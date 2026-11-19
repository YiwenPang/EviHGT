"""Packed contextual and proposition-scoped evidence pooling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.evidence_encoder import EvidenceEncoding
from evi_hgt.models.input_contract import PackedEvidenceBatch
from evi_hgt.models.segment_ops import segment_mean, segment_softmax, segment_sum


@dataclass(frozen=True, slots=True)
class EvidencePoolOutput:
    z_all: Tensor
    z_support: Tensor
    z_counter: Tensor
    fact_provenance: Tensor
    has_all_observation: Tensor
    has_support_observation: Tensor
    has_counter_observation: Tensor


@dataclass(frozen=True, slots=True)
class EvidenceStrengthOutput:
    support_score: Tensor
    counter_score: Tensor
    support_available: Tensor
    counter_available: Tensor


class PackedAttentionPool(nn.Module):
    """Attention-pool selected packed rows into fixed segment slots."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.Tanh(),
            nn.Linear(dimension, 1),
        )

    def forward(
        self,
        values: Tensor,
        segment_index: Tensor,
        num_segments: int,
        selection: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if selection is None:
            selection = torch.ones(
                values.shape[0], dtype=torch.bool, device=values.device
            )
        chosen_values = values[selection]
        chosen_index = segment_index[selection]
        if chosen_values.shape[0]:
            logits = self.scorer(chosen_values).squeeze(-1)
            weights = segment_softmax(logits, chosen_index, num_segments)
            pooled = segment_sum(
                chosen_values * weights.unsqueeze(-1),
                chosen_index,
                num_segments,
            )
            counts = segment_sum(
                torch.ones_like(chosen_index, dtype=values.dtype),
                chosen_index,
                num_segments,
            )
            has_observation = counts > 0
        else:
            pooled = values.new_zeros((num_segments, values.shape[-1]))
            has_observation = torch.zeros(
                num_segments, dtype=torch.bool, device=values.device
            )
        return pooled, has_observation


class EvidencePooler(nn.Module):
    """Pool all context and independent support/counter record subsets."""

    def __init__(self, config: EviHGTConfig) -> None:
        super().__init__()
        self.config = config
        self.all_pool = PackedAttentionPool(config.evidence_dim)
        self.support_pool = PackedAttentionPool(config.evidence_dim)
        self.counter_pool = PackedAttentionPool(config.evidence_dim)

    def forward(
        self,
        encoding: EvidenceEncoding,
        batch: PackedEvidenceBatch,
    ) -> EvidencePoolOutput:
        records = encoding.record_embeddings
        z_all, has_all = self.all_pool(
            records, batch.record_fact_index, batch.num_facts
        )
        polarity_enabled = bool(
            self.config.use_evidence and self.config.use_polarity_branches
        )
        support_selection = (
            batch.support_observed
            if polarity_enabled
            else torch.zeros_like(batch.support_observed)
        )
        counter_selection = (
            batch.counter_observed
            if polarity_enabled
            else torch.zeros_like(batch.counter_observed)
        )
        z_support, has_support = self.support_pool(
            records,
            batch.record_fact_index,
            batch.num_facts,
            support_selection,
        )
        z_counter, has_counter = self.counter_pool(
            records,
            batch.record_fact_index,
            batch.num_facts,
            counter_selection,
        )
        fact_provenance = segment_mean(
            encoding.provenance_embeddings,
            batch.record_fact_index,
            batch.num_facts,
        )
        return EvidencePoolOutput(
            z_all=z_all,
            z_support=z_support,
            z_counter=z_counter,
            fact_provenance=fact_provenance,
            has_all_observation=has_all,
            has_support_observation=has_support,
            has_counter_observation=has_counter,
        )


class EvidenceStrengthScorer(nn.Module):
    """Independently score verified support and counter channels."""

    def __init__(self, config: EviHGTConfig) -> None:
        super().__init__()
        self.config = config
        self.support = nn.Linear(config.evidence_dim + 1, 1)
        self.counter = nn.Linear(config.evidence_dim + 1, 1)

    def forward(
        self,
        pooled: EvidencePoolOutput,
        batch: PackedEvidenceBatch,
    ) -> EvidenceStrengthOutput:
        dtype = pooled.z_all.dtype
        support_input = torch.cat(
            (
                pooled.z_support,
                pooled.has_support_observation.to(dtype=dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        counter_input = torch.cat(
            (
                pooled.z_counter,
                pooled.has_counter_observation.to(dtype=dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        enabled = bool(self.config.use_evidence and self.config.use_polarity_branches)
        support_available = batch.fact_support_available & enabled
        counter_available = batch.fact_counter_available & enabled
        support_raw = torch.sigmoid(self.support(support_input).squeeze(-1))
        counter_raw = torch.sigmoid(self.counter(counter_input).squeeze(-1))
        support_score = torch.where(
            support_available, support_raw, torch.zeros_like(support_raw)
        )
        counter_score = torch.where(
            counter_available, counter_raw, torch.zeros_like(counter_raw)
        )
        if not torch.isfinite(support_score).all() or not torch.isfinite(
            counter_score
        ).all():
            raise FloatingPointError("Evidence strength scorer produced non-finite values")
        return EvidenceStrengthOutput(
            support_score=support_score,
            counter_score=counter_score,
            support_available=support_available,
            counter_available=counter_available,
        )

