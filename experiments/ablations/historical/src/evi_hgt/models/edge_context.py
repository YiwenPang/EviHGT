"""Canonical-fact edge-context composition for EviHGT."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.evidence_pooling import (
    EvidencePoolOutput,
    EvidenceStrengthOutput,
)
from evi_hgt.models.four_valued_logic import LogicState
from evi_hgt.models.input_contract import PackedEvidenceBatch


class EdgeContextBuilder(nn.Module):
    """Compose relation, contextual evidence, provenance, logic, and masks."""

    def __init__(
        self,
        config: EviHGTConfig,
        relation_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.relation_embedding = relation_embedding or nn.Embedding(
            config.num_edge_types, config.relation_dim
        )
        input_dim = config.relation_dim
        if config.use_evidence_context:
            input_dim += 3 * config.evidence_dim + config.provenance_dim + 4 + 6
        self.projection = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )

    def relation_only(self, batch: PackedEvidenceBatch) -> Tensor:
        """Build typed relation context without reading evidence enhancements."""

        if self.config.use_evidence_context:
            raise RuntimeError("relation-only context requires its explicit config gate")
        context = self.projection(
            self.relation_embedding(batch.fact_edge_type_id)
        )
        if not torch.isfinite(context).all():
            raise FloatingPointError("relation-only edge context contains non-finite values")
        return context

    def forward(
        self,
        pooled: EvidencePoolOutput,
        strengths: EvidenceStrengthOutput,
        logic: LogicState,
        batch: PackedEvidenceBatch,
    ) -> Tensor:
        if not self.config.use_evidence_context:
            raise RuntimeError(
                "enhanced edge-context forward is disabled for this model variant"
            )
        dtype = pooled.z_all.dtype
        support_gate = strengths.support_available.to(dtype=dtype).unsqueeze(-1)
        counter_gate = strengths.counter_available.to(dtype=dtype).unsqueeze(-1)
        logic_gate = logic.available.to(dtype=dtype).unsqueeze(-1)
        bits = torch.stack(
            (
                strengths.support_available,
                strengths.counter_available,
                logic.available,
                pooled.has_all_observation,
                pooled.has_support_observation,
                pooled.has_counter_observation,
            ),
            dim=-1,
        ).to(dtype=dtype)
        joined = torch.cat(
            (
                self.relation_embedding(batch.fact_edge_type_id),
                pooled.z_all,
                support_gate * pooled.z_support,
                counter_gate * pooled.z_counter,
                pooled.fact_provenance,
                logic_gate * logic.masses.to(dtype=dtype),
                bits,
            ),
            dim=-1,
        )
        context = self.projection(joined)
        if not torch.isfinite(context).all():
            raise FloatingPointError("Edge context contains non-finite values")
        return context
