"""Packed provenance encoding without hand-written source authority scores."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.input_contract import PackedEvidenceBatch
from evi_hgt.models.segment_ops import segment_mean


class ProvenanceEncoder(nn.Module):
    """Encode token identity, direct/indirect role, kind, and overlap category."""

    def __init__(self, config: EviHGTConfig) -> None:
        super().__init__()
        self.config = config
        enabled = config.use_provenance
        self.token_embedding = (
            nn.Embedding(config.provenance_token_buckets, config.provenance_dim)
            if enabled
            else None
        )
        self.role_embedding = (
            nn.Embedding(config.num_provenance_roles, config.provenance_dim)
            if enabled
            else None
        )
        self.kind_embedding = (
            nn.Embedding(config.num_provenance_kinds, config.provenance_dim)
            if enabled
            else None
        )
        self.overlap_embedding = (
            nn.Embedding(config.num_overlap_categories, config.provenance_dim)
            if enabled
            else None
        )
        self.normalization = nn.LayerNorm(config.provenance_dim) if enabled else None
        self.register_buffer("_zero_anchor", torch.empty(0), persistent=False)

    def forward(self, batch: PackedEvidenceBatch) -> Tensor:
        record_count = batch.num_records
        if not self.config.use_provenance:
            return self._zero_anchor.new_zeros(
                (record_count, self.config.provenance_dim)
            )
        if any(
            module is None
            for module in (
                self.token_embedding,
                self.role_embedding,
                self.kind_embedding,
                self.overlap_embedding,
                self.normalization,
            )
        ):  # pragma: no cover - constructor invariant.
            raise RuntimeError("enabled provenance encoder modules are missing")
        if batch.provenance_token_ids.numel():
            mentions = (
                self.token_embedding(batch.provenance_token_ids)
                + self.role_embedding(batch.provenance_role_ids)
                + self.kind_embedding(batch.provenance_kind_ids)
            )
            pooled = segment_mean(
                mentions, batch.provenance_record_index, record_count
            )
        else:
            pooled = self.token_embedding.weight.new_zeros(
                (record_count, self.config.provenance_dim)
            )
        encoded = self.normalization(
            pooled + self.overlap_embedding(batch.overlap_category_id)
        )
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("ProvenanceEncoder produced non-finite values")
        return encoded
