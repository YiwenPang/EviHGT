"""Relation/schema-aware encoding of packed evidence records."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.input_contract import PackedEvidenceBatch
from evi_hgt.models.provenance_encoder import ProvenanceEncoder
from evi_hgt.models.segment_ops import segment_mean


@dataclass(frozen=True, slots=True)
class EvidenceEncoding:
    record_embeddings: Tensor
    provenance_embeddings: Tensor


class EvidenceRecordEncoder(nn.Module):
    """Shared record encoder conditioned by the formal typed-edge identity."""

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
        self.property_token_embedding = nn.Embedding(
            config.property_token_buckets, config.evidence_dim
        )
        self.evidence_token_embedding = nn.Embedding(
            config.evidence_token_buckets, config.evidence_dim
        )
        self.provenance_encoder = ProvenanceEncoder(config)
        input_dim = (
            2 * config.num_numeric_features
            + 2 * config.num_boolean_features
            + config.evidence_dim
            + config.evidence_dim
            + config.provenance_dim
            + config.relation_dim
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.evidence_dim),
            nn.GELU(),
            nn.LayerNorm(config.evidence_dim),
        )

    @staticmethod
    def _packed_token_mean(
        embedding: nn.Embedding,
        token_ids: Tensor,
        record_index: Tensor,
        record_count: int,
    ) -> Tensor:
        if not token_ids.numel():
            return embedding.weight.new_zeros((record_count, embedding.embedding_dim))
        return segment_mean(embedding(token_ids), record_index, record_count)

    def forward(self, batch: PackedEvidenceBatch) -> EvidenceEncoding:
        record_count = batch.num_records
        dtype = self.relation_embedding.weight.dtype
        numeric_mask = batch.numeric_mask.to(dtype=dtype)
        boolean_mask = batch.boolean_mask.to(dtype=dtype)
        if self.config.use_property:
            numeric = torch.cat(
                (batch.numeric_values.to(dtype=dtype) * numeric_mask, numeric_mask),
                dim=-1,
            )
            boolean = torch.cat(
                (batch.boolean_values.to(dtype=dtype) * boolean_mask, boolean_mask),
                dim=-1,
            )
            property_tokens = self._packed_token_mean(
                self.property_token_embedding,
                batch.property_token_ids,
                batch.property_token_record_index,
                record_count,
            )
        else:
            numeric = self.relation_embedding.weight.new_zeros(
                (record_count, 2 * self.config.num_numeric_features)
            )
            boolean = self.relation_embedding.weight.new_zeros(
                (record_count, 2 * self.config.num_boolean_features)
            )
            property_tokens = self.property_token_embedding.weight.new_zeros(
                (record_count, self.config.evidence_dim)
            )
        if self.config.use_evidence:
            evidence_tokens = self._packed_token_mean(
                self.evidence_token_embedding,
                batch.evidence_token_ids,
                batch.evidence_token_record_index,
                record_count,
            )
        else:
            evidence_tokens = self.evidence_token_embedding.weight.new_zeros(
                (record_count, self.config.evidence_dim)
            )
        provenance = self.provenance_encoder(batch)
        relation = self.relation_embedding(batch.record_edge_type_id)
        joined = torch.cat(
            (numeric, boolean, property_tokens, evidence_tokens, provenance, relation),
            dim=-1,
        )
        encoded = self.encoder(joined)
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("EvidenceRecordEncoder produced non-finite values")
        return EvidenceEncoding(
            record_embeddings=encoded,
            provenance_embeddings=provenance,
        )

