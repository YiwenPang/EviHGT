"""Fixed endpoint query representation for graph encoder DDA subgraphs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig


class QueryEncoder(nn.Module):
    def __init__(self, config: EviHGTConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(
        self,
        node_embeddings: Tensor,
        drug_node_index: Tensor,
        disease_node_index: Tensor,
    ) -> Tensor:
        drug = node_embeddings[drug_node_index]
        disease = node_embeddings[disease_node_index]
        query = self.encoder(
            torch.cat(
                (drug, disease, drug * disease, (drug - disease).abs()), dim=-1
            )
        )
        if not torch.isfinite(query).all():
            raise FloatingPointError("QueryEncoder produced non-finite values")
        return query

