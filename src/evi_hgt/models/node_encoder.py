"""Cold-start-capable node initialization for EviHGT."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.input_contract import ModelInputError, NodeBatch


class NodeEncoder(nn.Module):
    """Sum enabled type, real-feature, and optional transductive ID terms."""

    def __init__(self, config: EviHGTConfig) -> None:
        super().__init__()
        self.config = config
        self.type_embedding = (
            nn.Embedding(config.num_node_types, config.hidden_dim)
            if config.use_node_type_embedding
            else None
        )
        self.feature_projection = (
            nn.Linear(config.node_feature_dim, config.hidden_dim, bias=True)
            if config.use_node_features
            else None
        )
        self.node_id_embedding = (
            nn.Embedding(config.num_nodes, config.hidden_dim)
            if config.use_node_id_embedding
            else None
        )

    def forward(self, nodes: NodeBatch) -> Tensor:
        contributions: list[Tensor] = []
        if self.type_embedding is not None:
            contributions.append(self.type_embedding(nodes.node_type_id))
        if self.feature_projection is not None:
            if nodes.features is None:
                raise ModelInputError(
                    "node features are required when use_node_features is enabled"
                )
            contributions.append(self.feature_projection(nodes.features))
        if self.node_id_embedding is not None:
            if nodes.node_id is None:
                raise ModelInputError(
                    "node_id is required when use_node_id_embedding is enabled"
                )
            contributions.append(self.node_id_embedding(nodes.node_id))
        if not contributions:  # Configuration validation normally prevents this.
            raise RuntimeError("NodeEncoder has no enabled representation source")
        output = contributions[0]
        for contribution in contributions[1:]:
            output = output + contribution
        if not torch.isfinite(output).all():
            raise FloatingPointError("NodeEncoder produced non-finite values")
        return output

