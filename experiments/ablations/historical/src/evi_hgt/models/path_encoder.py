"""Packed Phase 6 path tokenization, Transformer encoding, and step pooling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evi_hgt.sampling.path_schema import PackedPathBatch


@dataclass(frozen=True, slots=True)
class PathEncoderConfig:
    core_hidden_dim: int
    path_hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.1
    max_length: int = 4
    ffn_multiplier: int = 2

    def __post_init__(self) -> None:
        for name in (
            "core_hidden_dim",
            "path_hidden_dim",
            "num_heads",
            "num_layers",
            "max_length",
            "ffn_multiplier",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.path_hidden_dim % self.num_heads:
            raise ValueError("path_hidden_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class PathEncoderOutput:
    path_representation: Tensor
    step_states: Tensor
    step_weights: Tensor


class PathEncoder(nn.Module):
    """Encode real bounded path steps using Phase 4 node and fact contexts."""

    def __init__(self, config: PathEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_projection = nn.Sequential(
            nn.Linear(3 * config.core_hidden_dim, config.path_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.path_hidden_dim),
        )
        self.position_embedding = nn.Embedding(config.max_length, config.path_hidden_dim)
        self.direction_embedding = nn.Embedding(2, config.path_hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.path_hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_multiplier * config.path_hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.step_projection = nn.Linear(config.path_hidden_dim, config.path_hidden_dim)
        self.step_score = nn.Linear(config.path_hidden_dim, 1, bias=False)

    def forward(
        self,
        paths: PackedPathBatch,
        node_embeddings: Tensor,
        edge_context: Tensor,
    ) -> PathEncoderOutput:
        if node_embeddings.ndim != 2 or node_embeddings.shape[1] != self.config.core_hidden_dim:
            raise ValueError("node embeddings have the wrong Phase 4 hidden dimension")
        if edge_context.ndim != 2 or edge_context.shape[1] != self.config.core_hidden_dim:
            raise ValueError("edge context has the wrong Phase 4 hidden dimension")
        if paths.max_length != self.config.max_length:
            raise ValueError("packed path max_length does not match path encoder config")
        paths.validate(num_nodes=node_embeddings.shape[0], num_facts=edge_context.shape[0])
        if paths.num_paths == 0:
            empty_states = node_embeddings.new_empty(
                (0, self.config.max_length, self.config.path_hidden_dim)
            )
            return PathEncoderOutput(
                path_representation=node_embeddings.new_empty(
                    (0, self.config.path_hidden_dim)
                ),
                step_states=empty_states,
                step_weights=node_embeddings.new_empty((0, self.config.max_length)),
            )
        source = node_embeddings[paths.step_source_index]
        target = node_embeddings[paths.step_target_index]
        facts = edge_context[paths.step_fact_index]
        tokens = self.token_projection(torch.cat((source, facts, target), dim=-1))
        positions = torch.arange(
            self.config.max_length,
            device=tokens.device,
            dtype=torch.long,
        ).unsqueeze(0)
        tokens = (
            tokens
            + self.position_embedding(positions)
            + self.direction_embedding(paths.step_direction)
        )
        states = self.transformer(tokens, src_key_padding_mask=~paths.step_mask)
        logits = self.step_score(torch.tanh(self.step_projection(states))).squeeze(-1)
        work = logits.float().masked_fill(~paths.step_mask, -torch.inf)
        weights = torch.softmax(work, dim=1).masked_fill(~paths.step_mask, 0.0)
        weights = weights.to(dtype=states.dtype)
        representation = (weights.unsqueeze(-1) * states).sum(dim=1)
        if not all(
            torch.isfinite(value).all() for value in (states, weights, representation)
        ):
            raise FloatingPointError("path encoder produced non-finite values")
        return PathEncoderOutput(
            path_representation=representation,
            step_states=states,
            step_weights=weights,
        )

