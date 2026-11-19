"""Query-conditioned path encoder path scoring, pooling, and residual fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evi_hgt.models.path_encoder import (
    PathEncoder,
    PathEncoderConfig,
    PathEncoderOutput,
)
from evi_hgt.models.segment_ops import segment_softmax, segment_sum
from evi_hgt.sampling.path_schema import PackedPathBatch


@dataclass(frozen=True, slots=True)
class PathReasoningConfig:
    core_hidden_dim: int
    path_hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 1
    dropout: float = 0.1
    max_length: int = 4
    ffn_multiplier: int = 2

    def encoder_config(self) -> PathEncoderConfig:
        return PathEncoderConfig(
            core_hidden_dim=self.core_hidden_dim,
            path_hidden_dim=self.path_hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_length=self.max_length,
            ffn_multiplier=self.ffn_multiplier,
        )


@dataclass(frozen=True, slots=True)
class PathReasoningOutput:
    path_representation: Tensor
    step_states: Tensor
    step_weights: Tensor
    path_score: Tensor
    beta: Tensor
    path_context: Tensor
    path_available: Tensor


@dataclass(frozen=True, slots=True)
class PathFusionOutput:
    query_representation: Tensor
    path_gate: Tensor
    path_delta: Tensor


class PathReasoner(nn.Module):
    """Encode paths and normalize model relevance within each query."""

    def __init__(self, config: PathReasoningConfig) -> None:
        super().__init__()
        self.config = config
        self.path_encoder = PathEncoder(config.encoder_config())
        self.path_projection = nn.Linear(config.path_hidden_dim, config.path_hidden_dim)
        self.query_projection = nn.Linear(config.core_hidden_dim, config.path_hidden_dim)
        self.score_vector = nn.Linear(config.path_hidden_dim, 1, bias=False)

    def forward(
        self,
        paths: PackedPathBatch,
        node_embeddings: Tensor,
        edge_context: Tensor | None,
        query_embedding: Tensor,
    ) -> PathReasoningOutput:
        if query_embedding.shape != (paths.num_queries, self.config.core_hidden_dim):
            raise ValueError("query embedding shape does not match packed paths")
        available = paths.query_path_count > 0
        if paths.num_paths == 0:
            empty_representation = query_embedding.new_empty(
                (0, self.config.path_hidden_dim)
            )
            return PathReasoningOutput(
                path_representation=empty_representation,
                step_states=query_embedding.new_empty(
                    (0, self.config.max_length, self.config.path_hidden_dim)
                ),
                step_weights=query_embedding.new_empty((0, self.config.max_length)),
                path_score=query_embedding.new_empty((0,)),
                beta=query_embedding.new_empty((0,)),
                path_context=query_embedding.new_zeros(
                    (paths.num_queries, self.config.path_hidden_dim)
                ),
                path_available=available,
            )
        if edge_context is None:
            raise ValueError("non-empty path reasoning requires graph encoder edge context")
        encoded: PathEncoderOutput = self.path_encoder(
            paths, node_embeddings, edge_context
        )
        conditioned = self.path_projection(encoded.path_representation) + self.query_projection(
            query_embedding[paths.query_index]
        )
        scores = self.score_vector(torch.tanh(conditioned)).squeeze(-1)
        beta = segment_softmax(scores.float(), paths.query_index, paths.num_queries).to(
            dtype=encoded.path_representation.dtype
        )
        context = segment_sum(
            beta.unsqueeze(-1) * encoded.path_representation,
            paths.query_index,
            paths.num_queries,
        )
        if not all(torch.isfinite(value).all() for value in (scores, beta, context)):
            raise FloatingPointError("path reasoner produced non-finite values")
        return PathReasoningOutput(
            path_representation=encoded.path_representation,
            step_states=encoded.step_states,
            step_weights=encoded.step_weights,
            path_score=scores,
            beta=beta,
            path_context=context,
            path_available=available,
        )


class GatedResidualPathFusion(nn.Module):
    """Availability-masked residual fusion with exact core-only fallback."""

    def __init__(self, *, core_hidden_dim: int, path_hidden_dim: int) -> None:
        super().__init__()
        self.core_hidden_dim = core_hidden_dim
        self.path_hidden_dim = path_hidden_dim
        self.path_projection = nn.Linear(path_hidden_dim, core_hidden_dim)
        self.gate = nn.Linear(core_hidden_dim + path_hidden_dim, 1)

    def forward(
        self,
        core_representation: Tensor,
        path_context: Tensor,
        path_available: Tensor,
    ) -> PathFusionOutput:
        query_count = core_representation.shape[0]
        if core_representation.shape != (query_count, self.core_hidden_dim):
            raise ValueError("core query representation has the wrong shape")
        if path_context.shape != (query_count, self.path_hidden_dim):
            raise ValueError("path context has the wrong shape")
        if path_available.shape != (query_count,) or path_available.dtype != torch.bool:
            raise ValueError("path availability must be a Boolean query vector")
        raw_gate = torch.sigmoid(
            self.gate(torch.cat((core_representation, path_context), dim=-1)).squeeze(-1)
        )
        gate = torch.where(path_available, raw_gate, torch.zeros_like(raw_gate))
        delta = gate.unsqueeze(-1) * self.path_projection(path_context)
        fused = torch.where(
            path_available.unsqueeze(-1),
            core_representation + delta,
            core_representation,
        )
        if not all(torch.isfinite(value).all() for value in (gate, delta, fused)):
            raise FloatingPointError("path fusion produced non-finite values")
        return PathFusionOutput(
            query_representation=fused,
            path_gate=gate,
            path_delta=delta,
        )
