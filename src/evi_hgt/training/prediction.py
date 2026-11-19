"""Query-level dual-channel prediction on top of the closed graph encoder core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evi_hgt.models.evi_hgt import EviHGT, EviHGTOutput
from evi_hgt.models.input_contract import EviHGTBatch, QueryBatch
from evi_hgt.models.four_valued_logic import LogicState
from evi_hgt.models.path_reasoning import (
    GatedResidualPathFusion,
    PathReasoner,
    PathReasoningConfig,
    PathReasoningOutput,
)
from evi_hgt.sampling.path_schema import PackedPathBatch


@dataclass(frozen=True, slots=True)
class QueryLogicMasses:
    """Latent model-derived masses from two uncalibrated monotone scores."""

    support_score: Tensor
    unsuitability_score: Tensor
    m_T: Tensor
    m_F: Tensor
    m_B: Tensor
    m_U: Tensor

    @property
    def support_probability(self) -> Tensor:
        """Deprecated compatibility alias; this is not a calibrated probability."""

        return self.support_score

    @property
    def unsuitability_probability(self) -> Tensor:
        """Deprecated compatibility alias; this is not a calibrated probability."""

        return self.unsuitability_score


@dataclass(frozen=True, slots=True)
class QueryPrediction:
    """Compact DDA prediction contract returned by :class:`EviHGTForDDA`."""

    support_logit: Tensor
    unsuitability_logit: Tensor
    support_score: Tensor
    unsuitability_score: Tensor
    m_T: Tensor
    m_F: Tensor
    m_B: Tensor
    m_U: Tensor
    ranking_score: Tensor
    query_representation: Tensor
    path_context: Tensor | None
    path_available: Tensor
    path_gate: Tensor | None
    path_reasoning: PathReasoningOutput | None
    edge_logic_state: LogicState | None

    @property
    def support_probability(self) -> Tensor:
        """Deprecated compatibility alias; this is not a calibrated probability."""

        return self.support_score

    @property
    def unsuitability_probability(self) -> Tensor:
        """Deprecated compatibility alias; this is not a calibrated probability."""

        return self.unsuitability_score


def query_logic_masses(
    support_logit: Tensor,
    unsuitability_logit: Tensor,
) -> QueryLogicMasses:
    """Map two independent logits to normalized predicted T/F/B/U masses."""

    if support_logit.shape != unsuitability_logit.shape:
        raise ValueError("support and unsuitability logits must have the same shape")
    support_score = torch.sigmoid(support_logit)
    unsuitability_score = torch.sigmoid(unsuitability_logit)
    support_complement = 1.0 - support_score
    unsuitability_complement = 1.0 - unsuitability_score
    return QueryLogicMasses(
        support_score=support_score,
        unsuitability_score=unsuitability_score,
        m_T=support_score * unsuitability_complement,
        m_F=support_complement * unsuitability_score,
        m_B=support_score * unsuitability_score,
        m_U=support_complement * unsuitability_complement,
    )


class QueryReadout(nn.Module):
    """Project final endpoints and the graph encoder query embedding back to ``H``."""

    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden_dim = hidden_dim
        self.core_pair_projector = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        core_output: EviHGTOutput,
        queries: QueryBatch,
        *,
        path_context: Tensor | None = None,
        path_available: bool = False,
    ) -> Tensor:
        if path_available or path_context is not None:
            raise NotImplementedError(
                "Path context is reserved for path module and is unavailable in prediction wrapper"
            )
        final_nodes = core_output.node_embeddings
        drug = final_nodes[queries.drug_node_index]
        disease = final_nodes[queries.disease_node_index]
        phase4_query = core_output.query_embedding
        expected = (queries.num_queries, self.hidden_dim)
        if drug.shape != expected or disease.shape != expected:
            raise ValueError(f"query endpoint embeddings must have shape {expected}")
        if phase4_query.shape != expected:
            raise ValueError(f"graph encoder query embedding must have shape {expected}")
        features = torch.cat(
            (drug, disease, drug * disease, (drug - disease).abs(), phase4_query),
            dim=-1,
        )
        return self.core_pair_projector(features)


class IndependentPredictionHead(nn.Module):
    """Small channel-specific ``H -> H -> 1`` prediction head."""

    def __init__(self, hidden_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_representation: Tensor) -> Tensor:
        return self.network(query_representation).squeeze(-1)


class EviHGTForDDA(nn.Module):
    """Trainable DDA predictor with a shared graph encoder."""

    def __init__(
        self,
        encoder: EviHGT,
        *,
        dropout: float | None = None,
        path_config: PathReasoningConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_dim = encoder.config.hidden_dim
        prediction_dropout = encoder.config.dropout if dropout is None else dropout
        self.query_readout = QueryReadout(hidden_dim, dropout=prediction_dropout)
        self.support_head = IndependentPredictionHead(
            hidden_dim, dropout=prediction_dropout
        )
        self.contraindication_unsuitability_head = IndependentPredictionHead(
            hidden_dim, dropout=prediction_dropout
        )
        if path_config is not None and path_config.core_hidden_dim != hidden_dim:
            raise ValueError("path core hidden dimension must match the graph encoder encoder")
        self.path_reasoner = PathReasoner(path_config) if path_config is not None else None
        self.path_fusion = (
            GatedResidualPathFusion(
                core_hidden_dim=hidden_dim,
                path_hidden_dim=path_config.path_hidden_dim,
            )
            if path_config is not None
            else None
        )

    def forward(
        self,
        batch: EviHGTBatch,
        *,
        path_batch: PackedPathBatch | None = None,
        path_context: Tensor | None = None,
        path_available: bool | Tensor = False,
        use_path_reasoning: bool = True,
    ) -> QueryPrediction:
        core_output = self.encoder(batch)
        core_representation = self.query_readout(
            core_output,
            batch.queries,
            path_context=None,
            path_available=False,
        )
        query_count = batch.queries.num_queries
        available = torch.zeros(
            query_count, dtype=torch.bool, device=core_representation.device
        )
        used_context: Tensor | None = None
        used_gate: Tensor | None = None
        reasoning_output: PathReasoningOutput | None = None
        query_representation = core_representation
        path_requested = path_batch is not None or path_context is not None or (
            isinstance(path_available, bool) and path_available
        ) or (isinstance(path_available, Tensor) and bool(path_available.any()))
        if use_path_reasoning and path_requested:
            if self.path_reasoner is None or self.path_fusion is None:
                raise NotImplementedError(
                    "Path reasoning requires a path module path_config"
                )
            if path_batch is not None and path_context is not None:
                raise ValueError("provide either path_batch or external path_context, not both")
            if path_batch is not None:
                if path_batch.num_queries != query_count:
                    raise ValueError("packed path query count does not match model batch")
                reasoning_output = self.path_reasoner(
                    path_batch,
                    core_output.node_embeddings,
                    core_output.edge_context,
                    core_output.query_embedding,
                )
                used_context = reasoning_output.path_context
                available = reasoning_output.path_available
            elif path_context is not None:
                used_context = path_context
                if isinstance(path_available, Tensor):
                    available = path_available.to(
                        device=core_representation.device, dtype=torch.bool
                    )
                else:
                    available = torch.full(
                        (query_count,),
                        bool(path_available),
                        dtype=torch.bool,
                        device=core_representation.device,
                    )
            if used_context is not None:
                fusion = self.path_fusion(
                    core_representation, used_context, available
                )
                query_representation = fusion.query_representation
                used_gate = fusion.path_gate
        support_logit = self.support_head(query_representation)
        unsuitability_logit = self.contraindication_unsuitability_head(
            query_representation
        )
        logic = query_logic_masses(support_logit, unsuitability_logit)
        tensors = (
            support_logit,
            unsuitability_logit,
            query_representation,
            logic.m_T,
            logic.m_F,
            logic.m_B,
            logic.m_U,
        )
        if not all(torch.isfinite(value).all() for value in tensors):
            raise FloatingPointError("query prediction produced non-finite values")
        return QueryPrediction(
            support_logit=support_logit,
            unsuitability_logit=unsuitability_logit,
            support_score=logic.support_score,
            unsuitability_score=logic.unsuitability_score,
            m_T=logic.m_T,
            m_F=logic.m_F,
            m_B=logic.m_B,
            m_U=logic.m_U,
            ranking_score=logic.m_T,
            query_representation=query_representation,
            path_context=used_context,
            path_available=available,
            path_gate=used_gate,
            path_reasoning=reasoning_output,
            edge_logic_state=core_output.logic_state,
        )
