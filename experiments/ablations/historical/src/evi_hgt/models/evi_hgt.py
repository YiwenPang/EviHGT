"""End-to-end Phase 4 EviHGT core without prediction heads or path module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.edge_context import EdgeContextBuilder
from evi_hgt.models.evidence_encoder import EvidenceRecordEncoder
from evi_hgt.models.evidence_pooling import EvidencePooler, EvidenceStrengthScorer
from evi_hgt.models.evi_hgt_layer import EviHGTLayer
from evi_hgt.models.four_valued_logic import FourValuedLogic, LogicState
from evi_hgt.models.input_contract import EviHGTBatch
from evi_hgt.models.node_encoder import NodeEncoder
from evi_hgt.models.query_encoder import QueryEncoder


@dataclass(frozen=True, slots=True)
class EviHGTOutput:
    node_embeddings: Tensor
    query_embedding: Tensor
    attention: tuple[Tensor, ...] | None
    edge_context: Tensor | None
    logic_state: LogicState | None


class EviHGT(nn.Module):
    """Core query-conditioned evidence/logic-aware heterogeneous Transformer."""

    def __init__(self, config: EviHGTConfig) -> None:
        super().__init__()
        self.config = config
        self.relation_embedding = nn.Embedding(
            config.num_edge_types, config.relation_dim
        )
        self.node_encoder = NodeEncoder(config)
        self.query_encoder = QueryEncoder(config)
        self.evidence_encoder = (
            EvidenceRecordEncoder(config, self.relation_embedding)
            if config.use_evidence_context
            else None
        )
        self.evidence_pooler = (
            EvidencePooler(config) if config.use_evidence_context else None
        )
        self.strength_scorer = (
            EvidenceStrengthScorer(config) if config.use_evidence_context else None
        )
        self.logic = FourValuedLogic() if config.use_evidence_context else None
        self.edge_context_builder = EdgeContextBuilder(
            config, self.relation_embedding
        )
        self.layers = nn.ModuleList(
            EviHGTLayer(config) for _ in range(config.num_layers)
        )

    def forward(self, batch: EviHGTBatch) -> EviHGTOutput:
        batch.validate(self.config)
        hidden = self.node_encoder(batch.nodes)
        query = self.query_encoder(
            hidden,
            batch.queries.drug_node_index,
            batch.queries.disease_node_index,
        )
        if self.config.use_evidence_context:
            if any(
                module is None
                for module in (
                    self.evidence_encoder,
                    self.evidence_pooler,
                    self.strength_scorer,
                    self.logic,
                )
            ):  # pragma: no cover - constructor invariant.
                raise RuntimeError("enhanced evidence-context modules are missing")
            evidence = self.evidence_encoder(batch.evidence)
            pooled = self.evidence_pooler(evidence, batch.evidence)
            strengths = self.strength_scorer(pooled, batch.evidence)
            logic_available = (
                strengths.support_available
                & strengths.counter_available
                & bool(self.config.use_four_valued_logic)
            )
            logic = self.logic(
                strengths.support_score,
                strengths.counter_score,
                logic_available,
            )
            fact_context = self.edge_context_builder(
                pooled, strengths, logic, batch.evidence
            )
        else:
            fact_count = batch.evidence.num_facts
            logic = LogicState(
                masses=hidden.new_zeros((fact_count, 4), dtype=torch.float32),
                available=torch.zeros(
                    fact_count, dtype=torch.bool, device=hidden.device
                ),
            )
            fact_context = self.edge_context_builder.relation_only(batch.evidence)
        traversal_context = fact_context[batch.edges.fact_index]
        attention_values: list[Tensor] = []
        for layer in self.layers:
            layer_output = layer(
                hidden,
                batch.nodes,
                batch.edges,
                traversal_context,
                logic,
                query,
                self.config.return_attention,
            )
            hidden = layer_output.node_embeddings
            if self.config.return_attention:
                weights = layer_output.attention.attention_weights
                if weights is None:  # pragma: no cover - internal invariant.
                    raise RuntimeError("attention diagnostics were requested but absent")
                attention_values.append(weights)
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("EviHGT produced non-finite node embeddings")
        return EviHGTOutput(
            node_embeddings=hidden,
            query_embedding=query,
            attention=(tuple(attention_values) if self.config.return_attention else None),
            edge_context=(fact_context if self.config.return_edge_context else None),
            logic_state=(logic if self.config.return_logic_state else None),
        )
