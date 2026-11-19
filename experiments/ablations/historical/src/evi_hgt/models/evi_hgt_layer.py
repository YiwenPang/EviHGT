"""Query-conditioned evidence/logic-aware heterogeneous Transformer layer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.four_valued_logic import LogicState
from evi_hgt.models.input_contract import EdgeBatch, NodeBatch
from evi_hgt.models.segment_ops import segment_softmax, segment_sum


@dataclass(frozen=True, slots=True)
class AttentionOutput:
    node_updates: Tensor
    attention_weights: Tensor | None
    attention_logits: Tensor | None
    query_gate: Tensor | None


@dataclass(frozen=True, slots=True)
class LayerOutput:
    node_embeddings: Tensor
    post_attention: Tensor
    attention: AttentionOutput


class EviHGTAttention(nn.Module):
    """Typed multi-head attention with edge/query logits and logic messages."""

    T_INDEX = 0
    F_INDEX = 1
    B_INDEX = 2
    U_INDEX = 3
    NA_INDEX = 4

    def __init__(self, config: EviHGTConfig) -> None:
        super().__init__()
        self.config = config
        self.use_logic_mass_conditioning = True
        shape = (
            config.num_node_types,
            config.num_heads,
            config.hidden_dim,
            config.head_dim,
        )
        self.query_weight = nn.Parameter(torch.empty(shape))
        self.key_weight = nn.Parameter(torch.empty(shape))
        self.value_weight = nn.Parameter(torch.empty(shape))
        self.edge_key_projection = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )
        # Keep historical state-dict keys across the ablation boundary. Disabled
        # modules are frozen and bypassed in ``forward``.
        self.query_projection = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )
        self.query_context_projection = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )
        self.relation_bias = nn.Embedding(config.num_edge_types, config.num_heads)
        self.logic_transforms = nn.Parameter(
            torch.empty(
                5,
                config.num_heads,
                config.head_dim,
                config.head_dim,
            )
        )
        self.edge_value_projection = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.reset_parameters()
        if not config.use_query_conditioning:
            self.query_projection.requires_grad_(False)
            self.query_context_projection.requires_grad_(False)
        if not config.use_four_valued_logic:
            self.logic_transforms.requires_grad_(False)

    def reset_parameters(self) -> None:
        for value in (self.query_weight, self.key_weight, self.value_weight):
            nn.init.xavier_uniform_(value)
        nn.init.xavier_uniform_(self.logic_transforms)
        nn.init.xavier_uniform_(self.edge_key_projection.weight)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.query_context_projection.weight)
        nn.init.xavier_uniform_(self.edge_value_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.zeros_(self.relation_bias.weight)

    @staticmethod
    def _typed_project(hidden: Tensor, node_type_id: Tensor, weight: Tensor) -> Tensor:
        selected = weight[node_type_id]
        return torch.einsum("ni,nhid->nhd", hidden, selected)

    def logic_condition_values(
        self,
        source_values: Tensor,
        masses: Tensor,
        available: Tensor,
    ) -> Tensor:
        """Apply four mass-weighted transforms or the independent NA transform."""

        if not self.config.use_four_valued_logic:
            return source_values
        transformed = torch.einsum(
            "ehd,shdf->eshf", source_values, self.logic_transforms
        )
        # confirmatory ablation: only the direct four-transform mixture loses dynamic
        # conditioning. Do not mutate LogicState or the independent NA gate.
        if not self.use_logic_mass_conditioning:
            masses = torch.full_like(masses, 0.25)
        available_path = (
            transformed[:, :4]
            * masses.to(dtype=transformed.dtype).unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)
        na_path = transformed[:, self.NA_INDEX]
        return torch.where(
            available.unsqueeze(-1).unsqueeze(-1), available_path, na_path
        )

    def forward(
        self,
        hidden: Tensor,
        nodes: NodeBatch,
        edges: EdgeBatch,
        edge_context: Tensor,
        fact_logic: LogicState,
        query_embedding: Tensor,
        return_attention: bool = False,
    ) -> AttentionOutput:
        node_count = hidden.shape[0]
        edge_count = edges.num_edges
        q_nodes = self._typed_project(
            hidden, nodes.node_type_id, self.query_weight
        )
        k_nodes = self._typed_project(hidden, nodes.node_type_id, self.key_weight)
        v_nodes = self._typed_project(
            hidden, nodes.node_type_id, self.value_weight
        )
        edge_key = self.edge_key_projection(edge_context).reshape(
            edge_count, self.config.num_heads, self.config.head_dim
        )
        q_edge = q_nodes[edges.dst]
        k_edge = k_nodes[edges.src] + edge_key
        logits = (q_edge * k_edge).sum(dim=-1) / math.sqrt(self.config.head_dim)
        logits = logits + self.relation_bias(edges.edge_type_id)
        if self.config.use_query_conditioning:
            edge_query = query_embedding[edges.graph_id]
            projected_query = self.query_projection(edge_query).reshape(
                edge_count, self.config.num_heads, self.config.head_dim
            )
            projected_context = self.query_context_projection(edge_context).reshape(
                edge_count, self.config.num_heads, self.config.head_dim
            )
            query_gate = (projected_query * projected_context).sum(dim=-1) / math.sqrt(
                self.config.head_dim
            )
        else:
            query_gate = logits.new_zeros((edge_count, self.config.num_heads))
        logits = logits + query_gate
        weights = segment_softmax(logits, edges.dst, node_count)

        source_values = v_nodes[edges.src]
        edge_masses = fact_logic.masses[edges.fact_index]
        edge_logic_available = fact_logic.available[edges.fact_index]
        logic_values = self.logic_condition_values(
            source_values, edge_masses, edge_logic_available
        )
        edge_value = self.edge_value_projection(edge_context).reshape(
            edge_count, self.config.num_heads, self.config.head_dim
        )
        messages = (logic_values + edge_value) * weights.unsqueeze(-1)
        aggregated = segment_sum(messages, edges.dst, node_count).reshape(
            node_count, self.config.hidden_dim
        )
        node_updates = self.output_projection(aggregated)
        if not torch.isfinite(node_updates).all() or not torch.isfinite(weights).all():
            raise FloatingPointError("EviHGT attention produced non-finite values")
        return AttentionOutput(
            node_updates=node_updates,
            attention_weights=weights if return_attention else None,
            attention_logits=logits if return_attention else None,
            query_gate=query_gate if return_attention else None,
        )


class EviHGTLayer(nn.Module):
    """Standard Pre-LN attention/FFN residual block."""

    def __init__(self, config: EviHGTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attention = EviHGTAttention(config)
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_multiplier * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_multiplier * config.hidden_dim, config.hidden_dim),
        )
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: Tensor,
        nodes: NodeBatch,
        edges: EdgeBatch,
        edge_context: Tensor,
        fact_logic: LogicState,
        query_embedding: Tensor,
        return_attention: bool = False,
    ) -> LayerOutput:
        normalized = self.norm1(hidden)
        attention = self.attention(
            normalized,
            nodes,
            edges,
            edge_context,
            fact_logic,
            query_embedding,
            return_attention,
        )
        post_attention = hidden + self.dropout1(attention.node_updates)
        feed_forward = self.ffn(self.norm2(post_attention))
        output = post_attention + self.dropout2(feed_forward)
        if not torch.isfinite(output).all():
            raise FloatingPointError("EviHGT layer produced non-finite values")
        return LayerOutput(
            node_embeddings=output,
            post_attention=post_attention,
            attention=attention,
        )
