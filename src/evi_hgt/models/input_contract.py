"""Bounded tensor input contract for the graph encoder EviHGT core."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from evi_hgt.models.config import EviHGTConfig


class ModelInputError(ValueError):
    """Raised when a bounded model batch violates the declared contract."""


@dataclass(frozen=True, slots=True)
class NodeBatch:
    node_type_id: Tensor
    graph_id: Tensor
    features: Tensor | None = None
    node_id: Tensor | None = None

    @property
    def num_nodes(self) -> int:
        return int(self.node_type_id.numel())


@dataclass(frozen=True, slots=True)
class EdgeBatch:
    src: Tensor
    dst: Tensor
    edge_type_id: Tensor
    fact_index: Tensor
    graph_id: Tensor

    @property
    def num_edges(self) -> int:
        return int(self.src.numel())


@dataclass(frozen=True, slots=True)
class PackedEvidenceBatch:
    num_facts: int
    fact_edge_type_id: Tensor
    fact_graph_id: Tensor
    record_fact_index: Tensor
    record_edge_type_id: Tensor
    numeric_values: Tensor
    numeric_mask: Tensor
    boolean_values: Tensor
    boolean_mask: Tensor
    property_token_ids: Tensor
    property_token_record_index: Tensor
    evidence_token_ids: Tensor
    evidence_token_record_index: Tensor
    provenance_token_ids: Tensor
    provenance_role_ids: Tensor
    provenance_kind_ids: Tensor
    provenance_record_index: Tensor
    overlap_category_id: Tensor
    support_observed: Tensor
    counter_observed: Tensor
    fact_support_available: Tensor
    fact_counter_available: Tensor

    @property
    def num_records(self) -> int:
        return int(self.record_fact_index.numel())


@dataclass(frozen=True, slots=True)
class QueryBatch:
    drug_node_index: Tensor
    disease_node_index: Tensor
    graph_id: Tensor

    @property
    def num_queries(self) -> int:
        return int(self.graph_id.numel())


def _require_vector(name: str, value: Tensor, length: int, *, dtype: torch.dtype) -> None:
    if value.ndim != 1 or value.numel() != length:
        raise ModelInputError(f"{name} must have shape [{length}]")
    if value.dtype != dtype:
        raise ModelInputError(f"{name} must have dtype {dtype}")


def _require_index_bounds(name: str, value: Tensor, upper: int) -> None:
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= upper):
        raise ModelInputError(f"{name} contains an out-of-range index")


def _require_token_group(
    *,
    token_name: str,
    token_ids: Tensor,
    index_name: str,
    record_index: Tensor,
    bucket_count: int,
    record_count: int,
) -> None:
    _require_vector(token_name, token_ids, token_ids.numel(), dtype=torch.long)
    _require_vector(index_name, record_index, token_ids.numel(), dtype=torch.long)
    _require_index_bounds(token_name, token_ids, bucket_count)
    _require_index_bounds(index_name, record_index, record_count)


@dataclass(frozen=True, slots=True)
class EviHGTBatch:
    nodes: NodeBatch
    edges: EdgeBatch
    evidence: PackedEvidenceBatch
    queries: QueryBatch

    def validate(self, config: EviHGTConfig) -> None:
        nodes = self.nodes
        node_count = nodes.num_nodes
        _require_vector("node_type_id", nodes.node_type_id, node_count, dtype=torch.long)
        _require_vector("node graph_id", nodes.graph_id, node_count, dtype=torch.long)
        _require_index_bounds("node_type_id", nodes.node_type_id, config.num_node_types)
        if nodes.graph_id.numel() and int(nodes.graph_id.min()) < 0:
            raise ModelInputError("node graph_id must be non-negative")
        if nodes.features is not None:
            if nodes.features.shape != (node_count, config.node_feature_dim):
                raise ModelInputError(
                    "node features must have shape "
                    f"[{node_count}, {config.node_feature_dim}]"
                )
            if not torch.isfinite(nodes.features).all():
                raise ModelInputError("node features must be finite")
        if nodes.node_id is not None:
            _require_vector("node_id", nodes.node_id, node_count, dtype=torch.long)
            _require_index_bounds("node_id", nodes.node_id, config.num_nodes)

        edges = self.edges
        edge_count = edges.num_edges
        for name, value in (
            ("src", edges.src),
            ("dst", edges.dst),
            ("edge_type_id", edges.edge_type_id),
            ("fact_index", edges.fact_index),
            ("edge graph_id", edges.graph_id),
        ):
            _require_vector(name, value, edge_count, dtype=torch.long)
        _require_index_bounds("src", edges.src, node_count)
        _require_index_bounds("dst", edges.dst, node_count)
        _require_index_bounds("edge_type_id", edges.edge_type_id, config.num_edge_types)
        _require_index_bounds("fact_index", edges.fact_index, self.evidence.num_facts)
        if edge_count:
            if not torch.equal(nodes.graph_id[edges.src], edges.graph_id) or not torch.equal(
                nodes.graph_id[edges.dst], edges.graph_id
            ):
                raise ModelInputError("edge endpoint graph does not match edge graph_id")

        evidence = self.evidence
        if evidence.num_facts < 0:
            raise ModelInputError("num_facts must be non-negative")
        fact_count = evidence.num_facts
        _require_vector(
            "fact_edge_type_id", evidence.fact_edge_type_id, fact_count, dtype=torch.long
        )
        _require_vector("fact_graph_id", evidence.fact_graph_id, fact_count, dtype=torch.long)
        _require_index_bounds(
            "fact_edge_type_id", evidence.fact_edge_type_id, config.num_edge_types
        )
        if evidence.fact_graph_id.numel() and int(evidence.fact_graph_id.min()) < 0:
            raise ModelInputError("fact_graph_id must be non-negative")
        if edge_count:
            if not torch.equal(
                evidence.fact_edge_type_id[edges.fact_index], edges.edge_type_id
            ):
                raise ModelInputError("edge_type_id does not match its canonical fact")
            if not torch.equal(evidence.fact_graph_id[edges.fact_index], edges.graph_id):
                raise ModelInputError("edge graph_id does not match its canonical fact")

        record_count = evidence.num_records
        _require_vector(
            "record_fact_index", evidence.record_fact_index, record_count, dtype=torch.long
        )
        _require_vector(
            "record_edge_type_id",
            evidence.record_edge_type_id,
            record_count,
            dtype=torch.long,
        )
        _require_index_bounds("record_fact_index", evidence.record_fact_index, fact_count)
        _require_index_bounds(
            "record_edge_type_id", evidence.record_edge_type_id, config.num_edge_types
        )
        if record_count and not torch.equal(
            evidence.fact_edge_type_id[evidence.record_fact_index],
            evidence.record_edge_type_id,
        ):
            raise ModelInputError("record_edge_type_id does not match its canonical fact")

        expected_numeric = (record_count, config.num_numeric_features)
        if evidence.numeric_values.shape != expected_numeric:
            raise ModelInputError(f"numeric_values must have shape {expected_numeric}")
        if evidence.numeric_mask.shape != expected_numeric:
            raise ModelInputError(f"numeric_mask must have shape {expected_numeric}")
        if evidence.numeric_mask.dtype != torch.bool:
            raise ModelInputError("numeric_mask must be Boolean")
        if not torch.isfinite(evidence.numeric_values).all():
            raise ModelInputError("numeric_values must be finite")

        expected_boolean = (record_count, config.num_boolean_features)
        if evidence.boolean_values.shape != expected_boolean:
            raise ModelInputError(f"boolean_values must have shape {expected_boolean}")
        if evidence.boolean_mask.shape != expected_boolean:
            raise ModelInputError(f"boolean_mask must have shape {expected_boolean}")
        if evidence.boolean_values.dtype != torch.bool:
            raise ModelInputError("boolean_values must be Boolean")
        if evidence.boolean_mask.dtype != torch.bool:
            raise ModelInputError("boolean_mask must be Boolean")

        _require_token_group(
            token_name="property_token_ids",
            token_ids=evidence.property_token_ids,
            index_name="property_token_record_index",
            record_index=evidence.property_token_record_index,
            bucket_count=config.property_token_buckets,
            record_count=record_count,
        )
        _require_token_group(
            token_name="evidence_token_ids",
            token_ids=evidence.evidence_token_ids,
            index_name="evidence_token_record_index",
            record_index=evidence.evidence_token_record_index,
            bucket_count=config.evidence_token_buckets,
            record_count=record_count,
        )
        mention_count = int(evidence.provenance_token_ids.numel())
        for name, value in (
            ("provenance_token_ids", evidence.provenance_token_ids),
            ("provenance_role_ids", evidence.provenance_role_ids),
            ("provenance_kind_ids", evidence.provenance_kind_ids),
            ("provenance_record_index", evidence.provenance_record_index),
        ):
            _require_vector(name, value, mention_count, dtype=torch.long)
        _require_index_bounds(
            "provenance_token_ids",
            evidence.provenance_token_ids,
            config.provenance_token_buckets,
        )
        _require_index_bounds(
            "provenance_role_ids",
            evidence.provenance_role_ids,
            config.num_provenance_roles,
        )
        _require_index_bounds(
            "provenance_kind_ids",
            evidence.provenance_kind_ids,
            config.num_provenance_kinds,
        )
        _require_index_bounds(
            "provenance_record_index", evidence.provenance_record_index, record_count
        )
        _require_vector(
            "overlap_category_id",
            evidence.overlap_category_id,
            record_count,
            dtype=torch.long,
        )
        _require_index_bounds(
            "overlap_category_id",
            evidence.overlap_category_id,
            config.num_overlap_categories,
        )

        for name, value, length in (
            ("support_observed", evidence.support_observed, record_count),
            ("counter_observed", evidence.counter_observed, record_count),
            ("fact_support_available", evidence.fact_support_available, fact_count),
            ("fact_counter_available", evidence.fact_counter_available, fact_count),
        ):
            _require_vector(name, value, length, dtype=torch.bool)

        queries = self.queries
        query_count = queries.num_queries
        for name, value in (
            ("drug_node_index", queries.drug_node_index),
            ("disease_node_index", queries.disease_node_index),
            ("query graph_id", queries.graph_id),
        ):
            _require_vector(name, value, query_count, dtype=torch.long)
        _require_index_bounds("drug_node_index", queries.drug_node_index, node_count)
        _require_index_bounds("disease_node_index", queries.disease_node_index, node_count)
        if query_count:
            expected_graph_ids = torch.arange(
                query_count, device=queries.graph_id.device, dtype=torch.long
            )
            if not torch.equal(queries.graph_id, expected_graph_ids):
                raise ModelInputError(
                    "query graph_id must be contiguous and ordered from zero"
                )
            if not torch.equal(
                nodes.graph_id[queries.drug_node_index], queries.graph_id
            ) or not torch.equal(
                nodes.graph_id[queries.disease_node_index], queries.graph_id
            ):
                raise ModelInputError("query endpoint graph does not match query graph_id")
        for name, graph_ids in (
            ("node graph_id", nodes.graph_id),
            ("edge graph_id", edges.graph_id),
            ("fact_graph_id", evidence.fact_graph_id),
        ):
            _require_index_bounds(name, graph_ids, query_count)
