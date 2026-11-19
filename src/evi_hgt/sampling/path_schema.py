"""Bounded structural and tensor contracts for path reasoning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PathExtractionConfig:
    min_length: int = 2
    max_length: int = 4
    max_paths: int = 16
    beam_width: int = 32
    max_neighbors_per_expansion: int = 64
    use_hub_penalty: bool = True
    hub_penalty_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.min_length < 1:
            raise ValueError("min_length must be positive")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be at least min_length")
        if self.max_paths <= 0:
            raise ValueError("max_paths must be positive")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if self.max_neighbors_per_expansion <= 0:
            raise ValueError("max_neighbors_per_expansion must be positive")
        if self.hub_penalty_weight < 0.0:
            raise ValueError("hub_penalty_weight must be non-negative")


@dataclass(frozen=True, slots=True)
class PathFact:
    fact_index: int
    canonical_fact_id: int
    source_node_index: int
    target_node_index: int
    relation: str
    undirected: bool
    edge_type_id: int


@dataclass(frozen=True, slots=True)
class PathTraversal:
    source_node_index: int
    target_node_index: int
    fact_index: int
    direction: int


@dataclass(frozen=True, slots=True)
class PathSource:
    """One bounded, split-safe query graph used by both core and path search."""

    split_kind: str
    query_pair_id: str
    node_ids: tuple[int, ...]
    node_types: tuple[str, ...]
    facts: tuple[PathFact, ...]
    traversals: tuple[PathTraversal, ...]
    drug_node_index: int
    disease_node_index: int
    forbidden_fact_ids: frozenset[int]
    heldout_node_ids: frozenset[int]
    phase3_digest: str
    split_digest: str
    split_partition: str = "train"

    def validate(self) -> None:
        node_count = len(self.node_ids)
        if node_count != len(self.node_types):
            raise ValueError("node IDs and node types must have equal length")
        if len(set(self.node_ids)) != node_count:
            raise ValueError("PathSource node IDs must be unique")
        if not 0 <= self.drug_node_index < node_count:
            raise ValueError("drug node index is out of range")
        if not 0 <= self.disease_node_index < node_count:
            raise ValueError("disease node index is out of range")
        if self.node_types[self.drug_node_index] != "DRG":
            raise ValueError("path source drug endpoint must have type DRG")
        if self.node_types[self.disease_node_index] != "DIS":
            raise ValueError("path source disease endpoint must have type DIS")
        if not self.split_kind or not self.query_pair_id:
            raise ValueError("path source split and query bindings are required")
        if self.split_partition not in {"train", "validation", "test"}:
            raise ValueError("path source split partition is invalid")
        if not self.phase3_digest or not self.split_digest:
            raise ValueError("path source artifact digests are required")
        if tuple(fact.fact_index for fact in self.facts) != tuple(range(len(self.facts))):
            raise ValueError("PathSource fact indices must be contiguous")
        canonical_ids = [fact.canonical_fact_id for fact in self.facts]
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("canonical fact IDs must be unique")
        for fact in self.facts:
            if fact.canonical_fact_id < 0 or fact.edge_type_id < 0:
                raise ValueError("canonical fact and edge type IDs must be non-negative")
            if not 0 <= fact.source_node_index < node_count or not 0 <= fact.target_node_index < node_count:
                raise ValueError("canonical fact endpoint is out of range")
            if not fact.relation:
                raise ValueError("canonical fact relation must be non-empty")
            if fact.canonical_fact_id in self.forbidden_fact_ids:
                raise ValueError("forbidden canonical fact survived in PathSource")
        for traversal in self.traversals:
            if not 0 <= traversal.fact_index < len(self.facts):
                raise ValueError("traversal canonical fact index is out of range")
            if traversal.direction not in (0, 1):
                raise ValueError("traversal direction must be 0 or 1")
            fact = self.facts[traversal.fact_index]
            if traversal.direction == 0:
                expected = (fact.source_node_index, fact.target_node_index)
            else:
                if not fact.undirected:
                    raise ValueError("reverse traversal is invalid for a directed canonical fact")
                expected = (fact.target_node_index, fact.source_node_index)
            observed = (traversal.source_node_index, traversal.target_node_index)
            if observed != expected:
                raise ValueError("traversal does not match its canonical fact direction")


@dataclass(frozen=True, slots=True)
class PathCandidate:
    query_index: int
    path_id: str
    node_indices: tuple[int, ...]
    node_ids: tuple[int, ...]
    node_types: tuple[str, ...]
    fact_indices: tuple[int, ...]
    canonical_fact_ids: tuple[int, ...]
    relations: tuple[str, ...]
    directions: tuple[int, ...]
    structural_score: float

    @property
    def length(self) -> int:
        return len(self.fact_indices)

    @classmethod
    def create(
        cls,
        *,
        query_index: int,
        node_indices: tuple[int, ...],
        node_ids: tuple[int, ...],
        node_types: tuple[str, ...],
        fact_indices: tuple[int, ...],
        canonical_fact_ids: tuple[int, ...],
        relations: tuple[str, ...],
        directions: tuple[int, ...],
        structural_score: float,
    ) -> "PathCandidate":
        step_count = len(fact_indices)
        if query_index < 0:
            raise ValueError("query_index must be non-negative")
        if step_count <= 0:
            raise ValueError("a path candidate must contain at least one fact")
        if len(node_indices) != step_count + 1:
            raise ValueError("path node count must equal step count plus one")
        if len(node_ids) != len(node_indices) or len(node_types) != len(node_indices):
            raise ValueError("path node metadata lengths do not match")
        if not all(len(values) == step_count for values in (canonical_fact_ids, relations, directions)):
            raise ValueError("path step metadata lengths do not match")
        if len(set(node_indices)) != len(node_indices):
            raise ValueError("path candidates must be node-simple")
        if len(set(canonical_fact_ids)) != step_count:
            raise ValueError("path candidates must be canonical-fact-simple")
        if any(direction not in (0, 1) for direction in directions):
            raise ValueError("path direction must be 0 or 1")
        identity = {
            "node_ids": node_ids,
            "canonical_fact_ids": canonical_fact_ids,
            "directions": directions,
        }
        path_id = hashlib.sha256(
            json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return cls(
            query_index=query_index,
            path_id=path_id,
            node_indices=node_indices,
            node_ids=node_ids,
            node_types=node_types,
            fact_indices=fact_indices,
            canonical_fact_ids=canonical_fact_ids,
            relations=relations,
            directions=directions,
            structural_score=float(structural_score),
        )


@dataclass(frozen=True, slots=True)
class PackedPathBatch:
    step_source_index: Tensor
    step_target_index: Tensor
    step_fact_index: Tensor
    step_direction: Tensor
    step_mask: Tensor
    path_length: Tensor
    query_index: Tensor
    query_path_count: Tensor
    path_ids: tuple[str, ...]
    max_length: int

    @property
    def num_paths(self) -> int:
        return int(self.query_index.numel())

    @property
    def num_queries(self) -> int:
        return int(self.query_path_count.numel())

    def validate(self, *, num_nodes: int, num_facts: int) -> None:
        expected_steps = (self.num_paths, self.max_length)
        for name, value in (
            ("step_source_index", self.step_source_index),
            ("step_target_index", self.step_target_index),
            ("step_fact_index", self.step_fact_index),
            ("step_direction", self.step_direction),
        ):
            if value.shape != expected_steps or value.dtype != torch.long:
                raise ValueError(f"{name} must be a long tensor with shape {expected_steps}")
        if self.step_mask.shape != expected_steps or self.step_mask.dtype != torch.bool:
            raise ValueError(f"step_mask must be Boolean with shape {expected_steps}")
        if self.path_length.shape != (self.num_paths,) or self.path_length.dtype != torch.long:
            raise ValueError("path_length must be a long vector")
        if self.query_index.shape != (self.num_paths,) or self.query_index.dtype != torch.long:
            raise ValueError("query_index must be a long vector")
        if self.query_path_count.dtype != torch.long:
            raise ValueError("query_path_count must be a long vector")
        if len(self.path_ids) != self.num_paths or len(set(self.path_ids)) != self.num_paths:
            raise ValueError("path IDs must be unique and match the packed path count")
        if self.num_paths:
            if int(self.query_index.min()) < 0 or int(self.query_index.max()) >= self.num_queries:
                raise ValueError("packed path query index is out of range")
            if int(self.path_length.min()) <= 0 or int(self.path_length.max()) > self.max_length:
                raise ValueError("packed path length is out of range")
            expected_mask = torch.arange(
                self.max_length, device=self.path_length.device
            ).unsqueeze(0) < self.path_length.unsqueeze(1)
            if not torch.equal(self.step_mask, expected_mask):
                raise ValueError("step mask does not match packed path lengths")
            for name, value, upper in (
                ("source", self.step_source_index, num_nodes),
                ("target", self.step_target_index, num_nodes),
                ("fact", self.step_fact_index, num_facts),
            ):
                selected = value[self.step_mask]
                if selected.numel() and (int(selected.min()) < 0 or int(selected.max()) >= upper):
                    raise ValueError(f"packed path {name} index is out of range")
            directions = self.step_direction[self.step_mask]
            if directions.numel() and not bool(((directions == 0) | (directions == 1)).all()):
                raise ValueError("packed path direction must be 0 or 1")
        observed_counts = torch.bincount(
            self.query_index, minlength=self.num_queries
        ) if self.num_queries else torch.empty(0, dtype=torch.long, device=self.query_index.device)
        if not torch.equal(observed_counts, self.query_path_count):
            raise ValueError("query path counts do not match packed query indices")

    def to(self, device: str | torch.device) -> "PackedPathBatch":
        target = torch.device(device)
        return replace(
            self,
            step_source_index=self.step_source_index.to(target),
            step_target_index=self.step_target_index.to(target),
            step_fact_index=self.step_fact_index.to(target),
            step_direction=self.step_direction.to(target),
            step_mask=self.step_mask.to(target),
            path_length=self.path_length.to(target),
            query_index=self.query_index.to(target),
            query_path_count=self.query_path_count.to(target),
        )


def pack_path_candidates(
    candidates: tuple[PathCandidate, ...],
    *,
    num_queries: int,
    max_length: int,
    device: str | torch.device = "cpu",
) -> PackedPathBatch:
    if num_queries <= 0 or max_length <= 0:
        raise ValueError("num_queries and max_length must be positive")
    target = torch.device(device)
    path_count = len(candidates)
    source = torch.zeros((path_count, max_length), dtype=torch.long, device=target)
    destination = torch.zeros_like(source)
    facts = torch.zeros_like(source)
    directions = torch.zeros_like(source)
    mask = torch.zeros((path_count, max_length), dtype=torch.bool, device=target)
    lengths = torch.empty(path_count, dtype=torch.long, device=target)
    queries = torch.empty(path_count, dtype=torch.long, device=target)
    path_ids: list[str] = []
    for index, candidate in enumerate(candidates):
        if candidate.length > max_length:
            raise ValueError("candidate exceeds packed max_length")
        if candidate.query_index >= num_queries:
            raise ValueError("candidate query index is out of range")
        length = candidate.length
        source[index, :length] = torch.tensor(candidate.node_indices[:-1], device=target)
        destination[index, :length] = torch.tensor(candidate.node_indices[1:], device=target)
        facts[index, :length] = torch.tensor(candidate.fact_indices, device=target)
        directions[index, :length] = torch.tensor(candidate.directions, device=target)
        mask[index, :length] = True
        lengths[index] = length
        queries[index] = candidate.query_index
        path_ids.append(candidate.path_id)
    counts = torch.bincount(queries, minlength=num_queries) if path_count else torch.zeros(
        num_queries, dtype=torch.long, device=target
    )
    packed = PackedPathBatch(
        step_source_index=source,
        step_target_index=destination,
        step_fact_index=facts,
        step_direction=directions,
        step_mask=mask,
        path_length=lengths,
        query_index=queries,
        query_path_count=counts,
        path_ids=tuple(path_ids),
        max_length=max_length,
    )
    return packed
