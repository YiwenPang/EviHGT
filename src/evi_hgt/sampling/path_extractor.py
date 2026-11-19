"""Deterministic bounded beam search over a leakage-safe query graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

from evi_hgt.sampling.path_schema import (
    PathCandidate,
    PathExtractionConfig,
    PathSource,
    PathTraversal,
)


@dataclass(frozen=True, slots=True)
class _BeamState:
    node_indices: tuple[int, ...]
    fact_indices: tuple[int, ...]
    directions: tuple[int, ...]
    structural_score: float

    @property
    def last_node(self) -> int:
        return self.node_indices[-1]


class PathCandidateExtractor:
    """Extract node/fact-simple drug-to-disease candidates with hard bounds."""

    algorithm_version = "bounded-deterministic-beam-v1"

    def __init__(self, config: PathExtractionConfig) -> None:
        self.config = config

    def extract(self, source: PathSource, *, query_index: int = 0) -> tuple[PathCandidate, ...]:
        source.validate()
        if query_index < 0:
            raise ValueError("query_index must be non-negative")
        adjacency: dict[int, list[PathTraversal]] = {}
        for traversal in source.traversals:
            adjacency.setdefault(traversal.source_node_index, []).append(traversal)
        degree = {node: len(values) for node, values in adjacency.items()}

        def neighbor_key(traversal: PathTraversal) -> tuple[float, int, int, int]:
            fact = source.facts[traversal.fact_index]
            hub = math.log1p(degree.get(traversal.target_node_index, 0))
            return (
                hub if self.config.use_hub_penalty else 0.0,
                fact.canonical_fact_id,
                source.node_ids[traversal.target_node_index],
                traversal.direction,
            )

        for values in adjacency.values():
            unique = {
                (value.target_node_index, value.fact_index, value.direction): value
                for value in values
            }
            values[:] = sorted(unique.values(), key=neighbor_key)

        beam = (_BeamState((source.drug_node_index,), (), (), 0.0),)
        completed: dict[str, PathCandidate] = {}
        for _ in range(self.config.max_length):
            expanded: list[_BeamState] = []
            for state in beam:
                neighbors = adjacency.get(state.last_node, ())[
                    : self.config.max_neighbors_per_expansion
                ]
                for traversal in neighbors:
                    fact = source.facts[traversal.fact_index]
                    if traversal.target_node_index in state.node_indices:
                        continue
                    if traversal.fact_index in state.fact_indices:
                        continue
                    hub = math.log1p(degree.get(traversal.target_node_index, 0))
                    penalty = 1.0
                    if self.config.use_hub_penalty:
                        penalty += self.config.hub_penalty_weight * hub
                    candidate_state = _BeamState(
                        node_indices=(*state.node_indices, traversal.target_node_index),
                        fact_indices=(*state.fact_indices, traversal.fact_index),
                        directions=(*state.directions, traversal.direction),
                        structural_score=state.structural_score - penalty,
                    )
                    if traversal.target_node_index == source.disease_node_index:
                        if len(candidate_state.fact_indices) >= self.config.min_length:
                            candidate = self._candidate(source, candidate_state, query_index)
                            completed.setdefault(candidate.path_id, candidate)
                        continue
                    expanded.append(candidate_state)
            beam = tuple(
                sorted(expanded, key=lambda state: self._state_key(source, state))[
                    : self.config.beam_width
                ]
            )
            if not beam:
                break
        return tuple(
            sorted(completed.values(), key=self._candidate_key)[: self.config.max_paths]
        )

    @staticmethod
    def _state_key(source: PathSource, state: _BeamState) -> tuple[object, ...]:
        fact_ids = tuple(source.facts[index].canonical_fact_id for index in state.fact_indices)
        node_ids = tuple(source.node_ids[index] for index in state.node_indices)
        return (-state.structural_score, node_ids, fact_ids, state.directions)

    @staticmethod
    def _candidate_key(candidate: PathCandidate) -> tuple[object, ...]:
        return (
            -candidate.structural_score,
            candidate.node_ids,
            candidate.canonical_fact_ids,
            candidate.directions,
        )

    @staticmethod
    def _candidate(
        source: PathSource, state: _BeamState, query_index: int
    ) -> PathCandidate:
        facts = tuple(source.facts[index] for index in state.fact_indices)
        return PathCandidate.create(
            query_index=query_index,
            node_indices=state.node_indices,
            node_ids=tuple(source.node_ids[index] for index in state.node_indices),
            node_types=tuple(source.node_types[index] for index in state.node_indices),
            fact_indices=state.fact_indices,
            canonical_fact_ids=tuple(fact.canonical_fact_id for fact in facts),
            relations=tuple(fact.relation for fact in facts),
            directions=state.directions,
            structural_score=state.structural_score,
        )

