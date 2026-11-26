"""Export exact forward paths without post-hoc path construction."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch

from evi_hgt.models.path_reasoning import PathReasoningOutput
from evi_hgt.sampling.path_schema import PackedPathBatch, PathCandidate


@dataclass(frozen=True, slots=True)
class FactExplanationContext:
    canonical_fact_id: int
    provenance: tuple[str, ...]
    logic_state_available: bool
    logic_masses: tuple[float, float, float, float] | None

    def __post_init__(self) -> None:
        if self.canonical_fact_id < 0:
            raise ValueError("fact explanation ID must be non-negative")
        if self.logic_state_available:
            if self.logic_masses is None or len(self.logic_masses) != 4:
                raise ValueError("available edge logic requires four masses")
            if abs(sum(self.logic_masses) - 1.0) > 1e-5:
                raise ValueError("available edge logic masses must sum to one")
        elif self.logic_masses is not None:
            raise ValueError("logic-unavailable edge must not manufacture U or other masses")


@dataclass(frozen=True, slots=True)
class PathEdgeExplanation:
    canonical_fact_id: int
    relation: str
    traversal_direction: str
    provenance: tuple[str, ...]
    logic_state_available: bool
    logic_masses: tuple[float, float, float, float] | None


@dataclass(frozen=True, slots=True)
class PathExplanation:
    query_pair_id: str
    drug_id: str
    disease_id: str
    path_id: str
    path_length: int
    node_ids: tuple[int, ...]
    node_types: tuple[str, ...]
    canonical_fact_ids: tuple[int, ...]
    relations: tuple[str, ...]
    traversal_directions: tuple[str, ...]
    path_score: float
    beta: float
    step_weights: tuple[float, ...]
    path_available: bool
    edges: tuple[PathEdgeExplanation, ...]


def _validate_forward_alignment(
    index: int,
    candidate: PathCandidate,
    packed: PackedPathBatch,
) -> None:
    if candidate.path_id != packed.path_ids[index]:
        raise ValueError("candidate path ID does not match the packed forward input")
    if candidate.query_index != int(packed.query_index[index]):
        raise ValueError("candidate query does not match the packed forward input")
    length = candidate.length
    observed_source = tuple(int(value) for value in packed.step_source_index[index, :length])
    observed_target = tuple(int(value) for value in packed.step_target_index[index, :length])
    observed_fact = tuple(int(value) for value in packed.step_fact_index[index, :length])
    observed_direction = tuple(int(value) for value in packed.step_direction[index, :length])
    if observed_source != candidate.node_indices[:-1]:
        raise ValueError("candidate source steps do not match packed forward input")
    if observed_target != candidate.node_indices[1:]:
        raise ValueError("candidate target steps do not match packed forward input")
    if observed_fact != candidate.fact_indices:
        raise ValueError("candidate fact IDs do not match packed forward input")
    if observed_direction != candidate.directions:
        raise ValueError("candidate directions do not match packed forward input")


def build_path_explanations(
    candidates: tuple[PathCandidate, ...],
    packed: PackedPathBatch,
    forward: PathReasoningOutput,
    *,
    query_pair_ids: tuple[str, ...],
    drug_ids: tuple[str, ...],
    disease_ids: tuple[str, ...],
    fact_context: Mapping[int, FactExplanationContext],
    top_k: int | None = 5,
) -> tuple[PathExplanation, ...]:
    if len(candidates) != packed.num_paths:
        raise ValueError("candidate count does not match packed forward input")
    if not all(len(values) == packed.num_queries for values in (query_pair_ids, drug_ids, disease_ids)):
        raise ValueError("query explanation metadata count does not match packed queries")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when supplied")
    expected_path = (packed.num_paths,)
    if forward.path_score.shape != expected_path or forward.beta.shape != expected_path:
        raise ValueError("forward path score/beta shape does not match packed paths")
    if forward.step_weights.shape != (packed.num_paths, packed.max_length):
        raise ValueError("forward gamma shape does not match packed paths")
    if forward.path_available.shape != (packed.num_queries,):
        raise ValueError("forward path availability shape does not match packed queries")

    built: list[PathExplanation] = []
    for index, candidate in enumerate(candidates):
        _validate_forward_alignment(index, candidate, packed)
        contexts: list[FactExplanationContext] = []
        for fact_id in candidate.canonical_fact_ids:
            context = fact_context.get(fact_id)
            if context is None or context.canonical_fact_id != fact_id:
                raise ValueError("missing or mismatched fact explanation context")
            contexts.append(context)
        directions = tuple(
            "forward" if direction == 0 else "reverse"
            for direction in candidate.directions
        )
        edges = tuple(
            PathEdgeExplanation(
                canonical_fact_id=fact_id,
                relation=relation,
                traversal_direction=direction,
                provenance=context.provenance,
                logic_state_available=context.logic_state_available,
                logic_masses=context.logic_masses,
            )
            for fact_id, relation, direction, context in zip(
                candidate.canonical_fact_ids,
                candidate.relations,
                directions,
                contexts,
                strict=True,
            )
        )
        query_index = candidate.query_index
        built.append(
            PathExplanation(
                query_pair_id=query_pair_ids[query_index],
                drug_id=drug_ids[query_index],
                disease_id=disease_ids[query_index],
                path_id=candidate.path_id,
                path_length=candidate.length,
                node_ids=candidate.node_ids,
                node_types=candidate.node_types,
                canonical_fact_ids=candidate.canonical_fact_ids,
                relations=candidate.relations,
                traversal_directions=directions,
                path_score=float(forward.path_score[index].detach().float().cpu()),
                beta=float(forward.beta[index].detach().float().cpu()),
                step_weights=tuple(
                    float(value)
                    for value in forward.step_weights[index, : candidate.length]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                ),
                path_available=bool(forward.path_available[query_index]),
                edges=edges,
            )
        )
    if top_k is None:
        return tuple(built)
    selected: list[PathExplanation] = []
    for query_index in range(packed.num_queries):
        query_paths = [
            value
            for value, candidate in zip(built, candidates, strict=True)
            if candidate.query_index == query_index
        ]
        selected.extend(sorted(query_paths, key=lambda value: (-value.beta, value.path_id))[:top_k])
    return tuple(selected)


def write_path_explanations(
    path: str | Path, explanations: tuple[PathExplanation, ...]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_explanation_is_causal_proof": False,
        "scientific_boundary": (
            "A high path weight indicates model-assigned relevance for the current "
            "query under the implemented architecture; it is not causal proof."
        ),
        "paths": [asdict(value) for value in explanations],
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination.resolve()

