"""Candidate-level leakage checks after path extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from evi_hgt.data.leakage import DDA_CLINICAL_RELATIONS
from evi_hgt.sampling.path_schema import PathCandidate


class PathLeakageError(RuntimeError):
    """Raised when a path candidate violates query-graph exclusions."""


@dataclass(frozen=True, slots=True)
class PathLeakagePolicy:
    split_kind: str
    query_pair_id: str
    drug_node_id: int
    disease_node_id: int
    forbidden_fact_ids: frozenset[int]
    heldout_node_ids: frozenset[int]
    target_relation_aliases: Mapping[str, str]
    target_metadata_fact_ids: frozenset[int]
    cache_split_kind: str | None = None


@dataclass(frozen=True, slots=True)
class PathLeakageAudit:
    passed: bool
    split_kind: str
    query_pair_id: str
    candidate_count: int
    fact_count: int
    target_relation_survivor_count: int
    forbidden_fact_survivor_count: int
    target_metadata_survivor_count: int
    heldout_entity_survivor_count: int
    cache_split_mismatch_count: int


def audit_path_leakage(
    candidates: tuple[PathCandidate, ...], policy: PathLeakagePolicy
) -> PathLeakageAudit:
    if policy.cache_split_kind is not None and policy.cache_split_kind != policy.split_kind:
        raise PathLeakageError("path cache split does not match query split")
    fact_count = 0
    for candidate in candidates:
        fact_count += candidate.length
        if policy.heldout_node_ids.intersection(candidate.node_ids):
            raise PathLeakageError("path contains a held-out entity")
        if policy.forbidden_fact_ids.intersection(candidate.canonical_fact_ids):
            raise PathLeakageError("path contains a split or target forbidden fact")
        if policy.target_metadata_fact_ids.intersection(candidate.canonical_fact_ids):
            raise PathLeakageError("path contains target metadata or provenance")
        for index, relation in enumerate(candidate.relations):
            canonical_relation = policy.target_relation_aliases.get(relation, relation)
            endpoints = {
                candidate.node_ids[index],
                candidate.node_ids[index + 1],
            }
            target_endpoints = {policy.drug_node_id, policy.disease_node_id}
            if endpoints == target_endpoints and canonical_relation in DDA_CLINICAL_RELATIONS:
                raise PathLeakageError("path contains target-pair clinical relation")
    return PathLeakageAudit(
        passed=True,
        split_kind=policy.split_kind,
        query_pair_id=policy.query_pair_id,
        candidate_count=len(candidates),
        fact_count=fact_count,
        target_relation_survivor_count=0,
        forbidden_fact_survivor_count=0,
        target_metadata_survivor_count=0,
        heldout_entity_survivor_count=0,
        cache_split_mismatch_count=0,
    )

