"""Split-aware graph views and canonical target-answer masking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import polars as pl


DDA_CLINICAL_RELATIONS = frozenset(
    {"INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE"}
)


class LeakageError(RuntimeError):
    """Raised when target or split information survives a protected view."""


@dataclass(frozen=True, slots=True)
class GraphFrameView:
    facts: pl.DataFrame
    evidence: pl.DataFrame
    pair_features: pl.DataFrame = field(default_factory=pl.DataFrame)


@dataclass(frozen=True, slots=True)
class TargetMaskAudit:
    target_answer_facts_before: int
    target_answer_facts_after: int
    target_evidence_records_removed: int
    target_pair_feature_rows_removed: int
    reverse_facts_removed: int
    equivalent_alias_facts_removed: int


@dataclass(frozen=True, slots=True)
class SplitViewAudit:
    nontrain_dda_facts_removed: int
    heldout_entity_incident_facts_removed: int
    evidence_records_removed: int


@dataclass(frozen=True, slots=True)
class ExtractedSubgraph:
    node_ids: tuple[int, ...]
    fact_ids: tuple[int, ...]
    traversal_edges: tuple[tuple[int, int, int], ...]


def _pair_expr(drug_id: str, disease_id: str) -> pl.Expr:
    forward = (
        (pl.col("source_node_type") == "DRG")
        & (pl.col("source_node_id") == drug_id)
        & (pl.col("target_node_type") == "DIS")
        & (pl.col("target_node_id") == disease_id)
    )
    reverse = (
        (pl.col("source_node_type") == "DIS")
        & (pl.col("source_node_id") == disease_id)
        & (pl.col("target_node_type") == "DRG")
        & (pl.col("target_node_id") == drug_id)
    )
    return forward | reverse


def _canonical_relation_series(
    facts: pl.DataFrame, aliases: Mapping[str, str]
) -> pl.Series:
    return pl.Series(
        "_canonical_relation",
        [aliases.get(str(value), str(value)) for value in facts["relation"]],
        dtype=pl.String,
    )


def _target_answer_rows(
    facts: pl.DataFrame,
    *,
    drug_id: str,
    disease_id: str,
    relation_aliases: Mapping[str, str] | None = None,
) -> pl.DataFrame:
    aliases = relation_aliases or {}
    return facts.with_columns(_canonical_relation_series(facts, aliases)).filter(
        _pair_expr(drug_id, disease_id)
        & pl.col("_canonical_relation").is_in(DDA_CLINICAL_RELATIONS)
    )


def mask_target_pair(
    view: GraphFrameView,
    *,
    drug_id: str,
    disease_id: str,
    relation_aliases: Mapping[str, str] | None = None,
) -> tuple[GraphFrameView, TargetMaskAudit]:
    """Remove an entire target clinical fact, evidence, and cached pair features."""

    aliases = relation_aliases or {}
    target = _target_answer_rows(
        view.facts,
        drug_id=drug_id,
        disease_id=disease_id,
        relation_aliases=aliases,
    )
    fact_ids = target["canonical_fact_id"].to_list()
    reverse_removed = target.filter(
        pl.col("source_node_type") == "DIS"
    ).height
    alias_removed = sum(
        1
        for raw in target["relation"].to_list()
        if aliases.get(str(raw), str(raw)) != str(raw)
    )
    facts = view.facts.filter(~pl.col("canonical_fact_id").is_in(fact_ids))
    evidence_before = view.evidence.height
    evidence = view.evidence.filter(~pl.col("canonical_fact_id").is_in(fact_ids))
    pair_features = view.pair_features
    feature_removed = 0
    if {"drug_id", "disease_id"}.issubset(pair_features.columns):
        pair_feature_mask = (pl.col("drug_id") == drug_id) & (
            pl.col("disease_id") == disease_id
        )
        feature_removed = pair_features.filter(pair_feature_mask).height
        pair_features = pair_features.filter(~pair_feature_mask)
    masked = GraphFrameView(
        facts=facts,
        evidence=evidence,
        pair_features=pair_features,
    )
    remaining = _target_answer_rows(
        masked.facts,
        drug_id=drug_id,
        disease_id=disease_id,
        relation_aliases=aliases,
    ).height
    audit = TargetMaskAudit(
        target_answer_facts_before=target.height,
        target_answer_facts_after=remaining,
        target_evidence_records_removed=evidence_before - evidence.height,
        target_pair_feature_rows_removed=feature_removed,
        reverse_facts_removed=reverse_removed,
        equivalent_alias_facts_removed=alias_removed,
    )
    assert_no_target_answer(
        masked,
        drug_id=drug_id,
        disease_id=disease_id,
        relation_aliases=aliases,
    )
    return masked, audit


def assert_no_target_answer(
    view: GraphFrameView,
    *,
    drug_id: str,
    disease_id: str,
    relation_aliases: Mapping[str, str] | None = None,
) -> None:
    """Assert that neither answer facts nor target-keyed feature rows remain."""

    target = _target_answer_rows(
        view.facts,
        drug_id=drug_id,
        disease_id=disease_id,
        relation_aliases=relation_aliases,
    )
    if target.height:
        raise LeakageError(f"Target clinical facts remain: {target.height}")
    if {"drug_id", "disease_id"}.issubset(view.pair_features.columns):
        if view.pair_features.filter(
            (pl.col("drug_id") == drug_id)
            & (pl.col("disease_id") == disease_id)
        ).height:
            raise LeakageError("Target-keyed cached feature row remains")
    dangling = set(view.evidence["canonical_fact_id"].to_list()) - set(
        view.facts["canonical_fact_id"].to_list()
    )
    if dangling:
        raise LeakageError(f"Evidence references removed facts: {len(dangling)}")


def _dda_fact_expr() -> pl.Expr:
    typed_pair = (
        ((pl.col("source_node_type") == "DRG") & (pl.col("target_node_type") == "DIS"))
        | ((pl.col("source_node_type") == "DIS") & (pl.col("target_node_type") == "DRG"))
    )
    return typed_pair & pl.col("relation").is_in(DDA_CLINICAL_RELATIONS)


def _with_dda_endpoints(facts: pl.DataFrame) -> pl.DataFrame:
    return facts.with_columns(
        pl.when(pl.col("source_node_type") == "DRG")
        .then(pl.col("source_node_id"))
        .otherwise(pl.col("target_node_id"))
        .alias("_drug_id"),
        pl.when(pl.col("source_node_type") == "DIS")
        .then(pl.col("source_node_id"))
        .otherwise(pl.col("target_node_id"))
        .alias("_disease_id"),
    )


def _incident_to(
    *, node_type: str, identifiers: set[str]
) -> pl.Expr:
    values = sorted(identifiers)
    return (
        ((pl.col("source_node_type") == node_type) & pl.col("source_node_id").is_in(values))
        | ((pl.col("target_node_type") == node_type) & pl.col("target_node_id").is_in(values))
    )


def _heldout_identifiers(
    split_manifest: pl.DataFrame, *, entity: str
) -> set[str]:
    partition_column = (
        "drug_partition" if entity == "drug_id" else "disease_partition"
    )
    if partition_column in split_manifest.columns:
        rows = split_manifest.filter(
            pl.col(partition_column).is_in(["validation", "test"])
        )
    else:
        rows = split_manifest.filter(
            pl.col("split_assignment").is_in(["validation", "test"])
        )
    return set(rows[entity].to_list())


def training_forbidden_facts(
    facts: pl.DataFrame, split_manifest: pl.DataFrame
) -> pl.DataFrame:
    """Return one auditable exclusion record per training-hidden fact.

    The returned list-valued reasons are suitable for a compact graph-view
    index: consumers exclude the fact ID and its attached evidence/provenance
    records without copying the retained graph.
    """

    kinds = set(split_manifest["split_kind"].unique().to_list())
    if len(kinds) != 1:
        raise LeakageError("Split manifest must contain exactly one split kind")
    kind = str(next(iter(kinds)))
    reason_rows: list[pl.DataFrame] = []
    dda = _with_dda_endpoints(facts.filter(_dda_fact_expr())).join(
        split_manifest.select(
            pl.col("drug_id").alias("_drug_id"),
            pl.col("disease_id").alias("_disease_id"),
            "split_assignment",
        ),
        on=["_drug_id", "_disease_id"],
        how="left",
        validate="m:1",
    )
    nontrain = dda.filter(
        (pl.col("split_assignment") != "train").fill_null(True)
    )
    if nontrain.height:
        reason_rows.append(
            nontrain.select("canonical_fact_id").with_columns(
                pl.lit("NONTRAIN_DDA_ANSWER").alias("exclusion_reason")
            )
        )

    if kind in {"cold_drug", "double_cold"}:
        heldout_drugs = _heldout_identifiers(
            split_manifest, entity="drug_id"
        )
        incident = facts.filter(
            _incident_to(node_type="DRG", identifiers=heldout_drugs)
        )
        if incident.height:
            reason_rows.append(
                incident.select("canonical_fact_id").with_columns(
                    pl.lit("HELDOUT_DRUG_INCIDENT").alias(
                        "exclusion_reason"
                    )
                )
            )
    if kind in {"cold_disease", "double_cold"}:
        heldout_diseases = _heldout_identifiers(
            split_manifest, entity="disease_id"
        )
        incident = facts.filter(
            _incident_to(node_type="DIS", identifiers=heldout_diseases)
        )
        if incident.height:
            reason_rows.append(
                incident.select("canonical_fact_id").with_columns(
                    pl.lit("HELDOUT_DISEASE_INCIDENT").alias(
                        "exclusion_reason"
                    )
                )
            )
    if not reason_rows:
        return pl.DataFrame(
            schema={
                "canonical_fact_id": pl.UInt64,
                "exclusion_reasons": pl.List(pl.String),
            }
        )
    return (
        pl.concat(reason_rows)
        .group_by("canonical_fact_id")
        .agg(
            pl.col("exclusion_reason")
            .unique()
            .sort()
            .alias("exclusion_reasons")
        )
        .sort("canonical_fact_id")
    )


def build_training_visible_view(
    view: GraphFrameView,
    split_manifest: pl.DataFrame,
    *,
    exclude_heldout_entity_incident_edges: bool = True,
) -> tuple[GraphFrameView, SplitViewAudit]:
    """Create a strict training view without validation/test DDA answers."""

    kinds = set(split_manifest["split_kind"].unique().to_list())
    if len(kinds) != 1:
        raise LeakageError("Split manifest must contain exactly one split kind")
    kind = str(next(iter(kinds)))
    forbidden = training_forbidden_facts(view.facts, split_manifest)
    if not exclude_heldout_entity_incident_edges and not forbidden.is_empty():
        forbidden = forbidden.with_columns(
            pl.col("exclusion_reasons")
            .list.eval(
                pl.element().filter(pl.element() == "NONTRAIN_DDA_ANSWER")
            )
            .alias("exclusion_reasons")
        ).filter(pl.col("exclusion_reasons").list.len() > 0)
    forbidden_ids = forbidden["canonical_fact_id"].to_list()
    facts = view.facts.filter(
        ~pl.col("canonical_fact_id").is_in(forbidden_ids)
    )
    nontrain_removed = forbidden.filter(
        pl.col("exclusion_reasons").list.contains("NONTRAIN_DDA_ANSWER")
    ).height
    heldout_removed = forbidden.filter(
        (
            pl.col("exclusion_reasons").list.contains(
                "HELDOUT_DRUG_INCIDENT"
            )
            | pl.col("exclusion_reasons").list.contains(
                "HELDOUT_DISEASE_INCIDENT"
            )
        )
        & ~pl.col("exclusion_reasons").list.contains(
            "NONTRAIN_DDA_ANSWER"
        )
    ).height

    retained_fact_ids = facts["canonical_fact_id"].to_list()
    evidence = view.evidence.filter(
        pl.col("canonical_fact_id").is_in(retained_fact_ids)
    )
    pair_features = view.pair_features
    if {"drug_id", "disease_id"}.issubset(pair_features.columns):
        pair_features = pair_features.join(
            split_manifest.select("drug_id", "disease_id", "split_assignment"),
            on=["drug_id", "disease_id"],
            how="left",
        ).filter(pl.col("split_assignment") == "train").drop("split_assignment")
    result = GraphFrameView(
        facts=facts.sort("canonical_fact_id"),
        evidence=evidence.sort("evidence_record_id"),
        pair_features=pair_features,
    )
    return result, SplitViewAudit(
        nontrain_dda_facts_removed=nontrain_removed,
        heldout_entity_incident_facts_removed=heldout_removed,
        evidence_records_removed=view.evidence.height - evidence.height,
    )


def extract_csr_subgraph(
    csr_dir: str | Path,
    *,
    seed_node_ids: list[int],
    max_hops: int,
    max_nodes: int,
    max_edges: int,
    forbidden_fact_ids: set[int] | None = None,
) -> ExtractedSubgraph:
    """Deterministically traverse a bounded memory-mapped CSR graph."""

    if max_hops < 0 or max_nodes <= 0 or max_edges <= 0:
        raise ValueError("Invalid bounded-subgraph limits")
    root = Path(csr_dir)
    indptr = np.load(root / "indptr.npy", mmap_mode="r")
    indices = np.load(root / "indices.npy", mmap_mode="r")
    fact_ids = np.load(root / "fact_ids.npy", mmap_mode="r")
    node_count = len(indptr) - 1
    seeds = sorted(set(int(value) for value in seed_node_ids))
    if any(value < 0 or value >= node_count for value in seeds):
        raise ValueError("Seed node ID is outside the CSR index")
    forbidden = forbidden_fact_ids or set()
    visited = set(seeds)
    frontier = seeds
    traversal: list[tuple[int, int, int]] = []
    observed_facts: list[int] = []
    observed_fact_set: set[int] = set()
    for _ in range(max_hops):
        next_frontier: list[int] = []
        for source in frontier:
            for position in range(int(indptr[source]), int(indptr[source + 1])):
                fact = int(fact_ids[position])
                if fact in forbidden:
                    continue
                target = int(indices[position])
                if target not in visited and len(visited) >= max_nodes:
                    continue
                traversal.append((source, target, fact))
                if fact not in observed_fact_set:
                    observed_fact_set.add(fact)
                    observed_facts.append(fact)
                if target not in visited:
                    visited.add(target)
                    next_frontier.append(target)
                if len(traversal) >= max_edges:
                    break
            if len(traversal) >= max_edges:
                break
        frontier = sorted(set(next_frontier))
        if not frontier or len(traversal) >= max_edges:
            break
    return ExtractedSubgraph(
        node_ids=tuple(sorted(visited)),
        fact_ids=tuple(observed_facts),
        traversal_edges=tuple(traversal),
    )
