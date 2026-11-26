"""Schema-aware Phase 2.5 scientific-gate statistics.

The functions in this module operate only on verified OptimusKG relation fields.
They do not create training labels, and they never interpret a missing value as
negative evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl


EXPRESSION_PRESENT = "EXPRESSION_PRESENT"
EXPRESSION_ABSENT = "EXPRESSION_ABSENT"

INDICATION = "INDICATION"
CONTRAINDICATION = "CONTRAINDICATION"
OFF_LABEL_USE = "OFF_LABEL_USE"

POLICY_DETAILS: dict[str, dict[str, str]] = {
    "A_CONSERVATIVE": {
        "support_rule": (
            "Final INDICATION with DRUG_CENTRAL in direct provenance."
        ),
        "expected_label_noise": (
            "Lowest of the three candidates, but not zero because release-level "
            "relation collapse and source-version alignment remain unresolved."
        ),
        "expected_source_bias": "Severe DrugCentral curation/coverage bias.",
        "expected_cold_start_consequence": (
            "Highest cold-start sparsity; many drugs and diseases lose support labels."
        ),
        "potential_leakage_risk": (
            "Remove every target-pair DDA edge, reverse traversal, relation, and label-"
            "defining property from query subgraphs."
        ),
        "strengths": "Clearest retained therapeutic indication semantics.",
        "weaknesses": "Small support set and narrow entity coverage.",
    },
    "B_MODERATE": {
        "support_rule": (
            "Policy A, plus final INDICATION with highest_clinical_trial_phase >= 2."
        ),
        "expected_label_noise": (
            "Moderate: phase >= 2 is a proposed gate, not a verified OptimusKG "
            "strong/weak-evidence boundary."
        ),
        "expected_source_bias": (
            "DrugCentral plus strong Open Targets/clinical-development bias."
        ),
        "expected_cold_start_consequence": (
            "Moderate sparsity with substantially broader entity coverage than A."
        ),
        "potential_leakage_risk": (
            "Clinical phase defines the label policy and must be removed with the "
            "target edge and equivalent reverse traversal during prediction."
        ),
        "strengths": "Balances coverage with a transparent clinical-development gate.",
        "weaknesses": (
            "Threshold requires sensitivity analysis; trial phase is not approval or "
            "proof of efficacy."
        ),
    },
    "C_BROAD": {
        "support_rule": "Every final INDICATION pair.",
        "expected_label_noise": (
            "Highest: includes early, unknown, and heterogeneous clinical-development "
            "contexts that the upstream TODO does not stratify."
        ),
        "expected_source_bias": "Dominated by Open Targets coverage.",
        "expected_cold_start_consequence": (
            "Lowest sparsity and widest entity coverage, but noisier cold-start labels."
        ),
        "potential_leakage_risk": (
            "All target-pair DDA relations and label-defining properties must be removed; "
            "broad source overlap can amplify provenance leakage."
        ),
        "strengths": "Maximum scientifically arguable therapeutic-context coverage.",
        "weaknesses": "Largest semantic heterogeneity and expected label noise.",
    },
}


@dataclass(slots=True)
class ExpressionAuditResult:
    """Machine-readable expression audit outputs."""

    summary: dict[str, Any]
    overlapping_pairs: pl.DataFrame
    source_distribution: pl.DataFrame
    property_distribution: pl.DataFrame


@dataclass(slots=True)
class DDAAuditResult:
    """Machine-readable Drug-Disease audit outputs."""

    summary: dict[str, Any]
    relation_statistics: pl.DataFrame
    source_distribution: pl.DataFrame
    phase_distribution: pl.DataFrame
    degree_distribution: pl.DataFrame
    policy_comparison: pl.DataFrame
    policy_source_distribution: pl.DataFrame
    policy_phase_distribution: pl.DataFrame


def _require_columns(frame: pl.DataFrame, names: Iterable[str]) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _struct_field_names(dtype: pl.DataType) -> set[str]:
    if not isinstance(dtype, pl.Struct):
        return set()
    return {field.name for field in dtype.fields}


def _empty_overlap_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "anatomy_id": pl.String,
            "gene_id": pl.String,
            "present_edge_count": pl.UInt32,
            "absent_edge_count": pl.UInt32,
            "present_fingerprints_json": pl.String,
            "absent_fingerprints_json": pl.String,
            "classification": pl.String,
        }
    )


def _source_distribution(
    frame: pl.DataFrame,
    *,
    relation_column: str = "relation",
    direct_column: str = "direct_sources",
    indirect_column: str = "indirect_sources",
) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for role, column in (("direct", direct_column), ("indirect", indirect_column)):
        selected = frame.select(
            pl.col(relation_column).cast(pl.String).alias("relation"),
            pl.col(column)
            .fill_null(pl.lit([]).cast(pl.List(pl.String)))
            .alias("source_values"),
        )
        values = (
            selected.explode("source_values", empty_as_null=True)
            .filter(pl.col("source_values").is_not_null())
            .group_by(["relation", "source_values"])
            .agg(pl.len().alias("edge_mentions"))
            .rename({"source_values": "source"})
            .with_columns(pl.lit(role).alias("source_role"))
        )
        missing = (
            selected.filter(pl.col("source_values").list.len() == 0)
            .group_by("relation")
            .agg(pl.len().alias("edge_mentions"))
            .with_columns(
                pl.lit("__MISSING_OR_EMPTY__").alias("source"),
                pl.lit(role).alias("source_role"),
            )
        )
        rows.extend([values, missing])
    nonempty = [item for item in rows if item.height]
    if not nonempty:
        return pl.DataFrame(
            schema={
                "relation": pl.String,
                "source_role": pl.String,
                "source": pl.String,
                "edge_mentions": pl.UInt32,
            }
        )
    return (
        pl.concat(nonempty, how="diagonal_relaxed")
        .select("relation", "source_role", "source", "edge_mentions")
        .sort(["relation", "source_role", "source"])
    )


def _property_distribution(
    frame: pl.DataFrame,
    properties: Iterable[str],
) -> pl.DataFrame:
    rows: list[pl.DataFrame] = []
    for name in properties:
        if name not in frame.columns:
            continue
        rows.append(
            frame.select(
                pl.col("relation"),
                pl.col(name)
                .cast(pl.String, strict=False)
                .fill_null("__MISSING__")
                .alias("value"),
            )
            .group_by(["relation", "value"])
            .agg(pl.len().alias("edge_count"))
            .with_columns(pl.lit(name).alias("property"))
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "relation": pl.String,
                "property": pl.String,
                "value": pl.String,
                "edge_count": pl.UInt32,
            }
        )
    return (
        pl.concat(rows, how="diagonal_relaxed")
        .select("relation", "property", "value", "edge_count")
        .sort(["relation", "property", "value"])
    )


def _expression_fingerprint(row: dict[str, Any]) -> str:
    value = {
        "direct_sources": sorted(row.get("direct_sources") or []),
        "indirect_sources": sorted(row.get("indirect_sources") or []),
        "expression_rank": row.get("expression_rank"),
        "call_quality": row.get("call_quality"),
    }
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def analyze_expression_edges(frame: pl.DataFrame) -> ExpressionAuditResult:
    """Audit PRESENT/ABSENT states for identical canonical anatomy-gene pairs."""

    _require_columns(frame, ["from", "to", "relation", "properties"])
    property_fields = _struct_field_names(frame.schema["properties"])
    if "sources" not in property_fields:
        raise ValueError("properties.sources is required for the expression audit")

    work = frame.with_columns(
        pl.col("properties")
        .struct.field("sources")
        .struct.field("direct")
        .alias("direct_sources"),
        pl.col("properties")
        .struct.field("sources")
        .struct.field("indirect")
        .alias("indirect_sources"),
        (
            pl.col("properties").struct.field("expression_rank")
            if "expression_rank" in property_fields
            else pl.lit(None, dtype=pl.Int32)
        ).alias("expression_rank"),
        (
            pl.col("properties").struct.field("call_quality")
            if "call_quality" in property_fields
            else pl.lit(None, dtype=pl.String)
        ).alias("call_quality"),
    )
    present_rows = work.filter(pl.col("relation") == EXPRESSION_PRESENT)
    absent_rows = work.filter(pl.col("relation") == EXPRESSION_ABSENT)
    present_pairs = present_rows.select("from", "to").unique()
    absent_pairs = absent_rows.select("from", "to").unique()
    overlap = present_pairs.join(absent_pairs, on=["from", "to"], how="inner")

    if overlap.is_empty():
        overlapping_pairs = _empty_overlap_frame()
    else:
        overlap_rows = work.join(overlap, on=["from", "to"], how="inner")
        grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
        for row in overlap_rows.iter_rows(named=True):
            pair = (str(row["from"]), str(row["to"]))
            grouped.setdefault(pair, {EXPRESSION_PRESENT: [], EXPRESSION_ABSENT: []})
            relation = row["relation"]
            if relation in grouped[pair]:
                grouped[pair][relation].append(_expression_fingerprint(row))

        records: list[dict[str, Any]] = []
        for (anatomy_id, gene_id), evidence in sorted(grouped.items()):
            present_fingerprints = sorted(set(evidence[EXPRESSION_PRESENT]))
            absent_fingerprints = sorted(set(evidence[EXPRESSION_ABSENT]))
            classification = (
                "DUPLICATE_OR_NOT_INDEPENDENT_CONFLICT"
                if present_fingerprints == absent_fingerprints
                else "BOTH_DISTINCT_AVAILABLE_EVIDENCE"
            )
            records.append(
                {
                    "anatomy_id": anatomy_id,
                    "gene_id": gene_id,
                    "present_edge_count": len(evidence[EXPRESSION_PRESENT]),
                    "absent_edge_count": len(evidence[EXPRESSION_ABSENT]),
                    "present_fingerprints_json": json.dumps(
                        present_fingerprints, ensure_ascii=False
                    ),
                    "absent_fingerprints_json": json.dumps(
                        absent_fingerprints, ensure_ascii=False
                    ),
                    "classification": classification,
                }
            )
        overlapping_pairs = pl.DataFrame(records)

    genuine_both = int(
        overlapping_pairs.filter(
            pl.col("classification") == "BOTH_DISTINCT_AVAILABLE_EVIDENCE"
        ).height
    )
    duplicate_conflict = int(overlapping_pairs.height - genuine_both)
    present_pair_count = present_pairs.height
    absent_pair_count = absent_pairs.height
    overlap_count = overlap.height
    known_rows = work.filter(
        pl.col("relation").is_in([EXPRESSION_PRESENT, EXPRESSION_ABSENT])
    )
    pair_rows = known_rows.group_by(["from", "to"]).agg(
        pl.len().alias("row_count"),
        pl.col("relation").n_unique().alias("relation_count"),
    )
    summary: dict[str, Any] = {
        "row_count": frame.height,
        "present_edge_count": present_rows.height,
        "absent_edge_count": absent_rows.height,
        "unknown_or_other_edge_count": frame.height
        - present_rows.height
        - absent_rows.height,
        "present_pair_count": present_pair_count,
        "absent_pair_count": absent_pair_count,
        "overlap_pair_count": overlap_count,
        "overlap_percentage_of_present": (
            100.0 * overlap_count / present_pair_count if present_pair_count else 0.0
        ),
        "overlap_percentage_of_absent": (
            100.0 * overlap_count / absent_pair_count if absent_pair_count else 0.0
        ),
        "support_only_pair_count": present_pair_count - overlap_count,
        "refutation_only_pair_count": absent_pair_count - overlap_count,
        "genuine_both_pair_count": genuine_both,
        "duplicate_not_independent_conflict_pair_count": duplicate_conflict,
        "unique_known_pair_count": present_pairs.join(
            absent_pairs, on=["from", "to"], how="full", coalesce=True
        ).height,
        "pairs_with_duplicate_rows": pair_rows.filter(
            pl.col("row_count") > pl.col("relation_count")
        ).height,
        "verified_property_fields": sorted(property_fields),
        "reference_field_status": (
            "PRESENT" if "references" in property_fields else "NOT PRESENT"
        ),
        "observed_both_conclusion": (
            "GENUINE OBSERVED BOTH EXISTS IN THE AVAILABLE FIELDS"
            if genuine_both
            else "NO OBSERVED PRESENT/ABSENT PAIR-LEVEL BOTH IN THE PUBLISHED TABLE"
        ),
    }
    return ExpressionAuditResult(
        summary=summary,
        overlapping_pairs=overlapping_pairs,
        source_distribution=_source_distribution(known_rows),
        property_distribution=_property_distribution(
            known_rows, ["expression_rank", "call_quality"]
        ),
    )


def _degree_summary(frame: pl.DataFrame, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column, entity_type in (("from", "drug"), ("to", "disease")):
        degree = frame.group_by(column).agg(pl.len().alias("degree"))
        if degree.is_empty():
            rows.append(
                {
                    "scope": scope,
                    "entity_type": entity_type,
                    "entity_count": 0,
                    "degree_min": None,
                    "degree_mean": None,
                    "degree_p50": None,
                    "degree_p90": None,
                    "degree_p95": None,
                    "degree_max": None,
                }
            )
            continue
        stats = degree.select(
            pl.len().alias("entity_count"),
            pl.col("degree").min().alias("degree_min"),
            pl.col("degree").mean().alias("degree_mean"),
            pl.col("degree").quantile(0.5, interpolation="nearest").alias("degree_p50"),
            pl.col("degree").quantile(0.9, interpolation="nearest").alias("degree_p90"),
            pl.col("degree").quantile(0.95, interpolation="nearest").alias("degree_p95"),
            pl.col("degree").max().alias("degree_max"),
        ).to_dicts()[0]
        rows.append({"scope": scope, "entity_type": entity_type, **stats})
    return rows


def _unique_pairs(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select("from", "to").unique()


def _phase_distribution(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={"relation": pl.String, "phase": pl.String, "edge_count": pl.UInt32}
        )
    return (
        frame.select(
            pl.col("relation"),
            pl.col("phase")
            .cast(pl.String, strict=False)
            .fill_null("__MISSING__")
            .alias("phase"),
        )
        .group_by(["relation", "phase"])
        .agg(pl.len().alias("edge_count"))
        .sort(["relation", "phase"])
    )


def _policy_frames(work: pl.DataFrame) -> dict[str, pl.DataFrame]:
    indication = pl.col("relation") == INDICATION
    drugcentral = pl.col("direct_sources").list.contains("DRUG_CENTRAL")
    phase_two_or_more = pl.col("phase").fill_null(-999.0) >= 2.0
    return {
        "A_CONSERVATIVE": work.filter(indication & drugcentral),
        "B_MODERATE": work.filter(indication & (drugcentral | phase_two_or_more)),
        "C_BROAD": work.filter(indication),
    }


def analyze_dda_edges(frame: pl.DataFrame) -> DDAAuditResult:
    """Audit DDA relations and calculate three non-applied label policies."""

    _require_columns(
        frame, ["from", "to", "relation", "undirected", "properties"]
    )
    property_fields = _struct_field_names(frame.schema["properties"])
    if "sources" not in property_fields:
        raise ValueError("properties.sources is required for the DDA audit")

    def property_expr(name: str, dtype: pl.DataType) -> pl.Expr:
        if name in property_fields:
            return pl.col("properties").struct.field(name)
        return pl.lit(None, dtype=dtype)

    work = frame.with_columns(
        pl.col("properties")
        .struct.field("sources")
        .struct.field("direct")
        .fill_null(pl.lit([]).cast(pl.List(pl.String)))
        .alias("direct_sources"),
        pl.col("properties")
        .struct.field("sources")
        .struct.field("indirect")
        .fill_null(pl.lit([]).cast(pl.List(pl.String)))
        .alias("indirect_sources"),
        property_expr("structure_id", pl.String).alias("structure_id"),
        property_expr("drug_disease_id", pl.String).alias("drug_disease_id"),
        property_expr("reference_ids", pl.List(pl.String))
        .fill_null(pl.lit([]).cast(pl.List(pl.String)))
        .alias("reference_ids"),
        property_expr("highest_clinical_trial_phase", pl.Float64).alias("phase"),
        property_expr("evidence_score", pl.Float64).alias("evidence_score"),
    )

    relation_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    for relation in sorted(
        str(value) for value in work["relation"].drop_nulls().unique().to_list()
    ):
        subset = work.filter(pl.col("relation") == relation)
        pairs = _unique_pairs(subset)
        pair_sizes = subset.group_by(["from", "to"]).agg(pl.len().alias("rows"))
        reference_lengths = subset["reference_ids"].list.len()
        evidence = subset["evidence_score"].drop_nulls()
        relation_rows.append(
            {
                "relation": relation,
                "edge_count": subset.height,
                "unique_pair_count": pairs.height,
                "duplicate_pair_count": pair_sizes.filter(pl.col("rows") > 1).height,
                "duplicate_edge_count": subset.height - pairs.height,
                "drug_count": subset["from"].n_unique(),
                "disease_count": subset["to"].n_unique(),
                "all_undirected": bool(subset["undirected"].all()),
                "direct_source_combinations": json.dumps(
                    {
                        row["combination"]: row["count"]
                        for row in (
                            subset.select(
                                pl.col("direct_sources")
                                .list.sort()
                                .list.join("|")
                                .replace("", "__MISSING_OR_EMPTY__")
                                .fill_null("__MISSING_OR_EMPTY__")
                                .alias("combination")
                            )
                            .group_by("combination")
                            .agg(pl.len().alias("count"))
                            .sort("combination")
                            .to_dicts()
                        )
                    },
                    sort_keys=True,
                ),
                "clinical_phase_non_null": subset["phase"].is_not_null().sum(),
                "clinical_phase_missing": subset["phase"].is_null().sum(),
                "clinical_phase_missing_percent": (
                    100.0 * subset["phase"].is_null().sum() / subset.height
                    if subset.height
                    else 0.0
                ),
                "clinical_phase_min": subset["phase"].min(),
                "clinical_phase_max": subset["phase"].max(),
                "reference_rows_non_empty": int((reference_lengths > 0).sum()),
                "reference_id_mentions": int(reference_lengths.sum()),
                "structure_id_non_null": subset["structure_id"].is_not_null().sum(),
                "drug_disease_id_non_null": subset[
                    "drug_disease_id"
                ].is_not_null().sum(),
                "evidence_score_status": (
                    "PRESENT" if "evidence_score" in property_fields else "NOT PRESENT"
                ),
                "evidence_score_non_null": evidence.len(),
                "evidence_score_min": evidence.min() if evidence.len() else None,
                "evidence_score_p50": (
                    evidence.quantile(0.5, interpolation="nearest")
                    if evidence.len()
                    else None
                ),
                "evidence_score_max": evidence.max() if evidence.len() else None,
            }
        )
        degree_rows.extend(_degree_summary(pairs, f"RELATION:{relation}"))

    policy_frames = _policy_frames(work)
    contraindication = work.filter(pl.col("relation") == CONTRAINDICATION)
    contraindication_pairs = _unique_pairs(contraindication)
    off_label = work.filter(pl.col("relation") == OFF_LABEL_USE)

    policy_rows: list[dict[str, Any]] = []
    policy_sources: list[pl.DataFrame] = []
    policy_phases: list[pl.DataFrame] = []
    for name, support in policy_frames.items():
        support_pairs = _unique_pairs(support)
        overlap_count = support_pairs.join(
            contraindication_pairs, on=["from", "to"], how="inner"
        ).height
        union_pairs = pl.concat(
            [support_pairs, contraindication_pairs], how="vertical"
        ).unique()
        support_degrees = _degree_summary(support_pairs, f"POLICY_SUPPORT:{name}")
        degree_rows.extend(support_degrees)
        degree_lookup = {row["entity_type"]: row for row in support_degrees}
        policy_rows.append(
            {
                "policy": name,
                "support_rule": POLICY_DETAILS[name]["support_rule"],
                "support_pair_count": support_pairs.height,
                "unsuitability_pair_count": contraindication_pairs.height,
                "observed_final_table_overlap_pair_count": overlap_count,
                "off_label_supervision_pair_count": 0,
                "support_drug_count": support_pairs["from"].n_unique(),
                "support_disease_count": support_pairs["to"].n_unique(),
                "supervision_union_drug_count": union_pairs["from"].n_unique(),
                "supervision_union_disease_count": union_pairs["to"].n_unique(),
                "support_drug_degree_mean": degree_lookup["drug"]["degree_mean"],
                "support_drug_degree_p50": degree_lookup["drug"]["degree_p50"],
                "support_drug_degree_max": degree_lookup["drug"]["degree_max"],
                "support_disease_degree_mean": degree_lookup["disease"]["degree_mean"],
                "support_disease_degree_p50": degree_lookup["disease"]["degree_p50"],
                "support_disease_degree_max": degree_lookup["disease"]["degree_max"],
                **{
                    key: POLICY_DETAILS[name][key]
                    for key in (
                        "expected_label_noise",
                        "expected_source_bias",
                        "expected_cold_start_consequence",
                        "potential_leakage_risk",
                        "strengths",
                        "weaknesses",
                    )
                },
            }
        )
        source_table = _source_distribution(support)
        if source_table.height:
            policy_sources.append(
                source_table.with_columns(pl.lit(name).alias("policy"))
            )
        phase_table = _phase_distribution(support)
        if phase_table.height:
            policy_phases.append(phase_table.with_columns(pl.lit(name).alias("policy")))

    indication_pairs = _unique_pairs(work.filter(pl.col("relation") == INDICATION))
    off_label_pairs = _unique_pairs(off_label)
    pair_sizes = work.group_by(["from", "to"]).agg(
        pl.len().alias("rows"), pl.col("relation").n_unique().alias("relations")
    )
    actual_relations = set(work["relation"].drop_nulls().to_list())
    summary: dict[str, Any] = {
        "row_count": work.height,
        "unique_pair_count": _unique_pairs(work).height,
        "duplicate_pair_count": pair_sizes.filter(pl.col("rows") > 1).height,
        "multi_relation_pair_count": pair_sizes.filter(pl.col("relations") > 1).height,
        "actual_relations": sorted(str(value) for value in actual_relations),
        "expected_relations_missing": sorted(
            {INDICATION, CONTRAINDICATION, OFF_LABEL_USE} - actual_relations
        ),
        "other_relation_edge_count": work.filter(
            ~pl.col("relation").is_in([INDICATION, CONTRAINDICATION, OFF_LABEL_USE])
        ).height,
        "off_label_context_pair_count": off_label_pairs.height,
        "observed_indication_contraindication_overlap_pair_count": indication_pairs.join(
            contraindication_pairs, on=["from", "to"], how="inner"
        ).height,
        "observed_indication_off_label_overlap_pair_count": indication_pairs.join(
            off_label_pairs, on=["from", "to"], how="inner"
        ).height,
        "observed_contraindication_off_label_overlap_pair_count": contraindication_pairs.join(
            off_label_pairs, on=["from", "to"], how="inner"
        ).height,
        "contraindication_with_open_targets_candidate_count": contraindication.filter(
            pl.col("direct_sources").list.contains("OPEN_TARGETS")
        ).height,
        "off_label_with_open_targets_candidate_count": off_label.filter(
            pl.col("direct_sources").list.contains("OPEN_TARGETS")
        ).height,
        "observed_dda_both_recoverable": False,
        "observed_dda_both_conclusion": (
            "OBSERVED DDA BOTH/CONFLICT IS NOT RELIABLY RECOVERABLE FROM THE "
            "PUBLISHED OPTIMUSKG RELEASE."
        ),
        "cross_source_candidate_warning": (
            "OPEN_TARGETS provenance on final CONTRAINDICATION/OFF_LABEL_USE rows is a "
            "candidate trace of collapsed therapeutic context, not an observed Both label."
        ),
        "verified_property_fields": sorted(property_fields),
        "evidence_score_status": (
            "PRESENT" if "evidence_score" in property_fields else "NOT PRESENT"
        ),
        "unknown_is_false": False,
    }

    policy_source_frame = (
        pl.concat(policy_sources, how="diagonal_relaxed")
        .select("policy", "relation", "source_role", "source", "edge_mentions")
        .sort(["policy", "source_role", "source"])
        if policy_sources
        else pl.DataFrame(
            schema={
                "policy": pl.String,
                "relation": pl.String,
                "source_role": pl.String,
                "source": pl.String,
                "edge_mentions": pl.UInt32,
            }
        )
    )
    policy_phase_frame = (
        pl.concat(policy_phases, how="diagonal_relaxed")
        .select("policy", "relation", "phase", "edge_count")
        .sort(["policy", "phase"])
        if policy_phases
        else pl.DataFrame(
            schema={
                "policy": pl.String,
                "relation": pl.String,
                "phase": pl.String,
                "edge_count": pl.UInt32,
            }
        )
    )
    return DDAAuditResult(
        summary=summary,
        relation_statistics=pl.DataFrame(relation_rows).sort("relation"),
        source_distribution=_source_distribution(work),
        phase_distribution=_phase_distribution(work),
        degree_distribution=pl.DataFrame(degree_rows).sort(
            ["scope", "entity_type"]
        ),
        policy_comparison=pl.DataFrame(policy_rows),
        policy_source_distribution=policy_source_frame,
        policy_phase_distribution=policy_phase_frame,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _format_count(value: Any) -> str:
    if value is None:
        return "NOT PRESENT"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _expression_report(
    result: ExpressionAuditResult,
    input_integrity: dict[str, Any] | None,
) -> str:
    summary = result.summary
    integrity = input_integrity or {}
    integrity_rows = "\n".join(
        f"- {key}: `{value}`" for key, value in sorted(integrity.items())
    ) or "- No extra integrity metadata supplied."
    checksum_status = str(integrity.get("checksum_status", ""))
    checksum_warning = (
        "\n\n**CHECKSUM ALIGNMENT UNRESOLVED:** the current official object is a "
        "valid Parquet input, but its observed digest does not match the Dataverse "
        "release manifest. Counts below describe the explicitly recorded object "
        "digest and must not be presented as checksum-aligned release bytes."
        if "MISMATCH" in checksum_status or "DOES_NOT_MATCH" in checksum_status
        else ""
    )
    return f"""# OptimusKG Expression Polarity Audit

## Input integrity

{integrity_rows}{checksum_warning}

## Verified counts

| statistic | count |
| --- | ---: |
| EXPRESSION_PRESENT edges | {summary['present_edge_count']} |
| EXPRESSION_ABSENT edges | {summary['absent_edge_count']} |
| unique PRESENT anatomy-gene pairs | {summary['present_pair_count']} |
| unique ABSENT anatomy-gene pairs | {summary['absent_pair_count']} |
| PRESENT/ABSENT pair overlap | {summary['overlap_pair_count']} |
| overlap as % of PRESENT | {summary['overlap_percentage_of_present']:.8f}% |
| overlap as % of ABSENT | {summary['overlap_percentage_of_absent']:.8f}% |

## Pair-state classification

| state | pair count |
| --- | ---: |
| support-only (PRESENT only) | {summary['support_only_pair_count']} |
| refutation-only (ABSENT only) | {summary['refutation_only_pair_count']} |
| Both with distinct available evidence | {summary['genuine_both_pair_count']} |
| duplicate/not-independent conflict | {summary['duplicate_not_independent_conflict_pair_count']} |

Canonical pairs are exact stored `from` anatomy IDs and `to` gene IDs. Missing
relations are not ABSENT. Opposite relations would count as independently
supported Both only when the same canonical pair has distinguishable available
source/property fingerprints. Identical available fingerprints are conservatively
classified as duplicate or not independently supported conflict.

## Evidence fields

- Verified properties: `{', '.join(summary['verified_property_fields'])}`.
- Reference field: `{summary['reference_field_status']}`.
- Full source counts: `source_distribution.csv`.
- Expression-rank/call-quality counts: `evidence_property_distribution.csv`.
- Pair-level details: `overlapping_pairs.parquet` and `overlapping_pairs.csv`.

## Conclusion

**{summary['observed_both_conclusion']}**

The published table therefore supplies relation-specific PRESENT and ABSENT
context, but this release does not provide an observed edge-level Both example.
It does not justify treating missing expression knowledge as absence.
"""


def _dda_report(result: DDAAuditResult) -> str:
    summary = result.summary
    relation_lines = [
        "| relation | edges | unique pairs | drugs | diseases | phase missing | references | structure_id | drug_disease_id |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.relation_statistics.to_dicts():
        relation_lines.append(
            "| {relation} | {edge_count} | {unique_pair_count} | {drug_count} | "
            "{disease_count} | {clinical_phase_missing_percent:.4f}% | "
            "{reference_rows_non_empty} | {structure_id_non_null} | "
            "{drug_disease_id_non_null} |".format(**row)
        )
    policy_lines = [
        "| policy | support pairs | unsuitability pairs | drugs | diseases | final-table overlap | rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result.policy_comparison.to_dicts():
        policy_lines.append(
            "| {policy} | {support_pair_count} | {unsuitability_pair_count} | "
            "{support_drug_count} | {support_disease_count} | "
            "{observed_final_table_overlap_pair_count} | {support_rule} |".format(
                **row
            )
        )
    return f"""# OptimusKG Drug-Disease Label Policy Audit

## Release-data statistics

{chr(10).join(relation_lines)}

- Total rows: {summary['row_count']}.
- Unique final-table pairs: {summary['unique_pair_count']}.
- Duplicate final-table pairs: {summary['duplicate_pair_count']}.
- Multi-relation final-table pairs: {summary['multi_relation_pair_count']}.
- DDA evidence_score: `{summary['evidence_score_status']}`.

Detailed direct/indirect source, clinical-phase, reference/property, and degree
statistics are provided in the adjacent CSV/JSON files. Fields are extracted
per relation from the real schema; absent fields are reported as absent.

## Upstream aggregation and conflict recoverability

The final table contains one selected relation per canonical drug-disease pair.
It contains {summary['contraindication_with_open_targets_candidate_count']}
`CONTRAINDICATION` rows and {summary['off_label_with_open_targets_candidate_count']}
`OFF_LABEL_USE` rows carrying `OPEN_TARGETS` provenance. These are candidate
traces of collapsed source context; the source-specific original relation label
does not survive and current source code is not proven to be the release build.

**{summary['observed_dda_both_conclusion']}**

## Candidate policies (not applied)

{chr(10).join(policy_lines)}

All policies keep the {result.policy_comparison['unsuitability_pair_count'][0]}
final `CONTRAINDICATION` pairs in a separate contraindication/unsuitability
channel, not as proof of biological inefficacy. All keep `OFF_LABEL_USE` out of
support and refutation supervision. Policy B is the recommended later experiment
policy, subject to user approval and phase-threshold sensitivity analysis.

## Semantic decisions

- `INDICATION` is therapeutic-support context under the selected policy.
- `CONTRAINDICATION` is clinical contraindication/unsuitability evidence, not the
  logical complement of efficacy and not a generic false DDA.
- `OFF_LABEL_USE` remains graph context and a separate descriptive/evaluation
  category; it is not merged automatically into `INDICATION`.
- Missing DDA knowledge remains unlabeled/Unknown.
- Predicted query-level Both may be learned from simultaneous output strengths,
  but it must not be described as an observed ground-truth Both label here.

## Leakage warning

For every evaluated pair, remove all DDA target edges, reverse traversal,
semantically equivalent relations, and label-defining properties (including
clinical phase and source membership) from the query subgraph.
"""


def write_expression_audit(
    result: ExpressionAuditResult,
    output_dir: str | Path,
    *,
    input_integrity: dict[str, Any] | None = None,
) -> None:
    """Write the required expression audit artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = dict(result.summary)
    summary["input_integrity"] = input_integrity or {}
    _write_json(destination / "summary.json", summary)
    result.overlapping_pairs.write_parquet(destination / "overlapping_pairs.parquet")
    result.overlapping_pairs.write_csv(destination / "overlapping_pairs.csv")
    result.source_distribution.write_csv(destination / "source_distribution.csv")
    result.property_distribution.write_csv(
        destination / "evidence_property_distribution.csv"
    )
    (destination / "audit_report.md").write_text(
        _expression_report(result, input_integrity), encoding="utf-8"
    )


def write_dda_audit(
    result: DDAAuditResult,
    output_dir: str | Path,
) -> None:
    """Write the required DDA relation and label-policy audit artifacts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "summary.json", result.summary)
    result.relation_statistics.write_csv(destination / "relation_statistics.csv")
    result.source_distribution.write_csv(destination / "source_distribution.csv")
    result.phase_distribution.write_csv(
        destination / "clinical_phase_distribution.csv"
    )
    result.degree_distribution.write_csv(
        destination / "degree_distribution_summary.csv"
    )
    result.policy_comparison.write_csv(destination / "policy_comparison.csv")
    result.policy_source_distribution.write_csv(
        destination / "policy_source_distribution.csv"
    )
    result.policy_phase_distribution.write_csv(
        destination / "policy_phase_distribution.csv"
    )
    (destination / "audit_report.md").write_text(
        _dda_report(result), encoding="utf-8"
    )
