"""Leakage-aware, positive-only partial supervision for Drug-Disease pairs."""

from __future__ import annotations

from typing import Any

import polars as pl

from evi_hgt.preprocessing.raw_source.phase3_schema import (
    SupervisionSemanticsError,
    validate_supervision_frame,
)
from evi_hgt.preprocessing.raw_source.scientific_gate import (
    CONTRAINDICATION,
    INDICATION,
    OFF_LABEL_USE,
)


class DDASupervisionError(SupervisionSemanticsError):
    """Raised when real DDA rows violate the accepted supervision protocol."""


_POLICIES = {"A", "B", "C"}
_RELATIONS = {INDICATION, CONTRAINDICATION, OFF_LABEL_USE}


def _property_fields(frame: pl.DataFrame) -> set[str]:
    dtype = frame.schema.get("properties")
    if dtype == pl.String:
        return set()
    if not isinstance(dtype, pl.Struct):
        raise DDASupervisionError("DDA properties must be a Struct")
    return {field.name for field in dtype.fields}


def _property_expr(
    fields: set[str], name: str, dtype: pl.DataType, properties_dtype: pl.DataType
) -> pl.Expr:
    if properties_dtype == pl.String:
        value = pl.col("properties").str.json_path_match(f"$.{name}")
        if isinstance(dtype, pl.List):
            return value.str.json_decode(dtype)
        return value.cast(dtype, strict=False)
    if name in fields:
        return pl.col("properties").struct.field(name)
    return pl.lit(None, dtype=dtype)


def _source_role_expr(
    *, role: str, properties_dtype: pl.DataType
) -> pl.Expr:
    if properties_dtype == pl.String:
        return (
            pl.col("properties")
            .str.json_path_match(f"$.sources.{role}")
            .str.json_decode(pl.List(pl.String))
        )
    return (
        pl.col("properties")
        .struct.field("sources")
        .struct.field(role)
    )


def build_dda_supervision(
    frame: pl.DataFrame,
    *,
    policy: str = "B",
    clinical_phase_threshold: float = 2.0,
) -> pl.DataFrame:
    """Build one nullable supervision row per retained final DDA pair.

    `CONTRAINDICATION` is an incomplete retained positive observation for the
    clinical-unsuitability channel.  Its absence never creates target zero.
    """

    policy_name = policy.upper()
    if policy_name not in _POLICIES:
        raise DDASupervisionError(f"Unknown DDA policy: {policy}")
    required = {"from", "to", "relation", "undirected", "properties"}
    missing = required - set(frame.columns)
    if missing:
        raise DDASupervisionError(f"Missing DDA columns: {sorted(missing)}")
    if frame.select("from", "to").is_duplicated().any():
        raise DDASupervisionError(
            "Final DDA canonical pair must occur exactly once before supervision"
        )
    actual_relations = set(frame["relation"].drop_nulls().to_list())
    unexpected = actual_relations - _RELATIONS
    if unexpected:
        raise DDASupervisionError(
            f"Unexpected final DDA relations: {sorted(unexpected)}"
        )
    if frame.filter(~pl.col("undirected")).height:
        raise DDASupervisionError("Release 2.0 DDA facts must be undirected")

    properties_dtype = frame.schema["properties"]
    fields = _property_fields(frame)
    if properties_dtype != pl.String and "sources" not in fields:
        raise DDASupervisionError("DDA properties.sources is required")
    empty_strings = pl.lit([]).cast(pl.List(pl.String))
    work = frame.with_columns(
        _source_role_expr(role="direct", properties_dtype=properties_dtype)
        .fill_null(empty_strings)
        .alias("direct_sources"),
        _source_role_expr(role="indirect", properties_dtype=properties_dtype)
        .fill_null(empty_strings)
        .alias("indirect_sources"),
        _property_expr(
            fields, "reference_ids", pl.List(pl.String), properties_dtype
        )
        .fill_null(empty_strings)
        .alias("reference_ids"),
        _property_expr(
            fields,
            "highest_clinical_trial_phase",
            pl.Float64,
            properties_dtype,
        ).alias(
            "clinical_phase"
        ),
        _property_expr(
            fields, "structure_id", pl.String, properties_dtype
        ).alias("structure_id"),
        _property_expr(
            fields, "drug_disease_id", pl.String, properties_dtype
        ).alias(
            "drug_disease_id"
        ),
    )
    indication = pl.col("relation") == INDICATION
    contraindication = pl.col("relation") == CONTRAINDICATION
    off_label = pl.col("relation") == OFF_LABEL_USE
    policy_a = indication & pl.col("direct_sources").list.contains("DRUG_CENTRAL")
    policy_b = policy_a | (
        indication
        & (pl.col("clinical_phase").fill_null(float("-inf")) >= clinical_phase_threshold)
    )
    policy_c = indication
    active_expr = {"A": policy_a, "B": policy_b, "C": policy_c}[policy_name]

    artifact = (
        work.with_columns(
            pl.concat_str(["from", "to"], separator="\u001f").alias(
                "canonical_pair_key"
            ),
            policy_a.alias("policy_a_member"),
            policy_b.alias("policy_b_member"),
            policy_c.alias("policy_c_member"),
            active_expr.alias("support_mask"),
            contraindication.alias("unsuitability_mask"),
            off_label.alias("context_only"),
        )
        .with_columns(
            pl.when(pl.col("support_mask"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("support_target"),
            pl.when(pl.col("unsuitability_mask"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("unsuitability_target"),
            contraindication.alias("unsuitability_observation_incomplete"),
            pl.lit(True).alias("retained_after_upstream_aggregation"),
            pl.lit(policy_name).alias("active_policy"),
            pl.lit(False).alias("supervision_missingness_is_hard_unknown"),
        )
        .select(
            pl.col("from").alias("drug_id"),
            pl.col("to").alias("disease_id"),
            "canonical_pair_key",
            pl.col("relation").alias("raw_relation"),
            "active_policy",
            "policy_a_member",
            "policy_b_member",
            "policy_c_member",
            "support_target",
            "support_mask",
            "unsuitability_target",
            "unsuitability_mask",
            "context_only",
            "unsuitability_observation_incomplete",
            "retained_after_upstream_aggregation",
            "supervision_missingness_is_hard_unknown",
            "direct_sources",
            "indirect_sources",
            "reference_ids",
            "clinical_phase",
            "structure_id",
            "drug_disease_id",
        )
        .sort(["drug_id", "disease_id"])
    )
    try:
        validate_supervision_frame(artifact)
    except SupervisionSemanticsError as exc:
        raise DDASupervisionError(str(exc)) from exc
    return artifact


def _scope_counts(frame: pl.DataFrame) -> dict[str, int]:
    return {
        "pairs": frame.height,
        "drugs": frame["drug_id"].n_unique(),
        "diseases": frame["disease_id"].n_unique(),
    }


def supervision_statistics(artifact: pl.DataFrame) -> dict[str, Any]:
    """Summarize policy membership without interpreting unlabeled as zero."""

    validate_supervision_frame(artifact)
    return {
        "row_count": artifact.height,
        "policy_a": _scope_counts(
            artifact.filter(pl.col("policy_a_member"))
        ),
        "policy_b": _scope_counts(
            artifact.filter(pl.col("policy_b_member"))
        ),
        "policy_c": _scope_counts(
            artifact.filter(pl.col("policy_c_member"))
        ),
        "retained_unsuitability": _scope_counts(
            artifact.filter(pl.col("unsuitability_mask"))
        ),
        "off_label_context_pairs": artifact.filter(pl.col("context_only")).height,
        "target_zero_count": int(
            artifact.select(
                (pl.col("support_target") == 0).sum()
                + (pl.col("unsuitability_target") == 0).sum()
            ).item()
        ),
        "hard_four_value_target_present": bool(
            {"four_value_label", "hard_unknown"} & set(artifact.columns)
        ),
        "retained_contraindication_is_complete": False,
    }
