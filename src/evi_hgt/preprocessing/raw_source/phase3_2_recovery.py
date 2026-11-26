"""Fail-closed primitives for the Phase 3.2 aggregate recovery audit.

This module does not alter a Phase 3 component binding, quarantine state, or
formal artifact.  A positive recovery decision is only a candidate requiring a
later explicit user decision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import Any

import polars as pl


class DataUseClass(StrEnum):
    """Exactly one permitted purpose for every Phase 3.2 acquisition."""

    RECOVERY_ONLY = "RECOVERY_ONLY"
    VALIDATION_ONLY = "VALIDATION_ONLY"
    FORMAL_MODEL_INPUT = "FORMAL_MODEL_INPUT"


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    """Version-aware download record with an authoritative use classification."""

    name: str
    use_class: DataUseClass
    dataset_source: str
    version: str
    provider: str
    license_or_terms_status: str
    source_identity: str
    expected_size: int | None
    observed_size: int
    manifest_md5: str | None
    observed_md5: str
    observed_sha256: str
    destination: str
    purpose: str
    specific_question: str
    used_in_recovery_decision: bool
    formal_input_authorization_basis: str | None = None
    acquired_at_utc: str | None = None

    def __post_init__(self) -> None:
        try:
            selected = DataUseClass(self.use_class)
        except (TypeError, ValueError) as exc:
            raise ValueError("Every acquisition requires exactly one valid use class") from exc
        object.__setattr__(self, "use_class", selected)
        if self.observed_size < 0:
            raise ValueError("Observed acquisition size cannot be negative")
        if (
            selected is DataUseClass.FORMAL_MODEL_INPUT
            and not self.formal_input_authorization_basis
        ):
            raise ValueError(
                "FORMAL_MODEL_INPUT requires an explicit authorization basis"
            )

    @property
    def authorized_as_formal_model_input(self) -> bool:
        return self.use_class is DataUseClass.FORMAL_MODEL_INPUT

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["use_class"] = self.use_class.value
        value["authorized_as_formal_model_input"] = (
            self.authorized_as_formal_model_input
        )
        return value


class AggregateBindingStatus(StrEnum):
    """Strict byte/format outcomes for an aggregate release object."""

    MANIFEST_MATCH_VERIFIED = "MANIFEST_MATCH_VERIFIED"
    RELEASE_1_CARRYOVER = "RELEASE_1_CARRYOVER"
    MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
    PARQUET_INVALID = "PARQUET_INVALID"


@dataclass(frozen=True, slots=True)
class AggregateBindingDecision:
    name: str
    status: AggregateBindingStatus
    usable_for_recovery: bool
    manifest_size: int
    observed_size: int
    manifest_md5: str
    observed_md5: str
    observed_sha256: str
    release_1_md5: str | None
    observed_matches_release_1: bool
    observed_matches_release_2: bool
    parquet_readable: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


def validate_aggregate_binding(
    *,
    name: str,
    manifest_size: int,
    manifest_md5: str,
    release_1_md5: str | None,
    observed_size: int,
    observed_md5: str,
    observed_sha256: str,
    parquet_readable: bool,
) -> AggregateBindingDecision:
    """Require release-2.0 size/MD5 and explicitly reject v1 carryover bytes."""

    observed = observed_md5.lower()
    expected = manifest_md5.lower()
    v1 = release_1_md5.lower() if release_1_md5 else None
    matches_v1 = bool(v1 and observed == v1)
    matches_v2 = observed_size == manifest_size and observed == expected
    if not parquet_readable:
        status = AggregateBindingStatus.PARQUET_INVALID
    elif matches_v1:
        status = AggregateBindingStatus.RELEASE_1_CARRYOVER
    elif matches_v2:
        status = AggregateBindingStatus.MANIFEST_MATCH_VERIFIED
    else:
        status = AggregateBindingStatus.MANIFEST_MISMATCH
    return AggregateBindingDecision(
        name=name,
        status=status,
        usable_for_recovery=status is AggregateBindingStatus.MANIFEST_MATCH_VERIFIED,
        manifest_size=manifest_size,
        observed_size=observed_size,
        manifest_md5=expected,
        observed_md5=observed,
        observed_sha256=observed_sha256.lower(),
        release_1_md5=v1,
        observed_matches_release_1=matches_v1,
        observed_matches_release_2=matches_v2,
        parquet_readable=parquet_readable,
    )


@dataclass(frozen=True, slots=True)
class LosslessProperty:
    """Raw and parsed views of one aggregate JSON property object."""

    raw: str
    parsed: dict[str, Any]
    canonical_json: str


def _reject_nonstandard_json(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is not accepted: {value}")


def parse_property_losslessly(raw: str) -> LosslessProperty:
    """Parse JSON without dropping unknown keys or replacing the raw value."""

    try:
        parsed = json.loads(raw, parse_constant=_reject_nonstandard_json)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Invalid aggregate properties JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Aggregate properties must decode to a JSON object")
    canonical = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return LosslessProperty(raw=raw, parsed=parsed, canonical_json=canonical)


NODE_PARTITIONS = {
    "ANA": "nodes/anatomy.parquet",
    "BPO": "nodes/biological_process.parquet",
    "CCO": "nodes/cellular_component.parquet",
    "DIS": "nodes/disease.parquet",
    "DRG": "nodes/drug.parquet",
    "EXP": "nodes/exposure.parquet",
    "GEN": "nodes/gene.parquet",
    "MFN": "nodes/molecular_function.parquet",
    "PWY": "nodes/pathway.parquet",
    "PHE": "nodes/phenotype.parquet",
}

EDGE_PARTITIONS = {
    "ANA-ANA": "edges/anatomy_anatomy.parquet",
    "ANA-GEN": "edges/anatomy_gene.parquet",
    "BPO-BPO": "edges/biological_process_biological_process.parquet",
    "BPO-GEN": "edges/biological_process_gene.parquet",
    "CCO-CCO": "edges/cellular_component_cellular_component.parquet",
    "CCO-GEN": "edges/cellular_component_gene.parquet",
    "DIS-DIS": "edges/disease_disease.parquet",
    "DIS-GEN": "edges/disease_gene.parquet",
    "DIS-PHE": "edges/disease_phenotype.parquet",
    "DRG-BPO": "edges/drug_biological_process.parquet",
    "DRG-DIS": "edges/drug_disease.parquet",
    "DRG-DRG": "edges/drug_drug.parquet",
    "DRG-GEN": "edges/drug_gene.parquet",
    "DRG-PHE": "edges/drug_phenotype.parquet",
    "EXP-BPO": "edges/exposure_biological_process.parquet",
    "EXP-CCO": "edges/exposure_cellular_component.parquet",
    "EXP-DIS": "edges/exposure_disease.parquet",
    "EXP-EXP": "edges/exposure_exposure.parquet",
    "EXP-GEN": "edges/exposure_gene.parquet",
    "EXP-MFN": "edges/exposure_molecular_function.parquet",
    "GEN-GEN": "edges/gene_gene.parquet",
    "MFN-GEN": "edges/molecular_function_gene.parquet",
    "MFN-MFN": "edges/molecular_function_molecular_function.parquet",
    "PWY-GEN": "edges/pathway_gene.parquet",
    "PWY-PWY": "edges/pathway_pathway.parquet",
    "PHE-GEN": "edges/phenotype_gene.parquet",
    "PHE-PHE": "edges/phenotype_phenotype.parquet",
}


def node_partition_path(label: str) -> str:
    try:
        return NODE_PARTITIONS[label]
    except KeyError as exc:
        raise ValueError(f"Unknown aggregate node family: {label}") from exc


def edge_partition_path(label: str) -> str:
    try:
        return EDGE_PARTITIONS[label]
    except KeyError as exc:
        raise ValueError(f"Unknown aggregate edge family: {label}") from exc


_AGGREGATE_SCHEMAS = {
    "nodes": {
        "id": pl.String,
        "label": pl.String,
        "properties": pl.String,
    },
    "edges": {
        "from": pl.String,
        "to": pl.String,
        "label": pl.String,
        "relation": pl.String,
        "undirected": pl.Boolean,
        "properties": pl.String,
    },
}
_AGGREGATE_TOTALS = {"nodes": 190_531, "edges": 21_813_816}


def validate_aggregate_schema(
    kind: str, schema: pl.Schema, row_count: int
) -> dict[str, Any]:
    """Validate the exact unified base schema and accepted release total."""

    if kind not in _AGGREGATE_SCHEMAS:
        raise ValueError(f"Unknown aggregate kind: {kind}")
    expected = _AGGREGATE_SCHEMAS[kind]
    missing = sorted(set(expected) - set(schema.names()))
    unexpected = sorted(set(schema.names()) - set(expected))
    type_mismatches = {
        name: {"expected": str(dtype), "observed": str(schema[name])}
        for name, dtype in expected.items()
        if name in schema and schema[name] != dtype
    }
    row_count_match = row_count == _AGGREGATE_TOTALS[kind]
    return {
        "kind": kind,
        "row_count": row_count,
        "expected_row_count": _AGGREGATE_TOTALS[kind],
        "row_count_match": row_count_match,
        "observed_schema": {name: str(dtype) for name, dtype in schema.items()},
        "expected_schema": {name: str(dtype) for name, dtype in expected.items()},
        "missing_columns": missing,
        "unexpected_columns": unexpected,
        "type_mismatches": type_mismatches,
        "valid": not missing
        and not unexpected
        and not type_mismatches
        and row_count_match,
    }


def compare_partition_counts(
    observed: pl.DataFrame,
    *,
    expected: dict[str, int],
    kind: str,
) -> pl.DataFrame:
    """Compare deterministic aggregate family counts to a pinned baseline."""

    required = {"label", "aggregate_derived_count"}
    if missing := required - set(observed.columns):
        raise ValueError(f"Observed partition counts lack {sorted(missing)}")
    if observed["label"].n_unique() != observed.height:
        raise ValueError("Observed partition labels must be unique")
    path_function = node_partition_path if kind == "node" else edge_partition_path
    unknown = set(observed["label"].to_list()) - set(expected)
    if unknown:
        raise ValueError(f"Unexpected observed {kind} labels: {sorted(unknown)}")
    observed_lookup = {
        str(row["label"]): int(row["aggregate_derived_count"])
        for row in observed.iter_rows(named=True)
    }
    rows = []
    for label, expected_count in sorted(expected.items()):
        actual = observed_lookup.get(label)
        rows.append(
            {
                "partition": path_function(label),
                "kind": kind,
                "label": label,
                "aggregate_derived_count": actual,
                "phase2_expected_count": int(expected_count),
                "count_difference": (
                    int(actual) - int(expected_count) if actual is not None else None
                ),
                "match": actual == int(expected_count),
            }
        )
    return pl.DataFrame(rows)


def normalized_semantic_hashes(frame: pl.DataFrame) -> pl.Series:
    """Return deterministic row hashes after canonical JSON normalization.

    This helper is intended for bounded tests and partition cross-checks.  Full
    aggregate audits may use the same normalization in streaming partitions.
    """

    if "properties" not in frame.columns:
        raise ValueError("Semantic rows require a properties column")
    dtype = frame.schema["properties"]
    if dtype == pl.String:
        canonical = frame["properties"].map_elements(
            lambda raw: parse_property_losslessly(str(raw)).canonical_json,
            return_dtype=pl.String,
        )
    elif isinstance(dtype, pl.Struct):
        canonical = frame["properties"].struct.json_encode().map_elements(
            lambda raw: parse_property_losslessly(str(raw)).canonical_json,
            return_dtype=pl.String,
        )
    else:
        raise ValueError(f"Unsupported properties dtype: {dtype}")
    normalized = frame.with_columns(canonical.alias("properties"))
    return normalized.hash_rows(seed=42, seed_1=43, seed_2=44, seed_3=45).sort()


def semantic_frames_equivalent(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    """Compare normalized row multisets without relying on input row order."""

    if left.columns != right.columns or left.height != right.height:
        return False
    return normalized_semantic_hashes(left).equals(normalized_semantic_hashes(right))


def audit_endpoint_integrity(
    nodes: pl.DataFrame, edges: pl.DataFrame
) -> dict[str, Any]:
    """Resolve aggregate endpoints and enumerate missing/wrong family types."""

    node_required = {"id", "label"}
    edge_required = {"from", "to", "label", "relation"}
    if missing := node_required - set(nodes.columns):
        raise ValueError(f"Node endpoint table lacks {sorted(missing)}")
    if missing := edge_required - set(edges.columns):
        raise ValueError(f"Edge endpoint table lacks {sorted(missing)}")
    if nodes["id"].n_unique() != nodes.height:
        raise ValueError("Aggregate node identifiers must be unique")
    unknown_families = set(edges["label"].drop_nulls().unique().to_list()) - set(
        EDGE_PARTITIONS
    )
    if unknown_families:
        raise ValueError(f"Unknown aggregate edge families: {sorted(unknown_families)}")
    expected = edges.with_columns(
        pl.col("label").str.split_exact("-", 1).struct.field("field_0").alias(
            "expected_source_label"
        ),
        pl.col("label").str.split_exact("-", 1).struct.field("field_1").alias(
            "expected_target_label"
        ),
    )
    source_nodes = nodes.select(
        pl.col("id").alias("from"), pl.col("label").alias("source_node_label")
    )
    target_nodes = nodes.select(
        pl.col("id").alias("to"), pl.col("label").alias("target_node_label")
    )
    resolved = expected.join(source_nodes, on="from", how="left", validate="m:1").join(
        target_nodes, on="to", how="left", validate="m:1"
    )
    missing_source = pl.col("source_node_label").is_null()
    missing_target = pl.col("target_node_label").is_null()
    wrong_source = (~missing_source) & (
        pl.col("source_node_label") != pl.col("expected_source_label")
    )
    wrong_target = (~missing_target) & (
        pl.col("target_node_label") != pl.col("expected_target_label")
    )
    metrics = resolved.select(
        missing_source.sum().alias("missing_source_count"),
        missing_target.sum().alias("missing_target_count"),
        wrong_source.sum().alias("wrong_source_type_count"),
        wrong_target.sum().alias("wrong_target_type_count"),
        (
            (pl.col("label") == "DIS-DIS")
            & (pl.col("source_node_label") == "PHE")
        )
        .sum()
        .alias("disease_disease_phe_source_count"),
    ).row(0, named=True)
    return {key: int(value) for key, value in metrics.items()}


class RecoveryClassification(StrEnum):
    AGGREGATE_RECOVERY_NOT_POSSIBLE = "AGGREGATE_RECOVERY_NOT_POSSIBLE"
    AGGREGATE_RECOVERY_CANDIDATE = "AGGREGATE_RECOVERY_CANDIDATE"
    AGGREGATE_RECOVERY_CANDIDATE_WITH_EXCLUSIONS = (
        "AGGREGATE_RECOVERY_CANDIDATE_WITH_EXCLUSIONS"
    )
    MANIFEST_STRATIFIED_RECOVERY_FOUND = "MANIFEST_STRATIFIED_RECOVERY_FOUND"
    UPSTREAM_RECONSTRUCTION_CANDIDATE = "UPSTREAM_RECONSTRUCTION_CANDIDATE"


@dataclass(frozen=True, slots=True)
class RecoveryGates:
    """The sixteen mandatory clean aggregate recovery conditions."""

    nodes_manifest_match: bool
    edges_manifest_match: bool
    not_release_1_carryover: bool
    parquet_structurally_valid: bool
    total_counts_match: bool
    expected_families_present: bool
    partition_counts_match: bool
    dda_relation_counts_match: bool
    policy_counts_match: bool
    contraindication_count_match: bool
    off_label_count_match: bool
    expression_counts_match: bool
    provenance_recoverable: bool
    valid_stratified_semantically_equivalent: bool
    endpoint_anomalies_enumerated: bool
    no_unexplained_semantic_discrepancy: bool

    def failed_conditions(self) -> tuple[str, ...]:
        return tuple(
            item.name for item in fields(self) if not bool(getattr(self, item.name))
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    classification: RecoveryClassification
    gates: RecoveryGates
    failed_gates: tuple[str, ...]
    disease_disease_exclusion_required: bool
    user_decision_required: bool
    phase3_resumed: bool = False
    phase4_started: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.value,
            "gates": self.gates.to_dict(),
            "failed_gates": list(self.failed_gates),
            "disease_disease_exclusion_required": (
                self.disease_disease_exclusion_required
            ),
            "user_decision_required": self.user_decision_required,
            "phase3_resumed": self.phase3_resumed,
            "phase4_started": self.phase4_started,
        }


def decide_recovery(
    gates: RecoveryGates,
    *,
    disease_disease_exclusion_required: bool = False,
    manifest_stratified_recovery_found: bool = False,
    upstream_reconstruction_feasible: bool = False,
) -> RecoveryDecision:
    """Select exactly one recovery class without resuming Phase 3."""

    failed = gates.failed_conditions()
    if manifest_stratified_recovery_found:
        classification = RecoveryClassification.MANIFEST_STRATIFIED_RECOVERY_FOUND
    elif not failed:
        classification = (
            RecoveryClassification.AGGREGATE_RECOVERY_CANDIDATE_WITH_EXCLUSIONS
            if disease_disease_exclusion_required
            else RecoveryClassification.AGGREGATE_RECOVERY_CANDIDATE
        )
    elif upstream_reconstruction_feasible:
        classification = RecoveryClassification.UPSTREAM_RECONSTRUCTION_CANDIDATE
    else:
        classification = RecoveryClassification.AGGREGATE_RECOVERY_NOT_POSSIBLE
    return RecoveryDecision(
        classification=classification,
        gates=gates,
        failed_gates=failed,
        disease_disease_exclusion_required=disease_disease_exclusion_required,
        user_decision_required=classification
        is not RecoveryClassification.AGGREGATE_RECOVERY_NOT_POSSIBLE,
    )
