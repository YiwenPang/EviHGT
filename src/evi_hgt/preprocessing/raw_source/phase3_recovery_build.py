"""Authorized aggregate-parent recovery primitives for formal Phase 3.

The historical stratified quarantine is never consulted by this module.  Every
derived row comes from one of the two user-accepted release-2.0 aggregates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import pyarrow.parquet as pq

from evi_hgt.preprocessing.raw_source.phase3_2_recovery import EDGE_PARTITIONS, NODE_PARTITIONS


FORMAL_PARENT_STATUS = "AUTHORITATIVE_OPTIMUSKG_PARENT_BINDING"
FORMAL_USE_CLASS = "FORMAL_MODEL_INPUT"
ENDPOINT_EXCLUSION_REASON = "UPSTREAM_ENDPOINT_TYPE_INVARIANT_VIOLATION"
FULL_GRAPH_LEAKAGE_PASS = "FULL_GRAPH_LEAKAGE_AUDIT_PASS"
FULL_GRAPH_AUDIT_SCOPE = "REAL_FORMAL_PHASE3_GRAPH_ARTIFACTS"


class FormalRecoveryError(RuntimeError):
    """Raised when the accepted recovery policy cannot be reproduced exactly."""


@dataclass(frozen=True, slots=True)
class AggregateParentExpectation:
    kind: str
    release: str
    size_bytes: int
    manifest_md5: str
    sha256: str


@dataclass(frozen=True, slots=True)
class FormalAggregateParentBinding:
    kind: str
    path: Path
    release: str
    size_bytes: int
    manifest_md5: str
    observed_md5: str
    observed_sha256: str
    status: str = FORMAL_PARENT_STATUS
    use_class: str = FORMAL_USE_CLASS

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = str(self.path)
        return value


def _digests(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5()  # noqa: S324 - official release fixity field.
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return size, md5.hexdigest(), sha256.hexdigest()


def _sha256(path: Path) -> str:
    return _digests(path)[2]


def bind_formal_aggregate_parent(
    path: str | Path,
    expected: AggregateParentExpectation,
) -> FormalAggregateParentBinding:
    """Promote only an exact user-authorized aggregate payload."""

    source = Path(path).resolve()
    if expected.kind not in {"nodes", "edges"}:
        raise FormalRecoveryError(f"Unknown aggregate parent kind: {expected.kind}")
    if expected.release != "2.0" or not source.is_file():
        raise FormalRecoveryError("Formal Phase 3 requires an existing release-2.0 aggregate")
    size, observed_md5, observed_sha256 = _digests(source)
    matches = (
        size == expected.size_bytes
        and observed_md5.lower() == expected.manifest_md5.lower()
        and observed_sha256.lower() == expected.sha256.lower()
    )
    if not matches:
        raise FormalRecoveryError(
            f"{expected.kind} does not match the authorized aggregate binding"
        )
    try:
        parquet = pq.ParquetFile(source)
        if parquet.metadata.num_rows <= 0:
            raise ValueError("empty parquet")
    except Exception as exc:
        raise FormalRecoveryError(
            f"Authorized {expected.kind} aggregate is not readable Parquet"
        ) from exc
    return FormalAggregateParentBinding(
        kind=expected.kind,
        path=source,
        release=expected.release,
        size_bytes=size,
        manifest_md5=expected.manifest_md5.lower(),
        observed_md5=observed_md5.lower(),
        observed_sha256=observed_sha256.lower(),
    )


def _config_digest(build_config: dict[str, Any]) -> str:
    payload = json.dumps(
        build_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _partition_catalog(
    binding: FormalAggregateParentBinding,
) -> tuple[dict[str, str], str]:
    if binding.kind == "nodes":
        return NODE_PARTITIONS, "label"
    if binding.kind == "edges":
        return EDGE_PARTITIONS, "label"
    raise FormalRecoveryError(f"Unknown binding kind: {binding.kind}")


def _derive_parent_partitions(
    binding: FormalAggregateParentBinding,
    *,
    output_root: Path,
    transformation_version: str,
    config_sha256: str,
    code_identifier: str,
) -> list[dict[str, Any]]:
    catalog, family_column = _partition_catalog(binding)
    observed_families = sorted(
        str(value)
        for value in (
            pl.scan_parquet(binding.path)
            .select(family_column)
            .unique()
            .collect()[family_column]
            .to_list()
        )
    )
    unknown = set(observed_families) - set(catalog)
    if unknown:
        raise FormalRecoveryError(
            f"Unknown {binding.kind} aggregate families: {sorted(unknown)}"
        )
    rows: list[dict[str, Any]] = []
    for family in observed_families:
        relative_path = catalog[family]
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FormalRecoveryError(
                f"Derived partition already exists and will not be overwritten: {destination}"
            )
        lazy = pl.scan_parquet(binding.path).with_row_index(
            "aggregate_source_row_index"
        ).filter(pl.col(family_column) == family)
        lazy.sink_parquet(
            destination,
            compression="zstd",
            maintain_order=True,
        )
        parquet = pq.ParquetFile(destination)
        schema = pl.scan_parquet(destination).collect_schema()
        rows.append(
            {
                "kind": binding.kind,
                "family": family,
                "relative_path": relative_path,
                "parent_aggregate_path": str(binding.path),
                "parent_aggregate_sha256": binding.observed_sha256,
                "release": binding.release,
                "transformation_version": transformation_version,
                "row_count": int(parquet.metadata.num_rows),
                "schema": {name: str(dtype) for name, dtype in schema.items()},
                "derived_sha256": _sha256(destination),
                "build_config_sha256": config_sha256,
                "code_identifier": code_identifier,
                "artifact_identity": (
                    "EviHGT-derived OptimusKG release-2.0 partition"
                ),
                "official_stratified_release_file": False,
            }
        )
    return rows


def derive_local_partitions(
    node_parent: FormalAggregateParentBinding,
    edge_parent: FormalAggregateParentBinding,
    *,
    output_root: str | Path,
    transformation_version: str,
    build_config: dict[str, Any],
    code_identifier: str,
    require_complete_catalog: bool = True,
) -> list[dict[str, Any]]:
    """Deterministically derive raw-JSON partitions from the formal parents."""

    if node_parent.kind != "nodes" or edge_parent.kind != "edges":
        raise FormalRecoveryError("Formal parents must be supplied as nodes then edges")
    if node_parent.release != edge_parent.release or node_parent.release != "2.0":
        raise FormalRecoveryError("Aggregate parent releases do not agree on 2.0")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_sha256 = _config_digest(build_config)
    rows = [
        *_derive_parent_partitions(
            node_parent,
            output_root=root,
            transformation_version=transformation_version,
            config_sha256=config_sha256,
            code_identifier=code_identifier,
        ),
        *_derive_parent_partitions(
            edge_parent,
            output_root=root,
            transformation_version=transformation_version,
            config_sha256=config_sha256,
            code_identifier=code_identifier,
        ),
    ]
    if require_complete_catalog:
        node_count = sum(row["kind"] == "nodes" for row in rows)
        edge_count = sum(row["kind"] == "edges" for row in rows)
        if (node_count, edge_count) != (10, 27):
            raise FormalRecoveryError(
                f"Derived partition catalog is incomplete: nodes={node_count}, edges={edge_count}"
            )
    return sorted(rows, key=lambda row: str(row["relative_path"]))


def build_endpoint_exclusion_artifact(
    nodes_path: str | Path,
    edges_path: str | Path,
    *,
    output_path: str | Path,
    parent_edges_sha256: str,
    transformation_version: str,
    expected_exclusion_count: int | None = 142,
) -> dict[str, Any]:
    """Enumerate endpoint-family violations without changing any raw edge."""

    nodes = Path(nodes_path).resolve()
    edges = Path(edges_path).resolve()
    destination = Path(output_path).resolve()
    if _sha256(edges).lower() != parent_edges_sha256.lower():
        raise FormalRecoveryError("Exclusion audit parent edge digest mismatch")
    node_types = pl.scan_parquet(nodes).select("id", "label")
    source_types = node_types.select(
        pl.col("id").alias("from"),
        pl.col("label").alias("resolved_source_node_type"),
    )
    target_types = node_types.select(
        pl.col("id").alias("to"),
        pl.col("label").alias("resolved_target_node_type"),
    )
    resolved = (
        pl.scan_parquet(edges)
        .with_row_index("source_row_index")
        .with_columns(
            pl.col("label")
            .str.split_exact("-", 1)
            .struct.field("field_0")
            .alias("expected_source_node_type"),
            pl.col("label")
            .str.split_exact("-", 1)
            .struct.field("field_1")
            .alias("expected_target_node_type"),
        )
        .join(source_types, on="from", how="left")
        .join(target_types, on="to", how="left")
    )
    invalid = resolved.filter(
        pl.col("resolved_source_node_type").is_null()
        | pl.col("resolved_target_node_type").is_null()
        | (
            pl.col("resolved_source_node_type")
            != pl.col("expected_source_node_type")
        )
        | (
            pl.col("resolved_target_node_type")
            != pl.col("expected_target_node_type")
        )
    ).collect(engine="streaming")
    if expected_exclusion_count is not None and invalid.height != expected_exclusion_count:
        raise FormalRecoveryError(
            f"Expected {expected_exclusion_count} endpoint exclusions; observed {invalid.height}"
        )
    if invalid.height:
        accepted_pattern = invalid.filter(
            (pl.col("label") == "DIS-DIS")
            & (pl.col("resolved_source_node_type") == "PHE")
            & (pl.col("resolved_target_node_type") == "DIS")
            & (pl.col("expected_source_node_type") == "DIS")
            & (pl.col("expected_target_node_type") == "DIS")
        )
        if accepted_pattern.height != invalid.height:
            raise FormalRecoveryError(
                "Endpoint exclusion set contains an anomaly outside the authorized 142-row pattern"
            )
    artifact = invalid.select(
        "source_row_index",
        pl.col("from").alias("raw_source_id"),
        pl.col("to").alias("raw_target_id"),
        pl.col("label").alias("raw_edge_family"),
        pl.col("relation").alias("raw_relation"),
        "resolved_source_node_type",
        "resolved_target_node_type",
        "expected_source_node_type",
        "expected_target_node_type",
        pl.col("properties").alias("raw_properties"),
    ).with_columns(
        pl.lit(parent_edges_sha256.lower()).alias("parent_aggregate_sha256"),
        pl.lit(ENDPOINT_EXCLUSION_REASON).alias("exclusion_reason"),
        pl.lit(transformation_version).alias("transformation_version"),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FormalRecoveryError(
            f"Exclusion artifact already exists and will not be overwritten: {destination}"
        )
    artifact.write_parquet(destination, compression="zstd")
    raw_count = int(pq.ParquetFile(edges).metadata.num_rows)
    return {
        "parent_aggregate_sha256": parent_edges_sha256.lower(),
        "raw_aggregate_edge_count": raw_count,
        "excluded_canonical_invalid_rows": artifact.height,
        "canonical_input_count": raw_count - artifact.height,
        "exclusion_reason": ENDPOINT_EXCLUSION_REASON,
        "transformation_version": transformation_version,
        "artifact_path": str(destination),
        "artifact_sha256": _sha256(destination),
    }


def filter_canonical_input(
    frame: pl.DataFrame, exclusions: pl.DataFrame
) -> pl.DataFrame:
    """Remove only explicitly indexed rows from a canonicalization input."""

    if "source_row_index" not in exclusions.columns:
        raise FormalRecoveryError("Exclusion artifact lacks source_row_index")
    index_column = (
        "aggregate_source_row_index"
        if "aggregate_source_row_index" in frame.columns
        else "source_row_index"
    )
    work = (
        frame
        if index_column in frame.columns
        else frame.with_row_index("source_row_index")
    )
    result = work.join(
        exclusions.select(pl.col("source_row_index").alias(index_column)),
        on=index_column,
        how="anti",
    )
    if index_column == "source_row_index" and "source_row_index" not in frame.columns:
        result = result.drop("source_row_index")
    if result.height != frame.height - exclusions.height:
        raise FormalRecoveryError("Canonical exclusion indexes were not one-to-one")
    return result


_LEAKAGE_ZERO_FIELDS = (
    "target_indication_survivors",
    "target_contraindication_survivors",
    "target_off_label_use_survivors",
    "reverse_inverse_equivalent_survivors",
    "canonical_duplicate_survivors",
    "clinical_phase_feature_survivors",
    "source_provenance_feature_survivors",
    "cached_target_pair_feature_survivors",
    "train_validation_test_contamination",
    "cold_start_contamination",
)


def validate_full_graph_leakage_audit(audit: dict[str, Any]) -> None:
    """Reject unit/toy assertions and require all real-artifact counts to be zero."""

    if audit.get("audit_scope") != FULL_GRAPH_AUDIT_SCOPE:
        raise FormalRecoveryError(
            "Full-graph leakage PASS must be computed from real formal artifacts"
        )
    if audit.get("status") != FULL_GRAPH_LEAKAGE_PASS:
        raise FormalRecoveryError("Formal full-graph leakage status is not PASS")
    missing = [name for name in _LEAKAGE_ZERO_FIELDS if name not in audit]
    if missing:
        raise FormalRecoveryError(f"Formal leakage audit lacks fields: {missing}")
    nonzero = {name: audit[name] for name in _LEAKAGE_ZERO_FIELDS if audit[name] != 0}
    if nonzero:
        raise FormalRecoveryError(f"Formal graph leakage detected: {nonzero}")


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdefABCDEF" for character in text)


def validate_formal_completion_gates(
    manifest: dict[str, Any], *, enforce_official_baselines: bool = True
) -> None:
    """Validate the non-neural publication contract before atomic promotion."""

    if manifest.get("status") != "PHASE3_ARTIFACT_BUILD_COMPLETE":
        raise FormalRecoveryError("Formal completion status is absent")
    parent = manifest.get("aggregate_parent", {})
    if not _valid_sha256(parent.get("nodes_sha256")) or not _valid_sha256(
        parent.get("edges_sha256")
    ):
        raise FormalRecoveryError("Aggregate parent digests are incomplete")
    exclusion = manifest.get("canonical_exclusion", {})
    if enforce_official_baselines and exclusion.get("row_count") != 142:
        raise FormalRecoveryError("Formal exclusion count is not exactly 142")
    if not _valid_sha256(exclusion.get("artifact_sha256")):
        raise FormalRecoveryError("Formal exclusion artifact digest is incomplete")
    if exclusion.get("reason") != ENDPOINT_EXCLUSION_REASON:
        raise FormalRecoveryError("Formal exclusion reason is not authorized")
    if enforce_official_baselines and manifest.get("derived_partition_count") != 37:
        raise FormalRecoveryError("Formal derived partition catalog is incomplete")
    if manifest.get("formal_full_graph_leakage_audit") != FULL_GRAPH_LEAKAGE_PASS:
        raise FormalRecoveryError("Formal full-graph leakage gate did not pass")
    if manifest.get("all_completion_gates_passed") is not True:
        raise FormalRecoveryError("One or more formal Phase 3 gates failed")


def publish_formal_staging(
    staging_root: str | Path,
    final_root: str | Path,
    completion_manifest: dict[str, Any],
    *, enforce_official_baselines: bool = True,
) -> Path:
    """Atomically publish a staging tree only after every formal gate passes."""

    staging = Path(staging_root).resolve()
    final = Path(final_root).resolve()
    validate_formal_completion_gates(
        completion_manifest, enforce_official_baselines=enforce_official_baselines
    )
    if not staging.is_dir():
        raise FormalRecoveryError(f"Formal staging tree does not exist: {staging}")
    if final.exists():
        raise FormalRecoveryError(
            f"Formal artifact tree already exists and will not be overwritten: {final}"
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final)
    return final
