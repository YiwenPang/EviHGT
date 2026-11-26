"""Atomic formal Phase 3 build from the authorized aggregate parent binding."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evi_hgt.preprocessing.raw_source.phase3_cache import ComponentBinding, InputIntegrityError
from evi_hgt.preprocessing.raw_source.phase3_pipeline import (
    FormalRecoveryBuildContext,
    Phase3BuildResult,
    build_phase3_artifacts,
)
from evi_hgt.preprocessing.raw_source.phase3_recovery_build import (
    AggregateParentExpectation,
    bind_formal_aggregate_parent,
    build_endpoint_exclusion_artifact,
    derive_local_partitions,
    publish_formal_staging,
    validate_formal_completion_gates,
)
from evi_hgt.preprocessing.raw_source.paths import resolve_project_path


def _digests(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5()  # noqa: S324 - Parquet artifact fixity field.
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            size += len(block)
            md5.update(block)
            sha256.update(block)
    return size, md5.hexdigest(), sha256.hexdigest()


def _code_identifier(root: Path, configured_version: str) -> str:
    source_root = Path(__file__).resolve().parent
    paths = sorted([*source_root.glob("*.py"), source_root.parent / "canonical.py"],
                   key=lambda path: path.as_posix())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(source_root.parent).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"{configured_version}+source-sha256:{digest.hexdigest()}"


def _expectation(kind: str, config: dict[str, Any]) -> AggregateParentExpectation:
    return AggregateParentExpectation(
        kind=kind,
        release="2.0",
        size_bytes=int(config["size_bytes"]),
        manifest_md5=str(config["md5"]),
        sha256=str(config["sha256"]),
    )


def _derived_component_bindings(
    staging: Path,
    partition_manifest: list[dict[str, Any]],
) -> tuple[ComponentBinding, ...]:
    bindings = []
    for index, row in enumerate(
        sorted(partition_manifest, key=lambda value: value["relative_path"]),
        start=1,
    ):
        relative = str(row["relative_path"])
        path = staging / "partitions" / relative
        size, md5, sha256 = _digests(path)
        if sha256 != row["derived_sha256"] or size <= 0:
            raise InputIntegrityError(
                f"Derived partition fixity changed before construction: {relative}"
            )
        bindings.append(
            ComponentBinding(
                release_version="2.0",
                file_id=-index,
                relative_path=relative,
                cache_path=path,
                size_bytes=size,
                manifest_md5=md5,
                observed_md5=md5,
                observed_sha256=sha256,
                status="EVHGT_DERIVED_OPTIMUSKG_RELEASE_2_0_PARTITION",
                usable=True,
                manifest_size_bytes=size,
                content_type="application/vnd.apache.parquet",
                source_url=str(row["parent_aggregate_path"]),
            )
        )
    return tuple(bindings)


def run_authorized_phase3_recovery_build(
    *,
    config: dict[str, Any],
    project_root: str | Path,
    enforce_official_baselines: bool = True,
    progress: Any = None,
) -> Phase3BuildResult:
    """Build and atomically publish formal v1 from the two aggregate parents."""

    root = Path(project_root).resolve()
    phase3 = config["phase3"]
    recovery = phase3["formal_recovery"]
    if recovery.get("enabled") is not True:
        raise InputIntegrityError("Formal aggregate recovery is not enabled")
    if recovery.get("quarantined_stratified_inputs_allowed") is not False:
        raise InputIntegrityError("Quarantined stratified inputs must remain forbidden")
    final_root = resolve_project_path(root, phase3["paths"]["processed_root"])
    if final_root.exists():
        raise FileExistsError(
            f"Formal Phase 3 root already exists and will not be overwritten: {final_root}"
        )
    staging_parent = resolve_project_path(
        root, phase3["paths"]["staging_root"]
    )
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / (
        "formal_v1_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    )
    audit_root = resolve_project_path(root, recovery["formal_audit_root"])
    if audit_root.exists():
        raise FileExistsError(
            f"Formal audit root already exists and will not be overwritten: {audit_root}"
        )
    staging.mkdir(parents=True)

    nodes_path = resolve_project_path(root, recovery["nodes_parent"]["path"])
    edges_path = resolve_project_path(root, recovery["edges_parent"]["path"])
    nodes = bind_formal_aggregate_parent(
        nodes_path, _expectation("nodes", recovery["nodes_parent"])
    )
    edges = bind_formal_aggregate_parent(
        edges_path, _expectation("edges", recovery["edges_parent"])
    )
    code_identifier = _code_identifier(
        root, str(recovery["build_code_version"])
    )
    if progress:
        progress("按实体与关系类型派生 aggregate 分区")
    partition_manifest = derive_local_partitions(
        nodes,
        edges,
        output_root=staging / "partitions",
        transformation_version=str(recovery["transformation_version"]),
        build_config=phase3,
        code_identifier=code_identifier,
        require_complete_catalog=enforce_official_baselines,
    )
    derived_node_rows = sum(
        int(row["row_count"])
        for row in partition_manifest
        if row["kind"] == "nodes"
    )
    derived_edge_rows = sum(
        int(row["row_count"])
        for row in partition_manifest
        if row["kind"] == "edges"
    )
    if enforce_official_baselines and (derived_node_rows, derived_edge_rows) != (190_531, 21_813_816):
        raise InputIntegrityError(
            "Aggregate-derived partition totals do not reproduce the release"
        )

    exclusion_path = staging / "manifest/canonical_endpoint_exclusions.parquet"
    exclusion_summary = build_endpoint_exclusion_artifact(
        nodes.path,
        edges.path,
        output_path=exclusion_path,
        parent_edges_sha256=edges.observed_sha256,
        transformation_version=str(recovery["transformation_version"]),
        expected_exclusion_count=int(
            recovery["canonical_exclusion"]["expected_count"]
        ),
    )
    exclusion_summary["artifact_path"] = (
        "manifest/canonical_endpoint_exclusions.parquet"
    )
    bindings = _derived_component_bindings(staging, partition_manifest)
    parent_binding = {
        "release": "2.0",
        "binding_status": "AUTHORITATIVE_OPTIMUSKG_PARENT_BINDING",
        "nodes": nodes.to_dict(),
        "edges": edges.to_dict(),
        "derived_partitions_are_official_stratified_files": False,
        "quarantined_stratified_objects_used": False,
    }
    context = FormalRecoveryBuildContext(
        aggregate_parent=parent_binding,
        derived_partitions=tuple(partition_manifest),
        exclusion_artifact_path=exclusion_path,
        exclusion_summary=exclusion_summary,
        drugbank_interim_root=resolve_project_path(
            root, recovery["drugbank_interim_root"]
        ),
        transformation_version=str(recovery["transformation_version"]),
        code_identifier=code_identifier,
        published_root=final_root,
    )
    if progress:
        progress("规范化事实、合并 DrugBank 证据并构建 CSR 索引")
    drugbank_identity = recovery.get("drugbank_source_identity")
    if drugbank_identity is not None:
        (staging / "manifest/drugbank_source_identity.json").write_text(
            json.dumps(drugbank_identity, indent=2) + "\n", encoding="utf-8"
        )
    result = build_phase3_artifacts(
        bindings,
        config=config,
        processed_root=staging,
        audit_root=audit_root,
        enforce_official_baselines=enforce_official_baselines,
        formal_recovery_context=context,
    )
    completion_path = staging / "manifest/completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion_path.unlink()
    validate_formal_completion_gates(completion, enforce_official_baselines=enforce_official_baselines)
    # Confirm the files still identify the same raw inputs after every derived read.
    for parent in (nodes, edges):
        if _digests(parent.path)[2] != parent.observed_sha256:
            raise InputIntegrityError(f"Raw aggregate changed during build: {parent.path}")
    if drugbank_identity is not None:
        for filename, identity in drugbank_identity["files"].items():
            if _digests(Path(filename))[2] != identity["sha256"]:
                raise InputIntegrityError(f"DrugBank input changed during build: {filename}")
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    published = publish_formal_staging(
        staging, final_root, completion, enforce_official_baselines=enforce_official_baselines
    )
    return Phase3BuildResult(
        processed_root=published,
        audit_root=result.audit_root,
        node_count=result.node_count,
        canonical_fact_count=result.canonical_fact_count,
        evidence_record_count=result.evidence_record_count,
        traversal_edge_count=result.traversal_edge_count,
        supervision_statistics=result.supervision_statistics,
    )

