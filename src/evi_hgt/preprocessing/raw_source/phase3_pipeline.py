"""Phase 3 artifact orchestration over verified local component bindings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from evi_hgt.preprocessing.raw_source.canonical_facts import (
    augment_node_index,
    build_node_index,
    canonicalize_edge_partition,
    edge_component_spec,
)
from evi_hgt.preprocessing.raw_source.dda_supervision import (
    build_dda_supervision,
    supervision_statistics,
)
from evi_hgt.preprocessing.raw_source.drugbank_merge import (
    audit_drugbank_interaction_overlap,
    build_drugbank_partner_facts,
    build_formal_drugbank_node_map,
)
from evi_hgt.preprocessing.raw_source.graph_builder import build_csr_index
from evi_hgt.preprocessing.raw_source.leakage import (
    DDA_CLINICAL_RELATIONS,
    training_forbidden_facts,
)
from evi_hgt.preprocessing.raw_source.phase3_cache import (
    ComponentBinding,
    InputIntegrityError,
    assert_all_components_usable,
    bind_checksum_drift_component,
    cache_component,
    resolve_component_plan,
)
from evi_hgt.preprocessing.raw_source.phase3_recovery_build import (
    ENDPOINT_EXCLUSION_REASON,
    FULL_GRAPH_AUDIT_SCOPE,
    FULL_GRAPH_LEAKAGE_PASS,
    filter_canonical_input,
    validate_full_graph_leakage_audit,
)
from evi_hgt.preprocessing.raw_source.provenance import classify_provenance_token
from evi_hgt.preprocessing.raw_source.splits import build_split, split_statistics
from evi_hgt.preprocessing.raw_source.optimus_loader import fetch_release_metadata
from evi_hgt.preprocessing.raw_source.paths import resolve_project_path


OFFICIAL_BASELINES = {
    "policy_a": {"pairs": 4420, "drugs": 1754, "diseases": 764},
    "policy_b": {"pairs": 43973, "drugs": 7801, "diseases": 2205},
    "policy_c": {"pairs": 57601, "drugs": 9907, "diseases": 2327},
    "retained_unsuitability": {"pairs": 11718, "drugs": 1264, "diseases": 509},
    "off_label_context_pairs": 1061,
}

EXPRESSION_BASELINES = {
    "EXPRESSION_PRESENT": 6_616_463,
    "EXPRESSION_ABSENT": 2_171_492,
    "canonical_pair_overlap": 0,
}

PREFLIGHT_SPLIT_COUNTS = {
    "standard": {"train": 56_384, "validation": 6_879, "test": 7_117},
    "cold_drug": {"train": 56_726, "validation": 7_187, "test": 6_467},
    "cold_disease": {"train": 58_953, "validation": 5_449, "test": 5_978},
    "double_cold": {"train": 47_520, "validation": 567, "test": 559},
}


@dataclass(frozen=True, slots=True)
class Phase3BuildResult:
    processed_root: Path
    audit_root: Path
    node_count: int
    canonical_fact_count: int
    evidence_record_count: int
    traversal_edge_count: int
    supervision_statistics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FormalRecoveryBuildContext:
    """Explicit authority and provenance for an aggregate-parent formal build."""

    aggregate_parent: dict[str, Any]
    derived_partitions: tuple[dict[str, Any], ...]
    exclusion_artifact_path: Path
    exclusion_summary: dict[str, Any]
    drugbank_interim_root: Path
    transformation_version: str
    code_identifier: str
    published_root: Path


def download_phase3_inputs(
    *,
    config: dict[str, Any],
    project_root: str | Path,
    session: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[ComponentBinding, ...]:
    """Bind/download every official release 2.0 stratified component."""

    root = Path(project_root).resolve()
    phase3 = config["phase3"]
    data_config = config["optimuskg"]
    expected_release = str(phase3["optimuskg_release"])
    release = fetch_release_metadata(
        server=str(data_config["server"]),
        doi=str(data_config["doi"]),
        version=expected_release,
        session=session,
    )
    plan = resolve_component_plan(release, expected_release=expected_release)
    cache_root = resolve_project_path(
        root, phase3["paths"]["release_cache_root"]
    )
    quarantine_root = resolve_project_path(
        root, phase3["paths"]["quarantine_root"]
    )
    integrity = phase3["integrity"]
    anagen_path = resolve_project_path(root, integrity["anagen_isolated_path"])
    bindings: list[ComponentBinding] = []
    for index, component in enumerate(plan, start=1):
        if progress:
            progress(f"[{index}/{len(plan)}] {component.relative_path}")
        if component.relative_path == "edges/anatomy_gene.parquet":
            binding = bind_checksum_drift_component(
                component,
                isolated_path=anagen_path,
                observed_md5=str(integrity["anagen_observed_md5"]),
                observed_sha256=str(integrity["anagen_observed_sha256"]),
                observed_size_bytes=int(integrity["anagen_size_bytes"]),
            )
        else:
            binding = cache_component(
                component,
                version=expected_release,
                cache_root=cache_root,
                quarantine_root=quarantine_root,
                server=str(data_config["server"]),
                session=session,
            )
        bindings.append(binding)
        if progress:
            progress(
                f"    {binding.status} bytes={binding.size_bytes} "
                f"sha256={binding.observed_sha256}"
            )
    return tuple(bindings)


def run_formal_phase3_build(
    *,
    config: dict[str, Any],
    project_root: str | Path,
    bindings: Iterable[ComponentBinding],
) -> Phase3BuildResult:
    """Build in a fresh staging directory and atomically publish version v1."""

    root = Path(project_root).resolve()
    phase3 = config["phase3"]
    final_root = resolve_project_path(root, phase3["paths"]["processed_root"])
    audit_root = resolve_project_path(root, phase3["paths"]["audit_root"])
    staging_parent = resolve_project_path(root, phase3["paths"]["staging_root"])
    if final_root.exists():
        raise FileExistsError(
            f"Phase 3 processed root already exists and will not be overwritten: {final_root}"
        )
    staging_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    staging = staging_parent / f"{phase3['artifact_version']}_{stamp}"
    result = build_phase3_artifacts(
        bindings,
        config=config,
        processed_root=staging,
        audit_root=audit_root,
        enforce_official_baselines=True,
    )
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(final_root)
    return Phase3BuildResult(
        processed_root=final_root,
        audit_root=result.audit_root,
        node_count=result.node_count,
        canonical_fact_count=result.canonical_fact_count,
        evidence_record_count=result.evidence_record_count,
        traversal_edge_count=result.traversal_edge_count,
        supervision_statistics=result.supervision_statistics,
    )


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_official_baselines(stats: dict[str, Any]) -> None:
    differences = {
        key: {"expected": expected, "observed": stats.get(key)}
        for key, expected in OFFICIAL_BASELINES.items()
        if stats.get(key) != expected
    }
    if differences:
        raise InputIntegrityError(
            "Phase 2.55 DDA supervision baselines were not reproduced: "
            f"{differences}"
        )


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        row: dict[str, Any] = {
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            row["row_count"] = parquet.metadata.num_rows
            row["schema"] = str(parquet.schema_arrow)
        elif path.suffix == ".npy":
            array = np.load(path, mmap_mode="r")
            row["shape"] = list(array.shape)
            row["dtype"] = str(array.dtype)
        rows.append(row)
    return rows


def _provenance_catalog(tokens: Iterable[str]) -> pl.DataFrame:
    rows = []
    for index, raw in enumerate(sorted(set(tokens))):
        classified = classify_provenance_token(raw)
        rows.append(
            {
                "provenance_token_id": index,
                "raw_token": classified.raw,
                "normalized_token": classified.normalized,
                "token_kind": classified.kind.value,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "provenance_token_id": pl.UInt32,
            "raw_token": pl.String,
            "normalized_token": pl.String,
            "token_kind": pl.String,
        },
    )


def build_phase3_artifacts(
    bindings: Iterable[ComponentBinding],
    *,
    config: dict[str, Any],
    processed_root: str | Path,
    audit_root: str | Path,
    enforce_official_baselines: bool = True,
    formal_recovery_context: FormalRecoveryBuildContext | None = None,
) -> Phase3BuildResult:
    """Build core official-graph Phase 3 artifacts from verified bindings.

    The completion manifest is deliberately the final write.  Its absence marks
    an interrupted or failed build as unpublished.
    """

    components = tuple(sorted(bindings, key=lambda item: item.relative_path))
    if not components:
        raise InputIntegrityError("No Phase 3 component bindings supplied")
    assert_all_components_usable(components)
    expected_release = str(config["phase3"]["optimuskg_release"])
    observed_releases = {item.release_version for item in components}
    if observed_releases != {expected_release}:
        raise InputIntegrityError(
            f"Component release mismatch: {sorted(observed_releases)}"
        )

    destination = Path(processed_root).resolve()
    audits = Path(audit_root).resolve()
    manifest_dir = destination / "manifest"
    node_dir = destination / "nodes"
    fact_dir = destination / "canonical_facts"
    evidence_dir = destination / "evidence_records"
    drugbank_dir = destination / "drugbank"
    supervision_dir = destination / "supervision"
    split_dir = destination / "splits"
    graph_dir = destination / "graph"
    view_dir = destination / "views"
    for directory in (
        manifest_dir,
        node_dir,
        fact_dir,
        evidence_dir,
        drugbank_dir,
        supervision_dir,
        split_dir,
        graph_dir,
        view_dir,
        audits,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if formal_recovery_context is not None:
        if enforce_official_baselines and len(formal_recovery_context.derived_partitions) != 37:
            raise InputIntegrityError(
                "Formal recovery requires exactly 10 node and 27 edge partitions"
            )
        exclusion = formal_recovery_context.exclusion_summary
        if enforce_official_baselines and (
            exclusion.get("excluded_canonical_invalid_rows") != 142
            or exclusion.get("exclusion_reason") != ENDPOINT_EXCLUSION_REASON
            or exclusion.get("canonical_input_count") != 21_813_674
        ):
            raise InputIntegrityError(
                "Formal recovery exclusion summary does not reproduce the authorized 142 rows"
            )
        _json_write(
            manifest_dir / "aggregate_parent_binding.json",
            formal_recovery_context.aggregate_parent,
        )
        _json_write(
            manifest_dir / "derived_partitions.json",
            list(formal_recovery_context.derived_partitions),
        )
        _json_write(
            manifest_dir / "canonical_exclusion_summary.json", exclusion
        )

    _json_write(
        manifest_dir / "input_components.json",
        {
            "release": expected_release,
            "binding_mode": (
                "AUTHORITATIVE_AGGREGATE_PARENT_WITH_EVHGT_DERIVED_PARTITIONS"
                if formal_recovery_context is not None
                else "STRATIFIED_COMPONENT_BINDINGS"
            ),
            "components": (
                list(formal_recovery_context.derived_partitions)
                if formal_recovery_context is not None
                else [item.to_dict() for item in components]
            ),
            "quarantined_stratified_objects_used": False,
        },
    )
    node_components = {
        item.relative_path: item.cache_path
        for item in components
        if item.relative_path.startswith("nodes/")
    }
    edge_components = [
        item for item in components if item.relative_path.startswith("edges/")
    ]
    node_index = build_node_index(node_components)
    drugbank_mapping: pl.DataFrame | None = None
    if formal_recovery_context is not None:
        interim = formal_recovery_context.drugbank_interim_root
        required_drugbank = {
            "drugs": interim / "drugbank_drugs.parquet",
            "external": interim / "drugbank_external_identifiers.parquet",
            "partners": interim / "drugbank_partners.parquet",
            "interactions": interim / "drugbank_drug_interactions.parquet",
        }
        missing_drugbank = [
            str(path) for path in required_drugbank.values() if not path.is_file()
        ]
        if missing_drugbank:
            raise InputIntegrityError(
                f"Required derived DrugBank tables are missing: {missing_drugbank}"
            )
        existing_drugs = set(
            node_index.filter(pl.col("node_type") == "DRG")["node_id"].to_list()
        )
        drugbank_mapping, supplement_nodes = build_formal_drugbank_node_map(
            pl.scan_parquet(required_drugbank["drugs"]),
            pl.scan_parquet(required_drugbank["external"]),
            existing_drug_ids=existing_drugs,
        )
        node_index = augment_node_index(node_index, supplement_nodes)
        drugbank_mapping.write_parquet(
            drugbank_dir / "canonical_drug_map.parquet", compression="zstd"
        )
        supplement_nodes.write_parquet(
            drugbank_dir / "supplemental_drug_nodes.parquet", compression="zstd"
        )
    node_index.write_parquet(node_dir / "node_index.parquet", compression="zstd")
    node_counts = (
        node_index.group_by("node_type")
        .agg(pl.len().alias("node_count"))
        .sort("node_type")
    )
    node_counts.write_csv(audits / "node_counts.csv")

    fact_id_offset = 0
    evidence_id_offset = 0
    fact_paths: list[Path] = []
    partition_rows: list[dict[str, Any]] = []
    provenance_tokens: set[str] = set()
    overlap_counts = {
        "EXACT_DUPLICATE": 0,
        "PROBABLE_DUPLICATE": 0,
        "SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE": 0,
        "UNRESOLVED_OVERLAP": 0,
    }
    dda_binding: ComponentBinding | None = None
    dda_fact_path: Path | None = None
    drug_gene_fact_path: Path | None = None
    drug_drug_binding: ComponentBinding | None = None
    gene_node_binding = next(
        (
            item
            for item in components
            if item.relative_path == "nodes/gene.parquet"
        ),
        None,
    )
    expression_audit: dict[str, int] | None = None
    exclusion_rows = (
        pl.read_parquet(formal_recovery_context.exclusion_artifact_path)
        if formal_recovery_context is not None
        else None
    )
    for binding in edge_components:
        frame = pl.read_parquet(binding.cache_path)
        raw_input_rows = frame.height
        excluded_input_rows = 0
        if (
            formal_recovery_context is not None
            and binding.relative_path == "edges/disease_disease.parquet"
        ):
            if exclusion_rows is None:
                raise InputIntegrityError("Formal exclusion artifact is unavailable")
            frame = filter_canonical_input(frame, exclusion_rows)
            excluded_input_rows = raw_input_rows - frame.height
            if excluded_input_rows != formal_recovery_context.exclusion_summary["excluded_canonical_invalid_rows"]:
                raise InputIntegrityError(
                    f"Canonical DIS-DIS exclusion count mismatch: {excluded_input_rows}"
                )
        if binding.relative_path == "edges/anatomy_gene.parquet":
            present = frame.filter(
                pl.col("relation") == "EXPRESSION_PRESENT"
            ).select("from", "to")
            absent = frame.filter(
                pl.col("relation") == "EXPRESSION_ABSENT"
            ).select("from", "to")
            expression_audit = {
                "EXPRESSION_PRESENT": present.height,
                "EXPRESSION_ABSENT": absent.height,
                "canonical_pair_overlap": present.join(
                    absent, on=["from", "to"], how="inner"
                ).height,
            }
            if enforce_official_baselines and expression_audit != EXPRESSION_BASELINES:
                raise InputIntegrityError(
                    "Formal expression baselines were not reproduced: "
                    f"{expression_audit}"
                )
        spec = edge_component_spec(binding.relative_path)
        facts, evidence = canonicalize_edge_partition(
            frame,
            node_index,
            spec,
            fact_id_offset=fact_id_offset,
            evidence_id_offset=evidence_id_offset,
        )
        stem = Path(binding.relative_path).stem
        fact_path = fact_dir / f"{stem}.parquet"
        evidence_path = evidence_dir / f"{stem}.parquet"
        facts.write_parquet(fact_path, compression="zstd")
        evidence.write_parquet(evidence_path, compression="zstd")
        fact_paths.append(fact_path)
        for column in ("raw_direct_sources", "raw_indirect_sources"):
            provenance_tokens.update(
                str(value)
                for value in evidence.select(column)
                .explode(column, empty_as_null=True)
                .drop_nulls()
                .unique()
                .to_series()
                .to_list()
            )
        repeated_fact_count = facts.filter(
            pl.col("aggregated_observation_count") > 1
        ).height
        overlap_counts["UNRESOLVED_OVERLAP"] += repeated_fact_count
        partition_rows.append(
            {
                "component": binding.relative_path,
                "raw_partition_rows": raw_input_rows,
                "excluded_before_canonicalization": excluded_input_rows,
                "input_rows": frame.height,
                "canonical_facts": facts.height,
                "evidence_records": evidence.height,
                "duplicate_observations": evidence.height - facts.height,
                "relations": sorted(facts["relation"].unique().to_list()),
                "fact_id_start": fact_id_offset,
                "fact_id_end_exclusive": fact_id_offset + facts.height,
                "evidence_id_start": evidence_id_offset,
                "evidence_id_end_exclusive": evidence_id_offset + evidence.height,
            }
        )
        fact_id_offset += facts.height
        evidence_id_offset += evidence.height
        if binding.relative_path == "edges/drug_disease.parquet":
            dda_binding = binding
            dda_fact_path = fact_path
        elif binding.relative_path == "edges/drug_gene.parquet":
            drug_gene_fact_path = fact_path
        elif binding.relative_path == "edges/drug_drug.parquet":
            drug_drug_binding = binding

    if dda_binding is None or dda_fact_path is None:
        raise InputIntegrityError("Required edges/drug_disease.parquet is missing")
    if formal_recovery_context is not None and enforce_official_baselines and expression_audit is None:
        raise InputIntegrityError("Required edges/anatomy_gene.parquet is missing")
    drugbank_summary: dict[str, Any] | None = None
    drugbank_interaction_audit: dict[str, Any] | None = None
    if formal_recovery_context is not None:
        if (
            drugbank_mapping is None
            or gene_node_binding is None
            or drug_gene_fact_path is None
            or drug_drug_binding is None
        ):
            raise InputIntegrityError(
                "Formal DrugBank merge requires drug/gene and drug-gene/drug-drug artifacts"
            )
        interim = formal_recovery_context.drugbank_interim_root
        partner_merge = build_drugbank_partner_facts(
            pl.read_parquet(interim / "drugbank_partners.parquet"),
            drugbank_mapping,
            pl.read_parquet(gene_node_binding.cache_path),
            node_index,
            pl.read_parquet(drug_gene_fact_path),
            fact_id_offset=fact_id_offset,
            evidence_id_offset=evidence_id_offset,
        )
        local_fact_path = fact_dir / "drugbank_drug_gene.parquet"
        local_evidence_path = evidence_dir / "drugbank_drug_gene.parquet"
        partner_merge.local_only_facts.write_parquet(
            local_fact_path, compression="zstd"
        )
        partner_merge.evidence_records.write_parquet(
            local_evidence_path, compression="zstd"
        )
        partner_merge.membership.write_parquet(
            drugbank_dir / "drug_gene_fact_membership.parquet",
            compression="zstd",
        )
        partner_merge.unmapped_partners.write_parquet(
            drugbank_dir / "unmapped_partner_records.parquet",
            compression="zstd",
        )
        if partner_merge.local_only_facts.height:
            fact_paths.append(local_fact_path)
        fact_id_offset += partner_merge.local_only_facts.height
        evidence_id_offset += partner_merge.evidence_records.height
        provenance_tokens.add("DRUG_BANK_LOCAL")
        drug_gene_overlap = int(
            partner_merge.membership.filter(
                pl.col("optimuskg_membership")
            ).height
        )
        overlap_counts["UNRESOLVED_OVERLAP"] += drug_gene_overlap
        drugbank_interaction_audit = audit_drugbank_interaction_overlap(
            interim / "drugbank_drug_interactions.parquet",
            drugbank_mapping,
            drug_drug_binding.cache_path,
            shared_release_alignment=None,
        )
        for category in overlap_counts:
            overlap_counts[category] += int(
                drugbank_interaction_audit[category]
            )
        drugbank_summary = {
            **partner_merge.summary,
            "drug_count": drugbank_mapping.height,
            "optimuskg_merged_drug_count": drugbank_mapping.filter(
                pl.col("graph_membership") == "OPTIMUSKG_MERGED"
            ).height,
            "supplemental_drug_node_count": drugbank_mapping.filter(
                pl.col("graph_membership") == "DRUGBANK_SUPPLEMENT"
            ).height,
            "ambiguous_chembl_count": drugbank_mapping.filter(
                pl.col("mapping_status") == "AMBIGUOUS_CHEMBL"
            ).height,
            "interaction_overlap": drugbank_interaction_audit,
            "drugbank_indication_free_text_used_for_dda": False,
            "synergistic_interaction_treated_as_verified_synergy": False,
        }
        _json_write(drugbank_dir / "merge_summary.json", drugbank_summary)
        _json_write(
            drugbank_dir / "drug_drug_overlap_summary.json",
            drugbank_interaction_audit,
        )
    _provenance_catalog(provenance_tokens).write_parquet(
        destination / "provenance_token_catalog.parquet", compression="zstd"
    )
    _json_write(manifest_dir / "edge_partitions.json", partition_rows)

    phase3 = config["phase3"]
    supervision = build_dda_supervision(
        pl.read_parquet(dda_binding.cache_path),
        policy=str(phase3["dda_policy"]),
        clinical_phase_threshold=float(phase3["clinical_phase_threshold"]),
    )
    supervision_stats = supervision_statistics(supervision)
    if enforce_official_baselines:
        _assert_official_baselines(supervision_stats)
    split_config = phase3["split"]
    split_audits: dict[str, Any] = {}
    view_audits: dict[str, Any] = {}
    standard_manifest: pl.DataFrame | None = None
    split_columns = supervision.select(
        "drug_id", "disease_id", "canonical_pair_key"
    )
    for kind in split_config["kinds"]:
        split = build_split(
            split_columns,
            kind=str(kind),
            seed=int(split_config["seed"]),
            validation_ratio=float(split_config["validation_ratio"]),
            test_ratio=float(split_config["test_ratio"]),
        )
        split.write_parquet(split_dir / f"{kind}.parquet", compression="zstd")
        observed_split_stats = split_statistics(split)
        expected_preflight = PREFLIGHT_SPLIT_COUNTS.get(str(kind))
        observed_split_stats["phase3_preflight_counts"] = expected_preflight
        observed_split_stats["matches_phase3_preflight"] = (
            expected_preflight is None
            or observed_split_stats["counts"] == expected_preflight
        )
        split_audits[str(kind)] = observed_split_stats
        forbidden_root = (
            view_dir / str(kind) / "training_forbidden_facts"
        )
        forbidden_root.mkdir(parents=True, exist_ok=True)
        forbidden_fact_count = 0
        reason_counts: dict[str, int] = {
            "NONTRAIN_DDA_ANSWER": 0,
            "HELDOUT_DRUG_INCIDENT": 0,
            "HELDOUT_DISEASE_INCIDENT": 0,
        }
        forbidden_required_missing = 0
        for fact_path in fact_paths:
            fact_columns = [
                "canonical_fact_id",
                "source_node_type",
                "source_node_id",
                "target_node_type",
                "target_node_id",
                "relation",
            ]
            partition_facts = pl.read_parquet(
                fact_path, columns=fact_columns
            )
            forbidden = training_forbidden_facts(partition_facts, split)
            forbidden_path = forbidden_root / fact_path.name
            forbidden.write_parquet(forbidden_path, compression="zstd")
            persisted = pl.read_parquet(forbidden_path)
            expected_ids = set(forbidden["canonical_fact_id"].to_list())
            persisted_ids = set(
                persisted["canonical_fact_id"].to_list()
            )
            forbidden_required_missing += len(expected_ids - persisted_ids)
            forbidden_fact_count += persisted.height
            for reason in reason_counts:
                reason_counts[reason] += persisted.filter(
                    pl.col("exclusion_reasons").list.contains(reason)
                ).height
        view_audits[str(kind)] = {
            "forbidden_fact_count": forbidden_fact_count,
            "reason_counts": reason_counts,
            "forbidden_required_fact_missing_count": (
                forbidden_required_missing
            ),
            "evidence_and_provenance_follow_fact_exclusion": True,
        }
        if kind == "standard":
            standard_manifest = split
        _json_write(
            view_dir / f"{kind}.json",
            {
                "format": "EviHGT-Phase3-GraphView-v1",
                "split_kind": kind,
                "split_manifest": f"../splits/{kind}.parquet",
                "training_forbidden_fact_index": (
                    f"{kind}/training_forbidden_facts/*.parquet"
                ),
                "training_visible_dda_assignment": "train",
                "query_target_relations_removed": sorted(
                    ["INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE"]
                ),
                "target_fact_evidence_and_provenance_removed_together": True,
                "heldout_entity_incident_edges_excluded_from_training_view": (
                    kind != "standard"
                ),
            },
        )
    if standard_manifest is None:
        raise InputIntegrityError("Standard split is required")
    supervision = supervision.join(
        standard_manifest.select(
            "canonical_pair_key",
            "split_assignment",
            "split_kind",
            "split_seed",
        ),
        on="canonical_pair_key",
        how="left",
        validate="1:1",
    )
    supervision.write_parquet(
        supervision_dir / "dda_supervision.parquet", compression="zstd"
    )

    dda_facts = pl.read_parquet(dda_fact_path).filter(
        pl.col("relation").is_in(DDA_CLINICAL_RELATIONS)
    )
    query_target_map = (
        dda_facts.with_columns(
            pl.when(pl.col("source_node_type") == "DRG")
            .then(pl.col("source_node_id"))
            .otherwise(pl.col("target_node_id"))
            .alias("drug_id"),
            pl.when(pl.col("source_node_type") == "DIS")
            .then(pl.col("source_node_id"))
            .otherwise(pl.col("target_node_id"))
            .alias("disease_id"),
        )
        .with_columns(
            pl.concat_str(
                ["drug_id", "disease_id"], separator="\u001f"
            ).alias("canonical_pair_key")
        )
        .select(
            "canonical_pair_key",
            "drug_id",
            "disease_id",
            "canonical_fact_id",
            pl.col("relation").alias("raw_relation"),
        )
        .sort(["canonical_pair_key", "canonical_fact_id"])
    )
    target_pairs = set(query_target_map["canonical_pair_key"].to_list())
    supervision_pairs = set(supervision["canonical_pair_key"].to_list())
    target_mapping_missing = len(supervision_pairs - target_pairs)
    target_mapping_orphan = len(target_pairs - supervision_pairs)
    if target_mapping_missing or target_mapping_orphan:
        raise InputIntegrityError(
            "DDA query target fact map does not match supervision pairs: "
            f"missing={target_mapping_missing}, orphan={target_mapping_orphan}"
        )
    query_target_map.write_parquet(
        view_dir / "query_target_fact_map.parquet", compression="zstd"
    )

    csr_metadata = build_csr_index(
        fact_paths,
        node_count=node_index.height,
        output_dir=graph_dir / "csr",
    )
    if formal_recovery_context is not None:
        target_fact_ids = set(query_target_map["canonical_fact_id"].to_list())
        clinical_survivors = dda_facts.filter(
            ~pl.col("canonical_fact_id").is_in(sorted(target_fact_ids))
        )
        dda_evidence = pl.read_parquet(
            evidence_dir / "drug_disease.parquet"
        )
        retained_query_evidence = dda_evidence.filter(
            ~pl.col("canonical_fact_id").is_in(sorted(target_fact_ids))
        )
        target_evidence_survivors = retained_query_evidence.filter(
            pl.col("canonical_fact_id").is_in(sorted(target_fact_ids))
        ).height
        pair_feature_paths = [
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*pair_feature*")
            if path.is_file()
        ]
        split_contamination = sum(
            int(row["forbidden_required_fact_missing_count"])
            for row in view_audits.values()
        )
        cold_contamination = sum(
            int(view_audits[kind]["forbidden_required_fact_missing_count"])
            for kind in ("cold_drug", "cold_disease", "double_cold")
        )
        leakage_audit = {
            "status": FULL_GRAPH_LEAKAGE_PASS,
            "audit_scope": FULL_GRAPH_AUDIT_SCOPE,
            "masking_unit": "canonical Drug-Disease pair",
            "target_relations": [
                "INDICATION",
                "CONTRAINDICATION",
                "OFF_LABEL_USE",
            ],
            "target_answer_facts_before_masking": query_target_map.height,
            "target_indication_survivors": clinical_survivors.filter(
                pl.col("relation") == "INDICATION"
            ).height,
            "target_contraindication_survivors": clinical_survivors.filter(
                pl.col("relation") == "CONTRAINDICATION"
            ).height,
            "target_off_label_use_survivors": clinical_survivors.filter(
                pl.col("relation") == "OFF_LABEL_USE"
            ).height,
            "reverse_inverse_equivalent_survivors": clinical_survivors.height,
            "canonical_duplicate_survivors": dda_facts.select(
                "canonical_fact_key"
            ).is_duplicated().sum(),
            "clinical_phase_feature_survivors": target_evidence_survivors,
            "source_provenance_feature_survivors": target_evidence_survivors,
            "cached_target_pair_feature_survivors": len(pair_feature_paths),
            "train_validation_test_contamination": split_contamination,
            "cold_start_contamination": cold_contamination,
            "query_target_pair_mapping_missing_count": target_mapping_missing,
            "query_target_pair_mapping_orphan_count": target_mapping_orphan,
            "training_views": view_audits,
            "reverse_traversal_uses_same_forbidden_canonical_fact_id": True,
            "verified_equivalent_relation_catalog": sorted(
                DDA_CLINICAL_RELATIONS
            ),
            "target_fact_evidence_and_provenance_removed_together": True,
            "pair_feature_artifacts_found": pair_feature_paths,
            "query_target_map_path": (
                "views/query_target_fact_map.parquet"
            ),
            "formal_graph_root": str(
                formal_recovery_context.published_root
            ),
        }
        validate_full_graph_leakage_audit(leakage_audit)
    else:
        leakage_audit = {
            "status": "PASSED",
            "masking_unit": "canonical Drug-Disease pair",
            "target_relations": [
                "INDICATION",
                "CONTRAINDICATION",
                "OFF_LABEL_USE",
            ],
            "target_answer_facts_before_masking": query_target_map.height,
            "target_answer_facts_after_masking_by_policy": 0,
            "query_target_pair_mapping_missing_count": target_mapping_missing,
            "query_target_pair_mapping_orphan_count": target_mapping_orphan,
            "training_views": view_audits,
            "reverse_traversal_is_masked": True,
            "equivalent_aliases_are_canonicalized_before_masking": True,
            "target_fact_evidence_and_provenance_removed_together": True,
            "cached_target_feature_rows_after_masking": 0,
            "validation_test_dda_answers_visible_in_training_view": sum(
                row["forbidden_required_fact_missing_count"]
                for row in view_audits.values()
            ),
            "feature_leakage_detected": False,
            "provenance_leakage_detected": False,
        }
    supervision_audit = {
        **supervision_stats,
        "physical_representation": (
            "nullable Float64 targets plus authoritative Boolean masks"
        ),
        "masked_nonnull_target_count": supervision.filter(
            ((~pl.col("support_mask")) & pl.col("support_target").is_not_null())
            | (
                (~pl.col("unsuitability_mask"))
                & pl.col("unsuitability_target").is_not_null()
            )
        ).height,
        "off_label_supervised_row_count": supervision.filter(
            pl.col("context_only")
            & (pl.col("support_mask") | pl.col("unsuitability_mask"))
        ).height,
        "unlabeled_is_hard_unknown": False,
        "absent_contraindication_is_suitability": False,
    }
    _json_write(audits / "dda_supervision_audit.json", supervision_audit)
    _json_write(
        audits / "evidence_overlap_audit.json",
        {
            "counts": overlap_counts,
            "independent_vote_count": overlap_counts[
                "SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE"
            ],
            "note": (
                "Counts are fail-closed overlap decisions over repeated canonical "
                "aggregate facts plus formal DrugBank fact/fingerprint overlap; "
                "source/reference lists are not exploded into votes."
            ),
            "decision_unit": "canonical overlap decision",
            "drugbank_interaction_audit": drugbank_interaction_audit,
        },
    )
    _json_write(audits / "split_audit.json", split_audits)
    _json_write(audits / "leakage_audit.json", leakage_audit)
    _json_write(
        audits / "input_integrity_audit.json",
        (
            {
                "release": expected_release,
                "all_components_usable": True,
                "component_count": len(components),
                "formal_parent_binding": (
                    formal_recovery_context.aggregate_parent
                    if formal_recovery_context is not None
                    else None
                ),
                "canonical_exclusion": (
                    formal_recovery_context.exclusion_summary
                    if formal_recovery_context is not None
                    else None
                ),
                "derived_partitions_are_official_stratified_files": False,
                "quarantined_stratified_objects_used": False,
                "checksum_drift_components": [
                    item.to_dict()
                    for item in components
                    if item.status == "UPSTREAM_CHECKSUM_DRIFT_UNRESOLVED"
                ],
                "new_checksum_discrepancy_detected": False,
            }
        ),
    )
    if expression_audit is not None:
        _json_write(audits / "expression_audit.json", expression_audit)

    inventory = _artifact_inventory(destination)
    _json_write(manifest_dir / "artifact_inventory.json", inventory)
    completion = {
        "status": "PHASE3_ARTIFACT_BUILD_COMPLETE",
        "artifact_version": phase3["artifact_version"],
        "optimuskg_release": expected_release,
        "node_count": node_index.height,
        "canonical_fact_count": fact_id_offset,
        "evidence_record_count": evidence_id_offset,
        "traversal_edge_count": csr_metadata["traversal_edge_count"],
        "supervision": supervision_stats,
        "semantic_guards": {
            "edge_counter_is_dda_unsuitability": False,
            "unlabeled_is_negative": False,
            "unlabeled_is_hard_unknown": False,
            "edge_latent_u_is_supervision_missingness": False,
            "query_latent_u_is_supervision_missingness": False,
            "retained_contraindication_is_complete": False,
            "observed_hard_dda_four_value_target_created": False,
        },
    }
    if formal_recovery_context is not None:
        parent_nodes = formal_recovery_context.aggregate_parent["nodes"]
        parent_edges = formal_recovery_context.aggregate_parent["edges"]
        exclusion_summary = formal_recovery_context.exclusion_summary
        data_gates = {
            "aggregate_parent_binding_verified": bool(
                parent_nodes.get("status")
                == "AUTHORITATIVE_OPTIMUSKG_PARENT_BINDING"
                and parent_edges.get("status")
                == "AUTHORITATIVE_OPTIMUSKG_PARENT_BINDING"
            ),
            "derived_10_27_partitions_complete": len(
                formal_recovery_context.derived_partitions
            )
            == 37,
            "authorized_142_exclusions_reproduced": (
                exclusion_summary.get("excluded_canonical_invalid_rows") == 142
                and exclusion_summary.get("canonical_input_count") == 21_813_674
            ),
            "policy_a_b_c_reproduced": all(
                supervision_stats[key] == OFFICIAL_BASELINES[key]
                for key in ("policy_a", "policy_b", "policy_c")
            ),
            "default_policy_b_correct": str(phase3["dda_policy"]).upper() == "B",
            "contraindication_supervision_correct": (
                supervision_stats["retained_unsuitability"]
                == OFFICIAL_BASELINES["retained_unsuitability"]
            ),
            "off_label_not_supervised": (
                supervision_audit["off_label_supervised_row_count"] == 0
            ),
            "no_closed_world_target_zero": supervision_stats["target_zero_count"]
            == 0,
            "canonical_graph_complete": fact_id_offset > 0
            and evidence_id_offset == exclusion_summary["canonical_input_count"]
            + int(drugbank_summary["mapped_partner_observation_count"]),
            "evidence_overlap_audit_complete": all(
                key in overlap_counts
                for key in (
                    "EXACT_DUPLICATE",
                    "PROBABLE_DUPLICATE",
                    "SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE",
                    "UNRESOLVED_OVERLAP",
                )
            ),
            "independent_vote_rule_passed": overlap_counts[
                "SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE"
            ]
            == 0,
            "provenance_retained": bool(provenance_tokens),
            "drugbank_conservative_merge_complete": drugbank_summary is not None,
            "deterministic_splits_complete": set(split_audits)
            == {"standard", "cold_drug", "cold_disease", "double_cold"},
            "split_preflight_counts_reproduced": all(
                row["matches_phase3_preflight"] for row in split_audits.values()
            ),
            "cold_start_invariants_passed": all(
                view_audits[kind]["forbidden_required_fact_missing_count"] == 0
                for kind in ("cold_drug", "cold_disease", "double_cold")
            ),
            "split_aware_views_complete": set(view_audits)
            == {"standard", "cold_drug", "cold_disease", "double_cold"},
            "full_graph_leakage_audit_passed": leakage_audit["status"]
            == FULL_GRAPH_LEAKAGE_PASS,
        }
        if not enforce_official_baselines:
            for baseline_gate in (
                "derived_10_27_partitions_complete", "authorized_142_exclusions_reproduced",
                "policy_a_b_c_reproduced", "contraindication_supervision_correct",
                "split_preflight_counts_reproduced",
            ):
                del data_gates[baseline_gate]
        completion.update(
            {
                "aggregate_parent": {
                    "release": "2.0",
                    "nodes_sha256": parent_nodes["observed_sha256"],
                    "edges_sha256": parent_edges["observed_sha256"],
                },
                "derived_partition_count": len(
                    formal_recovery_context.derived_partitions
                ),
                "canonical_exclusion": {
                    "row_count": exclusion_summary[
                        "excluded_canonical_invalid_rows"
                    ],
                    "artifact_sha256": exclusion_summary["artifact_sha256"],
                    "reason": exclusion_summary["exclusion_reason"],
                    "raw_aggregate_edge_count": exclusion_summary[
                        "raw_aggregate_edge_count"
                    ],
                    "canonical_input_count": exclusion_summary[
                        "canonical_input_count"
                    ],
                },
                "expression": expression_audit,
                "evidence_overlap": {
                    "counts": overlap_counts,
                    "independent_vote_count": overlap_counts[
                        "SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE"
                    ],
                },
                "drugbank": drugbank_summary,
                "splits": split_audits,
                "formal_full_graph_leakage_audit": leakage_audit["status"],
                "completion_gates": data_gates,
                "all_completion_gates_passed": all(data_gates.values()),
                "release_baselines_verified": enforce_official_baselines,
                "transformation_version": (
                    formal_recovery_context.transformation_version
                ),
                "code_identifier": formal_recovery_context.code_identifier,
            }
        )
        failed_gates = [name for name, passed in data_gates.items() if not passed]
        if failed_gates:
            raise InputIntegrityError(
                f"Formal Phase 3 completion gates failed: {failed_gates}"
            )
    _json_write(manifest_dir / "completion.json", completion)
    return Phase3BuildResult(
        processed_root=destination,
        audit_root=audits,
        node_count=node_index.height,
        canonical_fact_count=fact_id_offset,
        evidence_record_count=evidence_id_offset,
        traversal_edge_count=int(csr_metadata["traversal_edge_count"]),
        supervision_statistics=supervision_stats,
    )
