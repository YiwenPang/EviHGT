"""Build the canonical graph from the pinned OptimusKG 2.0 aggregate release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4

from lxml import etree

from .raw_source.drugbank_loader import DRUGBANK_NAMESPACE, convert_drugbank_xml
from .raw_source.phase3_formal_pipeline import run_authorized_phase3_recovery_build
from .raw_source.phase3_recovery_build import (
    AggregateParentExpectation, bind_formal_aggregate_parent,
)


OPTIMUSKG_20 = {
    "nodes": {
        "size_bytes": 163857672,
        "md5": "5e0609a5cfdde70ecf86cbfdef701432",
        "sha256": "0601aaf9a659ca0f89aa9ab8b8baa2d3318b2d21e1d93d678d53e800a86eddad",
    },
    "edges": {
        "size_bytes": 177526798,
        "md5": "24dd404e9f52b935c91849a99aaed517",
        "sha256": "2a551d832c7d10758d2d5a6a636fb8eef863beeb65967b74546385ba3e7f6940",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _drugbank_envelope(xml: Path, xsd: Path) -> dict:
    """Compile the supplied XSD and check the XML envelope, not every XML record."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    schema_doc = etree.parse(str(xsd), parser)
    etree.XMLSchema(schema_doc)
    if schema_doc.getroot().get("targetNamespace") != DRUGBANK_NAMESPACE:
        raise ValueError("DrugBank XSD targetNamespace does not match the parser")
    with xml.open("rb") as handle:
        context = etree.iterparse(handle, events=("start",), resolve_entities=False, no_network=True)
        _, root = next(context)
        if root.tag != f"{{{DRUGBANK_NAMESPACE}}}drugbank":
            raise ValueError("Expected the DrugBank XML root and namespace")
        if root.get("version") != "5.1":
            raise ValueError("UNVERIFIED DrugBank XML schema version; expected 5.1")
        header = dict(root.attrib)
        del context
    return {
        "xml_header": header,
        "namespace": DRUGBANK_NAMESPACE,
        "validation_scope": "XSD compilation and XML root/version; structured streaming parser",
        "full_document_xsd_validation_performed": False,
    }


def _build_config(nodes: Path, edges: Path, destination: Path, audit: Path,
                  staging: Path, interim: Path, identities: dict,
                  expected_exclusions: int = 142) -> dict:
    return {"phase3": {
        "artifact_version": "v1", "optimuskg_release": "2.0", "dda_policy": "B",
        "clinical_phase_threshold": 2.0,
        "paths": {"processed_root": str(destination), "staging_root": str(staging)},
        "split": {"seed": 42, "validation_ratio": 0.1, "test_ratio": 0.1,
                  "kinds": ["standard", "cold_drug", "cold_disease", "double_cold"]},
        "formal_recovery": {
            "enabled": True, "quarantined_stratified_inputs_allowed": False,
            "transformation_version": "aggregate-restratification-v1",
            "build_code_version": "EviHGT-public-raw-canonical-v1",
            "formal_audit_root": str(audit), "drugbank_interim_root": str(interim),
            "nodes_parent": {"path": str(nodes), **identities["nodes"]},
            "edges_parent": {"path": str(edges), **identities["edges"]},
            "canonical_exclusion": {"expected_count": expected_exclusions},
        },
    }}


def build_canonical(nodes_path: str | Path, edges_path: str | Path,
                    drugbank_xml: str | Path, drugbank_xsd: str | Path,
                    destination: str | Path, *, audit_dir: str | Path,
                    progress: Callable[[str], None] | None = None) -> Path:
    """Create a new canonical artifact tree; existing destinations are never overwritten.

    OptimusKG aggregate bytes are pinned to release 2.0. DrugBank 5.1 exports may
    differ in record counts, so their observed identities are recorded separately.
    Derived conversion and interrupted graph files remain in a unique staging
    directory next to the destination. Audits remain in the supplied audit directory.
    """
    nodes, edges, xml, xsd = (Path(p).resolve() for p in
                              (nodes_path, edges_path, drugbank_xml, drugbank_xsd))
    final = Path(destination).resolve()
    audits = Path(audit_dir).resolve()
    if final.exists():
        raise FileExistsError(f"Canonical destination already exists: {final}")
    for source in (nodes, edges, xml, xsd):
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.is_relative_to(final) or source.is_relative_to(audits):
            raise ValueError("Output directories may not contain raw input files")
    if audits.is_relative_to(final) or final.is_relative_to(audits):
        raise ValueError("audit_dir must be external to the canonical data tree")
    if progress:
        progress("校验 OptimusKG 2.0 原始文件身份与 DrugBank XML/XSD 结构")
    for kind, source in (("nodes", nodes), ("edges", edges)):
        identity = OPTIMUSKG_20[kind]
        bind_formal_aggregate_parent(source, AggregateParentExpectation(
            kind, "2.0", identity["size_bytes"], identity["md5"], identity["sha256"]))
    envelope = _drugbank_envelope(xml, xsd)
    raw_identity = {str(source): {"size_bytes": source.stat().st_size,
                    "sha256": _sha256(source)} for source in (xml, xsd)}
    run_id = uuid4().hex
    staging = final.parent / ".staging" / f"raw_{run_id}"
    interim = staging / "drugbank_interim"
    run_audit = audits / f"canonical_{run_id}"
    run_audit.mkdir(parents=True)
    (run_audit / "raw_input_identity.json").write_text(json.dumps({
        "optimuskg_release": "2.0", "optimuskg": OPTIMUSKG_20,
        "drugbank": raw_identity, "drugbank_schema": envelope,
    }, indent=2) + "\n", encoding="utf-8")
    if progress:
        progress("流式解析 DrugBank XML 并生成派生 Parquet 表")
    convert_drugbank_xml(xml, interim)
    # The graph writer expects its own fresh audit folder and publishes completion last.
    config = _build_config(nodes, edges, final, run_audit / "graph", staging,
                           interim, OPTIMUSKG_20)
    config["phase3"]["formal_recovery"]["drugbank_source_identity"] = {
        "files": raw_identity, "schema_envelope": envelope,
    }
    result = run_authorized_phase3_recovery_build(
        config=config, project_root=final.parent, progress=progress)
    if progress:
        progress("规范图已完成完整性校验并原子封存")
    return result.processed_root
