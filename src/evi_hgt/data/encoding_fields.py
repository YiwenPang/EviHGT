"""Exact raw-property and provenance tokenization contract."""
from __future__ import annotations
import hashlib
from typing import Any, Iterable
from evi_hgt.data.provenance import classify_provenance_token

NUMERIC_FIELDS = (
    "expression_rank",
    "evidence_score",
    "evidence_count",
    "evidence_index",
    "number_of_pmids",
    "highest_clinical_trial_phase",
)

BOOLEAN_FIELDS = ("qualifier_not",)

EVIDENCE_FIELD_MARKERS = (
    "evidence",
    "reference",
    "eco_id",
    "source_id",
    "source_url",
    "mechanism",
)

RECORD_IDENTIFIER_FIELDS = {"structure_id", "drug_disease_id"}

EXPECTED_TOKEN_FIELDS: dict[str, tuple[str, ...]] = {
    "anatomy_gene": ("call_quality",),
    "disease_phenotype": ("evidence_type", "references"),
    "drug_disease": ("reference_ids", "structure_id", "drug_disease_id"),
    "drug_gene": ("mechanisms_of_action", "source_ids", "source_urls"),
}

OVERLAP_IDS = {
    None: 0,
    "EXACT_DUPLICATE": 1,
    "PROBABLE_DUPLICATE": 2,
    "SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE": 3,
    "UNRESOLVED_OVERLAP": 4,
}

PROVENANCE_KIND_IDS = {
    "DATABASE": 0,
    "PMID": 1,
    "DOI": 2,
    "GO_REF": 3,
    "RECORD_IDENTIFIER": 4,
    "UNKNOWN_RAW": 5,
}

PROVENANCE_ROLE_IDS = {
    "DIRECT": 0,
    "INDIRECT": 1,
    "REFERENCE": 2,
    "RECORD_IDENTIFIER": 3,
    "SOURCE_RELEASE": 4,
}

def _stable_bucket(namespace: str, value: Any, buckets: int) -> int:
    payload = f"phase4-token-v1\x1f{namespace}\x1f{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % buckets

def _flatten_properties(
    value: dict[str, Any], prefix: str = ""
) -> Iterable[tuple[str, Any]]:
    for key in sorted(value):
        item = value[key]
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            yield from _flatten_properties(item, path)
        else:
            yield path, item

def _as_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def _provenance_kind(value: str) -> int:
    kind = classify_provenance_token(value).kind.value
    return PROVENANCE_KIND_IDS[kind]
