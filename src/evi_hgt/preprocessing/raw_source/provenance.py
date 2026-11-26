"""Conservative classification of heterogeneous provenance tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class ProvenanceKind(StrEnum):
    DATABASE = "DATABASE"
    PMID = "PMID"
    DOI = "DOI"
    GO_REF = "GO_REF"
    RECORD_IDENTIFIER = "RECORD_IDENTIFIER"
    UNKNOWN_RAW = "UNKNOWN_RAW"


class ProvenanceRole(StrEnum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"


@dataclass(frozen=True, slots=True)
class ProvenanceToken:
    raw: str
    normalized: str
    kind: ProvenanceKind
    role: ProvenanceRole | None = None


_DATABASE_NAMES = {
    "BGEE",
    "CHEBI",
    "DRUG_BANK",
    "DRUG_CENTRAL",
    "GENE_ONTOLOGY",
    "GWAS_CATALOG",
    "HUMAN_PHENOTYPE_ONTOLOGY",
    "MONDO",
    "OPEN_TARGETS",
    "REACTOME",
}
_PMID = re.compile(r"^PMID:\d+$", re.IGNORECASE)
_DOI = re.compile(r"^(?:DOI:)?10\.\d{4,9}/\S+$", re.IGNORECASE)
_GO_REF = re.compile(r"^GO_REF:\d+$", re.IGNORECASE)
_RECORD_ID = re.compile(
    r"^(?:NCT\d{8}|(?:REACTOME|CLINICAL_TRIAL|DRUGBANK_REF):\S+)$",
    re.IGNORECASE,
)


def classify_provenance_token(
    raw: str, *, role: ProvenanceRole | None = None
) -> ProvenanceToken:
    """Classify only tokens with verified patterns and always retain raw text."""

    value = raw.strip()
    upper = value.upper()
    if upper in _DATABASE_NAMES:
        kind = ProvenanceKind.DATABASE
    elif _PMID.fullmatch(value):
        kind = ProvenanceKind.PMID
    elif _DOI.fullmatch(value):
        kind = ProvenanceKind.DOI
    elif _GO_REF.fullmatch(value):
        kind = ProvenanceKind.GO_REF
    elif _RECORD_ID.fullmatch(value):
        kind = ProvenanceKind.RECORD_IDENTIFIER
    else:
        kind = ProvenanceKind.UNKNOWN_RAW
    return ProvenanceToken(raw=raw, normalized=value, kind=kind, role=role)


def classify_provenance_values(
    direct: Iterable[str] | None,
    indirect: Iterable[str] | None,
) -> tuple[ProvenanceToken, ...]:
    """Preserve provenance list membership without treating it as evidence votes."""

    return tuple(
        [
            classify_provenance_token(value, role=ProvenanceRole.DIRECT)
            for value in (direct or ())
        ]
        + [
            classify_provenance_token(value, role=ProvenanceRole.INDIRECT)
            for value in (indirect or ())
        ]
    )

