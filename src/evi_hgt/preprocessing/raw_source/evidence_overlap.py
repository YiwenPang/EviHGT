"""Fail-closed evidence overlap decisions for canonical biomedical facts."""

from __future__ import annotations

from dataclasses import dataclass

from evi_hgt.preprocessing.raw_source.phase3_schema import OverlapCategory


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    """Only verified fields that may contribute to an overlap decision."""

    source_release: str | None
    source_record_id: str | None
    reference_ids: tuple[str, ...]
    source_membership: tuple[str, ...]
    semantic_fingerprint: str


@dataclass(frozen=True, slots=True)
class OverlapDecision:
    category: OverlapCategory
    independent_vote_eligible: bool
    decision_basis: str
    unresolved_reason: str | None = None


def independent_vote_eligible(category: OverlapCategory | str) -> bool:
    """Return the one accepted vote-eligible overlap category."""

    return OverlapCategory(category) is (
        OverlapCategory.SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE
    )


def _decision(
    category: OverlapCategory,
    basis: str,
    unresolved_reason: str | None = None,
) -> OverlapDecision:
    return OverlapDecision(
        category=category,
        independent_vote_eligible=independent_vote_eligible(category),
        decision_basis=basis,
        unresolved_reason=unresolved_reason,
    )


def classify_overlap(
    left: EvidenceIdentity,
    right: EvidenceIdentity,
    *,
    verified_distinct_records: bool = False,
    verified_disjoint_provenance_chains: bool = False,
) -> OverlapDecision:
    """Classify two observations without inferring independence from sources.

    The caller must supply independently verified record and provenance-chain
    distinctness.  Different database names alone never set those flags.
    """

    if left.semantic_fingerprint != right.semantic_fingerprint:
        return _decision(
            OverlapCategory.UNRESOLVED_OVERLAP,
            "canonical fact overlaps but available semantic fingerprints differ",
            "semantic identity cannot be established",
        )

    shared_references = set(left.reference_ids) & set(right.reference_ids)
    if (
        verified_distinct_records
        and verified_disjoint_provenance_chains
        and left.source_record_id
        and right.source_record_id
        and left.source_record_id != right.source_record_id
        and not shared_references
    ):
        return _decision(
            OverlapCategory.SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE,
            "verified distinct records with disjoint provenance chains",
        )

    if (
        left.source_release
        and left.source_release == right.source_release
        and left.source_record_id
        and left.source_record_id == right.source_record_id
        and left.reference_ids == right.reference_ids
        and left.source_membership == right.source_membership
    ):
        return _decision(
            OverlapCategory.EXACT_DUPLICATE,
            "same pinned release, source record, references, provenance, and semantics",
        )

    if (
        left.source_membership == right.source_membership
        and left.reference_ids == right.reference_ids
        and (not left.source_release or not right.source_release)
    ):
        return _decision(
            OverlapCategory.PROBABLE_DUPLICATE,
            "matching available semantics/provenance without a shared versioned record fingerprint",
        )

    reason = (
        "shared PMID/DOI cannot prove independence"
        if shared_references
        else "record identity or provenance-chain independence is unverified"
    )
    return _decision(
        OverlapCategory.UNRESOLVED_OVERLAP,
        "available fields do not establish identity or independence",
        reason,
    )

