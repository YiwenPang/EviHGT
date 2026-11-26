"""Phase 3 scientific schema and fail-fast supervision invariants.

Edge-level counter-evidence is relation-specific.  It is deliberately distinct
from the Drug-Disease contraindication/treatment-unsuitability target.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl


class EdgeEvidenceRole(StrEnum):
    """Optional proposition-scoped evidence roles for general graph edges."""

    SUPPORT = "edge_support"
    COUNTER = "edge_counter_evidence"


class DDATargetChannel(StrEnum):
    """Independent partially observed Drug-Disease supervision channels."""

    SUPPORT = "dda_support"
    UNSUITABILITY = "dda_unsuitability"


class OverlapCategory(StrEnum):
    """Accepted evidence identity/overlap decisions."""

    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    PROBABLE_DUPLICATE = "PROBABLE_DUPLICATE"
    SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE = (
        "SAME_CANONICAL_FACT_INDEPENDENT_EVIDENCE"
    )
    UNRESOLVED_OVERLAP = "UNRESOLVED_OVERLAP"


class SupervisionSemanticsError(ValueError):
    """Raised when an artifact would fabricate closed-world supervision."""


@dataclass(frozen=True, slots=True)
class DDASupervisionRow:
    """Minimal nullable representation of independent DDA observations."""

    drug_id: str
    disease_id: str
    support_target: float | None
    support_mask: bool
    unsuitability_target: float | None
    unsuitability_mask: bool

    @classmethod
    def unlabeled(cls, drug_id: str, disease_id: str) -> "DDASupervisionRow":
        """Create missing supervision without inventing zero or hard Unknown."""

        return cls(
            drug_id=drug_id,
            disease_id=disease_id,
            support_target=None,
            support_mask=False,
            unsuitability_target=None,
            unsuitability_mask=False,
        )


_REQUIRED_SUPERVISION_COLUMNS = {
    "drug_id",
    "disease_id",
    "support_target",
    "support_mask",
    "unsuitability_target",
    "unsuitability_mask",
}
_FORBIDDEN_HARD_STATE_COLUMNS = {
    "four_value_label",
    "hard_unknown",
    "observed_four_value_state",
}


def validate_supervision_frame(frame: pl.DataFrame) -> None:
    """Validate positive-only masked DDA supervision.

    A false mask requires a null target.  A true mask currently permits only an
    explicit positive observation (1.0); Phase 3 has no scientifically verified
    target-zero source.  Missingness is not a hard four-valued Unknown state.
    """

    missing = _REQUIRED_SUPERVISION_COLUMNS - set(frame.columns)
    if missing:
        raise SupervisionSemanticsError(
            f"Missing supervision columns: {sorted(missing)}"
        )
    forbidden = _FORBIDDEN_HARD_STATE_COLUMNS & set(frame.columns)
    if forbidden:
        raise SupervisionSemanticsError(
            "Observed hard four-valued supervision is forbidden: "
            f"{sorted(forbidden)}"
        )

    duplicate_pairs = frame.select("drug_id", "disease_id").is_duplicated()
    if duplicate_pairs.any():
        raise SupervisionSemanticsError("Canonical DDA pairs must be unique")

    for target, mask in (
        ("support_target", "support_mask"),
        ("unsuitability_target", "unsuitability_mask"),
    ):
        if frame.filter(~pl.col(mask) & pl.col(target).is_not_null()).height:
            raise SupervisionSemanticsError(
                f"A masked channel target must be null: {target}"
            )
        if frame.filter(pl.col(mask) & pl.col(target).is_null()).height:
            raise SupervisionSemanticsError(
                f"An observed channel must have a target: {target}"
            )
        invalid_observed = frame.filter(
            pl.col(target).is_not_null() & (pl.col(target) != 1.0)
        )
        if invalid_observed.height:
            raise SupervisionSemanticsError(
                "Phase 3 targets are positive observations only; no verified "
                f"closed-world zero exists for {target}"
            )

    if frame.filter(pl.col("support_mask") & pl.col("unsuitability_mask")).height:
        raise SupervisionSemanticsError(
            "Observed hard Both supervision is unavailable in release 2.0"
        )

