"""Independent supervision roles for observed and unlabeled drug-disease pairs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import polars as pl


PAIR_SEPARATOR = "\x1f"


class SampleRole(str, Enum):
    POSITIVE = "POSITIVE"
    UNLABELED = "UNLABELED"
    EXCLUDED_CONTEXT = "EXCLUDED_CONTEXT"


@dataclass(frozen=True, slots=True)
class ChannelSupervision:
    role: SampleRole
    target: float | None
    observed_mask: bool
    pu_risk_eligible: bool


@dataclass(frozen=True, slots=True)
class QuerySupervision:
    drug_id: str
    disease_id: str
    canonical_pair_key: str
    support: ChannelSupervision
    unsuitability: ChannelSupervision
    sample_source: str
    observed_relation: str | None
    observed_four_value_target: None = None

    @property
    def loss_eligible(self) -> bool:
        return any(
            channel.role is SampleRole.POSITIVE or channel.pu_risk_eligible
            for channel in (self.support, self.unsuitability)
        )


def _validate_channel(
    row: Mapping[str, Any],
    channel: str,
) -> tuple[float | None, bool]:
    target_value = row.get(f"{channel}_target")
    mask = bool(row.get(f"{channel}_mask", False))
    target = None if target_value is None else float(target_value)
    if mask:
        if target != 1.0:
            if target == 0.0:
                raise ValueError(f"{channel} contains an observed zero target")
            raise ValueError(f"masked {channel} target must equal 1.0")
    elif target is not None:
        raise ValueError(f"unmasked {channel} target must remain null")
    return target, mask


def validate_phase3_supervision_frame(frame: pl.DataFrame) -> None:
    """Validate nullable targets and observation masks for both channels."""

    if any("four_value" in name.lower() for name in frame.columns):
        raise ValueError("observed four-value supervision is forbidden")
    required = {
        "support_target",
        "support_mask",
        "unsuitability_target",
        "unsuitability_mask",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing Phase 3 supervision columns: {sorted(missing)}")
    for row in frame.iter_rows(named=True):
        _validate_channel(row, "support")
        _validate_channel(row, "unsuitability")


def _channel(target: float | None, mask: bool) -> ChannelSupervision:
    if mask:
        return ChannelSupervision(
            role=SampleRole.POSITIVE,
            target=target,
            observed_mask=True,
            pu_risk_eligible=False,
        )
    return ChannelSupervision(
        role=SampleRole.UNLABELED,
        target=None,
        observed_mask=False,
        pu_risk_eligible=False,
    )


def _excluded_channel() -> ChannelSupervision:
    return ChannelSupervision(
        role=SampleRole.EXCLUDED_CONTEXT,
        target=None,
        observed_mask=False,
        pu_risk_eligible=False,
    )


def derive_observed_query_supervision(row: Mapping[str, Any]) -> QuerySupervision:
    """Translate an observed relation without filling missing labels."""

    support_target, support_mask = _validate_channel(row, "support")
    unsuitability_target, unsuitability_mask = _validate_channel(
        row, "unsuitability"
    )
    relation = str(row["raw_relation"])
    policy_b_member = bool(row.get("policy_b_member", False))
    excluded = relation == "OFF_LABEL_USE" or (
        relation == "INDICATION" and not policy_b_member
    )
    if excluded:
        support = _excluded_channel()
        unsuitability = _excluded_channel()
    else:
        support = _channel(support_target, support_mask)
        unsuitability = _channel(unsuitability_target, unsuitability_mask)
    drug_id = str(row["drug_id"])
    disease_id = str(row["disease_id"])
    return QuerySupervision(
        drug_id=drug_id,
        disease_id=disease_id,
        canonical_pair_key=str(
            row.get("canonical_pair_key", f"{drug_id}{PAIR_SEPARATOR}{disease_id}")
        ),
        support=support,
        unsuitability=unsuitability,
        sample_source="OBSERVED_PHASE3_DDA",
        observed_relation=relation,
    )


def make_unlabeled_query_supervision(
    drug_id: str,
    disease_id: str,
) -> QuerySupervision:
    """Create an unlabeled candidate role record, never a zero-valued label."""

    channel = ChannelSupervision(
        role=SampleRole.UNLABELED,
        target=None,
        observed_mask=False,
        pu_risk_eligible=True,
    )
    return QuerySupervision(
        drug_id=drug_id,
        disease_id=disease_id,
        canonical_pair_key=f"{drug_id}{PAIR_SEPARATOR}{disease_id}",
        support=channel,
        unsuitability=channel,
        sample_source="LAZY_UNOBSERVED_DDA_CANDIDATE",
        observed_relation=None,
    )
