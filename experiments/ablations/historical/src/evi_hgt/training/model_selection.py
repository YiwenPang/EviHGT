"""Frozen Phase 8 model-selection and early-stopping protocol helpers.

This module is pure protocol logic.  It does not load a model, create an
optimizer, read test labels, or write a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from evi_hgt.training.formal_objective import PRIMARY_OBJECTIVE_ID


LR_SELECTION_PROTOCOL_ID = "EVIHGT_LR_SELECTION_V1"
EARLY_STOPPING_PROTOCOL_ID = "PRACTICAL_FORMAL_V1"
COARSE_LEARNING_RATE_CANDIDATES = (0.0001, 0.0002, 0.0003, 0.0005, 0.001)
LEARNING_RATE_CANDIDATES = (0.00002, 0.00005, 0.0001, 0.0003, 0.001)
VALIDATION_TIE_TOLERANCE = 1.0e-8
FORMAL_MAX_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.0
VALIDATION_CADENCE = "ONCE_PER_EPOCH"
SELECTION_CRITERION = "MINIMUM_DETERMINISTIC_VALIDATION_PRIMARY_OBJECTIVE"


class ModelSelectionProtocolError(ValueError):
    """Raised when selection input violates the frozen validation-only protocol."""


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_selection_protocol_payload(*, candidate_learning_rates=LEARNING_RATE_CANDIDATES) -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_id": LR_SELECTION_PROTOCOL_ID,
        "candidate_learning_rates": list(candidate_learning_rates),
        "selection_cell": {
            "model_variant": "FULL_EVIHGT",
            "policy": "B",
            "split": "standard",
            "seed": 42,
        },
        "criterion": SELECTION_CRITERION,
        "lr_level_score": "MINIMUM_OF_TWO_COMPLETE_CYCLE_VALIDATION_OBJECTIVES",
        "objective_id": PRIMARY_OBJECTIVE_ID,
        "tie_tolerance_absolute": VALIDATION_TIE_TOLERANCE,
        "tie_rule": "PREFER_SMALLER_LEARNING_RATE",
        "test_used_for_selection": False,
        "architecture_search": False,
        "formal_selection_executed": False,
    }
    return {**payload, "protocol_digest": _digest(payload)}


def early_stopping_protocol_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "protocol_id": EARLY_STOPPING_PROTOCOL_ID,
        "max_epochs": FORMAL_MAX_EPOCHS,
        "training_unit": "COMPLETE_POSITIVE_COVERAGE_EPOCH",
        "stopping_rule": "FIXED_MAXIMUM_2_COMPLETE_COVERAGE_CYCLES",
        "patience": EARLY_STOPPING_PATIENCE,
        "patience_role": "INERT_LEGACY_MECHANISM",
        "min_delta": EARLY_STOPPING_MIN_DELTA,
        "validation_cadence": VALIDATION_CADENCE,
        "criterion": SELECTION_CRITERION,
        "objective_id": PRIMARY_OBJECTIVE_ID,
        "tie_tolerance_absolute": 0.0,
        "equal_objective_tie_rule": "PREFER_EARLIEST_EPOCH",
        "best_checkpoint_role": "FORMAL_TEST_EVALUATION",
        "last_checkpoint_role": "RESUME_RECOVERY_ONLY",
        "test_metrics_permitted": False,
    }
    return {**payload, "protocol_digest": _digest(payload)}


@dataclass(frozen=True, slots=True)
class LearningRateSelectionDecision:
    candidate_validation_objectives: tuple[tuple[float, float], ...]
    selected_learning_rate: float
    tie: bool
    protocol_digest: str
    selection_rule: str = "MIN_VALIDATION_OBJECTIVE_THEN_SMALLER_LR"
    test_used_for_selection: bool = False

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_validation_objectives"] = [
            {"learning_rate": learning_rate, "validation_objective": objective}
            for learning_rate, objective in self.candidate_validation_objectives
        ]
        return payload


def select_learning_rate(
    *,
    validation_objectives: Mapping[float, float],
    test_metrics: Mapping[float, float] | None = None,
) -> LearningRateSelectionDecision:
    """Select five BEST scores; ties are measured against the global minimum."""

    if test_metrics is not None:
        raise ModelSelectionProtocolError(
            "test evidence is forbidden for learning-rate selection"
        )
    observed = {float(key): float(value) for key, value in validation_objectives.items()}
    if set(observed) != set(LEARNING_RATE_CANDIDATES):
        raise ModelSelectionProtocolError(
            "validation objectives must contain exactly the five frozen LR candidates"
        )
    if not all(math.isfinite(value) for value in observed.values()):
        raise ModelSelectionProtocolError("validation objectives must be finite")
    minimum = min(observed.values())
    tied = [lr for lr in LEARNING_RATE_CANDIDATES
            if abs(observed[lr] - minimum) <= VALIDATION_TIE_TOLERANCE]
    tie = len(tied) > 1
    selected = min(tied)
    protocol = model_selection_protocol_payload()
    return LearningRateSelectionDecision(
        candidate_validation_objectives=tuple(
            (learning_rate, observed[learning_rate])
            for learning_rate in LEARNING_RATE_CANDIDATES
        ),
        selected_learning_rate=selected,
        tie=tie,
        protocol_digest=str(protocol["protocol_digest"]),
    )


@dataclass(frozen=True, slots=True)
class CheckpointValidationRecord:
    epoch: int
    validation_objective: float
    checkpoint_digest: str

    def __post_init__(self) -> None:
        if self.epoch <= 0:
            raise ModelSelectionProtocolError("checkpoint epoch must be positive")
        if not math.isfinite(self.validation_objective):
            raise ModelSelectionProtocolError(
                "checkpoint validation objective must be finite"
            )
        if len(self.checkpoint_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.checkpoint_digest
        ):
            raise ModelSelectionProtocolError(
                "checkpoint digest must be a lowercase SHA-256"
            )


def select_best_checkpoint(
    records: Sequence[CheckpointValidationRecord],
) -> CheckpointValidationRecord:
    """Return the lowest valid validation objective, tying to earliest epoch."""

    if not records:
        raise ModelSelectionProtocolError("at least one valid checkpoint is required")
    minimum = min(record.validation_objective for record in records)
    tied = [
        record
        for record in records
        if record.validation_objective == minimum
    ]
    return min(tied, key=lambda record: record.epoch)


@dataclass(frozen=True, slots=True)
class EarlyStoppingState:
    best_checkpoint: CheckpointValidationRecord | None = None
    last_epoch: int = 0
    epochs_without_improvement: int = 0
    should_stop: bool = False


def advance_early_stopping(
    state: EarlyStoppingState,
    current: CheckpointValidationRecord,
) -> EarlyStoppingState:
    """Advance once after one epoch-level deterministic validation call."""

    if state.should_stop:
        raise ModelSelectionProtocolError("early stopping state is already terminal")
    if current.epoch != state.last_epoch + 1:
        raise ModelSelectionProtocolError(
            "validation must occur exactly once per consecutive epoch"
        )
    if state.best_checkpoint is None:
        best = current
        stale = 0
    elif (
        current.validation_objective
        < state.best_checkpoint.validation_objective
        - EARLY_STOPPING_MIN_DELTA
    ):
        best = current
        stale = 0
    else:
        best = state.best_checkpoint
        stale = state.epochs_without_improvement + 1
    return EarlyStoppingState(
        best_checkpoint=best,
        last_epoch=current.epoch,
        epochs_without_improvement=stale,
        should_stop=(stale >= EARLY_STOPPING_PATIENCE or current.epoch >= FORMAL_MAX_EPOCHS),
    )


def checkpoint_for_formal_test(
    *,
    best_checkpoint: CheckpointValidationRecord,
    last_checkpoint: CheckpointValidationRecord,
) -> CheckpointValidationRecord:
    """Freeze formal test evaluation to best; last is only a resume artifact."""

    del last_checkpoint
    return best_checkpoint


__all__ = [
    "EARLY_STOPPING_MIN_DELTA",
    "EARLY_STOPPING_PATIENCE",
    "EARLY_STOPPING_PROTOCOL_ID",
    "FORMAL_MAX_EPOCHS",
    "LEARNING_RATE_CANDIDATES",
    "LR_SELECTION_PROTOCOL_ID",
    "VALIDATION_CADENCE",
    "VALIDATION_TIE_TOLERANCE",
    "CheckpointValidationRecord",
    "EarlyStoppingState",
    "LearningRateSelectionDecision",
    "ModelSelectionProtocolError",
    "advance_early_stopping",
    "checkpoint_for_formal_test",
    "early_stopping_protocol_payload",
    "model_selection_protocol_payload",
    "select_best_checkpoint",
    "select_learning_rate",
]
