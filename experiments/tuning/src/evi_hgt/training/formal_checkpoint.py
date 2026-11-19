"""Exact-run BEST/LAST checkpointing for formal Phase 8 training."""

from __future__ import annotations

import hashlib
import json
import math
import random
import os
import tempfile
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import numpy as np
from torch import nn
from evi_hgt.training.formal_recovery import FORMAL_RECOVERY_POLICY

from evi_hgt.training.checkpoint import (
    FULL_PROTOCOL_RESUME,
    WEIGHTS_ONLY_PROTOCOL_TRANSITION,
    CheckpointProtocolMismatchError,
    read_training_checkpoint,
    restore_training_checkpoint_payload,
    save_training_checkpoint,
)
from evi_hgt.training.model_selection import (
    LEARNING_RATE_CANDIDATES,
    VALIDATION_TIE_TOLERANCE,
    CheckpointValidationRecord,
    EarlyStoppingState,
    advance_early_stopping,
)


FORMAL_CHECKPOINT_FORMAT = "evihgt-formal-checkpoint-v2"
LEGACY_FORMAL_CHECKPOINT_FORMAT = "evihgt-formal-checkpoint-v1"
FORMAL_IDENTITY_FORMAT = "evihgt-formal-checkpoint-identity-v1"
START_LEGITIMATE_INITIAL_RUN = "START_LEGITIMATE_INITIAL_RUN"
RESUME_ALLOWED = "RESUME_ALLOWED"

_IDENTITY_FIELDS = (
    "formal_run_id",
    "model_variant",
    "policy",
    "split",
    "seed",
    "learning_rate",
    "objective_id",
    "data_digest",
    "scientific_config_digest",
    "architecture_config_digest",
    "learning_protocol_digest",
    "training_protocol_digest",
    "validation_manifest_digest",
    "candidate_protocol_digest",
    "evaluation_protocol_digest",
    "authorization_identity_digest",
    "execution_id",
)

_LEGACY_RESUME_IDENTITY_FIELDS = (
    "run_id",
    "model_variant",
    "policy",
    "split",
    "seed",
    "selected_learning_rate",
    "learning_protocol_id",
    "learning_protocol_digest",
    "candidate_protocol_digest",
    "evaluation_protocol_digest",
    "validation_manifest_digest",
    "data_digest",
    "architecture_config_digest",
    "training_config_digest",
    "phase8_authorization_state",
)


class FormalCheckpointResumeBlockedError(CheckpointProtocolMismatchError):
    """A LAST artifact exists but cannot be used for an exact-run resume."""


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FormalCheckpointMetadata:
    """Phase 7.3 compatibility schema retained for prior protocol artifacts."""

    run_id: str
    model_variant: str
    policy: str
    split: str
    seed: int
    selected_learning_rate: float
    learning_protocol_id: str
    learning_protocol_digest: str
    candidate_protocol_digest: str
    evaluation_protocol_digest: str
    validation_manifest_digest: str
    data_digest: str
    epoch: int
    global_optimizer_step: int
    best_validation_objective: float | None
    architecture_config_digest: str
    training_config_digest: str
    checkpoint_role: str
    phase8_authorization_state: str

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "learning_protocol_digest",
            "candidate_protocol_digest",
            "evaluation_protocol_digest",
            "validation_manifest_digest",
            "data_digest",
            "architecture_config_digest",
            "training_config_digest",
        ):
            _validate_digest(name, str(getattr(self, name)))
        if self.policy not in {"A", "B", "C"}:
            raise ValueError("checkpoint policy must be A, B, or C")
        if self.split not in {"standard", "cold_drug", "cold_disease", "double_cold"}:
            raise ValueError("checkpoint split is unsupported")
        if self.seed not in {42, 43, 44}:
            raise ValueError("checkpoint seed is outside the frozen set")
        if self.selected_learning_rate not in LEARNING_RATE_CANDIDATES:
            raise ValueError("checkpoint LR is outside the frozen candidates")
        if self.epoch <= 0 or self.global_optimizer_step < 0:
            raise ValueError("checkpoint progress is invalid")
        if self.best_validation_objective is not None and not math.isfinite(
            self.best_validation_objective
        ):
            raise ValueError("best validation objective must be finite when present")
        if self.checkpoint_role not in {"BEST", "LAST"}:
            raise ValueError("checkpoint role must be BEST or LAST")
        if not self.phase8_authorization_state:
            raise ValueError("checkpoint authorization state is required")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def validate_formal_resume_consistency(
    *,
    expected: FormalCheckpointMetadata,
    observed: FormalCheckpointMetadata,
    allow_weights_only_protocol_transition: bool = False,
) -> str:
    """Reject legacy formal identity drift before any checkpoint state mutation."""

    mismatches = [
        name
        for name in _LEGACY_RESUME_IDENTITY_FIELDS
        if getattr(expected, name) != getattr(observed, name)
    ]
    if not mismatches:
        return FULL_PROTOCOL_RESUME
    if allow_weights_only_protocol_transition:
        return WEIGHTS_ONLY_PROTOCOL_TRANSITION
    raise CheckpointProtocolMismatchError(
        "formal checkpoint resume mismatch: " + ", ".join(mismatches)
    )


@dataclass(frozen=True, slots=True)
class FormalCheckpointIdentity:
    """Immutable scientific and authorization identity for one formal run."""

    formal_run_id: str
    model_variant: str
    policy: str
    split: str
    seed: int
    learning_rate: float
    objective_id: str
    data_digest: str
    scientific_config_digest: str
    architecture_config_digest: str
    learning_protocol_digest: str
    training_protocol_digest: str
    validation_manifest_digest: str
    candidate_protocol_digest: str
    evaluation_protocol_digest: str
    authorization_identity_digest: str
    execution_id: str | None = None

    def __post_init__(self) -> None:
        if self.execution_id is not None:
            _validate_digest('execution_id', self.execution_id)
        for name in (
            "formal_run_id",
            "data_digest",
            "scientific_config_digest",
            "architecture_config_digest",
            "learning_protocol_digest",
            "training_protocol_digest",
            "validation_manifest_digest",
            "candidate_protocol_digest",
            "evaluation_protocol_digest",
            "authorization_identity_digest",
        ):
            _validate_digest(name, str(getattr(self, name)))
        if not self.model_variant:
            raise ValueError("formal checkpoint model variant is required")
        if self.policy not in {"A", "B", "C"}:
            raise ValueError("formal checkpoint policy must be A, B, or C")
        if self.split not in {"standard", "cold_drug", "cold_disease", "double_cold"}:
            raise ValueError("formal checkpoint split is unsupported")
        if self.seed not in {42, 43, 44}:
            raise ValueError("formal checkpoint seed is outside the frozen set")
        if self.learning_rate not in LEARNING_RATE_CANDIDATES:
            raise ValueError("formal checkpoint learning rate is outside the frozen set")
        if not self.objective_id:
            raise ValueError("formal checkpoint objective identity is required")

    @property
    def identity_digest(self) -> str:
        return _canonical_digest(self._identity_fields())

    def _identity_fields(self) -> dict[str, object]:
        value = asdict(self)
        if self.execution_id is None:
            value.pop('execution_id')
        return value

    def to_payload(self) -> dict[str, object]:
        return {
            "format": FORMAL_IDENTITY_FORMAT,
            **self._identity_fields(),
            "identity_digest": self.identity_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FormalCheckpointIdentity":
        if payload.get("format") != FORMAL_IDENTITY_FORMAT:
            raise ValueError("unsupported formal checkpoint identity format")
        try:
            identity = cls(**{name: payload.get(name) if name == 'execution_id' else payload[name] for name in _IDENTITY_FIELDS})
        except (KeyError, TypeError) as exc:
            raise ValueError(f"formal checkpoint identity is incomplete: {exc}") from exc
        if payload.get("identity_digest") != identity.identity_digest:
            raise ValueError("formal checkpoint identity digest mismatch")
        return identity


def formal_checkpoint_identity_from_execution(execution: Any) -> FormalCheckpointIdentity:
    """Build the checkpoint identity from PATCH.1's already-audited execution."""

    run = execution.run
    authorization = execution.authorization
    if execution.run_id not in authorization["authorized_run_ids"]:
        raise ValueError("formal run is absent from the current exact authorization")
    validation_binding = {
        channel: {
            "manifest_digest": value["manifest_digest"],
            "source_artifact_sha256": value["source_artifact_sha256"],
        }
        for channel, value in sorted(execution.validation_manifests.items())
    }
    authorization_binding = {
        "formal_run_id": execution.run_id,
        "phase8_execution_authorized": authorization[
            "phase8_execution_authorized"
        ],
        "phase_state": authorization["phase_state"],
        "authorization_source": authorization["authorization_source"],
        "authorized_run_ids": authorization["authorized_run_ids"],
        "cli_override_permitted": authorization["cli_override_permitted"],
    }
    return FormalCheckpointIdentity(
        formal_run_id=execution.run_id,
        model_variant=str(run["model_variant"]),
        policy=str(run["policy"]),
        split=str(run["split"]),
        seed=int(run["seed"]),
        learning_rate=float(run["learning_rate"]),
        objective_id=str(run["learning_protocol"]["objective_id"]),
        data_digest=str(run["data_digest"]),
        scientific_config_digest=str(execution.config_digest),
        architecture_config_digest=str(run["architecture_digest"]),
        learning_protocol_digest=str(run["learning_protocol_digest"]),
        training_protocol_digest=_canonical_digest(execution.training_protocol),
        validation_manifest_digest=_canonical_digest(validation_binding),
        candidate_protocol_digest=str(run["candidate_protocol_digest"]),
        evaluation_protocol_digest=str(run["evaluation_protocol_digest"]),
        authorization_identity_digest=_canonical_digest(authorization_binding),
        execution_id=(run['execution_attempts'][-1]['execution_id'] if run.get('execution_attempts') else None),
    )


def validate_formal_checkpoint_identity(
    *, expected: FormalCheckpointIdentity, observed: FormalCheckpointIdentity
) -> None:
    mismatches = [
        name for name in _IDENTITY_FIELDS if getattr(expected, name) != getattr(observed, name)
    ]
    if mismatches:
        raise CheckpointProtocolMismatchError(
            "formal checkpoint resume mismatch: " + ", ".join(mismatches)
        )


def _record_to_payload(record: CheckpointValidationRecord | None) -> object:
    return None if record is None else asdict(record)


def _record_from_payload(payload: object) -> CheckpointValidationRecord | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("formal early-stopping best checkpoint must be a mapping")
    try:
        return CheckpointValidationRecord(
            epoch=int(payload["epoch"]),
            validation_objective=float(payload["validation_objective"]),
            checkpoint_digest=str(payload["checkpoint_digest"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"formal early-stopping best checkpoint is invalid: {exc}") from exc


def _early_stopping_to_payload(state: EarlyStoppingState) -> dict[str, object]:
    return {
        "best_checkpoint": _record_to_payload(state.best_checkpoint),
        "last_epoch": state.last_epoch,
        "epochs_without_improvement": state.epochs_without_improvement,
        "should_stop": state.should_stop,
    }


def _early_stopping_from_payload(payload: object) -> EarlyStoppingState:
    if not isinstance(payload, Mapping):
        raise ValueError("formal early-stopping state must be a mapping")
    try:
        return EarlyStoppingState(
            best_checkpoint=_record_from_payload(payload["best_checkpoint"]),
            last_epoch=int(payload["last_epoch"]),
            epochs_without_improvement=int(payload["epochs_without_improvement"]),
            should_stop=bool(payload["should_stop"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"formal early-stopping state is incomplete: {exc}") from exc


@dataclass(frozen=True, slots=True)
class FormalCheckpointSnapshot:
    identity: FormalCheckpointIdentity
    checkpoint_role: str
    validation_objective: float | None
    last_completed_epoch: int
    next_epoch: int
    global_optimizer_step: int
    engine_state: dict[str, int]
    early_stopping_state: EarlyStoppingState
    logical_checkpoint_digest: str
    amp_scaler_used: bool
    sampler_state_saved: bool
    dataloader_generator_state_saved: bool
    best_artifact: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_role not in {"BEST", "LAST"}:
            raise ValueError("formal checkpoint role must be BEST or LAST")
        if self.identity.execution_id is not None:
            if not self.sampler_state_saved or not self.dataloader_generator_state_saved or self.amp_scaler_used:
                raise ValueError('formal execution requires sampler/generator state and no GradScaler')
            if self.checkpoint_role == 'LAST' and self.best_artifact is None:
                raise ValueError('formal execution LAST requires committed BEST linkage')
        if not math.isfinite(self.validation_objective):
            raise ValueError("formal validation objective must be finite")
        if self.last_completed_epoch <= 0:
            raise ValueError("formal checkpoint epoch must be positive")
        if self.next_epoch != self.last_completed_epoch + 1:
            raise ValueError("formal checkpoint next epoch is inconsistent")
        if self.global_optimizer_step < 0:
            raise ValueError("formal global optimizer step must be non-negative")
        required_engine = {
            "epoch",
            "micro_step",
            "optimizer_step",
            "accumulation_position",
        }
        if set(self.engine_state) != required_engine:
            raise ValueError("formal checkpoint engine state is incomplete")
        if self.engine_state["epoch"] != self.last_completed_epoch:
            raise ValueError("formal checkpoint epoch differs from engine state")
        if self.engine_state["optimizer_step"] != self.global_optimizer_step:
            raise ValueError("formal checkpoint optimizer step differs from engine state")
        if self.engine_state["accumulation_position"] != 0:
            raise ValueError("formal epoch checkpoint requires complete accumulation")
        if self.early_stopping_state.last_epoch != self.last_completed_epoch:
            raise ValueError("formal early-stopping epoch differs from checkpoint")
        if self.early_stopping_state.best_checkpoint is None:
            raise ValueError("formal epoch checkpoint requires a BEST record")
        if self.early_stopping_state.best_checkpoint.epoch > self.last_completed_epoch:
            raise ValueError("formal BEST epoch cannot be after LAST progress")
        if self.early_stopping_state.epochs_without_improvement < 0:
            raise ValueError("formal early-stopping patience state is invalid")
        best = self.early_stopping_state.best_checkpoint
        if (
            self.validation_objective
            < best.validation_objective - VALIDATION_TIE_TOLERANCE
        ):
            raise ValueError("formal early-stopping BEST is stale")
        _validate_digest("logical_checkpoint_digest", self.logical_checkpoint_digest)
        expected_logical_digest = _canonical_digest(
            {
                "formal_identity_digest": self.identity.identity_digest,
                "epoch": self.last_completed_epoch,
                "validation_objective": self.validation_objective,
            }
        )
        if self.logical_checkpoint_digest != expected_logical_digest:
            raise ValueError("formal logical checkpoint digest mismatch")
        if self.checkpoint_role == "BEST" and (
            best.epoch != self.last_completed_epoch
            or abs(best.validation_objective - self.validation_objective)
            > VALIDATION_TIE_TOLERANCE
            or best.checkpoint_digest != self.logical_checkpoint_digest
        ):
            raise ValueError("BEST checkpoint state does not describe its selected epoch")

    @property
    def best_epoch(self) -> int:
        assert self.early_stopping_state.best_checkpoint is not None
        return self.early_stopping_state.best_checkpoint.epoch

    @property
    def best_validation_objective(self) -> float:
        assert self.early_stopping_state.best_checkpoint is not None
        return self.early_stopping_state.best_checkpoint.validation_objective

    def to_payload(self) -> dict[str, object]:
        return {
            "format": FORMAL_CHECKPOINT_FORMAT,
            "identity": self.identity.to_payload(),
            "checkpoint_role": self.checkpoint_role,
            "validation_objective": self.validation_objective,
            "last_completed_epoch": self.last_completed_epoch,
            "next_epoch": self.next_epoch,
            "global_optimizer_step": self.global_optimizer_step,
            "engine_state": dict(self.engine_state),
            "early_stopping_state": _early_stopping_to_payload(
                self.early_stopping_state
            ),
            "logical_checkpoint_digest": self.logical_checkpoint_digest,
            "amp_scaler_used": self.amp_scaler_used,
            "sampler_state_saved": self.sampler_state_saved,
            "dataloader_generator_state_saved": (
                self.dataloader_generator_state_saved
            ),
            "recovery_policy": FORMAL_RECOVERY_POLICY,
            "best_artifact": self.best_artifact,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FormalCheckpointSnapshot":
        if payload.get("format") not in {FORMAL_CHECKPOINT_FORMAT, LEGACY_FORMAL_CHECKPOINT_FORMAT}:
            raise ValueError("unsupported formal checkpoint state format")
        if payload.get("recovery_state") is not None:
            raise ValueError("mid-epoch durable recovery is retired")
        if payload.get("format") == FORMAL_CHECKPOINT_FORMAT and payload.get("recovery_policy") != FORMAL_RECOVERY_POLICY:
            raise ValueError("formal recovery policy is missing or incompatible")
        try:
            engine_state = {
                name: int(value)
                for name, value in dict(payload["engine_state"]).items()
            }
            return cls(
                identity=FormalCheckpointIdentity.from_payload(payload["identity"]),
                checkpoint_role=str(payload["checkpoint_role"]),
                validation_objective=(None if payload["validation_objective"] is None else float(payload["validation_objective"])),
                last_completed_epoch=int(payload["last_completed_epoch"]),
                next_epoch=int(payload["next_epoch"]),
                global_optimizer_step=int(payload["global_optimizer_step"]),
                engine_state=engine_state,
                early_stopping_state=_early_stopping_from_payload(
                    payload["early_stopping_state"]
                ),
                logical_checkpoint_digest=str(payload["logical_checkpoint_digest"]),
                amp_scaler_used=bool(payload["amp_scaler_used"]),
                sampler_state_saved=bool(payload["sampler_state_saved"]),
                dataloader_generator_state_saved=bool(
                    payload["dataloader_generator_state_saved"]
                ),
                best_artifact=payload.get('best_artifact'),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"formal checkpoint state is incomplete: {exc}") from exc


@dataclass(frozen=True, slots=True)
class FormalCheckpointPaths:
    best: Path
    last: Path

    @classmethod
    def from_directory(cls, directory: str | Path) -> "FormalCheckpointPaths":
        root = Path(directory)
        return cls(best=root / "best.pt", last=root / "last.pt")


@dataclass(frozen=True, slots=True)
class FormalCheckpointInspection:
    path: Path
    artifact_sha256: str
    snapshot: FormalCheckpointSnapshot
    payload: Mapping[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class FormalCheckpointDecision:
    action: str
    paths: FormalCheckpointPaths
    identity: FormalCheckpointIdentity
    inspection: FormalCheckpointInspection | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def snapshot(self) -> FormalCheckpointSnapshot | None:
        return None if self.inspection is None else self.inspection.snapshot


def _logical_checkpoint_digest(
    identity: FormalCheckpointIdentity, epoch: int, validation_objective: float
) -> str:
    return _canonical_digest(
        {
            "formal_identity_digest": identity.identity_digest,
            "epoch": epoch,
            "validation_objective": validation_objective,
        }
    )


def save_formal_checkpoint(
    path: str | Path,
    *,
    identity: FormalCheckpointIdentity,
    checkpoint_role: str,
    validation_objective: float | None,
    early_stopping_state: EarlyStoppingState,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    engine_state: Mapping[str, int],
    runtime_config: Mapping[str, Any],
    candidate_universe_version: str,
    sampler: Any | None,
    dataloader_generator: torch.Generator | None,
    amp_scaler: Any | None,
    best_artifact: dict[str, str] | None = None,
) -> FormalCheckpointSnapshot:
    """Persist one self-contained formal checkpoint using the shared atomic writer."""

    normalized_engine = {name: int(value) for name, value in engine_state.items()}
    epoch = normalized_engine.get("epoch", 0)
    optimizer_step = normalized_engine.get("optimizer_step", -1)
    sampler_state = sampler.state_dict() if sampler is not None else None
    generator_state = (
        dataloader_generator.get_state()
        if dataloader_generator is not None
        else None
    )
    scaler_state = amp_scaler.state_dict() if amp_scaler is not None else None
    snapshot = FormalCheckpointSnapshot(
        identity=identity,
        checkpoint_role=checkpoint_role,
        validation_objective=(None if validation_objective is None else float(validation_objective)),
        last_completed_epoch=epoch,
        next_epoch=epoch + 1,
        global_optimizer_step=optimizer_step,
        engine_state=normalized_engine,
        early_stopping_state=early_stopping_state,
        logical_checkpoint_digest=_logical_checkpoint_digest(
            identity, epoch, float(validation_objective)
        ),
        amp_scaler_used=amp_scaler is not None,
        sampler_state_saved=sampler_state is not None,
        dataloader_generator_state_saved=generator_state is not None,
        best_artifact=best_artifact,
    )
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        engine_state=normalized_engine,
        runtime_config=runtime_config,
        artifact_binding={"formal_checkpoint_identity": identity.to_payload()},
        split_kind=identity.split,
        loss_mode=identity.objective_id,
        class_priors={},
        prior_provenance="NOT_REQUIRED_BY_FORMAL_PRIMARY",
        candidate_universe_version=candidate_universe_version,
        formal_checkpoint_state=snapshot.to_payload(),
        sampler_state=sampler_state,
        dataloader_generator_state=generator_state,
        amp_scaler_state=scaler_state,
    )
    return snapshot


def inspect_formal_checkpoint_file(
    path: str | Path,
    *,
    expected_identity: FormalCheckpointIdentity,
    expected_role: str,
    map_location: str | torch.device,
) -> FormalCheckpointInspection:
    """Read and validate every identity/state layer without runtime mutation."""

    checkpoint_path = Path(path)
    payload = read_training_checkpoint(checkpoint_path, map_location=map_location)
    rng = payload['rng_state']
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator().set_state(rng['torch_cpu'].cpu())
    if not isinstance(rng['torch_cuda'], list) or any(not isinstance(x, torch.Tensor) for x in rng['torch_cuda']):
        raise ValueError('formal checkpoint CUDA RNG is invalid')
    for index, state in enumerate(rng['torch_cuda']):
        if state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError('formal checkpoint CUDA RNG dtype/shape is invalid')
        if torch.cuda.is_available():
            if index >= torch.cuda.device_count():
                raise ValueError('formal checkpoint CUDA RNG device count differs')
            torch.Generator(device=f'cuda:{index}').set_state(state.cpu())
    if str(payload['metadata']['runtime_config'].get('device', '')).startswith('cuda') and not rng['torch_cuda']:
        raise ValueError('formal CUDA checkpoint RNG is missing')
    formal_state = payload.get("formal_checkpoint_state")
    if not isinstance(formal_state, Mapping):
        raise ValueError("formal checkpoint state is missing")
    snapshot = FormalCheckpointSnapshot.from_payload(formal_state)
    validate_formal_checkpoint_identity(
        expected=expected_identity, observed=snapshot.identity
    )
    if snapshot.checkpoint_role != expected_role:
        raise CheckpointProtocolMismatchError(
            "formal checkpoint resume mismatch: checkpoint_role"
        )
    if dict(payload["engine_state"]) != snapshot.engine_state:
        raise ValueError("formal checkpoint binary and metadata engine states differ")
    binding = payload["metadata"].get("artifact_binding", {}).get(
        "formal_checkpoint_identity"
    )
    if binding != snapshot.identity.to_payload():
        raise ValueError("formal checkpoint binary and metadata identities differ")
    if payload["metadata"].get("split_kind") != snapshot.identity.split:
        raise ValueError("formal checkpoint binary and identity splits differ")
    if payload["metadata"].get("loss_mode") != snapshot.identity.objective_id:
        raise ValueError("formal checkpoint binary and identity objectives differ")
    groups = payload['optimizer_state_dict'].get('param_groups')
    if not isinstance(groups, list) or not groups or any(
        float(group.get('lr', float('nan'))) != snapshot.identity.learning_rate
        for group in groups
    ):
        raise ValueError('formal checkpoint optimizer learning rate differs from identity')
    if payload['scheduler_state_dict'] is not None or payload['pending_gradients']:
        raise ValueError('formal epoch boundary forbids scheduler/pending gradients')
    if (payload.get("amp_scaler_state_dict") is not None) != snapshot.amp_scaler_used:
        raise ValueError("formal checkpoint AMP state is inconsistent")
    if (payload.get("sampler_state") is not None) != snapshot.sampler_state_saved:
        raise ValueError("formal checkpoint sampler state is inconsistent")
    if (
        payload.get("dataloader_generator_state") is not None
    ) != snapshot.dataloader_generator_state_saved:
        raise ValueError("formal checkpoint DataLoader generator state is inconsistent")
    sampler_state = payload.get("sampler_state")
    if snapshot.sampler_state_saved:
        if not isinstance(sampler_state, Mapping):
            raise ValueError("formal checkpoint sampler state must be a mapping")
        if sampler_state.get("format") != "evihgt-lazy-unlabeled-sampler-v1":
            raise ValueError("formal checkpoint sampler format is unsupported")
        if int(sampler_state.get("seed", -1)) != snapshot.identity.seed:
            raise ValueError("formal checkpoint sampler seed differs from identity")
        _validate_digest(
            "sampler_universe_digest", str(sampler_state.get("universe_digest", ""))
        )
        sampler_verifier = random.Random()
        sampler_verifier.setstate(sampler_state.get("rng_state"))
    generator_state = payload.get("dataloader_generator_state")
    if snapshot.dataloader_generator_state_saved:
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("formal DataLoader generator state must be a tensor")
        generator_verifier = torch.Generator()
        generator_verifier.set_state(generator_state.cpu())
    scaler_state = payload.get("amp_scaler_state_dict")
    if snapshot.amp_scaler_used and not isinstance(scaler_state, Mapping):
        raise ValueError("formal AMP scaler state must be a mapping")
    if snapshot.best_artifact is not None:
        _validate_best_artifact(checkpoint_path.parent, snapshot)
    if expected_role == "BEST":
        committed = inspect_formal_checkpoint_file(
            checkpoint_path.parent / 'last.pt', expected_identity=expected_identity,
            expected_role='LAST', map_location='cpu').snapshot
        selected = committed.early_stopping_state.best_checkpoint
        if selected is None or selected.checkpoint_digest != snapshot.logical_checkpoint_digest:
            raise ValueError('BEST is not authoritative in committed LAST')
        if committed.best_artifact is not None and _file_sha256(checkpoint_path) != committed.best_artifact['sha256']:
            raise ValueError('BEST binary differs from committed LAST linkage')
    return FormalCheckpointInspection(
        path=checkpoint_path.resolve(),
        artifact_sha256=_file_sha256(checkpoint_path),
        snapshot=snapshot,
        payload=payload,
    )


def _validate_best_artifact(directory: Path, snapshot: FormalCheckpointSnapshot) -> Path:
    binding = snapshot.best_artifact
    name = binding['filename']
    if Path(name).name != name or not name.startswith('best.') or not name.endswith('.pt'):
        raise ValueError('formal immutable BEST path is invalid')
    path = directory / name
    if not path.is_file() or _file_sha256(path) != binding['sha256']:
        raise ValueError('formal immutable BEST binary digest mismatch')
    payload = read_training_checkpoint(path, map_location='cpu')
    selected = FormalCheckpointSnapshot.from_payload(payload['formal_checkpoint_state'])
    validate_formal_checkpoint_identity(expected=snapshot.identity, observed=selected.identity)
    best = snapshot.early_stopping_state.best_checkpoint
    if selected.checkpoint_role != 'BEST' or best is None or selected.logical_checkpoint_digest != best.checkpoint_digest:
        raise ValueError('formal immutable BEST selection identity mismatch')
    return path


def _atomic_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _save_immutable_best(path: Path, **kwargs) -> FormalCheckpointSnapshot:
    """Idempotent publication after a failed LAST write never rewrites a version."""
    if not path.exists():
        return save_formal_checkpoint(path, **kwargs)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
        candidate = Path(handle.name)
    try:
        snapshot = save_formal_checkpoint(candidate, **kwargs)
        previous = read_training_checkpoint(path, map_location='cpu')
        replay = read_training_checkpoint(candidate, map_location='cpu')
        if previous.get('payload_digest') != replay.get('payload_digest'):
            raise ValueError('formal immutable BEST replay state differs')
        return snapshot
    finally:
        candidate.unlink(missing_ok=True)


def inspect_formal_checkpoint_state(
    *,
    identity: FormalCheckpointIdentity,
    run_status: str,
    checkpoint_directory: str | Path,
    map_location: str | torch.device,
) -> FormalCheckpointDecision:
    """Choose legitimate initial start or exact LAST resume; never fall back."""

    paths = FormalCheckpointPaths.from_directory(checkpoint_directory)
    if run_status not in {'PLANNED', 'REARMED', 'INTERRUPTED', 'RUNNING'}:
        raise FormalCheckpointResumeBlockedError('RESUME_BLOCKED: failed/completed trajectory cannot resume')
    best_present = paths.best.exists() or any(paths.best.parent.glob('best.*.pt'))
    if run_status == 'REARMED' and (paths.last.exists() or best_present):
        raise FormalCheckpointResumeBlockedError('RESUME_BLOCKED: fresh execution directory contains learned state')
    if not paths.last.exists():
        if best_present:
            raise FormalCheckpointResumeBlockedError(
                'RESUME_BLOCKED: uncommitted BEST without LAST requires a new execution attempt')
        if run_status not in {"PLANNED", 'REARMED'}:
            raise FormalCheckpointResumeBlockedError(
                "RESUME_BLOCKED: non-PLANNED run has no LAST checkpoint"
            )
        return FormalCheckpointDecision(
            action=START_LEGITIMATE_INITIAL_RUN,
            paths=paths,
            identity=identity,
        )
    try:
        inspection = inspect_formal_checkpoint_file(
            paths.last,
            expected_identity=identity,
            expected_role="LAST",
            map_location=map_location,
        )
    except Exception as exc:
        raise FormalCheckpointResumeBlockedError(
            f"RESUME_BLOCKED: unusable LAST checkpoint: {exc}"
        ) from exc
    return FormalCheckpointDecision(
        action=RESUME_ALLOWED,
        paths=paths,
        identity=identity,
        inspection=inspection,
    )


def restore_formal_checkpoint(
    inspection: FormalCheckpointInspection,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer: Any,
    sampler: Any | None,
    dataloader_generator: torch.Generator | None,
    amp_scaler: Any | None,
) -> FormalCheckpointSnapshot:
    """Restore a preflight-validated LAST, including trajectory-specific state."""

    try:
        return _restore_formal_checkpoint(inspection, model=model, optimizer=optimizer,
            trainer=trainer, sampler=sampler, dataloader_generator=dataloader_generator,
            amp_scaler=amp_scaler)
    except FormalCheckpointResumeBlockedError:
        raise
    except Exception as exc:
        raise FormalCheckpointResumeBlockedError(f'RESUME_BLOCKED: incompatible runtime state: {exc}') from exc


def _validate_runtime_tensor_state(payload, model, optimizer) -> None:
    """Reject incompatible model/AdamW state before loading any runtime tensors."""
    expected = model.state_dict()
    actual = payload['model_state_dict']
    if expected.keys() != actual.keys():
        raise ValueError('formal checkpoint model parameter keys differ')
    for name, target in expected.items():
        value = actual[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f'formal checkpoint model tensor differs: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'formal checkpoint model tensor is nonfinite: {name}')
    saved = payload['optimizer_state_dict']
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('formal checkpoint requires AdamW')
    current_groups = optimizer.param_groups
    saved_groups = saved['param_groups']
    if len(current_groups) != len(saved_groups):
        raise ValueError('formal checkpoint optimizer group count differs')
    parameters = {}
    for current, group in zip(current_groups, saved_groups, strict=True):
        if {k: v for k, v in current.items() if k != 'params'} != {k: v for k, v in group.items() if k != 'params'}:
            raise ValueError('formal checkpoint optimizer configuration differs')
        if len(current['params']) != len(group['params']):
            raise ValueError('formal checkpoint optimizer parameter count differs')
        for key, parameter in zip(group['params'], current['params'], strict=True):
            if key in parameters:
                raise ValueError('formal checkpoint optimizer parameter is duplicated')
            parameters[key] = parameter
    states = saved.get('state')
    if not isinstance(states, Mapping) or (payload['engine_state']['optimizer_step'] > 0 and not states):
        raise ValueError('formal checkpoint optimizer state is missing')
    for key, state in states.items():
        if key not in parameters or not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys():
            raise ValueError('formal checkpoint AdamW state is incomplete')
        step = state['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all() or not 0 <= float(step) <= payload['engine_state']['optimizer_step']:
            raise ValueError('formal checkpoint AdamW step is invalid')
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name not in state:
                continue
            value = state[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameters[key].shape or value.dtype != parameters[key].dtype or not torch.isfinite(value).all():
                raise ValueError(f'formal checkpoint AdamW buffer differs: {name}')


def _restore_formal_checkpoint(inspection, *, model, optimizer, trainer, sampler,
                               dataloader_generator, amp_scaler):

    snapshot = inspection.snapshot
    if snapshot.checkpoint_role != "LAST":
        raise FormalCheckpointResumeBlockedError(
            "RESUME_BLOCKED: only LAST may restore training progress"
        )
    payload = inspection.payload
    if getattr(trainer, "model", None) is not model or getattr(
        trainer, "optimizer", None
    ) is not optimizer:
        raise FormalCheckpointResumeBlockedError(
            "RESUME_BLOCKED: trainer/model/optimizer binding differs"
        )
    if any(
        float(group.get("lr", float("nan"))) != snapshot.identity.learning_rate
        for group in optimizer.param_groups
    ):
        raise FormalCheckpointResumeBlockedError(
            "RESUME_BLOCKED: runtime optimizer learning rate differs"
        )
    if snapshot.amp_scaler_used != (amp_scaler is not None):
        raise FormalCheckpointResumeBlockedError(
            "RESUME_BLOCKED: AMP scaler applicability differs"
        )
    if snapshot.sampler_state_saved != (sampler is not None):
        raise FormalCheckpointResumeBlockedError(
            "RESUME_BLOCKED: sampler state applicability differs"
        )
    if snapshot.dataloader_generator_state_saved != (
        dataloader_generator is not None
    ):
        raise FormalCheckpointResumeBlockedError(
            "RESUME_BLOCKED: DataLoader generator applicability differs"
        )
    if sampler is not None and hasattr(sampler, "validate_state_dict"):
        sampler.validate_state_dict(payload["sampler_state"])
    _validate_runtime_tensor_state(payload, model, optimizer)
    restored = restore_training_checkpoint_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=None,
    )
    trainer.load_state_dict(restored.engine_state)
    if amp_scaler is not None:
        amp_scaler.load_state_dict(payload["amp_scaler_state_dict"])
    if sampler is not None:
        sampler.load_state_dict(payload["sampler_state"])
    if dataloader_generator is not None:
        dataloader_generator.set_state(
            payload["dataloader_generator_state"].cpu()
        )
    if snapshot.best_artifact is not None:
        best_path = _validate_best_artifact(inspection.path.parent, snapshot)
        _atomic_copy(best_path, inspection.path.parent / 'best.pt')
    return snapshot


@dataclass(frozen=True, slots=True)
class FormalEpochCheckpointResult:
    best_updated: bool
    early_stopping_state: EarlyStoppingState
    best_snapshot: FormalCheckpointSnapshot | None
    last_snapshot: FormalCheckpointSnapshot


@dataclass(slots=True)
class FormalCheckpointLifecycle:
    """Small epoch-boundary adapter around the existing trainer/checkpoint logic."""

    identity: FormalCheckpointIdentity
    paths: FormalCheckpointPaths
    trainer: Any
    runtime_config: Mapping[str, Any]
    candidate_universe_version: str
    sampler: Any | None
    dataloader_generator: torch.Generator | None
    amp_scaler: Any | None
    early_stopping_state: EarlyStoppingState = field(
        default_factory=EarlyStoppingState
    )
    best_artifact: dict[str, str] | None = None
    last_snapshot: FormalCheckpointSnapshot | None = None

    def record_completed_epoch(
        self, validation_objective: float
    ) -> FormalEpochCheckpointResult:
        """Select BEST from one validation scalar and always atomically write LAST."""

        if not math.isfinite(float(validation_objective)):
            raise ValueError("formal deterministic validation objective must be finite")
        engine_state = {
            name: int(value) for name, value in self.trainer.state_dict().items()
        }
        epoch = engine_state.get("epoch", 0)
        record = CheckpointValidationRecord(
            epoch=epoch,
            validation_objective=float(validation_objective),
            checkpoint_digest=_logical_checkpoint_digest(
                self.identity, epoch, float(validation_objective)
            ),
        )
        advanced = advance_early_stopping(self.early_stopping_state, record)
        best_updated = advanced.best_checkpoint == record
        best_snapshot = None
        if best_updated:
            version_path = self.paths.best.with_name(f'best.{record.checkpoint_digest}.pt')
            best_snapshot = _save_immutable_best(
                version_path,
                identity=self.identity,
                checkpoint_role="BEST",
                validation_objective=float(validation_objective),
                early_stopping_state=advanced,
                model=self.trainer.model,
                optimizer=self.trainer.optimizer,
                engine_state=engine_state,
                runtime_config=self.runtime_config,
                candidate_universe_version=self.candidate_universe_version,
                sampler=self.sampler,
                dataloader_generator=self.dataloader_generator,
                amp_scaler=self.amp_scaler,
            )
            best_artifact = {'filename': version_path.name, 'sha256': _file_sha256(version_path)}
        else:
            best_artifact = self.best_artifact
        last_snapshot = save_formal_checkpoint(
            self.paths.last,
            identity=self.identity,
            checkpoint_role="LAST",
            validation_objective=float(validation_objective),
            early_stopping_state=advanced,
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            engine_state=engine_state,
            runtime_config=self.runtime_config,
            candidate_universe_version=self.candidate_universe_version,
            sampler=self.sampler,
            dataloader_generator=self.dataloader_generator,
            amp_scaler=self.amp_scaler,
            best_artifact=best_artifact,
        )
        self.early_stopping_state = advanced
        self.last_snapshot = last_snapshot
        self.best_artifact = best_artifact
        if best_artifact is not None:
            _atomic_copy(self.paths.best.parent / best_artifact['filename'], self.paths.best)
        return FormalEpochCheckpointResult(
            best_updated=best_updated,
            early_stopping_state=advanced,
            best_snapshot=best_snapshot,
            last_snapshot=last_snapshot,
        )


__all__ = [
    "FORMAL_CHECKPOINT_FORMAT",
    "RESUME_ALLOWED",
    "START_LEGITIMATE_INITIAL_RUN",
    "FormalCheckpointDecision",
    "FormalCheckpointIdentity",
    "FormalCheckpointInspection",
    "FormalCheckpointLifecycle",
    "FormalCheckpointMetadata",
    "FormalCheckpointPaths",
    "FormalCheckpointResumeBlockedError",
    "FormalCheckpointSnapshot",
    "FormalEpochCheckpointResult",
    "formal_checkpoint_identity_from_execution",
    "inspect_formal_checkpoint_file",
    "inspect_formal_checkpoint_state",
    "restore_formal_checkpoint",
    "save_formal_checkpoint",
    "validate_formal_checkpoint_identity",
    "validate_formal_resume_consistency",
]
