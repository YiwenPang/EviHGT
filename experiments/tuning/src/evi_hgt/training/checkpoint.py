"""Deterministic Phase 5 checkpoint and RNG-state round trip."""

from __future__ import annotations

import random
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import hashlib
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


CHECKPOINT_FORMAT = "evihgt-training-checkpoint-v2"
LEGACY_CHECKPOINT_FORMAT = "evihgt-phase5-checkpoint-v1"
FULL_PROTOCOL_RESUME = "FULL_PROTOCOL_RESUME"
WEIGHTS_ONLY_PROTOCOL_TRANSITION = "WEIGHTS_ONLY_PROTOCOL_TRANSITION"


class CheckpointProtocolMismatchError(RuntimeError):
    """Raised before state mutation when a formal resume protocol mismatches."""


@dataclass(frozen=True, slots=True)
class CheckpointLoadResult:
    engine_state: dict[str, int]
    metadata: dict[str, Any]
    transition: str = FULL_PROTOCOL_RESUME


_REQUIRED_CHECKPOINT_KEYS = {
    "format",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "engine_state",
    "pending_gradients",
    "rng_state",
    "metadata",
}


def checkpoint_payload_digest(payload: Mapping[str, Any]) -> str:
    """Hash semantic state, including tensor bytes, independently of torch ZIP layout."""
    result = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            result.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            result.update(str((value.dtype.str, value.shape)).encode())
            result.update(value.tobytes())
        elif isinstance(value, Mapping):
            result.update(b'{')
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
            result.update(b'}')
        elif isinstance(value, (tuple, list)):
            result.update(b'[')
            for item in value:
                visit(item)
            result.update(b']')
        else:
            encoded = repr(value).encode('utf-8')
            result.update(str(len(encoded)).encode() + b':' + encoded)
    visit({k:v for k,v in payload.items() if k != 'payload_digest'})
    return result.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    """Serialize beside the target, then replace it without exposing a partial file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        torch.save(dict(payload), temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        os.replace(temporary_path, destination)
    except BaseException:
        temporary.close()
        temporary_path.unlink(missing_ok=True)
        raise


def read_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    """Deserialize and structurally validate a checkpoint without mutating runtime."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("EviHGT checkpoint payload must be a mapping")
    if payload.get("format") not in {CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT}:
        raise ValueError("unsupported EviHGT checkpoint format")
    missing = _REQUIRED_CHECKPOINT_KEYS - set(payload)
    if missing:
        raise ValueError(f"EviHGT checkpoint fields missing: {sorted(missing)}")
    for name in (
        "model_state_dict",
        "optimizer_state_dict",
        "engine_state",
        "pending_gradients",
        "rng_state",
        "metadata",
    ):
        if not isinstance(payload[name], Mapping):
            raise ValueError(f"EviHGT checkpoint field must be a mapping: {name}")
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if missing_rng := required_rng - set(payload["rng_state"]):
        raise ValueError(f"EviHGT checkpoint RNG fields missing: {sorted(missing_rng)}")
    if 'payload_digest' in payload and checkpoint_payload_digest(payload) != payload['payload_digest']:
        raise ValueError('checkpoint binary/state integrity digest mismatch')
    if payload.get('formal_checkpoint_state') is not None and 'payload_digest' not in payload:
        raise ValueError('recovery checkpoint state integrity digest is missing')
    return payload


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    engine_state: Mapping[str, int],
    runtime_config: Mapping[str, Any],
    artifact_binding: Mapping[str, Any],
    split_kind: str,
    loss_mode: str,
    class_priors: Mapping[str, float],
    prior_provenance: str,
    candidate_universe_version: str,
    formal_protocol_binding: Mapping[str, Any] | None = None,
    candidate_universe_digest: str | None = None,
    phase6_path_config_digest: str | None = None,
    formal_training_authorization_state: str | None = None,
    formal_checkpoint_state: Mapping[str, Any] | None = None,
    sampler_state: Mapping[str, Any] | None = None,
    dataloader_generator_state: torch.Tensor | None = None,
    amp_scaler_state: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    metadata: dict[str, Any] = {
        "runtime_config": dict(runtime_config),
        "artifact_binding": dict(artifact_binding),
        "split_kind": split_kind,
        "loss_mode": loss_mode,
        "class_priors": dict(class_priors),
        "prior_provenance": prior_provenance,
        "candidate_universe_version": candidate_universe_version,
    }
    if formal_protocol_binding is not None:
        protocol = dict(formal_protocol_binding)
        required = {
            "formal_learning_protocol_id",
            "formal_learning_protocol_digest",
            "primary_objective",
            "requires_class_prior",
            "support_policy",
            "temperature",
            "lambda_support",
            "lambda_unsuitability",
            "hard_negative_mining",
        }
        if missing := required - set(protocol):
            raise ValueError(
                f"formal checkpoint protocol fields missing: {sorted(missing)}"
            )
        if protocol["requires_class_prior"] is not False:
            raise ValueError("formal primary checkpoint must be class-prior-free")
        if protocol["hard_negative_mining"] is not False:
            raise ValueError("formal primary checkpoint forbids hard-negative mining")
        if str(protocol.get("split_kind")) != split_kind:
            raise ValueError("checkpoint split does not match protocol binding")
        if not candidate_universe_digest:
            raise ValueError("formal checkpoint requires a candidate universe digest")
        if not phase6_path_config_digest:
            raise ValueError("formal checkpoint requires a Phase 6 path config digest")
        if not formal_training_authorization_state:
            raise ValueError("formal checkpoint requires authorization state")
        metadata.update(protocol)
        metadata.update(
            {
                "candidate_universe_digest": candidate_universe_digest,
                "phase6_path_config_digest": phase6_path_config_digest,
                "formal_training_authorization_state": (
                    formal_training_authorization_state
                ),
            }
        )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "engine_state": dict(engine_state),
        "pending_gradients": gradients,
        "rng_state": capture_rng_state(),
        "metadata": metadata,
        "formal_checkpoint_state": (
            dict(formal_checkpoint_state)
            if formal_checkpoint_state is not None
            else None
        ),
        "sampler_state": dict(sampler_state) if sampler_state is not None else None,
        "dataloader_generator_state": (
            dataloader_generator_state.detach().cpu().clone()
            if dataloader_generator_state is not None
            else None
        ),
        "amp_scaler_state_dict": (
            dict(amp_scaler_state) if amp_scaler_state is not None else None
        ),
    }
    if formal_checkpoint_state is not None:
        payload['payload_digest'] = checkpoint_payload_digest(payload)
    _atomic_torch_save(payload, destination)
    return destination.resolve()


def restore_training_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
) -> CheckpointLoadResult:
    """Apply an already validated full-protocol payload to the supplied runtime."""

    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler_state is not None and scheduler is None:
        raise ValueError("checkpoint contains a scheduler but none was supplied")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    for parameter in model.parameters():
        parameter.grad = None
    named_parameters = dict(model.named_parameters())
    for name, gradient in payload.get("pending_gradients", {}).items():
        if name not in named_parameters:
            raise ValueError(f"checkpoint gradient has no model parameter: {name}")
        parameter = named_parameters[name]
        parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)
    restore_rng_state(payload["rng_state"])
    return CheckpointLoadResult(
        engine_state={
            name: int(value) for name, value in payload["engine_state"].items()
        },
        metadata=dict(payload["metadata"]),
        transition=FULL_PROTOCOL_RESUME,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    map_location: str | torch.device,
    expected_protocol_id: str | None = None,
    expected_protocol_digest: str | None = None,
    allow_weights_only_protocol_transition: bool = False,
) -> CheckpointLoadResult:
    payload = read_training_checkpoint(path, map_location=map_location)
    metadata = dict(payload["metadata"])
    mismatches: list[str] = []
    if expected_protocol_id is not None:
        observed_id = metadata.get("formal_learning_protocol_id")
        if observed_id != expected_protocol_id:
            mismatches.append(
                f"protocol id: checkpoint={observed_id!r}, current={expected_protocol_id!r}"
            )
    if expected_protocol_digest is not None:
        observed_digest = metadata.get("formal_learning_protocol_digest")
        if observed_digest != expected_protocol_digest:
            mismatches.append(
                "protocol digest: "
                f"checkpoint={observed_digest!r}, current={expected_protocol_digest!r}"
            )
    if mismatches and not allow_weights_only_protocol_transition:
        raise CheckpointProtocolMismatchError("; ".join(mismatches))
    if mismatches:
        model.load_state_dict(payload["model_state_dict"])
        for parameter in model.parameters():
            parameter.grad = None
        transition_metadata = {
            **metadata,
            "checkpoint_transition": WEIGHTS_ONLY_PROTOCOL_TRANSITION,
            "protocol_mismatch": tuple(mismatches),
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "rng_state_restored": False,
            "pending_gradients_restored": False,
        }
        return CheckpointLoadResult(
            engine_state={
                "epoch": 0,
                "micro_step": 0,
                "optimizer_step": 0,
                "accumulation_position": 0,
            },
            metadata=transition_metadata,
            transition=WEIGHTS_ONLY_PROTOCOL_TRANSITION,
        )
    return restore_training_checkpoint_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
