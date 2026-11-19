"""Complete training-state snapshots and atomic commits; no G3 authorization is implied.

Storage is currently restricted to isolated TEST_ONLY artifacts. The same complete
state format is usable by the approved production adapters after a separate G3 gate.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import uuid

import numpy as np
import torch

from .identity import ROOT, G1_SHA256, canonical, file_sha

TEST_ROOT = (ROOT / "outputs/phase9/phase92b/test_only").resolve()
SCHEMA = "PHASE9_COMPLETE_TRAINING_STATE_V1"
COMMIT_SCHEMA = "PHASE9_ATOMIC_CHECKPOINT_COMMIT_V1"
REQUIRED = {"schema", "identity", "cursor", "sampler", "model_metadata", "implementation_sha256",
            "model", "optimizer", "optimizer_recipe", "scheduler", "scheduler_recipe", "rng",
            "module_modes", "state_digest"}


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return copy.deepcopy(value)
    raise ValueError("Unsupported full-state value: " + type(value).__name__)


def training_state_digest(value):
    """Content integrity (not an authenticity signature), including RNG/moments/cursors."""
    def encode(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            if tensor.layout != torch.strided:
                raise ValueError("Only dense checkpoint tensors are supported")
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return ["tensor", str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]
        if isinstance(item, dict):
            return ["dict", [[type(key).__name__, str(key), encode(v)] for key, v in sorted(item.items(), key=lambda x: (type(x[0]).__name__, str(x[0])))]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [encode(v) for v in item]]
        if item is None or type(item) in (str, int, float, bool):
            return item
        raise ValueError("Unsupported full-state value")
    return hashlib.sha256(canonical(encode(value)).encode()).hexdigest()


def _finite(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite checkpoint scalar")


def _identity(identity):
    if not isinstance(identity, dict) or not {"method", "repeat", "fold", "g1_sha256"} <= identity.keys():
        raise ValueError("Incomplete method/cell/source identity")
    if identity["method"] not in {"M0", "M1", "M2", "M3"} or identity["g1_sha256"] != G1_SHA256:
        raise ValueError("Unexpected method or frozen G1 identity")
    if type(identity["repeat"]) is not int or not 0 <= identity["repeat"] < 3 or type(identity["fold"]) is not int or not 0 <= identity["fold"] < 5:
        raise ValueError("Invalid original cell")
    canonical(identity)  # Identities/cursors are plain JSON; no pickle objects or tensors.


def _cursor(cursor):
    if not isinstance(cursor, dict) or type(cursor.get("actual_optimizer_updates")) is not int or cursor["actual_optimizer_updates"] < 0:
        raise ValueError("Invalid committed optimizer cursor")
    canonical(cursor)


def _rng():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cpu": torch.get_rng_state().clone(),
            "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}


def _optimizer_recipe(model, optimizer, scheduler):
    if not isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
        raise ValueError("Only audited Adam/AdamW state formats are supported")
    names = {id(p): name for name, p in model.named_parameters()}
    seen = set()
    groups = []
    for group in optimizer.param_groups:
        layout = []
        for parameter in group["params"]:
            if id(parameter) not in names or id(parameter) in seen:
                raise ValueError("Optimizer ownership/parameter order differs")
            seen.add(id(parameter))
            layout.append(names[id(parameter)])
        options = {key: value for key, value in group.items() if key not in {"params", "lr", "initial_lr"}}
        options["base_lr"] = group.get("initial_lr", group["lr"]) if scheduler is not None else group["lr"]
        groups.append({"parameter_names": layout, "options": options})
    return {"class": type(optimizer).__module__ + "." + type(optimizer).__qualname__, "groups": groups}


def _scheduler_recipe(scheduler):
    if scheduler is None:
        return None
    dynamic = {"last_epoch", "_step_count", "_last_lr", "_get_lr_called_within_step", "_is_initial",
               "best", "num_bad_epochs", "cooldown_counter"}
    config = {key: value for key, value in scheduler.state_dict().items() if key not in dynamic}
    return {"class": type(scheduler).__module__ + "." + type(scheduler).__qualname__, "config": config}


def capture_training_state(model, optimizer, identity, cursor, *, scheduler=None, sampler=None, model_metadata=None):
    _identity(identity)
    _cursor(cursor)
    if scheduler is not None and scheduler.optimizer is not optimizer:
        raise ValueError("Scheduler optimizer ownership differs")
    if (cursor["actual_optimizer_updates"] == 0) != (len(optimizer.state) == 0):
        raise ValueError("Optimizer moments and actual cursor differ")
    state = _snapshot({"schema": SCHEMA, "identity": identity, "cursor": cursor, "sampler": sampler,
             "model_metadata": model_metadata, "implementation_sha256": file_sha(__file__),
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "optimizer_recipe": _optimizer_recipe(model, optimizer, scheduler),
             "scheduler": scheduler.state_dict() if scheduler is not None else None,
             "scheduler_recipe": _scheduler_recipe(scheduler), "rng": _rng(),
             "module_modes": {name: module.training for name, module in model.named_modules()}})
    _finite(state)
    state["state_digest"] = training_state_digest(state)
    return state


def _basic_integrity(state):
    if not isinstance(state, dict) or set(state) != REQUIRED or state["schema"] != SCHEMA:
        raise ValueError("Missing or unexpected complete-state fields/schema")
    if state["state_digest"] != training_state_digest({key: value for key, value in state.items() if key != "state_digest"}):
        raise ValueError("Full checkpoint integrity differs")
    _identity(state["identity"])
    _cursor(state["cursor"])
    if state["implementation_sha256"] != file_sha(__file__):
        raise ValueError("Production checkpoint implementation source differs")
    _finite(state)


def _validate_rng(rng):
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "cpu", "cuda"}:
        raise ValueError("Missing complete RNG streams")
    try:
        random.Random().setstate(rng["python"])
        numpy_value = rng["numpy"]
        numpy_state = (numpy_value[0], np.asarray(numpy_value[1], dtype=np.uint32), *numpy_value[2:])
        np.random.RandomState().set_state(numpy_state)
        if not isinstance(rng["cpu"], torch.Tensor) or rng["cpu"].dtype != torch.uint8:
            raise ValueError("Invalid CPU RNG state dtype")
        torch.Generator(device="cpu").set_state(rng["cpu"].cpu())
        if not isinstance(rng["cuda"], list):
            raise ValueError("Invalid CUDA RNG streams")
        if rng["cuda"]:
            if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG topology differs")
            for index, value in enumerate(rng["cuda"]):
                if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
                    raise ValueError("Invalid CUDA RNG dtype")
                torch.Generator(device=f"cuda:{index}").set_state(value.cpu())
        elif torch.cuda.is_initialized():
            raise ValueError("CPU-only checkpoint cannot restore an initialized CUDA training context")
    except (TypeError, IndexError, KeyError) as exc:
        raise ValueError("Malformed RNG state") from exc
    return numpy_state


def _validate_optimizer(state, model, optimizer, scheduler):
    recipe = _optimizer_recipe(model, optimizer, scheduler)
    if state["optimizer_recipe"] != recipe:
        raise ValueError("Optimizer recipe/parameter layout differs")
    saved = state["optimizer"]
    if not isinstance(saved, dict) or set(saved) != {"param_groups", "state"} or len(saved["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("Malformed optimizer state")
    parameters = []
    saved_ids = []
    for actual, current, declared in zip(saved["param_groups"], optimizer.param_groups, recipe["groups"]):
        options = {key: value for key, value in actual.items() if key not in {"params", "lr", "initial_lr"}}
        options["base_lr"] = actual.get("initial_lr", actual["lr"]) if scheduler is not None else actual["lr"]
        if options != declared["options"] or len(actual["params"]) != len(current["params"]) or not 0 <= float(actual["lr"]):
            raise ValueError("Saved optimizer groups differ")
        saved_ids.extend(actual["params"])
        parameters.extend(current["params"])
    if saved_ids != list(range(len(parameters))):
        raise ValueError("Saved optimizer parameter order differs")
    moments = saved["state"]
    step = state["cursor"]["actual_optimizer_updates"]
    if not isinstance(moments, dict) or ((step == 0) != (len(moments) == 0)):
        raise ValueError("Optimizer cursor/moments differ")
    maximum_step = 0
    for index, values in moments.items():
        if type(index) is not int or not 0 <= index < len(parameters) or not {"step", "exp_avg", "exp_avg_sq"} <= values.keys() or not set(values) <= {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            raise ValueError("Malformed Adam moments")
        optstep = float(values["step"])
        if not optstep.is_integer() or not 1 <= optstep <= step:
            raise ValueError("Optimizer step differs from committed cursor")
        maximum_step = max(maximum_step, int(optstep))
        for key in set(values) - {"step"}:
            tensor = values[key]
            if tensor.shape != parameters[index].shape or tensor.dtype != parameters[index].dtype:
                raise ValueError("Optimizer moment tensor schema differs")
    # All approved M0-M3 trajectories have a shared prediction head active on
    # every optimizer update; inactive evidence branches may have lower counters.
    if maximum_step != step:
        raise ValueError("Committed optimizer cursor overreports the actual shared-head updates")
    if state["scheduler"] is not None and "_last_lr" in state["scheduler"]:
        last_lrs = state["scheduler"]["_last_lr"]
        if len(last_lrs) != len(saved["param_groups"]) or any(float(lr) != float(group["lr"]) for lr, group in zip(last_lrs, saved["param_groups"])):
            raise ValueError("Scheduler learning rate differs from optimizer groups")


def restore_training_state(state, model, optimizer, identity, *, scheduler=None, model_metadata=None):
    # Everything below is validated before model/optimizer/scheduler/global RNG mutation.
    _basic_integrity(state)
    _identity(identity)
    if state["identity"] != identity or state["model_metadata"] != model_metadata:
        raise ValueError("Cross-arm/cell/flags/source/data restore forbidden")
    current = model.state_dict()
    if current.keys() != state["model"].keys() or any(current[key].shape != value.shape or current[key].dtype != value.dtype for key, value in state["model"].items()):
        raise ValueError("Model state schema differs")
    modes = state["module_modes"]
    modules = dict(model.named_modules())
    if modes.keys() != modules.keys() or any(type(value) is not bool for value in modes.values()):
        raise ValueError("Model module modes differ")
    if _scheduler_recipe(scheduler) != state["scheduler_recipe"] or (scheduler is None) != (state["scheduler"] is None):
        raise ValueError("Scheduler recipe differs")
    if scheduler is not None:
        if scheduler.optimizer is not optimizer or set(scheduler.state_dict()) != set(state["scheduler"]):
            raise ValueError("Scheduler state schema differs")
        # A deep-copied scheduler can validate load behavior without touching the receiver.
        copy.deepcopy(scheduler).load_state_dict(copy.deepcopy(state["scheduler"]))
    _validate_optimizer(state, model, optimizer, scheduler)
    numpy_state = _validate_rng(state["rng"])
    model.load_state_dict(state["model"], strict=True)
    # Optimizer.load_state_dict may retain CPU tensor aliases; later training must
    # never mutate the captured snapshot or invalidate its already committed hash.
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    if scheduler is not None:
        scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    for name, module in modules.items():
        module.training = modes[name]
    random.setstate(state["rng"]["python"])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["rng"]["cpu"].cpu())
    if state["rng"]["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
    return {"cursor": copy.deepcopy(state["cursor"]), "sampler": copy.deepcopy(state["sampler"])}


def _allowed(path, mode="TEST_ONLY", permit=None):
    path = Path(path).resolve()
    if mode == "TEST_ONLY":
        allowed = TEST_ROOT
    elif mode == "FORMAL":
        if not isinstance(permit, dict) or "output_root" not in permit:
            raise PermissionError("Independent G3 execution permit required")
        allowed = Path(permit["output_root"]).resolve()
        if not allowed.is_relative_to((ROOT / "outputs/phase9/formal").resolve()):
            raise PermissionError("Formal checkpoint output root differs")
    else:
        raise PermissionError("Unknown checkpoint publication mode")
    if not path.is_relative_to(allowed) or path == allowed:
        raise PermissionError("Checkpoint must be inside the exact authorized publication namespace")
    return path


def _commit_digest(receipt):
    return hashlib.sha256(canonical({key: value for key, value in receipt.items() if key != "commit_digest"}).encode()).hexdigest()


def verify_commit(commit_path):
    commit_path = Path(commit_path).resolve()
    if commit_path.is_relative_to(TEST_ROOT):
        expected_mode = "TEST_ONLY"
    elif commit_path.is_relative_to((ROOT / "outputs/phase9/formal").resolve()):
        from .execution import APPROVED_EXECUTIONS
        if not APPROVED_EXECUTIONS:
            raise PermissionError("G3_NOT_AUTHORIZED: formal checkpoint access refused")
        expected_mode = "FORMAL"
    else:
        raise PermissionError("Checkpoint is outside Phase9 publication namespaces")
    if not commit_path.name.endswith(".COMMIT.json"):
        raise ValueError("Only a final COMMIT.json marker can authorize resume")
    receipt = json.loads(commit_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != COMMIT_SCHEMA or receipt.get("status") != "COMMITTED" or receipt.get("mode") != expected_mode:
        raise ValueError("Not a committed checkpoint in the permitted mode")
    if receipt.get("commit_digest") != _commit_digest(receipt):
        raise ValueError("Atomic commit marker integrity differs")
    permit = None
    if expected_mode == "FORMAL":
        from .execution import validate_execution_manifest
        identity = receipt["identity"]
        permit = validate_execution_manifest(receipt["execution_manifest_path"], identity["method"], identity["repeat"], identity["fold"])
        if permit["_manifest_sha256"] != receipt["execution_manifest_sha256"] or identity.get("mode") != "FORMAL":
            raise PermissionError("Formal checkpoint authorization identity differs")
    path = _allowed(receipt["checkpoint_path"], expected_mode, permit)
    root = _allowed(receipt["store_root"], expected_mode, permit)
    _allowed(commit_path, expected_mode, permit)
    if Path(receipt["checkpoint_commit_path"]).resolve() != commit_path or path.parent != root or commit_path.parent != root or not path.name.endswith(".state.pt"):
        raise ValueError("Checkpoint committed path/root differs")
    if path.stat().st_size != receipt["checkpoint_size"] or file_sha(path) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint file content/size differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    _basic_integrity(state)
    if receipt["identity"] != state["identity"] or receipt["cursor"] != state["cursor"] or receipt["state_digest"] != state["state_digest"]:
        raise ValueError("Commit/state identity differs")
    return receipt


class AtomicCheckpointStore:
    def __init__(self, root, mode="TEST_ONLY", execution_permit=None):
        self.permit = None
        if mode == "FORMAL":
            from .execution import verify_permit
            self.permit = verify_permit(execution_permit, "M0", 0, 0)
        elif mode != "TEST_ONLY":
            raise PermissionError("Unknown checkpoint publication mode")
        self.root = _allowed(root, mode, self.permit)
        self.mode = mode
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, name, state):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", name):
            raise ValueError("Checkpoint name must be a safe basename")
        _basic_integrity(state)
        if self.mode == "FORMAL":
            from .execution import verify_permit
            identity = state["identity"]
            self.permit = verify_permit(self.permit, identity["method"], identity["repeat"], identity["fold"])
            if identity.get("mode") != "FORMAL":
                raise PermissionError("TEST_ONLY state cannot be published as FORMAL")
        elif state["identity"].get("mode", "TEST_ONLY") != "TEST_ONLY":
            raise PermissionError("FORMAL state cannot be published into TEST_ONLY")
        state = _snapshot(state)
        marker = self.root / (name + ".COMMIT.json")
        final = self.root / (name + ".state.pt")
        if marker.exists():
            previous = verify_commit(marker)
            if previous["state_digest"] == state["state_digest"]:
                return previous
            raise FileExistsError("Cannot overwrite a committed checkpoint")
        if final.exists():
            raise FileExistsError("Uncommitted checkpoint file exists; fail closed")
        lock = self.root / (name + ".publish.lock")
        token = uuid.uuid4().hex
        with lock.open("xb") as handle:
            handle.write(token.encode())
            handle.flush()
            os.fsync(handle.fileno())
        partial = self.root / (name + "." + token + ".state.pt.partial")
        marker_partial = self.root / (name + "." + token + ".COMMIT.json.partial")
        try:
            with partial.open("xb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            sha = file_sha(partial)
            size = partial.stat().st_size
            os.replace(partial, final)
            receipt = {"schema": COMMIT_SCHEMA, "status": "COMMITTED", "mode": self.mode,
                       "identity": state["identity"], "cursor": state["cursor"],
                       "store_root": str(self.root), "checkpoint_path": str(final), "checkpoint_sha256": sha,
                       "checkpoint_size": size, "state_digest": state["state_digest"], "checkpoint_commit_path": str(marker)}
            if self.permit is not None:
                receipt["execution_manifest_path"] = self.permit["_manifest_path"]
                receipt["execution_manifest_sha256"] = self.permit["_manifest_sha256"]
            receipt["commit_digest"] = _commit_digest(receipt)
            with marker_partial.open("x", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(marker_partial, marker)  # Commit marker is the last publication action.
            return verify_commit(marker)
        finally:
            if lock.exists() and lock.read_text() == token:
                lock.unlink()

    def load(self, commit_path):
        receipt = verify_commit(commit_path)
        if Path(receipt["store_root"]).resolve() != self.root:
            raise ValueError("Checkpoint belongs to a different store")
        return torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=True)
