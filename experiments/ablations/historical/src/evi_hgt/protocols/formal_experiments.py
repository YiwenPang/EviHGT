"""Production-side loading and validation of the frozen formal run manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from copy import deepcopy
from collections.abc import Callable
from typing import Any, Mapping


LEGACY_FORMAL_MANIFEST_RELATIVE_PATH = (
    "outputs/phase7_3_3_1_protocol/formal_experiment_manifest.json"
)
FORMAL_MANIFEST_RELATIVE_PATH = (
    "outputs/phase8_practical_formal_v1/formal_experiment_manifest.json"
)
PRACTICAL_PROTOCOL_ID = "PRACTICAL_FORMAL_V1"
# Pinned after static construction; includes the complete training horizon.
PRACTICAL_PROTOCOL_SHA256 = "99e5e6f21409063bd95f985fb01f2cd447634fe9b7a492cc5a9ca15b38dbc2f0"
VALIDATION_MANIFEST_RELATIVE_PATH = (
    "outputs/phase7_3_1_protocol/validation_manifest_protocol.json"
)
LEGACY_FORMAL_RUN_ID_SEQUENCE_SHA256 = (
    "c3ffa5487ecd404698fb9f88ee32c4821ea63d714ba7589a3cb8cfbcf39ec5b4"
)
PRACTICAL_ORIGINAL_RUN_ID_SEQUENCE_SHA256 = (
    "cb0b6a4fb82fd75d03f843cff27f0769ec08052b330fe06cafdb9e8887154a29"
)
FORMAL_RUN_ID_SEQUENCE_SHA256 = 'b65d6a16f7708c60ef8f1fc3a78bb54af6ab0bacdbc65f7f55b06bb76a78f38b'
LR_SWEEP_EXTENSION_ID = 'FIVE_POINT_LR_SWEEP_V1'
LR_REFINEMENT_EXTENSION_ID = 'LOWER_LR_REFINEMENT_V1'
LOWER_LR_RUN_ID_SEQUENCE_SHA256 = 'c27de5dc1d444e16af7d8d2b1ed1c2ecc6770e187eb0106f3954e098a6c0a762'
MANIFEST_REPLACE_RETRY_DELAYS = (.05, .1, .2, .4, .8, 1.6)
FORMAL_RUN_STATUSES = (
    "PLANNED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "INTERRUPTED",
    "REARMED",
)
EXPECTED_FAMILY_COUNTS = {
    "lr_tuning": 2,
    "full_primary": 12,
    "base_hgt": 12,
    "pair_mlp": 0,
    "ablations": 15,
    "policy_sensitivity": 6,
}
FORMAL_RUN_STATUS_TRANSITIONS = frozenset(
    {
        ("PLANNED", "RUNNING"),
        ("INTERRUPTED", "RUNNING"),
        ("RUNNING", "COMPLETED"),
        ("RUNNING", "INTERRUPTED"),
        ("RUNNING", "FAILED"),
        ("REARMED", "RUNNING"),
        ("RUNNING", "RUNNING"),
    }
)


class FormalManifestError(ValueError):
    """Raised when a formal run cannot be validated or resolved exactly."""


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_formal_run_id(run: Mapping[str, Any]) -> str:
    """Preserve historical IDs; bind the practical scientific training budget."""

    identity = {
        "data_digest": run["data_digest"],
        "policy": run["policy"],
        "split": run["split"],
        "seed": int(run["seed"]),
        "model_variant": run["model_variant"],
        "learning_protocol_digest": run["learning_protocol_digest"],
        "evaluation_protocol_digest": run["evaluation_protocol_digest"],
        "candidate_protocol_digest": run["candidate_protocol_digest"],
        "architecture_digest": run["architecture_digest"],
        "run_family": run["run_family"],
        "requirement": run["requirement"],
        "learning_rate_binding": run["learning_rate_binding"],
    }
    if "formal_training_protocol_digest" in run:
        identity["formal_training_protocol_digest"] = run["formal_training_protocol_digest"]
    if 'execution_scope' in run:
        for key in ('execution_scope', 'source_cell_id', 'selected_lr_source'):
            identity[key] = run[key]
        if 'cold_visibility_policy_digest' in run:
            identity['cold_visibility_policy_digest'] = run['cold_visibility_policy_digest']
        if 'amendment_digest' in run:
            identity['amendment_digest'] = run['amendment_digest']
    return _canonical_digest(identity)


def practical_training_protocol_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read scientific settings only; never build a dataset, model or runtime."""
    phase = config["phase7_3_3_2"]
    paths = config["phase6"]["paths"]
    optimizer_step = config["phase5"]["training"]
    return {
        "protocol_id": PRACTICAL_PROTOCOL_ID,
        "formal_training": dict(phase["formal_training"]),
        "early_stopping": dict(phase["early_stopping"]),
        "checkpoint_selection": dict(phase["checkpoint_selection"]),
        "graph": dict(config["subgraph"]),
        "paths": {key: paths[key] for key in (
            "enabled", "algorithm", "algorithm_version", "min_length", "max_length",
            "max_paths", "beam_width", "max_neighbors_per_expansion",
            "use_hub_penalty", "hub_penalty_weight", "target_mask_policy",
        )},
        "optimizer_step": {key: optimizer_step[key] for key in (
            "gradient_accumulation_steps", "max_grad_norm", "scheduler", "num_workers",
        )},
        "recovery_policy": "COMPLETE_EPOCH_BOUNDARY_ONLY",
    }


def manifest_run_id_sequence_digest(manifest: Mapping[str, Any]) -> str:
    run_ids = [str(run["run_id"]) for run in manifest["runs"]]
    return hashlib.sha256("\n".join(run_ids).encode("utf-8")).hexdigest()


def validate_formal_experiment_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise FormalManifestError("formal manifest runs must be a list")
    refined = manifest.get('lr_sweep_extension') == LR_REFINEMENT_EXTENSION_ID
    extended = refined or manifest.get('lr_sweep_extension') == LR_SWEEP_EXTENSION_ID
    if 'lr_sweep_extension' in manifest and not extended:
        raise FormalManifestError('unknown LR sweep extension')
    expected_count = 52 if refined else 50 if extended else 47
    if len(runs) != expected_count:
        raise FormalManifestError(f"formal manifest must contain exactly {expected_count} runs")
    expected_families = {**EXPECTED_FAMILY_COUNTS, 'lr_tuning': 7 if refined else 5 if extended else 2}
    if manifest.get("family_counts") != expected_families:
        raise FormalManifestError("formal manifest family counts drifted")
    if manifest.get("pair_mlp_status") != (
        "NOT_IMPLEMENTED_FEATURE_CONTRACT_NOT_DEFENSIBLE"
    ):
        raise FormalManifestError("Pair MLP status drifted")
    run_ids = [str(run.get("run_id", "")) for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise FormalManifestError("formal manifest contains duplicate run IDs")
    practical = "formal_training_protocol_digest" in manifest
    if extended:
        from evi_hgt.training.model_selection import (
            model_selection_protocol_payload, LEARNING_RATE_CANDIDATES, COARSE_LEARNING_RATE_CANDIDATES,
        )
        candidates = LEARNING_RATE_CANDIDATES if refined else COARSE_LEARNING_RATE_CANDIDATES
        if not practical or manifest.get('lr_tuning_selection') != model_selection_protocol_payload(candidate_learning_rates=candidates):
            raise FormalManifestError('five-point LR selection protocol drifted')
        if refined:
            amendment = manifest.get('lr_search_amendment', {})
            active_runs = [r for r in runs if r['run_family']=='lr_tuning'
                           and r['learning_rate'] in candidates]
            active_ids = [r['run_id'] for r in sorted(active_runs, key=lambda r:r['learning_rate'])]
            if (amendment.get('active_run_ids') != active_ids or len(active_ids) != 5
                or amendment.get('classification') != 'HYPERPARAMETER_SEARCH_PLAN_AMENDMENT'
                or amendment.get('adaptive_refinement') is not True
                or amendment.get('all_final_points_preregistered_initially') is not False):
                raise FormalManifestError('lower-LR search amendment provenance drifted')
    if practical and (
        manifest["formal_training_protocol_digest"] != PRACTICAL_PROTOCOL_SHA256
        or _canonical_digest(manifest.get("formal_training_protocol")) != PRACTICAL_PROTOCOL_SHA256
    ):
        raise FormalManifestError("practical scientific training protocol drifted")
    for run in runs:
        if practical:
            if (run.get("formal_training_protocol_digest") != PRACTICAL_PROTOCOL_SHA256
                or run.get("formal_training_protocol") != manifest["formal_training_protocol"]):
                raise FormalManifestError("run practical training protocol binding drifted")
        elif "formal_training_protocol_digest" in run or "formal_training_protocol" in run:
            raise FormalManifestError("historical manifest cannot contain practical bindings")
        status = str(run.get("execution_status", ""))
        if status not in FORMAL_RUN_STATUSES:
            raise FormalManifestError(f"invalid formal run status: {status!r}")
        if compute_formal_run_id(run) != run["run_id"]:
            raise FormalManifestError(f"formal run identity drifted: {run['run_id']}")
        if run['learning_rate'] != run['learning_rate_binding']:
            raise FormalManifestError('learning rate differs from identity binding')
        validate_execution_lineage(run)
    # Project to the original immutable 47-cell matrix: this validates both
    # generations without changing historical IDs or searching a new matrix.
    matrix_ids = []
    for run in runs[:47]:
        original = dict(run)
        original.pop("formal_training_protocol_digest", None)
        matrix_ids.append(compute_formal_run_id(original))
    if hashlib.sha256("\n".join(matrix_ids).encode("utf-8")).hexdigest() != LEGACY_FORMAL_RUN_ID_SEQUENCE_SHA256:
        raise FormalManifestError("formal run ID sequence drifted from variant catalog")
    expected_sequence = (LOWER_LR_RUN_ID_SEQUENCE_SHA256 if refined else
                         FORMAL_RUN_ID_SEQUENCE_SHA256 if extended else
                         PRACTICAL_ORIGINAL_RUN_ID_SEQUENCE_SHA256 if practical else
                         LEGACY_FORMAL_RUN_ID_SEQUENCE_SHA256)
    if manifest_run_id_sequence_digest(manifest) != expected_sequence:
        raise FormalManifestError("formal run ID sequence differs from its protocol generation")
    required = sum(run["requirement"] == "REQUIRED" for run in runs)
    optional = sum(run["requirement"] == "OPTIONAL" for run in runs)
    tuning = sum(run["requirement"] == "TUNING" for run in runs)
    if (required, optional, tuning) != (42, 3, 7 if refined else 5 if extended else 2):
        raise FormalManifestError("formal required/optional/tuning budget drifted")
    return dict(manifest)


def _manifest_path_for_run(project_root, run_id=None):
    root = Path(project_root).resolve()
    historical = root / FORMAL_MANIFEST_RELATIVE_PATH
    if run_id is not None:
        payload = json.loads(historical.read_text(encoding='utf-8'))
        if not any(r['run_id'] == run_id for r in payload['runs']):
            from evi_hgt.protocols.scoped_execution import SCOPED_PATH
            scoped = root / SCOPED_PATH
            if scoped.is_file() and any(r['run_id'] == run_id for r in
                    json.loads(scoped.read_text(encoding='utf-8'))['runs']):
                return scoped
            from evi_hgt.protocols.confirmatory_ablation import MANIFEST_PATH
            supplemental = root / MANIFEST_PATH
            if supplemental.is_file() and any(r['run_id'] == run_id for r in
                    json.loads(supplemental.read_text(encoding='utf-8'))['runs']):
                return supplemental
            if scoped.is_file():
                return scoped
    return historical


def _validate_manifest_at_root(root, payload):
    from evi_hgt.protocols.confirmatory_ablation import SCOPE, validate_confirmatory_manifest
    if payload.get('execution_scope') == SCOPE:
        return validate_confirmatory_manifest(root, payload)
    if 'execution_scope' in payload:
        from evi_hgt.protocols.scoped_execution import validate_scoped_manifest
        return validate_scoped_manifest(root, payload)
    return validate_formal_experiment_manifest(payload)


def load_formal_experiment_manifest(project_root: str | Path, *, run_id=None) -> dict[str, Any]:
    path = _manifest_path_for_run(project_root, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"formal experiment manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalManifestError("formal experiment manifest must be an object")
    return _validate_manifest_at_root(project_root, payload)


def transition_formal_run_status(
    project_root: str | Path,
    *,
    run_id: str,
    expected_status: str,
    target_status: str,
) -> dict[str, Any]:
    """Atomically transition exactly one manifest run through the frozen state machine."""

    source = str(expected_status)
    target = str(target_status)
    if (source, target) not in FORMAL_RUN_STATUS_TRANSITIONS:
        raise FormalManifestError(
            f"illegal formal run status transition: {source} -> {target}"
        )
    def update(validated):
        matches = [run for run in validated['runs'] if run['run_id'] == run_id]
        if len(matches) != 1:
            raise FormalManifestError('formal run not found for the supplied identity')
        observed = str(matches[0].get('execution_status'))
        if observed != source:
            raise FormalManifestError(f'formal run status compare-and-swap failed: expected {source}, observed {observed}')
        matches[0]['execution_status'] = target
        if matches[0].get('execution_attempts'):
            attempt = matches[0]['execution_attempts'][-1]
            if 'record_digest' in attempt:
                raise FormalManifestError('sealed execution lineage is immutable')
            attempt['status'] = target
            if target == 'FAILED':
                attempt['recoverable_optimizer_steps'] = 0
                attempt['safe_to_resume'] = False
                attempt['trajectory_classification'] = 'INVALID'
    updated = _update_formal_manifest(project_root, update, run_id=run_id)
    return resolve_formal_run(updated, run_id=run_id)


def _replace_manifest_with_retry(source: Path, destination: Path) -> None:
    """Bounded Windows lock tolerance; keep the same atomic replacement payload."""
    for attempt in range(len(MANIFEST_REPLACE_RETRY_DELAYS) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            # Access denied / sharing / lock violations may be transient on Windows.
            # A missing/other code (including ordinary POSIX permissions) is fatal.
            if getattr(exc, 'winerror', None) not in {5, 32, 33} or attempt == len(MANIFEST_REPLACE_RETRY_DELAYS):
                raise
            time.sleep(MANIFEST_REPLACE_RETRY_DELAYS[attempt])


def _update_formal_manifest(project_root, update: Callable, *, run_id=None) -> dict[str, Any]:
    """One locked atomic writer for status transitions and lineage reconciliation."""
    root = Path(project_root).resolve()
    path = _manifest_path_for_run(root, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"formal experiment manifest is missing: {path}")
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FormalManifestError(
            "formal manifest status transition is already locked"
        ) from exc
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise FormalManifestError("formal experiment manifest must be an object")
        validated = _validate_manifest_at_root(root, payload)
        before = deepcopy(validated)
        update(validated)
        # Sealing is append-only history, even if a caller recomputes a digest.
        for old_run in before['runs']:
            sealed = [a for a in old_run.get('execution_attempts', []) if 'record_digest' in a]
            if sealed:
                new_run = next((r for r in validated['runs'] if r['run_id'] == old_run['run_id']), {})
                new_attempts = new_run.get('execution_attempts', [])
                if any(a['execution_sequence'] > len(new_attempts) or
                       new_attempts[a['execution_sequence']-1] != a for a in sealed):
                    raise FormalManifestError('sealed execution lineage is immutable')
        updated = _validate_manifest_at_root(root, validated)
        if updated == before:
            return updated
        encoded = (
            json.dumps(updated, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        temporary = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        )
        temporary_path = Path(temporary.name)
        try:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary.close()
            _replace_manifest_with_retry(temporary_path, path)
        except BaseException:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            raise
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
    return updated


def execution_identity(run_id: str, sequence: int) -> dict[str, Any]:
    if len(run_id) != 64 or type(sequence) is not int or sequence <= 0:
        raise FormalManifestError('execution identity is invalid')
    fields = {'formal_run_id': run_id, 'execution_sequence': sequence}
    return {**fields, 'execution_id': _canonical_digest(fields)}


def validate_execution_lineage(run: Mapping[str, Any]) -> None:
    attempts = run.get('execution_attempts')
    if attempts is None:
        if run['execution_status'] == 'REARMED':
            raise FormalManifestError('REARMED requires explicit execution lineage')
        return
    if not isinstance(attempts, list) or not attempts:
        raise FormalManifestError('execution lineage must be a nonempty list')
    for sequence, attempt in enumerate(attempts, 1):
        expected = execution_identity(str(run['run_id']), sequence)
        if any(attempt.get(k) != v for k,v in expected.items()):
            raise FormalManifestError('execution lineage identity mismatch')
        if sequence > 1:
            from evi_hgt.formal_execution import formal_output_directory
            expected_directory = f"{formal_output_directory(run['run_family'], run['run_id'])}/attempts/{sequence:04d}-{expected['execution_id']}"
            if attempt.get('output_directory') != expected_directory or attempt.get('predecessor_record_digest') != attempts[sequence-2].get('record_digest'):
                raise FormalManifestError('execution lineage directory/predecessor binding mismatch')
        if attempt.get('status') not in FORMAL_RUN_STATUSES:
            raise FormalManifestError('execution lineage status is invalid')
        if sequence < len(attempts) and 'record_digest' not in attempt:
            raise FormalManifestError('previous execution lineage must be sealed')
        if 'record_digest' in attempt:
            digest = _canonical_digest({k:v for k,v in attempt.items() if k!='record_digest'})
            from evi_hgt.protocols.scoped_execution import SCOPE
            external = run.get('execution_scope') == SCOPE and attempt['status'] == 'INTERRUPTED'
            if external:
                from evi_hgt.protocols.scoped_recovery import validate_external_interruption
                validate_external_interruption(attempt)
            if digest != attempt['record_digest'] or not (external or attempt['status'] in {'FAILED','COMPLETED'}):
                raise FormalManifestError('sealed execution lineage digest/status mismatch')
        observed = attempt.get('observed_optimizer_steps')
        recoverable = attempt.get('recoverable_optimizer_steps')
        if type(observed) is not int or type(recoverable) is not int or not 0 <= recoverable <= observed:
            raise FormalManifestError('execution lineage counters are invalid')
        if attempt.get('status') == 'REARMED' and (
            observed != 0 or recoverable != 0 or attempt.get('completed_epochs') != 0
            or not attempt.get('fresh_start_only') or attempt.get('best_exists') or attempt.get('last_exists')
        ):
            raise FormalManifestError('rearmed execution lineage must be fresh with zero counters')
    if attempts[-1]['status'] != run['execution_status']:
        raise FormalManifestError('execution lineage and current cell status differ')


def formal_step_accounting(manifest: Mapping[str, Any]) -> dict[str, int]:
    historical = current = recoverable = 0
    for run in manifest['runs']:
        attempts = run.get('execution_attempts', [])
        if not attempts and run['execution_status'] != 'PLANNED':
            raise FormalManifestError('formal execution accounting is unreconciled')
        historical += sum(a['observed_optimizer_steps'] for a in attempts)
        if attempts and attempts[-1]['status'] != 'FAILED':
            current += attempts[-1]['observed_optimizer_steps']
            recoverable += attempts[-1]['recoverable_optimizer_steps']
    return {'historical_observed_optimizer_steps': historical,
            'current_attempt_optimizer_steps': current,
            'current_recoverable_optimizer_steps': recoverable}


def record_execution_progress(project_root, *, run_id: str, execution_id: str,
                              observed_step: int, recoverable_step: int,
                              completed_epochs: int, last_exists: bool,
                              best_exists: bool, resumed: bool) -> None:
    """Persist observed high-water marks and committed complete-epoch LAST counts.

    The driver supplies a successfully written snapshot or startup/boundary
    inspection. Partial epoch execution never advances checkpoint progress.
    """
    if any(type(value) is not int or value < 0 for value in
           (observed_step, recoverable_step, completed_epochs)) or recoverable_step > observed_step:
        raise FormalManifestError('execution progress counters are invalid')
    def update(manifest):
        run = next((r for r in manifest['runs'] if r['run_id'] == run_id), None)
        if run is None or run['execution_status'] != 'RUNNING' or not run.get('execution_attempts'):
            raise FormalManifestError('execution progress requires a running attempt')
        attempt = run['execution_attempts'][-1]
        if attempt['execution_id'] != execution_id or 'record_digest' in attempt:
            raise FormalManifestError('execution progress identity/sealed lineage mismatch')
        attempt.update(observed_optimizer_steps=max(attempt['observed_optimizer_steps'], observed_step),
            recoverable_optimizer_steps=recoverable_step, completed_epochs=completed_epochs,
            last_exists=last_exists, best_exists=best_exists,
            validation_reached=completed_epochs > 0, resumed=attempt.get('resumed', False) or resumed)
        manifest['real_optimizer_step_count'] = formal_step_accounting(manifest)['historical_observed_optimizer_steps']
    _update_formal_manifest(project_root, update, run_id=run_id)


def reconcile_failed_attempt(project_root, *, run_id: str, record: Mapping[str, Any]):
    """Seal forensic evidence without manufacturing checkpoint/model state."""
    root = Path(project_root).resolve()
    evidence = dict(record)
    required = {'status','trajectory_classification','observed_optimizer_steps','recoverable_optimizer_steps',
        'completed_epochs','validation_reached','best_exists','last_exists','safe_to_resume','formal_test_count',
        'selected_lr','root_cause','repair_classification','run_log_path','run_log_sha256'}
    if not required <= evidence.keys() or evidence['status'] != 'FAILED' or evidence['trajectory_classification'] != 'INVALID':
        raise FormalManifestError('failed lineage evidence is incomplete')
    if evidence['safe_to_resume'] is not False or evidence['recoverable_optimizer_steps'] != 0:
        raise FormalManifestError('nonrecoverable failed lineage cannot restore state')
    log = (root / evidence['run_log_path']).resolve()
    if not log.is_relative_to(root / 'outputs/phase8_formal') or not log.is_file() or _file_sha256(log) != evidence['run_log_sha256']:
        raise FormalManifestError('failed lineage run log binding mismatch')
    if list(log.parent.parent.rglob('*.pt')):
        raise FormalManifestError('failed lineage claims no checkpoints but learned artifacts exist')
    record = {**evidence, **execution_identity(run_id, 1)}
    record['record_digest'] = _canonical_digest(record)
    def update(manifest):
        run = next((r for r in manifest['runs'] if r['run_id']==run_id), None)
        if run is None or run['execution_status'] != 'FAILED':
            raise FormalManifestError('failed lineage requires FAILED cell')
        if run.get('execution_attempts'):
            if run['execution_attempts'] != [record]:
                raise FormalManifestError('sealed failed execution lineage is immutable')
            return
        run['execution_attempts'] = [record]
        manifest['real_optimizer_step_count'] = formal_step_accounting(manifest)['historical_observed_optimizer_steps']
        manifest['optimizer_step_accounting_contract'] = 'HISTORICAL_OBSERVED_ACROSS_ATTEMPTS_NOT_RECOVERABLE_STATE'
    return _update_formal_manifest(root,update)


REARM_REQUIRED_GATES = frozenset({'lineage','identity','counter_semantics','mid_epoch_recovery',
    'exact_equivalence','cuda_bf16_recovery','checkpoint_fail_closed','historical_tests',
    'full_pytest','compileall','frozen_integrity','authorization','experiment_isolation'})


def rearm_fresh_execution(project_root, *, run_id: str, gate_report: Mapping[str, Any]):
    """Arm a fresh trajectory only after all patch gates; never construct a runtime."""
    gates = gate_report.get('gates', {})
    if gate_report.get('formal_run_id') != run_id or set(gates) != REARM_REQUIRED_GATES or not all(v is True for v in gates.values()):
        raise FormalManifestError('fresh execution rearm gate is incomplete or failed')
    from evi_hgt.utils.config import load_config
    from evi_hgt.formal_execution import phase8_authorization_payload, formal_output_directory
    root = Path(project_root).resolve()
    authorization = phase8_authorization_payload(load_config(root / 'configs/default.yaml'))
    if not authorization['phase8_execution_authorized'] or authorization['authorized_run_ids'] != [run_id]:
        raise FormalManifestError('fresh execution rearm authorization gate differs')
    def update(manifest):
        run = next((r for r in manifest['runs'] if r['run_id']==run_id), None)
        if run is None or run['execution_status'] != 'FAILED' or not run.get('execution_attempts'):
            raise FormalManifestError('rearm requires sealed failed execution lineage')
        previous = run['execution_attempts'][-1]
        if previous.get('safe_to_resume') is not False or 'record_digest' not in previous:
            raise FormalManifestError('rearm requires immutable nonresumable history')
        log = (root / previous['run_log_path']).resolve()
        if not log.is_relative_to(root / 'outputs/phase8_formal') or _file_sha256(log) != previous['run_log_sha256']:
            raise FormalManifestError('failed execution log changed before rearm')
        identity = execution_identity(run_id,len(run['execution_attempts'])+1)
        directory = f"{formal_output_directory(run['run_family'],run_id)}/attempts/{identity['execution_sequence']:04d}-{identity['execution_id']}"
        if (root / directory).exists():
            raise FormalManifestError('fresh execution output directory already exists')
        run['execution_attempts'].append({**identity, 'status':'REARMED','fresh_start_only':True,
            'resumed':False,'observed_optimizer_steps':0,'recoverable_optimizer_steps':0,
            'completed_epochs':0,'validation_reached':False,'best_exists':False,'last_exists':False,
            'output_directory':directory,'predecessor_record_digest':previous['record_digest'],
            'rearm_gate_digest':_canonical_digest(gate_report)})
        run['execution_status']='REARMED'
    return _update_formal_manifest(root,update)


def resolve_formal_run(
    manifest: Mapping[str, Any],
    *,
    run_id: str | None = None,
    experiment_family: str | None = None,
    model_variant: str | None = None,
    policy: str | None = None,
    split: str | None = None,
    seed: int | None = None,
    learning_rate: float | str | None = None,
) -> dict[str, Any]:
    """Resolve exactly one formal cell by stable ID or explicit selectors."""

    runs = list(manifest["runs"])
    selectors: tuple[tuple[str, object | None], ...] = (
        ("run_family", experiment_family),
        ("model_variant", model_variant),
        ("policy", policy),
        ("split", split),
        ("seed", seed),
        ("learning_rate", learning_rate),
    )
    if run_id is not None:
        runs = [run for run in runs if run["run_id"] == run_id]
    for field, value in selectors:
        if value is not None:
            runs = [run for run in runs if run[field] == value]
    if not runs:
        raise FormalManifestError("formal run not found for the supplied identity")
    if len(runs) != 1:
        raise FormalManifestError(
            f"formal run selectors are ambiguous ({len(runs)} matches); use --run-id"
        )
    return dict(runs[0])


def load_validation_manifest_bindings(
    project_root: str | Path, *, split: str
) -> dict[str, dict[str, object]]:
    """Return compact immutable bindings for both validation P/U channels."""

    path = Path(project_root).resolve() / VALIDATION_MANIFEST_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        split_manifests = payload["manifests"][split]
    except (KeyError, TypeError) as exc:
        raise FormalManifestError(f"validation manifest split not found: {split}") from exc
    result: dict[str, dict[str, object]] = {}
    for channel in ("support", "unsuitability"):
        raw = split_manifests[channel]
        result[channel] = {
            "source_artifact": VALIDATION_MANIFEST_RELATIVE_PATH,
            "source_artifact_sha256": _file_sha256(path),
            "manifest_id": raw["manifest_id"],
            "manifest_digest": raw["manifest_digest"],
            "split_kind": raw["split_kind"],
            "channel": raw["channel"],
            "partition": raw["partition"],
            "objective_id": raw["objective_id"],
            "positive_count": int(raw["positive_count"]),
            "unlabeled_count": int(raw["unlabeled_count"]),
            "exact_pair_comparison_count": int(raw["exact_pair_comparison_count"]),
            "fixed_across_epochs": bool(raw["fixed_across_epochs"]),
            "used_for_training": bool(raw["used_for_training"]),
            "test_labels_used": bool(raw["test_labels_used"]),
        }
    return result


__all__ = [
    "EXPECTED_FAMILY_COUNTS",
    "FORMAL_RUN_ID_SEQUENCE_SHA256",
    "LEGACY_FORMAL_RUN_ID_SEQUENCE_SHA256",
    "FORMAL_MANIFEST_RELATIVE_PATH",
    "LEGACY_FORMAL_MANIFEST_RELATIVE_PATH",
    "PRACTICAL_PROTOCOL_ID",
    "PRACTICAL_PROTOCOL_SHA256",
    "practical_training_protocol_from_config",
    "FORMAL_RUN_STATUS_TRANSITIONS",
    "FORMAL_RUN_STATUSES",
    "FormalManifestError",
    "compute_formal_run_id",
    "load_formal_experiment_manifest",
    "load_validation_manifest_bindings",
    "manifest_run_id_sequence_digest",
    "resolve_formal_run",
    "transition_formal_run_status",
    "validate_formal_experiment_manifest",
]
