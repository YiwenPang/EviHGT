"""Integrated production epoch and deterministic-validation driver for Phase 8."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
import logging
import math
from typing import Any

import torch
from evi_hgt.utils.progress import EpochProgress

from evi_hgt.evaluation.protocol import DeterministicValidationPUManifest
from evi_hgt.evaluation.validation import (
    DualValidationObjectiveResult,
    evaluate_dual_validation_objective,
    load_validation_manifests,
)
from evi_hgt.protocols.formal_experiments import transition_formal_run_status
from evi_hgt.sampling.path_leakage import PathLeakagePolicy, audit_path_leakage
from evi_hgt.training.batcher import TrainBatch
from evi_hgt.training.supervision import (
    PAIR_SEPARATOR,
    QuerySupervision,
    SampleRole,
    make_unlabeled_query_supervision,
)
from evi_hgt.utils.run_logging import (
    RUN_SUBDIRECTORIES,
    RunContext,
    configure_run_logging,
)


FORMAL_ROLE_BATCHING_ID = "FORMAL_DUAL_CHANNEL_EXACT_COVER_V1"
FORMAL_TERMINAL_REASONS = frozenset({"MAX_TRAINING_HORIZON_REACHED", "EARLY_STOPPING"})


class FormalRunDriverError(RuntimeError):
    """Raised when epoch orchestration cannot preserve the frozen protocol."""


@dataclass(frozen=True, slots=True)
class FormalEpochTrainingResult:
    epoch: int
    microbatch_count: int
    optimizer_steps: int
    global_optimizer_step: int
    training_objective: float
    support_risk: float
    unsuitability_risk: float
    mean_gradient_norm: float
    batching_id: str


@dataclass(frozen=True, slots=True)
class FormalValidationInferenceResult:
    objective: DualValidationObjectiveResult
    support_score_by_pair: Mapping[str, float]
    unsuitability_score_by_pair: Mapping[str, float]
    inference_pair_count: int
    support_pair_comparison_count: int
    unsuitability_pair_comparison_count: int
    optimizer_step_before: int
    optimizer_step_after: int

    @property
    def objective_value(self) -> float:
        return float(self.objective.risk.total.detach().float().cpu())


@dataclass(frozen=True, slots=True)
class FormalRunTrainingResult:
    run_id: str
    start_epoch: int
    last_completed_epoch: int
    global_optimizer_step: int
    completed_epochs: tuple[FormalEpochTrainingResult, ...]
    best_epoch: int
    best_validation_objective: float
    terminal_reason: str
    final_status: str = "COMPLETED"
    formal_test_evaluation_count: int = 0


def train_formal_epoch(
    trainer: Any,
    microbatches: Iterable[Iterable[TrainBatch]],
    *,
    epoch: int,
    progress_writer: Callable[[], None] | None = None,
    progress_display: EpochProgress | None = None,
) -> FormalEpochTrainingResult:
    """Consume one epoch while leaving every optimizer semantic in the trainer."""

    before = {name: int(value) for name, value in trainer.state_dict().items()}
    if int(epoch) != before["epoch"] + 1:
        raise FormalRunDriverError("formal epochs must be consecutive")
    microbatch_count = 0
    optimizer_step_count = 0
    total_loss_sum = support_risk_sum = unsuitability_risk_sum = gradient_norm_sum = 0.0
    step_at_start = before['optimizer_step']
    for group in microbatches:
        logical_batch = tuple(group)
        if not logical_batch:
            raise FormalRunDriverError("formal microbatch must not be empty")
        result = trainer.train_microbatch_group(logical_batch)
        if progress_writer is not None:
            progress_writer()
        if not math.isfinite(float(result.total_loss)):
            raise FormalRunDriverError("formal training objective must be finite")
        microbatch_count += 1
        optimizer_step_count += int(bool(result.optimizer_stepped))
        total_loss_sum += float(result.total_loss)
        support_risk_sum += float(result.support_risk)
        unsuitability_risk_sum += float(result.unsuitability_risk)
        gradient_norm_sum += float(result.gradient_norm)
        if progress_display is not None:
            progress_display.update(global_optimizer_step=step_at_start + optimizer_step_count)
    if microbatch_count == 0:
        raise FormalRunDriverError("formal epoch did not consume a training batch")
    boundary = {name: int(value) for name, value in trainer.state_dict().items()}
    if boundary["accumulation_position"] != 0:
        raise FormalRunDriverError(
            "formal epoch ended inside a gradient-accumulation window"
        )
    trainer.mark_epoch_completed(int(epoch))
    after = {name: int(value) for name, value in trainer.state_dict().items()}
    optimizer_steps = after["optimizer_step"] - step_at_start
    if optimizer_steps != optimizer_step_count:
        raise FormalRunDriverError("formal optimizer-step accounting drifted")
    return FormalEpochTrainingResult(
        epoch=int(epoch),
        microbatch_count=microbatch_count,
        optimizer_steps=optimizer_steps,
        global_optimizer_step=after["optimizer_step"],
        training_objective=total_loss_sum / microbatch_count,
        support_risk=support_risk_sum / microbatch_count,
        unsuitability_risk=unsuitability_risk_sum / microbatch_count,
        mean_gradient_norm=gradient_norm_sum / microbatch_count,
        batching_id=FORMAL_ROLE_BATCHING_ID,
    )


def _balanced_nonempty_chunks(
    values: tuple[QuerySupervision, ...], group_count: int
) -> tuple[tuple[QuerySupervision, ...], ...]:
    if group_count <= 0 or len(values) < group_count:
        raise FormalRunDriverError("formal role stream cannot form nonempty groups")
    return tuple(
        values[(index * len(values)) // group_count : ((index + 1) * len(values)) // group_count]
        for index in range(group_count)
    )


def make_formal_role_groups(
    records: tuple[QuerySupervision, ...],
) -> tuple[tuple[QuerySupervision, ...], ...]:
    support = tuple(
        record
        for record in records
        if record.support.role is SampleRole.POSITIVE
    )
    unsuitability = tuple(
        record
        for record in records
        if record.unsuitability.role is SampleRole.POSITIVE
    )
    unlabeled = tuple(
        record
        for record in records
        if record.observed_relation is None
        and record.support.role is SampleRole.UNLABELED
        and record.support.pu_risk_eligible
        and record.unsuitability.role is SampleRole.UNLABELED
        and record.unsuitability.pu_risk_eligible
    )
    expected_groups = min(len(support), len(unsuitability))
    if len(unlabeled) != expected_groups or expected_groups <= 0:
        raise FormalRunDriverError(
            "formal exact-cover dataset has an invalid P/U role cardinality"
        )
    support_chunks = _balanced_nonempty_chunks(support, expected_groups)
    unsuitability_chunks = _balanced_nonempty_chunks(
        unsuitability, expected_groups
    )
    groups: list[tuple[QuerySupervision, ...]] = []
    for index in range(expected_groups):
        ordered: dict[str, QuerySupervision] = {}
        for record in (
            *support_chunks[index],
            *unsuitability_chunks[index],
            unlabeled[index],
        ):
            ordered.setdefault(record.canonical_pair_key, record)
        group = tuple(ordered.values())
        if not any(
            record.support.role is SampleRole.POSITIVE for record in group
        ) or not any(
            record.unsuitability.role is SampleRole.POSITIVE for record in group
        ):
            raise FormalRunDriverError("formal role group lost a positive stream")
        groups.append(group)
    return tuple(groups)


def _path_enabled(runtime: Any) -> bool:
    architecture = runtime.execution.run.get("architecture", {})
    return bool(architecture.get("use_path_reasoning", True))


def _assemble_production_query(
    runtime: Any,
    query: QuerySupervision,
    *,
    partition: str | None,
) -> TrainBatch:
    components = runtime.components
    batch, assembly = components.assembler.assemble(
        query,
        split_kind=str(runtime.execution.run["split"]),
        device=components.device,
        evaluation_partition=partition,
    )
    if any(
        int(getattr(assembly, name))
        for name in (
            "target_answer_fact_survivor_count",
            "split_forbidden_fact_survivor_count",
            "target_evidence_survivor_count",
            "cached_target_pair_feature_survivor_count",
        )
    ):
        raise FormalRunDriverError("formal query graph leakage audit failed")
    if not _path_enabled(runtime):
        return batch
    source = batch.path_source
    if source is None:
        raise FormalRunDriverError("formal path-enabled query has no PathSource")
    if source.split_partition != (partition or "train"):
        raise FormalRunDriverError("formal PathSource partition binding drifted")
    candidates = components.path_extractor.extract(source)
    path_audit = audit_path_leakage(
        candidates,
        PathLeakagePolicy(
            split_kind=source.split_kind,
            query_pair_id=source.query_pair_id,
            drug_node_id=source.node_ids[source.drug_node_index],
            disease_node_id=source.node_ids[source.disease_node_index],
            forbidden_fact_ids=source.forbidden_fact_ids,
            heldout_node_ids=source.heldout_node_ids,
            target_relation_aliases={},
            target_metadata_fact_ids=source.forbidden_fact_ids,
            cache_split_kind=source.split_kind,
        ),
    )
    if not path_audit.passed:
        raise FormalRunDriverError("formal post-path leakage audit failed")
    packed = components.path_packer(
        candidates,
        num_queries=1,
        max_length=int(components.path_extractor.config.max_length),
        device=torch.device(components.device),
    )
    return replace(batch, path_batch=packed)


def build_formal_epoch_microbatches(
    runtime: Any, epoch: int, *, progress_display: EpochProgress | None = None,
) -> Iterable[tuple[TrainBatch, ...]]:
    """Yield bounded four-stream groups from the existing role dataset/loader."""

    execution = runtime.execution
    components = runtime.components
    phase3_root = execution.project_root / str(
        execution.config["phase7_3_3_1"]["data"]["source"]
    )
    seed = int(execution.run["seed"])
    dataset = components.dataset_factory(
        phase3_root,
        split_kind=str(execution.run["split"]),
        seed=seed,
    )
    loader = components.dataloader_factory(
        dataset,
        batch_size=len(dataset),
        seed=seed,
        shuffle=True,
        num_workers=int(components.runtime_config.num_workers),
    )
    role_batches = list(loader)
    if len(role_batches) != 1 or len(role_batches[0]) != len(dataset):
        raise FormalRunDriverError("formal role DataLoader must produce one exact epoch")
    role_groups = make_formal_role_groups(tuple(role_batches[0]))
    if progress_display is not None:
        progress_display.set_total(len(role_groups))
    for role_group in role_groups:
        yield tuple(
            _assemble_production_query(runtime, query, partition=None)
            for query in role_group
        )


def _load_runtime_validation_manifests(
    runtime: Any,
) -> dict[str, DeterministicValidationPUManifest]:
    execution = runtime.execution
    bindings = execution.validation_manifests
    if set(bindings) != {"support", "unsuitability"}:
        raise FormalRunDriverError("formal validation bindings are incomplete")
    source_paths = {
        str(value["source_artifact"]) for value in bindings.values()
    }
    if len(source_paths) != 1:
        raise FormalRunDriverError("formal validation channels use different sources")
    all_manifests = load_validation_manifests(
        execution.project_root / next(iter(source_paths))
    )
    split = str(execution.run["split"])
    selected = {
        channel: all_manifests[(split, channel)]
        for channel in ("support", "unsuitability")
    }
    for channel, manifest in selected.items():
        if manifest.digest != bindings[channel]["manifest_digest"]:
            raise FormalRunDriverError(
                f"formal validation manifest digest drifted: {channel}"
            )
    return selected


def _build_validation_query_batch(
    runtime: Any, pair_id: str, partition: str
) -> TrainBatch:
    parts = pair_id.split(PAIR_SEPARATOR)
    if len(parts) != 2 or not all(parts):
        raise FormalRunDriverError("formal validation pair ID is invalid")
    query = make_unlabeled_query_supervision(parts[0], parts[1])
    if query.canonical_pair_key != pair_id:
        raise FormalRunDriverError("formal validation pair identity drifted")
    return _assemble_production_query(runtime, query, partition=partition)


def _optimizer_step(runtime: Any) -> int:
    state = runtime.trainer.state_dict()
    return int(state["optimizer_step"])


def run_deterministic_validation_inference(
    runtime: Any,
    *,
    manifests: Mapping[str, DeterministicValidationPUManifest] | None = None,
    query_batch_builder: Callable[[Any, str, str], Any] = _build_validation_query_batch,
    evaluator: Callable[..., DualValidationObjectiveResult] = evaluate_dual_validation_objective,
) -> FormalValidationInferenceResult:
    """Infer frozen validation score maps and delegate objective math unchanged."""

    selected = (
        _load_runtime_validation_manifests(runtime)
        if manifests is None
        else dict(manifests)
    )
    if set(selected) != {"support", "unsuitability"}:
        raise FormalRunDriverError("formal validation requires exactly two channels")
    support = selected["support"]
    unsuitability = selected["unsuitability"]
    split = str(runtime.execution.run["split"])
    if (
        support.channel != "support"
        or unsuitability.channel != "unsuitability"
        or support.split_kind != split
        or unsuitability.split_kind != split
    ):
        raise FormalRunDriverError("formal validation manifest namespace drifted")
    support_ids = (*support.positive_pair_ids, *support.unlabeled_pair_ids)
    unsuitability_ids = (
        *unsuitability.positive_pair_ids,
        *unsuitability.unlabeled_pair_ids,
    )
    ordered_union = tuple(dict.fromkeys((*support_ids, *unsuitability_ids)))
    model = runtime.components.model_variant.model
    device = torch.device(runtime.components.runtime_config.device)
    precision = str(runtime.components.runtime_config.precision)
    optimizer_before = _optimizer_step(runtime)
    previous_training = bool(model.training)
    all_support: dict[str, float] = {}
    all_unsuitability: dict[str, float] = {}
    model.eval()
    try:
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda" and precision == "bf16"),
        ):
            for pair_id in ordered_union:
                batch = query_batch_builder(runtime, pair_id, "validation")
                prediction = model(
                    batch.model_batch,
                    path_batch=batch.path_batch,
                )
                if (
                    prediction.support_logit.numel() != 1
                    or prediction.unsuitability_logit.numel() != 1
                ):
                    raise FormalRunDriverError(
                        "formal validation requires one scalar pair prediction"
                    )
                support_value = float(
                    prediction.support_logit.detach().float().cpu().reshape(())
                )
                unsuitability_value = float(
                    prediction.unsuitability_logit.detach().float().cpu().reshape(())
                )
                if not math.isfinite(support_value) or not math.isfinite(
                    unsuitability_value
                ):
                    raise FormalRunDriverError(
                        "formal validation model scores must be finite"
                    )
                all_support[pair_id] = support_value
                all_unsuitability[pair_id] = unsuitability_value
    finally:
        model.train(previous_training)
    optimizer_after = _optimizer_step(runtime)
    if optimizer_after != optimizer_before:
        raise FormalRunDriverError("formal validation changed optimizer state")
    support_scores = {pair_id: all_support[pair_id] for pair_id in support_ids}
    unsuitability_scores = {
        pair_id: all_unsuitability[pair_id] for pair_id in unsuitability_ids
    }
    objective = evaluator(
        support_manifest=support,
        unsuitability_manifest=unsuitability,
        support_score_by_pair=support_scores,
        unsuitability_score_by_pair=unsuitability_scores,
        objective=runtime.components.objective,
    )
    value = float(objective.risk.total.detach().float().cpu())
    if not math.isfinite(value):
        raise FormalRunDriverError("formal deterministic validation is non-finite")
    return FormalValidationInferenceResult(
        objective=objective,
        support_score_by_pair=support_scores,
        unsuitability_score_by_pair=unsuitability_scores,
        inference_pair_count=len(ordered_union),
        support_pair_comparison_count=support.exact_pair_comparison_count,
        unsuitability_pair_comparison_count=(
            unsuitability.exact_pair_comparison_count
        ),
        optimizer_step_before=optimizer_before,
        optimizer_step_after=optimizer_after,
    )


def _default_runtime_factory(execution: Any) -> Any:
    from evi_hgt.formal_execution import invoke_formal_learning_trainer

    return invoke_formal_learning_trainer(execution)


def _default_state_writer(execution: Any, source: str, target: str) -> None:
    if (
        source == execution.state_transition.source_status
        and target == execution.state_transition.target_status
    ):
        execution.state_transition.apply(execution.project_root)
        return
    transition_formal_run_status(
        execution.project_root,
        run_id=execution.run_id,
        expected_status=source,
        target_status=target,
    )


def _default_logger_factory(execution: Any) -> logging.Logger:
    run_dir = (execution.project_root / execution.output_directory).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    directories = {name: run_dir / name for name in RUN_SUBDIRECTORIES}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    log_file = directories["logs"] / "run.log"
    log_file.touch(exist_ok=True)
    return configure_run_logging(
        RunContext(run_dir=run_dir, directories=directories, log_file=log_file)
    )


def _healthy_last_checkpoint(runtime: Any) -> bool:
    from evi_hgt.training.formal_checkpoint import inspect_formal_checkpoint_file

    lifecycle = runtime.checkpoint_lifecycle
    try:
        inspect_formal_checkpoint_file(
            lifecycle.paths.last,
            expected_identity=lifecycle.identity,
            expected_role="LAST",
            map_location="cpu",
        )
    except Exception:
        return False
    return True


def _persist_execution_progress(runtime: Any, *, inspect_last: bool = False) -> None:
    from evi_hgt.protocols.formal_experiments import record_execution_progress
    from evi_hgt.training.formal_checkpoint import inspect_formal_checkpoint_file

    execution = runtime.execution
    attempts = execution.run.get('execution_attempts')
    if not attempts:
        return  # Historical fixtures predate the execution-lineage contract.
    lifecycle = runtime.checkpoint_lifecycle
    snapshot = lifecycle.last_snapshot
    if inspect_last and lifecycle.paths.last.exists():
        snapshot = inspect_formal_checkpoint_file(lifecycle.paths.last,
            expected_identity=lifecycle.identity, expected_role='LAST', map_location='cpu').snapshot
        lifecycle.last_snapshot = snapshot
    record_execution_progress(execution.project_root, run_id=execution.run_id,
        execution_id=attempts[-1]['execution_id'],
        observed_step=int(runtime.trainer.state_dict()['optimizer_step']),
        recoverable_step=0 if snapshot is None else snapshot.global_optimizer_step,
        completed_epochs=0 if snapshot is None else snapshot.early_stopping_state.last_epoch,
        last_exists=snapshot is not None,
        best_exists=snapshot is not None and snapshot.best_artifact is not None,
        resumed=runtime.checkpoint_decision.action == 'RESUME_ALLOWED')


def run_formal_training(
    execution: Any,
    *,
    runtime_factory: Callable[[Any], Any] = _default_runtime_factory,
    epoch_batch_factory: Callable[[Any, int], Iterable[Iterable[TrainBatch]]] = build_formal_epoch_microbatches,
    validation_driver: Callable[[Any], Any] = run_deterministic_validation_inference,
    state_writer: Callable[[Any, str, str], None] = _default_state_writer,
    logger_factory: Callable[[Any], Any] = _default_logger_factory,
) -> FormalRunTrainingResult:
    """Run the sole production epoch/validation/checkpoint/status lifecycle."""

    early_config = execution.config["phase7_3_3_2"]["early_stopping"]
    max_epochs = int(early_config["max_epochs"])
    if max_epochs <= 0:
        raise FormalRunDriverError("formal max_epochs must be positive")
    if str(early_config["validation_cadence"]) != "ONCE_PER_EPOCH":
        raise FormalRunDriverError("formal validation cadence drifted")
    runtime = runtime_factory(execution)
    trainer_state = {
        name: int(value) for name, value in runtime.trainer.state_dict().items()
    }
    early_state = runtime.checkpoint_lifecycle.early_stopping_state
    if early_state.last_epoch != trainer_state["epoch"]:
        raise FormalRunDriverError(
            "restored trainer and early-stopping epochs are inconsistent"
        )
    start_epoch = trainer_state["epoch"] + 1
    terminal = start_epoch > max_epochs or early_state.should_stop
    if terminal and runtime.checkpoint_decision.action != 'RESUME_ALLOWED':
        raise FormalRunDriverError("formal terminal state requires verified LAST")
    source_status = str(execution.state_transition.source_status)
    if source_status != str(execution.run["execution_status"]):
        raise FormalRunDriverError("formal execution source status drifted")
    state_writer(execution, source_status, "RUNNING")
    logger: Any = logging.getLogger("evi_hgt")
    try:
        logger = logger_factory(execution)
        if state_writer is _default_state_writer:
            _persist_execution_progress(runtime, inspect_last=True)
        logger.info(
            "Formal run started run_id=%s start_epoch=%d checkpoint_action=%s",
            execution.run_id,
            start_epoch,
            runtime.checkpoint_decision.action,
        )
        completed: list[FormalEpochTrainingResult] = []
        terminal_reason = (
            'EARLY_STOPPING' if early_state.epochs_without_improvement >= int(early_config['patience'])
            else 'MAX_TRAINING_HORIZON_REACHED'
        ) if terminal else ''
        # A crash after terminal LAST publication only needs status reconciliation.
        for epoch in (() if terminal else range(start_epoch, max_epochs + 1)):
            with EpochProgress(mode=getattr(execution, 'progress_mode', 'auto'),
                               epoch=epoch, max_epochs=max_epochs,
                               learning_rate=execution.run.get('learning_rate')) as display:
                batches = (epoch_batch_factory(runtime, epoch, progress_display=display)
                           if epoch_batch_factory is build_formal_epoch_microbatches
                           else epoch_batch_factory(runtime, epoch))
                training = train_formal_epoch(
                    runtime.trainer, batches, epoch=epoch, progress_display=display,
                    progress_writer=(lambda: _persist_execution_progress(runtime)) if state_writer is _default_state_writer else None,
                )
            validation = validation_driver(runtime)
            checkpoint = runtime.checkpoint_lifecycle.record_completed_epoch(
                float(validation.objective_value)
            )
            if state_writer is _default_state_writer:
                _persist_execution_progress(runtime, inspect_last=True)
            completed.append(training)
            logger.info(
                "Formal epoch=%d train_objective=%.9f validation_objective=%.9f "
                "support_risk=%.9f unsuitability_risk=%.9f mean_gradient_norm=%.9f "
                "microbatches=%d optimizer_step=%d validation_pairs=%s "
                "best_updated=%s stale_epochs=%d",
                epoch,
                training.training_objective,
                float(validation.objective_value),
                training.support_risk,
                training.unsuitability_risk,
                training.mean_gradient_norm,
                training.microbatch_count,
                training.global_optimizer_step,
                getattr(validation, "inference_pair_count", "test-fixture"),
                checkpoint.best_updated,
                checkpoint.early_stopping_state.epochs_without_improvement,
            )
            if checkpoint.early_stopping_state.should_stop:
                terminal_reason = (
                    "EARLY_STOPPING"
                    if checkpoint.early_stopping_state.epochs_without_improvement
                    >= int(early_config["patience"])
                    else "MAX_TRAINING_HORIZON_REACHED"
                )
                break
            if epoch == max_epochs:
                terminal_reason = "MAX_TRAINING_HORIZON_REACHED"
                break
        if terminal_reason not in FORMAL_TERMINAL_REASONS:
            raise FormalRunDriverError("formal epoch loop has no terminal decision")
        final_early = runtime.checkpoint_lifecycle.early_stopping_state
        best = final_early.best_checkpoint
        if best is None:
            raise FormalRunDriverError("formal completed run has no BEST checkpoint")
        final_state = {
            name: int(value)
            for name, value in runtime.trainer.state_dict().items()
        }
        state_writer(execution, "RUNNING", "COMPLETED")
        logger.info(
            "Formal run completed run_id=%s reason=%s epoch=%d optimizer_step=%d",
            execution.run_id,
            terminal_reason,
            final_state["epoch"],
            final_state["optimizer_step"],
        )
        return FormalRunTrainingResult(
            run_id=execution.run_id,
            start_epoch=start_epoch,
            last_completed_epoch=final_state["epoch"],
            global_optimizer_step=final_state["optimizer_step"],
            completed_epochs=tuple(completed),
            best_epoch=best.epoch,
            best_validation_objective=best.validation_objective,
            terminal_reason=terminal_reason,
        )
    except KeyboardInterrupt:
        if state_writer is _default_state_writer:
            _persist_execution_progress(runtime, inspect_last=True)
        target = "INTERRUPTED" if _healthy_last_checkpoint(runtime) else "FAILED"
        state_writer(execution, "RUNNING", target)
        logger.exception(
            "Formal run externally interrupted run_id=%s status=%s",
            execution.run_id,
            target,
        )
        raise
    except BaseException:
        state_writer(execution, "RUNNING", "FAILED")
        logger.exception("Formal run failed run_id=%s", execution.run_id)
        raise


__all__ = [
    "FORMAL_ROLE_BATCHING_ID",
    "FormalEpochTrainingResult",
    "FormalRunDriverError",
    "FormalRunTrainingResult",
    "FormalValidationInferenceResult",
    "build_formal_epoch_microbatches",
    "make_formal_role_groups",
    "run_deterministic_validation_inference",
    "run_formal_training",
    "train_formal_epoch",
]
