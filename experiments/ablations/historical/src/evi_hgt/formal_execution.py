"""Run-scoped formal study planning and protected formal-training orchestration."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from evi_hgt.protocols.formal_experiments import (
    compute_formal_run_id,
    load_formal_experiment_manifest,
    load_validation_manifest_bindings,
    resolve_formal_run,
    transition_formal_run_status,
)


PHASE8_EXECUTION_AUTHORIZATION_REQUIRED = (
    "PHASE8_EXECUTION_AUTHORIZATION_REQUIRED"
)
VARIANT_CATALOG_RELATIVE_PATH = (
    "outputs/phase7_3_3_1_protocol/variant_catalog.json"
)
ENVIRONMENT_SNAPSHOT_FIELDS = (
    "python_version",
    "pytorch_version",
    "pyg_version",
    "cuda_version",
    "gpu",
    "vram_bytes",
    "cpu",
    "ram_bytes",
    "seed",
    "model_variant",
    "policy",
    "split",
    "config_digest",
    "data_digest",
    "protocol_digests",
    "git_repository",
)
_OUTPUT_FAMILY = {
    "lr_tuning": "tuning",
    "full_primary": "primary",
    "base_hgt": "baselines",
    "ablations": "ablations",
    "policy_sensitivity": "policy_sensitivity",
}
_FORMAL_MODES = {"train", "evaluate", "full"}


class FormalRunSelectionError(ValueError):
    """Raised when production execution is not bound to one manifest run ID."""


class FormalRunAuthorizationError(RuntimeError):
    """Raised when the requested manifest run is outside the allowed scope."""


class FormalRunStateError(RuntimeError):
    """Raised when a fresh formal invocation targets a non-PLANNED run."""


class FormalRunIntegrityError(RuntimeError):
    """Raised when frozen run, protocol, config, or source bindings drift."""


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class EvaluationIdentity:
    """Identity that a future formal evaluation artifact must bind."""

    checkpoint_digest: str
    candidate_protocol_digest: str
    evaluation_protocol_digest: str
    split_partition: str
    ranking_score: str

    def __post_init__(self) -> None:
        _validate_sha256(self.checkpoint_digest, name="checkpoint_digest")
        _validate_sha256(
            self.candidate_protocol_digest, name="candidate_protocol_digest"
        )
        _validate_sha256(
            self.evaluation_protocol_digest, name="evaluation_protocol_digest"
        )
        if self.split_partition not in {"validation", "test"}:
            raise ValueError("split_partition must be validation or test")
        if self.ranking_score not in {
            "support_logit",
            "unsuitability_logit",
            "m_T",
            "m_F",
        }:
            raise ValueError("ranking_score is not a frozen formal score")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FormalRunStateTransition:
    """Compare-and-swap claim used by the sole formal run driver."""

    run_id: str
    source_status: str = "PLANNED"
    target_status: str = "RUNNING"

    def apply(self, project_root: str | Path) -> Mapping[str, Any]:
        return transition_formal_run_status(
            project_root,
            run_id=self.run_id,
            expected_status=self.source_status,
            target_status=self.target_status,
        )


@dataclass(frozen=True, slots=True)
class FormalTrainingExecution:
    """Validated, single-run execution context with no scheduler-wide state."""

    project_root: Path
    config: Mapping[str, Any]
    config_digest: str
    run: Mapping[str, Any]
    model: Mapping[str, Any]
    validation_manifests: Mapping[str, Mapping[str, object]]
    authorization: Mapping[str, Any]
    training_protocol: Mapping[str, Any]
    output_directory: str
    state_transition: FormalRunStateTransition
    manifest_run_count: int
    other_run_count: int
    resume_requested: bool = False
    progress_mode: str = 'auto'  # Execution UI only; never part of config/run digests.

    @property
    def run_id(self) -> str:
        return str(self.run["run_id"])


@dataclass(frozen=True, slots=True)
class FormalTrainingComponents:
    """Existing production components bound to one validated formal run."""

    assembler: object
    candidate_universe: object
    unlabeled_sampler: object
    dataset_factory: object
    dataloader_factory: object
    path_extractor: object
    path_packer: object
    model_variant: object
    objective: object
    runtime_config: object
    device: str
    dataloader_generator: object | None = None


@dataclass(frozen=True, slots=True)
class FormalTrainingRuntime:
    """Exact-run runtime after checkpoint preflight and optional resume restore."""

    execution: FormalTrainingExecution
    components: FormalTrainingComponents
    trainer: object
    checkpoint_decision: object
    checkpoint_lifecycle: object


def phase8_authorization_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    phase8 = config["phase7_3_3_2"]["phase8"]
    authorized = phase8["execution_authorized"]
    cli_override = phase8["cli_override_permitted"]
    if not isinstance(authorized, bool) or not isinstance(cli_override, bool):
        raise TypeError("formal study authorization fields must be booleans")
    if cli_override:
        raise RuntimeError("formal study authorization must not be CLI-overridable")
    authorized_run_ids = phase8.get("authorized_run_ids")
    if not isinstance(authorized_run_ids, list) or (authorized and not authorized_run_ids):
        raise TypeError("enabled formal study authorization requires a non-empty run-ID list")
    if len(authorized_run_ids) != len(set(authorized_run_ids)):
        raise ValueError("formal study authorization contains duplicate run IDs")
    for run_id in authorized_run_ids:
        if not isinstance(run_id, str):
            raise TypeError("formal study authorized run IDs must be strings")
        _validate_sha256(run_id, name="authorized_run_id")
    return {
        "phase8_execution_authorized": authorized,
        "phase_state": str(phase8["phase_state"]),
        "authorization_source": str(phase8["authorization_source"]),
        "authorized_run_ids": list(authorized_run_ids),
        "cli_override_permitted": cli_override,
        "dry_run_allowed": True,
    }


def enforce_phase8_execution_authorization(
    config: Mapping[str, Any], *, mode: str
) -> None:
    """Fail before any formal state mutation unless formal study is authorized."""

    if mode not in _FORMAL_MODES:
        raise ValueError(f"not a formal execution mode: {mode}")
    if not phase8_authorization_payload(config)["phase8_execution_authorized"]:
        raise RuntimeError(PHASE8_EXECUTION_AUTHORIZATION_REQUIRED)


def formal_output_directory(run_family: str, run_id: str) -> str:
    try:
        output_family = _OUTPUT_FAMILY[run_family]
    except KeyError as exc:
        raise ValueError(f"unknown formal run family: {run_family}") from exc
    _validate_sha256(run_id, name="run_id")
    return f"outputs/phase8_formal/{output_family}/{run_id}"


def _load_variant_record(project_root: Path, variant_id: str) -> dict[str, Any]:
    from evi_hgt.protocols.confirmatory_ablation import VARIANT, supplemental_variant_record
    if variant_id == VARIANT:
        return supplemental_variant_record(project_root)
    path = project_root / VARIANT_CATALOG_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in payload["variants"] if row["variant_id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"model variant is not uniquely catalogued: {variant_id}")
    record = matches[0]
    if record["status"] != "IMPLEMENTED":
        raise ValueError(f"model variant is unavailable: {variant_id}")
    return {
        "variant_id": variant_id,
        "status": record["status"],
        "total_parameters": int(record["total_parameters"]),
        "trainable_parameters": int(record["trainable_parameters"]),
        "catalog_artifact": VARIANT_CATALOG_RELATIVE_PATH,
        "catalog_artifact_sha256": _file_sha256(path),
    }


def _require_integrity(condition: bool, message: str) -> None:
    if not condition:
        raise FormalRunIntegrityError(message)


def _validate_formal_run_integrity(
    project_root: Path,
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    validation_manifests: Mapping[str, Mapping[str, object]],
) -> None:
    """Bind the resolved manifest cell to frozen payloads, config, and code."""

    try:
        from evi_hgt.evaluation.validation import load_validation_manifests

        _require_integrity(
            compute_formal_run_id(run) == run["run_id"],
            "formal run identity does not match its run ID",
        )
        for payload_name, digest_name in (
            ("architecture", "architecture_digest"),
            ("learning_protocol", "learning_protocol_digest"),
            ("validation_protocol", "validation_protocol_digest"),
            ("candidate_protocol", "candidate_protocol_digest"),
            ("evaluation_protocol", "evaluation_protocol_digest"),
        ):
            _require_integrity(
                _canonical_digest(run[payload_name]) == run[digest_name],
                f"{payload_name} digest does not match its frozen payload",
            )

        matrix = config["phase7_3_3_1"]
        protocols = matrix["protocols"]
        learning = config["phase7_2"]["learning_protocol"]
        formal_training = config["phase7_3_3_2"]["formal_training"]
        early_stopping = config["phase7_3_3_2"]["early_stopping"]
        phase5_training = config["phase5"]["training"]
        architecture = run["architecture"]

        _require_integrity(
            run["data_digest"] == matrix["data"]["formal_tree_sha256"],
            "formal data digest differs from the frozen matrix",
        )
        data_path = project_root / str(matrix["data"]["source"])
        _require_integrity(
            data_path.is_dir(),
            "formal Phase 3 data artifact is unavailable",
        )
        _require_integrity(
            run["learning_protocol"]["objective_id"]
            == protocols["learning_objective"]
            == learning["id"],
            "learning_protocol objective differs from frozen config",
        )
        _require_integrity(
            run["learning_protocol"]["objective_source_sha256"]
            == protocols["learning_objective_source_sha256"],
            "learning_protocol source binding differs from frozen config",
        )
        _require_integrity(
            run["evaluation_protocol"]["protocol_id"]
            == protocols["evaluation"],
            "evaluation_protocol identity differs from frozen config",
        )
        _require_integrity(
            run["evaluation_protocol"]["source_sha256"]
            == protocols["evaluation_source_sha256"],
            "evaluation_protocol source binding differs from frozen config",
        )
        _require_integrity(
            run["validation_protocol"]["protocol_id"]
            == protocols["validation"],
            "validation_protocol identity differs from frozen config",
        )
        _require_integrity(
            run["candidate_protocol"]["training_candidate_source"]
            == protocols["candidate"]
            == config["phase7_2"]["candidate_universe"]["version"],
            "candidate_protocol identity differs from frozen config",
        )
        _require_integrity(
            run["optimizer_family"]
            == protocols["optimizer_family"]
            == formal_training["optimizer"],
            "optimizer family differs from frozen config",
        )
        _require_integrity(
            int(phase5_training["gradient_accumulation_steps"]) == 1
            and float(phase5_training["max_grad_norm"]) == 1.0
            and str(phase5_training["scheduler"]) == "none"
            and int(phase5_training["num_workers"]) == 0,
            "formal optimizer-step runtime differs from the frozen protocol",
        )
        _require_integrity(
            int(early_stopping["max_epochs"]) == 2
            and int(early_stopping["patience"]) == 10
            and float(early_stopping["min_delta"]) == 0.0
            and str(early_stopping["validation_cadence"])
            == "ONCE_PER_EPOCH",
            "formal early-stopping runtime differs from the frozen protocol",
        )
        from evi_hgt.protocols.formal_experiments import (
            PRACTICAL_PROTOCOL_SHA256, practical_training_protocol_from_config,
        )
        practical = practical_training_protocol_from_config(config)
        _require_integrity(
            _canonical_digest(practical) == PRACTICAL_PROTOCOL_SHA256
            and run.get("formal_training_protocol_digest") == PRACTICAL_PROTOCOL_SHA256
            and run.get("formal_training_protocol") == practical,
            "practical scientific training protocol/config binding differs",
        )

        objective_source = project_root / "src/evi_hgt/training/formal_objective.py"
        evaluation_source = project_root / "src/evi_hgt/evaluation/protocol.py"
        _require_integrity(
            objective_source.is_file()
            and _file_sha256(objective_source)
            == run["learning_protocol"]["objective_source_sha256"],
            "learning_protocol implementation digest mismatch",
        )
        _require_integrity(
            evaluation_source.is_file()
            and _file_sha256(evaluation_source)
            == run["evaluation_protocol"]["source_sha256"],
            "evaluation_protocol implementation digest mismatch",
        )

        for key in (
            "hidden_dim",
            "evidence_dim",
            "relation_dim",
            "provenance_dim",
            "num_heads",
            "num_layers",
            "ffn_multiplier",
            "dropout",
            "use_node_features",
            "use_node_type_embedding",
            "use_node_id_embedding",
            "path_hidden_dim",
            "path_num_heads",
            "path_num_layers",
            "path_max_length",
        ):
            _require_integrity(
                architecture[key] == matrix["architecture"][key],
                f"architecture field drifted: {key}",
            )
        _require_integrity(
            architecture["model_variant"] == run["model_variant"],
            "architecture model variant differs from run identity",
        )
        _require_integrity(
            architecture["variant_status"] == model["status"] == "IMPLEMENTED",
            "model variant is not available for formal execution",
        )
        phase6_paths = config["phase6"]["paths"]
        _require_integrity(
            phase6_paths["enabled"] is True
            and phase6_paths["algorithm"] == "bounded_deterministic_beam"
            and phase6_paths["algorithm_version"]
            == "bounded-deterministic-beam-v1"
            and phase6_paths["max_length"] == architecture["path_max_length"],
            "path extraction protocol differs from the frozen architecture",
        )

        for config_key, formal_key in (
            ("temperature", "temperature"),
            ("lambda_support", "lambda_support"),
            ("lambda_unsuitability", "lambda_unsuitability"),
            ("explicit_regularization_weight", "lambda_regularization"),
            ("pairwise_chunk_size", "pairwise_chunk_size"),
        ):
            _require_integrity(
                learning[config_key] == formal_training[formal_key],
                f"formal learning setting drifted: {config_key}",
            )
        _require_integrity(
            learning["requires_class_prior"] is False
            and learning["use_hard_negative_mining"] is False,
            "formal primary learning semantics drifted",
        )
        _require_integrity(
            run["policy"] == run["learning_protocol"]["primary_policy"],
            "run policy differs from learning protocol",
        )
        _require_integrity(
            isinstance(run["learning_rate"], (int, float))
            and float(run["learning_rate"]) > 0.0
            and run["learning_rate"] == run["learning_rate_binding"],
            "authorized formal train requires a resolved positive learning rate",
        )

        _require_integrity(
            set(validation_manifests) == {"support", "unsuitability"},
            "deterministic validation requires both channels",
        )
        validation_path = project_root / str(
            next(iter(validation_manifests.values()))["source_artifact"]
        )
        verified_manifests = load_validation_manifests(validation_path)
        for channel, binding in validation_manifests.items():
            _require_integrity(
                binding["channel"] == channel
                and binding["split_kind"] == run["split"]
                and binding["partition"] == "validation"
                and binding["objective_id"] == learning["id"]
                and binding["fixed_across_epochs"] is True
                and binding["used_for_training"] is False
                and binding["test_labels_used"] is False,
                f"deterministic validation binding drifted: {channel}",
            )
            _require_integrity(
                verified_manifests[(str(run["split"]), channel)].digest
                == binding["manifest_digest"],
                f"deterministic validation payload digest drifted: {channel}",
            )
    except FormalRunIntegrityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalRunIntegrityError(
            f"formal run integrity payload is invalid: {exc}"
        ) from exc


def prepare_formal_training_execution(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    run_id: str | None,
    resume: bool = False,
) -> FormalTrainingExecution:
    """Resolve, authorize, and lock exactly one PLANNED manifest run without writes."""

    enforce_phase8_execution_authorization(config, mode="train")
    if run_id is None:
        raise FormalRunSelectionError(
            "production formal train requires one exact run ID"
        )
    try:
        _validate_sha256(run_id, name="run_id")
    except ValueError as exc:
        raise FormalRunSelectionError(str(exc)) from exc

    root = Path(project_root).resolve()
    manifest = load_formal_experiment_manifest(root, run_id=run_id)
    run = resolve_formal_run(manifest, run_id=run_id)
    execution_status = str(run["execution_status"])
    if execution_status not in {"PLANNED", "INTERRUPTED", "REARMED"} and not (execution_status == 'RUNNING' and resume):
        raise FormalRunStateError(
            "formal train requires manifest status PLANNED or INTERRUPTED; "
            f"observed {execution_status}"
        )
    if resume and execution_status in {'PLANNED', 'REARMED'}:
        raise FormalRunStateError('RESUME_BLOCKED: fresh execution cannot be requested as resume')

    authorization = phase8_authorization_payload(config)
    if run_id not in authorization["authorized_run_ids"]:
        raise FormalRunAuthorizationError(
            f"formal run is not authorized by exact run ID: {run_id}"
        )

    from evi_hgt.training.trainer import validate_formal_learning_configuration

    phase7_2 = config["phase7_2"]
    learning = phase7_2["learning_protocol"]
    validate_formal_learning_configuration(
        formal_training_enabled=bool(
            phase7_2["training"]["formal_training_enabled"]
        ),
        phase8_execution_authorized=bool(
            phase7_2["training"]["phase8_execution_authorized"]
        ),
        use_node_id_embedding=bool(config["model"]["use_node_id_embedding"]),
        candidate_source=str(phase7_2["candidate_universe"]["source"]),
        objective_id=str(learning["id"]),
        requires_class_prior=bool(learning["requires_class_prior"]),
        hard_negative_mining=bool(learning["use_hard_negative_mining"]),
    )

    model = _load_variant_record(root, str(run["model_variant"]))
    validation_manifests = load_validation_manifest_bindings(
        root, split=str(run["split"])
    )
    _validate_formal_run_integrity(
        root,
        config,
        run,
        model=model,
        validation_manifests=validation_manifests,
    )
    manifest_run_count = len(manifest["runs"])
    return FormalTrainingExecution(
        project_root=root,
        config=config,
        config_digest=_canonical_digest(config),
        run=run,
        model=model,
        validation_manifests=validation_manifests,
        authorization=authorization,
        training_protocol=dict(config["phase7_3_3_2"]["formal_training"]),
        output_directory=(run['execution_attempts'][-1]['output_directory'] if run.get('execution_attempts') else formal_output_directory(
            str(run["run_family"]), str(run["run_id"])
        )),
        state_transition=FormalRunStateTransition(
            run_id=str(run["run_id"]), source_status=execution_status
        ),
        manifest_run_count=manifest_run_count,
        other_run_count=manifest_run_count - 1,
        resume_requested=resume,
    )


def build_formal_training_components(
    execution: FormalTrainingExecution,
    *,
    input_index: Any | None = None,
    enable_indexed_training: bool = True,
) -> FormalTrainingComponents:
    """Construct one run's inputs, with one attempt-owned training index.

    Engineering callers may supply the same already-built index or explicitly
    disable indexed training. Evaluation always uses the original assembler path.
    """

    from evi_hgt.models.path_reasoning import PathReasoningConfig
    from evi_hgt.sampling.path_extractor import PathCandidateExtractor
    from evi_hgt.sampling.path_schema import (
        PathExtractionConfig,
        pack_path_candidates,
    )
    from evi_hgt.training.batcher import Phase3QueryBatchAssembler
    from evi_hgt.training.candidate_universe import CandidateUniverse
    from evi_hgt.training.dataset import (
        make_formal_epoch_dataset,
        make_phase5_dataloader,
    )
    from evi_hgt.training.formal_objective import (
        DualChannelPairwiseRankingObjective,
    )
    from evi_hgt.training.model_variants import build_model_variant
    from evi_hgt.training.trainer import TrainingRuntimeConfig
    from evi_hgt.training.unlabeled_sampler import LazyUnlabeledSampler
    from evi_hgt.utils.device import select_device
    import torch

    config = execution.config
    run = execution.run
    subgraph = config["subgraph"]
    if not enable_indexed_training:
        if input_index is not None:
            raise ValueError('cannot supply an index while indexed training is disabled')
    elif input_index is None:
        from evi_hgt.training.input_index import StaticInputIndex
        input_index = StaticInputIndex.build(
            execution.project_root,
            token_buckets=int(config["model"]["property_token_buckets"]),
        )
    assembler = Phase3QueryBatchAssembler(
        execution.project_root,
        max_hops=int(subgraph["max_hops"]),
        max_nodes=int(subgraph["max_nodes"]),
        max_edges=int(subgraph["max_edges"]),
        token_buckets=int(config["model"]["property_token_buckets"]),
        input_index=input_index,
        cold_visibility_policy_digest=run.get('cold_visibility_policy_digest'),
    )
    phase6 = config["phase6"]
    phase6_paths = phase6["paths"]
    architecture = run["architecture"]
    path_config = None
    if bool(architecture["use_path_reasoning"]):
        path_config = PathReasoningConfig(
            core_hidden_dim=int(architecture["hidden_dim"]),
            path_hidden_dim=int(architecture["path_hidden_dim"]),
            num_heads=int(architecture["path_num_heads"]),
            num_layers=int(architecture["path_num_layers"]),
            dropout=float(phase6["path_encoder"]["dropout"]),
            max_length=int(architecture["path_max_length"]),
            ffn_multiplier=int(phase6["path_encoder"]["ffn_multiplier"]),
        )
    built = build_model_variant(
        str(run["model_variant"]),
        base_config=assembler.config,
        path_config=path_config,
        dropout=float(architecture["dropout"]),
    )

    device_config = config["device"]
    selection = select_device(
        str(device_config["device"]),
        mixed_precision=bool(device_config["mixed_precision"]),
    )
    if selection.device.type == "cuda" and selection.precision_name != "bf16":
        raise FormalRunIntegrityError(
            "formal CUDA execution requires measured BF16 support"
        )
    built.model.to(selection.device)

    actual_total = sum(parameter.numel() for parameter in built.model.parameters())
    actual_trainable = sum(
        parameter.numel()
        for parameter in built.model.parameters()
        if parameter.requires_grad
    )
    _require_integrity(
        actual_total == execution.model["total_parameters"]
        and actual_trainable == execution.model["trainable_parameters"],
        "constructed model parameter counts differ from the frozen catalog",
    )

    phase3_root = execution.project_root / str(
        config["phase7_3_3_1"]["data"]["source"]
    )
    universe = CandidateUniverse.from_phase3(
        phase3_root, split_kind=str(run["split"])
    )
    sampler = LazyUnlabeledSampler(universe, seed=int(run["seed"]))
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(int(run["seed"]))
    path_extractor = PathCandidateExtractor(
        PathExtractionConfig(
            min_length=int(phase6_paths["min_length"]),
            max_length=int(phase6_paths["max_length"]),
            max_paths=int(phase6_paths["max_paths"]),
            beam_width=int(phase6_paths["beam_width"]),
            max_neighbors_per_expansion=int(
                phase6_paths["max_neighbors_per_expansion"]
            ),
            use_hub_penalty=bool(phase6_paths["use_hub_penalty"]),
            hub_penalty_weight=float(phase6_paths["hub_penalty_weight"]),
        )
    )
    learning = config["phase7_2"]["learning_protocol"]
    objective = DualChannelPairwiseRankingObjective(
        temperature=float(learning["temperature"]),
        lambda_support=float(learning["lambda_support"]),
        lambda_unsuitability=float(learning["lambda_unsuitability"]),
        explicit_regularization_weight=float(
            learning["explicit_regularization_weight"]
        ),
        pairwise_chunk_size=int(learning["pairwise_chunk_size"]),
    )
    phase5_training = config["phase5"]["training"]
    runtime_config = TrainingRuntimeConfig(
        device=selection.selected_device,
        precision="bf16" if selection.device.type == "cuda" else "fp32",
        learning_rate=float(run["learning_rate"]),
        weight_decay=float(execution.training_protocol["weight_decay"]),
        gradient_accumulation_steps=int(
            phase5_training["gradient_accumulation_steps"]
        ),
        max_grad_norm=float(phase5_training["max_grad_norm"]),
        num_workers=int(phase5_training["num_workers"]),
        scheduler=str(phase5_training["scheduler"]),
    )
    return FormalTrainingComponents(
        assembler=assembler,
        candidate_universe=universe,
        unlabeled_sampler=sampler,
        dataset_factory=partial(
            make_formal_epoch_dataset,
            unlabeled_sampler=sampler,
        ),
        dataloader_factory=partial(
            make_phase5_dataloader,
            generator=dataloader_generator,
        ),
        path_extractor=path_extractor,
        path_packer=pack_path_candidates,
        model_variant=built,
        objective=objective,
        runtime_config=runtime_config,
        device=selection.selected_device,
        dataloader_generator=dataloader_generator,
    )


def invoke_formal_learning_trainer(
    execution: FormalTrainingExecution,
    *,
    component_builder: Callable[
        [FormalTrainingExecution], FormalTrainingComponents
    ] = build_formal_training_components,
    trainer_factory: Callable[..., object] | None = None,
    checkpoint_directory: str | Path | None = None,
) -> FormalTrainingRuntime:
    """Preflight checkpoint state, bind the trainer, and deliberately take no step."""

    from evi_hgt.training.formal_checkpoint import (
        RESUME_ALLOWED,
        FormalCheckpointLifecycle,
        formal_checkpoint_identity_from_execution,
        inspect_formal_checkpoint_state,
        restore_formal_checkpoint,
    )
    from evi_hgt.training.model_selection import EarlyStoppingState

    checkpoint_identity = formal_checkpoint_identity_from_execution(execution)
    checkpoint_root = (
        execution.project_root / execution.output_directory / "checkpoints"
        if checkpoint_directory is None
        else Path(checkpoint_directory)
    )
    checkpoint_decision = inspect_formal_checkpoint_state(
        identity=checkpoint_identity,
        run_status=str(execution.run["execution_status"]),
        checkpoint_directory=checkpoint_root,
        map_location="cpu",
    )
    from evi_hgt.utils.seed import set_global_seed

    set_global_seed(int(execution.run["seed"]))

    if trainer_factory is None:
        from evi_hgt.training.trainer import FormalLearningTrainer

        trainer_factory = FormalLearningTrainer
    components = component_builder(execution)
    training = execution.config["phase7_2"]["training"]
    trainer = trainer_factory(
        components.model_variant.model,
        components.objective,
        components.runtime_config,
        synthetic_mode=False,
        formal_training_enabled=bool(training["formal_training_enabled"]),
        phase8_execution_authorized=bool(training["phase8_execution_authorized"]),
        use_node_id_embedding=bool(
            execution.config["model"]["use_node_id_embedding"]
        ),
        candidate_source=str(
            execution.config["phase7_2"]["candidate_universe"]["source"]
        ),
        formal_run_id=execution.run_id,
    )
    if checkpoint_decision.action == RESUME_ALLOWED:
        assert checkpoint_decision.inspection is not None
        restored_snapshot = restore_formal_checkpoint(
            checkpoint_decision.inspection,
            model=components.model_variant.model,
            optimizer=trainer.optimizer,
            trainer=trainer,
            sampler=components.unlabeled_sampler,
            dataloader_generator=components.dataloader_generator,
            amp_scaler=None,
        )
        early_stopping_state = restored_snapshot.early_stopping_state
    else:
        early_stopping_state = EarlyStoppingState()
    runtime_payload = (
        components.runtime_config.to_dict()
        if hasattr(components.runtime_config, "to_dict")
        else dict(components.runtime_config)
    )
    checkpoint_lifecycle = FormalCheckpointLifecycle(
        identity=checkpoint_identity,
        paths=checkpoint_decision.paths,
        trainer=trainer,
        runtime_config=runtime_payload,
        candidate_universe_version=str(
            execution.run["candidate_protocol"]["training_candidate_source"]
        ),
        sampler=components.unlabeled_sampler,
        dataloader_generator=components.dataloader_generator,
        amp_scaler=None,
        early_stopping_state=early_stopping_state,
        last_snapshot=checkpoint_decision.snapshot,
        best_artifact=(restored_snapshot.best_artifact if checkpoint_decision.action == RESUME_ALLOWED else None),
    )
    return FormalTrainingRuntime(
        execution=execution,
        components=components,
        trainer=trainer,
        checkpoint_decision=checkpoint_decision,
        checkpoint_lifecycle=checkpoint_lifecycle,
    )


def execute_formal_training(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    run_id: str | None,
    runner: Callable[[FormalTrainingExecution], object] | None = None,
    resume: bool = False,
    progress: str = 'auto',
) -> object:
    """Enter exactly one protected formal path and delegate one locked context."""

    from dataclasses import replace
    if progress not in {'auto', 'on', 'off'}:
        raise ValueError('progress must be auto, on or off')
    execution = prepare_formal_training_execution(project_root, config, run_id=run_id, **({'resume': True} if resume else {}))
    if progress != 'auto':
        execution = replace(execution, progress_mode=progress)
    if runner is None:
        from evi_hgt.training.formal_driver import run_formal_training

        selected_runner = run_formal_training
    else:
        selected_runner = runner
    if runner is not None:
        return selected_runner(execution)
    with formal_execution_lease(execution.project_root / execution.output_directory):
        return selected_runner(execution)


@contextmanager
def formal_execution_lease(directory: str | Path):
    """Kernel-held lease is released on process exit, including hard termination."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.execution.lock').open('a+b') as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('formal execution is already running: lease unavailable') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_formal_dry_run_plan(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    mode: str,
    run_id: str | None = None,
    experiment_family: str | None = None,
    model_variant: str | None = None,
    policy: str | None = None,
    split: str | None = None,
    seed: int | None = None,
    learning_rate: float | str | None = None,
) -> dict[str, Any]:
    """Resolve a complete immutable plan and perform no execution or writes."""

    if mode not in _FORMAL_MODES:
        raise ValueError("dry-run is supported only for train, evaluate, and full")
    root = Path(project_root).resolve()
    manifest = load_formal_experiment_manifest(root, run_id=run_id)
    run = resolve_formal_run(
        manifest,
        run_id=run_id,
        experiment_family=experiment_family,
        model_variant=model_variant,
        policy=policy,
        split=split,
        seed=seed,
        learning_rate=learning_rate,
    )
    model = _load_variant_record(root, str(run["model_variant"]))
    validation_manifests = load_validation_manifest_bindings(
        root, split=str(run["split"])
    )
    phase = config["phase7_3_3_2"]
    authorization = phase8_authorization_payload(config)
    run_output = (run['execution_attempts'][-1].get('output_directory') if run.get('execution_attempts') else None) or formal_output_directory(str(run["run_family"]), str(run["run_id"]))
    evaluation_output = f"outputs/phase8_formal/evaluation/{run['run_id']}"
    output_directory = evaluation_output if mode == "evaluate" else run_output
    data_relative_path = str(config["phase7_3_3_1"]["data"]["source"])
    protocol_digests = {
        "formal_training": run["formal_training_protocol_digest"],
        "learning": run["learning_protocol_digest"],
        "validation": run["validation_protocol_digest"],
        "candidate": run["candidate_protocol_digest"],
        "evaluation": run["evaluation_protocol_digest"],
        "architecture_config": run["architecture_digest"],
    }
    checkpoint_selection = dict(phase["checkpoint_selection"])
    return {
        "protocol_id": "EVIHGT_FORMAL_CLI_DRY_RUN_V1",
        "dry_run": True,
        "mode": mode,
        "resolved_run": run,
        "model": model,
        "data_artifact": {
            "relative_path": data_relative_path,
            "exists": (root / data_relative_path).is_dir(),
            "data_digest": run["data_digest"],
        },
        "config_digest": _canonical_digest(config),
        "protocol_digests": protocol_digests,
        "learning_protocol": run["learning_protocol"],
        "training_protocol": dict(phase["formal_training"]),
        "validation_manifests": validation_manifests,
        "candidate_protocol": run["candidate_protocol"],
        "evaluation_protocol": run["evaluation_protocol"],
        "output_directory": output_directory,
        "run_output_directory": run_output,
        "evaluation_output_directory": evaluation_output,
        "checkpoint_rule": checkpoint_selection,
        "future_checkpoint": {
            "required_for_formal_test": "BEST",
            "resolved": False,
            "digest": None,
            "reason": "NO_PHASE8_CHECKPOINT_EXISTS_DURING_DRY_RUN",
        },
        "environment_snapshot_fields": list(ENVIRONMENT_SNAPSHOT_FIELDS),
        "authorization": authorization,
        "planned_training_mode": (
            "FORMAL_TRAIN" if mode in {"train", "full"} else "NONE"
        ),
        "planned_evaluation_mode": (
            "FORMAL_TEST_AFTER_BEST_CHECKPOINT"
            if mode in {"evaluate", "full"}
            else "NONE"
        ),
        "would_execute_training": False,
        "would_execute_test": False,
        "would_create_optimizer": False,
        "would_mutate_checkpoint": False,
        "real_optimizer_step_count": 0,
        "formal_result": False,
    }


__all__ = [
    "ENVIRONMENT_SNAPSHOT_FIELDS",
    "EvaluationIdentity",
    "FormalRunAuthorizationError",
    "FormalRunIntegrityError",
    "FormalRunSelectionError",
    "FormalRunStateError",
    "FormalRunStateTransition",
    "FormalTrainingComponents",
    "FormalTrainingExecution",
    "FormalTrainingRuntime",
    "PHASE8_EXECUTION_AUTHORIZATION_REQUIRED",
    "build_formal_training_components",
    "build_formal_dry_run_plan",
    "enforce_phase8_execution_authorization",
    "execute_formal_training",
    "formal_output_directory",
    "invoke_formal_learning_trainer",
    "phase8_authorization_payload",
    "prepare_formal_training_execution",
]
