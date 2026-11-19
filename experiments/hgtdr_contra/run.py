"""Run the HGTDR contraindication baseline on the original R0 folds."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from filelock import FileLock, Timeout

METHOD = "HGTDR_ADAPTED_CONTRA"
SCOPE = "R0_INNER_DEVELOPMENT"
FOLDS = ("R0F1", "R0F0", "R0F2", "R0F3", "R0F4")
CHECKPOINT_SCHEMA = "HGTDR_CONTRA_EPOCH_CHECKPOINT_V1"


def add_research_source(research_root: Path) -> Path:
    research_root = research_root.resolve()
    source = research_root / "src"
    required = (
        source / "evi_hgt/phase9/hgtdr_adapter.py",
        source / "evi_hgt_phase98/data.py",
        research_root.parent / "HGTDR/src/competition.py",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(
            "Use --research-root with the original EviHGT research checkout and its sibling HGTDR reconstruction."
        )
    sys.path.insert(0, str(source))
    return research_root


def save_checkpoint(path: Path, payload: dict) -> None:
    partial = path.with_name(f"{path.name}.{uuid4().hex}.partial")
    try:
        with partial.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def load_checkpoint(path: Path, binding: dict) -> dict | None:
    if not path.exists():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["binding"] != binding:
        raise ValueError("Checkpoint binding differs from the current fold or protocol")
    return saved


def publish_json(path: Path, value: dict) -> None:
    from evi_hgt_phase98.common import read, write_new

    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing artifact differs: {path}")
    else:
        write_new(path, value)


def completed_fold(folder: Path, protocol: dict) -> dict | None:
    from evi_hgt_phase98.common import read, sha

    path = folder / "COMPLETE.json"
    if not path.exists():
        return None
    record = read(path)
    if record["status"] != "COMPLETE" or record["binding"]["protocol"] != protocol:
        raise ValueError(f"Completed fold protocol differs: {path}")
    if record["binding"]["cell"] != folder.name:
        raise ValueError(f"Completed fold identity differs: {path}")
    for file_name, expected in record["artifacts_sha256"].items():
        if sha(folder / file_name) != expected:
            raise ValueError(f"Completed artifact hash differs: {folder / file_name}")
    return record


def restore_epoch(trainer, saved: dict) -> None:
    from evi_hgt.phase9.production_state import _validate_rng

    if saved["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("Unknown epoch checkpoint schema")
    trainer.model.load_state_dict(saved["model"], strict=True)
    trainer.optimizer.load_state_dict(saved["optimizer"])
    trainer.scheduler.load_state_dict(saved["scheduler"])
    trainer.load_control_state(saved["control"])
    completed = saved["completed_epoch"]
    if trainer.cursor["epoch"] != completed + 1 or trainer.selection.opportunities != completed:
        raise ValueError("Training and selection state do not describe the same completed epoch")
    random.setstate(saved["rng"]["python"])
    np.random.set_state(_validate_rng(saved["rng"]))
    torch.set_rng_state(saved["rng"]["cpu"])
    if saved["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(saved["rng"]["cuda"])


def run_fold(cell, folder, protocol, inputs):
    from evi_hgt.phase9.production_state import _rng, _snapshot, training_state_digest
    from evi_hgt_phase98.common import append, sha

    from .data import load_contra_fold
    from .evaluation import checkpoint_key, score_panel
    from .trainer import build_trainer

    folder.mkdir(parents=True, exist_ok=True)
    complete = completed_fold(folder, protocol)
    if complete is not None:
        print(f"{cell}: verified existing results", flush=True)
        return complete

    events = folder / "events.jsonl"
    append(events, dict(event="PREPARING", method=METHOD, cell=cell))
    print(f"{cell}: checking fold inputs and graph masks", flush=True)
    fold = load_contra_fold(cell, inputs)
    trainer = build_trainer(fold, device="cuda")
    epochs = trainer.config["epochs"]
    updates_per_epoch = math.ceil(len(fold.native.supports) / (trainer.config["batch_size"] // 2))
    binding = dict(protocol=protocol, cell=cell, data=fold.binding, trainer=trainer.metadata())
    publish_json(folder / "binding.json", binding)
    latest = folder / "latest.pt"
    saved = load_checkpoint(latest, binding)
    best_evaluation = None
    if saved is not None:
        restore_epoch(trainer, saved)
        best_evaluation = saved["best_evaluation"]
    start_epoch = trainer.cursor["epoch"]
    append(events, dict(event="READY", cell=cell, start_epoch=start_epoch, epochs=epochs,
                        train_positive_pairs=len(fold.native.supports), eval_pairs=len(fold.evaluation_rows),
                        updates_per_epoch=updates_per_epoch))
    print(f"{cell}: starting epoch {start_epoch} of {epochs}", flush=True)

    started = time.monotonic()
    reported_progress = started - 5
    for epoch in range(start_epoch, epochs + 1):
        while trainer.cursor["epoch"] == epoch:
            row = trainer.step()
            append(events, dict(event="UPDATE", cell=cell, **row))
            now = time.monotonic()
            if now - reported_progress >= 5:
                memory = torch.cuda.max_memory_allocated() / 2**30
                print(f"{cell} epoch {epoch}/{epochs} · update {row['optimizer_steps']} · "
                      f"loss {row['loss']:.6g} · GPU {memory:.3f} GiB", flush=True)
                reported_progress = now

        evaluation = score_panel(trainer, fold)
        metrics = evaluation["metrics"]
        trainer.selection.observe(epoch, metrics)
        if trainer.selection.best["epoch"] == epoch:
            trainer.best_model = _snapshot(trainer.model.state_dict())
            best_evaluation = evaluation
        save_checkpoint(latest, dict(
            schema=CHECKPOINT_SCHEMA, binding=binding, completed_epoch=epoch,
            control=trainer.control_state(), best_evaluation=best_evaluation,
            model=_snapshot(trainer.model.state_dict()),
            optimizer=_snapshot(trainer.optimizer.state_dict()),
            scheduler=trainer.scheduler.state_dict(), rng=_rng(),
        ))
        append(events, dict(
            event="EPOCH_COMMITTED", cell=cell, epoch=epoch,
            optimizer_updates=trainer.cursor["optimizer_steps"], selected_epoch=trainer.selection.best["epoch"],
            validation_pairs=metrics["pairs"], macro_auc=metrics["macro_auc"], macro_ap=metrics["macro_ap"],
            evaluation_seconds=trainer.actual_evaluation_seconds, training_seconds=trainer.actual_training_seconds,
        ))
        eta = (time.monotonic() - started) / (epoch - start_epoch + 1) * (epochs - epoch)
        print(f"{cell} epoch {epoch}/{epochs} saved · ROC AUC {metrics['macro_auc']:.6f} · "
              f"AP {metrics['macro_ap']:.6f} · NDCG@10 {metrics['ranking']['NDCG@10']:.6f} · "
              f"selected epoch {trainer.selection.best['epoch']} · {eta / 3600:.2f} h remaining", flush=True)
        if epoch == 10:
            if trainer.cursor["optimizer_steps"] != 10 * updates_per_epoch or trainer.selection.opportunities != 10:
                raise ValueError("Ten-epoch qualification update/selection count differs")
            append(events, dict(event="TEN_EPOCH_CHECK_PASSED", cell=cell, validation_pairs=metrics["pairs"]))

    if trainer.cursor["optimizer_steps"] != epochs * updates_per_epoch or trainer.selection.opportunities != epochs:
        raise ValueError("Incomplete fixed training or selection budget")
    best_entry = max(trainer.selection.history, key=lambda item: checkpoint_key(item["epoch"], item["metrics"]))
    if trainer.selection.best != best_entry:
        raise ValueError("Selected checkpoint differs from the development history")
    if best_evaluation is None or trainer.best_model is None:
        raise ValueError("Missing evaluated selected weights")
    selected = folder / "selected.weights.pt"
    selected_digest = training_state_digest(trainer.best_model)
    if selected.exists():
        if training_state_digest(torch.load(selected, weights_only=True, map_location="cpu")) != selected_digest:
            raise ValueError("Selected weights differ from the committed best checkpoint")
    else:
        save_checkpoint(selected, trainer.best_model)
    best_path = folder / "selected_evaluation.json"
    publish_json(best_path, dict(
        binding=binding, selected_epoch=trainer.selection.best["epoch"],
        weights_state_sha256=selected_digest, **best_evaluation,
    ))
    record = dict(
        status="COMPLETE", binding=binding, epochs=epochs, updates=trainer.cursor["optimizer_steps"],
        selected_epoch=trainer.selection.best["epoch"], metrics=best_evaluation["metrics"],
        selection_history=trainer.selection.history,
        artifacts_sha256={p.name: sha(p) for p in (selected, best_path)},
    )
    publish_json(folder / "COMPLETE.json", record)
    append(events, dict(event="COMPLETE", cell=cell, selected_epoch=record["selected_epoch"]))
    return record


def protocol_binding() -> dict:
    from evi_hgt_phase98.common import sha

    package = Path(__file__).resolve().parent
    sources = {path.name: sha(path) for path in sorted(package.glob("*.py"))}
    return dict(
        method=METHOD, scope=SCOPE, folds=list(FOLDS), training_target="CONTRAINDICATION",
        unlabeled_role="U", U_is_confirmed_negative=False, epochs=300,
        selection="CONTRA_INNER_MACRO_ROC_AUC_THEN_AP_THEN_EARLIER_EPOCH",
        source_sha256=sources,
    )


def run(output: Path) -> None:
    from evi_hgt_phase98.common import append
    from evi_hgt_phase98.data import TrainingInputs
    from .evaluation import aggregate_fold_results

    if not torch.cuda.is_available():
        raise RuntimeError("HGTDR training requires a CUDA device")
    torch.set_num_threads(4)
    protocol = protocol_binding()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "run.lock", timeout=0):
        publish_json(output / "protocol.json", protocol)
        append(output / "queue.jsonl", dict(event="START", pid=os.getpid(), folds=list(FOLDS)))
        inputs = TrainingInputs()
        metrics = {}
        for cell in FOLDS:
            record = run_fold(cell, output / cell, protocol, inputs)
            metrics[cell] = record["metrics"]
            append(output / "queue.jsonl", dict(event="FOLD_COMPLETE", cell=cell))
        aggregate = aggregate_fold_results(metrics)
        publish_json(output / "R0_contra_summary.json", dict(protocol=protocol, metrics=aggregate))
        append(output / "queue.jsonl", dict(event="COMPLETE", folds=list(FOLDS)))
        print(f"Completed all five folds. Summary: {output / 'R0_contra_summary.json'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run HGTDR on the frozen R0 contraindication folds")
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--research-root", type=Path, required=True, help="Original EviHGT research checkout")
    parser.add_argument("--output", type=Path, help="Fresh output directory or a compatible checkpoint directory")
    args = parser.parse_args(argv)
    try:
        research_root = add_research_source(args.research_root)
    except FileNotFoundError as error:
        parser.error(str(error))
    output_root = (research_root / "outputs").resolve()
    output = args.output.resolve() if args.output else output_root / "hgtdr_contra"
    if output == output_root or not output.is_relative_to(output_root):
        parser.error("--output must be a run directory under the research checkout outputs directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        run(output)
    except Timeout:
        print(f"A run already holds the lock in {output}", file=sys.stderr, flush=True)
        return 1
    except ModuleNotFoundError as error:
        print(f"Install or restore the research dependency {error.name} before running HGTDR.", file=sys.stderr)
        return 1
    except Exception:
        output.mkdir(parents=True, exist_ok=True)
        error_log = output / "error.log"
        with error_log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        print(f"The run stopped. Details: {error_log}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
