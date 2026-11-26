"""Train one portable ablation arm on the fixed three-cycle schedule."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from evi_hgt.console import Console
from evi_hgt.io import digest, write_json
from evi_hgt.model import VARIANTS, build_model
from evi_hgt.reporting import create_report
from evi_hgt.runner import (
    deterministic, environment, evaluate, load_inputs, precision_context,
    restore_rng, save_checkpoint, update_once,
)
from evi_hgt.settings import Settings
from evi_hgt.training.supervision import make_unlabeled_query_supervision

from .ablation_diagnostics import write_exposure
from .ablation_losses import sigmoid_group_risk


def evaluation_steps(groups):
    return 3 * groups // 2, 3 * groups


def _sigmoid_update(built, corpus, spec, optimizer, settings):
    queries = tuple(spec.queries)
    extra = () if spec.record is None else tuple(
        make_unlabeled_query_supervision(*pair) for pair in spec.record["extra_pairs"])
    graphs = tuple(corpus.assembler.assemble(query, domain="training", device=settings.device)
                   for query in queries + extra)
    optimizer.zero_grad(set_to_none=True)
    with deterministic(), precision_context(settings):
        predictions = tuple(built.forward(graph.batch.model_batch, path_batch=graph.batch.path_batch)
                            for graph in graphs)
        for prediction in predictions:
            if not (torch.isfinite(prediction.support_logit).all()
                    and torch.isfinite(prediction.unsuitability_logit).all()):
                raise FloatingPointError("Nonfinite independent-head prediction")
        risk = sigmoid_group_risk(queries, predictions[:len(queries)], record=spec.record,
                                  extra_predictions=predictions[len(queries):])
    with deterministic():
        risk.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(built.model.parameters(), settings.max_grad_norm,
                                              error_if_nonfinite=True)
        optimizer.step()
    return dict(loss=float(risk.total.detach()), support_loss=float(risk.support.detach()),
                contra_loss=float(risk.contra.detach()), gradient_norm=float(norm), queries=len(graphs))


def _restore(checkpoint, binding, built, optimizer, total_steps):
    if checkpoint is None:
        return 0
    # Only use checkpoints written locally by this experiment workflow.
    saved = torch.load(Path(checkpoint).resolve(strict=True), map_location="cpu", weights_only=False)
    if saved.get("schema") != "EVIHGT_CHECKPOINT_V1" or saved.get("binding") != binding:
        raise ValueError("Ablation checkpoint variant, loss, corpus or training settings differ")
    step = saved.get("step")
    if type(step) is not int or not 0 <= step <= total_steps:
        raise ValueError("Ablation checkpoint step is outside the fixed plan")
    built.model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    return step


def _run(settings, variant, loss, checkpoint, smoke):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    root = settings.resolve(settings.output_dir) / f"{tag}_{variant}_{loss}"
    root.mkdir(parents=True, exist_ok=False)
    with Console(root / "logs", refresh_seconds=settings.refresh_seconds) as console:
        console.banner()
        if smoke:
            console.say("[SYNTHETIC_TEST_ONLY] 小型 CPU 消融流程验证")
        started = time.monotonic()
        try:
            with console.phase(1, "校验消融训练语料"):
                corpus = load_inputs(settings, console)
                groups, total = corpus.plan.G, corpus.plan.total_steps
                if total != 3 * groups or (smoke and total != 3):
                    raise ValueError("The supplied ablation plan must have three complete cycles")
                planned_evaluations = evaluation_steps(groups)
                write_json(root / "configuration.json", {**settings.as_dict(), "loss": loss,
                                                         "evaluation_steps": planned_evaluations})
                write_json(root / "environment.json", environment())
            with console.phase(2, "按共同随机种子初始化模型"):
                if settings.device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable. Select an explicit CPU/FP32 configuration to run on CPU.")
                if settings.precision == "bf16" and not torch.cuda.is_bf16_supported():
                    raise RuntimeError("The selected GPU does not support BF16")
                random.seed(corpus.training_seed)
                np.random.seed(corpus.training_seed % 2**32)
                torch.manual_seed(corpus.training_seed)
                built = build_model(corpus.model_config, pretrained=corpus.pretrained,
                                    seed=corpus.training_seed, variant=variant)
                built.model.to(settings.device).train()
                optimizer = torch.optim.AdamW(built.model.parameters(), lr=settings.learning_rate,
                    weight_decay=settings.weight_decay, betas=(0.9, 0.999), eps=1e-8, foreach=None, fused=None)
                metadata = built.metadata()
                binding = dict(protocol="PAIRED_DEVELOPMENT_ABLATION", corpus=corpus.identity,
                    plan=corpus.plan.identity, variant=variant, loss=loss, model=metadata,
                    settings={key: getattr(settings, key) for key in (
                        "device", "precision", "learning_rate", "weight_decay", "max_grad_norm")})
                initial = _restore(checkpoint, binding, built, optimizer, total)
                write_json(root / "model.json", metadata)
                console.say(f"[模型] {variant} / {loss}，{settings.device}/{settings.precision}")
                if checkpoint is not None:
                    console.say(f"[恢复] 已从第 {initial}/{total} 次更新加载模型与优化器。")

            evaluated = []
            latest = root / "checkpoints/latest.pt"
            update = update_once if loss == "softplus" else _sigmoid_update

            def score(step):
                evaluate(built, corpus, settings, console, root / f"evaluation/step_{step}", step)
                evaluated.append(step)

            with console.phase(3, "执行三周期训练与固定步点评估"):
                if initial in planned_evaluations:
                    score(initial)
                for cycle in range(initial // groups + 1, 4):
                    start = max(initial, (cycle - 1) * groups)
                    end = cycle * groups
                    with console.progress(end - start, f"Epoch {cycle}/3", unit="update") as bar:
                        for step in range(start + 1, end + 1):
                            spec = corpus.plan.step(step)
                            row = update(built, corpus, spec, optimizer, settings)
                            extra_count = row["queries"] - len(spec.queries)
                            console.event("optimizer_update", step=step, cycle=cycle,
                                stage="GLOBAL" if spec.record is None else "CONDITIONAL",
                                variant=variant, loss_id=loss, training_member_sha256=spec.identity["member_digest"],
                                record_sha256=None if spec.record is None else digest(spec.record),
                                sampling_sha256=spec.identity["sampling"], original_queries=len(spec.queries),
                                conditional_extra_U_queries=extra_count,
                                support_pairs=sum(query.support.observed_mask for query in spec.queries),
                                contra_pairs=sum(query.unsuitability.observed_mask for query in spec.queries), **row)
                            memory = torch.cuda.memory_allocated() / 2**30 if settings.device == "cuda" else 0.0
                            bar.update(loss=row["loss"], GPU_GiB=memory)
                            if step % settings.checkpoint_every == 0 or step in planned_evaluations or step == end:
                                save_checkpoint(latest, built, optimizer, step, binding)
                            if step in planned_evaluations:
                                if step != total:
                                    partial = latest.with_name(f"step_{step}.pt.partial")
                                    shutil.copyfile(latest, partial)
                                    os.replace(partial, latest.with_name(f"step_{step}.pt"))
                                score(step)
                if not latest.exists():
                    save_checkpoint(latest, built, optimizer, total, binding)

            with console.phase(4, "统计固定训练面板中的证据与路径覆盖"):
                write_exposure(corpus, root / "diagnostics", step=total, variant=variant, loss=loss)
            diagnostics = "diagnostics/exposure.json"
            with console.phase(5, "汇总双通道排序与路径解释"):
                summary = create_report(built, corpus, settings, console, root, total, latest)
                write_json(root / "summary.json", {**summary, "loss": loss, "plan_identity": corpus.plan.identity,
                                                   "diagnostics": diagnostics})
            scope = "SYNTHETIC_TEST_ONLY" if smoke else "LOCAL_DEVELOPMENT_PANEL"
            write_json(root / "completion.json", dict(status="COMPLETE", mode="train", scope=scope,
                variant=variant, loss=loss, updates=total, total_updates=total, resumed_from_step=initial,
                updates_this_invocation=total-initial, elapsed_seconds=time.monotonic()-started,
                corpus_identity=corpus.identity, plan_identity=corpus.plan.identity,
                evaluation_steps=evaluated, planned_evaluation_steps=list(planned_evaluations),
                checkpoint="checkpoints/latest.pt", metrics=f"evaluation/step_{total}/metrics.json",
                report="report.html", summary="summary.json", explanations="explanations/index.json",
                diagnostics=diagnostics))
            return root
        except Exception as exc:
            console.failure(exc)
            exc.evihgt_reported = True
            raise


def run_ablation(settings: Settings, *, variant="FULL", loss="softplus", checkpoint=None, smoke=False) -> Path:
    """Run a fresh or locally resumed arm, with an explicitly bounded CPU test mode."""
    if variant not in VARIANTS or loss not in ("softplus", "sigmoid"):
        raise ValueError("Unknown portable ablation variant or loss")
    manifest = json.loads(settings.resolve(settings.manifest).read_text(encoding="utf-8"))
    synthetic = manifest.get("purpose") == "SYNTHETIC_TEST_ONLY"
    if synthetic != smoke:
        raise ValueError("Use smoke mode only with an explicitly synthetic corpus")
    if not smoke:
        return _run(settings, variant, loss, checkpoint, smoke)
    if (settings.device, settings.precision) != ("cpu", "fp32") or manifest["schedule"]["G"] != 1:
        raise ValueError("Synthetic ablation smoke requires CPU/FP32 and three optimizer updates")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return _run(settings, variant, loss, checkpoint, smoke)
    finally:
        torch.set_num_threads(previous_threads)
