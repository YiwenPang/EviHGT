"""Local single-model training and development evaluation from an explicit corpus.

No benchmark baseline, download, outer-test or final-holdout evaluator is included.
"""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import polars as pl
import torch

from .console import Console
from .io import digest, write_json
from .losses import group_risk
from .metrics import metric_pair
from .model import build_model
from .reporting import create_report
from .training.supervision import make_unlabeled_query_supervision


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def deterministic():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


def precision_context(settings):
    return torch.autocast("cuda", dtype=torch.bfloat16) if settings.precision == "bf16" else nullcontext()


def environment():
    packages = {}
    for package in ("torch", "numpy", "polars", "pyarrow", "PyYAML", "tqdm"):
        packages[package] = importlib.metadata.version(package)
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def load_inputs(settings, console):
    from .data.corpus import load_corpus
    manifest = settings.resolve(settings.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Prepared corpus not found: {manifest}. Obtain the required datasets and follow "
            "docs/data.md, then run main.py --mode prepare, or start main.py to prepare and train."
        )
    with console.stage_progress() as progress:
        corpus = load_corpus(manifest, progress=progress)
    console.say(f"[数据] 每周期 {corpus.plan.G} 组，共 {corpus.plan.total_steps} 次计划更新。")
    return corpus


def update_once(built, corpus, spec, optimizer, settings):
    queries = tuple(spec.queries)
    extra = () if spec.record is None else tuple(
        make_unlabeled_query_supervision(*pair) for pair in spec.record["extra_pairs"])
    graphs = tuple(corpus.assembler.assemble(q, domain="training", device=settings.device)
                   for q in queries + extra)
    optimizer.zero_grad(set_to_none=True)
    with deterministic(), precision_context(settings):
        predictions = tuple(built.forward(g.batch.model_batch, path_batch=g.batch.path_batch) for g in graphs)
        for prediction in predictions:
            if not (torch.isfinite(prediction.support_logit).all() and torch.isfinite(prediction.unsuitability_logit).all()):
                raise FloatingPointError("Nonfinite independent-head prediction")
        risk = group_risk(queries, predictions[:len(queries)], record=spec.record,
                          extra_predictions=predictions[len(queries):])
        # Loss reductions are FP32, regardless of forward precision.
        # Backward is outside autocast in the frozen recipe.
    with deterministic():
        risk.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(built.model.parameters(), settings.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
    return dict(loss=float(risk.total.detach()), support_loss=float(risk.support.detach()),
                contra_loss=float(risk.contra.detach()), gradient_norm=float(norm), queries=len(graphs))


def evaluate(built, corpus, settings, console, target, step):
    """Only explicit development panels; preserve training mode and RNG state."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    state = rng_state()
    modes = {name: module.training for name, module in built.model.named_modules()}
    scores = {}
    try:
        built.model.eval()
        for domain in ("validation", "auxiliary_validation"):
            frame = corpus.panels.get(domain)
            rows = []
            if frame is not None and frame.height:
                with console.progress(frame.height, f"评分 {domain}", unit="query") as bar:
                    with torch.no_grad(), precision_context(settings):
                        for row in frame.iter_rows(named=True):
                            query = make_unlabeled_query_supervision(row["drug_id"], row["disease_id"])
                            graph = corpus.assembler.assemble(query, domain=domain, device=settings.device)
                            pred = built.forward(graph.batch.model_batch, path_batch=graph.batch.path_batch)
                            scored = dict(row, support_logit=float(pred.support_logit.float().item()),
                                          unsuitability_logit=float(pred.unsuitability_logit.float().item()))
                            if not all(math.isfinite(scored[key]) for key in ("support_logit", "unsuitability_logit")):
                                raise FloatingPointError("Nonfinite development score")
                            rows.append(scored)
                            bar.update()
                pl.DataFrame(rows).write_parquet(target / f"{domain}.parquet")
            scores[domain] = rows
        support, auxiliary = scores["validation"], scores["auxiliary_validation"]
        if any(row.get("role") != "P" for row in auxiliary):
            raise ValueError("Auxiliary validation must contain verified contraindication positives only")
        metrics = metric_pair(support, auxiliary)
        write_json(target / "metrics.json", dict(step=step, corpus_identity=corpus.identity, metrics=metrics))
        console.metrics(metrics)
        return metrics
    finally:
        for name, module in built.model.named_modules():
            module.training = modes[name]
        restore_rng(state)


def save_checkpoint(path, built, optimizer, step, binding):
    """Replace the latest checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save(dict(schema="EVIHGT_CHECKPOINT_V1", binding=binding, step=step,
                    model=built.model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state()), temporary)
    os.replace(temporary, path)


def run(settings, *, variant="FULL", mode="train", checkpoint=None, report_settings=None):
    if mode not in ("train", "evaluate", "validate"):
        raise ValueError("Use train, evaluate or validate mode")
    if mode == "evaluate" and checkpoint is None:
        raise ValueError("Evaluation requires a trusted local checkpoint")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + variant
    root = settings.resolve(settings.output_dir) / run_id
    # Allocate a fresh output directory for each invocation, including resumes.
    root.mkdir(parents=True, exist_ok=False)
    with Console(root / "logs", refresh_seconds=settings.refresh_seconds) as console:
        console.banner()
        started = time.monotonic()
        try:
            with console.phase(1, "校验并加载训练语料"):
                corpus = load_inputs(settings, console)
                synthetic = getattr(corpus, "manifest", {}).get("purpose") == "SYNTHETIC_TEST_ONLY"
                if synthetic:
                    console.say("[SYNTHETIC_TEST_ONLY] 合成数据软件自检，结果不属于生物医学实验。")
                write_json(root / "configuration.json", settings.as_dict())
                write_json(root / "environment.json", environment())
            if mode == "validate":
                console.say(f"输入校验完成，记录目录：{root}")
                write_json(root / "completion.json", dict(status="INPUTS_VALIDATED", corpus_identity=corpus.identity))
                return root
            with console.phase(2, "初始化 EviHGT"):
                if settings.device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable. Install a compatible PyTorch build or explicitly use configs/cpu.yaml.")
                if settings.precision == "bf16" and not torch.cuda.is_bf16_supported():
                    raise RuntimeError("Configured BF16 is unsupported on this GPU; choose an explicit FP32 configuration.")
                random.seed(corpus.training_seed)
                np.random.seed(corpus.training_seed % 2**32)
                torch.manual_seed(corpus.training_seed)
                built = build_model(corpus.model_config, pretrained=corpus.pretrained,
                                    seed=corpus.training_seed, variant=variant)
                built.model.to(settings.device)
                built.model.train()
                optimizer = torch.optim.AdamW(built.model.parameters(), lr=settings.learning_rate,
                    weight_decay=settings.weight_decay, betas=(0.9, 0.999), eps=1e-8, foreach=None, fused=None)
                binding = dict(variant=variant, corpus=corpus.identity, model=built.metadata(),
                               settings={k: v for k, v in settings.as_dict().items()
                                         if k not in ("output_dir", "manifest", "refresh_seconds", "checkpoint_every", "evaluate_every_cycle")})
                step = 0
                if checkpoint is not None:
                    path = Path(checkpoint).resolve()
                    # Only load trusted checkpoints produced locally by this program.
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    if saved.get("schema") != "EVIHGT_CHECKPOINT_V1" or saved["binding"] != binding:
                        raise ValueError("Checkpoint model, inputs or training settings differ")
                    built.model.load_state_dict(saved["model"], strict=True)
                    optimizer.load_state_dict(saved["optimizer"])
                    step = int(saved["step"])
                    if not 0 <= step <= corpus.plan.total_steps:
                        raise ValueError("Checkpoint step is outside the fixed plan")
                    restore_rng(saved["rng"])
                    console.say(f"[恢复] 从 {path} 加载第 {step}/{corpus.plan.total_steps} 次更新。")
                if mode == "evaluate" and checkpoint is None:
                    raise ValueError("Evaluation requires --checkpoint from a completed or partial local training run")
                write_json(root / "model.json", built.metadata())
                console.say(f"[模型] {variant}；{sum(p.numel() for p in built.model.parameters()):,} 个参数；{settings.device}/{settings.precision}")
                initial_step = step
            if mode == "train":
                with console.phase(3, "按固定三周期计划训练"):
                    if corpus.plan.total_steps != 3 * corpus.plan.G:
                        raise ValueError("The supplied training plan is not exactly three complete cycles")
                    for cycle in range(step // corpus.plan.G + 1, 4):
                        end = cycle * corpus.plan.G
                        with console.progress(end - step, f"Epoch {cycle}/3", unit="update") as bar:
                            while step < end:
                                spec = corpus.plan.step(step + 1)
                                row = update_once(built, corpus, spec, optimizer, settings)
                                step += 1
                                console.event("optimizer_update", step=step, cycle=cycle, variant=variant, **row)
                                memory = torch.cuda.memory_allocated() / 2**30 if settings.device == "cuda" else 0.0
                                bar.update(loss=row["loss"], GPU_GiB=memory)
                                if step % settings.checkpoint_every == 0 or step == end:
                                    save_checkpoint(root / "checkpoints/latest.pt", built, optimizer, step, binding)
                        if settings.evaluate_every_cycle:
                            evaluate(built, corpus, settings, console, root / f"evaluation/step_{step}", step)
            with console.phase(4, "保存最终开发评估结果"):
                if not (root / f"evaluation/step_{step}/metrics.json").is_file():
                    evaluate(built, corpus, settings, console, root / f"evaluation/step_{step}", step)
            with console.phase(5, "导出药物排序、路径解释和结果报告"):
                final_checkpoint = root / "checkpoints/latest.pt"
                if not final_checkpoint.is_file():
                    final_checkpoint = Path(checkpoint).resolve()
                create_report(built, corpus, settings, console, root, step, final_checkpoint, report_settings)
                elapsed = time.monotonic() - started
                write_json(root / "completion.json", dict(status="COMPLETE", variant=variant, mode=mode,
                    updates=step, total_updates=corpus.plan.total_steps, elapsed_seconds=elapsed,
                    resumed_from_step=initial_step, updates_this_invocation=step-initial_step,
                    scope="SYNTHETIC_TEST_ONLY" if synthetic else "LOCAL_RUN_NOT_AN_AUTOMATIC_REPRODUCTION_OF_PUBLISHED_RESULTS",
                    corpus_identity=corpus.identity, report="report.html", summary="summary.json",
                    explanations="explanations/index.json"))
                console.say(f"[结果目录] {root}")
            console.say(f"完整工作流程已完成，用时 {elapsed:.2f}s。")
            return root
        except Exception as exc:
            console.failure(exc)
            exc.evihgt_reported = True
            raise
