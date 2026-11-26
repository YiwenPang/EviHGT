"""Run individual matched, frequency and native baselines on a portable development fold."""

from datetime import datetime, timezone
import math
import os

import polars as pl
import torch

from evi_hgt.console import Console
from evi_hgt.data.common import sha256
from evi_hgt.io import write_json
from evi_hgt.metrics import evaluate_panel
from evi_hgt.runner import load_inputs, evaluate, restore_rng, rng_state, update_once
from experiments.comparisons.src.evi_hgt_ranking_comparison.popularity import Popularity, member_digest

MATCHED = {"PAIR_MLP_D1_MATCHED", "HGT_D1_MATCHED"}
POPULARITY = "TRAIN_VISIBLE_POPULARITY"
HGTDR = {"HGTDR_ADAPTED": "support", "HGTDR_ADAPTED_CONTRA": "contra"}
TXGNN = "TXGNN_ADAPTED"
METHODS = tuple(sorted(MATCHED | HGTDR.keys() | {POPULARITY, TXGNN}))


def candidate_rows(corpus):
    support = list(corpus.panels["validation"].select("drug_id", "disease_id", "role").iter_rows(named=True))
    contra = [dict(r, role="P") for r in corpus.panels["auxiliary_validation"].select(
        "drug_id", "disease_id").iter_rows(named=True)]
    contra.extend(r.copy() for r in support if r["role"] == "U")
    order = lambda row: (row["drug_id"], row["disease_id"])
    return dict(support=sorted(support, key=order), contra=sorted(contra, key=order))


def scored_panel(root, head, members, scores):
    field = "support_logit" if head == "support" else "unsuitability_logit"
    rows = [dict(row, **{field: float(score)}) for row, score in zip(members, scores, strict=True)]
    metrics = evaluate_panel(rows, head=head, expected_members=members)
    if metrics["status"] != "COMPLETE":
        raise ValueError(f"No evaluable diseases in the {head} development panel")
    pl.DataFrame(rows).write_parquet(root / f"{head}.parquet")
    return metrics


def _popularity(corpus, panels, root):
    baseline = Popularity.build(
        cell=corpus.manifest.get("source_cell", "R0F1"), support=corpus.support,
        contra=corpus.unsuitability, expected_support_digest=member_digest(corpus.support),
        expected_contra_digest=member_digest(corpus.unsuitability), forbidden_pairs=corpus.blocked)
    metrics = {head: scored_panel(root, head, members, (baseline.score(row["drug_id"], head) for row in members))
               for head, members in panels.items()}
    return metrics, dict(updates=0, selection="TRAINING_FREQUENCY", baseline=baseline.identity)


def save_training_state(path, state):
    temporary = path.with_name(path.name + ".partial")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_training_state(path, binding):
    # Only user-selected, locally produced checkpoints are supported.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("binding") != binding:
        raise ValueError("Checkpoint method, corpus or training recipe differs")
    return state


def model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def selection_key(metrics, epoch):
    values = metrics["macro_auc"], metrics["macro_ap"]
    if not all(value is not None and math.isfinite(value) for value in values):
        raise ValueError("Checkpoint selection requires finite macro AUC and AP")
    return (*values, -epoch)


def _matched(method, corpus, settings, root, console, checkpoint):
    from evi_hgt import losses, model, runner
    from . import matched as architecture

    seed = corpus.training_seed
    torch.manual_seed(seed)
    # The historical matched controls copy the initial side encoder before
    # any D1 forward hook is installed. They each attach their own D1 residual.
    full = model.build_model(corpus.model_config, pretrained=None, seed=seed, variant="NO_D1")
    matched = architecture.build_matched_pair(seed, corpus.pretrained, corpus.index.edge_catalog,
                                 common_node_encoder=full.model.encoder.node_encoder)[method]
    del full
    matched.model.to(settings.device).train()
    optimizer = torch.optim.AdamW(matched.model.parameters(), lr=settings.learning_rate,
        weight_decay=settings.weight_decay, betas=(.9, .999), eps=1e-8, foreach=None, fused=None)
    recipe = {key: getattr(settings, key) for key in ("device", "precision", "learning_rate",
                                                      "weight_decay", "max_grad_norm")}
    sources = {name: sha256(module.__file__) for name, module in
               (("architecture", architecture), ("common_model", model), ("runner", runner), ("losses", losses))}
    binding = dict(method=method, corpus=corpus.identity, recipe=recipe, model=matched.metadata(),
                   sources=sources, comparison_source_sha256=sha256(__file__))
    write_json(root / "model.json", matched.metadata())
    step = 0
    if checkpoint is not None:
        saved = load_training_state(checkpoint, binding)
        step = saved["step"]
        if type(step) is not int or not 0 <= step <= corpus.plan.total_steps:
            raise ValueError("Checkpoint step is outside the fixed training plan")
        matched.model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
    for cycle in range(step // corpus.plan.G + 1, 4):
        end = cycle * corpus.plan.G
        with console.progress(end - step, f"Epoch {cycle}/3", unit="update") as bar:
            while step < end:
                record = update_once(matched, corpus, corpus.plan.step(step + 1), optimizer, settings)
                step += 1
                console.event("optimizer_update", step=step, cycle=cycle, **record)
                bar.update(loss=record["loss"])
                if step % settings.checkpoint_every == 0 or step == end:
                    save_training_state(root / "checkpoint.pt", dict(binding=binding, step=step,
                        model=matched.model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state()))
    metrics = evaluate(matched, corpus, settings, console, root / "evaluation", step)
    return metrics, dict(updates=step, selection="FIXED_3G")


def run_comparison(method, settings, *, smoke=False, checkpoint=None, context=None):
    """Train or score one baseline and save metrics on its required common panels."""
    if method not in METHODS:
        raise ValueError(f"Unknown comparison method: {method}")
    if smoke and settings.device != "cpu":
        raise ValueError("Smoke runs are CPU only")
    context = context or {}
    if method == TXGNN and not context.get("_txgnn_worker"):
        from .native_sources.txgnn_worker import run_worker
        return run_worker(settings, smoke=smoke, checkpoint=checkpoint, context=context)
    if checkpoint and method == POPULARITY:
        raise ValueError("Popularity has no training checkpoint")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + method
    root = settings.resolve(settings.output_dir) / run_id
    root.mkdir(parents=True, exist_ok=False)
    with Console(root / "logs", refresh_seconds=settings.refresh_seconds) as console:
        try:
            corpus = load_inputs(settings, console)
            synthetic = corpus.manifest.get("purpose") == "SYNTHETIC_TEST_ONLY"
            if smoke != synthetic:
                raise ValueError("Explicit smoke mode and a synthetic corpus must be used together")
            panels = candidate_rows(corpus)
            if method == POPULARITY:
                metrics, record = _popularity(corpus, panels, root)
            elif method in MATCHED:
                metrics, record = _matched(method, corpus, settings, root, console, checkpoint)
            elif method in HGTDR:
                from .native_sources.hgtdr import run_hgtdr
                metrics, record = run_hgtdr(method, HGTDR[method], corpus, panels, settings, root, console,
                                            smoke=smoke, checkpoint=checkpoint, context=context)
            else:
                from .native_sources.txgnn import run_txgnn
                metrics, record = run_txgnn(corpus, panels, root, console, smoke=smoke,
                                            checkpoint=checkpoint, context=context)
            expected_heads = {HGTDR[method]} if method in HGTDR else set(panels)
            if set(metrics) != expected_heads:
                raise ValueError(f"{method} did not score all required development heads")
            for head, metrics_for_head in metrics.items():
                expected = evaluate_panel(
                    [dict(row, **{("support_logit" if head == "support" else "unsuitability_logit"): 0.})
                     for row in panels[head]], head=head)["candidate_identity_sha256"]
                if metrics_for_head["candidate_identity_sha256"] != expected:
                    raise ValueError(f"{method} changed the common {head} candidates")
            if smoke:
                for item in metrics.values():
                    item["scope"] = "SYNTHETIC_TEST_ONLY"
            write_json(root / "metrics.json", metrics)
            write_json(root / "completion.json", dict(status="COMPLETE", method=method,
                corpus_identity=corpus.identity,
                scope="SYNTHETIC_TEST_ONLY" if smoke else "LOCAL_DEVELOPMENT_PANEL", **record))
            return root
        except Exception as error:
            console.failure(error)
            error.evihgt_reported = True
            raise

def check_native_requirements(context, methods, *, offline=False, smoke=False):
    """Check all requested optional runtimes before any raw preparation."""
    methods = set(methods)
    unknown = methods - set(METHODS) - {"FULL"}
    if unknown:
        raise ValueError(f"Unknown comparison methods: {sorted(unknown)}")
    context = dict(context or {})
    context["offline"] = bool(offline or context.get("offline"))
    if methods & (MATCHED | HGTDR.keys()):
        try:
            from torch_geometric.nn import HGTConv  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Install requirements-experiments.txt for matched controls and HGTDR") from error
    if methods & HGTDR.keys():
        from .native_sources.hgtdr import native_hgtdr
        native_hgtdr()
        if not hasattr(torch.ops.torch_sparse, "hgt_sample"):
            raise RuntimeError("The installed torch-sparse build has no native hgt_sample operator")
        if not smoke and not torch.cuda.is_available():
            raise RuntimeError("The recorded HGTDR recipe needs CUDA; CPU is supported only for synthetic smoke")
    if TXGNN in methods:
        from .native_sources.txgnn_worker import check_worker
        check_worker(context, smoke=smoke)
