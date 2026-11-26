"""Compare HGTDR and EviHGT on the same frozen cold-disease candidates."""

from dataclasses import replace
from pathlib import Path

from filelock import FileLock
import numpy as np
import polars as pl
import torch

from evi_hgt.console import Console
from evi_hgt.data.common import sha256
from evi_hgt.io import digest, write_json
from evi_hgt.metrics import evaluate_panel

from .cold import TEST_SCOPE, _preflight, _read_json, _verify_test_files, archived_scripts
from .comparisons import candidate_rows
from .native_sources.hgtdr import HGTDRFold, run_hgtdr


METHODS = {"contra": "HGTDR_ADAPTED_CONTRA", "support": "HGTDR_ADAPTED"}
TEST_DOMAIN = "cold_disease_sealed_test"


def _paired(peer, reference, seed):
    """Keep the original paired per-disease bootstrap and metric directions."""
    peer_cases = {r["disease_id"]: r for r in peer["disease_metrics"]}
    reference_cases = {r["disease_id"]: r for r in reference["disease_metrics"]}
    if (peer["candidate_identity_sha256"] != reference["candidate_identity_sha256"]
            or not peer_cases or peer_cases.keys() != reference_cases.keys()):
        raise ValueError("HGTDR and EviHGT cold-test candidate identities differ")
    diseases = sorted(peer_cases)
    if any(peer_cases[d][key] != reference_cases[d][key] for d in diseases for key in ("P", "U", "N")):
        raise ValueError("Per-disease cold-test counts differ")
    draws = np.random.default_rng(seed).integers(len(diseases), size=(10_000, len(diseases)))
    result = {}
    for k in ("auc", "ap", "NDCG@10"):
        values = np.array([reference_cases[d][k] - peer_cases[d][k] for d in diseases])
        low, high = np.quantile(values[draws].mean(axis=1), (0.025, 0.975))
        result[k] = dict(EviHGT_minus_HGTDR=float(values.mean()),
                         conditional_disease_bootstrap_95=[float(low), float(high)])
    return result


def _reference(path, manifest, receipt, score):
    result = _read_json(path / "result.json")
    metrics_path = path / "test/metrics.json"
    expected_hash = sha256(metrics_path)
    if (result.get("status") != "COMPLETE" or result.get("case") != "cold_disease"
            or result.get("metric_sha256") != expected_hash):
        raise ValueError("A completed, bound EviHGT cold-disease test is required")
    reference = _read_json(metrics_path)
    members = reference["membership"]
    panel_files = _read_json(receipt)["test_files"]
    if (reference.get("status") != "COMPLETE" or reference.get("case") != "cold_disease"
            or members["manifest_sha256"] != sha256(manifest)
            or members["preparation_audit_sha256"] != sha256(receipt)
            or members["panel_sha256"] != {head: panel_files[score.PANEL_NAMES[head]]
                                          for head in METHODS}):
        raise ValueError("EviHGT reference and HGTDR cold corpus do not share the sealed test")
    return reference, expected_hash


def _fit_heads(corpus, panels, root, settings, context, binding, *, smoke):
    receipts = {}
    for head, method in METHODS.items():
        folder = root / head
        folder.mkdir(exist_ok=True)
        commit, selected = folder / "trained.json", folder / "selected.pt"
        if commit.exists():
            receipt = _read_json(commit)
            if receipt["binding"] != binding or receipt["checkpoint_sha256"] != sha256(selected):
                raise ValueError(f"Selected cold HGTDR checkpoint changed for {head}")
        else:
            checkpoint = folder / "checkpoint.pt"
            with Console(folder / "logs", refresh_seconds=settings.refresh_seconds) as console:
                _, training = run_hgtdr(method, head, corpus, panels, settings, folder, console,
                    smoke=smoke, checkpoint=checkpoint if checkpoint.exists() else None, context=context)
            receipt = dict(training=training, checkpoint_sha256=sha256(selected), binding=binding)
            write_json(commit, receipt)
        receipts[head] = receipt
    return receipts


def _score_head(head, corpus, settings, context, root, members, reference, trained, *, smoke):
    target = root / "test"
    target.mkdir(exist_ok=True)
    folder = root / head
    path = folder / "selected.pt"
    if sha256(path) != trained["checkpoint_sha256"]:
        raise ValueError(f"Selected HGTDR checkpoint changed before scoring {head}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    fold = HGTDRFold(corpus, context, folder, smoke=smoke, device=settings.device)
    if checkpoint["binding"]["data"] != fold.identity or checkpoint["epoch"] != trained["training"]["selected_epoch"]:
        raise ValueError("HGTDR test graph, features or checkpoint selection differ from training")
    positives = corpus.support if head == "support" else corpus.unsuitability
    model = fold.make_model(fold.pair_ids(sorted(positives))[:fold.batch_size])
    model.load_state_dict(checkpoint["model"], strict=True)
    logits = fold.scores(model, members, head, domain=TEST_DOMAIN)
    field = "support_logit" if head == "support" else "unsuitability_logit"
    scored = [dict(row, **{field: float(value)}) for row, value in zip(members, logits, strict=True)]
    metrics = evaluate_panel(scored, head=head, expected_members=members, scope=TEST_DOMAIN)
    comparison = _paired(metrics, reference, corpus.training_seed)
    path = target / f"{head}.parquet"
    pl.DataFrame(scored).write_parquet(path)
    write_json(target / f"{head}_metrics.json", metrics)
    def summary(values):
        return {"macro_auc": values["macro_auc"], "macro_ap": values["macro_ap"],
                "NDCG@10": values["ranking"]["NDCG@10"]}

    return dict(training=trained["training"], HGTDR=summary(metrics), EviHGT=summary(reference),
                evaluable_diseases=metrics["evaluable_diseases"], paired_disease_analysis=comparison,
                scores_sha256=sha256(path), metrics_sha256=sha256(target / f"{head}_metrics.json"))


def run_cold_hgtdr(settings, case_info, reference_run, context, *, output=None, smoke=False):
    """Fit both scalar models on development candidates before reading test roles."""
    from evi_hgt.runner import load_inputs

    _, score = archived_scripts()
    if case_info["case"] != "cold_disease":
        raise ValueError("The HGTDR cold comparison is registered for cold_disease")
    manifest = Path(case_info["manifest"]).resolve(strict=True)
    content = _read_json(manifest)
    if smoke != (content.get("purpose") == TEST_SCOPE):
        raise ValueError("Native cold mode and corpus purpose differ")
    receipt = _verify_test_files(score, manifest)
    reference_path = Path(reference_run).resolve(strict=True)
    reference, reference_hash = _reference(reference_path, manifest, receipt, score)
    root = Path(output or settings.resolve(settings.output_dir) / "cold_disease_hgtdr").resolve()
    if root == reference_path or root.is_relative_to(reference_path) or reference_path.is_relative_to(root):
        raise ValueError("Choose a distinct HGTDR comparison output directory")
    if root.is_relative_to(manifest.parent) or manifest.is_relative_to(root):
        raise ValueError("Native cold outputs and its input corpus must be separate")
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "run.lock")), Console(root / "logs", refresh_seconds=settings.refresh_seconds) as console:
        result_path = root / "result.json"
        binding = dict(manifest_sha256=sha256(manifest), preparation_sha256=sha256(receipt),
            evihgt_metrics_sha256=reference_hash, scope=TEST_SCOPE if smoke else "PORTABLE_COLD_COMPARISON",
            device=settings.device, workflow_sha256=sha256(__file__), runtime_sha256=sha256(Path(
                __file__).parent / "native_sources/hgtdr.py"))
        if result_path.is_file():
            result = _read_json(result_path)
            if result["binding"] != binding or any(
                    sha256(root / "test" / f"{head}.parquet") != row["scores_sha256"]
                    or sha256(root / "test" / f"{head}_metrics.json") != row["metrics_sha256"]
                    for head, row in result["heads"].items()):
                raise ValueError("Completed HGTDR cold comparison inputs or outputs changed")
            _preflight(score, "cold_disease", manifest)
            return root
        with console.phase(1, "验证冷启动训练图和开发候选"):
            corpus = load_inputs(replace(settings, manifest=str(manifest)), console)
            panels = candidate_rows(corpus)
        with console.phase(2, "训练并在开发集选择 HGTDR 检查点"):
            trained = _fit_heads(corpus, panels, root, settings, context, binding, smoke=smoke)
        with console.phase(3, "评分同一留出面板并按疾病比较"):
            membership = _preflight(score, "cold_disease", manifest)
            if membership != reference["membership"]:
                raise ValueError("The training isolation and test candidate bindings differ")
            heads = {}
            for head in METHODS:
                path = manifest.parent / score.PANEL_NAMES[head]
                if sha256(path) != membership["panel_sha256"][head]:
                    raise ValueError(f"The cold test panel changed: {path}")
                members = pl.read_parquet(path, columns=score.PANEL_FIELDS).sort(score.PAIR_FIELDS).to_dicts()
                heads[head] = _score_head(head, corpus, settings, context, root, members,
                    reference["metrics"][head], trained[head], smoke=smoke)
        write_json(result_path, dict(status="COMPLETE", case="cold_disease", purpose=binding["scope"],
            binding=binding, binding_sha256=digest(binding), heads=heads))
    return root
