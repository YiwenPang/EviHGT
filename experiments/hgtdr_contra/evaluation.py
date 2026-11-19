"""Score the frozen R0 contra panels and aggregate their disease metrics."""

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import random
from statistics import fmean
from time import perf_counter

import numpy as np
import torch

try:
    from evi_hgt_phase98.metrics import RANKING_KEYS, evaluate_panel
except ModuleNotFoundError as error:
    if error.name != "evi_hgt_phase98":
        raise
    from evi_hgt.metrics import RANKING_KEYS, evaluate_panel


def candidate_digest(rows):
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@contextmanager
def deterministic_kernels():
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


R0_FOLDS = tuple(f"R0F{index}" for index in range(5))
R0_CONTRA_DISEASES = 314
METRIC_KEYS = ("auc", "ap", "aupr_trapezoid", *RANKING_KEYS)


def score_panel(trainer, fold_data):
    """Score the fold in frozen candidate order."""
    started = perf_counter()
    if trainer.data is not fold_data.native:
        raise ValueError("Contra scoring requires the trainer's own frozen fold")
    members = [dict(row) for row in fold_data.evaluation_rows]
    if not members or candidate_digest(members) != fold_data.binding["candidate_order_sha256"]:
        raise ValueError("Frozen contra candidate membership or order changed")

    lookup = trainer.data.lookup
    pairs = np.asarray(
        [[lookup["drug"][row["drug_id"]], lookup["disease"][row["disease_id"]]] for row in members],
        dtype=np.int64,
    )
    config = trainer.config
    batch_size = config["batch_size"]
    device = torch.device(trainer.device)
    devices = []
    if device.type == "cuda":
        devices.append(torch.cuda.current_device() if device.index is None else device.index)
    module_modes = [(module, module.training) for module in trainer.model.modules()]
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    scores = []
    batch_count = 0
    try:
        trainer.model.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), deterministic_kernels():
            for offset in range(0, len(pairs), batch_size):
                batch_pairs = pairs[offset:offset + batch_size]
                seed = trainer.native.seed_for(config["seed"], "evaluation", "auxiliary_validation", offset)
                batch = trainer.data.batch(batch_pairs, config, seed, trainer.device)
                values = trainer.model(batch).reshape(-1).cpu()
                if values.numel() != len(batch_pairs):
                    raise ValueError("HGTDR contra must return exactly one score per candidate")
                if not torch.isfinite(values).all():
                    raise ValueError("Missing or nonfinite HGTDR contra score")
                scores.extend(values.tolist())
                batch_count += 1
    finally:
        for module, was_training in module_modes:
            module.training = was_training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)

    rows = [
        dict(row, unsuitability_logit=float(score))
        for row, score in zip(members, scores, strict=True)
    ]
    metrics = evaluate_panel(rows, head="contra", expected_members=members)
    if metrics["status"] != "COMPLETE":
        raise ValueError("Frozen contra panel produced no evaluable diseases")
    trainer.forwards += batch_count
    trainer.query_forward_count += len(rows)
    trainer.actual_evaluation_seconds += perf_counter() - started
    return {"rows": rows, "metrics": metrics}


def checkpoint_key(epoch, metrics):
    """Rank checkpoints by development AUC, AP, and earliest epoch."""
    auc, ap = metrics.get("macro_auc"), metrics.get("macro_ap")
    if type(epoch) is not int or epoch < 1:
        raise ValueError("A positive integer epoch is required for checkpoint selection")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in (auc, ap)):
        raise ValueError("Checkpoint selection requires finite development ROC-AUC and AP")
    return auc, ap, -epoch


def aggregate_fold_results(fold_results):
    """Average each evaluable disease across its legal folds, then macro-average."""
    if set(fold_results) != set(R0_FOLDS):
        raise ValueError("The R0 summary requires all five frozen development folds")
    by_disease = defaultdict(list)
    for fold in R0_FOLDS:
        metrics = fold_results[fold]
        rows = metrics["disease_metrics"]
        if metrics["head"] != "contra" or metrics["status"] != "COMPLETE" or len(rows) != metrics["evaluable_diseases"]:
            raise ValueError(f"Incomplete contra metrics for {fold}")
        seen = set()
        for row in rows:
            disease = row["disease_id"]
            if disease in seen or row["P"] <= 0 or row["U"] <= 0:
                raise ValueError(f"Invalid evaluable disease in {fold}")
            if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) for key in METRIC_KEYS):
                raise ValueError(f"Missing or nonfinite disease metric in {fold}")
            seen.add(disease)
            by_disease[disease].append(row)
    if len(by_disease) != R0_CONTRA_DISEASES:
        raise ValueError("The summary must cover all 314 evaluable R0 contra diseases")

    diseases = [
        dict(disease_id=disease, legal_folds=len(rows), **{
            key: fmean(row[key] for row in rows) for key in METRIC_KEYS
        })
        for disease, rows in sorted(by_disease.items())
    ]
    means = {key: fmean(row[key] for row in diseases) for key in METRIC_KEYS}
    return {
        "scope": "R0_INNER_DEVELOPMENT_DISEASE_FIRST",
        "head": "contra",
        "folds": list(R0_FOLDS),
        "evaluable_diseases": len(diseases),
        "macro_auc": means["auc"],
        "macro_ap": means["ap"],
        "macro_aupr_trapezoid": means["aupr_trapezoid"],
        "ranking": {key: means[key] for key in RANKING_KEYS},
        "disease_metrics": diseases,
    }
