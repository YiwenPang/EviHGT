"""Train HGTDR's scalar models on the existing cold-disease split."""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
import torch
from filelock import FileLock

from . import data as cold_data
from evi_hgt.phase9 import hgtdr_adapter as hg
from evi_hgt.phase9.production_state import _rng, _snapshot
from experiments.hgtdr_contra.run import (
    CHECKPOINT_SCHEMA,
    load_checkpoint,
    publish_json,
    restore_epoch,
    save_checkpoint,
)
from evi_hgt_phase98.common import append, read, sha
from evi_hgt_phase98.metrics import SCORE_FIELDS, evaluate_panel
from evi_hgt_phase98.native_numerical_policy import hgtdr_numerical_policy


HEADS = ("contra", "support")
METHOD = {"support": "HGTDR-adapted", "contra": "HGTDR-adapted-contra"}


class ColdTrainer(hg.SingleLRTrainer):
    """The native model, sampler, optimizer, and epoch control under the cold protocol."""

    def __init__(self, data, head):
        with hgtdr_numerical_policy():
            super().__init__(data, device="cuda")
        self.execution_scope = {"mode": "SEALED_COLD_DISEASE", "prediction_target": head}

    def step(self):
        with hgtdr_numerical_policy():
            return super().step()


def score(trainer, data, head, members, domain):
    """Keep validation deterministic and apply the same cold test candidate ordering."""
    index = np.asarray(
        [(data.lookup["drug"][row["drug_id"]], data.lookup["disease"][row["disease_id"]])
         for row in members],
        dtype=np.int64,
    )
    batch_size = trainer.config["batch_size"]
    modes = [(layer, layer.training) for layer in trainer.model.modules()]
    scores = []
    started = time.perf_counter()
    try:
        trainer.model.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode(), hgtdr_numerical_policy():
            for offset in range(0, len(index), batch_size):
                pairs = index[offset:offset + batch_size]
                seed = trainer.native.seed_for(trainer.config["seed"], "evaluation", domain, offset)
                batch = data.batch(pairs, trainer.config, seed, trainer.device)
                output = trainer.model(batch).reshape(-1).cpu()
                if output.numel() != len(pairs) or not torch.isfinite(output).all():
                    raise ValueError("Expected one finite score for every candidate")
                scores.extend(output.tolist())
                trainer.forwards += 1
    finally:
        for layer, training in modes:
            layer.training = training
    trainer.query_forward_count += len(members)
    trainer.actual_evaluation_seconds += time.perf_counter() - started
    scored = [dict(row, **{SCORE_FIELDS[head]: value}) for row, value in zip(members, scores, strict=True)]
    result = evaluate_panel(scored, head=head, expected_members=members, scope=domain)
    if result["status"] != "COMPLETE":
        raise ValueError(f"No evaluable diseases for {head} in {domain}")
    return scored, result


def native_for(template, head):
    data = copy.copy(template)
    data.supports = template.training_positives[head]
    data.support_ids = template.training_positive_ids[head]
    return data


def train_head(output, template, head, validation, protocol, reference_binding):
    folder = output / head
    folder.mkdir(exist_ok=True)
    selected = folder / "selected.weights.pt"
    receipt_path = folder / "trained.json"
    if receipt_path.exists():
        return trained_receipt(output, head, protocol)

    data = native_for(template, head)
    trainer = ColdTrainer(data, head)
    epochs = trainer.config["epochs"]
    updates_per_epoch = math.ceil(len(data.supports) / (trainer.config["batch_size"] // 2))
    binding = dict(protocol=protocol, head=head, corpus=reference_binding, trainer=trainer.metadata())
    publish_json(folder / "binding.json", binding)
    latest = folder / "latest.pt"
    saved = load_checkpoint(latest, binding)
    if saved is not None:
        restore_epoch(trainer, saved)
        del saved

    start_epoch = trainer.cursor["epoch"]
    events = folder / "events.jsonl"
    started = time.perf_counter()
    last_display = 0.0
    for epoch in range(start_epoch, epochs + 1):
        while trainer.cursor["epoch"] == epoch:
            row = trainer.step()
            if time.monotonic() - last_display >= 5:
                print(f"{METHOD[head]} | Epoch {epoch}/{epochs} | loss {row['loss']:.6g} | "
                      f"{1 / row['elapsed_seconds']:.2f} updates/s | "
                      f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
                last_display = time.monotonic()
        domain = "validation" if head == "support" else "auxiliary_validation"
        _, metrics = score(trainer, data, head, validation, domain)
        trainer.selection.observe(epoch, metrics)
        if trainer.selection.best["epoch"] == epoch:
            trainer.best_model = _snapshot(trainer.model.state_dict())
        save_checkpoint(latest, dict(
            schema=CHECKPOINT_SCHEMA,
            binding=binding,
            completed_epoch=epoch,
            control=trainer.control_state(),
            model=_snapshot(trainer.model.state_dict()),
            optimizer=_snapshot(trainer.optimizer.state_dict()),
            scheduler=trainer.scheduler.state_dict(),
            rng=_rng(),
        ))
        summary = {
            "event": "EPOCH_COMMITTED",
            "head": head,
            "epoch": epoch,
            "epochs": epochs,
            "updates": trainer.cursor["optimizer_steps"],
            "dev_auc": metrics["macro_auc"],
            "dev_ap": metrics["macro_ap"],
            "best_epoch": trainer.selection.best["epoch"],
            "eta_seconds_head": round((time.perf_counter() - started) / (epoch - start_epoch + 1) * (epochs - epoch)),
        }
        append(events, summary)
        print(f"{METHOD[head]} | Epoch {epoch}/{epochs} | development "
              f"AUROC {metrics['macro_auc']:.6f} | AP {metrics['macro_ap']:.6f} | "
              f"best epoch {trainer.selection.best['epoch']} | "
              f"estimated remaining {summary['eta_seconds_head']} s", flush=True)
    if trainer.cursor["optimizer_steps"] != epochs * updates_per_epoch or trainer.selection.opportunities != epochs:
        raise ValueError("Interrupted or incorrect optimization budget")
    if selected.exists():
        previous = torch.load(selected, map_location="cpu", weights_only=True)
        if set(previous) != set(trainer.best_model) or any(
            not torch.equal(previous[key], value) for key, value in trainer.best_model.items()
        ):
            raise ValueError("Existing selected weights differ from the sealed final selection")
    else:
        save_checkpoint(selected, trainer.best_model)
    receipt = dict(
        status="TRAINED",
        head=head,
        method=METHOD[head],
        protocol_sha256=sha(output / "protocol.json"),
        weights_sha256=sha(selected),
        selected=trainer.selection.best,
        epochs=epochs,
        updates=trainer.cursor["optimizer_steps"],
    )
    publish_json(receipt_path, receipt)
    return receipt


def trained_receipt(output, head, protocol):
    """Authorize test-file reads only after the complete development selection."""
    folder = output / head
    receipt = read(folder / "trained.json")
    epochs = protocol["native_config"]["epochs"]
    positives = protocol["corpus"]["training_positive_counts"][head]
    steps = math.ceil(positives / (protocol["native_config"]["batch_size"] // 2))
    if (receipt["status"] != "TRAINED" or receipt["head"] != head
            or receipt["method"] != METHOD[head] or receipt["epochs"] != epochs
            or receipt["updates"] != epochs * steps or not 1 <= receipt["selected"]["epoch"] <= epochs
            or receipt["protocol_sha256"] != sha(output / "protocol.json")
            or receipt["weights_sha256"] != sha(folder / "selected.weights.pt")):
        raise ValueError(f"Completed training receipt differs for {head}")
    return receipt


def verified_reference(reference_run, binding):
    receipt = read(reference_run / "result.json")
    metric_path = reference_run / "test/metrics.json"
    checksum = sha(metric_path)
    if (receipt["status"] != "COMPLETE" or receipt["case"] != "cold_disease"
            or receipt["metric_sha256"] != checksum):
        raise ValueError("The EviHGT reference run must have sealed cold-disease scores")
    reference = read(metric_path)
    membership = reference["membership"]
    if (reference["status"] != "COMPLETE" or reference["case"] != "cold_disease"
            or membership["manifest_sha256"] != binding["manifest_sha256"]
            or membership["preparation_audit_sha256"] != binding["preparation_audit_sha256"]
            or not all(membership[key] for key in ("test_blocking_verified", "test_dev_disjoint",
                                                   "test_training_disjoint", "shared_u"))
            or membership["test_u_observed_overlap"] != 0
            or membership["panel_sha256"] != {head: binding["test_panel_sha256"][f"test_{head}.parquet"]
                                               for head in HEADS}):
        raise ValueError("The EviHGT reference and HGTDR corpus use different cold-test inputs")
    return reference, checksum


def final_scores(output, template, members, receipts, reference):
    metrics = {}
    for head in HEADS:
        chosen = receipts[head]
        folder = output / head
        selected = folder / "selected.weights.pt"
        if sha(selected) != chosen["weights_sha256"]:
            raise ValueError("Selected weights changed before the test phase")
        data = native_for(template, head)
        trainer = ColdTrainer(data, head)
        trainer.model.load_state_dict(torch.load(selected, weights_only=True, map_location="cpu"))
        rows, results = score(trainer, data, head, members[head], "cold_disease_sealed_test")
        base = reference["metrics"][head]
        for key in ("candidate_identity_sha256", "pairs", "P", "U", "evaluable_diseases", "skipped_diseases"):
            if results[key] != base[key]:
                raise ValueError(f"Cold test panel differs from the EviHGT reference on {head}:{key}")
        path = folder / "test_scores.parquet"
        if not path.exists():
            temporary = path.with_name(f"{path.name}.{uuid4().hex}.partial")
            pl.DataFrame(rows).write_parquet(temporary)
            os.replace(temporary, path)
        elif pl.read_parquet(path).to_dicts() != rows:
            raise ValueError(f"Previously scored test candidates differ for {head}")
        results["scores_sha256"] = sha(path)
        publish_json(folder / "test_metrics.json", results)
        metrics[head] = results
    return metrics


def paired_comparison(peer, reference, seed):
    reference_rows = {row["disease_id"]: row for row in reference["disease_metrics"]}
    peer_rows = {row["disease_id"]: row for row in peer["disease_metrics"]}
    if set(peer_rows) != set(reference_rows):
        raise ValueError("Per-disease evaluation sets differ")
    diseases = sorted(reference_rows)
    for disease in diseases:
        if any(peer_rows[disease][key] != reference_rows[disease][key] for key in ("P", "U", "N")):
            raise ValueError(f"Per-disease candidate counts differ for {disease}")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(diseases), size=(10_000, len(diseases)))
    results = {}
    for key in ("auc", "ap", "NDCG@10"):
        delta = np.array([reference_rows[disease][key] - peer_rows[disease][key] for disease in diseases])
        low, high = np.quantile(delta[draws].mean(axis=1), (0.025, 0.975))
        results[key] = dict(EviHGT_minus_HGTDR=float(delta.mean()),
                            conditional_disease_bootstrap_95=[float(low), float(high)])
    return results


def execute(mode, corpus, cache, output, reference_run):
    if mode != "validate" and not torch.cuda.is_available():
        raise RuntimeError("HGTDR training and scoring require a CUDA device")
    torch.set_num_threads(4)
    if mode == "validate":
        _, validation, binding = cold_data.prepare_training(corpus, cache)
        print(json.dumps(dict(
            status="VALIDATED",
            training_positives=binding["training_positive_counts"],
            development_candidates={head: len(rows) for head, rows in validation.items()},
            corpus_sha256=binding["manifest_sha256"],
        ), indent=2), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "run.lock", timeout=0):
        template, validation, binding = cold_data.prepare_training(corpus, cache)
        protocol = dict(
            case="cold_disease",
            objective="Common held-out diseases and test candidates",
            method=METHOD,
            reference_model="EviHGT fixed three-cycle final checkpoint",
            native_config=hg.fixed_config(template.training_seed),
            heads_trained_independently=True,
            selection_data="Disjoint cold-disease development candidates",
            test_phase="Score using both selected weights after development selection",
            comparison="Common split, graph exclusions and test candidates. Training recipes and feature sets are method specific.",
            unlabeled_role="U",
            U_is_confirmed_negative=False,
            source_sha256={name: sha(Path(__file__).with_name(name))
                           for name in ("run.py", "runtime.py", "data.py")},
            corpus=binding,
        )
        protocol_path = output / "protocol.json"
        if mode == "score" and not protocol_path.is_file():
            raise FileNotFoundError("Train both models before opening the cold test panels")
        publish_json(protocol_path, protocol)
        if mode in ("train", "run"):
            print("Cold-disease comparison | 300 epochs per HGTDR model", flush=True)
            for head in HEADS:
                train_head(output, template, head, validation[head], protocol, binding)
        if mode == "train":
            return
        receipts = {head: trained_receipt(output, head, protocol) for head in HEADS}
        reference, reference_sha = verified_reference(reference_run, binding)
        members = cold_data.load_test(corpus, binding)
        metrics = final_scores(output, template, members, receipts, reference)
        compact = {}
        for head, values in metrics.items():
            base = reference["metrics"][head]
            compact[head] = {
                "paired_disease_analysis": paired_comparison(values, base, template.training_seed),
                "evaluable_diseases": values["evaluable_diseases"],
                "HGTDR": {**{key: values[key] for key in ("macro_ap", "macro_auc")},
                          "NDCG@10": values["ranking"]["NDCG@10"]},
                "EviHGT": {**{key: base[key] for key in ("macro_ap", "macro_auc")},
                           "NDCG@10": base["ranking"]["NDCG@10"]},
            }
        result = dict(
            status="COMPLETE",
            protocol_sha256=sha(protocol_path),
            evihgt_metrics_sha256=reference_sha,
            checkpoints=receipts,
            test_metrics_sha256={head: sha(output / head / "test_metrics.json") for head in HEADS},
            paired_analysis_scope="Conditional on this split and both trained models. Drugs can recur across diseases.",
            comparison=compact,
        )
        publish_json(output / "result.json", result)
        print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
