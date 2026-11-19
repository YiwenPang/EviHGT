"""Validation-only P-vs-U diagnostics over uncalibrated raw logits.

This module deliberately does not know how to read formal-test artifacts.  Its
accepted partition is exactly ``validation`` and its second observed class is
named ``unlabeled`` rather than negative.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import csv
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


VALID_TASKS = ("support", "unsuitability")
VALID_MEMBERSHIPS = ("observed_positive", "unlabeled")
VALID_DIAGNOSTIC_SPLITS = ("standard", "cold_drug", "cold_disease")

_EPOCH_PATTERN = re.compile(
    r"Formal epoch=(?P<cycle>\d+) "
    r"train_objective=(?P<train>[-+0-9.eE]+) "
    r"validation_objective=(?P<validation>[-+0-9.eE]+) "
    r"support_risk=(?P<support>[-+0-9.eE]+) "
    r"unsuitability_risk=(?P<unsuitability>[-+0-9.eE]+).*?"
    r"microbatches=(?P<groups>\d+) optimizer_step=(?P<step>\d+)"
)
_COMPLETION_PATTERN = re.compile(
    r"Formal run completed run_id=(?P<run_id>[0-9a-f]{64}) "
    r"reason=(?P<reason>[A-Z0-9_]+) epoch=(?P<cycle>\d+) "
    r"optimizer_step=(?P<step>\d+)"
)


def _finite_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    vector = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite raw logits")
    return vector


def _threshold_curve(
    positive: np.ndarray, unlabeled: np.ndarray
) -> tuple[list[dict[str, float]], list[dict[str, float]], float, float]:
    scores = np.concatenate((positive, unlabeled))
    labels = np.concatenate(
        (np.ones(positive.size, dtype=np.int8), np.zeros(unlabeled.size, dtype=np.int8))
    )
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]

    roc = [{"fpr_proxy": 0.0, "tpr_observed_p": 0.0}]
    pr = [{"recall_observed_p": 0.0, "precision_p_vs_u": 1.0}]
    true_positive = false_positive = 0
    average_precision = 0.0
    previous_recall = 0.0
    cursor = 0
    while cursor < scores.size:
        end = cursor + 1
        while end < scores.size and scores[end] == scores[cursor]:
            end += 1
        block = labels[cursor:end]
        true_positive += int(block.sum())
        false_positive += int(block.size - block.sum())
        recall = true_positive / positive.size
        precision = true_positive / (true_positive + false_positive)
        roc.append(
            {
                "fpr_proxy": false_positive / unlabeled.size,
                "tpr_observed_p": recall,
            }
        )
        pr.append(
            {
                "recall_observed_p": recall,
                "precision_p_vs_u": precision,
            }
        )
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end

    roc_auc = float(
        np.trapezoid(
            [point["tpr_observed_p"] for point in roc],
            [point["fpr_proxy"] for point in roc],
        )
    )
    return roc, pr, roc_auc, float(average_precision)


def compute_pu_diagnostics(
    *, positive_logits: Iterable[float], unlabeled_logits: Iterable[float]
) -> dict[str, Any]:
    """Compute secondary P-vs-U diagnostics without treating U as negative."""

    positive = _finite_vector(positive_logits, name="positive_logits")
    unlabeled = _finite_vector(unlabeled_logits, name="unlabeled_logits")
    combined = np.concatenate((positive, unlabeled))
    roc, pr, roc_auc, average_precision = _threshold_curve(positive, unlabeled)
    return {
        "semantics": "P_VS_U_VALIDATION_ONLY",
        "formal_test_performance": False,
        "scores_are_calibrated_probabilities": False,
        "unlabeled_is_negative": False,
        "roc_fpr_is_proxy": True,
        "threshold_raw_logit": 0.0,
        "n_positive": int(positive.size),
        "n_unlabeled": int(unlabeled.size),
        "score_mean": float(combined.mean()),
        "score_std": float(combined.std(ddof=0)),
        "positive_score_mean": float(positive.mean()),
        "positive_score_median": float(np.median(positive)),
        "unlabeled_score_mean": float(unlabeled.mean()),
        "unlabeled_score_median": float(np.median(unlabeled)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "roc_curve": roc,
        "pr_curve": pr,
        "contingency": {
            "observed_positive": {
                "score_ge_0": int(np.count_nonzero(positive >= 0.0)),
                "score_lt_0": int(np.count_nonzero(positive < 0.0)),
            },
            "unlabeled": {
                "score_ge_0": int(np.count_nonzero(unlabeled >= 0.0)),
                "score_lt_0": int(np.count_nonzero(unlabeled < 0.0)),
            },
        },
    }


def validate_score_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Reject any score table that is not explicitly validation-only P-vs-U."""

    materialized = list(rows)
    if not materialized:
        raise ValueError("validation score rows must not be empty")
    required = {
        "run_id",
        "model",
        "model_variant",
        "split",
        "partition",
        "task",
        "pair_id",
        "query_id",
        "membership",
        "raw_logit",
        "checkpoint_identity",
        "checkpoint_sha256",
    }
    for row in materialized:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"validation score row is missing fields: {sorted(missing)}")
        if row["partition"] != "validation":
            raise ValueError("diagnostic evaluator is validation-only; formal test is sealed")
        if row["split"] not in VALID_DIAGNOSTIC_SPLITS:
            raise ValueError("validation diagnostic split is not permitted")
        if row["task"] not in VALID_TASKS:
            raise ValueError("validation diagnostic task must be support or unsuitability")
        if row["membership"] not in VALID_MEMBERSHIPS:
            raise ValueError("membership must be observed_positive or unlabeled")
        if not math.isfinite(float(row["raw_logit"])):
            raise ValueError("raw_logit must be finite")
        if len(str(row["run_id"])) != 64:
            raise ValueError("run_id must be a full SHA-256 identity")
        for field in ("checkpoint_identity", "checkpoint_sha256"):
            value = str(row[field])
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field} must be a lowercase SHA-256 identity")
        if not str(row["pair_id"]) or not str(row["query_id"]):
            raise ValueError("pair_id and query_id are required")
    return materialized


def analyze_score_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Analyze each run/task independently; never merge the two task heads."""

    materialized = validate_score_rows(rows)
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in materialized:
        grouped[
            (
                str(row["run_id"]),
                str(row["model"]),
                str(row["model_variant"]),
                str(row["split"]),
                str(row["task"]),
            )
        ].append(row)
    run_tasks: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for run_id, model, model_variant, split, task in grouped:
        run_tasks[(run_id, model, model_variant, split)].add(task)
    if any(tasks != set(VALID_TASKS) for tasks in run_tasks.values()):
        raise ValueError("each run requires both support and unsuitability tasks")

    analyses = []
    for (run_id, model, model_variant, split, task), task_rows in sorted(grouped.items()):
        positive = [float(row["raw_logit"]) for row in task_rows if row["membership"] == "observed_positive"]
        unlabeled = [float(row["raw_logit"]) for row in task_rows if row["membership"] == "unlabeled"]
        analyses.append(
            {
                "run_id": run_id,
                "model": model,
                "model_variant": model_variant,
                "split": split,
                "checkpoint_identity": str(task_rows[0]["checkpoint_identity"]),
                "checkpoint_sha256": str(task_rows[0]["checkpoint_sha256"]),
                "task": task,
                **compute_pu_diagnostics(
                    positive_logits=positive,
                    unlabeled_logits=unlabeled,
                ),
            }
        )
    return {
        "status": "VALIDATION-ONLY DIAGNOSTIC / NOT FORMAL TEST PERFORMANCE",
        "formal_test_count": 0,
        "analyses": analyses,
    }


def build_validation_score_rows(
    *,
    run_id: str,
    model: str,
    model_variant: str,
    split: str,
    checkpoint_identity: str,
    checkpoint_sha256: str,
    manifests: Mapping[str, Any],
    support_score_by_pair: Mapping[str, float],
    unsuitability_score_by_pair: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Join replayed logits to frozen P/U membership without relabeling U."""

    if set(manifests) != set(VALID_TASKS):
        raise ValueError("validation replay requires both frozen task manifests")
    score_maps = {
        "support": support_score_by_pair,
        "unsuitability": unsuitability_score_by_pair,
    }
    rows: list[dict[str, Any]] = []
    for task in VALID_TASKS:
        manifest = manifests[task]
        expected = {
            *manifest.positive_pair_ids,
            *manifest.unlabeled_pair_ids,
        }
        if set(score_maps[task]) != expected:
            raise ValueError(f"{task} score map differs from its frozen validation manifest")
        for membership, pair_ids in (
            ("observed_positive", manifest.positive_pair_ids),
            ("unlabeled", manifest.unlabeled_pair_ids),
        ):
            for pair_id in pair_ids:
                parts = str(pair_id).split("\u001f")
                if len(parts) != 2 or not all(parts):
                    raise ValueError("validation pair identity is invalid")
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "model_variant": model_variant,
                        "split": split,
                        "partition": "validation",
                        "task": task,
                        "pair_id": str(pair_id),
                        "query_id": parts[1],
                        "membership": membership,
                        "raw_logit": float(score_maps[task][pair_id]),
                        "checkpoint_identity": checkpoint_identity,
                        "checkpoint_sha256": checkpoint_sha256,
                    }
                )
    return [dict(row) for row in validate_score_rows(rows)]


def parse_training_log(text: str) -> dict[str, Any]:
    """Parse the two-cycle immutable formal-training summary from a run log."""

    cycles = []
    for match in _EPOCH_PATTERN.finditer(text):
        values = {
            "cycle": int(match.group("cycle")),
            "train_objective": float(match.group("train")),
            "validation_objective": float(match.group("validation")),
            "support_risk": float(match.group("support")),
            "unsuitability_risk": float(match.group("unsuitability")),
            "groups": int(match.group("groups")),
            "optimizer_step": int(match.group("step")),
        }
        if not all(
            math.isfinite(float(values[name]))
            for name in (
                "train_objective",
                "validation_objective",
                "support_risk",
                "unsuitability_risk",
            )
        ):
            raise ValueError("training log contains a non-finite objective")
        cycles.append(values)
    completion = _COMPLETION_PATTERN.search(text)
    if [row["cycle"] for row in cycles] != [1, 2] or completion is None:
        raise ValueError("training log must contain two complete cycles and a terminal record")
    if int(completion.group("cycle")) != 2:
        raise ValueError("training completion cycle differs from the frozen horizon")
    if int(completion.group("step")) != cycles[-1]["optimizer_step"]:
        raise ValueError("training completion optimizer step differs from cycle 2")
    return {
        "run_id": completion.group("run_id"),
        "cycles": cycles,
        "termination": completion.group("reason"),
        "optimizer_steps": int(completion.group("step")),
    }


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _write_validation_objective_figure(
    model_rows: list[dict[str, Any]], figures_directory: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for row in model_rows:
        cycles = row["cycles"]
        axis.plot(
            [cycle["cycle"] for cycle in cycles],
            [cycle["validation_objective"] for cycle in cycles],
            marker="o",
            linewidth=2,
            label=row["label"],
        )
    axis.set_xticks((1, 2))
    axis.set_xlabel("Complete coverage cycle")
    axis.set_ylabel("Deterministic validation objective")
    axis.set_title(
        "Standard-model validation objective comparison\n"
        "VALIDATION ONLY — lower is better; not formal retrieval ranking"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    figure.savefig(figures_directory / "validation_objective_comparison.png", dpi=600)
    figure.savefig(figures_directory / "validation_objective_comparison.pdf")
    plt.close(figure)


def _write_metric_figures(
    analyses: list[dict[str, Any]],
    score_rows: list[Mapping[str, Any]],
    figures_directory: Path,
    *,
    availability_note: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    standard_analyses = [row for row in analyses if row["split"] == "standard"]
    per_model_directory = figures_directory / "per_model"
    contingency_directory = figures_directory / "diagnostic_contingency"
    per_model_directory.mkdir(parents=True, exist_ok=True)
    contingency_directory.mkdir(parents=True, exist_ok=True)

    for task in VALID_TASKS:
        task_rows = [row for row in standard_analyses if row["task"] == task]
        for curve_name, x_key, y_key, label in (
            ("roc", "fpr_proxy", "tpr_observed_p", "P-vs-U diagnostic ROC"),
            ("pr", "recall_observed_p", "precision_p_vs_u", "P-vs-U diagnostic Precision-Recall"),
        ):
            figure, axis = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
            for row in task_rows:
                curve = row[f"{curve_name}_curve"]
                axis.plot(
                    [point[x_key] for point in curve],
                    [point[y_key] for point in curve],
                    linewidth=1.8,
                    label=row["model"],
                )
            axis.set_xlabel("false-positive-rate proxy" if curve_name == "roc" else "recall on observed P")
            axis.set_ylabel("true-positive-rate on observed P" if curve_name == "roc" else "precision among P and U")
            axis.set_title(
                f"{task}: VALIDATION ONLY\n{label}; U is unlabeled, not confirmed negative"
                + (f"\n{availability_note}" if availability_note else "")
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
            stem = f"{task}_{curve_name}_comparison"
            figure.savefig(figures_directory / f"{stem}.png", dpi=600)
            figure.savefig(figures_directory / f"{stem}.pdf")
            plt.close(figure)

            for row in task_rows:
                individual, individual_axis = plt.subplots(
                    figsize=(7.2, 5.4), constrained_layout=True
                )
                curve = row[f"{curve_name}_curve"]
                individual_axis.plot(
                    [point[x_key] for point in curve],
                    [point[y_key] for point in curve],
                    linewidth=2.0,
                )
                individual_axis.set_xlabel(axis.get_xlabel())
                individual_axis.set_ylabel(axis.get_ylabel())
                individual_axis.set_title(
                    f"{row['model']} / {task}\nVALIDATION ONLY — {label}"
                )
                individual_axis.grid(alpha=0.25)
                safe_model = re.sub(r"[^a-z0-9]+", "_", row["model"].lower()).strip("_")
                individual_stem = f"{safe_model}_{task}_{curve_name}"
                individual.savefig(per_model_directory / f"{individual_stem}.png", dpi=600)
                individual.savefig(per_model_directory / f"{individual_stem}.pdf")
                plt.close(individual)

        for metric_name, key in (("roc_auc", "roc_auc"), ("pr_auc", "average_precision")):
            labels = [row["model"] for row in task_rows]
            values = [row[key] for row in task_rows]
            x = np.arange(len(task_rows), dtype=float)
            figure, axis = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
            bars = axis.bar(x, values, color="#4C78A8")
            axis.set_xticks(x, labels, rotation=25, ha="right")
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("P-vs-U ROC-AUC" if key == "roc_auc" else "P-vs-U Average Precision")
            axis.set_title(
                f"{task}: VALIDATION ONLY\nP-vs-U diagnostic {'ROC-AUC' if key == 'roc_auc' else 'PR-AUC / Average Precision'}"
                + (f"\n{availability_note}" if availability_note else "")
            )
            axis.grid(axis="y", alpha=0.25)
            axis.bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=3, fontsize=8)
            stem = f"{task}_{metric_name}_comparison"
            figure.savefig(figures_directory / f"{stem}.png", dpi=600)
            figure.savefig(figures_directory / f"{stem}.pdf")
            plt.close(figure)

        figure, axes = plt.subplots(
            max(1, len(task_rows)), 2,
            figsize=(12.0, max(4.0, 3.0 * len(task_rows))),
            squeeze=False,
            constrained_layout=True,
        )
        for row_index, analysis in enumerate(task_rows):
            selected = [
                row for row in score_rows
                if row["run_id"] == analysis["run_id"] and row["task"] == task
            ]
            positive = np.asarray(
                [float(row["raw_logit"]) for row in selected if row["membership"] == "observed_positive"]
            )
            unlabeled = np.asarray(
                [float(row["raw_logit"]) for row in selected if row["membership"] == "unlabeled"]
            )
            axes[row_index, 0].hist(positive, bins=24, alpha=0.62, density=True, label="Observed Positive")
            axes[row_index, 0].hist(unlabeled, bins=24, alpha=0.62, density=True, label="Unlabeled")
            axes[row_index, 0].axvline(0.0, color="black", linewidth=0.8)
            axes[row_index, 0].set_title(f"{analysis['model']} — histogram")
            axes[row_index, 0].set_xlabel("Raw logit")
            axes[row_index, 0].set_ylabel("Density")
            axes[row_index, 0].legend(fontsize=7)
            for values, membership, linestyle in (
                (positive, "Observed Positive", "-"),
                (unlabeled, "Unlabeled", "--"),
            ):
                sorted_values = np.sort(values)
                axes[row_index, 1].plot(
                    sorted_values,
                    np.arange(1, len(sorted_values) + 1) / len(sorted_values),
                    linestyle=linestyle,
                    label=membership,
                )
            axes[row_index, 1].axvline(0.0, color="black", linewidth=0.8)
            axes[row_index, 1].set_title(f"{analysis['model']} — ECDF")
            axes[row_index, 1].set_xlabel("Raw logit")
            axes[row_index, 1].set_ylabel("Cumulative fraction")
            axes[row_index, 1].legend(fontsize=7)
        figure.suptitle(
            f"{task}: P-vs-U RAW LOGITS — VALIDATION ONLY\nNOT CALIBRATED PROBABILITIES"
            + (f"\n{availability_note}" if availability_note else ""),
            fontsize=13,
        )
        stem = f"{task}_score_distribution"
        figure.savefig(figures_directory / f"{stem}.png", dpi=600)
        figure.savefig(figures_directory / f"{stem}.pdf")
        plt.close(figure)

        for row in [analysis for analysis in analyses if analysis["task"] == task]:
            matrix = np.asarray(
                (
                    (
                        row["contingency"]["observed_positive"]["score_ge_0"],
                        row["contingency"]["observed_positive"]["score_lt_0"],
                    ),
                    (
                        row["contingency"]["unlabeled"]["score_ge_0"],
                        row["contingency"]["unlabeled"]["score_lt_0"],
                    ),
                )
            )
            figure, axis = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
            image = axis.imshow(matrix, cmap="Blues")
            for y in range(2):
                for x_index in range(2):
                    axis.text(x_index, y, str(int(matrix[y, x_index])), ha="center", va="center")
            axis.set_xticks((0, 1), ("score >= 0", "score < 0"))
            axis.set_yticks((0, 1), ("Observed Positive", "Unlabeled"))
            axis.set_title(
                f"{row['model']} / {row['split']} / {task}\nP-vs-U Diagnostic Contingency Matrix\nNOT A BINARY CONFUSION MATRIX"
            )
            figure.colorbar(image, ax=axis)
            safe_model = re.sub(r"[^a-z0-9]+", "_", row["model"].lower()).strip("_")
            stem = f"{safe_model}_{row['split']}_{task}_diagnostic_contingency"
            figure.savefig(contingency_directory / f"{stem}.png", dpi=600)
            figure.savefig(contingency_directory / f"{stem}.pdf")
            plt.close(figure)


def prepare_diagnostic_artifacts(
    *,
    project_root: str | Path,
    frozen_config: Mapping[str, Any],
    output_directory: str | Path,
    score_rows: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Write the validation-only report, tables, and permitted figures."""

    root = Path(project_root).resolve()
    output = Path(output_directory)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if frozen_config.get("formal_test_count") != 0:
        raise ValueError("formal test count must remain zero")
    if frozen_config.get("format") != "EVIHGT_VALIDATION_DIAGNOSTIC_CONFIG_V1":
        raise ValueError("unsupported validation diagnostic config")

    def load_model_rows(config_key: str, default_split: str) -> list[dict[str, Any]]:
        loaded = []
        for model in frozen_config.get(config_key, []):
            log_path = _resolve_path(root, str(model["log_path"]))
            parsed = parse_training_log(log_path.read_text(encoding="utf-8"))
            if parsed["run_id"] != model["run_id"]:
                raise ValueError("training log run identity differs from frozen config")
            best = min(
                parsed["cycles"],
                key=lambda row: (row["validation_objective"], row["cycle"]),
            )
            loaded.append(
                {
                    **dict(model),
                    "split": model.get("split", default_split),
                    **parsed,
                    "best_cycle": best["cycle"],
                    "best_validation_objective": best["validation_objective"],
                }
            )
        return loaded

    model_rows = load_model_rows("models", "standard")
    cold_model_rows = load_model_rows("cold_models", "cold")
    if not model_rows:
        raise ValueError("frozen diagnostic config has no standard models")
    _write_validation_objective_figure(model_rows, figures)

    materialized_scores = None if score_rows is None else validate_score_rows(score_rows)
    score_analysis = None if materialized_scores is None else analyze_score_rows(materialized_scores)
    if score_analysis is not None:
        configured = {
            (row["run_id"], row["label"], row["split"])
            for row in (*model_rows, *cold_model_rows)
        }
        observed = {
            (row["run_id"], row["model"], row["split"])
            for row in score_analysis["analyses"]
        }
        if observed != configured:
            raise ValueError("validation score runs differ from the frozen diagnostic config")
        _write_metric_figures(score_analysis["analyses"], materialized_scores, figures)

    csv_path = output / "validation_diagnostics.csv"
    columns = (
        "label",
        "model_variant",
        "run_id",
        "scores_available",
        "best_cycle",
        "best_validation_objective",
        "support_roc_auc",
        "support_pr_auc_average_precision",
        "unsuitability_roc_auc",
        "unsuitability_pr_auc_average_precision",
        "support_positive_mean_logit",
        "support_unlabeled_mean_logit",
        "unsuitability_positive_mean_logit",
        "unsuitability_unlabeled_mean_logit",
    )
    analyses_by_model_task = {}
    if score_analysis is not None:
        analyses_by_model_task = {
            (row["run_id"], row["task"]): row for row in score_analysis["analyses"]
        }
    def diagnostic_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for model in models:
            support = analyses_by_model_task.get((model["run_id"], "support"), {})
            unsuitability = analyses_by_model_task.get((model["run_id"], "unsuitability"), {})
            results.append(
                {
                    "label": model["label"],
                    "model_variant": model["model_variant"],
                    "run_id": model["run_id"],
                    "scores_available": score_analysis is not None,
                    "best_cycle": model["best_cycle"],
                    "best_validation_objective": model["best_validation_objective"],
                    "support_roc_auc": support.get("roc_auc", ""),
                    "support_pr_auc_average_precision": support.get("average_precision", ""),
                    "unsuitability_roc_auc": unsuitability.get("roc_auc", ""),
                    "unsuitability_pr_auc_average_precision": unsuitability.get("average_precision", ""),
                    "support_positive_mean_logit": support.get("positive_score_mean", ""),
                    "support_unlabeled_mean_logit": support.get("unlabeled_score_mean", ""),
                    "unsuitability_positive_mean_logit": unsuitability.get("positive_score_mean", ""),
                    "unsuitability_unlabeled_mean_logit": unsuitability.get("unlabeled_score_mean", ""),
                }
            )
        return results

    standard_diagnostics = diagnostic_rows(model_rows)
    cold_diagnostics = diagnostic_rows(cold_model_rows)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(standard_diagnostics)
    cold_csv_path = output / "cold_validation_diagnostics.csv"
    with cold_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cold_diagnostics)

    summary = {
        "status": "VALIDATION-ONLY DIAGNOSTIC / NOT FORMAL TEST PERFORMANCE",
        "formal_test_count": 0,
        "formal_test_accessed": False,
        "test_metrics_generated": False,
        "unknown_is_false": False,
        "unlabeled_is_negative": False,
        "scores_are_calibrated_probabilities": False,
        "diagnostic_semantics": "OBSERVED_POSITIVE_VS_UNLABELED",
        "existing_scores_available": score_analysis is not None,
        "validation_only_auc_aupr_generated": score_analysis is not None,
        "roc_curves_generated": score_analysis is not None,
        "pr_curves_generated": score_analysis is not None,
        "score_distribution_plots_generated": score_analysis is not None,
        "diagnostic_contingency_matrices_generated": score_analysis is not None,
        "validation_objective_comparison_generated": True,
        "models": model_rows,
        "cold_models": cold_model_rows,
        "standard_diagnostics": standard_diagnostics,
        "cold_diagnostics": cold_diagnostics,
        "cold_comparability": "NOT DIRECTLY COMPARABLE TO STANDARD SPLIT",
        "score_analysis": score_analysis,
    }
    (output / "validation_diagnostics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table_lines = [
        "| Model | BEST objective | Support ROC-AUC | Support PR-AUC | Unsuitability ROC-AUC | Unsuitability PR-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, diagnostic in zip(model_rows, standard_diagnostics, strict=True):
        def metric(name: str) -> str:
            value = diagnostic[name]
            return "not generated" if value == "" else f"{float(value):.6f}"

        table_lines.append(
            f"| {model['label']} | {model['best_validation_objective']:.9f} | "
            f"{metric('support_roc_auc')} | {metric('support_pr_auc_average_precision')} | "
            f"{metric('unsuitability_roc_auc')} | {metric('unsuitability_pr_auc_average_precision')} |"
        )
    cold_table_lines = [
        "| Model | Split | Support ROC-AUC | Support PR-AUC | Unsuitability ROC-AUC | Unsuitability PR-AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, diagnostic in zip(cold_model_rows, cold_diagnostics, strict=True):
        def cold_metric(name: str) -> str:
            value = diagnostic[name]
            return "not generated" if value == "" else f"{float(value):.6f}"

        cold_table_lines.append(
            f"| {model['label']} | {model['split']} | {cold_metric('support_roc_auc')} | "
            f"{cold_metric('support_pr_auc_average_precision')} | "
            f"{cold_metric('unsuitability_roc_auc')} | "
            f"{cold_metric('unsuitability_pr_auc_average_precision')} |"
        )
    report = "\n".join(
        (
            "# EviHGT Validation Diagnostic Report",
            "",
            "**VALIDATION-ONLY DIAGNOSTIC — NOT FORMAL TEST PERFORMANCE.**",
            "",
            "The formal test remains sealed (`formal test count = 0`; `test accessed = NO`). U is NOT confirmed negative. Unknown/unlabeled is not false. P-vs-U ROC-AUC and Average Precision are validation diagnostics only, not true binary classification metrics. Raw logits are not calibrated probabilities.",
            "",
            "## Standard validation diagnostics",
            "",
            *table_lines,
            "",
            "## Cold validation diagnostics",
            "",
            "**NOT DIRECTLY COMPARABLE TO STANDARD SPLIT.** Cold results use their own frozen visibility and validation semantics.",
            "",
            *cold_table_lines,
            "",
            "## Artifact semantics",
            "",
            f"Persisted per-pair raw validation scores available: **{'YES' if score_analysis is not None else 'NO'}**.",
            "",
            "- ROC false-positive rate is a P-vs-U proxy because U is not verified negative.",
            "- PR-AUC is reported as Average Precision over observed P versus protocol-defined U.",
            "- Threshold-zero tables are diagnostic contingency matrices, NOT binary confusion matrices.",
            "- No accuracy, specificity, true-negative rate, or binary F1 is generated.",
            "- The validation objective comparison retains its frozen values and lower-is-better meaning.",
            "",
            "No result in this report is formal test performance or a final retrieval-ranking conclusion.",
        )
    )
    (output / "VALIDATION_DIAGNOSTICS_REPORT.md").write_text(report + "\n", encoding="utf-8")
    return summary


__all__ = [
    "VALID_MEMBERSHIPS",
    "VALID_TASKS",
    "analyze_score_rows",
    "build_validation_score_rows",
    "compute_pu_diagnostics",
    "parse_training_log",
    "prepare_diagnostic_artifacts",
    "validate_score_rows",
]
