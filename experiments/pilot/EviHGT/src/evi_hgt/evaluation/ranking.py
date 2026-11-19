"""Deterministic known-positive ranking metrics for the frozen evaluation protocol."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from evi_hgt.evaluation.protocol import (
    BINARY_METRIC_SKIP_STATUS,
    CALIBRATION_METRIC_SKIP_STATUS,
    PRIMARY_RANKING_STATUS,
    RANKING_K,
    SECONDARY_PU_DIAGNOSTIC_STATUS,
    ZERO_RELEVANCE_STATUS,
    EvaluationProtocolBinding,
    effective_k,
    metric_validity_matrix,
)


HELD_OUT_RELEVANCE_GROUND_TRUTH_TYPE = "HELD_OUT_KNOWN_POSITIVE_RELEVANCE"
PU_GROUND_TRUTH_TYPE = "KNOWN_POSITIVE_VS_UNLABELED"
BINARY_GROUND_TRUTH_TYPE = "VERIFIED_BINARY_GROUND_TRUTH_REQUIRED"


@dataclass(frozen=True, slots=True)
class KnownPositiveMetricsAtK:
    requested_k: int
    effective_k: int
    candidate_count: int
    relevance_count: int
    hit_count: int
    known_positive_precision: float
    known_positive_recall: float
    hits_at_k: float
    ndcg: float

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiseaseRankingResult:
    split_kind: str
    partition: str
    channel: str
    disease_id: str
    status: str
    ground_truth_type: str
    candidate_count: int
    relevance_count: int
    ranked_drug_ids: tuple[str, ...]
    metrics: tuple[KnownPositiveMetricsAtK, ...]
    non_relevant_is_verified_negative: bool = False

    @property
    def evaluable(self) -> bool:
        return self.status == PRIMARY_RANKING_STATUS

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["evaluable"] = self.evaluable
        return payload


@dataclass(frozen=True, slots=True)
class MacroMetricsAtK:
    requested_k: int
    number_evaluable_diseases: int
    known_positive_precision_mean: float
    known_positive_recall_mean: float
    hits_at_k_mean: float
    ndcg_mean: float
    known_positive_precision_std: float | None
    known_positive_recall_std: float | None
    hits_at_k_std: float | None
    ndcg_std: float | None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MacroRankingResult:
    split_kind: str
    partition: str
    channel: str
    status: str
    ground_truth_type: str
    total_disease_count: int
    number_evaluable_diseases: int
    number_skipped_diseases: int
    metrics: tuple[MacroMetricsAtK, ...]
    aggregation: str = "MACRO_ACROSS_EVALUABLE_DISEASES"

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PUDiagnosticResult:
    status: str
    ground_truth_type: str
    positive_count: int
    unlabeled_count: int
    auroc: float
    auprc: float
    unlabeled_is_negative: bool = False
    true_binary_performance: bool = False

    @property
    def metric_names(self) -> tuple[str, str]:
        return ("P_VS_U_AUROC", "P_VS_U_AUPRC")

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "metric_names": self.metric_names,
                "P_VS_U_AUROC": self.auroc,
                "P_VS_U_AUPRC": self.auprc,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class MetricRequestDecision:
    metric: str
    family: str
    status: str
    valid: bool
    ground_truth_type: str
    interpretation: str

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def _finite_scores(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    scores = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in scores):
        raise ValueError(f"{name} scores must be finite")
    return scores


def evaluate_disease_ranking(
    *,
    binding: EvaluationProtocolBinding,
    disease_id: str,
    candidate_drug_ids: Sequence[str],
    scores: Sequence[float],
    relevant_drug_ids: Iterable[str],
) -> DiseaseRankingResult:
    """Evaluate one disease without treating non-relevant candidates as negatives."""

    if any(not isinstance(value, str) or not value.strip() for value in candidate_drug_ids):
        raise ValueError("candidate identifier must be a non-empty canonical string")
    candidate_ids = tuple(candidate_drug_ids)
    if not candidate_ids:
        raise ValueError("candidate drug set must not be empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate drug IDs must be unique")
    numeric_scores = _finite_scores(scores, name="ranking")
    if len(numeric_scores) != len(candidate_ids):
        raise ValueError("candidate drug IDs and scores must have equal length")
    relevance = {str(value) for value in relevant_drug_ids}
    unknown_relevance = relevance - set(candidate_ids)
    if unknown_relevance:
        raise ValueError("held-out relevance must be contained in the candidate set")
    ranked = tuple(
        drug_id
        for drug_id, _ in sorted(
            zip(candidate_ids, numeric_scores, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
    )
    if not relevance:
        return DiseaseRankingResult(
            split_kind=binding.split_kind,
            partition=binding.partition,
            channel=binding.channel,
            disease_id=str(disease_id),
            status=ZERO_RELEVANCE_STATUS,
            ground_truth_type=HELD_OUT_RELEVANCE_GROUND_TRUTH_TYPE,
            candidate_count=len(candidate_ids),
            relevance_count=0,
            ranked_drug_ids=ranked,
            metrics=(),
        )

    results: list[KnownPositiveMetricsAtK] = []
    for requested_k in RANKING_K:
        bounded_k = effective_k(requested_k, len(ranked))
        top = ranked[:bounded_k]
        hits = sum(drug_id in relevance for drug_id in top)
        dcg = sum(
            1.0 / math.log2(rank + 1.0)
            for rank, drug_id in enumerate(top, start=1)
            if drug_id in relevance
        )
        ideal_hits = min(len(relevance), bounded_k)
        idcg = sum(
            1.0 / math.log2(rank + 1.0)
            for rank in range(1, ideal_hits + 1)
        )
        results.append(
            KnownPositiveMetricsAtK(
                requested_k=requested_k,
                effective_k=bounded_k,
                candidate_count=len(ranked),
                relevance_count=len(relevance),
                hit_count=hits,
                known_positive_precision=hits / bounded_k,
                known_positive_recall=hits / len(relevance),
                hits_at_k=float(hits > 0),
                ndcg=dcg / idcg,
            )
        )
    return DiseaseRankingResult(
        split_kind=binding.split_kind,
        partition=binding.partition,
        channel=binding.channel,
        disease_id=str(disease_id),
        status=PRIMARY_RANKING_STATUS,
        ground_truth_type=HELD_OUT_RELEVANCE_GROUND_TRUTH_TYPE,
        candidate_count=len(candidate_ids),
        relevance_count=len(relevance),
        ranked_drug_ids=ranked,
        metrics=tuple(results),
    )


def _mean_std(values: Sequence[float], include_std: bool) -> tuple[float, float | None]:
    return (
        float(statistics.fmean(values)),
        float(statistics.pstdev(values)) if include_std else None,
    )


def aggregate_disease_rankings(
    results: Iterable[DiseaseRankingResult], *, include_std: bool = True
) -> MacroRankingResult:
    """Macro aggregate only evaluable diseases; skipped rows never become zeros."""

    rows = tuple(results)
    if not rows:
        raise ValueError("at least one disease ranking result is required")
    identity = {(row.split_kind, row.partition, row.channel) for row in rows}
    if len(identity) != 1:
        raise ValueError("macro aggregation requires one split/partition/channel")
    evaluable = tuple(row for row in rows if row.evaluable)
    skipped = tuple(row for row in rows if not row.evaluable)
    if any(row.status != ZERO_RELEVANCE_STATUS for row in skipped):
        raise ValueError("only zero-relevance disease results may be skipped")
    split_kind, partition, channel = next(iter(identity))
    metrics: list[MacroMetricsAtK] = []
    if evaluable:
        by_k = {
            requested_k: tuple(
                next(item for item in row.metrics if item.requested_k == requested_k)
                for row in evaluable
            )
            for requested_k in RANKING_K
        }
        for requested_k, values in by_k.items():
            precision_mean, precision_std = _mean_std(
                [value.known_positive_precision for value in values], include_std
            )
            recall_mean, recall_std = _mean_std(
                [value.known_positive_recall for value in values], include_std
            )
            hits_mean, hits_std = _mean_std(
                [value.hits_at_k for value in values], include_std
            )
            ndcg_mean, ndcg_std = _mean_std(
                [value.ndcg for value in values], include_std
            )
            metrics.append(
                MacroMetricsAtK(
                    requested_k=requested_k,
                    number_evaluable_diseases=len(evaluable),
                    known_positive_precision_mean=precision_mean,
                    known_positive_recall_mean=recall_mean,
                    hits_at_k_mean=hits_mean,
                    ndcg_mean=ndcg_mean,
                    known_positive_precision_std=precision_std,
                    known_positive_recall_std=recall_std,
                    hits_at_k_std=hits_std,
                    ndcg_std=ndcg_std,
                )
            )
    return MacroRankingResult(
        split_kind=split_kind,
        partition=partition,
        channel=channel,
        status=(PRIMARY_RANKING_STATUS if evaluable else ZERO_RELEVANCE_STATUS),
        ground_truth_type=HELD_OUT_RELEVANCE_GROUND_TRUTH_TYPE,
        total_disease_count=len(rows),
        number_evaluable_diseases=len(evaluable),
        number_skipped_diseases=len(skipped),
        metrics=tuple(metrics),
    )


def evaluate_p_vs_u_diagnostic(
    *, positive_scores: Sequence[float], unlabeled_scores: Sequence[float]
) -> PUDiagnosticResult:
    """Compute protocol-specific P-vs-U discrimination without a binary claim."""

    positives = _finite_scores(positive_scores, name="positive")
    unlabeled = _finite_scores(unlabeled_scores, name="unlabeled")
    if not positives or not unlabeled:
        raise ValueError("P-vs-U diagnostic requires non-empty positive and unlabeled sets")
    concordance = sum(
        1.0 if positive > unknown else 0.5 if positive == unknown else 0.0
        for positive in positives
        for unknown in unlabeled
    )
    auroc = concordance / (len(positives) * len(unlabeled))

    grouped: dict[float, list[int]] = {}
    for score in positives:
        grouped.setdefault(score, [0, 0])[0] += 1
    for score in unlabeled:
        grouped.setdefault(score, [0, 0])[1] += 1
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    auprc = 0.0
    for score in sorted(grouped, reverse=True):
        positive_count, unlabeled_count = grouped[score]
        true_positive += positive_count
        false_positive += unlabeled_count
        recall = true_positive / len(positives)
        precision = true_positive / (true_positive + false_positive)
        auprc += (recall - previous_recall) * precision
        previous_recall = recall
    return PUDiagnosticResult(
        status=SECONDARY_PU_DIAGNOSTIC_STATUS,
        ground_truth_type=PU_GROUND_TRUTH_TYPE,
        positive_count=len(positives),
        unlabeled_count=len(unlabeled),
        auroc=float(auroc),
        auprc=float(auprc),
    )


def metric_request_decision(metric: str) -> MetricRequestDecision:
    """Return the frozen validity decision instead of computing invalid metrics."""

    requested = str(metric)
    for record in metric_validity_matrix():
        if record.metric == requested:
            if record.family == "KNOWN_POSITIVE_RETRIEVAL":
                ground_truth_type = HELD_OUT_RELEVANCE_GROUND_TRUTH_TYPE
            elif record.family == "POSITIVE_UNLABELED_COMPARISON":
                ground_truth_type = PU_GROUND_TRUTH_TYPE
            else:
                ground_truth_type = BINARY_GROUND_TRUTH_TYPE
            return MetricRequestDecision(
                metric=record.metric,
                family=record.family,
                status=record.status,
                valid=record.valid,
                ground_truth_type=ground_truth_type,
                interpretation=record.interpretation,
            )
    raise ValueError(f"metric is not present in the frozen validity matrix: {requested}")


__all__ = [
    "BINARY_GROUND_TRUTH_TYPE",
    "HELD_OUT_RELEVANCE_GROUND_TRUTH_TYPE",
    "PU_GROUND_TRUTH_TYPE",
    "DiseaseRankingResult",
    "KnownPositiveMetricsAtK",
    "MacroMetricsAtK",
    "MacroRankingResult",
    "MetricRequestDecision",
    "PUDiagnosticResult",
    "aggregate_disease_rankings",
    "evaluate_disease_ranking",
    "evaluate_p_vs_u_diagnostic",
    "metric_request_decision",
]
