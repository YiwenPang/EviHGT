"""Frozen disease-ranking evaluation semantics without metric computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


EVALUATION_PROTOCOL_ID = "KNOWN_POSITIVE_RETRIEVAL_V1"
PRIMARY_EVALUATION = "KNOWN_POSITIVE_RETRIEVAL"
PRIMARY_OBJECTIVE_ID = "SYMMETRIC_PU_PAIRWISE_RANKING_V1"
PRIMARY_SUPPORT_RANKING_SCORE = "support_logit"
PRIMARY_UNSUITABILITY_RANKING_SCORE = "unsuitability_logit"
SECONDARY_LOGIC_SUPPORT_SCORE = "m_T"
SECONDARY_LOGIC_UNSUITABILITY_SCORE = "m_F"
VALIDATION_MANIFEST_ID = "DETERMINISTIC_VALIDATION_PU_MANIFEST_V1"
VALIDATION_BASE_SEED = 73101
VALIDATION_POSITIVE_COUNT = 128
VALIDATION_UNLABELED_COUNT = 512
PAIR_SEPARATOR = "\x1f"
RANKING_K = (5, 10, 20, 50)
VALID_SPLITS = ("standard", "cold_drug", "cold_disease", "double_cold")
VALID_PARTITIONS = ("validation", "test")
VALID_CHANNELS = ("support", "unsuitability")
TRAINING_KNOWN_RELATIONS = (
    "INDICATION",
    "CONTRAINDICATION",
    "OFF_LABEL_USE",
)
BINARY_METRIC_SKIP_STATUS = "SKIPPED_NO_VERIFIED_BINARY_GROUND_TRUTH"
CALIBRATION_METRIC_SKIP_STATUS = (
    "SKIPPED_NO_VALIDATED_PROBABILITY_CALIBRATION"
)
PRIMARY_RANKING_STATUS = "PRIMARY_KNOWN_POSITIVE_RETRIEVAL"
SECONDARY_PU_DIAGNOSTIC_STATUS = "SECONDARY_PROTOCOL_SPECIFIC_DIAGNOSTIC"
ZERO_RELEVANCE_STATUS = "SKIPPED_NO_HELD_OUT_RELEVANCE"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RankingScoreDecision:
    """Primary channel logits and secondary latent logic-score roles."""

    primary_support_ranking_score: str = PRIMARY_SUPPORT_RANKING_SCORE
    primary_unsuitability_ranking_score: str = (
        PRIMARY_UNSUITABILITY_RANKING_SCORE
    )
    secondary_logic_support_score: str = SECONDARY_LOGIC_SUPPORT_SCORE
    secondary_logic_unsuitability_score: str = (
        SECONDARY_LOGIC_UNSUITABILITY_SCORE
    )
    secondary_logic_state_scores: tuple[str, str] = ("m_B", "m_U")
    sigmoid_scores_are_monotone_display_only: bool = True
    scores_are_calibrated_probabilities: bool = False
    logic_masses_are_observed_labels: bool = False

    def __post_init__(self) -> None:
        if self.primary_support_ranking_score != PRIMARY_SUPPORT_RANKING_SCORE:
            raise ValueError("primary support ranking must use support_logit")
        if (
            self.primary_unsuitability_ranking_score
            != PRIMARY_UNSUITABILITY_RANKING_SCORE
        ):
            raise ValueError(
                "primary unsuitability ranking must use unsuitability_logit"
            )
        if self.secondary_logic_support_score != SECONDARY_LOGIC_SUPPORT_SCORE:
            raise ValueError("m_T is frozen as a secondary logic-aware score")
        if (
            self.secondary_logic_unsuitability_score
            != SECONDARY_LOGIC_UNSUITABILITY_SCORE
        ):
            raise ValueError("m_F is frozen as a secondary logic-aware score")
        if self.secondary_logic_state_scores != ("m_B", "m_U"):
            raise ValueError("m_B/m_U are frozen for latent-state analysis")
        if not self.sigmoid_scores_are_monotone_display_only:
            raise ValueError("sigmoid scores are display-only monotone transforms")
        if self.scores_are_calibrated_probabilities:
            raise ValueError("Ranking scores are not calibrated probabilities")
        if self.logic_masses_are_observed_labels:
            raise ValueError("latent logic masses are not observed labels")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return _digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class MetricValidityRecord:
    """Validity and interpretation of one metric family member."""

    metric: str
    family: str
    status: str
    valid: bool
    requires_verified_negatives: bool
    requires_probability_calibration: bool
    unlabeled_to_zero_allowed: bool
    interpretation: str
    comparison_scope: str | None = None

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def metric_validity_matrix() -> tuple[MetricValidityRecord, ...]:
    """Return the metric decision matrix for each supported split."""

    primary = tuple(
        MetricValidityRecord(
            metric=name,
            family="KNOWN_POSITIVE_RETRIEVAL",
            status=PRIMARY_RANKING_STATUS,
            valid=True,
            requires_verified_negatives=False,
            requires_probability_calibration=False,
            unlabeled_to_zero_allowed=False,
            interpretation=(
                "HELD_OUT_KNOWN_POSITIVE_RETRIEVAL_NOT_TRUE_BINARY_CLASSIFICATION"
            ),
        )
        for name in (
            "Known-Positive Precision@K",
            "Known-Positive Recall@K",
            "Hits@K",
            "NDCG@K",
        )
    )
    binary = tuple(
        MetricValidityRecord(
            metric=name,
            family="BINARY_CLASSIFICATION",
            status=BINARY_METRIC_SKIP_STATUS,
            valid=False,
            requires_verified_negatives=True,
            requires_probability_calibration=False,
            unlabeled_to_zero_allowed=False,
            interpretation="UNAVAILABLE_WITHOUT_VERIFIED_BINARY_GROUND_TRUTH",
        )
        for name in (
            "Accuracy",
            "binary Precision",
            "binary Recall",
            "F1",
            "MCC",
            "Balanced Accuracy",
            "Specificity",
            "Confusion Matrix",
        )
    )
    calibration = tuple(
        MetricValidityRecord(
            metric=name,
            family="PROBABILITY_CALIBRATION",
            status=CALIBRATION_METRIC_SKIP_STATUS,
            valid=False,
            requires_verified_negatives=True,
            requires_probability_calibration=True,
            unlabeled_to_zero_allowed=False,
            interpretation="UNAVAILABLE_WITHOUT_VALIDATED_PROBABILITY_CALIBRATION",
        )
        for name in (
            "Brier Score",
            "ECE",
            "NLL calibration claim",
            "Reliability Diagram",
        )
    )
    pu_diagnostics = tuple(
        MetricValidityRecord(
            metric=name,
            family="POSITIVE_UNLABELED_COMPARISON",
            status=SECONDARY_PU_DIAGNOSTIC_STATUS,
            valid=True,
            requires_verified_negatives=False,
            requires_probability_calibration=False,
            unlabeled_to_zero_allowed=False,
            interpretation=(
                "KNOWN_POSITIVE_VS_PROTOCOL_DEFINED_UNLABELED_NOT_TRUE_BINARY"
            ),
            comparison_scope="PU_FULL_CANDIDATE_COMPARISON",
        )
        for name in ("P-vs-U AUROC", "P-vs-U AUPRC")
    )
    return (*primary, *binary, *calibration, *pu_diagnostics)


@dataclass(frozen=True, slots=True)
class RankingCandidateProtocol:
    """Split-bound disease-wise known-positive retrieval candidate rules."""

    split_kind: str
    partition: str
    channel: str
    candidate_universe_version: str = "formal-dda-endpoints-v1"
    support_policy: str = "B"
    primary_evaluation: str = PRIMARY_EVALUATION
    ranking_k: tuple[int, ...] = RANKING_K
    disease_wise: bool = True
    global_cartesian_materialized: bool = False
    training_known_exclusion_relations: tuple[str, ...] = TRAINING_KNOWN_RELATIONS
    validation_test_label_pruning: bool = False
    held_out_relevance_must_remain: bool = True
    aggregation: str = "MACRO_ACROSS_EVALUABLE_DISEASES"
    tie_break: tuple[str, str] = (
        "SCORE_DESCENDING",
        "CANONICAL_DRUG_ID_ASCENDING",
    )
    zero_relevance_status: str = ZERO_RELEVANCE_STATUS
    non_relevant_interpretation: str = (
        "NOT_A_HELD_OUT_KNOWN_POSITIVE_UNDER_BENCHMARK_NOT_VERIFIED_NEGATIVE"
    )

    def __post_init__(self) -> None:
        if self.split_kind not in VALID_SPLITS:
            raise ValueError(f"unsupported split: {self.split_kind}")
        if self.partition not in VALID_PARTITIONS:
            raise ValueError(f"unsupported partition: {self.partition}")
        if self.channel not in VALID_CHANNELS:
            raise ValueError(f"unsupported channel: {self.channel}")
        if self.candidate_universe_version != "formal-dda-endpoints-v1":
            raise ValueError("candidate universe version is frozen")
        if self.support_policy != "B":
            raise ValueError("support Policy B is frozen")
        if self.primary_evaluation != PRIMARY_EVALUATION:
            raise ValueError("primary evaluation family is frozen")
        if self.ranking_k != RANKING_K:
            raise ValueError("ranking K values are frozen")
        if not self.disease_wise or self.global_cartesian_materialized:
            raise ValueError("ranking must use disease-wise lazy construction")
        if self.training_known_exclusion_relations != TRAINING_KNOWN_RELATIONS:
            raise ValueError("training-known DDA exclusion relations are frozen")
        if self.validation_test_label_pruning:
            raise ValueError("future-label candidate pruning is forbidden")
        if not self.held_out_relevance_must_remain:
            raise ValueError("held-out relevance must remain in candidates")
        if self.aggregation != "MACRO_ACROSS_EVALUABLE_DISEASES":
            raise ValueError("macro aggregation is frozen")
        if self.tie_break != (
            "SCORE_DESCENDING",
            "CANONICAL_DRUG_ID_ASCENDING",
        ):
            raise ValueError("deterministic tie breaking is frozen")
        if self.zero_relevance_status != ZERO_RELEVANCE_STATUS:
            raise ValueError("zero-relevance skip behavior is frozen")

    @property
    def candidate_drug_scope(self) -> str:
        if self.split_kind in {"cold_drug", "double_cold"}:
            return "MATCHING_HELD_OUT_DRUG_PARTITION"
        return "ALL_ELIGIBLE_DRUGS"

    @property
    def evaluation_disease_scope(self) -> str:
        if self.split_kind in {"cold_disease", "double_cold"}:
            return "MATCHING_HELD_OUT_DISEASE_PARTITION"
        return "OBSERVED_HELD_OUT_PARTITION_DISEASES"

    @property
    def relevance_definition(self) -> str:
        if self.channel == "support":
            return "HELD_OUT_POLICY_B_SUPPORT_POSITIVE"
        return "HELD_OUT_RETAINED_CONTRAINDICATION_OBSERVATION"

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "candidate_drug_scope": self.candidate_drug_scope,
                "evaluation_disease_scope": self.evaluation_disease_scope,
                "relevance_definition": self.relevance_definition,
            }
        )
        return payload

    @property
    def digest(self) -> str:
        return _digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class EvaluationProtocolBinding:
    """Complete channel/split/partition identity for later evaluator use."""

    split_kind: str
    partition: str
    channel: str
    evaluation_protocol_id: str = EVALUATION_PROTOCOL_ID
    primary_evaluation: str = PRIMARY_EVALUATION
    learning_objective_id: str = PRIMARY_OBJECTIVE_ID

    def __post_init__(self) -> None:
        RankingCandidateProtocol(self.split_kind, self.partition, self.channel)
        if self.evaluation_protocol_id != EVALUATION_PROTOCOL_ID:
            raise ValueError("evaluation protocol ID is frozen")
        if self.primary_evaluation != PRIMARY_EVALUATION:
            raise ValueError("primary evaluation is frozen")
        if self.learning_objective_id != PRIMARY_OBJECTIVE_ID:
            raise ValueError("validation learning objective is frozen")

    @property
    def candidate_protocol(self) -> RankingCandidateProtocol:
        return RankingCandidateProtocol(
            split_kind=self.split_kind,
            partition=self.partition,
            channel=self.channel,
        )

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["score_decision"] = RankingScoreDecision().to_payload()
        payload["candidate_protocol"] = self.candidate_protocol.to_payload()
        payload["metric_validity_digest"] = _digest(
            [record.to_payload() for record in metric_validity_matrix()]
        )
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_payload())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DeterministicValidationPUManifest:
    """Fixed validation P/U identities for the validation objective."""

    split_kind: str
    channel: str
    positive_pair_ids: tuple[str, ...]
    unlabeled_pair_ids: tuple[str, ...]
    candidate_universe_digest: str
    namespace_seed: int
    manifest_id: str = VALIDATION_MANIFEST_ID
    partition: str = "validation"
    objective_id: str = PRIMARY_OBJECTIVE_ID
    base_seed: int = VALIDATION_BASE_SEED
    fixed_across_epochs: bool = True
    used_for_training: bool = False
    test_labels_used: bool = False
    unlabeled_is_negative: bool = False

    def __post_init__(self) -> None:
        if self.split_kind not in VALID_SPLITS:
            raise ValueError(f"unsupported split: {self.split_kind}")
        if self.channel not in VALID_CHANNELS:
            raise ValueError(f"unsupported channel: {self.channel}")
        if self.manifest_id != VALIDATION_MANIFEST_ID:
            raise ValueError("validation manifest ID is frozen")
        if self.partition != "validation":
            raise ValueError("validation manifest partition is frozen")
        if self.objective_id != PRIMARY_OBJECTIVE_ID:
            raise ValueError("validation objective ID is frozen")
        if self.base_seed != VALIDATION_BASE_SEED:
            raise ValueError("validation base seed is frozen")
        if len(self.positive_pair_ids) != VALIDATION_POSITIVE_COUNT:
            raise ValueError("validation manifest requires exactly 128 positives")
        if len(self.unlabeled_pair_ids) != VALIDATION_UNLABELED_COUNT:
            raise ValueError("validation manifest requires exactly 512 unlabeled pairs")
        if len(set(self.positive_pair_ids)) != len(self.positive_pair_ids):
            raise ValueError("validation positive pair IDs must be unique")
        if len(set(self.unlabeled_pair_ids)) != len(self.unlabeled_pair_ids):
            raise ValueError("validation unlabeled pair IDs must be unique")
        if set(self.positive_pair_ids) & set(self.unlabeled_pair_ids):
            raise ValueError("validation P and U streams must be disjoint")
        for pair_id in (*self.positive_pair_ids, *self.unlabeled_pair_ids):
            parts = pair_id.split(PAIR_SEPARATOR)
            if len(parts) != 2 or not all(parts):
                raise ValueError("validation pair IDs must use canonical separator")
        if (
            len(self.candidate_universe_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.candidate_universe_digest
            )
        ):
            raise ValueError("candidate universe digest must be lowercase SHA-256")
        if self.namespace_seed < 0:
            raise ValueError("validation namespace seed must be non-negative")
        if not self.fixed_across_epochs:
            raise ValueError("validation manifest must remain fixed across epochs")
        if self.used_for_training:
            raise ValueError("validation manifest cannot be used for training")
        if self.test_labels_used:
            raise ValueError("validation manifest cannot use test labels")
        if self.unlabeled_is_negative:
            raise ValueError("validation U pairs are not negative labels")

    @property
    def exact_pair_comparison_count(self) -> int:
        return len(self.positive_pair_ids) * len(self.unlabeled_pair_ids)

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "positive_count": len(self.positive_pair_ids),
                "unlabeled_count": len(self.unlabeled_pair_ids),
                "exact_pair_comparison_count": self.exact_pair_comparison_count,
            }
        )
        return payload

    @property
    def digest(self) -> str:
        return _digest(self.to_payload())


def effective_k(requested_k: int, candidate_count: int) -> int:
    """Return the frozen candidate-bounded K while retaining requested K."""

    requested = int(requested_k)
    count = int(candidate_count)
    if requested not in RANKING_K:
        raise ValueError("requested K is not a frozen ranking value")
    if count < 0:
        raise ValueError("candidate count must be non-negative")
    return min(requested, count)
