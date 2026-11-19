"""Disease-wise, future-label-safe evaluation candidate construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

import polars as pl

from evi_hgt.evaluation.protocol import (
    TRAINING_KNOWN_RELATIONS,
    EvaluationProtocolBinding,
    RankingCandidateProtocol,
)


REQUIRED_SUPERVISION_COLUMNS = {
    "drug_id",
    "disease_id",
    "raw_relation",
    "policy_b_member",
    "support_mask",
    "unsuitability_mask",
}
REQUIRED_SPLIT_COLUMNS = {
    "drug_id",
    "disease_id",
    "split_assignment",
    "drug_partition",
    "disease_partition",
}


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pairs_by_disease(frame: pl.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for drug_id, disease_id in frame.select("drug_id", "disease_id").iter_rows():
        result.setdefault(str(disease_id), set()).add(str(drug_id))
    return result


@dataclass(frozen=True, slots=True)
class DiseaseCandidateSet:
    split_kind: str
    partition: str
    channel: str
    disease_id: str
    candidate_drug_ids: tuple[str, ...]
    held_out_relevant_drug_ids: tuple[str, ...]
    training_known_excluded_drug_ids: tuple[str, ...]
    candidate_protocol_digest: str
    candidate_membership_digest: str
    definition_digest: str
    ground_truth_type: str = "HELD_OUT_KNOWN_POSITIVE_RELEVANCE"
    unlabeled_is_negative: bool = False
    global_cartesian_materialized: bool = False

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_drug_ids)

    @property
    def relevance_count(self) -> int:
        return len(self.held_out_relevant_drug_ids)

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "candidate_count": self.candidate_count,
                "relevance_count": self.relevance_count,
            }
        )
        return payload


def iter_disease_candidates(
    supervision: pl.DataFrame,
    split_frame: pl.DataFrame,
    *,
    split_kind: str,
    partition: str,
    channel: str,
    formal_drug_ids: Iterable[str] | None = None,
) -> Iterator[DiseaseCandidateSet]:
    """Yield one canonical candidate definition at a time.

    Candidate membership depends on endpoint partitions and training-observed DDA
    pairs only. Held-out labels are read solely to attach relevance after membership
    has been fixed.
    """

    protocol = RankingCandidateProtocol(split_kind, partition, channel)
    binding = EvaluationProtocolBinding(split_kind, partition, channel)
    if missing := REQUIRED_SUPERVISION_COLUMNS - set(supervision.columns):
        raise ValueError(f"supervision columns missing: {sorted(missing)}")
    if missing := REQUIRED_SPLIT_COLUMNS - set(split_frame.columns):
        raise ValueError(f"split columns missing: {sorted(missing)}")
    observed_relations = {
        str(value) for value in supervision["raw_relation"].unique().to_list()
    }
    allowed_relations = set(TRAINING_KNOWN_RELATIONS)
    if unexpected := observed_relations - allowed_relations:
        raise ValueError(f"unexpected DDA relations: {sorted(unexpected)}")

    source = supervision.select(sorted(REQUIRED_SUPERVISION_COLUMNS))
    split = split_frame.select(sorted(REQUIRED_SPLIT_COLUMNS))
    joined = source.join(
        split,
        on=["drug_id", "disease_id"],
        how="inner",
        validate="1:1",
    )
    if joined.height != supervision.height or joined.height != split_frame.height:
        raise ValueError("split manifest must bind every supervision pair exactly once")
    assignments = {
        str(value)
        for value in joined["split_assignment"].unique().to_list()
        if value is not None
    }
    if not assignments <= {"train", "validation", "test"}:
        raise ValueError(f"unexpected split assignments: {sorted(assignments)}")

    if formal_drug_ids is None:
        all_drugs = tuple(
            sorted(str(value) for value in supervision["drug_id"].unique().to_list())
        )
    else:
        all_drugs = tuple(sorted({str(value) for value in formal_drug_ids}))
    if not all_drugs:
        raise ValueError("formal candidate drug scope must not be empty")
    if protocol.candidate_drug_scope == "MATCHING_HELD_OUT_DRUG_PARTITION":
        scoped = {
            str(value)
            for value in split_frame.filter(pl.col("drug_partition") == partition)[
                "drug_id"
            ].unique().to_list()
        }
        candidate_drug_scope = tuple(value for value in all_drugs if value in scoped)
    else:
        candidate_drug_scope = all_drugs
    if not candidate_drug_scope:
        raise ValueError("candidate drug scope must not be empty")
    candidate_drug_set = set(candidate_drug_scope)

    if protocol.evaluation_disease_scope == "MATCHING_HELD_OUT_DISEASE_PARTITION":
        evaluation_diseases = tuple(
            sorted(
                str(value)
                for value in split_frame.filter(
                    pl.col("disease_partition") == partition
                )["disease_id"].unique().to_list()
            )
        )
    else:
        evaluation_diseases = tuple(
            sorted(
                str(value)
                for value in split_frame.filter(
                    pl.col("split_assignment") == partition
                )["disease_id"].unique().to_list()
            )
        )
    if not evaluation_diseases:
        raise ValueError("evaluation disease scope must not be empty")

    training = joined.filter(
        (pl.col("split_assignment") == "train")
        & pl.col("raw_relation").is_in(TRAINING_KNOWN_RELATIONS)
    )
    held_out = joined.filter(pl.col("split_assignment") == partition)
    if channel == "support":
        relevant = held_out.filter(pl.col("support_mask") & pl.col("policy_b_member"))
    else:
        relevant = held_out.filter(pl.col("unsuitability_mask"))
    training_by_disease = _pairs_by_disease(training)
    relevance_by_disease = _pairs_by_disease(relevant)

    for disease_id in evaluation_diseases:
        exclusions = training_by_disease.get(disease_id, set()) & candidate_drug_set
        unscoped_relevance = relevance_by_disease.get(disease_id, set())
        outside_scope = unscoped_relevance - candidate_drug_set
        if outside_scope:
            raise ValueError(
                "held-out relevance falls outside the split-safe candidate drug scope"
            )
        candidate_ids = tuple(
            drug_id for drug_id in candidate_drug_scope if drug_id not in exclusions
        )
        if not candidate_ids:
            raise ValueError(f"candidate set is empty for disease {disease_id}")
        relevant_ids = tuple(sorted(unscoped_relevance))
        if not set(relevant_ids) <= set(candidate_ids):
            raise ValueError("held-out relevance must remain in the candidate set")
        excluded_ids = tuple(sorted(exclusions))
        membership_payload = {
            "candidate_universe_version": protocol.candidate_universe_version,
            "split_kind": split_kind,
            "partition": partition,
            "disease_id": disease_id,
            "candidate_drug_ids": candidate_ids,
            "training_known_excluded_drug_ids": excluded_ids,
        }
        membership_digest = _digest(membership_payload)
        definition_payload = {
            "binding": binding.to_payload(),
            "candidate_membership_digest": membership_digest,
            "held_out_relevant_drug_ids": relevant_ids,
        }
        yield DiseaseCandidateSet(
            split_kind=split_kind,
            partition=partition,
            channel=channel,
            disease_id=disease_id,
            candidate_drug_ids=candidate_ids,
            held_out_relevant_drug_ids=relevant_ids,
            training_known_excluded_drug_ids=excluded_ids,
            candidate_protocol_digest=protocol.digest,
            candidate_membership_digest=membership_digest,
            definition_digest=_digest(definition_payload),
        )


__all__ = ["DiseaseCandidateSet", "iter_disease_candidates"]
