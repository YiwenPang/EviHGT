"""Deterministic canonical-pair and cold-start split manifests."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import polars as pl


SPLIT_KINDS = {"standard", "cold_drug", "cold_disease", "double_cold"}
ASSIGNED_SPLITS = ("train", "validation", "test")


class SplitInvariantError(ValueError):
    """Raised when a split would permit pair or cold-start leakage."""


def _score(value: str, *, seed: int, namespace: str) -> float:
    payload = f"EviHGT-Phase3-v1\x1f{seed}\x1f{namespace}\x1f{value}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def _assignment(score: float, validation_ratio: float, test_ratio: float) -> str:
    if score < test_ratio:
        return "test"
    if score < test_ratio + validation_ratio:
        return "validation"
    return "train"


def _validate_ratios(validation_ratio: float, test_ratio: float) -> None:
    if validation_ratio <= 0 or test_ratio <= 0:
        raise SplitInvariantError("Validation and test ratios must be positive")
    if validation_ratio + test_ratio >= 1:
        raise SplitInvariantError("Validation plus test ratio must be less than one")


def _entity_assignments(
    values: Iterable[str],
    *,
    name: str,
    seed: int,
    validation_ratio: float,
    test_ratio: float,
) -> pl.DataFrame:
    unique = sorted(set(values))
    return pl.DataFrame(
        {
            name: unique,
            f"_{name}_split": [
                _assignment(
                    _score(value, seed=seed, namespace=name),
                    validation_ratio,
                    test_ratio,
                )
                for value in unique
            ],
        }
    )


def build_split(
    pairs: pl.DataFrame,
    *,
    kind: str,
    seed: int,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> pl.DataFrame:
    """Assign canonical DDA pairs using a stable SHA-256 based pseudo-random score."""

    if kind not in SPLIT_KINDS:
        raise SplitInvariantError(f"Unknown split kind: {kind}")
    _validate_ratios(validation_ratio, test_ratio)
    required = {"drug_id", "disease_id", "canonical_pair_key"}
    missing = required - set(pairs.columns)
    if missing:
        raise SplitInvariantError(f"Missing split columns: {sorted(missing)}")
    if pairs["canonical_pair_key"].n_unique() != pairs.height:
        raise SplitInvariantError("Canonical pair keys must be unique before splitting")

    if kind == "standard":
        assignment = [
            _assignment(
                _score(str(value), seed=seed, namespace="canonical_pair"),
                validation_ratio,
                test_ratio,
            )
            for value in pairs["canonical_pair_key"]
        ]
        result = pairs.with_columns(
            pl.Series("split_assignment", assignment, dtype=pl.String),
            pl.lit(None, dtype=pl.String).alias("split_exclusion_reason"),
            pl.lit(None, dtype=pl.String).alias("drug_partition"),
            pl.lit(None, dtype=pl.String).alias("disease_partition"),
        )
    elif kind in {"cold_drug", "cold_disease"}:
        entity = "drug_id" if kind == "cold_drug" else "disease_id"
        mapping = _entity_assignments(
            pairs[entity].to_list(),
            name=entity,
            seed=seed,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        partition_column = "drug_partition" if entity == "drug_id" else "disease_partition"
        other_partition = "disease_partition" if entity == "drug_id" else "drug_partition"
        result = pairs.join(mapping, on=entity, how="left", validate="m:1").with_columns(
            pl.col(f"_{entity}_split").alias("split_assignment"),
            pl.col(f"_{entity}_split").alias(partition_column),
            pl.lit(None, dtype=pl.String).alias(other_partition),
            pl.lit(None, dtype=pl.String).alias("split_exclusion_reason"),
        ).drop(f"_{entity}_split")
    else:
        drug_mapping = _entity_assignments(
            pairs["drug_id"].to_list(),
            name="drug_id",
            seed=seed,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        disease_mapping = _entity_assignments(
            pairs["disease_id"].to_list(),
            name="disease_id",
            seed=seed,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )
        result = (
            pairs.join(drug_mapping, on="drug_id", how="left", validate="m:1")
            .join(disease_mapping, on="disease_id", how="left", validate="m:1")
            .rename(
                {
                    "_drug_id_split": "drug_partition",
                    "_disease_id_split": "disease_partition",
                }
            )
            .with_columns(
                pl.when(pl.col("drug_partition") == pl.col("disease_partition"))
                .then(pl.col("drug_partition"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("split_assignment"),
                pl.when(pl.col("drug_partition") != pl.col("disease_partition"))
                .then(pl.lit("CROSS_PARTITION_ENDPOINTS"))
                .otherwise(pl.lit(None, dtype=pl.String))
                .alias("split_exclusion_reason"),
            )
        )

    result = result.with_columns(
        pl.lit(kind).alias("split_kind"),
        pl.lit(seed, dtype=pl.Int64).alias("split_seed"),
        pl.lit(validation_ratio).alias("validation_ratio"),
        pl.lit(test_ratio).alias("test_ratio"),
        pl.lit("sha256-threshold-v1").alias("split_algorithm"),
    ).sort("canonical_pair_key")
    assert_split_invariants(result, kind)
    return result


def _entity_sets(frame: pl.DataFrame, column: str) -> dict[str, set[str]]:
    return {
        split: set(
            frame.filter(pl.col("split_assignment") == split)[column].to_list()
        )
        for split in ASSIGNED_SPLITS
    }


def _assert_disjoint(sets: dict[str, set[str]], label: str) -> None:
    for index, left_name in enumerate(ASSIGNED_SPLITS):
        for right_name in ASSIGNED_SPLITS[index + 1 :]:
            overlap = sets[left_name] & sets[right_name]
            if overlap:
                raise SplitInvariantError(
                    f"{label} leakage between {left_name}/{right_name}: "
                    f"{len(overlap)} entities"
                )


def assert_split_invariants(frame: pl.DataFrame, kind: str) -> None:
    """Fail on pair overlap, invalid assignments, or cold-entity contamination."""

    if kind not in SPLIT_KINDS:
        raise SplitInvariantError(f"Unknown split kind: {kind}")
    if frame["canonical_pair_key"].n_unique() != frame.height:
        raise SplitInvariantError("Canonical pair keys must remain unique")
    observed = set(frame["split_assignment"].drop_nulls().unique().to_list())
    unexpected = observed - set(ASSIGNED_SPLITS)
    if unexpected:
        raise SplitInvariantError(f"Unexpected split assignments: {unexpected}")
    if kind != "double_cold" and frame["split_assignment"].null_count():
        raise SplitInvariantError(f"{kind} cannot contain unassigned pairs")
    if kind == "double_cold":
        invalid_missing = frame.filter(
            pl.col("split_assignment").is_null()
            & (pl.col("split_exclusion_reason") != "CROSS_PARTITION_ENDPOINTS")
        )
        if invalid_missing.height:
            raise SplitInvariantError("Double-cold exclusions must state their reason")
    if kind in {"cold_drug", "double_cold"}:
        _assert_disjoint(_entity_sets(frame, "drug_id"), "drug")
    if kind in {"cold_disease", "double_cold"}:
        _assert_disjoint(_entity_sets(frame, "disease_id"), "disease")


def split_statistics(frame: pl.DataFrame) -> dict[str, Any]:
    """Return auditable split sizes and train/test entity intersections."""

    kind = str(frame["split_kind"][0])
    assert_split_invariants(frame, kind)
    counts = {
        split: frame.filter(pl.col("split_assignment") == split).height
        for split in ASSIGNED_SPLITS
    }
    drugs = _entity_sets(frame, "drug_id")
    diseases = _entity_sets(frame, "disease_id")
    return {
        "kind": kind,
        "seed": int(frame["split_seed"][0]),
        "validation_ratio": float(frame["validation_ratio"][0]),
        "test_ratio": float(frame["test_ratio"][0]),
        "counts": counts,
        "excluded_pair_count": frame["split_assignment"].null_count(),
        "pair_overlap_count": 0,
        "train_test_drug_overlap_count": len(drugs["train"] & drugs["test"]),
        "train_test_disease_overlap_count": len(
            diseases["train"] & diseases["test"]
        ),
    }
