"""Optional drug approval-year cohort adapter; it does not date graph facts."""

from collections import Counter
import hashlib
from itertools import accumulate
from pathlib import Path

import polars as pl


APPROVAL_SOURCE = Path("data/cache/optimuskg/doi_10_7910_DVN_IYNGEV/2.0/nodes/drug.parquet")


def approval_partitions(
    workspace: Path, eligible_drugs: set[str], nodes_drugs: set[str],
) -> tuple[dict[str, str], dict]:
    """Assign all graph drugs and eligible IDs to chronological cohorts.

    Choose the train boundary nearest 80% of dated drugs (allowing a year for
    each later partition), then the validation boundary nearest cumulative 90%.
    Ties choose the earlier year. No labels or model scores enter the choice.
    """
    in_graph = eligible_drugs & nodes_drugs
    source = workspace / APPROVAL_SOURCE
    with source.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    records = (
        pl.scan_parquet(source)
        .filter(pl.col("id").is_in(sorted(nodes_drugs)))
        .select("id", pl.col("properties").struct.field("year_of_first_approval").alias("year"))
        .collect()
    )
    if records.schema != {"id": pl.String, "year": pl.Int64}:
        raise ValueError("Unexpected OptimusKG approval-year column schema")
    if records["id"].n_unique() != records.height:
        raise ValueError("OptimusKG approval-year records contain duplicate graph drug IDs")
    years: dict[str, int] = {}
    missing = Counter()
    found_eligible = 0
    for drug, year in records.iter_rows():
        found_eligible += drug in in_graph
        if year is None:
            missing["null_approval_year"] += drug in in_graph
            continue
        if year < 1 or year > 9999:
            raise ValueError(f"Invalid OptimusKG approval year for graph drug {drug}")
        years[drug] = year
    missing["no_optimuskg_record"] = len(in_graph) - found_eligible
    missing["outside_graph"] = len(eligible_drugs - in_graph)
    by_year = Counter(year for drug, year in years.items() if drug in in_graph)
    distinct_years = sorted(by_year)
    if len(distinct_years) < 3:
        raise ValueError("Approval cohort needs dated eligible graph drugs in at least three distinct years")

    n = sum(by_year.values())
    cumulative = list(accumulate(by_year[year] for year in distinct_years))
    i = min(range(len(distinct_years) - 2), key=lambda j: (abs(10 * cumulative[j] - 8 * n), j))
    k = min(range(i + 1, len(distinct_years) - 1), key=lambda j: (abs(10 * cumulative[j] - 9 * n), j))
    train_year, validation_year = distinct_years[i], distinct_years[k]
    partitions = {}
    for drug in sorted(eligible_drugs | nodes_drugs):
        year = years.get(drug)
        if year is None:
            partitions[drug] = "excluded"
        elif year <= train_year:
            partitions[drug] = "train"
        elif year <= validation_year:
            partitions[drug] = "validation"
        else:
            partitions[drug] = "test"
    audit = {
        "protocol": "APPROVAL_YEAR_DRUG_COHORT_V1",
        "source": APPROVAL_SOURCE.as_posix(),
        "source_sha256": source_sha256,
        "source_field": "properties.year_of_first_approval",
        "assignment_rule": "Nearest 80% and 90% whole-year cumulative dated drug counts with nonempty chronological partitions; earlier year breaks ties.",
        "date_meaning": "First drug approval year as recorded in OptimusKG; not development start or first known DDA date.",
        "dated_eligible_drugs": n,
        "approval_year_min": distinct_years[0],
        "approval_year_max": distinct_years[-1],
        "train_through_year": train_year,
        "validation_through_year": validation_year,
        "assignment_counts": dict(Counter(partitions[drug] for drug in eligible_drugs)),
        "graph_assignment_counts": dict(Counter(partitions[drug] for drug in nodes_drugs)),
        "exclusion_counts": dict(missing),
        "score_blind": True,
        "historical_graph_availability_verified": False,
    }
    return partitions, audit
