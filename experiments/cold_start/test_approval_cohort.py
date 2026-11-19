from collections import Counter
from pathlib import Path
import tempfile
import unittest

import polars as pl


# Assert the data contract independently of the production constant.
SOURCE = Path("data/cache/optimuskg/doi_10_7910_DVN_IYNGEV/2.0/nodes/drug.parquet")


def write_years(root: Path, records: list[tuple[str, int | None]]) -> None:
    source = root / SOURCE
    source.parent.mkdir(parents=True)
    pl.DataFrame(
        {"id": [key for key, _ in records],
         "year_of_first_approval": [year for _, year in records]},
        schema={"id": pl.String, "year_of_first_approval": pl.Int64},
    ).select("id", pl.struct("year_of_first_approval").alias("properties")).write_parquet(source)


class ApprovalCohortTests(unittest.TestCase):
    def test_whole_years_and_missing_dates_do_not_enter_cohorts(self) -> None:
        try:
            from approval_cohort import approval_partitions
        except ModuleNotFoundError as exc:
            if exc.name != "approval_cohort":
                raise
            self.fail("approval_partitions is not implemented")
        dated = [(f"d{i:02}", year) for i, year in enumerate(
            [2000] * 8 + [2010] * 7 + [2015] * 2 + [2020] + [2021] * 2
        )]
        eligible = {key for key, _ in dated} | {"null-date", "no-record", "outside-graph"}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            write_years(root, dated + [("null-date", None), ("outside-graph", 2015),
                                       ("non-eligible", 2022)])
            graph = (eligible - {"outside-graph"}) | {"non-eligible", "graph-unknown"}
            assignments, audit = approval_partitions(root, eligible, graph)
        self.assertEqual(set(assignments), eligible | graph)
        self.assertEqual(dict(Counter(assignments[drug] for drug in eligible)),
                         {"train": 15, "validation": 3, "test": 2, "excluded": 3})
        self.assertEqual(assignments["non-eligible"], "test")
        self.assertEqual(assignments["graph-unknown"], "excluded")
        self.assertEqual(audit["graph_assignment_counts"],
                         {"train": 15, "validation": 3, "test": 3, "excluded": 3})
        self.assertEqual({assignments[key] for key, year in dated if year == 2015},
                         {"validation"})
        self.assertEqual(audit["train_through_year"], 2010)
        self.assertEqual(audit["validation_through_year"], 2020)
        self.assertFalse(audit["historical_graph_availability_verified"])

    def test_too_few_years_fail_closed_instead_of_splitting_a_year(self) -> None:
        try:
            from approval_cohort import approval_partitions
        except ModuleNotFoundError as exc:
            if exc.name != "approval_cohort":
                raise
            self.fail("approval_partitions is not implemented")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            write_years(root, [("a", 2010), ("b", 2010), ("c", 2020)])
            with self.assertRaisesRegex(ValueError, "three distinct"):
                approval_partitions(root, {"a", "b", "c"}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
