import json
from pathlib import Path
import polars as pl
import pytest

from evi_hgt.data.common import sha256
from evi_hgt.settings import Settings


def test_approval_years_are_read_from_documented_aggregate_nodes(tmp_path):
    from experiments.workflow.cold import approval_partitions_from_aggregate

    source = tmp_path / "nodes.parquet"
    source_rows = [
        ("old", "DRG", '{"year_of_first_approval":1980}'),
        ("middle", "DRG", '{"year_of_first_approval":2010}'),
        ("latest", "DRG", '{"year_of_first_approval":2023}'),
        ("missing", "DRG", "{}"),
        ("disease", "DIS", '{"year_of_first_approval":2024}'),
    ]
    pl.DataFrame(source_rows, schema=["id", "label", "properties"], orient="row").write_parquet(source)
    ids = {"old", "middle", "latest", "missing", "drugbank"}
    assignment, receipt = approval_partitions_from_aggregate(source, ids, ids, tmp_path / "derived")
    assert assignment == {"old": "train", "middle": "validation", "latest": "test",
                          "missing": "excluded", "drugbank": "excluded"}
    assert receipt["source_sha256"] == sha256(source)
    assert receipt["historical_graph_availability_verified"] is False
    assert receipt["exclusion_counts"]["no_optimuskg_record"] == 1


def test_cold_preparation_and_scoring_keep_sealed_test_out_of_training(tmp_path):
    from experiments.workflow.cold import run_cold
    from experiments.workflow.cold_smoke import prepare_cold_smoke

    prepared = prepare_cold_smoke("cold_disease", tmp_path / "cold_disease", audit_dir=tmp_path / "audit")
    document = json.loads(Path(prepared["manifest"]).read_text(encoding="utf-8"))
    assert document["schedule"]["G"] == 1
    assert document["purpose"] == "SYNTHETIC_TEST_ONLY"
    assert not any(name.startswith("test_") for name in document["files"])
    settings = Settings(output_dir=str(tmp_path / "runs"), device="cpu", precision="fp32", checkpoint_every=1)
    run_dir = run_cold(settings, prepared, smoke=True)
    metrics = json.loads((run_dir / "test/metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "COMPLETE"
    assert metrics["updates"] == 3
    assert metrics["graph_coverage"]["disease_endpoint_with_fact"] == 0
    assert metrics["membership"]["test_training_disjoint"]
    assert metrics["membership"]["shared_u"]
    assert run_cold(settings, prepared, smoke=True) == run_dir
    (run_dir / "result.json").rename(run_dir / "previous_result.json")
    (run_dir / "training_complete.json").rename(run_dir / "previous_training.json")
    assert run_cold(settings, prepared, smoke=True) == run_dir
    assert len(list((run_dir / "frozen_runs").glob("*/checkpoints/latest.pt"))) == 1

    with Path(prepared["test_panels"]["support"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="test panel|test candidates"):
        run_cold(settings, prepared, smoke=True)



@pytest.mark.parametrize("case", ("cold_drug", "cold_disease", "double_cold", "first_approval_year_cohort"))
def test_prepare_every_cold_case_from_raw_layout_fixture(case, tmp_path):
    from experiments.workflow.cold_smoke import prepare_cold_smoke

    prepared = prepare_cold_smoke(case, tmp_path / case, audit_dir=tmp_path / "audit")
    manifest = json.loads(prepared["manifest"].read_text(encoding="utf-8"))
    audit = json.loads(prepared["preparation_audit"].read_text(encoding="utf-8"))
    assert manifest["schedule"]["G"] == 1
    assert audit["graph"]["heldout_facts_remaining"] == 0
    assert audit["graph"]["heldout_csr_neighbors_remaining"] == 0
    assert audit["retained_P"]["test"] == {"support": 1, "contra": 1}
    assert audit["test_U"] >= 1
    if case == "first_approval_year_cohort":
        assert audit["split"]["train_through_year"] == 1980
        assert audit["split"]["validation_through_year"] == 2005
        assert audit["graph"]["temporal_as_of_graph"] is False



def test_cold_native_hgtdr_selects_before_scoring_the_common_test(tmp_path):
    pytest.importorskip("torch_sparse")
    from experiments.workflow.cold import run_cold
    from experiments.workflow.cold_hgtdr import run_cold_hgtdr
    from experiments.workflow.cold_smoke import prepare_cold_smoke

    prepared = prepare_cold_smoke("cold_disease", tmp_path / "cold_disease", audit_dir=tmp_path / "audit")
    settings = Settings(output_dir=str(tmp_path / "runs"), device="cpu", precision="fp32", checkpoint_every=1)
    reference = run_cold(settings, prepared, smoke=True)
    run = run_cold_hgtdr(settings, prepared, reference, {}, smoke=True)
    report = json.loads((run / "result.json").read_text(encoding="utf-8"))
    assert report["purpose"] == "SYNTHETIC_TEST_ONLY"
    assert report["status"] == "COMPLETE"
    assert set(report["heads"]) == {"support", "contra"}
    for head in report["heads"]:
        assert report["heads"][head]["training"]["selected_epoch"] == 1
        assert report["heads"][head]["evaluable_diseases"] == 1
        assert (run / head / "trained.json").is_file()
        assert (run / "test" / f"{head}.parquet").is_file()
    assert run_cold_hgtdr(settings, prepared, reference, {}, smoke=True) == run
