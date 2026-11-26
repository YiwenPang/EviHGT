import json

import pytest

from evi_hgt.settings import Settings
from data_fixture import make_corpus
from experiments.workflow.comparisons import run_comparison


@pytest.mark.parametrize("method", ["TRAIN_VISIBLE_POPULARITY", "PAIR_MLP_D1_MATCHED", "HGT_D1_MATCHED"])
def test_portable_comparison_scores_both_frozen_panels(tmp_path, method):
    settings = Settings(manifest=str(make_corpus(tmp_path / "corpus")), output_dir=str(tmp_path / "run"),
                        device="cpu", precision="fp32")
    result = run_comparison(method, settings, smoke=True)
    completed = json.loads((result / "completion.json").read_text())
    metrics = json.loads((result / "metrics.json").read_text())
    assert completed["scope"] == "SYNTHETIC_TEST_ONLY"
    assert completed["updates"] == (0 if method == "TRAIN_VISIBLE_POPULARITY" else 3)
    assert metrics["support"]["pairs"] == metrics["contra"]["pairs"] == 2
    assert metrics["support"]["evaluable_diseases"] == metrics["contra"]["evaluable_diseases"] == 1
    if method == "TRAIN_VISIBLE_POPULARITY":
        assert metrics["support"]["macro_ap"] == metrics["contra"]["macro_ap"] == 1.0


def test_synthetic_corpus_requires_explicit_smoke_mode(tmp_path):
    settings = Settings(manifest=str(make_corpus(tmp_path / "corpus")), output_dir=str(tmp_path / "run"),
                        device="cpu", precision="fp32")
    with pytest.raises(ValueError, match="synthetic"):
        run_comparison("TRAIN_VISIBLE_POPULARITY", settings)

@pytest.mark.parametrize("method, expected", [("HGTDR_ADAPTED", {"support"}),
                                             ("HGTDR_ADAPTED_CONTRA", {"contra"}),
                                             ("TXGNN_ADAPTED", {"support", "contra"})])
def test_native_comparison_runs_real_tiny_update(tmp_path, method, expected):
    import os
    from experiments.workflow.comparisons import check_native_requirements
    from experiments.workflow.comparison_smoke import create_comparison_smoke_corpus

    context = {"offline": True}
    if method == "TXGNN_ADAPTED":
        if not os.environ.get("EVIHGT_TEST_TXGNN_SOURCE"):
            pytest.skip("Set EVIHGT_TEST_TXGNN_SOURCE to a pinned local TxGNN checkout")
        context["txgnn_source"] = os.environ["EVIHGT_TEST_TXGNN_SOURCE"]
    else:
        pytest.importorskip("torch_sparse")
    check_native_requirements(context, [method], smoke=True)
    manifest = create_comparison_smoke_corpus(tmp_path / "corpus")
    settings = Settings(manifest=str(manifest), output_dir=str(tmp_path / "run"), device="cpu", precision="fp32")
    result = run_comparison(method, settings, smoke=True, context=context)
    completed = json.loads((result / "completion.json").read_text(encoding="utf-8"))
    metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    assert completed["scope"] == "SYNTHETIC_TEST_ONLY"
    assert completed["updates"] == (2 if method == "TXGNN_ADAPTED" else 1)
    assert completed["selected_epoch"] == 1
    assert set(metrics) == expected
    assert all(row["pairs"] == 2 and row["evaluable_diseases"] == 1 for row in metrics.values())
    assert (result / "selected.pt").is_file()
    restored = run_comparison(method, settings, smoke=True, context=context, checkpoint=result / "checkpoint.pt")
    assert json.loads((restored / "metrics.json").read_text(encoding="utf-8")) == metrics
    events = (restored / "logs/events.jsonl").read_text(encoding="utf-8")
    assert '"event": "optimizer_update"' not in events
