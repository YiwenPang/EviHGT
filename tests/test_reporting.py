"""Complete-run outputs keep the two evaluation candidate pools and model state."""
import json
import random
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
import torch

from data_fixture import make_corpus
from evi_hgt.metrics import metric_pair


def saved_evaluation(root):
    root.mkdir()
    support = [
        dict(drug_id="B", disease_id="S0", role="U", raw_relation=None,
             support_logit=1., unsuitability_logit=1.),
        dict(drug_id="A", disease_id="S0", role="P", raw_relation="INDICATION",
             support_logit=1., unsuitability_logit=10.),
        dict(drug_id="A", disease_id="S1", role="U", raw_relation=None,
             support_logit=-1000., unsuitability_logit=1000.),
    ]
    auxiliary = [dict(drug_id="C", disease_id="S0", role="P", raw_relation="CONTRAINDICATION",
                      support_logit=-1., unsuitability_logit=0.)]
    pl.DataFrame(support).write_parquet(root / "validation.parquet")
    pl.DataFrame(auxiliary).write_parquet(root / "auxiliary_validation.parquet")
    (root / "metrics.json").write_text(json.dumps(dict(step=3, corpus_identity="fixture",
        metrics=metric_pair(support, auxiliary))), encoding="utf-8")
    return support, auxiliary


def test_rankings_keep_each_head_pool_and_the_metric_tie_rule(tmp_path):
    from evi_hgt.reporting import export_rankings
    saved_evaluation(tmp_path / "evaluation")
    result = export_rankings(tmp_path / "evaluation", tmp_path / "rankings", top_k=1)
    support = result["rows"]["support"]
    contra = result["rows"]["contra"]
    assert [(r["drug_id"], r["disease_id"], r["rank"]) for r in support] == [
        ("A", "S0", 1), ("B", "S0", 2), ("A", "S1", 1)]
    assert {(r["drug_id"], r["disease_id"]) for r in contra} == {("B", "S0"), ("C", "S0"), ("A", "S1")}
    assert next(r for r in contra if r["drug_id"] == "B")["role"] == "U"
    for row in support + contra:
        assert sum(row[name] for name in ("m_T", "m_F", "m_B", "m_U")) == pytest.approx(1.)
        assert 0 <= row["support_score"] <= 1 and 0 <= row["contra_score"] <= 1
    assert pl.read_csv(tmp_path / "rankings/support.csv").height == 3
    assert pl.read_csv(tmp_path / "rankings/support_top1.csv").height == 2
    assert result["metadata"]["support"]["candidate_identity_sha256"] == \
        json.loads((tmp_path / "evaluation/metrics.json").read_text())["metrics"]["support"]["candidate_identity_sha256"]


def test_rankings_reject_changed_predictions_before_publishing(tmp_path):
    from evi_hgt.reporting import export_rankings
    saved_evaluation(tmp_path / "evaluation")
    path = tmp_path / "evaluation/validation.parquet"
    pl.read_parquet(path).filter(pl.col("drug_id") != "B").write_parquet(path)
    with pytest.raises(ValueError, match="evaluation|metrics|candidate"):
        export_rankings(path.parent, tmp_path / "rankings", top_k=10)
    assert not (tmp_path / "rankings/support.csv").exists()


@pytest.mark.parametrize("values", [{"explanation_queries": -1}, {"explanation_queries": True},
    {"paths_per_query": 0}, {"ranking_top_k": 0}, {"paths_per_query": 1.5}])
def test_report_settings_reject_invalid_budgets(values):
    from evi_hgt.reporting import ReportSettings
    with pytest.raises(ValueError):
        ReportSettings(**values)


def test_diagnostics_restore_model_flags_rng_and_parameters(tmp_path):
    from evi_hgt.reporting import diagnostic_inference
    from evi_hgt.data.corpus import load_corpus
    from evi_hgt.model import build_model
    torch.set_num_threads(1)
    corpus = load_corpus(make_corpus(tmp_path / "corpus"))
    built = build_model(corpus.model_config, pretrained=corpus.pretrained, seed=42)
    built.model.train()
    config = built.model.encoder.config
    modes = [m.training for m in built.model.modules()]
    weights = {k: v.clone() for k, v in built.model.state_dict().items()}
    tensor_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    with pytest.raises(RuntimeError, match="injected"):
        with diagnostic_inference(built):
            assert built.model.encoder.config.return_logic_state
            assert not any(m.training for m in built.model.modules())
            random.random(); np.random.rand(); torch.rand(1)
            raise RuntimeError("injected")
    assert built.model.encoder.config is config
    assert [m.training for m in built.model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), tensor_rng)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert all(torch.equal(weights[k], v) for k, v in built.model.state_dict().items())


def test_full_runner_reports_are_bound_to_final_checkpoint(tmp_path):
    from evi_hgt.runner import run
    from evi_hgt.settings import Settings
    torch.set_num_threads(1)
    manifest = make_corpus(tmp_path / "corpus")
    settings = Settings(manifest=str(manifest), output_dir=str(tmp_path / "outputs"),
                        device="cpu", precision="fp32", checkpoint_every=1)
    result = run(settings)
    completion = json.loads((result / "completion.json").read_text())
    assert completion["report"] == "report.html"
    summary = json.loads((result / "summary.json").read_text())
    assert summary["schema"] == "EVIHGT_RUN_SUMMARY_V1"
    assert summary["step"] == 3
    assert summary["scope"] == "SYNTHETIC_TEST_ONLY"
    assert summary["checkpoint"]["sha256"]
    index = json.loads((result / "explanations/index.json").read_text())
    assert index["queries"]
    assert all((result / item["path"]).is_file() for item in index["queries"])
    html = (result / "report.html").read_text("utf-8")
    assert "SYNTHETIC_TEST_ONLY" in html and "rankings/support.csv" in html
    assert (result / "rankings/contra.csv").is_file()
    evaluated = run(settings, mode="evaluate", checkpoint=result / "checkpoints/latest.pt")
    repeated = json.loads((evaluated / "summary.json").read_text())
    assert repeated["checkpoint"]["sha256"] == summary["checkpoint"]["sha256"]
    assert repeated["metrics"] == summary["metrics"]


def test_report_failure_never_marks_workflow_complete(tmp_path, monkeypatch):
    from evi_hgt import runner
    from evi_hgt.settings import Settings
    torch.set_num_threads(1)
    manifest = make_corpus(tmp_path / "corpus")
    settings = Settings(manifest=str(manifest), output_dir=str(tmp_path / "outputs"),
                        device="cpu", precision="fp32", checkpoint_every=1)
    def fail(*args, **kwargs):
        raise RuntimeError("injected report export failure")
    monkeypatch.setattr(runner, "create_report", fail)
    with pytest.raises(RuntimeError, match="report export"):
        runner.run(settings)
    result, = (tmp_path / "outputs").iterdir()
    assert (result / "checkpoints/latest.pt").is_file()
    assert not (result / "completion.json").exists()
