"""Train, evaluate and resume on a synthetic corpus."""
import json
from pathlib import Path

import numpy as np
import torch

from data_fixture import make_corpus
from evi_hgt.runner import run
from evi_hgt.settings import Settings


def test_complete_synthetic_workflow_and_checkpoint_evaluation(tmp_path):
    torch.set_num_threads(1)
    manifest = make_corpus(tmp_path / "corpus")
    settings = Settings(manifest=str(manifest), output_dir=str(tmp_path / "outputs"),
                        device="cpu", precision="fp32", checkpoint_every=1)
    result = run(settings)
    completion = json.loads((result / "completion.json").read_text())
    assert completion["updates"] == completion["total_updates"] == 3
    assert completion["scope"] == "SYNTHETIC_TEST_ONLY"
    assert completion["updates_this_invocation"] == 3
    checkpoint = result / "checkpoints/latest.pt"
    assert checkpoint.is_file()
    metrics = json.loads((result / "evaluation/step_3/metrics.json").read_text())["metrics"]
    assert metrics["support"]["macro_auc"] is not None
    assert metrics["contra"]["macro_auc"] is not None
    evaluated = run(settings, mode="evaluate", checkpoint=checkpoint)
    again = json.loads((evaluated / "evaluation/step_3/metrics.json").read_text())["metrics"]
    assert again == metrics
    resumed = run(settings, checkpoint=checkpoint)
    assert json.loads((resumed / "completion.json").read_text())["updates"] == 3
    assert json.loads((resumed / "completion.json").read_text())["updates_this_invocation"] == 0
    assert (resumed / "evaluation/step_3/metrics.json").is_file()


def test_variants_execute_one_real_synthetic_update(tmp_path):
    from evi_hgt.data.corpus import load_corpus
    from evi_hgt.model import VARIANTS, build_model
    from evi_hgt.runner import update_once
    torch.set_num_threads(1)
    corpus = load_corpus(make_corpus(tmp_path / "corpus"))
    settings = Settings(device="cpu", precision="fp32")
    for variant in VARIANTS:
        built = build_model(corpus.model_config, pretrained=corpus.pretrained,
                            seed=corpus.training_seed, variant=variant)
        optimizer = torch.optim.AdamW(built.model.parameters(), lr=5e-5, weight_decay=1e-5)
        row = update_once(built, corpus, corpus.plan.step(2), optimizer, settings)
        assert np.isfinite(row["loss"])
        assert row["queries"] >= 3


def test_interrupted_training_resumes_exact_cpu_trajectory(tmp_path, monkeypatch):
    import pytest
    import evi_hgt.runner as runner
    torch.set_num_threads(1)
    manifest = make_corpus(tmp_path / "corpus")
    settings = Settings(manifest=str(manifest), output_dir=str(tmp_path / "outputs"),
                        device="cpu", precision="fp32", checkpoint_every=1)
    complete = run(settings)
    expected = torch.load(complete / "checkpoints/latest.pt", weights_only=False)
    real_update = runner.update_once
    def interrupt_after_first(built, corpus, spec, optimizer, settings):
        if spec.step == 2:
            raise RuntimeError("SYNTHETIC_TEST_ONLY injected interruption")
        return real_update(built, corpus, spec, optimizer, settings)
    monkeypatch.setattr(runner, "update_once", interrupt_after_first)
    with pytest.raises(RuntimeError, match="injected interruption"):
        run(settings)
    roots = sorted(Path(settings.output_dir).iterdir())
    partial = roots[-1] / "checkpoints/latest.pt"
    assert torch.load(partial, weights_only=False)["step"] == 1
    monkeypatch.setattr(runner, "update_once", real_update)
    recovered = run(settings, checkpoint=partial)
    actual = torch.load(recovered / "checkpoints/latest.pt", weights_only=False)
    assert actual["step"] == expected["step"] == 3
    for name, value in expected["model"].items():
        assert torch.equal(value, actual["model"][name]), name
    for key, state in expected["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, actual["optimizer"]["state"][key][name])
