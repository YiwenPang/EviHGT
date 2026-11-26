"""Ablation objectives and runner tested on the offline synthetic corpus."""
import importlib
import importlib.util
import json
from pathlib import Path
import random

import pytest
import torch

from evi_hgt.settings import Settings
from evi_hgt.synthetic import make_corpus


def module(name):
    package = importlib.util.find_spec("experiments.workflow")
    assert package is not None, "The public experiment workflow is missing"
    path = f"experiments.workflow.{name}"
    assert importlib.util.find_spec(path) is not None, f"The public {name} module is missing"
    return importlib.import_module(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def tiny_settings(tmp_path):
    return Settings(manifest=str(make_corpus(tmp_path / "corpus")),
                    output_dir=str(tmp_path / "runs"), device="cpu", precision="fp32",
                    checkpoint_every=1)


def test_sigmoid_control_preserves_the_gradient_at_saturated_margins():
    sigmoid_risk = module("ablation_losses").sigmoid_risk
    positive = torch.tensor([-20.0, 20.0], requires_grad=True)
    unlabeled = torch.tensor([0.0], requires_grad=True)

    loss = sigmoid_risk(positive, unlabeled)
    loss.backward()

    q = torch.exp(torch.tensor(-20.0))
    magnitude = q / (1 + q).square()
    torch.testing.assert_close(loss, torch.sigmoid(-positive.detach()).mean(), atol=0, rtol=0)
    torch.testing.assert_close(positive.grad, -magnitude.expand(2) / 2, atol=0, rtol=0)
    torch.testing.assert_close(unlabeled.grad, magnitude.reshape(1), atol=0, rtol=0)
    assert loss.detach().item() == 0.5


def test_paired_runs_bind_inputs_and_score_both_development_milestones(tiny_settings):
    ablations = module("ablations")
    roots = {loss: ablations.run_ablation(tiny_settings, loss=loss, smoke=True)
             for loss in ("softplus", "sigmoid")}
    witnesses = {}
    candidates = {}
    exposures = {}
    for loss, root in roots.items():
        completion = read(root / "completion.json")
        assert completion["status"] == "COMPLETE"
        assert completion["scope"] == "SYNTHETIC_TEST_ONLY"
        assert completion["variant"] == "FULL" and completion["loss"] == loss
        assert completion["updates"] == completion["total_updates"] == 3
        assert completion["evaluation_steps"] == [1, 3]
        assert (root / "report.html").is_file()
        assert not (root / "evaluation/step_2").exists()
        metrics = read(root / "evaluation/step_3/metrics.json")["metrics"]
        candidates[loss] = [metrics[head]["candidate_identity_sha256"] for head in ("support", "contra")]
        events = [json.loads(line) for line in (root / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()]
        updates = [row for row in events if row["event"] == "optimizer_update"]
        assert [row["step"] for row in updates] == [1, 2, 3]
        assert all(row["loss_id"] == loss for row in updates)
        witnesses[loss] = [(row["training_member_sha256"], row["record_sha256"], row["sampling_sha256"])
                           for row in updates]
        checkpoint = torch.load(root / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        assert checkpoint["binding"]["loss"] == loss
        assert bool(checkpoint["optimizer"]["state"])
        exposure = read(root / completion["diagnostics"])
        assert exposure["model_forwards"] == exposure["optimizer_updates"] == 0
        assert exposure["counts"]["groups"] == 3
        assert exposure["counts"]["query_occurrences"] == 11
        assert exposure["counts"]["available_facts"] <= exposure["counts"]["facts"]
        assert exposure["counts"]["available_path_steps"] <= exposure["counts"]["path_steps"]
        assert exposure["fractions"]["no_path_query_fraction"] is not None
        assert (root / "diagnostics/exposure.csv").is_file()
        exposures[loss] = (exposure["selection"], exposure["counts"], exposure["fractions"])
    assert witnesses["sigmoid"] == witnesses["softplus"]
    assert candidates["sigmoid"] == candidates["softplus"]
    assert exposures["sigmoid"] == exposures["softplus"]
    assert ablations.evaluation_steps(6847) == (10270, 20541)


def test_checkpoint_is_loss_bound_and_resumes_same_variant(tiny_settings):
    run = module("ablations").run_ablation
    first = run(tiny_settings, variant="CONTENT_ZERO", smoke=True)
    checkpoint = first / "checkpoints/step_1.pt"
    with pytest.raises(ValueError, match="checkpoint.*loss|Checkpoint.*loss"):
        run(tiny_settings, variant="CONTENT_ZERO", loss="sigmoid", checkpoint=checkpoint, smoke=True)

    resumed = run(tiny_settings, variant="CONTENT_ZERO", checkpoint=checkpoint, smoke=True)

    completion = read(resumed / "completion.json")
    assert completion["updates"] == 3 and completion["updates_this_invocation"] == 2
    assert completion["evaluation_steps"] == [1, 3]
    expected = torch.load(first / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    actual = torch.load(resumed / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert expected["binding"] == actual["binding"]
    for name, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][name], atol=0, rtol=0)


def test_smoke_is_explicit_and_report_failure_cannot_complete(tiny_settings, monkeypatch):
    ablations = module("ablations")
    with pytest.raises(ValueError, match="synthetic"):
        ablations.run_ablation(tiny_settings)

    def interrupted(*args, **kwargs):
        raise RuntimeError("Interrupted output export")

    monkeypatch.setattr(ablations, "create_report", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted output export"):
        ablations.run_ablation(tiny_settings, variant="FACT_QUALITY_CONSTANT", smoke=True)

    output = Path(tiny_settings.output_dir)
    assert not list(output.glob("*/completion.json"))
    errors, = output.glob("*/logs/errors.log")
    assert "Interrupted output export" in errors.read_text(encoding="utf-8")


def test_fixed_training_panel_keeps_undefined_path_fraction_empty(tiny_settings, monkeypatch):
    from dataclasses import replace

    from evi_hgt.data.corpus import load_corpus
    from evi_hgt.sampling.path_schema import pack_path_candidates

    diagnostics = module("ablation_diagnostics")
    corpus = load_corpus(Path(tiny_settings.manifest))
    assemble = corpus.assembler.assemble

    def pathless(query, **kwargs):
        graph = assemble(query, **kwargs)
        paths = pack_path_candidates((), num_queries=1, max_length=4)
        return replace(graph, batch=replace(graph.batch, path_batch=paths))

    monkeypatch.setattr(corpus.assembler, "assemble", pathless)
    torch_state, python_state = torch.get_rng_state(), random.getstate()
    exposure = diagnostics.fixed_panel_exposure(corpus)

    assert exposure["counts"]["path_steps"] == exposure["counts"]["available_path_steps"] == 0
    assert exposure["counts"]["queries_without_paths"] == exposure["counts"]["query_occurrences"] == 11
    assert exposure["fractions"]["no_path_query_fraction"] == 1.0
    assert exposure["fractions"]["path_step_availability_fraction"] is None
    assert torch.equal(torch.get_rng_state(), torch_state) and random.getstate() == python_state
