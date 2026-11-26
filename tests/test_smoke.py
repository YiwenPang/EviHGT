"""The public smoke entry point runs one bounded, offline FULL workflow."""
import importlib
import importlib.util
import json
from pathlib import Path
import socket

import polars as pl
import pytest
import torch


def smoke_module():
    assert importlib.util.find_spec("evi_hgt.smoke") is not None, "The packaged smoke entry point is missing"
    return importlib.import_module("evi_hgt.smoke")


def test_smoke_runs_three_real_updates_offline_with_isolated_inputs(tmp_path, monkeypatch, capsys):
    smoke = smoke_module()
    from evi_hgt import settings
    from evi_hgt.preprocessing import pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic smoke must not read default settings, prepare raw data, or access the network")

    monkeypatch.setattr(settings, "load_settings", forbidden)
    monkeypatch.setattr(pipeline, "prepare_raw", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    data_root, output_root = tmp_path / "derived", tmp_path / "runtime"
    data_root.mkdir()
    existing = data_root / "existing-input.txt"
    existing.write_bytes(b"DO NOT MODIFY")
    old_threads = torch.get_num_threads()

    result = smoke.run_smoke(data_root=data_root, output_root=output_root)

    assert torch.get_num_threads() == old_threads
    assert result.is_dir() and result.is_relative_to(output_root)
    completion = json.loads((result / "completion.json").read_text("utf8"))
    assert completion["status"] == "COMPLETE"
    assert completion["scope"] == completion["purpose"] == "SYNTHETIC_TEST_ONLY"
    assert completion["variant"] == "FULL"
    assert completion["updates"] == completion["total_updates"] == completion["updates_this_invocation"] == 3
    config = json.loads((result / "configuration.json").read_text("utf8"))
    assert (config["device"], config["precision"], config["checkpoint_every"]) == ("cpu", "fp32", 1)
    manifest_path = Path(config["manifest"])
    assert manifest_path.is_relative_to(data_root)
    manifest = json.loads(manifest_path.read_text("utf8"))
    assert manifest["purpose"] == "SYNTHETIC_TEST_ONLY" and manifest["schedule"]["G"] == 1
    assert pl.read_parquet(manifest_path.parent / manifest["nodes"]).height == 15
    assert sum(entry["rows"] for entry in manifest["fact_files"]) == 10
    assert existing.read_bytes() == b"DO NOT MODIFY"
    assert (result / "checkpoints/latest.pt").is_file()
    metrics = json.loads((result / "evaluation/step_3/metrics.json").read_text("utf8"))["metrics"]
    assert metrics["support"]["macro_auc"] is not None
    assert metrics["contra"]["macro_auc"] is not None
    events = [json.loads(line) for line in (result / "logs/events.jsonl").read_text("utf8").splitlines()]
    assert [row["step"] for row in events if row["event"] == "optimizer_update"] == [1, 2, 3]
    assert "SYNTHETIC_TEST_ONLY" in capsys.readouterr().out


def test_smoke_default_paths_are_unique_and_threads_restore_after_failure(tmp_path, monkeypatch):
    smoke = smoke_module()
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    captured = []

    def fail(settings, *, variant):
        assert torch.get_num_threads() == 1
        assert variant == "FULL"
        captured.append(settings)
        raise RuntimeError("Injected smoke interruption")

    monkeypatch.setattr(smoke, "run", fail)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Injected smoke interruption"):
                smoke.run_smoke()
            assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(old_threads)
    manifests = [Path(item.manifest) for item in captured]
    outputs = [Path(item.output_dir) for item in captured]
    assert len(set(manifests)) == len(set(outputs)) == 2
    for manifest, output in zip(manifests, outputs):
        assert manifest.parent.parent == tmp_path / "data/processed/smoke"
        assert output.parent == tmp_path / "outputs/smoke"
        assert manifest.parent.name == output.name
        assert len(output.name) == 32


def test_smoke_rejects_a_fixture_that_changes_the_training_budget(tmp_path, monkeypatch):
    smoke = smoke_module()
    original = smoke.make_corpus

    def oversized(directory):
        path = original(directory)
        value = json.loads(path.read_text("utf8"))
        value["schedule"]["G"] = 2
        path.write_text(json.dumps(value), encoding="utf8")
        return path

    monkeypatch.setattr(smoke, "make_corpus", oversized)
    monkeypatch.setattr(smoke, "run", lambda *args, **kwargs: pytest.fail("Invalid smoke budget reached training"))
    with pytest.raises(ValueError, match="SYNTHETIC_TEST_ONLY|三次|three"):
        smoke.run_smoke(data_root=tmp_path / "derived", output_root=tmp_path / "runtime")
