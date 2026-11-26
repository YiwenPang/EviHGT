"""The public primary entry point has an isolated tiny check and early guards."""
import sys

import pytest


def test_main_smoke_does_not_read_training_or_raw_configuration(monkeypatch, tmp_path):
    from evi_hgt import cli
    calls = []
    monkeypatch.setattr(cli, "run_smoke", lambda: calls.append("smoke") or tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("smoke read real training settings"))
    monkeypatch.setattr(cli, "load_report_settings", lambda *_: pytest.fail("smoke read custom output settings"))
    monkeypatch.setattr(cli, "prepare_raw", lambda *a, **kw: pytest.fail("smoke accessed raw preparation"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--smoke"])
    assert cli.main() == 0
    assert calls == ["smoke"]


@pytest.mark.parametrize("arguments", [["--config", "real.yaml"], ["--checkpoint", "real.pt"],
    ["--raw-config", "real.yaml"], ["--report-config", "real.yaml"], ["--mode", "evaluate"], ["--offline"]])
def test_smoke_refuses_options_that_could_select_a_real_run(monkeypatch, arguments):
    from evi_hgt import cli
    monkeypatch.setattr(sys, "argv", ["main.py", "--smoke", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_evaluation_without_checkpoint_fails_before_loading_inputs(monkeypatch):
    from evi_hgt import cli
    monkeypatch.setattr(cli, "load_settings", lambda *_: pytest.fail("loaded settings before checkpoint guard"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "evaluate"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_smoke_failure_returns_nonzero(monkeypatch):
    from evi_hgt import cli
    def fail():
        raise ValueError("SYNTHETIC_TEST_ONLY failure")
    errors = []
    monkeypatch.setattr(cli, "run_smoke", fail)
    monkeypatch.setattr(cli, "startup_failure", errors.append)
    monkeypatch.setattr(sys, "argv", ["main.py", "--smoke"])
    assert cli.main() == 1
    assert len(errors) == 1
