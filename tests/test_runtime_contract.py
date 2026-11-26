"""Check runtime behavior on small synthetic inputs."""
import io
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_half_softplus_and_conditional_weighting():
    from evi_hgt.losses import half_softplus, dual_risk
    p = torch.tensor([2., -1., 0.], requires_grad=True)
    u = torch.tensor([0.5])
    cp = torch.tensor([1.], requires_grad=True)
    cu = torch.tensor([-1.])
    expected = 0.5 * torch.nn.functional.softplus(-(p - u)).mean()
    assert torch.equal(half_softplus(p, u), expected)
    local = {"D1": torch.tensor([-0.5])}
    risk = dual_risk(p, u, cp, cu, support_diseases=["D1", "D1", "D2"], local_u=local)
    expected_support = (half_softplus(p[:2], local["D1"]) * (2/3)
                        + half_softplus(p[2:], u) * (1/3))
    assert torch.equal(risk.support, expected_support)
    assert torch.equal(risk.total, (risk.support + risk.contra) / 2)
    risk.total.backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(cp.grad).all()


def test_ties_and_missing_metrics_are_explicit():
    from evi_hgt.metrics import threshold_metrics, evaluate_panel
    result = threshold_metrics([0.], [0., 0., 0.])
    assert result["auc"] == 0.5
    assert result["ap"] == 0.25
    assert result["aupr_trapezoid"] == 0.625
    panel = evaluate_panel([dict(drug_id="A", disease_id="D", role="P", support_logit=1.)])
    assert panel["macro_auc"] is None
    assert panel["status"] == "NO_EVALUABLE_DISEASES"


def test_console_redirected_has_no_control_codes_and_preserves_small_loss(tmp_path):
    from evi_hgt.console import Console
    stream = io.StringIO()
    with Console(tmp_path, stream=stream, refresh_seconds=5) as console:
        console.banner()
        with console.phase(1, "Training"):
            with console.progress(2, "Epoch 1/3", unit="update") as progress:
                progress.update(loss=1e-10)
                progress.update(loss=2e-10)
    output = stream.getvalue()
    assert "Epoch 1/3" in output and "2e-10" in output
    assert "\x1b" not in output and "\r" not in output
    assert "完成" in output and "阶段" in output


def test_invalid_loss_stream_fails():
    from evi_hgt.losses import half_softplus
    with pytest.raises(ValueError):
        half_softplus(torch.tensor([]), torch.tensor([0.]))
    with pytest.raises(ValueError):
        half_softplus(torch.tensor([math.nan]), torch.tensor([0.]))


def test_missing_contra_panel_is_unavailable_not_failure():
    from evi_hgt.metrics import metric_pair
    metrics = metric_pair([dict(drug_id="A", disease_id="D", role="P", support_logit=1.)], [])
    assert metrics["support"]["macro_auc"] is None
    assert metrics["contra"]["macro_auc"] is None
    assert metrics["contra"]["status"] == "UNAVAILABLE"


def test_cli_reports_early_output_failure(tmp_path, monkeypatch, capsys):
    import evi_hgt.cli as cli
    from evi_hgt.settings import Settings
    target = tmp_path / "not_a_directory"
    target.write_text("SYNTHETIC_TEST_ONLY")
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(cli, "load_settings", lambda path: Settings(output_dir=str(target)))
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert "[Error]" in captured.err


@pytest.mark.parametrize("values", [{"learning_rate": math.nan}, {"weight_decay": math.inf},
    {"checkpoint_every": 1.5}, {"evaluate_every_cycle": "false"}, {"learning_rate": True}])
def test_configuration_rejects_ambiguous_values(values):
    from evi_hgt.settings import Settings
    with pytest.raises(ValueError):
        Settings(**values)
