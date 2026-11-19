"""Synthetic checks for the dependency-independent environment diagnostic."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[1] / "check_environment.py"
    spec = importlib.util.spec_from_file_location("environment_check_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_packages(checker, monkeypatch, *, missing=(), cuda=False, bf16=False):
    versions = {"torch": "2.13.0+cu132", "numpy": "2.4.0", "polars": "1.37.0",
                "pyarrow": "22.0.0", "PyYAML": "6.0.3", "tqdm": "4.67.1"}
    def version(name):
        if name in missing:
            raise checker.metadata.PackageNotFoundError(name)
        return versions[name]
    cuda_api = SimpleNamespace(
        is_available=lambda: cuda,
        device_count=lambda: 1 if cuda else 0,
        current_device=lambda: 0,
        is_bf16_supported=lambda: bf16,
        get_device_properties=lambda index: SimpleNamespace(name="Synthetic GPU", total_memory=16 * 2**30,
                                                            major=12, minor=0),
        mem_get_info=lambda index: (12 * 2**30, 16 * 2**30),
    )
    torch_module = SimpleNamespace(cuda=cuda_api, version=SimpleNamespace(cuda="13.2"))
    def import_module(name):
        assert name in ("torch", "numpy", "polars", "pyarrow", "yaml", "tqdm")
        return torch_module if name == "torch" else SimpleNamespace()
    monkeypatch.setattr(checker.metadata, "version", version)
    monkeypatch.setattr(checker.importlib, "import_module", import_module)
    monkeypatch.setattr(checker, "physical_memory", lambda: {"total_bytes": 64 * 2**30, "available_bytes": 32 * 2**30})


def read_report(path):
    reports = list(path.glob("*.json"))
    assert len(reports) == 1
    return json.loads(reports[0].read_text(encoding="utf-8"))


def test_help_uses_only_standard_library(checker, monkeypatch, capsys):
    def forbidden_import(name):
        raise AssertionError("Help must not import third-party packages")
    monkeypatch.setattr(checker.importlib, "import_module", forbidden_import)
    with pytest.raises(SystemExit) as result:
        checker.main(["--help"])
    assert result.value.code == 0
    assert "--device" in capsys.readouterr().out


def test_missing_torch_still_prints_table_and_writes_report(checker, monkeypatch, capsys, tmp_path):
    fake_packages(checker, monkeypatch, missing={"torch"})
    assert checker.main(["--output-dir", str(tmp_path)]) == 1
    report = read_report(tmp_path)
    assert report["exit_code"] == 1 and report["status"] == "NOT_READY"
    assert any(row["name"] == "torch" and row["status"] == "FAIL" for row in report["checks"])
    text = capsys.readouterr().out
    assert "Check" in text and "FAIL" in text and "torch" in text and "JSON report:" in text
    assert "\x1b" not in text and "\r" not in text


def test_cpu_target_does_not_require_cuda(checker, monkeypatch, tmp_path):
    fake_packages(checker, monkeypatch, cuda=False)
    assert checker.main(["--device", "cpu", "--output-dir", str(tmp_path)]) == 0
    report = read_report(tmp_path)
    assert report["status"] == "READY"
    assert all(row["status"] != "FAIL" for row in report["checks"])


def test_cuda_target_requires_bf16_and_reports_memory(checker, monkeypatch, tmp_path):
    fake_packages(checker, monkeypatch, cuda=True, bf16=False)
    assert checker.main(["--device", "cuda", "--output-dir", str(tmp_path)]) == 1
    report = read_report(tmp_path)
    assert any(row["name"] == "BF16" and row["status"] == "FAIL" for row in report["checks"])
    assert report["gpu"][0]["total_bytes"] == 16 * 2**30
    assert report["gpu"][0]["free_bytes"] == 12 * 2**30


def test_success_report_is_unique_and_cuda_check_has_no_training(checker, monkeypatch, tmp_path):
    fake_packages(checker, monkeypatch, cuda=True, bf16=True)
    assert checker.main(["--output-dir", str(tmp_path)]) == 0
    assert checker.main(["--output-dir", str(tmp_path)]) == 0
    reports = list(tmp_path.glob("*.json"))
    assert len(reports) == 2
    assert all(json.loads(path.read_text())["scope"] == "ENVIRONMENT_ONLY_NO_DATA_OR_MODEL_EXECUTION" for path in reports)


def test_version_bounds_are_checked_without_packaging(checker):
    assert checker.satisfies("2.13.0+cu132", ">=2.6,<3") is True
    assert checker.satisfies("2.5.1", ">=2.6,<3") is False
    assert checker.satisfies("3.0.0", ">=2.6,<3") is False
    assert checker.satisfies("2.6.0rc1", ">=2.6,<3") is None


def test_broken_import_is_reported_without_stopping_other_checks(checker, monkeypatch, tmp_path):
    fake_packages(checker, monkeypatch, cuda=True, bf16=True)
    original_import = checker.importlib.import_module
    def broken_import(name):
        if name == "pyarrow":
            raise ImportError("SYNTHETIC_TEST_ONLY missing shared library")
        return original_import(name)
    monkeypatch.setattr(checker.importlib, "import_module", broken_import)
    assert checker.main(["--output-dir", str(tmp_path)]) == 1
    report = read_report(tmp_path)
    assert any(row["name"] == "pyarrow" and row["status"] == "FAIL" for row in report["checks"])
    assert any(row["name"] == "BF16" and row["status"] == "PASS" for row in report["checks"])
    assert "Traceback" in next(error["traceback"] for error in report["errors"] if error["check"] == "pyarrow")


def test_report_write_failure_has_distinct_return_code(checker, monkeypatch, tmp_path, capsys):
    fake_packages(checker, monkeypatch, cuda=False)
    target = tmp_path / "existing_file"
    target.write_text("SYNTHETIC_TEST_ONLY")
    assert checker.main(["--device", "cpu", "--output-dir", str(target)]) == 2
    assert "Cannot save the environment report" in capsys.readouterr().err
    assert target.read_text() == "SYNTHETIC_TEST_ONLY"
