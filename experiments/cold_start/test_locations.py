import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import locations


def test_runtime_pin_detects_changes_and_extra_modules(monkeypatch):
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        files = ("src/evi_hgt/model.py", "tests/data_fixture.py", "configs/default.yaml")
        contents = b"# fixture\n"
        digest = hashlib.sha256(contents).hexdigest()
        expected = {name: [digest] for name in files}
        for name in files:
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents.replace(b"\n", b"\r\n"))
        (tmp_path / "runtime_sha256.json").write_text(json.dumps(expected), encoding="utf-8")
        monkeypatch.setattr(locations, "SOURCE", tmp_path)
        monkeypatch.setattr(locations, "FROZEN", tmp_path)

        assert locations.check_frozen()["files"] == len(files)
        model = tmp_path / "src/evi_hgt/model.py"
        model.write_bytes(b"# edited\n")
        with pytest.raises(RuntimeError, match="source differs"):
            locations.check_frozen()

        model.write_bytes(contents)
        extra = model.with_name("extra.py")
        extra.write_bytes(b"")
        with pytest.raises(RuntimeError, match="inventory differs"):
            locations.check_frozen()


def test_runtime_variants_require_one_complete_identity(monkeypatch, tmp_path):
    contents = b"# historical runtime\n"
    updated = b"# raw preparation CLI\n"
    base = ("src/evi_hgt/model.py", "tests/data_fixture.py", "configs/default.yaml")
    historical = {name: [hashlib.sha256(contents).hexdigest()] for name in base}
    raw = dict(historical)
    raw[base[0]] = [hashlib.sha256(updated).hexdigest()]
    addition = "src/evi_hgt/preprocessing/canonical.py"
    raw[addition] = [hashlib.sha256(contents).hexdigest()]
    (tmp_path / "runtime_sha256.json").write_text(json.dumps(historical), encoding="utf-8")
    (tmp_path / "runtime_sha256_raw_preparation.json").write_text(json.dumps(raw), encoding="utf-8")
    for name in base:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    monkeypatch.setattr(locations, "SOURCE", tmp_path)
    monkeypatch.setattr(locations, "FROZEN", tmp_path)

    historical_result = locations.check_frozen()
    assert historical_result["runtime_variant"] == "historical-release"
    assert historical_result["historical_runtime_identical"] is True
    new_module = tmp_path / addition
    new_module.parent.mkdir(parents=True, exist_ok=True)
    new_module.write_bytes(contents)
    (tmp_path / base[0]).write_bytes(updated.replace(b"\n", b"\r\n"))
    current_result = locations.check_frozen()
    assert current_result["runtime_variant"] == "raw-preparation-release"
    assert current_result["historical_runtime_identical"] is False
    assert current_result["files"] == 4
    assert current_result["identical"] is True

    # The enlarged inventory cannot borrow an old digest for a changed file.
    (tmp_path / base[0]).write_bytes(contents)
    with pytest.raises(RuntimeError, match="source differs"):
        locations.check_frozen()
    (tmp_path / base[0]).write_bytes(updated)
    extra = new_module.with_name("unregistered.py")
    extra.write_bytes(contents)
    with pytest.raises(RuntimeError, match="inventory differs"):
        locations.check_frozen()
    extra.unlink()
    (tmp_path / base[2]).write_bytes(b"changed: true\n")
    with pytest.raises(RuntimeError, match="source differs"):
        locations.check_frozen()
