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
