"""Checks that the cold-disease release can be inspected without native packages."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def test_summary_uses_only_the_standard_library():
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-m", "experiments.cold_start.hgtdr.run", "summary"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    )
    summary = json.loads(result.stdout)
    assert summary == json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    assert summary["comparison"]["support"]["evaluable_diseases"] == 196
    assert summary["comparison"]["contra"]["evaluable_diseases"] == 45
    assert summary["selection"]["HGTDR"]["selected_epoch"] == {"support": 149, "contra": 53}


def test_published_source_checksums():
    manifest = json.loads((HERE / "source_sha256.json").read_text(encoding="utf-8"))
    for name, hashes in manifest.items():
        content = (HERE / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == hashes["published_sha256_lf"]
