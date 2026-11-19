"""Paths and source identities for the published cold-start scripts."""

from hashlib import sha256
import json
import os
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
REPOSITORY = SOURCE.parents[1]
WORKSPACE = Path(os.environ.get("EVIHGT_RESEARCH_ROOT", REPOSITORY)).expanduser().resolve()
FROZEN = Path(os.environ.get("EVIHGT_FROZEN_ROOT", REPOSITORY)).expanduser().resolve()
D1_FEATURES = Path(os.environ.get(
    "EVIHGT_D1_FEATURES",
    WORKSPACE.parent / "HGTDR/data/processed/competition_pilot_v1/features",
)).expanduser().resolve()
# Relative paths in the manifests issued by the research workspace.
CANONICAL_GRAPH = Path("data/processed/phase3/v1")
DEVELOPMENT_INPUTS = Path("data/processed/phase85_part2")
DISEASE_GENE_INPUTS = Path("data/processed/phase85_part2b")
GRAPH_ACCEPTANCE = Path("outputs/phase85_part2b/graph_acceptance.json")
CASE_CACHE = Path("data/cache/phase98/cold_start_20261118")
CACHE = WORKSPACE / CASE_CACHE
EXPERIMENT = WORKSPACE / "outputs/phase9/phase98/cold_start_20261118"


def check_frozen():
    """Verify the historical or released spelling of each runtime source file."""
    expected = json.loads((SOURCE / "runtime_sha256.json").read_text(encoding="utf-8"))
    files = [*(FROZEN / "src").rglob("*.py"), *(FROZEN / "configs").rglob("*.yaml"),
             FROZEN / "tests/data_fixture.py"]
    actual = {file.relative_to(FROZEN).as_posix(): file for file in files}
    if actual.keys() != expected.keys():
        raise RuntimeError(f"Runtime inventory differs from the cold-start release at {FROZEN}")
    for name, choices in expected.items():
        normalized = actual[name].read_bytes().replace(b"\r\n", b"\n")
        if sha256(normalized).hexdigest() not in choices:
            raise RuntimeError(f"Cold-start runtime source differs: {name}")
    return {"files": len(actual), "identical": True}
