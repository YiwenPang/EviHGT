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
    """Match one complete registered runtime, retaining its distinct identity."""
    variants = [("historical-release", SOURCE / "runtime_sha256.json")]
    raw_manifest = SOURCE / "runtime_sha256_raw_preparation.json"
    if raw_manifest.is_file():
        variants.append(("raw-preparation-release", raw_manifest))
    workflow_manifest = SOURCE / "runtime_sha256_full_workflow.json"
    if workflow_manifest.is_file():
        variants.append(("full-workflow-release", workflow_manifest))
    files = [*(FROZEN / "src").rglob("*.py"), *(FROZEN / "configs").rglob("*.yaml"),
             FROZEN / "tests/data_fixture.py"]
    actual = {file.relative_to(FROZEN).as_posix(): file for file in files}
    mismatches = []
    observed = None
    for variant, manifest in variants:
        expected = json.loads(manifest.read_text(encoding="utf-8"))
        if actual.keys() != expected.keys():
            continue
        if observed is None:
            observed = {name: sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                        for name, path in actual.items()}
        changed = [name for name, choices in expected.items() if observed[name] not in choices]
        if not changed:
            return {"files": len(actual), "identical": True,
                    "runtime_variant": variant, "runtime_manifest": manifest.name,
                    "historical_runtime_identical": variant == "historical-release"}
        mismatches.append(f"{variant}: {', '.join(changed)}")
    if mismatches:
        raise RuntimeError("Cold-start runtime source differs: " + "; ".join(mismatches))
    raise RuntimeError(f"Runtime inventory differs from every registered cold-start release at {FROZEN}")
