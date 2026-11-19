import json
import os
from pathlib import Path
import uuid

import ablation_records as records
from evi_hgt.ranking_protocol.identity import ROOT, digest, file_sha
from evi_hgt_feature_baseline.common import read

OUT = ROOT / records.LOSS_OUTPUT
MANIFEST = OUT / records.LOSS_MANIFEST
ARMS = ("E0_D1_SIGMOID_CONTROL", "E1_D1_SOFTPLUS_RANKING")
LOSSES = dict(zip(ARMS, ("SYMMETRIC_BOUNDED_SIGMOID_STABLE_V1", "EMPIRICAL_PU_HALF_SOFTPLUS_V1")))
G = 6856
TOTAL = 20568
SEED = 1289779183
MILESTONES = (10284, 20568)
DOMAINS = ("validation", "auxiliary_validation")
OLD = ROOT / records.FEATURE_OUTPUT
RECOVERY = OLD / "recovery_v1"
SCHEDULE = ROOT / records.FEATURE_SCHEDULE
PAIRING = ROOT / records.FEATURE_PAIRING


def write(path, value, *, immutable=True):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise PermissionError("LOSS_COMPARISON output namespace required")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and immutable:
        if read(path) != value:
            raise FileExistsError("Immutable LOSS_COMPARISON artifact differs: " + str(path))
        return
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def journal(path, row):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise PermissionError("LOSS_COMPARISON log namespace required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def check_evaluation(step, domain):
    if step not in MILESTONES or domain not in DOMAINS:
        raise PermissionError("Only LOSS_COMPARISON R0F0 two registered development panels at 10284/20568")


def source_hashes():
    sources = dict(read(RECOVERY / "RECOVERY_EXECUTION_MANIFEST.json")["source_sha256"])
    for directory in records.LOSS_SOURCE_DIRS:
        for p in (ROOT / directory).glob("*"):
            if p.suffix in (".py", ".ps1"):
                sources[str(p.resolve())] = file_sha(p)
    sources[str((ROOT / "src/evi_hgt_console/__init__.py").resolve())] = file_sha(ROOT / "src/evi_hgt_console/__init__.py")
    return sources


def validate_manifest():
    m = read(MANIFEST)
    fixed = dict(schema=records.LOSS_SCHEMA, mode="DEVELOPMENT", arms=list(ARMS),
                 losses=LOSSES, updates_per_arm=TOTAL, total_logical_updates=2*TOTAL,
                 milestones=list(MILESTONES), evaluation_domains=list(DOMAINS),
                 repeat=0, fold=0, seed=SEED, G=G, G3=False, G4=False, G5=False,
                 old_G3="HELD", outer="UNSCORED", new_final="SEALED",
                 old_Formal_access=1, new_Final_access=0)
    if any(m.get(k) != v for k, v in fixed.items()):
        raise PermissionError("LOSS_COMPARISON authorization, loss or budget changed")
    for group in ("source_sha256", "artifact_sha256"):
        for p, h in m[group].items():
            if file_sha(p) != h:
                raise ValueError("Bound LOSS_COMPARISON source/artifact changed: " + p)
    return m


def identity(m, arm, mode="DEVELOPMENT"):
    if arm not in ARMS or mode != "DEVELOPMENT":
        raise PermissionError("Registered LOSS_COMPARISON development arm required")
    return dict(arm=arm, loss_id=LOSSES[arm], mode=mode, manifest_sha256=file_sha(MANIFEST),
                source_sha256=digest(m["source_sha256"]), data_bindings=m["data_bindings"],
                sampling_sha256=m["paired_stream_sha256"], model_initial_sha256=m["initial_tensor_sha256"])


def checkpoint_due(step):
    return 0 < step <= TOTAL and (step % 1024 == 0 or step in MILESTONES)
