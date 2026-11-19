import json
import os
from pathlib import Path
import uuid

from evi_hgt.phase9.identity import ROOT, digest, file_sha
from evi_hgt_dev94.common import read

OUT = ROOT / "outputs/phase9/phase95"
MANIFEST = OUT / "DEV95_EXECUTION_MANIFEST.json"
ARMS = ("E0_D1_SIGMOID_CONTROL", "E1_D1_SOFTPLUS_RANKING")
LOSSES = dict(zip(ARMS, ("SYMMETRIC_BOUNDED_SIGMOID_STABLE_V1", "EMPIRICAL_PU_HALF_SOFTPLUS_V1")))
G = 6856
TOTAL = 20568
SEED = 1289779183
MILESTONES = (10284, 20568)
DOMAINS = ("validation", "auxiliary_validation")
OLD = ROOT / "outputs/phase9/phase94"
RECOVERY = OLD / "recovery_v1"
SCHEDULE = ROOT / "data/processed/phase94/schedule_v1/schedule.json"
PAIRING = ROOT / "data/processed/phase94/conditional_v1/plan.json"


def write(path, value, *, immutable=True):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise PermissionError("DEV95 output namespace required")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and immutable:
        if read(path) != value:
            raise FileExistsError("Immutable DEV95 artifact differs: " + str(path))
        return
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def journal(path, row):
    path = Path(path).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise PermissionError("DEV95 log namespace required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def check_evaluation(step, domain):
    if step not in MILESTONES or domain not in DOMAINS:
        raise PermissionError("Only DEV95 R0F0 two registered development panels at 10284/20568")


def source_hashes():
    sources = dict(read(RECOVERY / "RECOVERY_EXECUTION_MANIFEST.json")["source_sha256"])
    for directory in ("src/evi_hgt_dev95", "scripts/phase95", "tests/phase95"):
        for p in (ROOT / directory).glob("*"):
            if p.suffix in (".py", ".ps1"):
                sources[str(p.resolve())] = file_sha(p)
    sources[str((ROOT / "src/evi_hgt_console/__init__.py").resolve())] = file_sha(ROOT / "src/evi_hgt_console/__init__.py")
    return sources


def validate_manifest():
    m = read(MANIFEST)
    fixed = dict(schema="DEV95_TWO_ARM_DEVELOPMENT_V1", mode="DEVELOPMENT", arms=list(ARMS),
                 losses=LOSSES, updates_per_arm=TOTAL, total_logical_updates=2*TOTAL,
                 milestones=list(MILESTONES), evaluation_domains=list(DOMAINS),
                 repeat=0, fold=0, seed=SEED, G=G, G3=False, G4=False, G5=False,
                 old_G3="HELD", outer="UNSCORED", new_final="SEALED",
                 old_Formal_access=1, new_Final_access=0)
    if any(m.get(k) != v for k, v in fixed.items()):
        raise PermissionError("DEV95 authorization, loss or budget changed")
    for group in ("source_sha256", "artifact_sha256"):
        for p, h in m[group].items():
            if file_sha(p) != h:
                raise ValueError("Bound DEV95 source/artifact changed: " + p)
    return m


def identity(m, arm, mode="DEVELOPMENT"):
    if arm not in ARMS or mode != "DEVELOPMENT":
        raise PermissionError("Registered DEV95 development arm required")
    return dict(arm=arm, loss_id=LOSSES[arm], mode=mode, manifest_sha256=file_sha(MANIFEST),
                source_sha256=digest(m["source_sha256"]), data_bindings=m["data_bindings"],
                sampling_sha256=m["paired_stream_sha256"], model_initial_sha256=m["initial_tensor_sha256"])


def checkpoint_due(step):
    return 0 < step <= TOTAL and (step % 1024 == 0 or step in MILESTONES)
