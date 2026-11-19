"""Check mirrored source or the sampling identities in the original development logs."""
import argparse
import hashlib
import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
PAIRING_FIELDS = (
    "training_member_sha256", "record_sha256", "sampling_sha256", "stage",
    "queries", "original_queries", "conditional_extra_U_queries", "support_pairs", "contra_pairs",
)
LOSS_ARMS = ("E0_D1_SIGMOID_CONTROL", "E1_D1_SOFTPLUS_RANKING")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(research_root):
    manifest = json.loads((DIRECTORY / "SHA256.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        source = DIRECTORY / record["file"]
        if sha(source) != record["lf_sha256"]:
            raise ValueError(f"Public source differs: {record['file']}")
        if research_root and sha(research_root / record["source"]) != record["source_sha256"]:
            raise ValueError(f"Research source differs: {record['source']}")
    if research_root:
        for name, receipt in manifest["receipts"].items():
            if sha(research_root / receipt["source"]) != receipt["sha256"]:
                raise ValueError(f"Historical receipt differs: {name}")
    print(f"Verified {len(manifest['files'])} source files.")


def read_steps(path, arm, total):
    steps = {}
    replays = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            step = row["optimizer_step"]
            if row["mode"] != "DEVELOPMENT" or row["arm"] != arm or row["planned_optimizer_steps"] != total:
                raise ValueError(f"{path}: unexpected arm, budget or run mode at step {step}")
            pairing = tuple(row[field] for field in PAIRING_FIELDS)
            if step in steps:
                if steps[step] != pairing:
                    raise ValueError(f"{path}: replay sample mismatch at step {step}")
                replays += 1
            steps[step] = pairing
    if sorted(steps) != list(range(1, total + 1)):
        raise ValueError(f"{path}: incomplete logical step sequence")
    return steps, replays


def verify_pairing(research_root):
    base = research_root / "outputs" / "phase9"
    for cell, total in (("R0F0", 20568), ("R0F1", 20541), ("R1F0", 20700)):
        loss_root = base / "phase95" if cell == "R0F0" else base / "phase96" / cell
        arms = [(arm, loss_root) for arm in LOSS_ARMS]
        if cell != "R0F0":
            arms.extend((arm, base / "phase97" / cell) for arm in ("A_EP_ZERO", "A_QUALITY_CONST"))
        reference = None
        for arm, directory in arms:
            rows, replays = read_steps(directory / arm / "updates.jsonl", arm, total)
            if reference is None:
                reference = rows
            elif rows != reference:
                mismatch = next(step for step in sorted(reference) if rows[step] != reference[step])
                raise ValueError(f"{cell}/{arm}: sampling differs at step {mismatch}")
            print(f"{cell}/{arm}: {len(rows)} matched logical updates, {replays} logged replays")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checks = parser.add_subparsers(dest="check", required=True)
    checks.add_parser("sources").add_argument("--research-root", type=Path)
    checks.add_parser("pairing").add_argument("--research-root", required=True, type=Path)
    args = parser.parse_args()
    if args.check == "sources":
        verify_sources(args.research_root)
    else:
        verify_pairing(args.research_root)


if __name__ == "__main__":
    main()
