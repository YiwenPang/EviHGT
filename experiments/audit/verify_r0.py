"""Check the five-fold comparison ledger. Optionally verify the original local artifacts."""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean

FOLDS = {f"R0F{i}" for i in range(5)}
METHODS = {"FULL", "HGT_D1_MATCHED", "PAIR_MLP_D1_MATCHED", "TRAIN_VISIBLE_POPULARITY",
           "HGTDR_ADAPTED", "TXGNN_ADAPTED"}


def checked_file(root, relative, expected_digest):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path outside supplied workspace: {relative}")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected_digest:
        raise ValueError(f"File digest differs: {relative}")
    return contents


def artifact_locations(root, index_path):
    if index_path is None:
        matches = sorted((root / "outputs").glob("*/*/analysis_*/metrics_audit.json"))
        if len(matches) == 1:
            index_path = matches[0]
    if index_path is not None:
        entries = json.loads(index_path.read_bytes())["records"]
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from replay import historical_sources

        with historical_sources() as history:
            ledger = history / "audit" / "r0_fold_audit.csv"
            with ledger.open(encoding="utf-8", newline="") as stream:
                entries = list(csv.DictReader(stream))
    locations = {}
    for entry in entries:
        key = (entry["method"], entry["cell"], entry["head"])
        if key in locations:
            raise ValueError(f"Duplicate local audit entry: {key}")
        paths = {}
        for field in ("metrics_path", "complete_path"):
            parts = PurePosixPath(entry[field].replace("\\", "/")).parts
            paths[field] = Path(*parts[parts.index("outputs"):])
        locations[key] = paths
    return locations


def verify_artifacts(root, row):
    checked_file(root, row["complete_path"], row["complete_sha256"])
    payload = json.loads(checked_file(root, row["metrics_path"], row["metrics_sha256"]))
    metrics = payload.get("metrics", payload)
    metrics = metrics.get(row["head"], metrics)
    panel_digest = metrics.get("candidate_identity_sha256")
    if "rows" in payload:
        members = sorted((r["drug_id"], r["disease_id"], r["role"]) for r in payload["rows"])
        roles = Counter(role for _, _, role in members)
        if roles != {"P": int(row["P"]), "U": int(row["U"])}:
            raise ValueError("Saved candidate roles differ")
        encoded = json.dumps(members, separators=(",", ":"), ensure_ascii=False).encode()
        panel_digest = hashlib.sha256(encoded).hexdigest()
    elif (metrics["P"], metrics["U"]) != (int(row["P"]), int(row["U"])):
        raise ValueError("Saved candidate role counts differ")
    if panel_digest != row["candidate_identity_sha256"] or metrics["pairs"] != int(row["pairs"]):
        raise ValueError("Saved candidate panel differs")
    diseases = metrics["disease_metrics"]
    if len(diseases) != int(row["evaluable_diseases"]):
        raise ValueError("Evaluable disease count differs")
    for metric, column in (("auc", "macro_auc"), ("ap", "macro_ap"), ("NDCG@10", "NDCG@10")):
        value = mean(d[metric] for d in diseases)
        if not math.isclose(value, float(row[column]), abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Fold metric differs: {row['method']} {row['cell']} {column}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, help="original workspace containing the bound results")
    parser.add_argument("--artifact-index", type=Path, help="local R0 metrics audit with artifact locations")
    args = parser.parse_args()
    root = args.research_root.resolve() if args.research_root else None
    if args.artifact_index and root is None:
        parser.error("--artifact-index requires --research-root")
    locations = artifact_locations(root, args.artifact_index) if root else {}
    expected = {(method, cell, head) for method in METHODS for cell in FOLDS
                for head in ("support", "contra") if (method, head) != ("HGTDR_ADAPTED", "contra")}
    seen = set()
    panels = defaultdict(set)
    with Path(__file__).with_name("r0_fold_audit.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["method"], row["cell"], row["head"])
            if key in seen or key not in expected or int(row["P"]) + int(row["U"]) != int(row["pairs"]):
                raise ValueError(f"Invalid or duplicate comparison row: {key}")
            seen.add(key)
            panels[key[1:]].add(tuple(row[col] for col in
                ("P", "U", "pairs", "evaluable_diseases", "candidate_identity_sha256")))
            if root:
                verify_artifacts(root, {**row, **locations[key]})
    if seen != expected or any(len(identities) != 1 for identities in panels.values()):
        raise ValueError("Incomplete roster or inconsistent within-fold candidate identities")
    print(f"Verified {len(seen)} method/fold/head entries across {len(panels)} common panels.")
    if root:
        print("Verified source checksums, candidate identities and fold-level macro metrics.")


if __name__ == "__main__":
    main()
