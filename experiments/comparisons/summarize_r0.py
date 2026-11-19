"""Recompute the common R0 table from saved disease-level audit records."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


METHODS = (
    "FULL", "HGTDR_ADAPTED", "TXGNN_ADAPTED", "PAIR_MLP_D1_MATCHED",
    "HGT_D1_MATCHED", "TRAIN_VISIBLE_POPULARITY",
)
FOLDS = tuple(f"R0F{fold}" for fold in range(5))
METRICS = (
    "auc", "ap", "P@5", "R@5", "Hits@5", "NDCG@5",
    "P@10", "R@10", "Hits@10", "NDCG@10",
)


def summarize(audit, legacy_scores):
    records = audit["records"]
    expected = {
        (method, head, fold)
        for method in METHODS for head in ("support", "contra") for fold in FOLDS
        if (method, head) != ("HGTDR_ADAPTED", "contra")
    }
    by_cell = {(r["method"], r["head"], r["cell"]): r for r in records}
    if len(records) != len(expected) or set(by_cell) != expected:
        raise ValueError("Incomplete or duplicate method, head, or fold records")
    legacy_key = "HGTDR_ADAPTED", "support", "R0F0"
    legacy_record = by_cell[legacy_key]
    if hashlib.sha256(legacy_scores).hexdigest() != legacy_record["metrics_sha256"]:
        raise ValueError("Legacy HGTDR R0F0 scores differ from the audit record")
    legacy_rows = json.loads(legacy_scores)["rows"]
    members = sorted((r["drug_id"], r["disease_id"], r["role"]) for r in legacy_rows)
    serialized = json.dumps(members, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    legacy_digest = hashlib.sha256(serialized).hexdigest()
    rows_by_cell = defaultdict(dict)
    for row in audit["disease_fold_metrics"]:
        key = row["method"], row["head"], row["cell"]
        disease = row["disease_id"]
        if key not in expected or disease in rows_by_cell[key]:
            raise ValueError("Unexpected or duplicate disease observation")
        if min(row["P"], row["U"]) <= 0 or row["P"] + row["U"] != row["N"]:
            raise ValueError("Expected an evaluable disease with recorded P and U")
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in METRICS):
            raise ValueError("Invalid disease metric")
        rows_by_cell[key][disease] = row

    summaries = []
    for method, head in dict.fromkeys((r["method"], r["head"]) for r in records):
        diseases = defaultdict(list)
        for fold in FOLDS:
            key = method, head, fold
            reference_key = "FULL", head, fold
            record, reference = by_cell[key], by_cell[reference_key]
            membership = legacy_digest if key == legacy_key else record["candidate_identity_sha256"]
            if not membership or membership != reference["candidate_identity_sha256"]:
                raise ValueError(f"Candidate identity differs for {key}")
            if any(record[k] != reference[k] for k in (
                "pairs", "evaluable_diseases", "skipped_diseases",
            )):
                raise ValueError(f"Candidate panel differs for {key}")
            cell, reference_cell = rows_by_cell[key], rows_by_cell[reference_key]
            if len(cell) != record["evaluable_diseases"] or cell.keys() != reference_cell.keys():
                raise ValueError(f"Evaluable disease membership differs for {key}")
            for disease, row in cell.items():
                if any(row[k] != reference_cell[disease][k] for k in ("P", "U")):
                    raise ValueError(f"Disease candidate counts differ for {key}")
                diseases[disease].append(row)
        values = {
            k: statistics.fmean(statistics.fmean(r[k] for r in rows) for rows in diseases.values())
            for k in METRICS
        }
        summaries.append(dict(method=method, head=head, diseases=len(diseases), **values))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path, help="Path to the local metrics_audit.json")
    parser.add_argument("legacy_scores", type=Path, help="Original HGTDR R0F0 selected validation.ROWS.json")
    args = parser.parse_args()
    summary = summarize(json.loads(args.audit.read_text(encoding="utf-8")), args.legacy_scores.read_bytes())
    writer = csv.DictWriter(sys.stdout, fieldnames=("method", "head", "diseases", *METRICS))
    writer.writeheader()
    writer.writerows(summary)
