"""Run one frozen FULL trajectory; attach separately sealed P/U test panels afterward."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import file_digest
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys
import time
import traceback

sys.dont_write_bytecode = True

from locations import CACHE as CORPORA, EXPERIMENT as HERE, FROZEN, check_frozen
CASES = {
    "cold_drug": ("drug",),
    "cold_disease": ("disease",),
    "double_cold": ("drug", "disease"),
    "first_approval_year_cohort": ("drug",),
}
RELATIONS = {"support": "INDICATION", "contra": "CONTRAINDICATION"}
PAIR_FIELDS = ("drug_id", "disease_id")
PANEL_FIELDS = (*PAIR_FIELDS, "role", "raw_relation")
PANEL_NAMES = {head: f"test_{head}.parquet" for head in RELATIONS}
TEST_SCOPE = "SINGLE_COLD_SPLIT_INDEPENDENT_P_VS_U_TEST_PANEL"


def now():
    return datetime.now(timezone.utc).isoformat()


def audit_memberships(case, panels, blocked, observed, train, dev, eligible_drugs, eligible_diseases):
    """Audit experimental test membership independently of the portable corpus reader."""
    if case not in CASES or set(panels) != set(RELATIONS):
        raise ValueError("Unknown experiment case or missing independent test head")
    counts = {}
    recorded = {}
    for head, relation in RELATIONS.items():
        rows = panels[head]
        if not rows:
            raise ValueError(f"Empty {head} test panel")
        by_role = {"P": set(), "U": set()}
        for row in rows:
            pair = tuple(row[key] for key in PAIR_FIELDS)
            role = row.get("role")
            if any(not isinstance(item, str) or not item for item in pair) or role not in by_role:
                raise ValueError(f"Invalid canonical P/U candidate in {head}")
            if row.get("raw_relation") != (relation if role == "P" else None):
                raise ValueError(f"Wrong raw clinical relation in {head}; U must be unlabelled")
            if pair in by_role["P"] or pair in by_role["U"]:
                raise ValueError(f"Duplicate pair in {head} test candidates")
            by_role[role].add(pair)
        if not all(by_role.values()):
            raise ValueError(f"Both P and U are required in the {head} test panel")
        if not by_role["P"] <= observed or by_role["U"] & observed:
            raise ValueError(f"Test P is missing from observed clinical pairs or test U is observed: {head}")
        recorded[head] = by_role
        counts[head] = {role: len(pairs) for role, pairs in by_role.items()}
    if recorded["support"]["U"] != recorded["contra"]["U"]:
        raise ValueError("Both test heads must use the same U candidate membership")
    pairs = set().union(*(part for roles in recorded.values() for part in roles.values()))
    if not pairs <= blocked:
        raise ValueError("One or more test pairs are absent from blocked training/graph membership")
    if pairs & (train | dev):
        raise ValueError("Test candidate leaked into training or development membership")
    for axis in CASES[case]:
        i = {"drug": 0, "disease": 1}[axis]
        test_entities = {p[i] for p in pairs}
        training_entities = {p[i] for p in train} | (eligible_drugs if i == 0 else eligible_diseases)
        if test_entities & training_entities:
            raise ValueError(f"Test entity is present in positive training or U eligibility on cold axis {axis}")
        if test_entities & {p[i] for p in dev}:
            raise ValueError(f"Test entity is present in the development panel on cold axis {axis}")
    return dict(test_union_pairs=len(pairs), by_head=counts, cold_axes=list(CASES[case]),
                test_blocking_verified=True, test_dev_disjoint=True, test_training_disjoint=True,
                shared_u=True, test_u_observed_overlap=0)


def tie_diagnostics(rows, score_field):
    by_disease = defaultdict(list)
    for row in rows:
        by_disease[row["disease_id"]].append(row)
    all_groups, evaluable_groups = [], []
    for members in by_disease.values():
        scores = Counter(row[score_field] for row in members)
        group = (len(scores) == 1, max(scores.values()) / len(members))
        all_groups.append(group)
        if {r["role"] for r in members} == {"P", "U"}:
            evaluable_groups.append(group)

    def summarize(groups):
        n = len(groups)
        same = sum(constant for constant, _ in groups)
        return dict(diseases=n, constant_score_diseases=same,
                    constant_score_fraction=same / n if n else None,
                    mean_largest_tie_fraction=fmean(size for _, size in groups) if n else None)

    return dict(all_diseases=summarize(all_groups), evaluable_diseases=summarize(evaluable_groups))


def _sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path, data, *, exclusive=False):
    serialized = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if exclusive:
        with path.open("x", encoding="utf8") as stream:
            stream.write(serialized)
        return
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(serialized, encoding="utf8")
    os.replace(tmp, path)


def _bootstrap():
    check_frozen()
    frozen_src = (FROZEN / "src").resolve(strict=True)
    sys.path.insert(0, str(frozen_src))
    from evi_hgt import runner, settings
    if not all(Path(m.__file__).resolve().is_relative_to(frozen_src) for m in (runner, settings)):
        raise RuntimeError("Model imported from outside frozen release src")
    return runner, settings


def _localized_console(base):
    phases = {
        "Validate and load prepared inputs": "Validate and load inputs",
        "Initialize EviHGT": "Initialize EviHGT",
        "Train the fixed three-cycle plan": "Train for three fixed cycles",
        "Save final development results": "Save development results",
    }

    class ColdConsole(base):
        def phase(self, index, title):
            return super().phase(index, phases.get(title, title))

        def banner(self):
            self.say("=" * 72)
            self.say("EviHGT cold start | Separate support and contra outputs")
            self.say("=" * 72)

        def failure(self, exc):
            with self.error_path.open("a", encoding="utf8") as stream:
                traceback.print_exception(exc, file=stream)
            self.event("failure", type=type(exc).__name__, message=str(exc))
            self.say(f"Stage failed ({type(exc).__name__}). Error log: {self.error_path}")

        def metrics(self, metrics):
            self.say("Development evaluation | Recorded positives and unlabeled candidates")
            _print_metrics(self.say, metrics)
            self.event("evaluation_completed", metrics=metrics)

    return ColdConsole


def _print_metrics(say, metrics):
    def fmt(value):
        return "Unavailable" if value is None else f"{value:.6g}"

    say(f"{'Head':<12} {'ROC-AUC':>12} {'AP':>12} {'Diseases':>8}")
    for head, result in metrics.items():
        say(f"{head:<12} {fmt(result.get('macro_auc')):>12} {fmt(result.get('macro_ap')):>12} "
            f"{result.get('evaluable_diseases', 0):>8}")
    say(f"{'Head':<12} {'K':>3} {'Precision':>12} {'Recall':>12} {'Hits':>12} {'NDCG':>12}")
    for head, result in metrics.items():
        for k in (5, 10):
            values = [fmt(result.get("ranking", {}).get(f"{key}@{k}")) for key in ("P", "R", "Hits", "NDCG")]
            say(f"{head:<12} {k:>3} " + " ".join(f"{value:>12}" for value in values))


def _corp_input(corpus_path, relative, inventory):
    if relative not in inventory or not isinstance(relative, str):
        raise ValueError("Portable corpus path is not content-bound")
    part = Path(relative)
    root = corpus_path.parent
    if part.is_absolute() or ".." in part.parts or "\\" in relative or ":" in relative:
        raise ValueError("Portable corpus input must be a confined relative POSIX path")
    path = (root / part).resolve(strict=True)
    if not path.is_relative_to(root) or _sha(path) != inventory[relative]:
        raise ValueError(f"Invalid portable corpus membership path or hash: {relative}")
    return path


def _preflight(case, corpus_path):
    import polars as pl

    manifest = _read_json(corpus_path)
    inventory = manifest["files"]
    claimed_test = {
        (corpus_path.parent / relative).resolve(strict=True)
        for relative in PANEL_NAMES.values()
    }
    if any(not path.is_relative_to(corpus_path.parent) for path in claimed_test):
        raise ValueError("Test inputs must stay in their case directory")
    if any((corpus_path.parent / name).resolve() in claimed_test for name in inventory):
        raise ValueError("Test panel was attached to the frozen training inventory")

    def pairs(relative):
        p = _corp_input(corpus_path, relative, inventory)
        return set(pl.read_parquet(p, columns=list(PAIR_FIELDS)).iter_rows())

    members, hashes = {}, {}
    for head, relative in PANEL_NAMES.items():
        path = corpus_path.parent / relative
        hashes[head] = _sha(path)
        members[head] = pl.read_parquet(path, columns=list(PANEL_FIELDS)).to_dicts()
        if _sha(path) != hashes[head]:
            raise ValueError(f"{head} test panel changed during preflight")
    dev = set().union(*(pairs(relative) for relative in manifest.get("panels", {}).values()))
    eligible = _read_json(_corp_input(corpus_path, manifest["eligibility"], inventory))
    audit = audit_memberships(case, members, pairs(manifest["blocked_pairs"]),
                              pairs(manifest["observed_pairs"]), pairs(manifest["training_pairs"]), dev,
                              set(eligible["eligible_drugs"]), set(eligible["eligible_diseases"]))
    audit.update(panel_sha256=hashes, training_seed=manifest["training_seed"],
                 expected_updates=3 * manifest["schedule"]["G"], manifest_sha256=_sha(corpus_path))
    audit_path = _corp_input(corpus_path, manifest["cold_case_receipt"], inventory)
    prepared = _read_json(audit_path)
    if (prepared.get("case") != case or prepared.get("train_updates") != audit["expected_updates"]
        or prepared.get("test_files") != {PANEL_NAMES[head]: hashed for head, hashed in hashes.items()}):
        raise ValueError("Preparation receipt does not bind this case's test panel identities")
    audit["preparation_audit_sha256"] = _sha(audit_path)
    return audit


def _training_settings(settings_mod, corpus_path, output):
    return replace(settings_mod.load_settings(FROZEN / "configs/default.yaml"),
                   manifest=str(corpus_path), output_dir=str(output))


def _checkpoint_receipt(path, root, audit):
    if not path.is_dir() or not path.is_relative_to(root):
        raise ValueError("Invalid frozen-run result directory")
    completed = _read_json(path / "completion.json")
    expected = audit["expected_updates"]
    if (completed.get("status"), completed.get("variant"), completed.get("mode"),
        completed.get("updates"), completed.get("total_updates")) != ("COMPLETE", "FULL", "train", expected, expected):
        raise ValueError("Frozen FULL did not complete all three cycles")
    cp = path / "checkpoints/latest.pt"
    return dict(completed, root=str(path), checkpoint=str(cp), checkpoint_sha256=_sha(cp),
                result_sha256=_sha(path / "completion.json"), recorded_at=now())


def _score(case, runner, settings, root, audit, training):
    import polars as pl
    import torch
    from evi_hgt.metrics import evaluate_panel

    started = time.monotonic()
    target = root / "test"
    target.mkdir(exist_ok=True)
    corpus_path = Path(settings.manifest)
    final = Path(training["checkpoint"]).resolve(strict=True)
    if not final.is_relative_to(root) or _sha(final) != training["checkpoint_sha256"]:
        raise ValueError("Final checkpoint identity changed or escaped its case")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    members = {}
    if _sha(corpus_path) != audit["manifest_sha256"]:
        raise ValueError("Training corpus manifest has changed before test evaluation")
    for head, filename in PANEL_NAMES.items():
        path = corpus_path.parent / filename
        if _sha(path) != audit["panel_sha256"][head]:
            raise ValueError("Sealed test candidates changed before evaluation")
        members[head] = pl.read_parquet(path, columns=list(PANEL_FIELDS)).to_dicts()
    with runner.Console(target / "logs", refresh_seconds=settings.refresh_seconds) as console:
        with console.phase(1, "Load the fixed graph and final checkpoint"):
            corpus = runner.load_inputs(settings, console)
            saved = torch.load(final, map_location="cpu", weights_only=False)
            if (saved.get("schema") != "EVIHGT_CHECKPOINT_V1" or saved.get("step") != corpus.plan.total_steps
                or training["corpus_identity"] != corpus.identity or saved["binding"].get("corpus") != corpus.identity
                or saved["binding"].get("variant") != "FULL"):
                raise ValueError("Final checkpoint does not bind this complete case corpus")
            built = runner.build_model(corpus.model_config, pretrained=corpus.pretrained,
                                       seed=corpus.training_seed, variant="FULL")
            if saved["binding"].get("model") != built.metadata():
                raise ValueError("Final model metadata differs from checkpoint binding")
            built.model.load_state_dict(saved["model"], strict=True)
            del saved
            built.model.to(settings.device)
            built.model.eval()
            pairs = set().union(*({tuple(r[key] for key in PAIR_FIELDS) for r in rows}
                                  for rows in members.values()))
            if not pairs <= corpus.blocked:
                raise ValueError("Test pairs missing frozen graph/training exclusions")
            if "cold_test_inference" in corpus.panels:
                raise ValueError("Test domain was visible during training")
            # This role-free dataset is attached only to the fresh post-training
            # instance, so QueryAssembler uses its unchanged membership guard.
            domain = "cold_test_inference"
            corpus.panels[domain] = None
            corpus.panel_pairs[domain] = pairs
        infer_start = time.monotonic()
        scores = {}
        with console.phase(2, "Score the sealed test candidates"):
            with console.progress(len(pairs), "Test scoring", unit="pair") as bar:
                with torch.no_grad(), runner.precision_context(settings):
                    for pair in sorted(pairs):
                        q = runner.make_unlabeled_query_supervision(*pair)
                        graph = corpus.assembler.assemble(q, domain=domain, device=settings.device)
                        pred = graph.forward(built)
                        support = float(pred.support_logit.float().item())
                        contra = float(pred.unsuitability_logit.float().item())
                        if not (math.isfinite(support) and math.isfinite(contra)):
                            raise FloatingPointError("Nonfinite frozen test score")
                        source = graph.batch.path_source
                        incident = {ix for f in source.facts for ix in (f.source_node_index, f.target_node_index)}
                        scores[pair] = dict(support_logit=support, unsuitability_logit=contra,
                                            graph_nodes=graph.biological_nodes, graph_facts=len(source.facts),
                                            biological_paths=graph.batch.path_batch.num_paths,
                                            drug_has_graph_fact=source.drug_node_index in incident,
                                            disease_has_graph_fact=source.disease_node_index in incident)
                        bar.update()
        infer_elapsed = time.monotonic() - infer_start
        with console.phase(3, "Save metrics, score ties and graph coverage"):
            metrics, ties, output_hashes = {}, {}, {}
            for head, rows in members.items():
                scored = [dict(row, support_logit=scores[tuple(row[k] for k in PAIR_FIELDS)]["support_logit"],
                               unsuitability_logit=scores[tuple(row[k] for k in PAIR_FIELDS)]["unsuitability_logit"])
                          for row in rows]
                metrics[head] = evaluate_panel(scored, head=head, expected_members=rows, scope=TEST_SCOPE)
                ties[head] = tie_diagnostics(scored, "support_logit" if head == "support" else "unsuitability_logit")
                out = target / f"{head}_scores.parquet"
                tmp = out.with_suffix(".parquet.partial")
                pl.DataFrame(scored).write_parquet(tmp)
                os.replace(tmp, out)
                output_hashes[out.name] = _sha(out)
            out = target / "pair_graph_coverage.parquet"
            tmp = out.with_suffix(".parquet.partial")
            pl.DataFrame([dict(drug_id=pair[0], disease_id=pair[1], **row)
                          for pair, row in scores.items()]).write_parquet(tmp)
            os.replace(tmp, out)
            output_hashes[out.name] = _sha(out)
            n = len(scores)
            coverage = dict(distinct_pairs=n,
                            pairs_with_any_graph_fact=sum(row["graph_facts"] > 0 for row in scores.values()),
                            pairs_with_path=sum(row["biological_paths"] > 0 for row in scores.values()),
                            drug_endpoint_with_fact=sum(row["drug_has_graph_fact"] for row in scores.values()),
                            disease_endpoint_with_fact=sum(row["disease_has_graph_fact"] for row in scores.values()))
            elapsed = time.monotonic() - started
            result = dict(status="COMPLETE", case=case, scope=TEST_SCOPE, training_seed=corpus.training_seed,
                          updates=corpus.plan.total_steps, frozen_source=str(FROZEN),
                          training_corpus_identity=corpus.identity, checkpoint_sha256=training["checkpoint_sha256"],
                          membership=audit, metrics=metrics, tie_diagnostics=ties, graph_coverage=coverage,
                          artifact_sha256=output_hashes, graph_scoring_seconds=infer_elapsed,
                          test_stage_seconds=elapsed, recorded_at=now())
            _write_json(target / "metrics.json", result)
            _print_metrics(console.say, metrics)
        return result


@contextmanager
def _case_lock(root):
    path = root / "active.lock"
    with path.open("x", encoding="utf8") as file:
        file.write(f"{os.getpid()} {now()}\n")
    try:
        yield
    finally:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "validate"), required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Explicit continuation of a failed invocation in this case only")
    parser.add_argument("--resume-test", action="store_true",
                        help="Explicitly continue scoring after already completed frozen 3G training")
    args = parser.parse_args()
    if args.resume_checkpoint and args.resume_test or args.mode == "validate" and (args.resume_checkpoint or args.resume_test):
        parser.error("Continuation options are exclusive and available only for --mode train")
    root = HERE / ("validations" if args.mode == "validate" else "runs") / args.case
    root.mkdir(parents=True, exist_ok=True)
    corpus_path = (CORPORA / args.case / "manifest.json").resolve()
    try:
        with _case_lock(root):
            invocation_started = time.monotonic()
            audit = _preflight(args.case, corpus_path)
            runner, settings_mod = _bootstrap()
            settings = _training_settings(settings_mod, corpus_path, root / "frozen_runs")
            runner.Console = _localized_console(runner.Console)
            if args.mode == "validate":
                result = runner.run(settings, variant="FULL", mode="validate")
                _write_json(root / "validation.json", dict(audit, status="VALIDATED", frozen_run=str(result), recorded_at=now()))
                return 0
            completed_path = root / "result.json"
            attempt_path = root / "first_invocation.json"
            training_path = root / "training_complete.json"
            if completed_path.is_file():
                raise FileExistsError(f"This case was completed once already: {completed_path}")
            resumed = bool(args.resume_checkpoint or args.resume_test)
            if resumed:
                initial = _read_json(attempt_path)
                if audit != initial["membership"]:
                    raise ValueError("Cannot continue with changed inputs or test panels")
                _write_json(root / f"continuation_{time.time_ns()}.json",
                            dict(started_at=now(), resume_test=args.resume_test,
                                 checkpoint=str(args.resume_checkpoint) if args.resume_checkpoint else None),
                            exclusive=True)
            else:
                _write_json(attempt_path, dict(started_at=now(), membership=audit,
                            frozen_source=str(FROZEN), variant="FULL"), exclusive=True)
            if args.resume_test:
                training = _read_json(training_path)
            else:
                checkpoint = None
                if args.resume_checkpoint:
                    if training_path.exists():
                        raise ValueError("Training already complete; use --resume-test for scoring")
                    checkpoint = args.resume_checkpoint.resolve(strict=True)
                    if not checkpoint.is_relative_to(root / "frozen_runs") or checkpoint.name != "latest.pt":
                        raise ValueError("Resume only from a trusted case-local latest checkpoint")
                result_dir = runner.run(settings, variant="FULL", mode="train", checkpoint=checkpoint).resolve()
                training = _checkpoint_receipt(result_dir, root, audit)
                _write_json(training_path, training, exclusive=True)
            result = _score(args.case, runner, settings, root, audit, training)
            total = time.monotonic() - invocation_started
            _write_json(completed_path, dict(status="COMPLETE", case=args.case, metric_file="test/metrics.json",
                        metric_sha256=_sha(root / "test/metrics.json"), metrics=result["metrics"],
                        test_scoring_seconds=result["graph_scoring_seconds"],
                        training_invocation_seconds=training["elapsed_seconds"], resumed=resumed,
                        this_invocation_seconds=total, recorded_at=now()), exclusive=True)
            print(f"Completed {args.case}. Result: {completed_path}", flush=True)
            return 0
    except Exception as exc:
        log = root / "workflow_errors.log"
        with log.open("a", encoding="utf8") as stream:
            stream.write(f"\n{now()} {args.case} {args.mode}\n")
            traceback.print_exc(file=stream)
        print(f"Cold-start run failed ({type(exc).__name__}). Error log: {log}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
