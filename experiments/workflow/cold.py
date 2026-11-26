"""Prepare, train and score the four portable cold-start experiments."""

from dataclasses import replace
from importlib import import_module
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from uuid import uuid4

import polars as pl

from evi_hgt.data.common import digest, sha256

from experiments.cold_start.approval_cohort import APPROVAL_SOURCE, approval_partitions
from .cold_graph import build_cold_graph, pair_frame


COLD_CASES = ("cold_drug", "cold_disease", "double_cold", "first_approval_year_cohort")
TEST_SCOPE = "SYNTHETIC_TEST_ONLY"
POOL_COLUMNS = ("drug_id", "disease_id")
RECIPE = "EVIHGT_PORTABLE_COLD_FROM_RAW_V1"


def archived_scripts():
    """Load the archived split and score scripts with their original sibling imports."""
    path = str(Path(__file__).resolve().parents[1] / "cold_start")
    sys.path.insert(0, path)
    try:
        prep = import_module("experiments.cold_start.prepare_case")
        score = import_module("experiments.cold_start.run_case")
    finally:
        sys.path.remove(path)
    return prep, score


def approval_partitions_from_aggregate(source, eligible_drugs, node_drugs, work):
    """Read only the checked approval-year property from OptimusKG aggregate nodes."""
    source = Path(source).resolve()
    original_hash = sha256(source)
    frame = (pl.scan_parquet(source).filter(pl.col("label") == "DRG")
             .select("id", "properties").collect())
    if frame.schema != {"id": pl.String, "properties": pl.String}:
        raise ValueError("Unexpected OptimusKG aggregate approval-year schema")
    if frame["id"].n_unique() != frame.height:
        raise ValueError("Duplicate OptimusKG drug IDs in aggregate nodes")
    values = []
    for drug, raw in frame.iter_rows():
        props = json.loads(raw) if raw is not None else {}
        if not isinstance(props, dict):
            raise ValueError("OptimusKG drug properties must be a JSON object")
        year = props.get("year_of_first_approval")
        if year is not None and (type(year) is not int or not 1 <= year <= 9999):
            raise ValueError(f"Invalid OptimusKG approval year for canonical drug {drug}")
        values.append((drug, year))
    typed = pl.DataFrame(values, schema={"id": pl.String, "year_of_first_approval": pl.Int64},
                         orient="row").select("id", pl.struct("year_of_first_approval").alias("properties"))
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="approval_", dir=work, ignore_cleanup_errors=False) as temporary:
        projection = Path(temporary) / APPROVAL_SOURCE
        projection.parent.mkdir(parents=True)
        typed.write_parquet(projection)
        assigned, audit = approval_partitions(Path(temporary), eligible_drugs, node_drugs)
    if sha256(source) != original_hash:
        raise ValueError("OptimusKG aggregate nodes changed during extraction")
    audit.update(source=str(source), source_sha256=original_hash,
                 source_field="properties.year_of_first_approval", source_row_filter="label == DRG",
                 source_layout="OPTIMUSKG_2_0_AGGREGATE_NODES_JSON_PROPERTIES",
                 projection_sha256=audit["source_sha256"])
    return assigned, audit


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_test_files(score, manifest):
    content = _read_json(manifest)
    audit = score._corp_input(manifest, content["cold_case_receipt"], content["files"])
    receipt = _read_json(audit)
    for name in score.PANEL_NAMES.values():
        candidate = manifest.parent / name
        expected = receipt["test_files"][name]
        if not candidate.is_file() or not candidate.resolve().is_relative_to(manifest.parent) or sha256(candidate) != expected:
            raise ValueError(f"Sealed cold test panel changed: {candidate}")
    return audit


def _preflight(score, case, manifest):
    _verify_test_files(score, manifest)
    return score._preflight(case, manifest)


def _portable_runtime(prep):
    root = Path(__file__).resolve().parents[2]
    if prep.FROZEN != root:
        raise ValueError("The portable cold workflow uses the registered runtime in this repository")
    return prep.check_frozen()


def _adapter_binding():
    here = Path(__file__).resolve().parent
    archive = here.parent / "cold_start"
    paths = [here / "cold.py", here / "cold_graph.py", *(archive / name for name in
        ("prepare_case.py", "approval_cohort.py", "graph_builder.py", "run_case.py", "locations.py"))]
    return {str(p.relative_to(here.parent)): sha256(p) for p in paths}


def _pool_sets(pools):
    support = {(r["drug_id"], r["disease_id"]) for r in pools["panel"] if r["role"] == "P"}
    contra, protected, observed = (set(pools[k]) for k in ("auxiliary", "frozen", "observed"))
    drugs, diseases = set(pools["eligible_drugs"]), set(pools["eligible_diseases"])
    legal = {(a, b) for a, b in observed - protected - set(pools["historical"]) if a in drugs and b in diseases}
    relations = pools["source"].select(*POOL_COLUMNS, "raw_relation")
    labels = {(a, b): rel for a, b, rel in relations.iter_rows()}
    if len(labels) != relations.height or set(labels) != observed:
        raise ValueError("Canonical source does not match the observed-pair pool")
    if (support & contra or (support | contra) - legal
            or any(labels[pair] != "INDICATION" for pair in support)
            or any(labels[pair] != "CONTRAINDICATION" for pair in contra)):
        raise ValueError("Development pools do not match the verified relation and protection rules")
    return support, contra, protected, observed, legal, drugs, diseases


def _case_result(case, destination, *, runtime=None):
    return {
        "case": case, "manifest": destination / "manifest.json",
        "preparation_audit": destination / "preparation_audit.json",
        "test_panels": {head: destination / f"test_{head}.parquet" for head in ("support", "contra")},
        "runtime": runtime,
    }


def prepare_cold_case(case, shared, pools, destination, *, audit_dir, progress=None):
    """Build one case with score-blind splits and test roles outside the trainer inventory."""
    prep, score = archived_scripts()
    runtime = _portable_runtime(prep)
    if case not in COLD_CASES:
        raise ValueError(f"Unsupported cold case: {case}")
    destination = Path(destination).resolve()
    canonical = Path(shared["canonical_root"]).resolve(strict=True)
    raw_nodes = Path(shared["nodes"]).resolve(strict=True)
    bound_hashes = {Path(p).resolve(): h for p, h in shared["file_hashes"].items()}
    source_identity = shared["source_identity"]
    synthetic = source_identity.get("purpose") == TEST_SCOPE
    if any(p.is_relative_to(destination) for p in bound_hashes):
        raise ValueError("Cold corpus output would contain one of its immutable inputs")
    source_used = {}

    def verified(path):
        path = Path(path).resolve(strict=True)
        expected = bound_hashes.get(path)
        if expected is None:
            raise ValueError(f"Cold input is absent from the verified preparation receipts: {path}")
        if path not in source_used and sha256(path) != expected:
            raise ValueError(f"Cold input has changed since raw preparation: {path}")
        source_used[path] = expected
        return path

    support, contra, protected, observed, legal, drugs, diseases = _pool_sets(pools)
    input_identity = digest(dict(schema=RECIPE, raw=source_identity, runtime=runtime, adapter=_adapter_binding(),
        source_artifact_hashes=sorted(bound_hashes.values()),
        pools={key: digest(sorted(value)) for key, value in (
            ("support", support), ("contra", contra), ("protected", protected), ("observed", observed),
            ("legal", legal), ("eligible_drugs", drugs), ("eligible_diseases", diseases))}))
    result = _case_result(case, destination, runtime=runtime)
    audit_path = Path(audit_dir) / f"{case}_preparation.json"
    if result["manifest"].is_file():
        receipt = _read_json(result["preparation_audit"])
        if receipt.get("input_identity") != input_identity:
            raise ValueError("Cold preparation inputs changed. Use a new case directory.")
        _preflight(score, case, result["manifest"])
        for name, expected in _read_json(result["manifest"])["files"].items():
            score._corp_input(result["manifest"], name, {name: expected})
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        prep.write_json(audit_path, receipt)
        return result
    if destination.exists():
        raise FileExistsError(destination)
    if progress:
        progress(f"准备冷启动划分 {case}", 0, 0, 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f"{destination.name}.attempt_{uuid4().hex}")
    stage.mkdir()

    nodes = pl.read_parquet(verified(canonical / "nodes/node_index.parquet")).sort("global_node_id")
    node_drugs = set(nodes.filter(node_type="DRG")["node_id"])
    node_diseases = set(nodes.filter(node_type="DIS")["node_id"])
    if not drugs <= node_drugs or not diseases <= node_diseases:
        raise ValueError("Cold candidate universe contains noncanonical endpoints")
    drug_part, disease_part = {}, {}
    split_audit = dict(protocol="EVIHGT_FROZEN_DEVELOPMENT_COLD_V1", seed=prep.SEED,
                       train_fraction_nominal=.8, stable_sha256_entity_grouping=True,
                       existing_protected_pair_roles_read=False)
    if case == "first_approval_year_cohort":
        drug_part, split_audit = approval_partitions_from_aggregate(verified(raw_nodes), drugs, node_drugs, stage)
    if case in ("cold_drug", "double_cold"):
        drug_part = {drug: prep.entity_partition(drug, "drug_id") for drug in node_drugs}
    if case in ("cold_disease", "double_cold"):
        disease_part = {disease: prep.entity_partition(disease, "disease_id") for disease in node_diseases}

    def matches(pair, partition):
        return ((not drug_part or drug_part[pair[0]] == partition)
                and (not disease_part or disease_part[pair[1]] == partition))
    drug_domain = {p: {d for d in drugs if not drug_part or drug_part[d] == p} for p in prep.PARTS}
    disease_domain = {p: {d for d in diseases if not disease_part or disease_part[d] == p} for p in prep.PARTS}
    positives = {p: tuple({pair for pair in pool if matches(pair, p)} for pool in (support, contra))
                 for p in prep.PARTS}
    if any(not a or not b for a, b in positives.values()):
        raise ValueError(f"{case} requires recorded support and contra positives in its three partitions")
    dev_support, dev_contra, dev_u = prep.make_panels(
        *positives["validation"], drug_domain["validation"], observed | protected, "validation")
    test_support, test_contra, test_u = prep.make_panels(
        *positives["test"], drug_domain["test"], observed | protected | dev_u, "test")
    if not dev_u or not test_u or dev_u & test_u:
        raise ValueError(f"{case} requires disjoint development and test U panels")
    blocked = protected | ((support | contra) - set.union(*positives["train"])) | dev_u | test_u
    G, groups, conditional = prep.make_schedule(*positives["train"], drug_domain["train"],
                                                 disease_domain["train"], observed | blocked, case)
    held_drugs = {k for k, value in drug_part.items() if value != "train"}
    held_diseases = {k for k, value in disease_part.items() if value != "train"}
    if progress:
        progress(f"物化冷启动隔离图 {case}", 0, 0, 0)

    files = {}

    def write(name, writer):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)
        files[name] = sha256(path)
        return name

    def write_json(name, value):
        return write(name, lambda path: prep.write_json(path, value))

    clean = shared["clean_dg"]
    graph, graph_receipt = build_cold_graph(canonical, clean["facts"], clean["evidence"], nodes, stage,
        write, verified, held_drugs, held_diseases, blocked, legal)
    feature_manifest = {}
    for section in ("raw_features", "pretrained"):
        feature_manifest[section] = {}
        for key, path in shared["features"][section].items():
            source = verified(path)
            suffix = ".json" if key == "metadata" else ".npy"
            name = write(f"{section}/{key}{suffix}", lambda output: shutil.copyfile(source, output))
            if files[name] != source_used[source]:
                raise ValueError(f"Copied feature differs from the certified source: {source}")
            feature_manifest[section][key] = name
    train = [(drug, disease, channel, relation)
             for channel, relation, pool in zip(("support", "unsuitability"),
                      ("INDICATION", "CONTRAINDICATION"), positives["train"], strict=True)
             for drug, disease in sorted(pool)]
    training = pl.DataFrame(train, schema={**prep.PAIR_SCHEMA, "channel": pl.String, "raw_relation": pl.String},
                            orient="row")
    test_files = {}
    for head, panel in (("support", test_support), ("contra", test_contra)):
        filename = f"test_{head}.parquet"
        panel.write_parquet(stage / filename)
        test_files[filename] = sha256(stage / filename)
    receipt = dict(case=case, split=split_audit, graph=graph_receipt, train_updates=3 * G,
        graph_drugs_excluded=len(held_drugs), graph_diseases_excluded=len(held_diseases),
        retained_P={p: {"support": len(a), "contra": len(b)} for p, (a, b) in positives.items()},
        dev_U=len(dev_u), test_U=len(test_u), test_files=test_files, input_identity=input_identity,
        test_labels_used_by_optimizer=False, original_protected_membership_roles_read=False,
        source_sha256={str(path): value for path, value in source_used.items()}, runtime=runtime)
    if any(sha256(path) != bound for path, bound in source_used.items()):
        raise ValueError("A bound cold preparation source changed during use")
    semantics = {name: True for name in (
        "canonical_entities_verified", "canonical_facts_deduplicated", "evidence_records_deduplicated",
        "clinical_labels_verified", "target_aliases_resolved", "disease_gene_label_leakage_removed",
        "split_mask_applied", "features_label_independent")}
    semantics["unknown_is_negative"] = False
    manifest = dict(schema="EVIHGT_PORTABLE_CORPUS_V1", preparation=source_identity,
        purpose=TEST_SCOPE if synthetic else RECIPE, training_seed=prep.SEED, token_buckets=8192,
        **graph, **feature_manifest, semantics=semantics,
        blocked_pairs=write("blocked_pairs.parquet", pair_frame(blocked).write_parquet),
        observed_pairs=write("observed_pairs.parquet", pair_frame(observed).write_parquet),
        eligibility=write_json("eligibility.json", dict(eligible_drugs=sorted(drug_domain["train"]),
                                                         eligible_diseases=sorted(disease_domain["train"]))),
        training_pairs=write("training_pairs.parquet", training.write_parquet),
        panels=dict(validation=write("validation.parquet", dev_support.write_parquet),
                    auxiliary_validation=write("auxiliary_validation.parquet",
                                               dev_contra.filter(role="P").write_parquet)),
        schedule=dict(G=G, groups=write("global_groups.parquet", groups.write_parquet),
                      conditional_records=write_json("conditional_records.json", conditional),
                      sampling_provenance=dict(namespace="COLD_START_3G_V1", case=case, seed=prep.SEED,
                                               full_positive_coverage=True)),
        cold_case_receipt=write_json("preparation_audit.json", receipt), files=files)
    prep.write_json(stage / "manifest.json", manifest)
    _preflight(score, case, stage / "manifest.json")
    stage.rename(destination)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    prep.write_json(audit_path, receipt)
    return result


def _finish_training(runner, score, settings, root, membership, checkpoint):
    run_root = root / "frozen_runs"
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve(strict=True)
        if not checkpoint.is_relative_to(run_root) or checkpoint.name != "latest.pt":
            raise ValueError("Resume only from a case-local latest checkpoint")
    else:
        # A failed invocation may leave an optimizer checkpoint before its report.
        candidates = sorted(run_root.glob("*/checkpoints/latest.pt"))
        checkpoint = candidates[-1] if candidates else None
    if checkpoint is not None:
        previous = checkpoint.parent.parent
        if (previous / "completion.json").is_file():
            return score._checkpoint_receipt(previous, root, membership)
    run = runner.run(settings, variant="FULL", mode="train", checkpoint=checkpoint).resolve()
    target = run / "checkpoints/latest.pt"
    if not target.is_file() and checkpoint is not None:
        # A report-only continuation already starts at the final optimizer step.
        # Preserve its existing in-case checkpoint and bind it to this completion.
        completed = _read_json(run / "completion.json")
        if (completed["updates"] != membership["expected_updates"]
                or completed["resumed_from_step"] != completed["updates"]):
            raise ValueError("The cold continuation did not commit its final checkpoint")
        return dict(completed, root=str(run), checkpoint=str(checkpoint),
                    checkpoint_sha256=sha256(checkpoint), result_sha256=sha256(run / "completion.json"),
                    recorded_at=score.now())
    return score._checkpoint_receipt(run, root, membership)


def run_cold(settings, case_info, *, checkpoint=None, smoke=False):
    """Run fixed FULL then score the independently sealed cold test panels."""
    from evi_hgt import runner

    prep, score = archived_scripts()
    runtime = _portable_runtime(prep)
    case, manifest = case_info["case"], Path(case_info["manifest"]).resolve(strict=True)
    content = _read_json(manifest)
    if smoke != (content.get("purpose") == TEST_SCOPE):
        raise ValueError("The run mode and cold corpus purpose differ")
    if smoke and (content["schedule"]["G"] != 1 or settings.device != "cpu" or settings.precision != "fp32"):
        raise ValueError("SYNTHETIC_TEST_ONLY cold runs require CPU/FP32 and three updates")
    root = Path(case_info.get("run_dir") or settings.resolve(settings.output_dir) / case).resolve()
    if root.is_relative_to(manifest.parent) or manifest.is_relative_to(root):
        raise ValueError("Choose separate cold training and corpus directories")
    root.mkdir(parents=True, exist_ok=True)
    run_settings = replace(settings, manifest=str(manifest), output_dir=str(root / "frozen_runs"))
    training_path, result_path, first_path = (root / name for name in
        ("training_complete.json", "result.json", "first_invocation.json"))
    with score._case_lock(root):
        membership = _preflight(score, case, manifest)
        invocation = dict(membership=membership, runtime=runtime, adapter=_adapter_binding(),
                          settings=run_settings.as_dict(), variant="FULL")
        if first_path.exists():
            if _read_json(first_path)["binding"] != invocation:
                raise ValueError("Cold runtime, settings or test panel binding changed.")
        else:
            score._write_json(first_path, dict(started_at=score.now(), binding=invocation), exclusive=True)
        if result_path.exists():
            result = _read_json(result_path)
            metric_path = root / "test/metrics.json"
            if result["metric_sha256"] != sha256(metric_path):
                raise ValueError("Completed cold-test metrics hash changed")
            metrics = _read_json(metric_path)
            if metrics["membership"] != membership or any(sha256(root / "test" / name) != expected
                    for name, expected in metrics["artifact_sha256"].items()):
                raise ValueError("Completed cold test candidate or score artifacts changed")
            return root
        if training_path.exists():
            training = _read_json(training_path)
        else:
            training = _finish_training(runner, score, run_settings, root, membership, checkpoint)
            score._write_json(training_path, training, exclusive=True)
        score_runner = SimpleNamespace(Console=score._localized_console(runner.Console), **{
            name: getattr(runner, name) for name in (
                "load_inputs", "build_model", "make_unlabeled_query_supervision", "precision_context"
            )})
        metrics = score._score(case, score_runner, run_settings, root, membership, training)
        metrics["purpose"] = content["purpose"]
        score._write_json(root / "test/metrics.json", metrics)
        score._write_json(result_path, dict(status="COMPLETE", case=case, metric_file="test/metrics.json",
            metric_sha256=sha256(root / "test/metrics.json"), metrics=metrics["metrics"],
            purpose=content["purpose"], runtime=runtime, recorded_at=score.now()), exclusive=True)
    return root
