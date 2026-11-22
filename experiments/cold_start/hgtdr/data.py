"""Build the HGTDR comparison inputs from the sealed cold-disease corpus."""

from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock
import numpy as np
import polars as pl

from evi_hgt.competition.data import UnlabeledPolicy, export_graph_arrays, normalize_nodes
from evi_hgt.phase9 import hgtdr_adapter as hg
from evi_hgt_phase98.common import ROOT, read, sha, write_new


CASE = "cold_disease"
HEADS = ("support", "contra")
CLINICAL_RELATION = {"support": "INDICATION", "contra": "CONTRAINDICATION"}
TRAIN_CHANNEL = {"support": "support", "contra": "unsuitability"}
PAIR_COLUMNS = ["drug_id", "disease_id"]
PANEL_COLUMNS = PAIR_COLUMNS + ["role", "raw_relation"]


def checked_file(root, relative, expected):
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"Unconfined input path: {relative}")
    path = (root / part).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or sha(path) != expected:
        raise ValueError(f"Input differs from its sealed SHA-256: {relative}")
    return path


def read_case(corpus):
    """Read training inventory and recorded test hashes without opening test files."""
    manifest_path = corpus / "manifest.json"
    manifest = read(manifest_path)
    audit_name = manifest["cold_case_receipt"]
    audit = read(bound_path(corpus, manifest, audit_name))
    test_hashes = audit["test_files"]
    graph = audit["graph"]
    if (audit["case"] != CASE or set(test_hashes) != {f"test_{head}.parquet" for head in HEADS}
            or set(test_hashes) & set(manifest["files"])
            or graph["graph_mask"] != "STRICT_ENTITY_INCIDENCE_ALL_DOMAINS"
            or graph["heldout_facts_remaining"] or graph["forbidden_equivalent_facts_remaining"]):
        raise ValueError("The prepared corpus does not have the strict cold-disease mask")
    if manifest["training_seed"] != 42:
        raise ValueError("The recorded cold-disease protocol uses seed 42")
    binding = dict(case=CASE, training_seed=manifest["training_seed"],
                   manifest_sha256=sha(manifest_path),
                   preparation_audit_sha256=manifest["files"][audit_name],
                   test_panel_sha256=test_hashes)
    return manifest, audit, binding


def bound_path(corpus, manifest, relative):
    return checked_file(corpus, relative, manifest["files"][relative])


def read_pairs(corpus, manifest, key):
    data = pl.read_parquet(bound_path(corpus, manifest, manifest[key]), columns=PAIR_COLUMNS)
    return frozenset(data.iter_rows())


def validate_panel(frame, head, lookup, blocked, observed, training_diseases):
    if frame.is_empty() or frame.select(PAIR_COLUMNS).n_unique() != frame.height:
        raise ValueError(f"Empty panel or duplicate pair for {head}")
    if set(frame["role"]) != {"P", "U"}:
        raise ValueError(f"Both original P and U roles are required for {head}")
    positives = frame.filter(pl.col("role") == "P")
    unlabeled = frame.filter(pl.col("role") == "U")
    if set(positives["raw_relation"]) != {CLINICAL_RELATION[head]} or set(unlabeled["raw_relation"]) != {None}:
        raise ValueError(f"Clinical relation does not match the candidate role for {head}")
    pos_pairs = set(positives.select(PAIR_COLUMNS).iter_rows())
    u_pairs = set(unlabeled.select(PAIR_COLUMNS).iter_rows())
    if not pos_pairs <= observed or u_pairs & observed or not (pos_pairs | u_pairs) <= blocked:
        raise ValueError(f"Observed or role-free mask does not match the panel for {head}")
    if set(frame["disease_id"]) & training_diseases:
        raise ValueError(f"Training disease is present in the held-out {head} panel")
    if not set(frame["drug_id"]) <= lookup["drug"].keys() or not set(frame["disease_id"]) <= lookup["disease"].keys():
        raise ValueError(f"Unknown canonical candidate identity for {head}")
    return u_pairs


def graph_inputs(corpus, cache, nodes, manifest, audit, blocked):
    graph_cache = cache / 'hgtdr_graph'
    fact_paths = [bound_path(corpus, manifest, row["path"]) for row in manifest["fact_files"]]
    catalog = read(bound_path(corpus, manifest, manifest["edge_catalog"]))
    exporter = ROOT / "src/evi_hgt/competition/data.py"
    graph_binding = dict(nodes_sha256=manifest["files"][manifest["nodes"]],
                         catalog_sha256=manifest["files"][manifest["edge_catalog"]],
                         facts_sha256={row["path"]: manifest["files"][row["path"]]
                                       for row in manifest["fact_files"]},
                         blocked_sha256=manifest["files"][manifest["blocked_pairs"]],
                         exporter_sha256=sha(exporter))
    cache.mkdir(parents=True, exist_ok=True)
    receipt_path = graph_cache / "receipt.json"
    with FileLock(cache / "hgtdr_graph.lock", timeout=600):
        if not graph_cache.exists():
            with TemporaryDirectory(prefix="graph_stage_", dir=cache) as tmp:
                stage = Path(tmp).resolve()
                if not (stage.is_relative_to(cache.resolve())
                        and graph_cache.resolve().is_relative_to(cache.resolve())):
                    raise ValueError("Graph cache staging path escaped the comparison")
                relations, construction = export_graph_arrays(nodes, fact_paths, catalog, stage,
                                                               forbidden_pairs=blocked)
                if construction["canonical_facts"] != audit["graph"]["canonical_facts"]:
                    raise ValueError("Native graph exported a different set of canonical facts")
                relations.write_parquet(stage / "relations.parquet")
                artifacts = {"relations.parquet": sha(stage / "relations.parquet")}
                for entry in relations.iter_rows(named=True):
                    for key in ("edge_index_path", "fact_ids_path"):
                        name = entry[key]
                        artifacts[name] = sha(stage / name)
                write_new(stage / receipt_path.name,
                          dict(binding=graph_binding, construction=construction, artifacts=artifacts))
                stage.rename(graph_cache)
        receipt = read(receipt_path)
        if receipt["binding"] != graph_binding or receipt["construction"]["canonical_facts"] != audit["graph"]["canonical_facts"]:
            raise ValueError("Cached native graph does not bind the original cold-disease facts")
        verified = {relative: checked_file(graph_cache, relative, expected)
                    for relative, expected in receipt["artifacts"].items()}
    return pl.read_parquet(verified["relations.parquet"]).to_dicts(), verified, receipt


class ColdHGTDRData(hg.HGTDRFoldData):
    """The existing native HGTDR batching with the cold split's eligible U domain."""

    def unlabeled(self, count, seed):
        rows = self.u_policy.sample(count, seed=seed)
        self.u_policy.validate(rows)
        return self.encode_pairs(rows)

    def encode_pairs(self, rows):
        return np.asarray([[self.lookup["drug"][drug], self.lookup["disease"][disease]]
                           for drug, disease in rows], dtype=np.int64).reshape(-1, 2)


def prepare_training(corpus, cache):
    """Return a copyable data template, dev-only candidate rows and an identity receipt."""
    manifest, audit, binding = read_case(corpus)
    native = ColdHGTDRData()
    native.training_seed = binding["training_seed"]
    native.nodes = normalize_nodes(pl.read_parquet(bound_path(corpus, manifest, manifest["nodes"])))
    native.features, native.available, native.counts, native.feature_binding = hg._read_features(native.nodes)
    d1_metadata = read(bound_path(corpus, manifest, manifest["pretrained"]["metadata"]))
    if d1_metadata["original_feature_manifest_sha256"] != native.feature_binding["feature_manifest_sha256"]:
        raise ValueError("HGTDR and the prepared EviHGT corpus must use the same frozen D1 feature source")
    native.lookup = {kind: dict(native.nodes.filter(pl.col("human_node_type") == kind)
                                .select("node_id", "type_local_id").iter_rows())
                     for kind in ("drug", "disease")}
    eligibility = read(bound_path(corpus, manifest, manifest["eligibility"]))
    training_drugs = set(eligibility["eligible_drugs"])
    training_diseases = set(eligibility["eligible_diseases"])
    if not training_drugs <= native.lookup["drug"].keys() or not training_diseases <= native.lookup["disease"].keys():
        raise ValueError("Training eligibility differs from canonical node identity")
    blocked, observed = (read_pairs(corpus, manifest, key) for key in ("blocked_pairs", "observed_pairs"))
    native.u_policy = UnlabeledPolicy(tuple(eligibility["eligible_drugs"]), tuple(eligibility["eligible_diseases"]),
                                     observed, blocked)

    training = pl.read_parquet(bound_path(corpus, manifest, manifest["training_pairs"]))
    if set(training["channel"]) != set(TRAIN_CHANNEL.values()):
        raise ValueError("Unexpected cold-disease training channel")
    native.training_positive_ids, native.training_positives = {}, {}
    for head, channel in TRAIN_CHANNEL.items():
        frame = training.filter(pl.col("channel") == channel).sort(PAIR_COLUMNS)
        pairs = tuple(frame.select(PAIR_COLUMNS).iter_rows())
        if (len(pairs) != audit["retained_P"]["train"][head] or len(set(pairs)) != len(pairs)
                or set(frame["raw_relation"]) != {CLINICAL_RELATION[head]}
                or set(pairs) & blocked or not set(pairs) <= observed
                or not set(frame["drug_id"]) <= training_drugs or not set(frame["disease_id"]) <= training_diseases):
            raise ValueError(f"Training positives violate frozen cold membership: {head}")
        native.training_positive_ids[head] = pairs
        native.training_positives[head] = native.encode_pairs(pairs)
    native.support_ids = native.training_positive_ids["support"]
    native.supports = native.training_positives["support"]

    manifests = manifest["panels"]
    support = pl.read_parquet(bound_path(corpus, manifest, manifests["validation"]), columns=PANEL_COLUMNS)
    auxiliary = pl.read_parquet(bound_path(corpus, manifest, manifests["auxiliary_validation"]), columns=PANEL_COLUMNS)
    validation = {"support": support, "contra": pl.concat((auxiliary, support.filter(pl.col("role") == "U")))}
    for head, rows in validation.items():
        common_u = validate_panel(rows, head, native.lookup, blocked, observed, training_diseases)
        if len(common_u) != audit["dev_U"] or rows.height - len(common_u) != audit["retained_P"]["validation"][head]:
            raise ValueError(f"Development candidate count differs from the original {head} run")
    native.relations, artifacts, native.graph_receipt = graph_inputs(corpus, cache, native.nodes, manifest, audit, blocked)
    arrays = [np.load(artifacts[row["edge_index_path"]], mmap_mode="r", allow_pickle=False) for row in native.relations]
    native.sampler = hg.native_module().QueryMaskedSampler(native.counts, native.relations, arrays)
    native.metadata = (list(native.counts), native.sampler.edge_types)
    binding.update(graph_receipt_sha256=sha(cache / "hgtdr_graph" / "receipt.json"), feature_binding=native.feature_binding,
                   development_panel_sha256={name: manifest["files"][path] for name, path in manifests.items()},
                   native_sources=hg.verify_native_identity(), data_adapter_sha256=sha(Path(__file__)),
                   u_policy="NATIVE_GLOBAL_WITH_REPLACEMENT_ON_FROZEN_COLD_TRAINING_DOMAIN",
                   training_positive_counts={head: len(ids) for head, ids in native.training_positive_ids.items()})
    native.bindings = dict(binding)
    return native, {head: rows.sort(PAIR_COLUMNS).to_dicts() for head, rows in validation.items()}, binding


def load_test(corpus, binding):
    """Read the original independent test panel only after development selection."""
    manifest, audit, current = read_case(corpus)
    if any(binding.get(key) != value for key, value in current.items()):
        raise ValueError("Selected training binding differs from the sealed cold-disease test")
    nodes = normalize_nodes(pl.read_parquet(bound_path(corpus, manifest, manifest["nodes"])))
    lookup = {kind: dict(nodes.filter(pl.col("human_node_type") == kind)
                         .select("node_id", "type_local_id").iter_rows())
              for kind in ("drug", "disease")}
    eligibility = read(bound_path(corpus, manifest, manifest["eligibility"]))
    training_diseases = set(eligibility["eligible_diseases"])
    dev_path = bound_path(corpus, manifest, manifest["panels"]["validation"])
    dev_diseases = set(pl.read_parquet(dev_path, columns=["disease_id"])["disease_id"])
    auxiliary_path = bound_path(corpus, manifest, manifest["panels"]["auxiliary_validation"])
    dev_diseases.update(pl.read_parquet(auxiliary_path, columns=["disease_id"])["disease_id"])
    blocked, observed = (read_pairs(corpus, manifest, key) for key in ("blocked_pairs", "observed_pairs"))
    rows, u_sets = {}, {}
    for head in HEADS:
        name = f"test_{head}.parquet"
        path = checked_file(corpus, name, current["test_panel_sha256"][name])
        frame = pl.read_parquet(path, columns=PANEL_COLUMNS)
        u_sets[head] = validate_panel(frame, head, lookup, blocked, observed, training_diseases | dev_diseases)
        if len(u_sets[head]) != audit["test_U"] or frame.height - len(u_sets[head]) != audit["retained_P"]["test"][head]:
            raise ValueError(f"Independent test candidate count differs for {head}")
        rows[head] = frame.sort(PAIR_COLUMNS).to_dicts()
    if u_sets["support"] != u_sets["contra"]:
        raise ValueError("Both heads must use the same test U candidates")
    return rows
