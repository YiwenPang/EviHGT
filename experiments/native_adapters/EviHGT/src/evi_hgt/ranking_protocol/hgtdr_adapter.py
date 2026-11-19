"""Single-LR HGTDR native adapter with training-only, fold-local graph wiring.

No historical CommonDataset/run/score_panel is called: those open pilot roles.
This revision permits isolated TEST_ONLY work; G3 remains closed.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import polars as pl
import torch

from .identity import ROOT, G1_SHA256, CellContract, frozen_science, canonical, digest, file_sha

HGTDR_ROOT = ROOT.parent / "HGTDR"
FEATURE_ROOT = HGTDR_ROOT / "data/processed/competition_pilot_v1/features"
TEST_ROOT = ROOT / "outputs/phase9/phase92b/test_only"
_NATIVE = None


def verify_native_identity():
    old = json.loads((ROOT / "outputs/phase9/phase92/G2_TEST_RESULTS.json").read_text(encoding="utf-8"))
    rows = [r for r in old["baseline_binding_verification"] if str(HGTDR_ROOT) in r["path"]]
    if len(rows) != 7:
        raise ValueError("Seven bound HGTDR code/feature identities required")
    result = []
    for row in rows:
        actual = file_sha(row["path"])
        if actual != row["expected"]:
            raise ValueError("HGTDR certified source/feature manifest changed: " + row["path"])
        result.append({"path": row["path"], "sha256": actual, "match": True})
    return result


def native_module():
    global _NATIVE
    verify_native_identity()
    if _NATIVE is None:
        # Native imports are src.runtime/src.vendor, not evi_hgt. Never replace an
        # already-loaded src package belonging to another project.
        existing = sys.modules.get("src")
        if existing is not None and getattr(existing, "__file__", None):
            if HGTDR_ROOT not in Path(existing.__file__).resolve().parents:
                raise ImportError("Conflicting native src namespace; use an independent process")
        sys.path.insert(0, str(HGTDR_ROOT))
        spec = importlib.util.spec_from_file_location("_phase92b_native_hgtdr", HGTDR_ROOT / "src/competition.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NATIVE = module
    return _NATIVE


def fixed_config(seed):
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Original cell seed required")
    return dict(seed=seed, batch_size=4096, fanout=[512, 512, 512], epochs=300,
                learning_rates=[.003], max_nodes=20000, max_edges=1500000,
                cpu_threads=4, evaluation_frequency=1,
                checkpoint_selection="INNER_AUC_THEN_AP_THEN_EARLIER_EPOCH")


def require_formal_execution(execution_manifest=None, *, repeat=None, fold=None):
    # No token is issued in this preparation task. A hand-made JSON, environment
    # variable or test flag must never turn the current authorization into G3.
    if repeat is None or fold is None:
        raise PermissionError("G3_NOT_AUTHORIZED: an independently issued, cell-bound permit is required")
    try:
        from .execution import verify_permit
    except ImportError as exc:
        raise PermissionError("G3_NOT_AUTHORIZED: execution verifier unavailable") from exc
    return verify_permit(execution_manifest, "M1", repeat, fold)


class InnerCheckpointSelection:
    def __init__(self):
        self.best = None
        self.opportunities = 0
        self.history = []

    def observe(self, epoch, metrics):
        if type(epoch) is not int or epoch != self.opportunities + 1 or epoch > 300:
            raise ValueError("Exactly sequential 300 inner selection opportunities required")
        auc, ap = float(metrics["macro_auc"]), float(metrics["macro_ap"])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (auc, ap)):
            raise ValueError("Inner metrics must be finite and within [0,1]")
        row = dict(epoch=epoch, metrics=dict(macro_auc=auc, macro_ap=ap))
        key = lambda v: (v["metrics"]["macro_auc"], v["metrics"]["macro_ap"], -v["epoch"])
        if self.best is None or key(row) > key(self.best):
            self.best = row
        self.opportunities = epoch
        self.history.append(copy.deepcopy(row))


def _test_destination(path):
    path = Path(path).resolve()
    if TEST_ROOT.resolve() not in path.parents:
        raise PermissionError("This phase writes HGTDR derived/test data only below phase92b/test_only")
    return path


def _read_features(nodes):
    """Use cached embeddings only; never walk input_manifest raw-source paths."""
    manifest_path = FEATURE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent = FEATURE_ROOT.parent
    if manifest["status"] != "COMPLETE" or manifest.get("optimizer_steps") != 0 or manifest.get("training") is not False:
        raise ValueError("Frozen inference features required")
    if file_sha(parent / "input_manifest.json") != manifest["input_manifest_sha256"]:
        raise ValueError("Feature provenance receipt changed")
    source = json.loads((parent / "input_manifest.json").read_text(encoding="utf-8"))
    if source.get("clinical_text_encoded") is not False:
        raise ValueError("Unexpected feature target-text boundary")
    if file_sha(parent / "feature_inputs.parquet") != manifest["feature_inputs_sha256"]:
        raise ValueError("Feature input identity changed")
    inputs = pl.read_parquet(parent / "feature_inputs.parquet").sort("global_node_id")
    for col in ("global_node_id", "type_local_id", "node_type", "node_id"):
        if inputs[col].to_list() != nodes[col].to_list():
            raise ValueError("Feature/canonical node row mismatch: " + col)
    features, available, counts, coverage = {}, {}, {}, []
    records = {r["node_type"]: r for r in manifest["types"]}
    for group in nodes.partition_by("node_type", maintain_order=True):
        group = group.sort("type_local_id")
        kind, abbreviation = group["human_node_type"][0], group["node_type"][0]
        r = records[abbreviation]
        path, mask_path = FEATURE_ROOT / f"type_{r['node_type_id']:02d}.npy", FEATURE_ROOT / f"type_{r['node_type_id']:02d}_available.npy"
        if file_sha(path) != r["sha256"] or file_sha(mask_path) != r["mask_sha256"]:
            raise ValueError("Cached feature/availability hash mismatch")
        values, mask = np.load(path, allow_pickle=False), np.load(mask_path, allow_pickle=False)
        if values.shape != (group.height, 768) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Invalid 768-dimensional native features")
        if mask.shape != (group.height,) or mask.dtype != np.bool_ or np.any(values[~mask]):
            raise ValueError("Invalid missing feature values/masks")
        expected = inputs.filter(pl.col("node_type") == abbreviation).sort("type_local_id")
        if not np.array_equal(mask, expected["available"].to_numpy()):
            raise ValueError("Feature mask/source input mismatch")
        h = hashlib.sha256()
        for entity, text, present in expected.select("node_id", "encoder_text", "available").iter_rows():
            h.update(json.dumps([entity, text if present else None, bool(present)], ensure_ascii=False).encode("utf-8"))
            h.update(b"\n")
        if h.hexdigest() != r["input_sha256"]:
            raise ValueError("Feature source-row digest mismatch")
        features[kind], available[kind], counts[kind] = torch.from_numpy(values), torch.from_numpy(mask), group.height
        coverage.append(dict(node_type=kind, rows=group.height, available=int(mask.sum()), missing=int((~mask).sum()), encoder=r["encoder"]))
    return features, available, counts, dict(feature_manifest_sha256=file_sha(manifest_path),
        feature_inputs_sha256=manifest["feature_inputs_sha256"], input_manifest_sha256=manifest["input_manifest_sha256"],
        clinical_text_encoded=False, coverage=coverage, fixed_encoder_specs=manifest.get("fixed_encoder_specs"),
        upstream_target_exposure="NOT_VERIFIED", raw_source_paths_followed=False)


def native_batch(data, pairs, config, seed, device):
    sample = data.sampler.sample(pairs, config["fanout"], seed=seed,
                                max_nodes=config["max_nodes"], max_edges=config["max_edges"])
    sample["features"] = {k: data.features[k][ids].to(device) for k, ids in sample["nodes"].items()}
    sample["available"] = {k: data.available[k][ids].to(device) for k, ids in sample["nodes"].items()}
    sample["edges"] = {k: v.to(device) for k, v in sample["edges"].items()}
    sample["pair_index"] = sample["pair_index"].to(device)
    return sample


def validate_cell_corpus(data, corpus):
    if any(corpus.get("bindings", {}).get(k) != v for k, v in data.bindings.items()):
        raise ValueError("Native corpus belongs to a different cell/data binding")
    contract = CellContract.load("M0", data.bindings["original_repeat_index"], data.bindings["fold_index"],
                                 seed=data.training_seed)
    if data.bindings["fold_manifest_digest"] != contract.bindings["cell_digest"]:
        raise ValueError("Native corpus current-cell member identity differs from G1")
    if len(data.support) != contract.budget["full_training_support_P"]:
        raise ValueError("Native corpus training support membership/count differs")
    return contract


class HGTDRFoldData:
    @classmethod
    def from_cell(cls, data, corpus, output, *, permit=None):
        from evi_hgt.competition.data import export_graph_arrays, normalize_nodes, UnlabeledPolicy
        validate_cell_corpus(data, corpus)
        verify_native_identity()
        if permit is None:
            output = _test_destination(output)
        else:
            current = require_formal_execution(permit, repeat=data.repeat_index, fold=data.cell["fold"])
            output = Path(output).resolve()
            if not output.is_relative_to(Path(current["output_root"]).resolve()):
                raise PermissionError("Native formal data output is outside approved execution root")
        output.mkdir(parents=True, exist_ok=True)
        obj = cls()
        obj.training_seed, obj.bindings = data.training_seed, copy.deepcopy(data.bindings)
        obj.cell_source = data
        nodes = normalize_nodes(pl.read_parquet(ROOT / "data/processed/phase3/v1/nodes/node_index.parquet"))
        obj.nodes = nodes
        obj.features, obj.available, obj.counts, obj.feature_binding = _read_features(nodes)
        obj.lookup = {kind: dict(nodes.filter(pl.col("human_node_type") == kind).select("node_id", "type_local_id").iter_rows())
                      for kind in ("drug", "disease")}
        obj.support_ids = tuple(sorted(data.support))
        if set(obj.support_ids) & set(data.blocked):
            raise ValueError("Protected support pair reached native training")
        for pair in obj.support_ids:
            data.pairs.assert_optimizer_pair(pair, "P")
        obj.supports = np.asarray([[obj.lookup["drug"][p[0]], obj.lookup["disease"][p[1]]] for p in obj.support_ids], dtype=np.int64)
        universe, gate = data.pairs.universe, data.pairs.gate
        obj.u_policy = UnlabeledPolicy(tuple(universe.eligible_drugs), tuple(universe.eligible_diseases),
            frozenset(universe.observed_pairs), frozenset(set(gate.formal) | set(gate.blocked) | set(gate.positive)))
        catalog_path = ROOT / "data/processed/phase3/v1/graph/csr/edge_type_catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        fact_paths = [Path(r["path"]) for r in corpus["fact_files"]]
        certified = dict(corpus["inputs"], **corpus["outputs"])
        current_sources = {str(p): file_sha(p) for p in fact_paths}
        if any(certified.get(p) != sha for p, sha in current_sources.items()):
            raise ValueError("Native fact payload differs from the certified current-fold corpus")
        if certified.get(str(catalog_path)) != file_sha(catalog_path):
            raise ValueError("Native relation catalog differs from current-fold binding")
        bindings = dict(g1_sha256=G1_SHA256, method="M1", cell=copy.deepcopy(data.bindings),
            corpus_digest=corpus["digest"], feature_binding=obj.feature_binding,
            source_hashes=current_sources, native_sources=verify_native_identity(),
            graph_policy="NATIVE_ALL_CANONICAL_FORWARD_PLUS_COMPUTED_INVERSE_CURRENT_FOLD_LEGAL_FACTS",
            fixed_config=fixed_config(data.training_seed), support_digest=digest(sorted(data.support)),
            blocked_digest=digest(sorted(data.blocked)), U_is_confirmed_negative=False,
            only_training_roles_loaded=True, reference_or_prototype="NOT_PRESENT_IN_HGTDR_NATIVE_ARCHITECTURE")
        if permit is not None:
            bindings.update(mode="FORMAL", execution_manifest_sha256=permit["_manifest_sha256"])
        receipt_path = output / ("FORMAL_native_graph.json" if permit is not None else "TEST_ONLY_native_graph.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["bindings"] != bindings:
                raise ValueError("Native graph cache binding differs")
            for relative, expected in receipt["artifacts"].items():
                if file_sha(output / relative) != expected:
                    raise ValueError("Native graph cache artifact differs")
            relations = pl.read_parquet(output / "relations.parquet")
        else:
            if (output / "edges").exists():
                raise ValueError("Uncommitted native graph export exists")
            relations, construction = export_graph_arrays(nodes, fact_paths, catalog, output, forbidden_pairs=set(data.blocked))
            relations.write_parquet(output / "relations.parquet")
            artifacts = {"relations.parquet": file_sha(output / "relations.parquet")}
            for r in relations.iter_rows(named=True):
                for key in ("edge_index_path", "fact_ids_path"):
                    artifacts[r[key]] = file_sha(output / r[key])
            receipt = dict(status="FORMAL_DATA_READY" if permit is not None else "TEST_ONLY_COMPLETE",
                           bindings=bindings, construction=construction, artifacts=artifacts)
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        obj.graph_receipt = receipt
        obj.relations = relations.to_dicts()
        arrays = [np.load(output / r["edge_index_path"], mmap_mode="r", allow_pickle=False) for r in obj.relations]
        obj.sampler = native_module().QueryMaskedSampler(obj.counts, obj.relations, arrays)
        obj.metadata = (list(obj.counts), obj.sampler.edge_types)
        return obj

    def unlabeled(self, count, seed):
        values = self.u_policy.sample(count, seed=seed)
        self.u_policy.validate(values)
        for pair in values:
            self.cell_source.pairs.assert_optimizer_pair(pair, "U")
        return np.asarray([[self.lookup["drug"][d], self.lookup["disease"][s]] for d, s in values], dtype=np.int64)

    def batch(self, pairs, config, seed, device):
        return native_batch(self, pairs, config, seed, device)


class SingleLRTrainer:
    """Native batches and exact epoch cursor; no implicit metric computation.

    This low-level engine is reused by bounded test workers. Formal dispatch is
    deliberately refused until the separately approved G3 execution gate exists.
    """
    def __init__(self, data, *, device="cuda", permit=None):
        self.data, self.device = data, device
        if permit is None:
            self.execution_scope = dict(mode="TEST_ONLY", formal_authorized=False)
        else:
            p = require_formal_execution(permit, repeat=data.bindings.get("original_repeat_index"), fold=data.bindings.get("fold_index"))
            self.execution_scope = dict(mode="FORMAL", formal_authorized=True,
                                       execution_manifest_sha256=p["_manifest_sha256"])
        self.config = fixed_config(data.training_seed)
        self.native = native_module()
        first = data.batch(data.supports[:min(self.config["batch_size"], len(data.supports))], self.config,
                           self.native.seed_for(data.training_seed, "initialize"), device)
        self.model = self.native.initialized_model(data, first, device, data.training_seed)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.003)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300, eta_min=0)
        self.cursor = dict(epoch=1, batch_index=0, optimizer_steps=0)
        self.selection = InnerCheckpointSelection()
        self.best_model = None
        self._epoch = None
        self.forwards = 1
        self.query_forward_count = min(self.config["batch_size"], len(data.supports))
        self.training_history = []
        self.epoch_accumulator = dict(weighted_loss=0., examples=0, P=0, U=0, masked=0)
        self.actual_training_seconds = 0.
        self.actual_evaluation_seconds = 0.
        self.actual_preparation_seconds = 0.

    def metadata(self):
        return dict(method="M1", has_refutation_head=False, config=self.config,
                    device=self.device, training_dtype="FP32_NATIVE", torch_version=str(torch.__version__),
                    execution_scope=copy.deepcopy(self.execution_scope),
                    checkpoint_cadence="EACH_EPOCH_PRE_INNER_AND_POST_INNER_COMPLETE_STATE_V1",
                    g1_sha256=G1_SHA256, data_bindings=self.data.bindings,
                    nominal_parameters=sum(p.numel() for p in self.model.parameters()),
                    adapter_sha256=file_sha(__file__), native_sources=verify_native_identity())

    def _schedule(self, epoch):
        if self._epoch is None or self._epoch[0] != epoch:
            seed = self.config["seed"]
            permutation = np.random.default_rng(self.native.seed_for(seed, "positive-order", epoch)).permutation(len(self.data.supports))
            unlabeled = self.data.unlabeled(len(self.data.supports), self.native.seed_for(seed, "unlabeled", epoch))
            self._epoch = epoch, permutation, unlabeled
        return self._epoch[1:]

    def step(self):
        epoch, batch_index = self.cursor["epoch"], self.cursor["batch_index"]
        if epoch > 300:
            raise ValueError("Frozen 300 epochs exhausted")
        permutation, unlabeled = self._schedule(epoch)
        half = self.config["batch_size"] // 2
        offset = batch_index * half
        indices = permutation[offset:offset + half]
        if not len(indices):
            raise ValueError("Invalid native training cursor")
        p, u = self.data.supports[indices], unlabeled[offset:offset + len(indices)]
        pairs = np.concatenate((p, u))
        labels = torch.cat((torch.ones(len(p), 1), torch.zeros(len(u), 1))).to(self.device)
        started = time.perf_counter()
        batch = self.data.batch(pairs, self.config, self.native.seed_for(self.config["seed"], "training-sample", epoch, batch_index), self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        scores = self.model(batch)
        self.forwards += 1
        self.query_forward_count += len(pairs)
        loss = self.native.compute_loss(scores, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite native HGTDR training loss")
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Missing or nonfinite native HGTDR gradients")
        self.optimizer.step()
        self.cursor["optimizer_steps"] += 1
        acc = self.epoch_accumulator
        acc["weighted_loss"] += float(loss.detach().cpu()) * len(pairs)
        acc["examples"] += len(pairs)
        acc["P"] += len(p)
        acc["U"] += len(u)
        acc["masked"] += batch["masked_traversals"]
        finished = offset + len(indices) == len(self.data.supports)
        self.cursor["batch_index"] += 1
        if finished:
            if acc["P"] != len(self.data.supports) or acc["U"] != len(self.data.supports):
                raise ValueError("Full native epoch did not preserve P/equal U coverage")
            self.training_history.append(dict(epoch=epoch, optimizer_steps=self.cursor["optimizer_steps"],
                train_loss=acc["weighted_loss"] / acc["examples"], positive_pairs_seen=acc["P"],
                unlabeled_pairs_seen=acc["U"], masked_traversals=acc["masked"],
                learning_rate=float(self.optimizer.param_groups[0]["lr"])))
            self.epoch_accumulator = dict(weighted_loss=0., examples=0, P=0, U=0, masked=0)
            self.scheduler.step()
            self.cursor.update(epoch=epoch + 1, batch_index=0)
            self._epoch = None
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.actual_training_seconds += elapsed
        return dict(status="FORMAL_TRAINING_UPDATE" if self.execution_scope["mode"] == "FORMAL" else "TEST_ONLY",
            epoch=epoch, batch_index=batch_index,
            optimizer_steps=self.cursor["optimizer_steps"], query_count=len(pairs), positive_count=len(p), U_count=len(u),
            loss=float(loss.detach().cpu()), finite=True, epoch_finished=finished,
            nodes=batch["node_count"], messages=batch["edge_count"], masked_current_query_traversals=batch["masked_traversals"],
            elapsed_seconds=elapsed, real_inner_scores_computed=0)

    def control_state(self):
        return dict(cursor=copy.deepcopy(self.cursor), selection=copy.deepcopy(vars(self.selection)), forwards=self.forwards,
                    query_forward_count=self.query_forward_count, training_history=copy.deepcopy(self.training_history),
                    epoch_accumulator=copy.deepcopy(self.epoch_accumulator),
                    actual_training_seconds=self.actual_training_seconds,
                    actual_evaluation_seconds=self.actual_evaluation_seconds,
                    actual_preparation_seconds=self.actual_preparation_seconds,
                    best_model=copy.deepcopy(self.best_model), metadata=self.metadata())

    def load_control_state(self, state):
        if state["metadata"] != self.metadata():
            raise ValueError("Native method/source/cell binding mismatch")
        cursor = state["cursor"]
        steps_per_epoch = math.ceil(len(self.data.supports) / (self.config["batch_size"] // 2))
        epoch, batch, step = (cursor[k] for k in ("epoch", "batch_index", "optimizer_steps"))
        if any(type(v) is not int for v in (epoch, batch, step)) or not 1 <= epoch <= 301 or not 0 <= batch < steps_per_epoch:
            raise ValueError("Invalid native epoch/batch cursor")
        if step != (epoch - 1) * steps_per_epoch + batch or (epoch == 301 and batch):
            raise ValueError("Native committed optimizer cursor mismatch")
        selection = state["selection"]
        if not 0 <= selection["opportunities"] <= min(epoch - 1, 300):
            raise ValueError("Selection opportunity/cursor mismatch")
        replay = InnerCheckpointSelection()
        if len(selection["history"]) != selection["opportunities"]:
            raise ValueError("Selection history/opportunity mismatch")
        for row in selection["history"]:
            replay.observe(row["epoch"], row["metrics"])
        if replay.best != selection["best"]:
            raise ValueError("Selection best checkpoint does not match committed history")
        best = state["best_model"]
        if best is not None:
            schema = self.model.state_dict()
            if set(best) != set(schema) or any(not isinstance(v, torch.Tensor) or v.shape != schema[k].shape
                    or v.dtype != schema[k].dtype or (v.is_floating_point() and not bool(torch.isfinite(v).all()))
                    for k, v in best.items()):
                raise ValueError("Best checkpoint schema/nonfinite value differs")
        if selection["opportunities"] == 0:
            if selection["best"] is not None or best is not None:
                raise ValueError("Best checkpoint exists without a selection opportunity")
        elif (not isinstance(selection["best"], dict) or best is None or
                not 1 <= selection["best"].get("epoch", 0) <= selection["opportunities"]):
            raise ValueError("Best checkpoint selection metadata is incomplete")
        if type(state["forwards"]) is not int or state["forwards"] < step + 1:
            raise ValueError("Native forward cursor invalid")
        if len(state["training_history"]) != epoch - 1 or any(row["epoch"] != i + 1 or
                row["positive_pairs_seen"] != len(self.data.supports) or row["unlabeled_pairs_seen"] != len(self.data.supports)
                for i, row in enumerate(state["training_history"])):
            raise ValueError("Native complete-epoch coverage/history differs")
        acc = state["epoch_accumulator"]
        expected_p = min(batch * (self.config["batch_size"] // 2), len(self.data.supports))
        if acc["P"] != expected_p or acc["U"] != expected_p or acc["examples"] != 2 * expected_p:
            raise ValueError("Native partial-epoch cursor/coverage differs")
        if type(state["query_forward_count"]) is not int or state["query_forward_count"] < state["forwards"]:
            raise ValueError("Native query-forward cursor invalid")
        for key in ("actual_training_seconds", "actual_evaluation_seconds", "actual_preparation_seconds"):
            if not math.isfinite(float(state[key])) or state[key] < 0:
                raise ValueError("Native committed timing invalid")
        self.cursor = copy.deepcopy(cursor)
        self.selection.best = copy.deepcopy(selection["best"])
        self.selection.opportunities = selection["opportunities"]
        self.selection.history = copy.deepcopy(selection["history"])
        self.best_model = copy.deepcopy(best)
        self.forwards = state["forwards"]
        self.query_forward_count = state["query_forward_count"]
        self.training_history = copy.deepcopy(state["training_history"])
        self.epoch_accumulator = copy.deepcopy(state["epoch_accumulator"])
        self.actual_training_seconds = state["actual_training_seconds"]
        self.actual_evaluation_seconds = state["actual_evaluation_seconds"]
        self.actual_preparation_seconds = state["actual_preparation_seconds"]
        self._epoch = None

    def score(self, domain):
        raise PermissionError("No new real inner/outer/final score in Phase92b: " + str(domain))

    def run_formal(self, execution_manifest=None, *, evaluate_inner=None, commit=None):
        require_formal_execution(execution_manifest,
            repeat=self.data.bindings["original_repeat_index"], fold=self.data.bindings["fold_index"])
        if not callable(evaluate_inner) or not callable(commit):
            raise ValueError("Authorized shared inner-metric and atomic commit adapters required")
        return execute_prepared_job(self, execution_manifest, evaluate_inner=evaluate_inner, commit=commit)


def native_state_identity(trainer, repeat, fold):
    return dict(method="M1", repeat=repeat, fold=fold, g1_sha256=G1_SHA256, mode=trainer.execution_scope["mode"],
                native_binding=trainer.metadata(), execution_scope=copy.deepcopy(trainer.execution_scope))


def capture_native_state(trainer, *, repeat, fold):
    from .production_state import capture_training_state
    cursor = dict(actual_optimizer_updates=trainer.cursor["optimizer_steps"],
                  completed_epochs=trainer.cursor["epoch"] - 1,
                  epoch=trainer.cursor["epoch"], batch_position=trainer.cursor["batch_index"],
                  selection_opportunities=trainer.selection.opportunities)
    return capture_training_state(trainer.model, trainer.optimizer, native_state_identity(trainer, repeat, fold), cursor,
        scheduler=trainer.scheduler, sampler={"native_control": trainer.control_state()}, model_metadata=trainer.metadata())


def restore_native_state(state, trainer, *, repeat, fold):
    from .production_state import restore_training_state
    control = state.get("sampler", {}).get("native_control")
    if not isinstance(control, dict):
        raise ValueError("Missing native cursor/selection/sampling control state")
    # Validate control on a shallow receiver before any live state is changed.
    probe = copy.copy(trainer)
    probe.selection = copy.deepcopy(trainer.selection)
    probe.load_control_state(control)
    c = state["cursor"]
    expected = dict(actual_optimizer_updates=control["cursor"]["optimizer_steps"],
                    completed_epochs=control["cursor"]["epoch"] - 1,
                    epoch=control["cursor"]["epoch"], batch_position=control["cursor"]["batch_index"],
                    selection_opportunities=control["selection"]["opportunities"])
    if c != expected:
        raise ValueError("Native committed receipt/control cursor differs")
    result = restore_training_state(state, trainer.model, trainer.optimizer,
        native_state_identity(trainer, repeat, fold), scheduler=trainer.scheduler, model_metadata=trainer.metadata())
    trainer.load_control_state(control)
    return result


def execute_prepared_job(trainer, execution_manifest, *, evaluate_inner, commit):
    """Prepared 300-epoch, one-LR control flow behind the closed G3 gate.

    evaluate_inner is the future *authorized inner-only* shared metric adapter;
    it cannot be supplied as a way to issue authorization. commit must publish
    model/optimizer/scheduler/RNG plus this complete control receipt atomically.
    No outer/final operation, LR loop, early stop or probe selection exists here.
    This task never issues a permit or executes the following full job.
    """
    if trainer is None:
        require_formal_execution(execution_manifest)
    bindings = trainer.data.bindings
    require_formal_execution(execution_manifest, repeat=bindings["original_repeat_index"], fold=bindings["fold_index"])
    return _run_fixed_epoch_control(trainer, evaluate_inner=evaluate_inner, commit=commit)


def _run_fixed_epoch_control(trainer, *, evaluate_inner, commit):
    """Control logic separable for zero-optimizer synthetic scheduling tests.

    Production callers must enter through execute_prepared_job's G3 verifier.
    This internal function never issues a permit or a formal receipt.
    """
    if trainer.config != fixed_config(trainer.data.training_seed):
        raise ValueError("Single-LR native configuration changed")
    actual_batches = 0

    def receipt(stage, **extra):
        return dict(stage=stage, control=trainer.control_state(),
                    actual_optimizer_updates=trainer.cursor["optimizer_steps"],
                    completed_epochs=trainer.cursor["epoch"] - 1,
                    epoch=trainer.cursor["epoch"], batch_position=trainer.cursor["batch_index"], **extra)

    def finish_pending_selection():
        completed_epoch = trainer.cursor["epoch"] - 1
        if trainer.selection.opportunities == completed_epoch:
            return
        if trainer.selection.opportunities != completed_epoch - 1 or trainer.cursor["batch_index"]:
            raise ValueError("Checkpoint selection opportunity/cursor cannot be recovered")
        metrics = evaluate_inner(trainer.model, "validation", completed_epoch)
        previous = trainer.selection.best
        trainer.selection.observe(completed_epoch, metrics)
        if trainer.selection.best != previous:
            trainer.best_model = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        commit(trainer, receipt("INNER_SELECTION_COMMITTED", best_epoch=trainer.selection.best["epoch"]))

    # A fully committed last batch may precede a crash in the inner-selection
    # callback; replay that authorized evaluation before taking another update.
    finish_pending_selection()
    while trainer.cursor["epoch"] <= 300:
        report = trainer.step()
        actual_batches += 1
        if report["epoch_finished"]:
            # Fixed engineering cadence: retain an exact pre-inner endpoint and
            # its completed inner-selection state. Uncommitted middle batches
            # replay from the previous full state/RNG after interruption.
            commit(trainer, receipt("EPOCH_TRAINING_COMMITTED", last_update=report))
            finish_pending_selection()
    if trainer.selection.opportunities != 300 or trainer.cursor["batch_index"]:
        raise ValueError("Native 300-epoch trajectory incomplete")
    return dict(status="TRAINING_COMPLETE_AWAITING_AUTHORIZED_ENDPOINT_DIAGNOSTIC",
                method="M1", epochs=300, selection_opportunities=300,
                optimizer_steps=trainer.cursor["optimizer_steps"], actual_batches_this_invocation=actual_batches,
                best_epoch=trainer.selection.best["epoch"], outer_scored=False, final_opened=False)


def validate_smoke_scope(repeat, fold, updates):
    if (repeat, fold) not in {(0, 1), (1, 0)} or type(updates) is not int or not 1 <= updates <= 3:
        raise PermissionError("TEST_ONLY native smoke is limited to the two prespecified cells and 1-3 updates")


def score_authorized_inner(trainer, data, name, permit):
    """Future native sampling plus unchanged shared metrics on approved inner members."""
    require_formal_execution(permit, repeat=data.repeat_index, fold=data.cell["fold"])
    if name not in {"validation", "train_probe"}:
        raise PermissionError("Native M1 metric adapter only supports authorized development panels")
    data.enable_development(permit, "M1")
    from evi_hgt.phase85_part2b.runtime import inner_metrics
    rows = data.cell[name]
    if any(r["role"] not in {"P", "U"} for r in rows):
        raise ValueError("Invalid original inner support roles")
    for row in rows:
        data.assert_score_domain((row["drug_id"], row["disease_id"]), name)
    frame = pl.DataFrame(rows).select("drug_id", "disease_id", "role").sort("drug_id", "disease_id")
    if frame.select("drug_id", "disease_id").n_unique() != frame.height:
        raise ValueError("Duplicate inner support member")
    native = trainer.data
    pairs = np.asarray([[native.lookup["drug"][r["drug_id"]], native.lookup["disease"][r["disease_id"]]]
                        for r in frame.iter_rows(named=True)], dtype=np.int64)
    values = []
    trainer.model.eval()
    with torch.inference_mode():
        for offset in range(0, len(pairs), trainer.config["batch_size"]):
            p = pairs[offset:offset + trainer.config["batch_size"]]
            batch = native.batch(p, trainer.config,
                trainer.native.seed_for(trainer.config["seed"], "evaluation", name, offset), trainer.device)
            result = trainer.model(batch).view(-1)
            trainer.forwards += 1
            trainer.query_forward_count += len(p)
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError("Nonfinite authorized inner support logit")
            values.extend(result.cpu().tolist())
    result = frame.with_columns(pl.Series("support_logit", values))
    return result, inner_metrics(result.to_dicts())


def _cached_native_panel(root, token, name, bindings, producer):
    """No authorization is issued; only an already-authorized caller supplies producer."""
    from .development_cache import once_panel
    if not re.fullmatch(r"epoch_[0-9]{4}|endpoint_selected", token):
        raise ValueError("Invalid bounded native development cache token")
    return once_panel(Path(root) / token, name, bindings, producer)


def _apply_cached_native_counters(trainer, result):
    current = dict(forwards=trainer.forwards, query_forwards=trainer.query_forward_count,
                   evaluation_seconds=trainer.actual_evaluation_seconds)
    if current == result["counters_before"]:
        after = result["counters_after"]
        trainer.forwards = after["forwards"]
        trainer.query_forward_count = after["query_forwards"]
        trainer.actual_evaluation_seconds = after["evaluation_seconds"]
    elif current != result["counters_after"]:
        raise ValueError("Cached native scoring counters differ from committed training state")


def _score_cached_authorized_inner(trainer, data, name, permit, root, token, checkpoint_sha256, epoch):
    require_formal_execution(permit, repeat=data.repeat_index, fold=data.cell["fold"])
    if name not in {"validation", "train_probe"}:
        raise PermissionError("No protected native score cache domain")
    from .state import tensor_digest
    bindings = dict(method="M1", repeat=data.repeat_index, fold=data.cell["fold"],
        cell_digest=data.contract.bindings["cell_digest"], epoch=epoch,
        member_digest=data.contract.bindings["membership_digests"][name],
        checkpoint_sha256=checkpoint_sha256, model_weight_digest=tensor_digest(trainer.model.state_dict()),
        g1_sha256=G1_SHA256, execution_manifest_sha256=permit["_manifest_sha256"],
        adapter_sha256=file_sha(__file__),
        development_cache_sha256=file_sha(Path(__file__).with_name("development_cache.py")))
    trainer.model.eval()

    def counters():
        return dict(forwards=trainer.forwards, query_forwards=trainer.query_forward_count,
                    evaluation_seconds=trainer.actual_evaluation_seconds)

    def producer():
        before = counters()
        started = time.perf_counter()
        frame, metrics = score_authorized_inner(trainer, data, name, permit)
        trainer.actual_evaluation_seconds += time.perf_counter() - started
        return dict(rows=frame.to_dicts(), metrics=metrics, counters_before=before, counters_after=counters())

    result = _cached_native_panel(root, token, name, bindings, producer)
    # A complete score cache may survive a crash before its training-state
    # commit. Restore the recorded work, without another physical forward.
    _apply_cached_native_counters(trainer, result)
    return pl.DataFrame(result["rows"]), result["metrics"]


def planned_native_updates(repeat, fold):
    cell = next(c for c in frozen_science()["cells"] if (c["repeat_index"], c["fold_index"]) == (repeat, fold))
    return 300 * math.ceil(cell["budget"]["full_training_support_P"] / (4096 // 2))


def build_trajectory_receipt(repeat, fold, checkpoint, *, actual_optimizer_updates,
                             completed_epochs, development_metrics_authorized_complete, **extra):
    expected = planned_native_updates(repeat, fold)
    identity, cursor = checkpoint["identity"], checkpoint["cursor"]
    if any(identity.get(k) != v for k, v in dict(method="M1", repeat=repeat, fold=fold, g1_sha256=G1_SHA256, mode="FORMAL").items()):
        raise ValueError("Formal native checkpoint method/cell/mode differs")
    if (actual_optimizer_updates != expected or completed_epochs != 300 or
            cursor.get("actual_optimizer_updates") != expected or cursor.get("completed_epochs") != 300 or
            cursor.get("selection_opportunities") != 300 or development_metrics_authorized_complete is not True):
        raise ValueError("Native fixed-epoch/selection/development endpoint incomplete")
    result = dict(schema="PHASE9_TRAJECTORY_RECEIPT_V1", mode="FORMAL", status="COMPLETE_VERIFIED",
        method="M1", repeat=repeat, fold=fold, g1_sha256=G1_SHA256,
        actual_optimizer_updates=actual_optimizer_updates, planned_optimizer_updates=expected, completed_epochs=300,
        development_metrics_authorized_complete=True, G4=False, G5=False,
        checkpoint_path=checkpoint["checkpoint_path"], checkpoint_commit_path=checkpoint["checkpoint_commit_path"],
        checkpoint_sha256=checkpoint["checkpoint_sha256"],
        current_training_model_location="checkpoint.model WITH ITS OWN FINAL OPTIMIZER/SCHEDULER",
        selected_model_location="checkpoint.sampler.native_control.best_model; SELECTED WEIGHTS ONLY")
    if set(result) & set(extra):
        raise ValueError("Additional fields cannot replace fixed native trajectory receipt fields")
    return dict(result, **extra)


def latest_native_commit(store):
    """Validate every small marker, then load the latest complete native state."""
    from .production_state import COMMIT_SCHEMA, _commit_digest
    candidates = []
    for path in store.root.glob("*.COMMIT.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("schema") != COMMIT_SCHEMA or row.get("status") != "COMMITTED" or
                row.get("mode") != store.mode or row.get("commit_digest") != _commit_digest(row)):
            raise ValueError("Native checkpoint commit marker integrity differs")
        cursor = row["cursor"]
        candidates.append(((cursor["actual_optimizer_updates"], cursor["selection_opportunities"]), path))
    if not candidates:
        raise FileNotFoundError("--resume requires an already committed same-arm checkpoint")
    return max(candidates, key=lambda item: item[0])[1]


def _validate_completed_native_receipt(receipt, checkpoint, output, manifest_sha256, repeat, fold):
    output = Path(output).resolve()
    expected = build_trajectory_receipt(repeat, fold, checkpoint,
        actual_optimizer_updates=receipt["actual_optimizer_updates"], completed_epochs=receipt["completed_epochs"],
        development_metrics_authorized_complete=receipt["development_metrics_authorized_complete"])
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Completed native trajectory/checkpoint identity differs")
    if (receipt.get("execution_manifest_sha256") != manifest_sha256 or
            receipt.get("final_report_absolute_path") != str(output / "receipt.json")):
        raise ValueError("Completed native execution/output binding differs")
    artifacts = receipt.get("endpoint_artifact_sha256", {})
    if set(artifacts) != {"train_probe_scores.parquet", "best_validation_scores.parquet"}:
        raise ValueError("Completed native development artifact set differs")
    if any(file_sha(output / name) != sha for name, sha in artifacts.items()):
        raise ValueError("Completed native development artifact content differs")
    return receipt


def _load_completed_native_receipt(output, permit, repeat, fold):
    from .production_state import verify_commit
    from evi_hgt.phase85_v2.common import read, verify
    require_formal_execution(permit, repeat=repeat, fold=fold)
    output = Path(output).resolve()
    receipt = verify(read(output / "receipt.json"))
    checkpoint = verify_commit(receipt["checkpoint_commit_path"])
    return _validate_completed_native_receipt(receipt, checkpoint, output, permit["_manifest_sha256"], repeat, fold)


def run_formal_job(repeat, fold, output, execution_manifest, *, resume=False):
    """G3-ready native driver; empty approval registry rejects before data/model I/O."""
    from .execution import validate_execution_manifest
    permit = validate_execution_manifest(execution_manifest, "M1", repeat, fold)
    output = Path(output).resolve()
    if not output.is_relative_to(Path(permit["output_root"]).resolve()):
        raise PermissionError("M1 output outside separately approved formal run")
    final_path = output / "receipt.json"
    if final_path.exists():
        # Verify source-bound authorization, full commit and endpoint artifacts,
        # then reuse before loading any real data or constructing another model.
        return _load_completed_native_receipt(output, permit, repeat, fold)
    from .real_data import RealInputs, build_fold_corpus
    from .production_state import AtomicCheckpointStore, verify_commit
    torch.set_num_threads(4)
    started = time.perf_counter()
    data = RealInputs().cell(repeat, fold)
    corpus = build_fold_corpus(data, permit=permit)
    native_data = HGTDRFoldData.from_cell(data, corpus, output / "native_graph", permit=permit)
    trainer = SingleLRTrainer(native_data, device="cuda", permit=permit)
    store = AtomicCheckpointStore(output / "checkpoints", mode="FORMAL", execution_permit=permit)
    last_receipt = None
    if resume:
        resume_path = latest_native_commit(store) if resume is True else resume
        state = store.load(resume_path)
        restore_native_state(state, trainer, repeat=repeat, fold=fold)
        last_receipt = verify_commit(resume_path)
    elif any(store.root.glob("*.COMMIT.json")):
        raise FileExistsError("Existing committed trajectory requires explicit same-arm resume")
    trainer.actual_preparation_seconds += time.perf_counter() - started
    latest_validation = {}

    def evaluate_inner(model, name, epoch):
        if last_receipt is None:
            raise ValueError("Native inner selection requires a committed current training endpoint")
        frame, metrics = _score_cached_authorized_inner(trainer, data, name, permit,
            output / "development_cache", f"epoch_{epoch:04d}", last_receipt["checkpoint_sha256"], epoch)
        latest_validation.update(epoch=epoch, frame=frame)
        return metrics

    def commit(active, receipt):
        nonlocal last_receipt
        require_formal_execution(permit, repeat=repeat, fold=fold)
        state = capture_native_state(active, repeat=repeat, fold=fold)
        stage = "inner" if receipt["stage"] == "INNER_SELECTION_COMMITTED" else "update"
        name = f"step_{active.cursor['optimizer_steps']:07d}_{stage}"
        # Save a complete resumable score artifact only when this authorized
        # epoch becomes the selected checkpoint, before publishing its complete
        # state marker. A crash before the marker replays pending selection.
        if stage == "inner" and latest_validation.get("epoch") == active.selection.best["epoch"]:
            temporary = output / "best_validation_scores.parquet.partial"
            latest_validation["frame"].write_parquet(temporary)
            temporary.replace(output / "best_validation_scores.parquet")
        last_receipt = store.commit(name, state)
        progress = dict(CURRENT_METHOD="M1", REPEAT=repeat, FOLD=fold, STAGE="NATIVE_EPOCH",
            OPTIMIZER_STEP=active.cursor["optimizer_steps"], EPOCH=active.cursor["epoch"], PLANNED_EPOCHS=300,
            PLANNED_STEPS=300 * math.ceil(len(active.data.supports) / (active.config["batch_size"] // 2)),
            ELAPSED_TIME=time.perf_counter() - started,
            TRAINING_LOSS=active.training_history[-1]["train_loss"] if active.training_history else None,
            LAST_VERIFIED_CHECKPOINT=last_receipt["checkpoint_commit_path"], FINITE_STATUS=True)
        print(json.dumps(progress), flush=True)

    try:
        result = execute_prepared_job(trainer, permit, evaluate_inner=evaluate_inner, commit=commit)
        if trainer.best_model is None:
            raise ValueError("Native selected model missing after 300 opportunities")
        # Diagnostic is explicitly after training, never an additional selector.
        trainer.model.load_state_dict(trainer.best_model)
        probe_scores, probe_metrics = _score_cached_authorized_inner(trainer, data, "train_probe", permit,
            output / "development_cache", "endpoint_selected", last_receipt["checkpoint_sha256"], trainer.selection.best["epoch"])
        probe_scores.write_parquet(output / "train_probe_scores.parquet")
        result = build_trajectory_receipt(repeat, fold, last_receipt,
            actual_optimizer_updates=trainer.cursor["optimizer_steps"], completed_epochs=300,
            development_metrics_authorized_complete=True, complete_verified=True,
            execution_manifest_sha256=permit["_manifest_sha256"],
            authorized_development_metrics=dict(
                best_inner=trainer.selection.best, train_probe=probe_metrics),
            actual_evaluation_seconds=trainer.actual_evaluation_seconds,
            actual_training_seconds=trainer.actual_training_seconds,
            actual_preparation_seconds=trainer.actual_preparation_seconds,
            current_invocation_wall_seconds=time.perf_counter() - started,
            cost_scope="COMMITTED_TRAIN_AND_EVAL_PLUS_CURRENT_ENDPOINT; TRAIN_INCLUDES_NATIVE_CPU_SAMPLING; UNCOMMITTED_CRASH_WORK_EXCLUDED",
            actual_forward_calls=trainer.forwards, actual_query_forwards=trainer.query_forward_count,
            last_verified_checkpoint=last_receipt["checkpoint_commit_path"],
            endpoint_artifact_sha256={"train_probe_scores.parquet": file_sha(output / "train_probe_scores.parquet"),
                "best_validation_scores.parquet": file_sha(output / "best_validation_scores.parquet")},
            final_report_absolute_path=str(final_path), outer_scored=False, final_opened=False)
        # Complete marker follows all mandatory training, state and endpoint work.
        from evi_hgt.phase85_v2.common import save
        save(final_path, result)
        return result
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = dict(status="FAILED", method="M1", repeat=repeat, fold=fold,
            error=f"{type(exc).__name__}: {exc}", cursor=trainer.cursor,
            last_verified_checkpoint=last_receipt["checkpoint_commit_path"] if last_receipt else None,
            final_report_absolute_path=str(output / "FAILED.json"), outer_scored=False, final_opened=False)
        (output / "FAILED.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


def run_test_preflight(repeat, fold, *, updates=1, device="cuda"):
    """Bounded actual training batches; no real metric or checkpoint selection."""
    validate_smoke_scope(repeat, fold, updates)
    from .real_data import RealInputs
    from evi_hgt.phase85_v2.common import read, verify
    torch.set_num_threads(4)
    started = time.perf_counter()
    data = RealInputs().cell(repeat, fold)
    if data.original_corpus_root.exists():
        data.corpus_root = data.original_corpus_root
    corpus = verify(read(data.corpus_root / "manifest.json"))
    output = TEST_ROOT / "hgtdr" / f"r{repeat}_f{fold}"
    native_data = HGTDRFoldData.from_cell(data, corpus, output / "native_graph")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; do not silently substitute hardware")
        torch.cuda.reset_peak_memory_stats()
    preparation_seconds = time.perf_counter() - started
    trainer = SingleLRTrainer(native_data, device=device)
    training_started = time.perf_counter()
    reports = [trainer.step() for _ in range(updates)]
    region_seconds = time.perf_counter() - training_started
    result = dict(status="TEST_ONLY_PREFLIGHT_PASS", method="M1", repeat=repeat, fold=fold,
        g1_sha256=G1_SHA256, training_seed=data.training_seed,
        actual_optimizer_updates=updates, forward_calls=trainer.forwards,
        query_count=sum(r["query_count"] for r in reports), update_reports=reports,
        preparation_seconds=preparation_seconds, train_region_seconds=region_seconds,
        wall_seconds=time.perf_counter() - started,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else 0,
        device=device, source_binding=trainer.metadata(), feature_binding=native_data.feature_binding,
        train_support_P_count=len(data.support), training_U_legal_domain=data.legal_u_domain(),
        common_support_membership_digest=data.contract.bindings["membership_digests"]["training"],
        native_graph=copy.deepcopy(native_data.graph_receipt["construction"]),
        selection_opportunities_used=trainer.selection.opportunities, real_inner_scores_computed=0,
        outer_roles_read=False, final_roles_read=False, formal_authorized=False, test_weights_reusable_for_formal=False)
    output.mkdir(parents=True, exist_ok=True)
    report = output / "TEST_ONLY_preflight.json"
    if report.exists():
        raise FileExistsError("Do not overwrite an existing actual update receipt")
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    del trainer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only-preflight", action="store_true")
    parser.add_argument("--mode", choices=("TEST_ONLY", "FORMAL"), default="TEST_ONLY")
    parser.add_argument("--method", choices=("M1",), default="M1")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "FORMAL":
        if args.test_only_preflight or args.output is None or args.execution_manifest is None or args.device != "cuda":
            raise PermissionError("FORMAL requires an independent exact execution manifest and CUDA output")
        result = run_formal_job(args.repeat, args.fold, args.output, args.execution_manifest, resume=args.resume)
        print(json.dumps(result), flush=True)
        return
    if not args.test_only_preflight or args.resume or args.execution_manifest or args.output:
        raise PermissionError("TEST_ONLY must explicitly select its isolated bounded preflight")
    result = run_test_preflight(args.repeat, args.fold, updates=args.updates, device=args.device)
    print(json.dumps({k: result[k] for k in ("status", "repeat", "fold", "actual_optimizer_updates",
        "forward_calls", "query_count", "train_region_seconds", "peak_gpu_allocated_bytes")}), flush=True)


if __name__ == "__main__":
    main()
