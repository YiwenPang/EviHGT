"""FEATURE_BASELINE frozen, label-independent DRG/DIS cache certification and lookup.

Only the precomputed inference arrays are read. This module never loads an
HGTDR checkpoint, calls its data loader, follows provenance raw-source paths,
opens label panels, downloads encoders, or changes any random-number stream.
"""
from __future__ import annotations

from ablation_protocol_ids import FEATURE_SCHEMA
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

VERSION = FEATURE_SCHEMA
ROOT = Path(__file__).resolve().parents[2]
HGTDR_ROOT = ROOT.parent / "HGTDR"
FEATURE_ROOT = HGTDR_ROOT / "data/processed/competition_pilot_v1/features"
NODE_INDEX = ROOT / "data/processed/phase3/v1/nodes/node_index.parquet"
NODE_TYPE_ORDER = ("ANA", "BPO", "CCO", "DIS", "DRG", "EXP", "GEN", "MFN", "PHE", "PWY")
FEATURE_MANIFEST_SHA256 = "29188a9135b71624b4d7eaa972659fd19a54931b1b75540828a9fec294ae3531"
# Identities already certified by ranking_protocol G2, checked without importing the
# generators (the generator's build_inputs function would read raw sources).
GENERATOR_SOURCES = {
    "src/competition_features.py": "dd4787d15900db5ef5c98c1762b0495709040d20301f5bbb700c95b6500ccca0",
    "src/rebuild_features.py": "b1b82ddf546231bd0eda53a52fd0cebda95cfe03a43de2b14358a781c3caf835",
    "src/rebuild_smiles.py": "a37930deaeb57f442ae8a5363e8f88e5c0b08f1d3b65da46f8cbc511f8aed39a",
}
ENCODER_SPECS = {
    "chemberta": dict(repo="seyonec/ChemBERTa-zinc-base-v1",
        revision="761d6a18cf99db371e0b43baf3e2d21b3e865a20",
        weight_sha256="bf8a23d93cca196c3712c94ec379494a74be64c83d86a63b08f03e26db547267",
        max_length=512),
    "biobert": dict(repo="dmis-lab/biobert-v1.1",
        revision="551ca18efd7f052c8dfa0b01c94c2a8e68bc5488",
        weight_sha256="8e87cb27d889186c442feff2bea53f8cce2c5fc24d4bc88bf08ccd3821b138dc",
        max_length=128),
}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer_vector(values, name):
    if torch.is_tensor(values):
        _require(values.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
                 name + " must contain integer IDs")
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    _require(values.ndim == 1 and (not values.size or values.dtype.kind in "iu"),
             name + " must be a one-dimensional integer ID vector")
    return values.astype(np.int64, copy=False)


class PretrainedStore:
    """Read-only mmap arrays, globally keyed and strictly type checked.

    The production defaults pin the historical feature manifest. Alternate
    paths plus an explicit expected hash exist for synthetic acceptance tests;
    callers must bind the resulting metadata into their own run identity.
    """

    def __init__(self, feature_root=FEATURE_ROOT, *, node_index=NODE_INDEX,
                 expected_manifest_sha256=FEATURE_MANIFEST_SHA256):
        directory, node_path = Path(feature_root).resolve(), Path(node_index).resolve()
        manifest_path = directory / "manifest.json"
        _require(file_sha(manifest_path) == expected_manifest_sha256, "Frozen feature manifest identity changed")
        manifest = _read(manifest_path)
        _require(manifest.get("schema") == "HGTDR_COMMON_FIXED_FEATURES_V1" and
                 manifest.get("status") == "COMPLETE" and manifest.get("training") is False and
                 manifest.get("optimizer_steps") == 0 and manifest.get("sealed_final_labels_read") is False,
                 "Frozen inference cache required; trained/supervised feature caches are forbidden")
        _require(manifest.get("width") == 768 and manifest.get("dtype") == "float32", "Wrong pretrained feature dimensions or dtype")
        generator_identity = {}
        for relative, expected in GENERATOR_SOURCES.items():
            path = HGTDR_ROOT / relative
            _require(file_sha(path) == expected, "Frozen encoder-generation source changed: " + relative)
            generator_identity[str(path)] = expected
        for encoder, expected in ENCODER_SPECS.items():
            actual = manifest.get("fixed_encoder_specs", {}).get(encoder, {})
            _require(all(actual.get(k) == v for k, v in expected.items()), "Pinned encoder source/version differs")

        input_manifest_path, inputs_path = directory.parent / "input_manifest.json", directory.parent / "feature_inputs.parquet"
        _require(file_sha(input_manifest_path) == manifest.get("input_manifest_sha256"), "Input provenance manifest identity changed")
        provenance = _read(input_manifest_path)
        _require(provenance.get("schema") == "HGTDR_COMMON_FEATURE_INPUTS_V1" and
                 provenance.get("status") == "COMPLETE" and provenance.get("clinical_text_encoded") is False and
                 provenance.get("sealed_final_labels_read") is False and provenance.get("optimizer_steps") == 0,
                 "Feature source includes unverified clinical/label encoding")
        _require(file_sha(node_path) == provenance.get("node_index_sha256"), "Canonical node index identity changed")
        inputs_sha = file_sha(inputs_path)
        _require(inputs_sha == manifest.get("feature_inputs_sha256") == provenance.get("feature_inputs_sha256"),
                 "Frozen feature inputs identity changed")
        identity_columns = ["global_node_id", "type_local_id", "node_type", "node_id"]
        nodes = pl.read_parquet(node_path, columns=identity_columns).sort("global_node_id")
        inputs = pl.read_parquet(inputs_path).sort("global_node_id")
        _require(nodes.height > 0 and nodes.height == inputs.height == provenance.get("rows"), "Node index/input length mismatch")
        for column in identity_columns:
            _require(nodes[column].null_count() == 0 and nodes[column].to_list() == inputs[column].to_list(),
                     "Canonical/global/type-local/entity mapping mismatch: " + column)
        _require(nodes.select("node_type", "node_id").n_unique() == nodes.height, "Duplicate canonical entity identity")
        _require(np.array_equal(nodes["global_node_id"].to_numpy(), np.arange(nodes.height)), "Global IDs must be unique and contiguous")
        _require(set(nodes["node_type"]) <= set(NODE_TYPE_ORDER), "Unverified canonical node type")
        for group in nodes.partition_by("node_type"):
            _require(np.array_equal(np.sort(group["type_local_id"].to_numpy()), np.arange(group.height)),
                     "Type-local IDs must be unique and contiguous")
        self.kind_by_global = np.asarray([NODE_TYPE_ORDER.index(k) for k in nodes["node_type"]], dtype=np.int16)
        self.local_by_global = nodes["type_local_id"].to_numpy().astype(np.int64)
        self.arrays, self.masks = {}, {}
        coverage, artifacts = {}, {str(manifest_path): expected_manifest_sha256,
            str(node_path): provenance["node_index_sha256"], str(input_manifest_path): manifest["input_manifest_sha256"],
            str(inputs_path): inputs_sha}
        records = manifest.get("types", [])
        _require(len({r["node_type"] for r in records}) == len(records), "Duplicate cache type records")
        records = {r["node_type"]: r for r in records}
        for kind, encoder in (("DRG", "chemberta"), ("DIS", "biobert")):
            selected = inputs.filter(pl.col("node_type") == kind).sort("type_local_id")
            record = records.get(kind, {})
            _require(record.get("node_type_id") == NODE_TYPE_ORDER.index(kind) and
                     record.get("encoder") == encoder and record.get("rows") == selected.height and
                     record.get("width") == 768, "Typed cache schema mismatch: " + kind)
            _require(selected["available"].dtype == pl.Boolean and selected["available"].null_count() == 0,
                     "Availability must be an explicit Boolean for every entity")
            allowed = {"optimus_canonical_smiles", "drugbank_structured_smiles"} if kind == "DRG" else {"canonical_name"}
            digest = hashlib.sha256()
            for row in selected.select("node_id", "encoder_text", "available", "source_field", "encoder").iter_rows(named=True):
                present, source_text = row["available"], row["encoder_text"]
                _require(row["encoder"] == encoder, "Entity encoder differs from its type")
                _require((present and row["source_field"] in allowed and isinstance(source_text, str) and bool(source_text.strip())) or
                         (not present and row["source_field"] is None and source_text is None),
                         "Unapproved source field or availability/text mismatch")
                digest.update(json.dumps([row["node_id"], source_text if present else None, bool(present)], ensure_ascii=False).encode("utf-8"))
                digest.update(b"\n")
            _require(digest.hexdigest() == record.get("input_sha256"), "Typed input row digest differs")
            array_path = directory / f"type_{record['node_type_id']:02d}.npy"
            mask_path = directory / f"type_{record['node_type_id']:02d}_available.npy"
            for path, expected in ((array_path, record.get("sha256")), (mask_path, record.get("mask_sha256"))):
                _require(file_sha(path) == expected, "Frozen feature array or mask identity changed")
                artifacts[str(path)] = expected
            values = np.load(array_path, mmap_mode="r", allow_pickle=False)
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            _require(values.shape == (selected.height, 768) and values.dtype == np.float32, "Invalid feature tensor shape or dtype")
            _require(mask.shape == (selected.height,) and mask.dtype == np.bool_, "Invalid availability mask shape or dtype")
            _require(np.array_equal(mask, selected["available"].to_numpy()), "Array mask/source availability differs")
            for offset in range(0, len(mask), 4096):
                chunk, present = values[offset:offset + 4096], mask[offset:offset + 4096]
                _require(np.isfinite(chunk).all() and not np.any(chunk[~present]), "Nonfinite features or nonzero missing feature rows")
            _require(int(mask.sum()) == record.get("available_rows") and int((~mask).sum()) == record.get("missing_rows"),
                     "Availability counts differ from cache manifest")
            self.arrays[kind], self.masks[kind] = values, mask
            coverage[kind] = dict(rows=len(mask), available=int(mask.sum()), missing=int((~mask).sum()),
                encoder=encoder, available_fraction=float(mask.mean()))
        mapping_hash = hashlib.sha256()
        for row in nodes.iter_rows():
            mapping_hash.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        self.metadata = dict(schema=VERSION, status="CERTIFIED_FROZEN_LOCAL_CACHE", width=768,
            dtype="float32", loaded_node_types=["DRG", "DIS"], coverage=coverage,
            source_artifact_sha256=artifacts, generator_source_sha256=generator_identity,
            source_code_sha256=file_sha(__file__), canonical_mapping_sha256=mapping_hash.hexdigest(),
            canonical_node_count=nodes.height, node_type_order=list(NODE_TYPE_ORDER),
            fixed_encoder_specs=ENCODER_SPECS, feature_manifest_sha256=expected_manifest_sha256,
            production_cache_identity=expected_manifest_sha256 == FEATURE_MANIFEST_SHA256,
            clinical_text_encoded=False, supervised_dda_features_present=False,
            trained_node_embeddings_loaded=False, prediction_scores_loaded=False,
            training=False, optimizer_steps=0, fold_specific_inputs=False,
            cross_fold_identity="SINGLE_GLOBAL_CANONICAL_CACHE; FOLD_IS_NOT_AN_INPUT_TO_LOOKUP",
            missing_policy="EXACT_ZERO_INCREMENT_WITH_FALSE_AVAILABLE_MASK",
            pooling="LAST_HIDDEN_STATE_MEAN_OVER_NONPADDING_NONSPECIAL_TOKENS",
            upstream_pretraining_target_exposure="NOT_VERIFIED", raw_source_paths_followed=False,
            protected_label_panels_opened=False, encoder_weights_loaded=False, network_used=False)

    def gather(self, global_ids, node_type_ids, drug_type=4, disease_type=3, device="cpu"):
        """Return detached float32 [N,768] and bool [N] in input-node order."""
        _require(drug_type == NODE_TYPE_ORDER.index("DRG") and disease_type == NODE_TYPE_ORDER.index("DIS"),
                 "Model node-type binding differs from the certified canonical order")
        ids = _integer_vector(global_ids, "global_ids")
        types = _integer_vector(node_type_ids, "node_type_ids")
        _require(ids.shape == types.shape, "Feature lookup vectors differ in length")
        _require(not len(ids) or (int(ids.min()) >= 0 and int(ids.max()) < len(self.kind_by_global)),
                 "Unknown global node identity")
        _require(np.array_equal(self.kind_by_global[ids], types), "Global entity/model node-type mismatch")
        result, available = np.zeros((len(ids), 768), dtype=np.float32), np.zeros(len(ids), dtype=np.bool_)
        for kind, type_id in (("DRG", drug_type), ("DIS", disease_type)):
            positions = np.flatnonzero(types == type_id)
            local = self.local_by_global[ids[positions]]
            result[positions] = self.arrays[kind][local]
            available[positions] = self.masks[kind][local]
        # Missing rows and all other types remain literal zeros; no learned
        # replacement or ID fallback is shared with the HGTDR model.
        return torch.from_numpy(result).to(device), torch.from_numpy(available).to(device)
