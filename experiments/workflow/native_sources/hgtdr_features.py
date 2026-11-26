"""Frozen canonical name features for HGTDR's eight additional node types."""

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from filelock import FileLock

from evi_hgt.data.common import sha256
from evi_hgt.io import digest, write_json
from evi_hgt.preprocessing.feature_encoders import encode_texts, load_encoder
from evi_hgt.settings import ROOT
from experiments.pilot.EviHGT.src.evi_hgt.competition.data import HUMAN_TYPES

EXTRA = tuple(kind for kind in HUMAN_TYPES if kind not in ("DRG", "DIS"))


def _extra_names(path, wanted):
    names = {}
    for chunk in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=["id", "label", "properties"]):
        for identifier, kind, serialized in pl.from_arrow(chunk).filter(pl.col("label").is_in(EXTRA)).iter_rows():
            key = kind, identifier
            if key in wanted:
                if key in names:
                    raise ValueError("Duplicate typed canonical name source")
                properties = json.loads(serialized) if isinstance(serialized, str) else serialized
                if not isinstance(properties, dict):
                    raise ValueError("Canonical name source properties must be a mapping")
                name = properties.get("name")
                names[key] = name if isinstance(name, str) and name.strip() else None
    return names


def _additional_features(nodes, context, original_nodes_sha256):
    source = Path(context["nodes"])
    extras = nodes.filter(pl.col("node_type").is_in(EXTRA))
    node_rows = extras.select("node_type", "node_id", "type_local_id").rows()
    source_sha = sha256(source)
    if source_sha != original_nodes_sha256:
        raise ValueError("HGTDR canonical name input differs from the aggregate used for this corpus")
    device = context.get("feature_device", "cuda")
    batch_size = context.get("feature_batch_size", 16)
    identity = dict(raw_nodes_sha256=source_sha, rows_sha256=digest(node_rows),
                    batch_size=batch_size, device=device, torch=str(torch.__version__),
                    recipe="HGTDR_CANONICAL_NAMES_BIOBERT_V1", code_sha256=sha256(__file__))
    cache = Path(context.get("native_cache", ROOT / "data/cache/experiments/native"))
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / ("hgtdr_extra_" + digest(identity))
    with FileLock(str(output) + ".lock"):
        if not output.exists():
            working = output.with_name(output.name + "." + uuid4().hex)
            working.mkdir()
            names = _extra_names(source, {(kind, value) for kind, value, _ in node_rows})
            tokenizer, model, encoder = load_encoder("biobert", context["model_cache"], device,
                                                       offline=context.get("offline", False))
            for kind in EXTRA:
                group = extras.filter(pl.col("node_type") == kind).sort("type_local_id")
                texts = [names.get((kind, value)) for value in group["node_id"]]
                mask = np.asarray([text is not None for text in texts], dtype=np.bool_)
                encode_texts(texts, mask, "biobert", working / f"{kind}.npy", tokenizer, model,
                             device=device, batch_size=batch_size)
                np.save(working / f"{kind}_available.npy", mask, allow_pickle=False)
            if sha256(source) != source_sha:
                raise ValueError("Canonical name source changed during frozen encoding")
            write_json(working / "manifest.json", dict(identity=identity, encoder=encoder,
                files={path.name: sha256(path) for path in sorted(working.glob("*.npy"))}))
            os.rename(working, output)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Extra HGTDR feature identity differs")
        names = {name for kind in EXTRA for name in (f"{kind}.npy", f"{kind}_available.npy")}
        if set(manifest["files"]) != names:
            raise ValueError("Incomplete or unexpected HGTDR feature cache files")
        for relative, expected in manifest["files"].items():
            if sha256(output / relative) != expected:
                raise ValueError("HGTDR feature cache differs")
        return output, manifest


def load_fixed_features(corpus, nodes, context, *, smoke):
    source = corpus.manifest.get("preparation", {}).get("raw_sha256", {}).get("nodes")
    extra_root, extra_manifest = (None, None) if smoke else _additional_features(nodes, context, source)
    values, masks = {}, {}
    for kind in nodes["node_type"].unique(maintain_order=True):
        group = nodes.filter(pl.col("node_type") == kind).sort("type_local_id")
        if kind in ("DRG", "DIS"):
            ids, feature, available = corpus.pretrained.arrays[kind]
            order = np.searchsorted(ids, group["global_node_id"].to_numpy())
            if not np.array_equal(ids[order], group["global_node_id"].to_numpy()):
                raise ValueError("Native D1 canonical node order differs")
            feature, available = feature[order], available[order]
        elif smoke:
            feature = np.zeros((group.height, 768), dtype=np.float32)
            available = np.zeros(group.height, dtype=np.bool_)
        else:
            feature = np.load(extra_root / f"{kind}.npy", allow_pickle=False)
            available = np.load(extra_root / f"{kind}_available.npy", allow_pickle=False)
        if (feature.shape != (group.height, 768) or available.shape != (group.height,)
            or feature.dtype != np.float32 or available.dtype != np.bool_
            or not np.isfinite(feature).all() or np.any(feature[~available])):
            raise ValueError("Invalid native feature values or availability mask")
        values[HUMAN_TYPES[kind]], masks[HUMAN_TYPES[kind]] = torch.from_numpy(feature), torch.from_numpy(available)
    return values, masks, None if smoke else digest(extra_manifest)
