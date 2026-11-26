"""Canonical graph export shared by the two native baselines."""

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from filelock import FileLock

from evi_hgt.data.common import sha256
from evi_hgt.io import write_json
from evi_hgt.settings import ROOT
from experiments.pilot.EviHGT.src.evi_hgt.competition.data import (
    HUMAN_TYPES, UnlabeledPolicy, export_graph_arrays, normalize_nodes,
)


def native_nodes(corpus):
    nodes = corpus.nodes
    if "type_local_id" not in nodes.columns:
        if corpus.manifest.get("purpose") != "SYNTHETIC_TEST_ONLY":
            raise ValueError("The portable corpus lacks canonical type-local IDs")
        nodes = nodes.with_columns((pl.col("global_node_id").rank("ordinal").over("node_type") - 1)
                                   .alias("type_local_id"))
    return normalize_nodes(nodes)


def native_graph(corpus, context, root, *, smoke):
    base = root if smoke else Path(context.get("native_cache", ROOT / "data/cache/experiments/native"))
    base.mkdir(parents=True, exist_ok=True)
    output = base / (corpus.identity + "_native_coo")
    nodes = native_nodes(corpus)
    files = [corpus.resolve(row["path"]) for row in corpus.manifest["fact_files"]]
    catalog = [dict(edge_type_id=i, source_node_type=source, relation=relation, target_node_type=target)
               for i, (source, relation, target) in enumerate(corpus.index.edge_catalog)]
    with FileLock(str(output) + ".lock"):
        if not output.exists():
            staging = output.with_name(output.name + "." + uuid4().hex)
            staging.mkdir()
            relations, _ = export_graph_arrays(nodes, files, catalog, staging, forbidden_pairs=corpus.blocked)
            relations.write_parquet(staging / "relations.parquet")
            inventory = {p.relative_to(staging).as_posix(): sha256(p)
                         for p in staging.rglob("*") if p.is_file()}
            write_json(staging / "manifest.json", dict(corpus=corpus.identity, files=inventory))
            os.rename(staging, output)
        record = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if record["corpus"] != corpus.identity or not record["files"]:
            raise ValueError("Native graph belongs to another portable corpus")
        for relative, expected in record["files"].items():
            path = (output / relative).resolve()
            if not path.is_relative_to(output.resolve()) or sha256(path) != expected:
                raise ValueError("Native graph cache payload changed")
        relations = pl.read_parquet(output / "relations.parquet").to_dicts()
        arrays = [np.load(output / row["edge_index_path"], mmap_mode="r", allow_pickle=False)
                  for row in relations]
        for row in relations[::2]:
            ids = np.load(output / row["fact_ids_path"], mmap_mode="r", allow_pickle=False)
            if corpus.forbidden and np.isin(ids, list(corpus.forbidden)).any():
                raise ValueError("Excluded canonical fact reached native export")
    return nodes, relations, arrays, record


def native_unlabeled(corpus):
    return UnlabeledPolicy(tuple(corpus.eligible_drugs), tuple(corpus.eligible_diseases),
                           frozenset(corpus.observed), frozenset(corpus.blocked))


def native_lookup(nodes):
    return {HUMAN_TYPES[kind]: dict(nodes.filter(pl.col("node_type") == kind)
                                     .select("node_id", "type_local_id").iter_rows())
            for kind in ("DRG", "DIS")}
