"""Compact typed graph indexes without NetworkX or GPU materialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl

from .csr_order_recipe import ORDER_POLICY, RELATION_GROUP_ORDER


_CSR_COLUMNS = [
    "canonical_fact_id",
    "source_global_id",
    "target_global_id",
    "source_node_type",
    "target_node_type",
    "relation",
    "undirected",
]


def _load_facts(value: pl.DataFrame | str | Path) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value.select(_CSR_COLUMNS)
    return pl.read_parquet(value, columns=_CSR_COLUMNS)


def _ordered_relation_groups(frame: pl.DataFrame, partition: pl.DataFrame | str | Path):
    columns = ("source_node_type", "relation", "target_node_type")
    groups = frame.select(columns).unique().iter_rows()
    component = Path(partition).stem if isinstance(partition, (str, Path)) else None
    recipe = RELATION_GROUP_ORDER.get(component, ())
    rank = {group: i for i, group in enumerate(recipe)}
    for group in sorted(groups, key=lambda value: (rank.get(value, len(rank)), value)):
        yield dict(zip(columns, group, strict=True))


def _scatter_edges(
    *,
    sources: np.ndarray,
    targets: np.ndarray,
    facts: np.ndarray,
    edge_type_id: int,
    cursor: np.ndarray,
    indices: np.memmap,
    fact_ids: np.memmap,
    edge_type_ids: np.memmap,
) -> None:
    if not sources.size:
        return
    order = np.argsort(sources, kind="stable")
    sorted_sources = sources[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_sources[1:] != sorted_sources[:-1]))
    )
    counts = np.diff(np.append(starts, sorted_sources.size))
    ranks = np.arange(sorted_sources.size, dtype=np.int64) - np.repeat(starts, counts)
    positions = cursor[sorted_sources] + ranks
    indices[positions] = targets[order]
    fact_ids[positions] = facts[order]
    edge_type_ids[positions] = edge_type_id
    cursor += np.bincount(sources, minlength=cursor.size)


def build_csr_index(
    partitions: Iterable[pl.DataFrame | str | Path],
    *,
    node_count: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build a memory-mappable global CSR traversal index.

    Undirected canonical facts create two traversal entries sharing one fact ID.
    """

    items = list(partitions)
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    edge_types = sorted(
        {
            (str(row[0]), str(row[1]), str(row[2]))
            for item in items
            for frame in [_load_facts(item)]
            for row in frame.select(
                "source_node_type", "relation", "target_node_type"
            ).unique().iter_rows()
        }
    )
    edge_type_lookup = {value: index for index, value in enumerate(edge_types)}
    if len(edge_types) >= np.iinfo(np.uint16).max:
        raise ValueError("Too many typed relations for UInt16 edge type IDs")

    degree = np.zeros(node_count, dtype=np.int64)
    canonical_fact_references = 0
    for item in items:
        frame = _load_facts(item)
        sources = frame["source_global_id"].to_numpy().astype(np.int64, copy=False)
        targets = frame["target_global_id"].to_numpy().astype(np.int64, copy=False)
        if (sources < 0).any() or (targets < 0).any():
            raise ValueError("Negative global node IDs are invalid")
        if (sources >= node_count).any() or (targets >= node_count).any():
            raise ValueError("Global node ID exceeds node_count")
        degree += np.bincount(sources, minlength=node_count)
        undirected = frame["undirected"].to_numpy()
        degree += np.bincount(targets[undirected], minlength=node_count)
        canonical_fact_references += frame.height

    indptr_values = np.empty(node_count + 1, dtype=np.int64)
    indptr_values[0] = 0
    np.cumsum(degree, out=indptr_values[1:])
    traversal_count = int(indptr_values[-1])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    indptr = np.lib.format.open_memmap(
        destination / "indptr.npy", mode="w+", dtype=np.int64, shape=(node_count + 1,)
    )
    indices = np.lib.format.open_memmap(
        destination / "indices.npy", mode="w+", dtype=np.uint32, shape=(traversal_count,)
    )
    fact_ids = np.lib.format.open_memmap(
        destination / "fact_ids.npy", mode="w+", dtype=np.uint64, shape=(traversal_count,)
    )
    edge_type_ids = np.lib.format.open_memmap(
        destination / "edge_type_ids.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(traversal_count,),
    )
    indptr[:] = indptr_values
    cursor = indptr_values[:-1].copy()
    for item in items:
        frame = _load_facts(item)
        source = frame["source_global_id"].to_numpy().astype(np.int64, copy=False)
        target = frame["target_global_id"].to_numpy().astype(np.int64, copy=False)
        facts = frame["canonical_fact_id"].to_numpy().astype(np.uint64, copy=False)
        undirected = frame["undirected"].to_numpy()
        for relation_row in _ordered_relation_groups(frame, item):
            mask = (
                (frame["source_node_type"] == relation_row["source_node_type"])
                & (frame["relation"] == relation_row["relation"])
                & (frame["target_node_type"] == relation_row["target_node_type"])
            ).to_numpy()
            edge_type = edge_type_lookup[
                (
                    str(relation_row["source_node_type"]),
                    str(relation_row["relation"]),
                    str(relation_row["target_node_type"]),
                )
            ]
            _scatter_edges(
                sources=source[mask],
                targets=target[mask],
                facts=facts[mask],
                edge_type_id=edge_type,
                cursor=cursor,
                indices=indices,
                fact_ids=fact_ids,
                edge_type_ids=edge_type_ids,
            )
            reverse = mask & undirected
            _scatter_edges(
                sources=target[reverse],
                targets=source[reverse],
                facts=facts[reverse],
                edge_type_id=edge_type,
                cursor=cursor,
                indices=indices,
                fact_ids=fact_ids,
                edge_type_ids=edge_type_ids,
            )
    if not np.array_equal(cursor, indptr_values[1:]):
        raise RuntimeError("CSR fill cursor does not match the counted degree")
    for array in (indptr, indices, fact_ids, edge_type_ids):
        array.flush()
    (destination / "relation_order.json").write_text(
        json.dumps({"policy": ORDER_POLICY,
                    "recipe": "evi_hgt.preprocessing.raw_source.csr_order_recipe",
                    "unlisted_relation_order": "lexical after listed relation tuples"}, indent=2) + "\n",
        encoding="utf-8",
    )

    catalog = [
        {
            "edge_type_id": index,
            "source_node_type": value[0],
            "relation": value[1],
            "target_node_type": value[2],
        }
        for index, value in enumerate(edge_types)
    ]
    (destination / "edge_type_catalog.json").write_text(
        json.dumps(catalog, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "format": "EviHGT-Phase3-CSR-v1",
        "node_count": node_count,
        "canonical_fact_reference_count": canonical_fact_references,
        "traversal_edge_count": traversal_count,
        "edge_type_count": len(edge_types),
        "undirected_representation": "two traversal entries sharing one canonical_fact_id",
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
