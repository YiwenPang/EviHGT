"""Bounded native-array traversal with the scalar CSR extractor's exact order.

The scalar oracle in ``data.leakage`` intentionally remains independent. This
path vectorizes physical edge blocks, retaining first-encounter node admission
and every accepted duplicate/reverse edge. It does not cache query results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from evi_hgt.data.leakage import ExtractedSubgraph


# Process high-degree rows in bounded blocks.
_EDGE_BLOCK_SIZE = 32_768


def extract_csr_subgraph_native(
    csr_dir: str | Path,
    *,
    seed_node_ids: list[int],
    max_hops: int,
    max_nodes: int,
    max_edges: int,
    forbidden_fact_ids: set[int] | frozenset[int] | None = None,
    forbidden_filter: Callable[[np.ndarray], np.ndarray] | None = None,
    csr_arrays: Mapping[str, np.ndarray] | None = None,
) -> ExtractedSubgraph:
    """Return exactly the scalar oracle's nodes, facts and physical traversals.

    ``forbidden_filter(ids)`` returns a boolean array of the same shape, True
    for hidden facts. Its exclusions are unioned with ``forbidden_fact_ids``.
    The indexed assembler uses this vector API for split/target masks instead
    of Python membership. ``csr_arrays`` may supply already-open read-only
    ``indptr``, ``indices`` and ``fact_ids`` arrays from the static index.

    Scratch consists of a node-count boolean visited bitmap, bounded edge
    blocks, and accepted output bounded by max_edges. No graph-sized Python
    objects, persistent state, or query cache are created.
    """
    if max_hops < 0 or max_nodes <= 0 or max_edges <= 0:
        raise ValueError("Invalid bounded-subgraph limits")
    if csr_arrays is None:
        root = Path(csr_dir)
        csr_arrays = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in ("indptr", "indices", "fact_ids")
        }
    indptr, indices, facts = (
        np.asarray(csr_arrays[name]) for name in ("indptr", "indices", "fact_ids")
    )
    if any(array.ndim != 1 or array.dtype.kind not in "iu" for array in (indptr, indices, facts)):
        raise ValueError("CSR arrays must be one-dimensional integer arrays")
    if not len(indptr) or len(indices) != len(facts) or int(indptr[0]) != 0 or int(indptr[-1]) != len(indices):
        raise ValueError("CSR array lengths or terminal offsets are invalid")
    node_count = len(indptr) - 1
    seeds = sorted(set(int(value) for value in seed_node_ids))
    if any(value < 0 or value >= node_count for value in seeds):
        raise ValueError("Seed node ID is outside the CSR index")
    if forbidden_fact_ids is not None and type(forbidden_fact_ids) not in (set, frozenset):
        raise TypeError("Use forbidden_filter for vectorized index membership")
    fact_range = np.iinfo(facts.dtype)
    # Avoid NumPy inferring float64 from mixed negative/large uint64 IDs.
    # IDs outside the physical fact dtype cannot match any stored fact.
    forbidden = np.fromiter(
        (value for value in (forbidden_fact_ids or ()) if fact_range.min <= value <= fact_range.max),
        dtype=facts.dtype,
    )
    visited = np.zeros(node_count, dtype=np.bool_)
    visited[seeds] = True
    visited_count = len(seeds)
    frontier = seeds
    accepted_count = 0
    blocks: list[tuple[int, np.ndarray, np.ndarray]] = []

    for _ in range(max_hops):
        next_frontier: list[int] = []
        sources = np.asarray(frontier, dtype=np.int64)
        starts, stops = indptr[sources], indptr[sources + 1]
        if np.any(starts > stops) or np.any(stops > len(indices)):
            raise ValueError("CSR row offsets are invalid")
        for source, row_start, row_stop in zip(frontier, starts.tolist(), stops.tolist()):
            for start in range(row_start, row_stop, _EDGE_BLOCK_SIZE):
                stop = min(row_stop, start + _EDGE_BLOCK_SIZE)
                row_facts = facts[start:stop]
                allowed = np.ones(row_facts.size, dtype=np.bool_)
                if forbidden.size:
                    allowed &= ~np.isin(row_facts, forbidden)
                if forbidden_filter is not None:
                    hidden = np.asarray(forbidden_filter(row_facts))
                    if hidden.shape != row_facts.shape or hidden.dtype != np.dtype(np.bool_):
                        raise ValueError("forbidden_filter must return a matching boolean array")
                    allowed &= ~hidden
                if not allowed.any():
                    continue
                targets = indices[start:stop][allowed]
                row_facts = row_facts[allowed]
                if np.any(targets < 0) or np.any(targets >= node_count):
                    raise ValueError("CSR target node ID is outside the index")
                known = visited[targets]
                room = max(0, max_nodes - visited_count)
                if not known.all():
                    if room:
                        # np.unique sorts IDs; sorting its first-occurrence
                        # positions restores physical first-encounter order.
                        unseen = targets[~known]
                        _, first = np.unique(unseen, return_index=True)
                        admitted = unseen[np.sort(first)[:room]]
                        allowed_nodes = known | np.isin(targets, admitted)
                    else:
                        # The node cap does not stop a row: later edges to
                        # already visited targets are still admitted.
                        allowed_nodes = known
                    targets, row_facts = targets[allowed_nodes], row_facts[allowed_nodes]
                remaining = max_edges - accepted_count
                targets, row_facts = targets[:remaining], row_facts[:remaining]
                if not targets.size:
                    continue
                # Only targets surviving the edge cap may become visited.
                new_nodes = np.unique(targets[~visited[targets]])
                visited[new_nodes] = True
                visited_count += new_nodes.size
                next_frontier.extend(new_nodes.tolist())
                blocks.append((source, targets, row_facts))
                accepted_count += targets.size
                if accepted_count >= max_edges:
                    break
            if accepted_count >= max_edges:
                break
        frontier = sorted(set(next_frontier))
        if not frontier or accepted_count >= max_edges:
            break

    traversal = tuple(
        (source, target, fact)
        for source, targets, row_facts in blocks
        for target, fact in zip(targets.tolist(), row_facts.tolist())
    )
    if blocks:
        observed = np.concatenate([row_facts for _, _, row_facts in blocks])
        _, first = np.unique(observed, return_index=True)
        ordered_facts = tuple(observed[np.sort(first)].tolist())
    else:
        ordered_facts = ()
    return ExtractedSubgraph(
        node_ids=tuple(np.flatnonzero(visited).tolist()),
        fact_ids=ordered_facts,
        traversal_edges=traversal,
    )
