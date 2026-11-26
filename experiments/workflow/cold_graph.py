"""Build a physically isolated cold-case graph from verified raw preparation inputs."""

from pathlib import Path
import shutil

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from experiments.cold_start.graph_builder import (
    _cold_incidence, _contains_sorted, _member, _with_direct_pair, filter_fact_identities, mask_csr,
)
from evi_hgt.data.common import sha256

FACT_ID = "canonical_fact_id"
PAIR = ("drug_id", "disease_id")
ENDPOINTS = ("source_node_type", "source_node_id", "target_node_type", "target_node_id")


def build_cold_graph(canonical, clean_facts, clean_evidence, nodes, stage, write, verified,
                     held_drugs, held_diseases, blocked, legal):
    """Keep the canonical traversal order, and bind evidence by surviving fact ID."""
    node_name = "graph/nodes.parquet"
    nodes_source = verified(canonical / "nodes/node_index.parquet")
    graph_nodes = write(node_name, lambda path: shutil.copyfile(nodes_source, path))
    catalog_source = verified(canonical / "graph/csr/edge_type_catalog.json")
    catalog = write("graph/edge_catalog.json", lambda path: shutil.copyfile(catalog_source, path))
    target = pl.read_parquet(verified(canonical / "views/query_target_fact_map.parquet"),
                             columns=[*PAIR, FACT_ID])
    blocked_frame = pair_frame(blocked)
    forbidden = target.join(blocked_frame, on=PAIR, how="semi")[FACT_ID].unique().sort()
    forbidden_path = write("graph/forbidden_facts.parquet", forbidden.to_frame().write_parquet)
    legal_frame = pair_frame(legal).rename({"drug_id": "_drug", "disease_id": "_disease"}).lazy()
    cold_mask = _cold_incidence(held_drugs, held_diseases)
    cold_nodes = nodes.filter(((pl.col("node_type") == "DRG") & _member("node_id", held_drugs))
                              | ((pl.col("node_type") == "DIS") & _member("node_id", held_diseases)))
    if cold_nodes.height != len(held_drugs) + len(held_diseases):
        raise ValueError("A held-out endpoint has no unique canonical identity")

    fact_files, evidence_files, retained, evidence_paths, component_counts = [], [], [], {}, {}
    has_clean_component = False
    for original in sorted((canonical / "canonical_facts").glob("*.parquet")):
        component = original.stem
        if component == "disease_gene":
            facts, evidence = verified(clean_facts), verified(clean_evidence)
            has_clean_component = True
        else:
            facts = verified(original)
            evidence = verified(canonical / "evidence_records" / original.name)
        identities = pl.scan_parquet(facts).select(FACT_ID, *ENDPOINTS)
        safe = filter_fact_identities(identities, held_drugs, held_diseases, blocked, forbidden)
        if component == "drug_disease":
            safe = _with_direct_pair(safe).join(legal_frame, on=["_drug", "_disease"], how="semi",
                                              maintain_order="left").drop("_drug", "_disease")
        identity = safe.collect()
        if identity.filter(cold_mask).height:
            raise ValueError(f"Held-out endpoints survived in {component}")
        direct = identity.filter(
            ((pl.col("source_node_type") == "DRG") & (pl.col("target_node_type") == "DIS"))
            | ((pl.col("source_node_type") == "DIS") & (pl.col("target_node_type") == "DRG"))
        )
        if direct.height:
            direct = _with_direct_pair(direct).select(FACT_ID, pl.col("_drug").alias("drug_id"),
                                                       pl.col("_disease").alias("disease_id"))
            if direct.join(blocked_frame, on=PAIR, how="semi").height:
                raise ValueError("A blocked drug–disease pair survived the graph mask")
            if direct.join(target, on=[FACT_ID, *PAIR], how="anti").height:
                raise ValueError("Unregistered direct or reverse drug–disease fact")
        admitted = identity[FACT_ID]
        retained.append(admitted.to_numpy())
        before = pq.ParquetFile(facts).metadata.num_rows
        fact_output = f"graph/canonical_facts/{original.name}"
        if len(admitted) == before:
            fact_output = write(fact_output, lambda path: shutil.copyfile(facts, path))
        else:
            filtered = pl.scan_parquet(facts).join(admitted.to_frame().lazy(), on=FACT_ID, how="semi",
                                                   maintain_order="left")
            fact_output = write(fact_output, filtered.sink_parquet)
        fact_files.append({"component": component, "path": fact_output, "rows": len(admitted)})
        component_counts[component] = {"facts_before": before, "facts_after": len(admitted),
                                      "evidence_before": pq.ParquetFile(evidence).metadata.num_rows}
        evidence_paths[component] = evidence

    if not has_clean_component:
        raise ValueError("The certified disease–gene component is missing")
    ids = np.sort(np.concatenate(retained))
    if (ids[1:] == ids[:-1]).any() or _contains_sorted(ids, forbidden.to_numpy()).any():
        raise ValueError("Duplicate or forbidden fact IDs in the admitted graph")
    seen = np.zeros(len(ids), dtype=np.bool_)
    for component, path in evidence_paths.items():
        name = f"graph/evidence_records/{component}.parquet"
        records = pl.scan_parquet(path)
        values = records.select(FACT_ID).unique().collect()[FACT_ID]
        keep = _contains_sorted(ids, values.to_numpy())
        if keep.all():
            name = write(name, lambda out: shutil.copyfile(path, out))
        else:
            rows = records.join(values.filter(pl.Series(keep)).to_frame().lazy(),
                                on=FACT_ID, how="semi", maintain_order="left")
            name = write(name, rows.sink_parquet)
        assigned = pl.scan_parquet(stage / name).select(FACT_ID).unique().collect()[FACT_ID].to_numpy()
        if not _contains_sorted(ids, assigned).all():
            raise ValueError("Evidence retained without its admitted canonical fact")
        seen[np.searchsorted(ids, assigned)] = True
        count = pq.ParquetFile(stage / name).metadata.num_rows
        evidence_files.append({"component": component, "path": name, "rows": count})
        component_counts[component]["evidence_after"] = count
    if not seen.all():
        raise ValueError("An admitted canonical fact is missing evidence")

    arrays = {name: np.load(verified(canonical / "graph/csr" / f"{name}.npy"),
                            mmap_mode="r", allow_pickle=False) for name in ("indptr", "indices", "fact_ids")}
    if len(arrays["indptr"]) != nodes.height + 1:
        raise ValueError("Canonical CSR row count does not match the node index")
    masked = mask_csr(arrays, ids, cold_nodes["global_node_id"].to_numpy())
    csr = {name: write(f"graph/csr/{name}.npy", lambda path, a=array: np.save(path, a, allow_pickle=False))
           for name, array in masked.items()}
    legal_target = target.filter(_member(FACT_ID, pl.Series(FACT_ID, ids)))
    target_output = write("graph/target_map.parquet", legal_target.write_parquet)
    graph = dict(nodes=graph_nodes, edge_catalog=catalog, target_map=target_output,
                 forbidden_facts=forbidden_path, fact_files=fact_files, evidence_files=evidence_files, csr=csr)
    audit = dict(graph_mask="STRICT_ENTITY_INCIDENCE_ALL_DOMAINS",
                 source_clean_graph_facts_sha256=sha256(clean_facts),
                 source_clean_graph_evidence_sha256=sha256(clean_evidence),
                 heldout_drugs=len(held_drugs), heldout_diseases=len(held_diseases),
                 heldout_facts_remaining=0, heldout_csr_rows_with_edges=0, heldout_csr_neighbors_remaining=0,
                 evidence_without_admitted_fact=0, forbidden_equivalent_facts_remaining=0,
                 blocked_pairs=len(blocked), forbidden_equivalent_facts=len(forbidden),
                 canonical_facts=len(ids), surviving_dda_target_facts=len(legal_target),
                 traversals=len(masked["fact_ids"]), components=component_counts, temporal_as_of_graph=False)
    return graph, audit


def pair_frame(pairs):
    return pl.DataFrame(sorted(pairs), schema={key: pl.String for key in PAIR}, orient="row")
