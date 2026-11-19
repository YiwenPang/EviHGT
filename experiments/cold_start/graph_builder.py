"""Build one isolated cold-start graph for the frozen EviHGT portable loader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import polars as pl
import pyarrow.parquet as pq

if __package__:
    from . import locations
else:
    import locations


_ID = "canonical_fact_id"
_ENDPOINTS = ("source_node_type", "source_node_id", "target_node_type", "target_node_id")


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _verified_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    candidate = {k: v for k, v in value.items() if k != "digest"}
    identity = hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if value.get("digest") != identity:
        raise ValueError(f"Audited source manifest identity changed: {path}")
    return value


def _pair_frame(pairs, drug="drug_id", disease="disease_id") -> pl.DataFrame:
    return pl.DataFrame(sorted(pairs), schema={drug: pl.String, disease: pl.String}, orient="row")


def _member(name, members, dtype=pl.String):
    if isinstance(members, pl.Series):
        if len(members) == 0:
            return pl.lit(False)
        return pl.col(name).is_in(members.implode())
    if not members:
        return pl.lit(False)
    return pl.col(name).is_in(pl.Series(sorted(members), dtype=dtype).implode())


def _cold_incidence(heldout_drugs, heldout_diseases):
    return pl.any_horizontal(*(
        (pl.col(f"{side}_node_type") == kind) & _member(f"{side}_node_id", ids)
        for side in ("source", "target")
        for kind, ids in (("DRG", heldout_drugs), ("DIS", heldout_diseases))
    ))


def _with_direct_pair(frame):
    return frame.with_columns(
        pl.when(pl.col("source_node_type") == "DRG").then(pl.col("source_node_id"))
          .when(pl.col("target_node_type") == "DRG").then(pl.col("target_node_id"))
          .otherwise(None).alias("_drug"),
        pl.when(pl.col("source_node_type") == "DIS").then(pl.col("source_node_id"))
          .when(pl.col("target_node_type") == "DIS").then(pl.col("target_node_id"))
          .otherwise(None).alias("_disease"),
    )


def filter_fact_identities(frame, heldout_drugs, heldout_diseases, blocked_pairs, forbidden_ids):
    """Filter typed endpoints in both directions and reject canonical aliases."""
    id_type = frame.collect_schema()[_ID]
    retained = frame.filter(
        ~_cold_incidence(heldout_drugs, heldout_diseases) & ~_member(_ID, forbidden_ids, id_type)
    )
    if blocked_pairs:
        retained = (_with_direct_pair(retained)
            .join(_pair_frame(blocked_pairs, "_drug", "_disease").lazy(),
                  on=["_drug", "_disease"], how="anti", maintain_order="left")
            .drop("_drug", "_disease"))
    return retained


def _contains_sorted(haystack, values):
    positions = np.searchsorted(haystack, values)
    result = positions < len(haystack)
    result[result] &= haystack[positions[result]] == values[result]
    return result


def mask_csr(csr, admitted, heldout_ids):
    """Keep the original adjacency order and remove all forbidden incidences."""
    ptr, neighbors, fact_ids = (csr[key] for key in ("indptr", "indices", "fact_ids"))
    n = len(ptr) - 1
    if (ptr.ndim != 1 or int(ptr[0]) != 0 or int(ptr[-1]) != len(neighbors)
            or len(neighbors) != len(fact_ids) or np.any(ptr[1:] < ptr[:-1])):
        raise ValueError("Invalid certified source CSR")
    if (len(neighbors) and (int(neighbors.min()) < 0 or int(neighbors.max()) >= n)):
        raise ValueError("Source CSR has unregistered node IDs")
    if len(admitted) > 1 and np.any(admitted[1:] <= admitted[:-1]):
        raise ValueError("Admitted fact IDs must be globally unique and sorted")
    if len(heldout_ids) and (int(heldout_ids.min()) < 0 or int(heldout_ids.max()) >= n):
        raise ValueError("Held-out endpoint has no canonical global row")
    heldout = np.zeros(n, dtype=np.bool_)
    heldout[heldout_ids] = True
    keep = np.empty(len(fact_ids), dtype=np.bool_)
    upper = int(admitted[-1]) if len(admitted) else -1
    dense = upper < 100_000_000 and upper < 8 * len(admitted)
    if dense:
        membership = np.zeros(upper + 1, dtype=np.bool_)
        membership[admitted] = True
    for start in range(0, len(keep), 2_000_000):
        end = min(start+2_000_000, len(keep))
        batch = fact_ids[start:end]
        if dense:
            current = batch <= upper
            current[current] &= membership[batch[current]]
        else:
            current = _contains_sorted(admitted, batch)
        keep[start:end] = current & ~heldout[neighbors[start:end]]
    for i in heldout_ids:
        keep[ptr[i]:ptr[i+1]] = False

    # Prefix counts only by CSR row, so no full-length int64 prefix allocation.
    lengths = np.diff(ptr)
    active = lengths != 0
    counts = np.zeros(n, dtype=np.int64)
    if len(keep):
        counts[active] = np.add.reduceat(keep, ptr[:-1][active], dtype=np.int64)
    result_ptr = np.empty(n+1, dtype=np.int64)
    result_ptr[0] = 0
    np.cumsum(counts, out=result_ptr[1:])
    retained_facts = fact_ids[keep]
    if dense:
        # Sanity-check that no canonical fact was silently orphaned by CSR.
        seen = np.zeros(upper+1, dtype=np.bool_)
        seen[retained_facts] = True
        if not np.array_equal(seen, membership):
            raise ValueError("Admitted canonical facts are missing from the certified traversal index")
    elif not np.array_equal(np.unique(retained_facts), admitted):
        raise ValueError("Admitted canonical facts are missing from the certified traversal index")
    result_neighbors = neighbors[keep]
    if np.any(counts[heldout]) or np.any(heldout[result_neighbors]):
        raise ValueError("Held-out endpoints remain incident to the materialized CSR")
    return {"indptr": result_ptr, "indices": result_neighbors, "fact_ids": retained_facts}


def build_graph(workspace: Path, dest: Path, heldout_drugs: set[str],
                heldout_diseases: set[str], blocked_pairs: set[tuple[str, str]]) -> dict:
    """Return graph fields to merge into a portable manifest at case dest."""
    workspace, dest = Path(workspace).resolve(), Path(dest).resolve()
    cache = (workspace / locations.CASE_CACHE).resolve()
    if not cache.is_relative_to(workspace) or not dest.is_relative_to(cache) or dest == cache:
        raise ValueError("Cold-start graph output must be below the new experiment cache")
    if not heldout_drugs and not heldout_diseases:
        raise ValueError("At least one cold endpoint axis must be held out")
    if not blocked_pairs:
        raise ValueError("Cold-start validation and test memberships must be specified")
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "graph"
    if final.exists():
        raise FileExistsError(final)
    stage = dest / ("graph.attempt_" + uuid.uuid4().hex)
    stage.mkdir()

    canonical = workspace / locations.CANONICAL_GRAPH
    development = workspace / locations.DEVELOPMENT_INPUTS
    parent = workspace / locations.DISEASE_GENE_INPUTS
    clean_dir = parent / "LEAKAGE_SAFE_OT_DISEASE_GENE_V1"
    pilot_dir = parent / "pilot_adapter_v1"
    clean = _verified_json(clean_dir / "manifest.json")
    accept = _verified_json(workspace / locations.GRAPH_ACCEPTANCE)
    pilot = _verified_json(pilot_dir / "manifest.json")
    dev = _verified_json(development / "development_manifest.json")
    if (accept["graph_provenance"] != "PASS" or accept["topology"] != "PASS"
            or accept["clean_graph_digest"] != clean["digest"]
            or pilot["contaminated_DG_evidence_imported"] is not False
            or pilot["heldout_DDA_properties_imported"] is not False
            or pilot["CTD_used"] is not False):
        raise ValueError("Certified leakage-safe parent graph gate failed")
    source_hashes = {Path(p).resolve(): h
                     for group in (clean["files"], pilot["inputs"], pilot["outputs"])
                     for p, h in group.items()}
    source_hashes[development / "legal_source_rows.parquet"] = dev["files"]["legal_source_rows.parquet"]
    source_checks = {}
    files = {}

    def verified(path):
        path = path.resolve()
        expected = source_hashes.get(path)
        if not path.is_relative_to(workspace) or expected is None:
            raise ValueError(f"Source not in the graph's certified inventory: {path}")
        if path not in source_checks:
            observed = _sha(path)
            if observed != expected:
                raise ValueError(f"Certified graph source hash changed: {path}")
            source_checks[path] = observed
        return path

    def write(name, writer):
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)
        relative = (Path("graph") / name).as_posix()
        files[relative] = _sha(path)
        return relative

    def copy(name, source):
        source = verified(source)
        relative = write(name, lambda path: shutil.copyfile(source, path))
        if files[relative] != source_checks[source]:
            raise ValueError(f"Copied graph source changed during materialization: {source}")
        return relative

    node_source = verified(canonical / "nodes/node_index.parquet")
    nodes = pl.read_parquet(node_source, columns=["global_node_id", "node_type", "node_id"])
    cold_nodes = nodes.filter(((pl.col("node_type") == "DRG") & _member("node_id", heldout_drugs))
                             | ((pl.col("node_type") == "DIS") & _member("node_id", heldout_diseases)))
    if cold_nodes.height != len(heldout_drugs)+len(heldout_diseases):
        raise ValueError("Unknown or duplicate held-out canonical endpoints")
    node_path = copy("nodes.parquet", node_source)
    edge_path = copy("edge_catalog.json", canonical / "graph/csr/edge_type_catalog.json")
    target_source = verified(canonical / "views/query_target_fact_map.parquet")
    target_map = pl.read_parquet(target_source, columns=["drug_id", "disease_id", _ID])
    blocked = _pair_frame(blocked_pairs)
    forbidden = target_map.join(blocked, on=["drug_id", "disease_id"], how="semi").get_column(_ID).unique().sort()
    forbidden_path = write("forbidden_facts.parquet", pl.DataFrame({_ID: forbidden}).write_parquet)
    legal = pl.read_parquet(verified(development / "legal_source_rows.parquet"), columns=["drug_id","disease_id"])
    legal = legal.rename({"drug_id": "_drug", "disease_id": "_disease"})
    fact_files, evidence_files, retained_id_parts = [], [], []
    components, evidence_sources = {}, {}

    for original in sorted((canonical / "canonical_facts").glob("*.parquet")):
        component = original.stem
        facts_source = clean_dir / "canonical_facts.parquet" if component == "disease_gene" else original
        evidence_source = (pilot_dir if component == "disease_gene" else canonical) / "evidence_records" / original.name
        facts_source, evidence_source = verified(facts_source), verified(evidence_source)
        fact_count = pq.ParquetFile(facts_source).metadata.num_rows
        evidence_count = pq.ParquetFile(evidence_source).metadata.num_rows
        identities = pl.scan_parquet(facts_source).select([_ID, *_ENDPOINTS])
        admitted = filter_fact_identities(identities, heldout_drugs, heldout_diseases,
                                          blocked_pairs, forbidden)
        if component == "drug_disease":
            admitted = _with_direct_pair(admitted).join(legal.lazy(), on=["_drug", "_disease"],
                                                       how="semi", maintain_order="left").drop("_drug", "_disease")
        identity = admitted.collect()
        if identity.filter(_cold_incidence(heldout_drugs, heldout_diseases)).height:
            raise ValueError(f"Held-out endpoints survived the canonical fact mask: {component}")
        ids = identity.get_column(_ID)
        direct = identity.filter(
            ((pl.col("source_node_type") == "DRG") & (pl.col("target_node_type") == "DIS"))
            | ((pl.col("source_node_type") == "DIS") & (pl.col("target_node_type") == "DRG")))
        if direct.height:
            direct = _with_direct_pair(direct).select(_ID, pl.col("_drug").alias("drug_id"),
                                                       pl.col("_disease").alias("disease_id"))
            if direct.join(blocked, on=["drug_id", "disease_id"], how="semi").height:
                raise ValueError("Blocked drug鈥揹isease endpoints survived the canonical fact mask")
            if direct.join(target_map, on=[_ID, "drug_id", "disease_id"], how="anti").height:
                raise ValueError("Unregistered direct or reverse drug鈥揹isease fact")
        name = f"canonical_facts/{original.name}"
        evidence_name = f"evidence_records/{original.name}"
        if len(ids) == fact_count:
            fp = copy(name, facts_source)
        else:
            selected = ids.to_frame().lazy()
            filtered_facts = pl.scan_parquet(facts_source).join(selected, on=_ID, how="semi",
                                                               maintain_order="left")
            fp = write(name, filtered_facts.sink_parquet)
        retained_id_parts.append(ids.to_numpy())
        fact_files.append(dict(path=fp, component=component, rows=len(ids)))
        components[component] = dict(facts_before=fact_count, facts_after=len(ids),
            evidence_before=evidence_count)
        evidence_sources[component] = evidence_name, evidence_source

    admitted_ids = np.concatenate(retained_id_parts)
    admitted_ids.sort()
    if np.any(admitted_ids[1:] == admitted_ids[:-1]) or np.any(_contains_sorted(admitted_ids, forbidden.to_numpy())):
        raise ValueError("Duplicate or forbidden canonical IDs after masking")
    # Source adapter names identify evidence provenance, not owning canonical
    # facts: DrugBank evidence can corroborate a fact owned by OptimusKG.
    evidence_seen = np.zeros(len(admitted_ids), dtype=np.bool_)
    retained_id_parts.clear()
    for component, (evidence_name, evidence_source) in evidence_sources.items():
        source = pl.scan_parquet(evidence_source)
        ids = source.select(_ID).unique().collect().get_column(_ID)
        admitted = _contains_sorted(admitted_ids, ids.to_numpy())
        if admitted.all():
            ep = copy(evidence_name, evidence_source)
        else:
            selected = ids.filter(pl.Series(admitted)).to_frame().lazy()
            filtered = source.join(selected, on=_ID, how="semi", maintain_order="left")
            ep = write(evidence_name, filtered.sink_parquet)
        ep_file = stage / evidence_name
        ep_rows = pq.ParquetFile(ep_file).metadata.num_rows
        ep_ids = pl.scan_parquet(ep_file).select(_ID).unique().collect().get_column(_ID).to_numpy()
        if not _contains_sorted(admitted_ids, ep_ids).all():
            raise ValueError(f"Evidence with inadmissible canonical ID survived masking: {component}")
        evidence_seen[np.searchsorted(admitted_ids, ep_ids)] = True
        evidence_files.append(dict(path=ep, component=component, rows=ep_rows))
        components[component]["evidence_after"] = ep_rows
    if not evidence_seen.all():
        raise ValueError("At least one admitted canonical fact is missing evidence after masking")
    masked = mask_csr({key: np.load(verified(canonical / "graph/csr" / f"{key}.npy"),
                                    mmap_mode="r", allow_pickle=False)
                       for key in ("indptr", "indices", "fact_ids")},
                      admitted_ids, cold_nodes["global_node_id"].to_numpy())
    if len(masked["indptr"]) != nodes.height+1:
        raise ValueError("CSR global row count does not match the canonical node index")
    csr = {k: write(f"csr/{k}.npy", lambda path, values=v: np.save(path, values, allow_pickle=False))
           for k, v in masked.items()}
    retained_map = target_map.filter(_member(_ID, pl.Series(_ID, admitted_ids)))
    target_map_path = write("target_map.parquet", retained_map.write_parquet)
    stage.rename(final)
    return dict(nodes=node_path, edge_catalog=edge_path, fact_files=fact_files, evidence_files=evidence_files,
        csr=csr, forbidden_facts=forbidden_path, target_map=target_map_path, files=files,
        graph_audit=dict(graph_mask="STRICT_ENTITY_INCIDENCE_ALL_DOMAINS",
            source_clean_graph_digest=clean["digest"], source_graph_acceptance_digest=accept["digest"],
            source_pilot_inventory_digest=pilot["digest"], source_development_inventory_digest=dev["digest"],
            checked_sources={path.relative_to(workspace).as_posix(): value for path,value in source_checks.items()},
            heldout_drugs=len(heldout_drugs), heldout_diseases=len(heldout_diseases),
            heldout_facts_remaining=0, heldout_csr_rows_with_edges=0,
            heldout_csr_neighbors_remaining=0, evidence_without_admitted_fact=0,
            blocked_pairs=len(blocked_pairs), forbidden_equivalent_facts=len(forbidden),
            forbidden_equivalent_facts_remaining=0, traversals=len(masked["fact_ids"]),
            canonical_facts=len(admitted_ids), surviving_dda_target_facts=len(retained_map),
            components=components, temporal_as_of_graph=False))
