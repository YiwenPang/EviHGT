"""Small, offline raw-layout fixture for all four cold workflows."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl

from evi_hgt.data.common import sha256
from evi_hgt.synthetic import make_corpus
from .cold import TEST_SCOPE, archived_scripts, prepare_cold_case


def _cohorts(prep, prefix, axis, counts):
    groups = {p: [] for p in prep.PARTS}
    index = 0
    while any(len(values) < counts for values in groups.values()):
        name = f"{prefix}{index}"
        index += 1
        group = groups[prep.entity_partition(name, axis)]
        if len(group) < counts:
            group.append(name)
    return groups


def _fixture(root):
    prep, _ = archived_scripts()
    base = make_corpus(root / "features").parent
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    nodes = pl.read_parquet(base / manifest["nodes"])
    drugs = _cohorts(prep, "tiny_drug_", "drug_id", 3)
    diseases = {p: values[0] for p, values in _cohorts(prep, "tiny_disease_", "disease_id", 1).items()}
    all_drugs = [drug for values in drugs.values() for drug in values]
    replace = dict(zip(("S0", "S1", "S2", "D0", "D1", "D2", "D3"),
                       (*diseases.values(), *all_drugs[:4]), strict=True))
    extras = len(all_drugs) - 4
    new_ids = list(range(nodes.height, nodes.height + extras))
    nodes = pl.concat([nodes.with_columns(pl.col("node_id").replace(replace)),
                      pl.DataFrame(dict(global_node_id=new_ids, node_id=all_drugs[4:],
                                        node_type=["DRG"] * extras))])
    canonical = root / "canonical"
    for sub in ("nodes", "graph/csr", "views", "canonical_facts", "evidence_records"):
        (canonical / sub).mkdir(parents=True)
    nodes.write_parquet(canonical / "nodes/node_index.parquet")
    catalog = json.loads((base / manifest["edge_catalog"]).read_text(encoding="utf-8"))
    target = pl.read_parquet(base / manifest["target_map"]).with_columns(
        pl.col("drug_id").replace(replace), pl.col("disease_id").replace(replace))
    target.write_parquet(canonical / "views/query_target_fact_map.parquet")
    facts = pl.read_parquet(base / manifest["fact_files"][0]["path"]).with_columns(
        pl.col("source_node_id").replace(replace), pl.col("target_node_id").replace(replace))
    evidence = pl.read_parquet(base / manifest["evidence_files"][0]["path"])
    node_id = dict(nodes.select("node_id", "global_node_id").iter_rows())
    # Cover native HGTDR's incoming-relation requirement for all ten node types.
    synthetic_edges = []
    for name, kind in nodes.select("node_id", "node_type").iter_rows():
        if kind in {"ANA", "DRG", "DIS", "GEN"}:
            continue
        catalog.append(dict(source_node_type=kind, relation="SYNTH_LINK", target_node_type="GEN"))
        edge_id = facts.height + len(synthetic_edges)
        synthetic_edges.append(dict(canonical_fact_id=edge_id, source_global_id=node_id[name],
            target_global_id=node_id["G0"], source_node_id=name, target_node_id="G0",
            source_node_type=kind, target_node_type="GEN", relation="SYNTH_LINK", undirected=True))
    regional_evidence = [dict(evidence.row(0, named=True), canonical_fact_id=row["canonical_fact_id"],
                            evidence_record_id=row["canonical_fact_id"], edge_counter_evidence_observed=False)
                         for row in synthetic_edges]
    facts = pl.concat([facts, pl.DataFrame(synthetic_edges, schema=facts.schema)])
    evidence = pl.concat([evidence, pl.DataFrame(regional_evidence, schema=evidence.schema)])
    (canonical / "graph/csr/edge_type_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    arcs = sorted((src, dst, fact) for a, b, fact in
                  facts.select("source_global_id", "target_global_id", "canonical_fact_id").iter_rows()
                  for src, dst in ((a, b), (b, a)))
    sources, targets, fact_ids = np.asarray(arcs, dtype=np.int64).T
    indptr = np.r_[0, np.cumsum(np.bincount(sources, minlength=nodes.height))]
    for key, values in (("indptr", indptr), ("indices", targets), ("fact_ids", fact_ids)):
        np.save(canonical / "graph/csr" / f"{key}.npy", values, allow_pickle=False)
    dda = pl.col("relation").is_in(["INDICATION", "CONTRAINDICATION"])
    dg = pl.col("relation") == "ASSOCIATED"
    for component, mask in (("disease_gene", dg), ("drug_disease", dda), ("remaining", ~(dg | dda))):
        selected = facts.filter(mask)
        selected.write_parquet(canonical / "canonical_facts" / f"{component}.parquet")
        evidence.join(selected.select("canonical_fact_id"), on="canonical_fact_id", how="semi").write_parquet(
            canonical / "evidence_records" / f"{component}.parquet")
    features = {section: {key: base / path for key, path in manifest[section].items()}
                for section in ("raw_features", "pretrained")}
    drug_ids = features["raw_features"]["drug_ids"]
    np.save(drug_ids, np.r_[np.load(drug_ids, allow_pickle=False), new_ids], allow_pickle=False)
    for path in (features["raw_features"]["drug_values"], features["pretrained"]["drug_values"],
                 features["pretrained"]["drug_available"]):
        values = np.load(path, allow_pickle=False)
        np.save(path, np.concatenate([values, np.repeat(values[:1], extras, axis=0)]), allow_pickle=False)
    aggregate = root / "nodes.parquet"
    years = dict(zip(prep.PARTS, (1980, 2005, 2020), strict=True))
    pl.DataFrame([(drug, "DRG", json.dumps(dict(year_of_first_approval=years[p])))
                  for p, values in drugs.items() for drug in values],
                 schema=["id", "label", "properties"], orient="row").write_parquet(aggregate)
    shared = dict(canonical_root=canonical, features=features, nodes=aggregate,
                  clean_dg=dict(facts=canonical / "canonical_facts/disease_gene.parquet",
                                evidence=canonical / "evidence_records/disease_gene.parquet"),
                  source_identity=dict(purpose=TEST_SCOPE),
                  file_hashes={path: sha256(path) for path in root.rglob("*") if path.is_file()})
    support = [(values[0], diseases[p]) for p, values in drugs.items()]
    contra = [(values[1], diseases[p]) for p, values in drugs.items()]
    source = pl.DataFrame([(a, b, relation) for relation, pairs in
                           (("INDICATION", support), ("CONTRAINDICATION", contra)) for a, b in pairs],
                          schema=["drug_id", "disease_id", "raw_relation"], orient="row")
    pools = dict(source=source, panel=[dict(drug_id=a, disease_id=b, role="P") for a, b in support],
                 auxiliary=contra, frozen=set(), historical=set(), observed=set(support + contra),
                 eligible_drugs=all_drugs, eligible_diseases=list(diseases.values()))
    return shared, pools


def prepare_cold_smoke(case, destination, *, audit_dir, progress=None):
    """Exercise the real cold preparation logic with one update per cycle."""
    destination = Path(destination)
    fixture = destination.parent / f"cold_fixture_{uuid4().hex}"
    shared, pools = _fixture(fixture)
    return prepare_cold_case(case, shared, pools, destination, audit_dir=audit_dir, progress=progress)
