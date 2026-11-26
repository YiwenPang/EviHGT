"""Prepare one explicitly synthetic fold usable by every comparison method."""

from pathlib import Path

import numpy as np
import polars as pl

from evi_hgt.data.common import sha256
from evi_hgt.data.corpus import SCHEMA
from evi_hgt.data.pretrained import ENCODER_SPECS, NODE_TYPES
from evi_hgt.data.raw_features import DISEASE_DIM, DRUG_DIM
from evi_hgt.io import write_json


def create_comparison_smoke_corpus(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)

    def parquet(name, rows, **kwargs):
        pl.DataFrame(rows, **kwargs).write_parquet(root / name)
        return name

    def array(name, values):
        np.save(root / name, values, allow_pickle=False)
        return name

    def document(name, value):
        write_json(root / name, value)
        return name

    disease_ids = [f"S{i}" for i in range(5)]
    drug_ids = [f"D{i}" for i in range(4)]
    typed = {kind: [kind + "0"] for kind in NODE_TYPES}
    typed.update(DIS=disease_ids, DRG=drug_ids)
    definitions = [(kind, node, local) for kind in NODE_TYPES for local, node in enumerate(typed[kind])]
    nodes = [dict(node_type=kind, node_id=node, type_local_id=local, global_node_id=i)
             for i, (kind, node, local) in enumerate(definitions)]
    identities = {n["node_id"]: (n["node_type"], n["global_node_id"]) for n in nodes}
    train_support = [("D0", disease) for disease in disease_ids[:-1]]
    train_contra = [("D1", disease) for disease in disease_ids[:-1]]
    specs = [(node, "GEN0", "CONTEXT") for kind, node, _ in definitions if kind not in ("DIS", "GEN")]
    specs.extend((disease, "GEN0", "ASSOCIATED_WITH") for disease in disease_ids)
    specs.extend((disease, "S4", "PARENT") for disease in disease_ids[:-1])
    specs.extend((drug, disease, "INDICATION") for drug, disease in train_support)
    specs.extend((drug, disease, "CONTRAINDICATION") for drug, disease in train_contra)
    facts, evidence, targets, traversals, source_relations = [], [], [], [], set()
    for fact_id, (source, target, relation) in enumerate(specs):
        (source_type, source_id), (target_type, target_id) = identities[source], identities[target]
        source_relations.add((source_type, relation, target_type))
        facts.append(dict(canonical_fact_id=fact_id, source_global_id=source_id, target_global_id=target_id,
            source_node_id=source, target_node_id=target, source_node_type=source_type,
            target_node_type=target_type, relation=relation, undirected=True))
        evidence.append(dict(evidence_record_id=fact_id, canonical_fact_id=fact_id,
            source_component_path="synthetic", source_row_number=fact_id,
            properties='{"evidence_count":1}', raw_direct_sources=["SYNTHETIC_TEST_ONLY"], raw_indirect_sources=[],
            overlap_category=None, edge_support_observed=False, edge_counter_evidence_observed=False))
        traversals.extend([(source_id, target_id, fact_id), (target_id, source_id, fact_id)])
        if relation in ("INDICATION", "CONTRAINDICATION"):
            targets.append(dict(drug_id=source, disease_id=target, canonical_fact_id=fact_id))
    traversals.sort()
    indptr = np.r_[0, np.cumsum(np.bincount([row[0] for row in traversals], minlength=len(nodes)))]
    semantics = dict.fromkeys(("canonical_entities_verified", "canonical_facts_deduplicated",
        "evidence_records_deduplicated", "clinical_labels_verified", "target_aliases_resolved",
        "disease_gene_label_leakage_removed", "split_mask_applied", "features_label_independent"), True)
    semantics["unknown_is_negative"] = False
    manifest = dict(schema=SCHEMA, purpose="SYNTHETIC_TEST_ONLY", training_seed=42, source_cell="R0F0",
                    token_buckets=8192, semantics=semantics)
    manifest["nodes"] = parquet("nodes.parquet", nodes)
    manifest["edge_catalog"] = document("relations.json", [
        dict(source_node_type=source, relation=relation, target_node_type=target)
        for source, relation, target in sorted(source_relations)])
    manifest["fact_files"] = [dict(path=parquet("facts.parquet", facts), component="synthetic", rows=len(facts))]
    manifest["evidence_files"] = [dict(path=parquet("evidence.parquet", evidence,
        schema_overrides={"raw_indirect_sources": pl.List(pl.String), "overlap_category": pl.String}),
        component="synthetic", rows=len(evidence))]
    manifest["csr"] = dict(indptr=array("indptr.npy", indptr),
        indices=array("indices.npy", np.array([row[1] for row in traversals], dtype=np.int64)),
        fact_ids=array("fact_ids.npy", np.array([row[2] for row in traversals], dtype=np.int64)))
    manifest["target_map"] = parquet("target_map.parquet", targets)
    manifest["forbidden_facts"] = parquet("forbidden.parquet", {"canonical_fact_id": []},
                                           schema={"canonical_fact_id": pl.Int64})
    blocked = [("D0", "S4"), ("D1", "S4"), ("D2", "S4")]
    observed = train_support + train_contra + blocked[:2]
    for key, pairs in (("blocked_pairs", blocked), ("observed_pairs", observed)):
        manifest[key] = parquet(key + ".parquet", pairs, schema=["drug_id", "disease_id"], orient="row")
    manifest["eligibility"] = document("eligibility.json", dict(eligible_drugs=drug_ids, eligible_diseases=disease_ids))
    manifest["training_pairs"] = parquet("training_pairs.parquet", [
        dict(drug_id=drug, disease_id=disease, channel=channel, raw_relation=relation)
        for channel, relation, pairs in (("support", "INDICATION", train_support),
                                        ("unsuitability", "CONTRAINDICATION", train_contra))
        for drug, disease in pairs])
    manifest["panels"] = dict(
        validation=parquet("validation.parquet", [
            dict(drug_id="D0", disease_id="S4", role="P", raw_relation="INDICATION"),
            dict(drug_id="D2", disease_id="S4", role="U", raw_relation=None)]),
        auxiliary_validation=parquet("auxiliary_validation.parquet", [
            dict(drug_id="D1", disease_id="S4", role="P", raw_relation="CONTRAINDICATION")]))
    raw, pretrained = {}, {}
    for prefix, kind, width in (("drug", "DRG", DRUG_DIM), ("disease", "DIS", DISEASE_DIM)):
        ids = np.asarray([node["global_node_id"] for node in nodes if node["node_type"] == kind])
        values = np.zeros((len(ids), width), dtype=np.uint8)
        values[:, -2:] = 1
        values[:, 0] = 1
        pretrained[prefix + "_ids"] = raw[prefix + "_ids"] = array(prefix + "_ids.npy", ids)
        raw[prefix + "_values"] = array(prefix + "_raw.npy", values)
        pretrained[prefix + "_values"] = array(prefix + "_d1.npy", np.full((len(ids), 768), .01, dtype=np.float32))
        pretrained[prefix + "_available"] = array(prefix + "_available.npy", np.ones(len(ids), dtype=np.bool_))
    pretrained["metadata"] = document("d1_metadata.json", dict(schema="EVIHGT_D1_FEATURES_V1",
        clinical_text_encoded=False, supervised_dda_features_present=False, trained_node_embeddings_loaded=False,
        prediction_scores_loaded=False, optimizer_steps=0, width=768, fixed_encoder_specs=ENCODER_SPECS,
        purpose="SYNTHETIC_TEST_ONLY"))
    manifest["raw_features"], manifest["pretrained"] = raw, pretrained
    groups, conditional = [], []
    for cycle in range(1, 4):
        for j, disease in enumerate(disease_ids[:-1]):
            step = (cycle - 1) * 4 + j + 1
            groups.extend(dict(step=step, cycle=cycle, position=i, role=role, drug_id=drug, disease_id=disease)
                          for i, (role, drug) in enumerate((("support", "D0"), ("unsuitability", "D1"), ("U", "D2"))))
            if cycle > 1:
                conditional.append(dict(logical_step=step, group_index=step - 5, original_global_u=["D2", disease],
                    original_support_u_count=1, support_positive_pairs=[["D0", disease]],
                    contra_positive_pairs=[["D1", disease]], disease_unlabeled_pairs={disease: [["D3", disease]]},
                    disease_status={disease: "DISEASE_CONDITIONAL_UNIFORM"}, draw_seed_by_disease={disease: step},
                    extra_pairs=[["D3", disease]]))
    manifest["schedule"] = dict(G=4, groups=parquet("groups.parquet", groups),
        conditional_records=document("conditional.json", conditional), sampling_provenance="SYNTHETIC_TEST_ONLY")
    manifest["files"] = {path.name: sha256(path) for path in root.iterdir() if path.is_file()}
    return root / document("manifest.json", manifest)
