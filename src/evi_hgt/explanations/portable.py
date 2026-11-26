"""Serialize an already completed bounded query forward and its source evidence."""
from dataclasses import asdict

import torch

from evi_hgt.explanations.path_explanation import FactExplanationContext, build_path_explanations
from evi_hgt.sampling.path_schema import PathCandidate


def _forward_candidates(corpus, graph, drug_id, disease_id):
    batch = graph.batch
    model, source, packed = batch.model_batch, batch.path_source, batch.path_batch
    if source is None or packed is None:
        raise ValueError("Exact forward explanations require PathSource and an explicit packed path batch")
    source.validate()
    if source.phase3_digest != corpus.identity or source.split_digest != corpus.identity:
        raise ValueError("PathSource corpus binding differs from the explanation evidence corpus")
    if model.queries.num_queries != 1 or packed.num_queries != 1:
        raise ValueError("A query explanation requires exactly one query")
    if len(batch.supervision) != 1:
        raise ValueError("Query explanation metadata count does not match the forward")
    query = batch.supervision[0]
    if (query.drug_id, query.disease_id) != (drug_id, disease_id) or source.query_pair_id != query.canonical_pair_key:
        raise ValueError("Query explanation identity does not match the original forward")
    if len(source.facts) != model.evidence.num_facts or len(source.node_ids) != model.nodes.num_nodes:
        raise ValueError("Canonical fact or node count differs from the original forward")
    if model.nodes.node_id is None or tuple(model.nodes.node_id.detach().cpu().tolist()) != source.node_ids:
        raise ValueError("Canonical node mapping differs from the original forward")
    endpoints = (int(model.queries.drug_node_index[0]), int(model.queries.disease_node_index[0]))
    if endpoints != (source.drug_node_index, source.disease_node_index):
        raise ValueError("Query endpoint mapping differs from the original forward")
    if tuple(map(int, corpus.index.node_global_ids((drug_id, disease_id)))) != tuple(source.node_ids[n] for n in endpoints):
        raise ValueError("Canonical query IDs differ from the original forward")
    if tuple(model.evidence.fact_edge_type_id.detach().cpu().tolist()) != tuple(f.edge_type_id for f in source.facts):
        raise ValueError("Canonical fact order differs from the original forward")
    metadata = {int(row["canonical_fact_id"]): row for row in
                corpus.index.fact_metadata([fact.canonical_fact_id for fact in source.facts]).iter_rows(named=True)}
    for fact in source.facts:
        row = metadata[fact.canonical_fact_id]
        expected = (int(row["source_global_id"]), int(row["target_global_id"]), row["relation"], row["undirected"])
        observed = (source.node_ids[fact.source_node_index], source.node_ids[fact.target_node_index],
                    fact.relation, fact.undirected)
        if observed != expected:
            raise ValueError("PathSource canonical fact metadata differs from the evidence corpus")
    packed.validate(num_nodes=model.nodes.num_nodes, num_facts=model.evidence.num_facts)

    traversals = {(t.source_node_index, t.target_node_index, t.fact_index, t.direction)
                  for t in source.traversals}
    sources = packed.step_source_index.detach().cpu().tolist()
    targets = packed.step_target_index.detach().cpu().tolist()
    facts = packed.step_fact_index.detach().cpu().tolist()
    directions = packed.step_direction.detach().cpu().tolist()
    lengths = packed.path_length.detach().cpu().tolist()
    queries = packed.query_index.detach().cpu().tolist()
    candidates = []
    for row, length in enumerate(lengths):
        nodes = (sources[row][0],) + tuple(targets[row][:length])
        if tuple(sources[row][:length]) != nodes[:-1]:
            raise ValueError("Packed path is not a continuous original traversal")
        fs, ds = tuple(facts[row][:length]), tuple(directions[row][:length])
        for start, end, fact, direction in zip(sources[row][:length], targets[row][:length], fs, ds, strict=True):
            canonical = source.facts[fact]
            expected = ((canonical.source_node_index, canonical.target_node_index) if direction == 0
                        else (canonical.target_node_index, canonical.source_node_index))
            if (start, end) != expected or (direction == 1 and not canonical.undirected):
                raise ValueError("Packed path step does not match its canonical fact/direction")
            if (start, end, fact, direction) not in traversals:
                raise ValueError("Packed path step was not in the original legal path traversals")
        if nodes[0] != source.drug_node_index or nodes[-1] != source.disease_node_index:
            raise ValueError("Packed path endpoints differ from the original query")
        candidate = PathCandidate.create(query_index=queries[row], node_indices=nodes,
            node_ids=tuple(source.node_ids[n] for n in nodes),
            node_types=tuple(source.node_types[n] for n in nodes), fact_indices=fs,
            canonical_fact_ids=tuple(source.facts[f].canonical_fact_id for f in fs),
            relations=tuple(source.facts[f].relation for f in fs), directions=ds, structural_score=0.0)
        if candidate.path_id != packed.path_ids[row]:
            raise ValueError("Packed path identity differs from its original canonical fact/directions")
        candidates.append(candidate)
    return tuple(candidates)


def _fact_context(corpus, source, prediction):
    ids = [fact.canonical_fact_id for fact in source.facts]
    records = {fact_id: [] for fact_id in ids}
    for row in corpus.index.evidence_for_facts(ids).iter_rows(named=True):
        records[int(row["canonical_fact_id"])].append(row)
    logic = prediction.edge_logic_state
    if logic is not None and (logic.masses.shape != (len(ids), 4) or logic.available.shape != (len(ids),)
                              or logic.available.dtype != torch.bool):
        raise ValueError("Forward edge logic does not match canonical fact order")
    contexts = {}
    for fact in source.facts:
        fact_id, index = fact.canonical_fact_id, fact.fact_index
        provenance = set()
        for row in records[fact_id]:
            for field in ("raw_direct_sources", "raw_indirect_sources"):
                provenance.update(row.get(field) or ())
        available = logic is not None and bool(logic.available[index])
        masses = tuple(logic.masses[index].detach().float().cpu().tolist()) if available else None
        contexts[fact_id] = FactExplanationContext(fact_id, tuple(sorted(provenance)), available, masses)
    return contexts, records


def _query_scalar(prediction, name):
    value = getattr(prediction, name)
    if value.shape != (1,) or not bool(torch.isfinite(value).all()):
        raise ValueError("Forward query value must be one finite scalar: " + name)
    return float(value[0].detach().float().cpu())


def build_query_explanation(corpus, graph, prediction, *, drug_id, disease_id, domain, top_k=5):
    """Export only the supplied forward; this function performs no path search or inference.

    ``graph`` must be the exact input of ``prediction``. The caller controls eval
    mode and optional edge-logic retention. Missing fact logic stays unavailable.
    Scores and T/F/B/U masses describe the model output and are uncalibrated.
    """
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(domain, str) or not domain:
        raise ValueError("An explicit query domain is required")
    candidates = _forward_candidates(corpus, graph, drug_id, disease_id)
    packed, source = graph.batch.path_batch, graph.batch.path_source
    reasoning = prediction.path_reasoning
    if prediction.path_available.shape != (1,) or prediction.path_available.dtype != torch.bool:
        raise ValueError("Forward path availability must describe one query")
    available = bool(prediction.path_available[0])
    payloads = []
    if reasoning is None:
        if available or prediction.path_gate is not None:
            raise ValueError("An unused path branch cannot have path availability or a fusion gate")
        path_status, forward_count = "PATH_BRANCH_DISABLED", 0
    else:
        if reasoning.path_available.shape != (1,) or available != bool(reasoning.path_available[0]) or available != bool(packed.num_paths):
            raise ValueError("Forward path availability differs from the packed input")
        contexts, records = _fact_context(corpus, source, prediction)
        paths = build_path_explanations(candidates, packed, reasoning,
            query_pair_ids=(source.query_pair_id,), drug_ids=(drug_id,), disease_ids=(disease_id,),
            fact_context=contexts, top_k=top_k)
        for path in paths:
            payload = asdict(path)
            for edge in payload["edges"]:
                edge["evidence_records"] = records[edge["canonical_fact_id"]]
            payloads.append(payload)
        path_status = "AVAILABLE" if available else "NO_LEGAL_FORWARD_PATHS"
        forward_count = packed.num_paths
    result = {name: _query_scalar(prediction, name) for name in (
        "support_logit", "unsuitability_logit", "support_score", "unsuitability_score", "ranking_score")}
    result.update(schema="EVIHGT_QUERY_EXPLANATION_V1", corpus_identity=corpus.identity,
        query_pair_id=source.query_pair_id, drug_id=drug_id, disease_id=disease_id, domain=domain,
        query_logic_masses={key: _query_scalar(prediction, "m_" + key) for key in "TFBU"},
        logic_mass_order=["T", "F", "B", "U"], scores_are_calibrated_probabilities=False,
        model_explanation_is_causal_proof=False,
        path_available=available,
        path_gate=None if prediction.path_gate is None else _query_scalar(prediction, "path_gate"),
        path_status=path_status, total_forward_paths=forward_count,
        total_candidate_paths=packed.num_paths, top_k=top_k, paths=payloads)
    return result
