"""Bounded query graphs, exact evidence gathers, and independent path directions."""
from dataclasses import dataclass, fields, replace
import time
import torch

from evi_hgt.data.native_csr import extract_csr_subgraph_native
from evi_hgt.training.indexed_assembly import tensorize_indexed, move_model_batch
from evi_hgt.sampling.path_schema import PathSource, PathFact, PathTraversal, PathExtractionConfig, pack_path_candidates
from evi_hgt.sampling.path_extractor import PathCandidateExtractor
from evi_hgt.sampling.path_leakage import PathLeakagePolicy, audit_path_leakage
from .direction_policy import apply_policy


@dataclass(frozen=True)
class QueryBatchEnvelope:
    model_batch: object
    supervision: tuple
    path_source: object
    path_batch: object


@dataclass(frozen=True)
class QueryGraph:
    batch: QueryBatchEnvelope
    biological_nodes: int
    timings: dict

    def forward(self, built):
        return built.forward(self.batch.model_batch, path_batch=self.batch.path_batch)

    def to(self, device):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if hasattr(value, '__dataclass_fields__'):
                return replace(value, **{f.name: move(getattr(value, f.name)) for f in fields(value)})
            return value
        return replace(self, batch=replace(self.batch,
            model_batch=move_model_batch(self.batch.model_batch, device), path_batch=move(self.batch.path_batch)))


class QueryAssembler:
    def __init__(self, corpus):
        self.corpus = corpus
        self.index = corpus.index
        self.extractor = PathCandidateExtractor(PathExtractionConfig())
        self.edge_types = {v: i for i, v in enumerate(self.index.edge_catalog)}

    def assemble(self, query, *, domain='training', device='cpu'):
        started = time.perf_counter()
        c, index = self.corpus, self.index
        pair = query.drug_id, query.disease_id
        if domain == 'training':
            role = 'U' if query.observed_relation is None else 'support' if query.support.observed_mask else 'unsuitability'
            c.assert_training_pair(pair, role)
        elif domain in c.panels:
            if pair not in c.panel_pairs[domain] or query.observed_relation is not None:
                raise ValueError('Evaluation graphs require a role-free member of the declared panel')
        else:
            raise ValueError('Unknown query domain: ' + str(domain))
        drug, disease = map(int, index.node_global_ids(pair))
        if index.node_metadata([drug, disease])['node_type'].to_list() != ['DRG', 'DIS']:
            raise ValueError('Query endpoints must be canonical drug and disease nodes')
        targets = c.target_ids.get(pair, set())
        extracted = extract_csr_subgraph_native(c.root, seed_node_ids=[drug, disease],
            max_hops=2, max_nodes=1000, max_edges=5000, forbidden_fact_ids=targets, csr_arrays=c.csr)
        forbidden = c.forbidden | targets
        if forbidden.intersection(extracted.fact_ids):
            raise ValueError('Forbidden target/equivalent fact survived query extraction')
        gathered = index.gather(extracted.fact_ids)
        fact_ids = list(map(int, gathered['fact_ids']))
        facts = index.fact_metadata(fact_ids)
        batch = tensorize_indexed(index, gathered, node_ids=extracted.node_ids,
            traversal_edges=extracted.traversal_edges, drug_global=drug, disease_global=disease)
        values = c.features.gather(batch.nodes.node_id.numpy(), index.node_metadata(batch.nodes.node_id.numpy())['node_type'].to_list())
        batch = replace(batch, nodes=replace(batch.nodes, features=torch.from_numpy(values)))
        nodes = tuple(sorted(extracted.node_ids))
        local = {n: i for i, n in enumerate(nodes)}
        pf = tuple(PathFact(i, int(r['canonical_fact_id']), local[int(r['source_global_id'])],
            local[int(r['target_global_id'])], r['relation'], r['undirected'],
            self.edge_types[(r['source_node_type'], r['relation'], r['target_node_type'])])
            for i, r in enumerate(facts.iter_rows(named=True)))
        fact_index = {f.canonical_fact_id: f.fact_index for f in pf}
        pt = tuple(PathTraversal(local[a], local[b], fact_index[f],
            0 if (pf[fact_index[f]].source_node_index, pf[fact_index[f]].target_node_index) == (local[a], local[b]) else 1)
            for a, b, f in extracted.traversal_edges)
        source = PathSource('standard', query.canonical_pair_key, nodes,
            tuple(index.node_metadata(nodes)['node_type']), pf, pt, local[drug], local[disease],
            frozenset(forbidden), frozenset(), c.identity, c.identity,
            'train' if domain == 'training' else 'validation')
        candidates = self.extractor.extract(source)
        audit_path_leakage(candidates, PathLeakagePolicy('standard', query.canonical_pair_key,
            drug, disease, source.forbidden_fact_ids, frozenset(), {}, source.forbidden_fact_ids, 'standard'))
        paths = pack_path_candidates(candidates, num_queries=1, max_length=4, device='cpu')
        graph = QueryGraph(QueryBatchEnvelope(batch, (query,), source, paths), len(nodes), {})
        graph, audit = apply_policy(graph, index, forbidden=forbidden)
        graph = replace(graph, timings={'assembly_seconds': time.perf_counter()-started,
            'canonical_expression_replacements': audit['replacement_count']})
        return graph if torch.device(device).type == 'cpu' else graph.to(device)
