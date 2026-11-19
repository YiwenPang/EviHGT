"""EXPRESSION_CANONICAL_PRESERVATION_V1, an explicit message-graph protocol change.

Only EdgeBatch is rebuilt. Canonical evidence and independent path provenance stay
unchanged. The production catalog assigns a fact's canonical type to both legal
traversal directions; it has no extra reverse type or direction feature in EdgeBatch.
No model, parameters, source datasets, neighbor selection, or path search is run here.
"""
from collections import Counter
from dataclasses import fields, replace
from numbers import Integral

import torch

from evi_hgt.models.input_contract import EdgeBatch
from evi_hgt.sampling.path_schema import PathCandidate


POLICY = 'EXPRESSION_CANONICAL_PRESERVATION_V1'
EXPRESSION = frozenset({'EXPRESSION_PRESENT', 'EXPRESSION_ABSENT'})


def _integer_set(values, name):
    result = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(name + ' must contain nonnegative integer canonical IDs')
        result.add(int(value))
    return result


def _vector(value, length, name, upper=None, *, device=None):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.shape != (length,):
        raise ValueError(name + ' must be a correctly sized int64 tensor')
    if device is not None and value.device != device:
        raise ValueError(name + ' must share the message tensor device')
    data = value.detach().cpu().tolist()
    if any(v < 0 or (upper is not None and v >= upper) for v in data):
        raise ValueError(name + ' contains an out-of-range index')
    return data


def _path_check(batch, source, node_count, fact_count):
    """Paths refer to canonical facts and their own directions, never edge rows."""
    paths = batch.path_batch
    if paths is None:
        raise ValueError('Expected the existing packed path batch, including an explicit empty batch')
    paths.validate(num_nodes=node_count, num_facts=fact_count)
    if paths.num_queries != batch.model_batch.queries.num_queries:
        raise ValueError('Path query count differs from model query count')
    traversals = {(t.source_node_index, t.target_node_index, t.fact_index, t.direction)
                  for t in source.traversals}
    src = paths.step_source_index.detach().cpu().tolist()
    dst = paths.step_target_index.detach().cpu().tolist()
    fact = paths.step_fact_index.detach().cpu().tolist()
    direction = paths.step_direction.detach().cpu().tolist()
    lengths = paths.path_length.detach().cpu().tolist()
    queries = paths.query_index.detach().cpu().tolist()
    for row, length in enumerate(lengths):
        nodes = (src[row][0],) + tuple(dst[row][:length])
        if tuple(src[row][:length]) != nodes[:-1]:
            raise ValueError('Packed path is not a continuous original traversal')
        fs = tuple(fact[row][:length]); ds = tuple(direction[row][:length])
        for s, t, f, d in zip(src[row][:length], dst[row][:length], fs, ds, strict=True):
            canonical = source.facts[f]
            expected = ((canonical.source_node_index, canonical.target_node_index) if d == 0
                        else (canonical.target_node_index, canonical.source_node_index))
            if (s,t) != expected or (d == 1 and not canonical.undirected):
                raise ValueError('Packed path step does not match its canonical fact/direction')
            if (s,t,f,d) not in traversals:
                raise ValueError('Packed path step was not in the original legal path traversals')
        if nodes[0] != source.drug_node_index or nodes[-1] != source.disease_node_index:
            raise ValueError('Packed path endpoints differ from the original query')
        candidate = PathCandidate.create(query_index=queries[row], node_indices=nodes,
            node_ids=tuple(source.node_ids[n] for n in nodes),
            node_types=tuple(source.node_types[n] for n in nodes), fact_indices=fs,
            canonical_fact_ids=tuple(source.facts[f].canonical_fact_id for f in fs),
            relations=tuple(source.facts[f].relation for f in fs), directions=ds, structural_score=0.0)
        if candidate.path_id != paths.path_ids[row]:
            raise ValueError('Packed path identity differs from its original canonical fact/directions')
    return paths.num_paths


def apply_policy(graph, index, forbidden=None):
    """Return a new QueryGraph and audit, replacing only each first reverse-only slot.

    ``forbidden`` may add already-resolved canonical IDs (including IDs of any
    equivalent facts). The caller supplies the existing query/split exclusion union;
    PathSource.forbidden_fact_ids is always included. Any forbidden input survivor
    aborts. Missing or contradictory metadata also aborts without mutating the input.
    """
    batch = graph.batch; model = batch.model_batch; source = batch.path_source
    if source is None:
        raise ValueError('Canonical PathSource metadata is required to certify preserved paths')
    source.validate()
    node_count = model.nodes.num_nodes; edge_count = model.edges.num_edges
    fact_count = model.evidence.num_facts; device = model.edges.src.device
    if node_count > 1000 or edge_count > 5000:
        raise ValueError('Original 1000-node/5000-message bound exceeded')
    if model.queries.num_queries != 1:
        raise ValueError('This bounded per-query policy requires exactly one query')
    if len(source.facts) != fact_count or len(source.node_ids) != node_count:
        raise ValueError('PathSource/model canonical fact or node count differs')
    if model.nodes.node_id is None:
        raise ValueError('Canonical node IDs are required')
    node_ids = _vector(model.nodes.node_id,node_count,'node_id',device=device)
    node_types = _vector(model.nodes.node_type_id,node_count,'node_type_id',len(index.node_type_catalog),device=device)
    node_graphs = _vector(model.nodes.graph_id,node_count,'node graph_id',device=device)
    if tuple(node_ids) != source.node_ids or tuple(index.node_type_catalog[t] for t in node_types) != source.node_types:
        raise ValueError('Canonical node mapping differs from original path metadata')
    if list(map(int,index.node_type_ids(node_ids))) != node_types:
        raise ValueError('Runtime node types differ from the existing static index')
    query_src = _vector(model.queries.drug_node_index,1,'query drug index',node_count,device=device)[0]
    query_dst = _vector(model.queries.disease_node_index,1,'query disease index',node_count,device=device)[0]
    query_graph = _vector(model.queries.graph_id,1,'query graph_id',device=device)[0]
    if (query_src,query_dst) != (source.drug_node_index,source.disease_node_index):
        raise ValueError('Query endpoint mapping differs from original path metadata')
    if any(g != query_graph for g in node_graphs):
        raise ValueError('Per-query graph contains foreign node graph IDs')
    canonical_ids = [f.canonical_fact_id for f in source.facts]
    banned = _integer_set(source.forbidden_fact_ids,'PathSource forbidden facts')
    banned |= _integer_set(() if forbidden is None else forbidden,'additional forbidden facts')
    if banned.intersection(canonical_ids):
        raise ValueError('Forbidden canonical/equivalent fact survives in the input graph')
    catalog = tuple(index.edge_catalog)
    if len(set(catalog)) != len(catalog):
        raise ValueError('Duplicate canonical relation/type catalog entries')
    lookup = {key:i for i,key in enumerate(catalog)}
    fact_types = _vector(model.evidence.fact_edge_type_id,fact_count,'fact_edge_type_id',len(catalog),device=device)
    fact_graphs = _vector(model.evidence.fact_graph_id,fact_count,'fact_graph_id',device=device)
    records = _vector(model.evidence.record_fact_index,model.evidence.num_records,'record_fact_index',fact_count,device=device)
    record_types = _vector(model.evidence.record_edge_type_id,len(records),'record_edge_type_id',len(catalog),device=device)
    if record_types != [fact_types[f] for f in records]:
        raise ValueError('Evidence record type differs from its canonical fact')
    metadata = index.fact_metadata(canonical_ids)
    rows = {int(r['canonical_fact_id']):r for r in metadata.iter_rows(named=True)}
    if metadata.height != fact_count or set(rows) != set(canonical_ids):
        raise ValueError('Static index did not supply exactly the selected canonical fact metadata')
    canonical = []
    for i,fact in enumerate(source.facts):
        raw = rows[fact.canonical_fact_id]
        key = (raw['source_node_type'],raw['relation'],raw['target_node_type'])
        if key not in lookup or lookup[key] != fact.edge_type_id or fact_types[i] != lookup[key]:
            raise ValueError('No matching existing canonical relation/type mapping')
        if type(raw['undirected']) is not bool or raw['undirected'] != fact.undirected:
            raise ValueError('Canonical legal traversal declaration differs')
        s,t = fact.source_node_index,fact.target_node_index
        if (raw['source_global_id'],raw['target_global_id']) != (node_ids[s],node_ids[t]):
            raise ValueError('Canonical fact endpoint missing or differs from original node mapping')
        if key != (source.node_types[s],fact.relation,source.node_types[t]):
            raise ValueError('Canonical source/target node types or relation differ')
        if fact_graphs[i] != node_graphs[s] or fact_graphs[i] != node_graphs[t]:
            raise ValueError('Canonical fact graph differs from endpoint graph')
        canonical.append((s,t,lookup[key],key in {('ANA',r,'GEN') for r in EXPRESSION}))
    values = {f.name:_vector(getattr(model.edges,f.name),edge_count,'edge '+f.name,
                            node_count if f.name in ('src','dst') else
                            fact_count if f.name=='fact_index' else len(catalog) if f.name=='edge_type_id' else None,
                            device=device) for f in fields(EdgeBatch)}
    slots = [[] for _ in range(fact_count)]; directions = []
    for row,(s,t,f,typ,g) in enumerate(zip(values['src'],values['dst'],values['fact_index'],
                                           values['edge_type_id'],values['graph_id'],strict=True)):
        cs,ct,canonical_type,_ = canonical[f]
        if typ != canonical_type or g != fact_graphs[f] or g != node_graphs[s] or g != node_graphs[t]:
            raise ValueError('Runtime edge type or graph differs from canonical fact/endpoints')
        if (s,t)==(cs,ct):direction=0
        elif (s,t)==(ct,cs) and source.facts[f].undirected:direction=1
        else:raise ValueError('Runtime message is not a legal canonical or reverse traversal')
        directions.append(direction); slots[f].append(row)
    if set(values['fact_index']) != set(range(fact_count)):
        raise ValueError('Selected canonical facts and represented runtime fact set differ')
    path_count = _path_check(batch,source,node_count,fact_count)
    updated = {key:list(value) for key,value in values.items()}
    replacements=[]; expression_facts=0; forward_present=0; reverse_only=0
    for f,(s,t,typ,eligible) in enumerate(canonical):
        if not eligible:continue
        expression_facts+=1
        if any(directions[row]==0 for row in slots[f]):
            forward_present+=1; continue
        reverse_only+=1
        row = slots[f][0]
        # Keep slot order and the one fact context. No direction-specific type or
        # extra edge-feature tensor exists in the actual five-field EdgeBatch.
        updated['src'][row]=s; updated['dst'][row]=t; updated['edge_type_id'][row]=typ
        updated['fact_index'][row]=f; updated['graph_id'][row]=fact_graphs[f]
        replacements.append(dict(slot=row,fact_index=f,canonical_fact_id=canonical_ids[f],
            relation=source.facts[f].relation,old_source_index=values['src'][row],
            old_target_index=values['dst'][row],source_index=s,target_index=t,
            source_global_id=node_ids[s],target_global_id=node_ids[t],edge_type_id=typ,
            old_direction=1,new_direction=0))
    replaced_rows=[r['slot'] for r in replacements]
    if len(set(replaced_rows)) != len(replaced_rows):
        raise ValueError('Duplicate replacement slot')
    nonexpression=lambda columns:Counter(tuple(columns[n][r] for n in ('src','dst','edge_type_id','fact_index','graph_id'))
        for r in range(edge_count) if not canonical[columns['fact_index'][r]][3])
    if nonexpression(values) != nonexpression(updated):
        raise ValueError('Non-expression message identity multiset changed')
    for r in range(edge_count):
        f=updated['fact_index'][r]; s,t,typ,_=canonical[f]
        endpoints=(updated['src'][r],updated['dst'][r])
        if endpoints!=(s,t) and not (source.facts[f].undirected and endpoints==(t,s)):
            raise ValueError('Rebuilt runtime message has illegal endpoints')
        if updated['edge_type_id'][r]!=typ or updated['graph_id'][r]!=fact_graphs[f]:
            raise ValueError('Rebuilt message type/fact/graph mapping differs')
    new_edges=EdgeBatch(**{name:torch.tensor(data,dtype=torch.long,device=device) for name,data in updated.items()})
    new_model=replace(model,edges=new_edges)
    result=replace(graph,batch=replace(batch,model_batch=new_model))
    for f in fields(batch):
        if f.name!='model_batch' and getattr(result.batch,f.name) is not getattr(batch,f.name):
            raise ValueError('TrainBatch metadata changed')
    if any(getattr(new_model,n) is not getattr(model,n) for n in ('nodes','evidence','queries')):
        raise ValueError('Non-message model input changed')
    if set(updated['fact_index'])!=set(values['fact_index']) or len(updated['src'])!=edge_count:
        raise ValueError('Unique canonical fact or message count changed')
    if banned.intersection(source.facts[f].canonical_fact_id for f in updated['fact_index']):
        raise ValueError('Forbidden fact in rebuilt message graph')
    audit=dict(policy=POLICY,message_graph_protocol_change=True,biological_facts_added=0,
        node_count=node_count,unique_fact_count=fact_count,evidence_record_count=model.evidence.num_records,
        message_count=edge_count,expression_fact_count=expression_facts,
        expression_forward_already_present=forward_present,expression_reverse_only=reverse_only,
        replacement_count=len(replacements),replacements=replacements,path_count=path_count,
        path_source_same_object=True,path_batch_same_object=True,path_semantics_preserved=True,
        context_binding='CANONICAL_FACT_INDEX_WITH_INDEPENDENT_PATH_DIRECTION',
        runtime_type_binding='EXISTING_CANONICAL_FACT_TYPE_FOR_BOTH_LEGAL_TRAVERSALS',
        direction_features='NONE_IN_EDGE_BATCH; PATH_DIRECTION_UNCHANGED',
        slot_order_preserved=True,nonexpression_message_multiset_preserved=True,
        forbidden_fact_survivors=0,forbidden_union_count=len(banned),
        node_fact_evidence_and_masks_preserved=True)
    return result,audit
