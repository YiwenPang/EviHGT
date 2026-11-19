"""Method-specific, training-only preparation over a certified cell assembler.

MLP gathers only the two immutable raw endpoint feature rows. HGT reuses the
same bounded CSR extraction and canonical message policy without gathering EP
records or constructing paths. No encoded/trainable representation is cached.
Enable only after the root's registered old/new input and update qualification.
"""
from dataclasses import asdict
from copy import deepcopy
import time
import numpy as np
import torch
from evi_hgt.data.native_csr import extract_csr_subgraph_native
from evi_hgt.models.input_contract import NodeBatch,EdgeBatch,QueryBatch,EviHGTBatch
from evi_hgt.training.batcher import TrainBatch
from evi_hgt.training.indexed_assembly import move_model_batch
from evi_hgt.training.supervision import make_unlabeled_query_supervision
from evi_hgt.inner_validation.production import assert_training_channel
from evi_hgt.query_graph.production import QueryGraph
from .training import PreparedStep,_validate_prepared,prepare_step
from .input_contract import NODE_TYPES,endpoint_nodes,validate_graph

MLP='PAIR_MLP_D1_MATCHED'
HGT='HGT_D1_MATCHED'
VERSION='RANKING_COMPARISON_METHOD_RAW_PREPARATION_V1'
EXPRESSION=frozenset({'EXPRESSION_PRESENT','EXPRESSION_ABSENT'})


def _cell(assembler):
    base=assembler.base.base
    if base.index is not assembler.index or base.index.binding.canonical_graph_digest!=base.corpus['digest']:
        raise ValueError('Canonical assembler, static index and cell corpus differ')
    return base


def _authorize_query(base,query):
    pair=(query.drug_id,query.disease_id)
    base.data.pairs.assert_optimizer_pair(pair,'U' if query.observed_relation is None else 'P')
    if query.observed_relation is not None:
        channel='support' if query.support.observed_mask else 'unsuitability'
        assert_training_channel(pair,channel,base.data.support,base.data.unsuitability)
    ids=list(map(int,base.index.node_global_ids(pair)))
    kinds=base.index.node_metadata(ids)['node_type'].to_list()
    if kinds!=['DRG','DIS']:raise ValueError('Query canonical endpoint types differ')
    return pair,ids


def _raw_nodes(index,features,ids):
    ids=list(ids);types=index.node_metadata(ids)['node_type'].to_list()
    return NodeBatch(torch.tensor(index.node_type_ids(ids).tolist(),dtype=torch.long),
        torch.zeros(len(ids),dtype=torch.long),torch.from_numpy(features.gather(ids,types)),
        torch.tensor(ids,dtype=torch.long))


def _mlp(base,query,pair,ids):
    z=torch.empty(0,dtype=torch.long)
    batch=EviHGTBatch(_raw_nodes(base.index,base.features,ids),EdgeBatch(z,z,z,z,z),None,
        QueryBatch(torch.tensor([0]),torch.tensor([1]),torch.tensor([0])))
    endpoint_nodes(batch)
    return batch,dict(graph_extraction=False,evidence_gather=False,path_extraction=False,
        target_fact_or_neighbor_features_consumed=False,unique_fact_count=0,
        replacement_count=0,forbidden_fact_survivors=0)


def _canonical_edges(index,extracted,forbidden):
    """Same sorted fact/node order and first reverse-only EXPRESSION slot rule."""
    nodes=sorted(map(int,extracted.node_ids));local={n:i for i,n in enumerate(nodes)}
    ids=sorted(set(map(int,extracted.fact_ids)))
    if forbidden.intersection(ids):raise RuntimeError('Forbidden fact survived production extraction')
    meta=index.fact_metadata(ids)
    rows=list(meta.iter_rows(named=True))
    if [int(r['canonical_fact_id']) for r in rows]!=ids:raise ValueError('Canonical metadata fact ordering differs')
    kinds=index.node_metadata(nodes)['node_type'].to_list()
    if len(kinds)!=len(nodes) or any(kind not in NODE_TYPES for kind in kinds):
        raise ValueError('Canonical node endpoint types differ')
    type_ids=list(map(int,index.node_type_ids(nodes)))
    if type_ids!=[NODE_TYPES.index(kind) for kind in kinds]:
        raise ValueError('Canonical node type IDs differ from node metadata')
    catalog=tuple(index.edge_catalog)
    if len(catalog)!=len(set(catalog)):raise ValueError('Duplicate relation catalog entry')
    lookup={key:i for i,key in enumerate(catalog)}
    canonical=[]
    for r in rows:
        key=(r['source_node_type'],r['relation'],r['target_node_type'])
        a,b=int(r['source_global_id']),int(r['target_global_id'])
        if key not in lookup or a not in local or b not in local or type(r['undirected']) is not bool:
            raise ValueError('Canonical metadata endpoint/type/traversal differs')
        if (kinds[local[a]],kinds[local[b]])!=(key[0],key[2]):
            raise ValueError('Canonical fact endpoint types differ from index node types')
        canonical.append((local[a],local[b],lookup[key],r['undirected'],
            key[0]=='ANA' and key[1] in EXPRESSION and key[2]=='GEN'))
    by_id={fact:i for i,fact in enumerate(ids)}
    values=dict(src=[],dst=[],edge_type_id=[],fact_index=[],graph_id=[])
    slots=[[] for _ in ids];directions=[]
    for i,(a,b,f) in enumerate(extracted.traversal_edges):
        if f not in by_id or a not in local or b not in local:raise ValueError('Traversal not in selected facts/nodes')
        fi=by_id[f];s,t,typ,undirected,_=canonical[fi];actual=(local[a],local[b])
        if actual==(s,t):direction=0
        elif actual==(t,s) and undirected:direction=1
        else:raise ValueError('Illegal canonical/reverse traversal')
        values['src'].append(actual[0]);values['dst'].append(actual[1]);values['edge_type_id'].append(typ)
        values['fact_index'].append(fi);values['graph_id'].append(0);slots[fi].append(i);directions.append(direction)
    if set(values['fact_index'])!=set(range(len(ids))):raise ValueError('Selected fact lacks runtime message')
    replacements=0
    for f,(s,t,typ,_,eligible) in enumerate(canonical):
        if eligible and not any(directions[i]==0 for i in slots[f]):
            i=slots[f][0];values['src'][i]=s;values['dst'][i]=t;replacements+=1
    edges=EdgeBatch(**{name:torch.tensor(v,dtype=torch.long) for name,v in values.items()})
    return nodes,local,edges,replacements


def _hgt(base,query,pair,ids):
    targets=base.target_ids.get(pair,set())
    extracted=extract_csr_subgraph_native(base.root/'csr',seed_node_ids=ids,
        max_hops=2,max_nodes=1000,max_edges=5000,forbidden_fact_ids=targets,csr_arrays=base.csr)
    nodes,local,edges,replacements=_canonical_edges(base.index,extracted,base.forbidden|targets)
    batch=EviHGTBatch(_raw_nodes(base.index,base.features,nodes),edges,None,
        QueryBatch(torch.tensor([local[ids[0]]]),torch.tensor([local[ids[1]]]),torch.tensor([0])))
    validate_graph(batch,base.index.edge_catalog)
    return batch,dict(graph_extraction=True,evidence_gather=False,path_extraction=False,
        canonical_message_policy='EXPRESSION_CANONICAL_PRESERVATION_V1',
        replacement_count=replacements,unique_fact_count=len(set(extracted.fact_ids)),
        forbidden_fact_survivors=0,slot_order_preserved=True)


def _record_prepared_coverage(assembler,method,prepared):
    """Observation only: count completed preparation, never claim model execution."""
    if method not in (MLP,HGT):raise ValueError('Only optimized matched preparation is counted')
    count=dict(preparation_calls=1,query_occurrences=0,extra_query_occurrences=len(prepared.extras),
        graph_extraction_queries=0,evidence_gather_queries=0,path_extraction_queries=0,
        node_occurrences=0,message_occurrences=0,unique_fact_occurrences=0,
        canonical_expression_replacements=0,forbidden_fact_survivors=0)
    for graph in prepared.graphs+prepared.extras:
        audit=graph.timings;batch=graph.batch.model_batch
        if audit.get('preparation_version')!=VERSION:raise ValueError('Coverage preparation identity differs')
        count['query_occurrences']+=1
        for source,target in [('graph_extraction','graph_extraction_queries'),
                              ('evidence_gather','evidence_gather_queries'),('path_extraction','path_extraction_queries')]:
            if type(audit[source]) is not bool:raise ValueError('Coverage operation flag is not Boolean')
            count[target]+=int(audit[source])
        count['node_occurrences']+=batch.nodes.num_nodes;count['message_occurrences']+=batch.edges.num_edges
        for source,target in [('unique_fact_count','unique_fact_occurrences'),
                              ('replacement_count','canonical_expression_replacements'),('forbidden_fact_survivors','forbidden_fact_survivors')]:
            value=audit[source]
            if type(value) is not int or value<0:raise ValueError('Coverage count must be a nonnegative integer')
            count[target]+=value
    current=deepcopy(getattr(assembler,'_ranking_comparison_preparation_coverage',{}))
    previous=current.get(method,{key:0 for key in count})
    if set(previous)!=set(count):raise ValueError('Preparation coverage schema changed')
    current[method]={key:previous[key]+value for key,value in count.items()}
    assembler._ranking_comparison_preparation_coverage=current


def coverage_summary(assembler,method=None):
    counts=deepcopy(getattr(assembler,'_ranking_comparison_preparation_coverage',{}))
    if method is not None:counts={method:counts[method]} if method in counts else {}
    return dict(schema='RANKING_COMPARISON_OPTIMIZED_PREPARATION_COVERAGE_V1',preparation_version=VERSION,
        scope='OPTIMIZED_MATCHED_INPUT_PREPARATION_ONLY',counts_are_query_occurrences=True,
        represents_training_execution=False,methods=counts)


def prepare_optimized(method,assembler,spec,device='cpu'):
    if method=='FULL':return prepare_step(assembler,spec,device=device)
    if method not in (MLP,HGT):raise ValueError('No preparation optimization for this method')
    spec.validate();base=_cell(assembler)
    if base.data.cell_id!=spec.cell or base.data.training_seed!=spec.identity['training_seed']:
        raise ValueError('Input factory cell/seed differs from sealed plan')
    extras=() if spec.record is None else tuple(make_unlabeled_query_supervision(*p) for p in spec.record['extra_pairs'])
    def one(query):
        started=time.perf_counter();pair,ids=_authorize_query(base,query)
        batch,audit=(_mlp if method==MLP else _hgt)(base,query,pair,ids)
        tb=TrainBatch.from_model_batch(move_model_batch(batch,device),(query,),split_kind='standard')
        return QueryGraph(tb,None,batch.nodes.num_nodes,dict(base_assembly=time.perf_counter()-started,
            similarity_gather=0.,preparation_version=VERSION,**audit))
    result=PreparedStep(spec,tuple(one(q) for q in spec.queries),tuple(one(q) for q in extras))
    _validate_prepared(result);_record_prepared_coverage(assembler,method,result);return result


def consumed_inputs(prepared,method):
    """Exact comparison projection of every field the selected model/loss uses."""
    if method not in (MLP,HGT):raise ValueError('Matched methods only')
    result=[]
    for graph in prepared.graphs+prepared.extras:
        tb=graph.batch;batch=tb.model_batch
        if method==MLP:
            inputs=dict(nodes=asdict(endpoint_nodes(batch)),query_order=['DRG','DIS'])
        else:inputs=dict(nodes=asdict(batch.nodes),edges=asdict(batch.edges),queries=asdict(batch.queries))
        inputs['loss_inputs']={name:getattr(tb,name) for name in (
            'support_positive_mask','support_unlabeled_mask','unsuitability_positive_mask','unsuitability_unlabeled_mask',
            'query_pair_ids','sample_sources')}
        result.append(inputs)
    return dict(spec=prepared.spec.metadata(method),content_digest=prepared.spec.content_digest,queries=result)
