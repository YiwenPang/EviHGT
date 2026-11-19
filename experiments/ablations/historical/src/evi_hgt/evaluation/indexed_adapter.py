"""Evaluation-only reuse of the existing immutable input index and exact CSR.

No model, objective, candidate, truncation, path-search or cold policy changes.
The scalar assembler remains the oracle; its training dispatch is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
import hashlib
import time
import numpy as np
import polars as pl
import pyarrow.ipc as ipc
import torch

from evi_hgt.data.native_csr import extract_csr_subgraph_native
from evi_hgt.training.indexed_assembly import tensorize_indexed,move_model_batch
from evi_hgt.training.input_index import _immutable
from evi_hgt.training.batcher import TrainBatch,BatchAssemblyAudit


@dataclass(frozen=True)
class _ReindexedEncoding:
    encoded: object
    original_to_safe: np.ndarray

    def gather(self,positions):
        safe=self.original_to_safe[positions]
        if np.any(safe<0):raise ValueError('uncertified cold evidence reached encoding')
        return self.encoded.gather(safe)


class _ColdIndex:
    """Sanitized model gather; raw evidence access also applies the frozen policy."""
    def __init__(self,index,visibility,startup_seconds,extra_bytes):
        self.index=index;self.visibility=visibility
        self.cold_startup_seconds=startup_seconds;self.cold_extra_bytes=extra_bytes

    def __getattr__(self,name):return getattr(self.index,name)

    def evidence_for_facts(self,ids):
        return self.visibility.evidence(self.index.evidence_for_facts(ids),set(map(int,ids)))

    def evidence_rows(self,ids):
        rows=self.index.evidence_rows(ids)
        return self.visibility.evidence(rows,set(rows['canonical_fact_id'].to_list()))

    def evidence_static(self,ids):
        positions=self.index._evidence_map.positions(ids)
        if not np.all(self.index._record_production_visible[positions]):
            raise ValueError('uncertified or non-owner cold record')
        return self.index._static_rows(positions)


def build_cold_index(index,visibility,*,extra_limit=8*1024**3):
    """Encode the existing sanitizer once in bounded Arrow blocks, sharing graph.

    Record offsets retain the original physical positions. Only certified,
    component-owned records are selected; duplicates/tokens/order/dtypes are
    exactly those of ColdVisibility.evidence followed by scalar tensorization.
    """
    from evi_hgt.training.input_encoding import encode_records
    start=time.perf_counter();blocks=[]
    allowed=np.zeros(index._evidence_ids.size,dtype=bool);extra=allowed.nbytes
    offsets=index._evidence_raw._offsets
    for block_number,raw in enumerate(index._evidence_raw._blocks):
        frame=pl.from_arrow(ipc.open_stream(raw).read_all())
        safe=visibility.evidence(frame,set(frame['canonical_fact_id'].to_list()))
        physical=index._evidence_map.positions(safe['evidence_record_id'].to_numpy())
        start_row,end_row=map(int,offsets[block_number:block_number+2])
        local=physical-start_row
        if np.any(local<0) or np.any(local>=end_row-start_row):raise ValueError('cold encoding row binding drift')
        mapping=np.full(frame.height,-1,dtype=np.int64);mapping[local]=np.arange(safe.height)
        allowed[physical]=True
        encoded=encode_records(safe,token_buckets=index.binding.token_buckets)
        extra+=mapping.nbytes+encoded.nbytes
        if extra>extra_limit:raise MemoryError('cold encoding hard memory budget exceeded; no truncation')
        blocks.append(_ReindexedEncoding(encoded,_immutable(mapping)))
    selected=replace(index,_encoded=tuple(blocks),_encoded_offsets=offsets,
        _record_production_visible=_immutable(allowed & index._record_production_visible))
    return _ColdIndex(selected,visibility,time.perf_counter()-start,extra)


def cold_forbidden_filter(mask):
    hidden=np.array(sorted(mask.hidden_nodes),dtype=np.int64)
    targets=np.array(sorted(mask.target_facts),dtype=np.int64)
    def forbidden(values):
        ids=np.asarray(values,dtype=np.int64);result=np.ones(ids.shape,dtype=bool)
        valid=(ids>=0)&(ids<mask.admitted.size);v=ids[valid]
        result[valid]=(~mask.admitted[v] | mask.permanent[v] | np.isin(v,targets)
            | np.isin(mask.source[v],hidden) | np.isin(mask.target[v],hidden))
        return result
    return forbidden


class IndexedEvaluationAssembler:
    """Bounded pair-dependent graph assembly with attempt-owned static inputs."""
    def __init__(self,scalar,index,*,split_frames,cold_visibility=None):
        if tuple(scalar.adapter.node_type_lookup)!=index.node_type_catalog:
            raise ValueError('evaluation index node catalog mismatch')
        if tuple(scalar.adapter.edge_type_lookup)!=index.edge_catalog:
            raise ValueError('evaluation index relation catalog mismatch')
        self.scalar=scalar;self.index=index;self.config=scalar.config
        self.cold_visibility=cold_visibility;self._cold_templates={}
        self._assignments={};self._partitions={};self._targets={}
        for row in scalar._target_map.iter_rows(named=True):
            self._targets.setdefault((row['drug_id'],row['disease_id']),set()).add(int(row['canonical_fact_id']))
        for split,frame in split_frames.items():
            pairs={}
            for row in frame.iter_rows(named=True):
                pair=(row['drug_id'],row['disease_id'])
                if pair in pairs:raise ValueError('duplicate split pair')
                pairs[pair]=row['split_assignment']
            self._assignments[split]=pairs
            if split in {'cold_drug','cold_disease'}:
                axis='drug' if split=='cold_drug' else 'disease';parts={}
                for entity,assignment in frame.select(f'{axis}_id',f'{axis}_partition').unique().iter_rows():
                    if assignment is not None:parts.setdefault(entity,set()).add(assignment)
                self._partitions[split]=parts

    def __getattr__(self,name):return getattr(self.scalar,name)

    def assemble(self,supervision,*,split_kind,device='cpu',evaluation_partition=None):
        if evaluation_partition not in {'validation','test'}:
            raise ValueError('indexed formal adapter is evaluation-only')
        if supervision.observed_relation is not None:
            raise ValueError('evaluation candidate assembly must not receive observed labels')
        pair=(supervision.drug_id,supervision.disease_id)
        universe=self.scalar._universes[split_kind]
        if pair in universe.observed_pairs and self._assignments[split_kind].get(pair) not in {'validation','test'}:
            raise ValueError('evaluation candidate must exclude training-known pairs')
        endpoints=self.index.node_global_ids(pair)
        if self.index.node_metadata(endpoints)['node_type'].to_list()!=['DRG','DIS']:
            raise ValueError('query endpoints must be DRG and DIS nodes')
        drug,disease=map(int,endpoints);targets=self._targets.get(pair,set())
        cold=None;mask=None
        if self.scalar.cold_visibility_policy_digest is not None and split_kind in {'cold_drug','cold_disease'}:
            cold=self.cold_visibility
            if cold is None or not isinstance(self.index,_ColdIndex):
                raise ValueError('cold query requires certified sanitized index')
            axis=0 if split_kind=='cold_drug' else 1
            if self._partitions[split_kind].get(pair[axis])!={evaluation_partition}:
                raise ValueError('current cold endpoint is not in requested partition')
            if split_kind not in self._cold_templates:
                # Reuse the authoritative implementation for permanent-mask
                # precedence. It reads the bound split once, not per candidate.
                self._cold_templates[split_kind]=cold.mask(self.scalar,split_kind,evaluation_partition,
                    pair,(drug,disease),targets,universe)
            mask=replace(self._cold_templates[split_kind],target_facts=targets,
                hidden_nodes=self.scalar._heldout_globals(split_kind,universe)-{int(endpoints[axis])})
            forbidden=cold_forbidden_filter(mask)
        else:
            blocked=self.scalar._forbidden_mask(split_kind)
            target_array=np.array(sorted(targets),dtype=np.int64)
            def forbidden(ids):return blocked[ids] | np.isin(ids,target_array)
        extracted=extract_csr_subgraph_native(self.scalar.formal_root/'graph/csr',seed_node_ids=[drug,disease],
            max_hops=self.scalar.max_hops,max_nodes=self.scalar.max_nodes,max_edges=self.scalar.max_edges,
            forbidden_filter=forbidden,csr_arrays=self.index._lookups['csr_readonly'])
        gathered=self.index.gather(extracted.fact_ids);facts=self.index.fact_metadata(extracted.fact_ids)
        arguments=dict(node_ids=extracted.node_ids,traversal_edges=extracted.traversal_edges,
                       drug_global=drug,disease_global=disease)
        batch=move_model_batch(tensorize_indexed(self.index,gathered,**arguments),device)
        batch.validate(self.config)
        source=self.scalar._build_path_source(facts,**arguments,split_kind=split_kind,
            query_pair_id=supervision.canonical_pair_key,target_fact_ids=targets,universe=universe,
            split_partition=evaluation_partition,node_rows=self.index.node_metadata(sorted(extracted.node_ids)))
        if cold is not None:
            source=replace(source,heldout_node_ids=frozenset(mask.hidden_nodes),
                split_digest=hashlib.sha256((source.split_digest+':'+self.scalar.cold_visibility_policy_digest).encode()).hexdigest())
            source.validate()
        target=np.isin(gathered['fact_ids'],np.array(sorted(targets),dtype=np.int64))
        audit=BatchAssemblyAudit(split_kind=split_kind,query_pair_id=supervision.canonical_pair_key,
            target_answer_fact_count=len(targets),target_answer_fact_survivor_count=int(target.sum()),
            split_forbidden_fact_survivor_count=int(forbidden(gathered['fact_ids']).sum()),
            target_evidence_survivor_count=int(target[gathered['evidence'].record_fact_index.numpy()].sum()),
            cached_target_pair_feature_survivor_count=0,target_endpoint_count=int(drug in extracted.node_ids)+int(disease in extracted.node_ids),
            node_count=len(extracted.node_ids),fact_count=len(extracted.fact_ids),
            evidence_record_count=len(gathered['evidence_ids']),traversal_edge_count=len(extracted.traversal_edges),
            reused_phase3_csr=True,reused_phase3_split_view=True)
        if any((audit.target_answer_fact_survivor_count,audit.split_forbidden_fact_survivor_count,audit.target_evidence_survivor_count)):
            raise RuntimeError('indexed evaluation leakage audit failed')
        return TrainBatch.from_model_batch(batch,(supervision,),split_kind=split_kind,path_source=source),audit
