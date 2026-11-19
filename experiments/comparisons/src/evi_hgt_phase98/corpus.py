"""Part2.1 physically masked cell corpora from certified immutable parents.

This module prepares data only: it never constructs a network or executes scoring.
It implements the original fold-corpus algorithm in the new authorized namespace.
No Phase93 permit is invented and no protected member/label file is opened.
"""
from pathlib import Path
import gc
import json
import os
import shutil
import threading
import time
import uuid

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import psutil

from evi_hgt.phase85_v2.common import DATA, bound, digest, read, verify
from evi_hgt.phase85_part2b.production import PHASE3, CLEAN, mask_endpoint_pairs
from evi_hgt.phase85_part3b1.data import assert_corpus_identity
from .common import ROOT, sha, write_new

OUT = ROOT/'outputs/phase9/phase98/part2_1/corpus'
CACHE = ROOT/'data/cache/phase98/corpora'
RESERVE_BYTES = 70_000_000_000
CELL_WORK_ALLOWANCE = 2_000_000_000
CELLS = tuple(f'R{r}F{f}' for r in range(3) for f in range(5))


def check_space(path, additional_bytes=CELL_WORK_ALLOWANCE):
    if type(additional_bytes) is not int or additional_bytes < 0:
        raise ValueError('Nonnegative exact byte allowance required')
    usage = shutil.disk_usage(Path(path).anchor)
    if usage.free < RESERVE_BYTES + additional_bytes:
        raise OSError('Insufficient corpus preparation disk headroom;70GB reserve preserved')
    return dict(free_bytes=usage.free,system_reserve_bytes=RESERVE_BYTES,additional_allowance_bytes=additional_bytes)


class HashVerifier:
    """Session cache hashes only while size and nanosecond mtime remain unchanged."""
    def __init__(self): self.cache = {}; self.bytes_hashed = 0
    def check(self, path, expected):
        p=Path(path);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
        value=self.cache.get(key)
        if value is None:
            value=sha(p);self.cache[key]=value;self.bytes_hashed+=s.st_size
        if value != expected: raise ValueError('Certified corpus source changed: '+str(p))
        return value


def roots_for(data):
    return [data.original_corpus_root,
        ROOT/f"outputs/phase9/phase92b/test_only/r{data.repeat_index}_f{data.cell['fold']}/corpus",
        ROOT/f"outputs/phase9/formal/PHASE9_G3_75_V1/data/r{data.repeat_index}_f{data.cell['fold']}/corpus",
        CACHE/data.cell_id]


def find_manifest(data):
    for root in roots_for(data):
        if (root/'manifest.json').is_file(): return root/'manifest.json'
    raise FileNotFoundError('No certified current-cell corpus; run prepare_corpora first')


def verify_manifest(data, path=None, *, verifier=None):
    path=Path(path or find_manifest(data));data.corpus_root=path.parent
    corpus=verify(read(path));assert_corpus_identity(data,corpus)
    verifier=verifier or HashVerifier()
    for p,h in {**corpus['inputs'],**corpus['outputs']}.items():verifier.check(p,h)
    return corpus


class ParentSources:
    def __init__(self, verifier=None):
        self.verifier=verifier or HashVerifier()
        self.clean=verify(read(CLEAN/'manifest.json'))
        acceptance=verify(read(ROOT/'outputs/phase85_part2b/graph_acceptance.json'))
        if (acceptance['graph_provenance']!='PASS' or acceptance['topology']!='PASS'
            or acceptance['clean_graph_digest']!=self.clean['digest']):raise ValueError('Clean graph gate failed')
        for p,h in {**self.clean['inputs'],**self.clean['files']}.items():self.verifier.check(p,h)
        self.certified=verify(read(ROOT/'data/processed/phase85_part2b/pilot_adapter_v1/manifest.json'))
        self.source_hashes={**self.certified['inputs'],**self.certified['outputs']}
        self.source_sha(PHASE3/'views/query_target_fact_map.parquet')
        self.targets=pl.read_parquet(PHASE3/'views/query_target_fact_map.parquet',columns=['drug_id','disease_id','canonical_fact_id'])
        self.source_sha(PHASE3/'nodes/node_index.parquet')
        self.source_sha(PHASE3/'graph/csr/edge_type_catalog.json')
        self.csr={}
        for name in ('indptr','indices','fact_ids','edge_type_ids'):
            path=PHASE3/'graph/csr'/f'{name}.npy';self.source_sha(path)
            self.csr[name]=np.load(path,mmap_mode='r',allow_pickle=False)
        # No role column or protected labels: this is the original legal-source
        # ID table, subsequently restricted by the current-cell complement mask.
        legal_path=DATA/'legal_source_rows.parquet'
        development=verify(read(DATA/'development_manifest.json'))
        self.legal_source_sha256=self.verifier.check(legal_path,development['files']['legal_source_rows.parquet'])
        self.legal_ids=pl.read_parquet(legal_path,columns=['drug_id','disease_id'])
        self.identities={}
        for f in sorted((PHASE3/'canonical_facts').glob('*.parquet')):
            if f.stem not in ('drug_disease','disease_gene'):
                self.source_sha(f);self.source_sha(PHASE3/'evidence_records'/f.name)
                self.identities[f.stem]=pl.read_parquet(f,columns=['canonical_fact_id','source_node_id','target_node_id','source_node_type','target_node_type'])

    def source_sha(self,path):
        path=Path(path);expected=self.source_hashes.get(str(path))
        if expected is None:raise ValueError('Path outside certified parent: '+str(path))
        return self.verifier.check(path,expected)


def materialize(data, parent):
    if data.cell_id not in CELLS:raise ValueError('Only the original fifteen cells')
    try:
        path=find_manifest(data)
    except FileNotFoundError: path=None
    if path is not None:
        return verify_manifest(data,path,verifier=parent.verifier),dict(reused=True,new_bytes=0)
    started=time.perf_counter();space=check_space(CACHE)
    dest=(CACHE/data.cell_id).resolve();base=CACHE.resolve()
    if not dest.is_relative_to(base) or dest==base:raise ValueError('New corpus must remain under Phase98 cache')
    if dest.exists():raise FileExistsError('Uncommitted corpus directory retained; manual inspection required')
    CACHE.mkdir(parents=True,exist_ok=True)
    staging=CACHE/(data.cell_id+'.attempt_'+uuid.uuid4().hex)
    if not staging.resolve().is_relative_to(base):raise ValueError('Invalid staging root')
    staging.mkdir();outputs={};inputs={}
    def source(path):inputs[str(path)]=parent.source_sha(path);return path
    def write(frame,name):
        p=staging/name;p.parent.mkdir(parents=True,exist_ok=True);frame.write_parquet(p)
        final=dest/name;outputs[str(final)]=sha(p);return final
    def local_path(p):
        p=Path(p)
        return staging/p.relative_to(dest) if p.is_relative_to(dest) else p
    try:
        source(PHASE3/'nodes/node_index.parquet');source(PHASE3/'graph/csr/edge_type_catalog.json')
        source(PHASE3/'views/query_target_fact_map.parquet')
        inputs[str(DATA/'legal_source_rows.parquet')]=parent.legal_source_sha256
        blocked=pl.DataFrame(sorted(data.blocked),schema=['drug_id','disease_id'],orient='row')
        forbidden=parent.targets.join(blocked,on=['drug_id','disease_id'],how='semi')['canonical_fact_id'].unique()
        write(pl.DataFrame({'canonical_fact_id':forbidden}),'forbidden_fact_ids.parquet')
        legal=parent.legal_ids.join(blocked,on=['drug_id','disease_id'],how='anti')
        dda_path=source(PHASE3/'canonical_facts/drug_disease.parquet')
        dda=(pl.scan_parquet(dda_path).join(legal.lazy().rename({'drug_id':'source_node_id','disease_id':'target_node_id'}),
            on=['source_node_id','target_node_id'],how='semi').filter(~pl.col('canonical_fact_id').is_in(forbidden.implode())).collect())
        if mask_endpoint_pairs(dda,data.blocked).height!=dda.height:raise ValueError('Forbidden endpoint entered DDA graph')
        fact_files=[];evidence_files=[]
        for f in sorted((PHASE3/'canonical_facts').glob('*.parquet')):
            e=PHASE3/'evidence_records'/f.name
            if f.stem=='drug_disease':
                fp=write(dda,'canonical_facts/drug_disease.parquet');source(e)
                ep=write(pl.scan_parquet(e).join(dda.lazy().select('canonical_fact_id'),on='canonical_fact_id',how='semi').collect(),'evidence_records/drug_disease.parquet')
            elif f.stem=='disease_gene':
                fp=CLEAN/'canonical_facts.parquet';ep=Path(next(v['path'] for v in parent.certified['evidence_files'] if v['component']=='disease_gene'))
                source(fp);source(ep)
            else:
                fp,ep=f,e;source(f);source(e);ids=parent.identities[f.stem]
                safe=mask_endpoint_pairs(ids,data.blocked).filter(~pl.col('canonical_fact_id').is_in(forbidden.implode()))
                if safe.height!=ids.height:
                    selected=safe.lazy().select('canonical_fact_id')
                    fp=write(pl.scan_parquet(f).join(selected,on='canonical_fact_id',how='semi').collect(),'canonical_facts/'+f.name)
                    ep=write(pl.scan_parquet(e).join(selected,on='canonical_fact_id',how='semi').collect(),'evidence_records/'+e.name)
            fact_files.append(dict(path=str(fp),component=f.stem,rows=pq.ParquetFile(local_path(fp)).metadata.num_rows))
            evidence_files.append(dict(path=str(ep),component=f.stem,rows=pq.ParquetFile(local_path(ep)).metadata.num_rows))
        admitted=np.concatenate([pl.read_parquet(local_path(v['path']),columns=['canonical_fact_id']).to_series().to_numpy() for v in fact_files])
        if len(np.unique(admitted))!=len(admitted) or np.isin(admitted,forbidden.to_numpy()).any():raise ValueError('Duplicate/forbidden fact in corpus')
        admitted.sort();keep=np.isin(parent.csr['fact_ids'],admitted)
        prefix=np.empty(len(keep)+1,dtype=np.int64);prefix[0]=0;np.cumsum(keep,dtype=np.int64,out=prefix[1:])
        csr={'indptr':prefix[parent.csr['indptr']]}
        for name in ('indices','fact_ids','edge_type_ids'):csr[name]=parent.csr[name][keep]
        for name,values in csr.items():
            source(PHASE3/'graph/csr'/f'{name}.npy')
            check_space(CACHE,values.nbytes+CELL_WORK_ALLOWANCE)
            p=staging/'csr'/f'{name}.npy';p.parent.mkdir(exist_ok=True)
            with p.open('xb') as h:np.save(h,values,allow_pickle=False);h.flush();os.fsync(h.fileno())
            outputs[str(dest/'csr'/f'{name}.npy')]=sha(p)
        legal_targets=parent.targets.filter(pl.col('canonical_fact_id').is_in(pl.Series(admitted).implode()))
        write(legal_targets,'legal_target_map.parquet')
        result=bound(dict(schema='PART3B1_FOLD_LOCAL_CORPUS_V1',corpus_root=str(dest),
            bindings=dict(data.bindings,clean_graph_digest=parent.clean['digest']),inputs=inputs,outputs=outputs,
            fact_files=fact_files,evidence_files=evidence_files,fact_count=sum(v['rows'] for v in fact_files),
            evidence_count=sum(v['rows'] for v in evidence_files),traversal_count=len(csr['fact_ids']),
            target_facts_permanently_removed=len(forbidden),legal_DDA_graph_facts=dda.height,
            clean_DG_facts=parent.clean['clean_fact_count'],certified_unlabeled_source_corpus_digest=parent.certified['digest'],
            source_properties_rebuilt=False,clean_DG_properties_reused=True,fold_local_DDA_and_CSR_materialized=True,
            pilot_DDA_or_pilot_CSR_reused=False,contaminated_DG_evidence_imported=False,heldout_DDA_properties_imported=False,
            CTD_used=False,role_free_target_map_projection=True))
        for p,h in outputs.items():
            if sha(local_path(p))!=h:raise ValueError('Staged payload changed')
        # All paths are verified to stay under the new named workspace cache.
        # Whole-directory atomic publication preserves partial attempts on error.
        write_new(staging/'manifest.json',result)
        os.rename(staging,dest)
        data.corpus_root=dest;assert_corpus_identity(data,result)
        return result,dict(reused=False,new_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),
            elapsed_seconds=time.perf_counter()-started,disk_before=space,publish='ATOMIC_DIRECTORY_RENAME',
            source_sha256=sha(__file__),provenance='Original fold algorithm; new permitted root; no full KG reconstruction')
    except BaseException as exc:
        if staging.exists():
            p=staging/'FAILED.json'
            if not p.exists():write_new(p,dict(error=repr(exc),partial_payloads_retained=True))
        raise


def loader_acceptance(data, corpus):
    """Actual index/assembler and one fixed legal training query, no network."""
    from evi_hgt.phase9.real_data import load_index,load_canonical_assembler
    started=time.perf_counter();index=load_index(corpus);loaded=time.perf_counter()
    assembler=load_canonical_assembler(data,corpus,index)
    pair=min(data.support);query=data.positive(pair,'support')
    graph=assembler.assemble(query,domain='training',device='cpu')
    batch=graph.batch.model_batch
    result=dict(status='PASS_REAL_INDEX_ASSEMBLER_TRAINING_QUERY_NO_MODEL',index_load_seconds=loaded-started,
        total_loader_seconds=time.perf_counter()-started,fixture_rule='lexicographically first training support pair',
        fixture_pair_sha256=digest(list(pair)),nodes=batch.nodes.num_nodes,messages=batch.edges.num_edges,
        no_network_or_model_constructed=True,model_forwards=0,query_scores=0,optimizer_updates=0)
    del graph,assembler,index;gc.collect()
    return result


def prepare_all(*,accept_loaders=True):
    from .data import TrainingInputs
    OUT.mkdir(parents=True,exist_ok=True);run=OUT/('run_'+uuid.uuid4().hex);run.mkdir()
    started=time.perf_counter();disk_before=check_space(CACHE,15*CELL_WORK_ALLOWANCE)
    process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.2):peak[0]=max(peak[0],process.memory_info().rss)
    watcher=threading.Thread(target=sample,daemon=True);watcher.start()
    rows=[];inputs=TrainingInputs();parent=ParentSources()
    try:
        for cell in CELLS:
            t=time.perf_counter();data=inputs.cell(cell)
            corpus,cost=materialize(data,parent)
            verified=verify_manifest(data,verifier=parent.verifier)
            acceptance=loader_acceptance(data,verified) if accept_loaders else {'status':'NOT_YET_RUN'}
            row=bound(dict(cell=cell,corpus_root=corpus['corpus_root'],corpus_digest=corpus['digest'],
                manifest_sha256=sha(Path(corpus['corpus_root'])/'manifest.json'),mask_digest=data.bindings['fold_mask_digest'],
                data_bindings=data.bindings,projection_audit=data.projection_audit,cost=cost,loader=acceptance,
                elapsed_seconds=time.perf_counter()-t,corpus_fact_count=corpus['fact_count'],traversal_count=corpus['traversal_count']))
            write_new(run/(cell+'.json'),row);rows.append(row)
            print(json.dumps(dict(cell=cell,status=acceptance['status'],reused=cost['reused'],seconds=row['elapsed_seconds']),ensure_ascii=False),flush=True)
            del data;gc.collect()
        result=bound(dict(schema='PHASE98_PART21_CORPUS_PREPARATION_V1',status='PASS_15_ACTUAL_CORPORA_AND_LOADERS' if accept_loaders else '15_PREPARED_LOADER_CHECK_PENDING',
            cells=rows,elapsed_seconds=time.perf_counter()-started,new_corpus_count=sum(not r['cost']['reused'] for r in rows),
            new_corpus_bytes=sum(r['cost']['new_bytes'] for r in rows),peak_cpu_rss_bytes=peak[0],rss_sampling_seconds=.2,
            source_bytes_hashed=parent.verifier.bytes_hashed,disk_before=disk_before,disk_after=check_space(CACHE,0),
            model_constructions=0,model_forwards=0,optimizer_updates=0,query_scores=0,raw_DrugBank_accesses=0,
            protected_outer_final_member_label_decodes=0,role_free_exclusion_ids_used=True,
            shared_assignment_container_bytes_scanned_or_hashed=True,network_calls=0,full_OptimusKG_reconstruction=False,
            source_sha256={str(Path(__file__)):sha(__file__),str(ROOT/'src/evi_hgt_phase98/data.py'):sha(ROOT/'src/evi_hgt_phase98/data.py')}))
        write_new(run/'ALL_CORPORA.json',result);return run/'ALL_CORPORA.json'
    except BaseException as exc:
        write_new(run/'FAILED.json',dict(error=repr(exc),completed_cells=[r['cell'] for r in rows],elapsed_seconds=time.perf_counter()-started,
            model_forwards=0,optimizer_updates=0,partial_artifacts_retained=True));raise
    finally:stop.set();watcher.join()
