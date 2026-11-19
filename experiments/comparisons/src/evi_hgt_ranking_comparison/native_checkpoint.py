"""New-root native checkpoint bridge; historical modules and files stay intact.

Raw HGTDR/Tx snapshots retain their native schema and tensor control state.
Publication sequence is independent of optimizer steps. HGT thinning requires
explicit receiver/replay/retention receipts; Tx keeps all committed payloads.
The private Tx module only redirects its commit verifier; its old formal
constructor/runner authorization is not replaced or implicitly authorized.
"""
import copy
import gzip
import tempfile
import importlib.util
import os
from pathlib import Path
import pickle
import sys
import uuid

import torch
from evi_hgt.ranking_protocol.production_state import _snapshot,_finite,_validate_rng,training_state_digest
from .checkpoint import _check_optimizer
from .common import ROOT,OUT,digest,sha,read,write_new,append

KINDS={'HGTDR':'hgtdr_adapter.py','TXGNN':'txgnn_recovery.py'}
SCHEMAS={'HGTDR':'RANKING_PROTOCOL_COMPLETE_TRAINING_STATE_V1','TXGNN':'TXGNN_PILOT_TXGNN_COMPLETE_STATE_V1'}
TX_FIELDS={'schema','identity','recipe','source','graph_binding','eid_binding','scope','cursor',
    'permutation','model','auxiliary_gates','optimizer','optimizer_recipe','scheduler',
    'scheduler_recipe','rng','module_modes','stream','transition','selection','external_control','state_digest'}
HGT_FIELDS={'schema','identity','cursor','sampler','model_metadata','implementation_sha256',
    'model','optimizer','optimizer_recipe','scheduler','scheduler_recipe','rng','module_modes','state_digest'}


def source_bindings(kind):
    if kind not in KINDS:raise ValueError('Unknown native kind')
    paths=[ROOT/'src/evi_hgt/ranking_protocol'/KINDS[kind],ROOT/'src/evi_hgt/ranking_protocol/production_state.py']
    return {str(path):sha(path) for path in paths}


def _weights(values,model):
    if not isinstance(values,dict) or values.keys()!=model.keys() or any(not isinstance(v,torch.Tensor) or v.shape!=model[k].shape or v.dtype!=model[k].dtype for k,v in values.items()):
        raise ValueError('Native selected tensor schema differs')


def validate_tx_selected_control(control,epoch,model):
    from evi_hgt.ranking_protocol.txgnn_protocol import InnerSelection
    selected=InnerSelection.from_state(control['selection']) if 'selection' in control else InnerSelection()
    opportunities=epoch//25;allowed={opportunities}
    if epoch and epoch%25==0:allowed.add(opportunities-1)
    if len(selected.history) not in allowed:raise ValueError('Tx selection history does not match native completed epochs')
    best=control.get('best_model')
    if (selected.best is None)!=(best is None):raise ValueError('Tx selection/best tensor state is incomplete')
    if best is not None:_weights(best,model)


def validate_raw_state(state,kind,identity):
    """No receiver/model construction; native restore adds live-layout checks."""
    fields=HGT_FIELDS if kind=='HGTDR' else TX_FIELDS
    if not isinstance(state,dict) or set(state)!=fields or state['schema']!=SCHEMAS[kind] or state['identity']!=identity:
        raise ValueError('Native schema/identity differs')
    if training_state_digest({k:v for k,v in state.items() if k!='state_digest'})!=state['state_digest']:raise ValueError('Native state digest differs')
    _finite(state);_validate_rng(state['rng'])
    if not isinstance(state['model'],dict) or not state['model'] or any(not isinstance(v,torch.Tensor) for v in state['model'].values()):raise ValueError('Native model tensors missing')
    if not isinstance(state['module_modes'],dict) or '' not in state['module_modes'] or any(type(v) is not bool for v in state['module_modes'].values()):raise ValueError('Native mode schema differs')
    c=state['cursor']
    if type(c.get('actual_optimizer_updates')) is not int or c['actual_optimizer_updates']<0:raise ValueError('Invalid native update cursor')
    active=c['actual_optimizer_updates']
    if kind=='HGTDR':
        if state['implementation_sha256']!=sha(ROOT/'src/evi_hgt/ranking_protocol/production_state.py'):raise ValueError('Native state helper source differs')
        control=state['sampler'].get('native_control')
        if not isinstance(control,dict):raise ValueError('Native control missing')
        nc=control['cursor'];selection=control['selection']
        expected=dict(actual_optimizer_updates=nc['optimizer_steps'],completed_epochs=nc['epoch']-1,
            epoch=nc['epoch'],batch_position=nc['batch_index'],selection_opportunities=selection['opportunities'])
        if c!=expected or any(type(v) is not int or v<0 for v in c.values()):raise ValueError('HGTDR control/cursor differs')
        history=selection['history'];count=selection['opportunities']
        if not 0<=count<=min(c['completed_epochs'],300) or len(history)!=count or [r['epoch'] for r in history]!=list(range(1,count+1)):raise ValueError('HGTDR selection history differs')
        best=max(history,key=lambda r:(r['metrics']['macro_auc'],r['metrics']['macro_ap'],-r['epoch'])) if history else None
        if selection['best']!=best or (best is None)!=(control['best_model'] is None):raise ValueError('HGTDR selected state missing/inconsistent')
        if best is not None:_weights(control['best_model'],state['model'])
    else:
        if set(c)!={'stage','pretrain_epoch','pretrain_batch','pretrain_updates','finetune_epoch','finetune_updates','actual_optimizer_updates','consumed_edges','negative_batches'} or any(type(v) is not int or v<0 for k,v in c.items() if k!='stage'):raise ValueError('Tx stage cursor malformed')
        if c['stage'] not in ('PRETRAIN','FINETUNE') or c['pretrain_epoch']!=1 or c['pretrain_updates']!=c['pretrain_batch'] or c['finetune_updates']!=c['finetune_epoch'] or c['actual_optimizer_updates']!=c['pretrain_updates']+c['finetune_updates'] or c['negative_batches']!=c['actual_optimizer_updates']:raise ValueError('Tx stage/update accounting differs')
        permutation=state['permutation'];scope=state['scope']
        if not isinstance(permutation,torch.Tensor) or permutation.dtype!=torch.int64 or permutation.ndim!=1 or not torch.equal(torch.sort(permutation).values,torch.arange(len(permutation))):raise ValueError('Tx exact loader permutation differs')
        if c['pretrain_batch']>scope['planned_pretrain_batches'] or c['finetune_epoch']>500 or c['consumed_edges']!=min(c['pretrain_batch']*scope['batch_size'],len(permutation)):raise ValueError('Tx loader cursor differs')
        active=c['pretrain_updates'] if c['stage']=='PRETRAIN' else c['finetune_updates']
        if c['stage']=='PRETRAIN':
            if state['transition'] is not None or c['finetune_epoch'] or state['scheduler'] is not None:raise ValueError('Illegal Tx pretrain transition')
        elif c['pretrain_batch']!=scope['planned_pretrain_batches'] or not isinstance(state['transition'],dict) or state['transition'].get('decoder_reset_count')!=1 or state['scheduler'] is None:raise ValueError('Tx completed pretrain/reset proof missing')
        if not isinstance(state['auxiliary_gates'],dict) or not isinstance(state['stream'],dict) or not isinstance(state['external_control'],dict):raise ValueError('Tx auxiliary/native control missing')
        best=state['external_control'].get('best_model')
        validate_tx_selected_control(state['external_control'],c['finetune_epoch'],state['model'])
    # Native Tx deliberately resets Adam at the transition. Validate moments
    # against the active-stage updates, never total two-stage updates.
    names=[n for g in state['optimizer_recipe']['groups'] for n in g['parameter_names']]
    _check_optimizer(dict(state,step=active,parameter_requires_grad={n:True for n in names}))


def gzip_plan(state):
    """Count the complete gzip stream without a disk write or omitted state."""
    class Counter:
        size=0
        def write(self,data):self.size+=len(data);return len(data)
        def flush(self):pass
    counter=Counter()
    with gzip.GzipFile(filename='',fileobj=counter,mode='wb',compresslevel=1,mtime=0) as stream:
        torch.save(state,stream);decoded=stream.tell()
    return dict(storage_codec='gzip-1',uncompressed_size=decoded,compressed_size=counter.size)


class NativeCheckpointStore:
    """Native commit(name,state)/load/verify API; retention remains retain-all.

    validate_state is mandatory for source/data/control checks specific to the
    caller's frozen native contract. It is run before publication and every read.
    """
    def __init__(self,root,identity,*,kind,native_identity,bindings,validate_state,capability=None,retain_all=True,retention_evidence=None):
        if kind not in KINDS or not callable(validate_state):raise ValueError('Native kind and strict validation callback required')
        if type(retain_all) is not bool or (kind=='TXGNN' and not retain_all):raise ValueError('Tx native payload thinning is UNQUALIFIED')
        self.retain_all=retain_all
        if not retain_all and identity.get('mode')=='RANKING_COMPARISON_FORMAL':
            if not retention_evidence:raise PermissionError('Runtime recovery and retention qualification receipt required')
            evidence=read(retention_evidence['path'])
            required=('HGTDR_recovery','HGTDR_retention','HGTDR_new_process_recovery','HGTDR_selection_control')
            if sha(retention_evidence['path'])!=retention_evidence['sha256'] or any(evidence.get(k)!='PASS' for k in required) or evidence.get('native_store_sha256')!=sha(__file__):raise PermissionError('Native retention qualification differs')
        self.root=Path(root).resolve();self.identity=copy.deepcopy(identity);self.kind=kind
        self.native_identity=copy.deepcopy(native_identity);self.bindings=copy.deepcopy(bindings);self.validator=validate_state
        if bindings!=source_bindings(kind):raise ValueError('Native source bindings differ')
        if native_identity.get('method')!=('M1' if kind=='HGTDR' else 'M4'):raise ValueError('Wrong native method identity')
        if any(not identity.get(k) for k in ('run_id','method','cell','source_digest','data_digest')):raise ValueError('Missing new run identity')
        allowed=(OUT/'TEST_ONLY').resolve()
        if identity.get('mode')=='RANKING_COMPARISON_FORMAL':
            from .authorization import ExecutionCapability
            if not isinstance(capability,ExecutionCapability):raise PermissionError('New RankingComparison G3 capability required')
            capability.permits(identity,self.root.parent if self.root.name=='checkpoints' else self.root);allowed=(ROOT/'outputs/ranking_protocol/ranking_comparison/runs').resolve()
        elif identity.get('mode')!='TEST_ONLY':raise PermissionError('Unknown native store mode')
        if identity.get('mode')=='TEST_ONLY' and self.root.is_relative_to((ROOT/'outputs/ranking_protocol/ranking_comparison/part2_1/TEST_ONLY').resolve()):
            allowed=(ROOT/'outputs/ranking_protocol/ranking_comparison/part2_1/TEST_ONLY').resolve()
        if identity.get('mode')=='TEST_ONLY' and self.root.is_relative_to((ROOT/'outputs/ranking_protocol/ranking_comparison/part2_2/TEST_ONLY').resolve()):
            allowed=(ROOT/'outputs/ranking_protocol/ranking_comparison/part2_2/TEST_ONLY').resolve()
        if identity.get('mode')=='TEST_ONLY' and self.root.is_relative_to((ROOT/'outputs/ranking_protocol/ranking_comparison/part2_3/TEST_ONLY').resolve()):
            allowed=(ROOT/'outputs/ranking_protocol/ranking_comparison/part2_3/TEST_ONLY').resolve()
        if identity.get('mode')=='TEST_ONLY' and self.root.is_relative_to((ROOT/'outputs/ranking_protocol/ranking_comparison/part2_4/TEST_ONLY').resolve()):
            allowed=(ROOT/'outputs/ranking_protocol/ranking_comparison/part2_4/TEST_ONLY').resolve()
        if self.root==allowed or not self.root.is_relative_to(allowed):raise PermissionError('Isolated new RankingComparison root required')
        self.root.mkdir(parents=True,exist_ok=True)

    def _inside(self,path):
        p=Path(path).resolve()
        if p.parent!=self.root:raise PermissionError('Native store path escape')
        return p

    def _validate(self,state):
        if self.bindings!=source_bindings(self.kind):raise ValueError('Native source changed during run')
        try:
            validate_raw_state(state,self.kind,self.native_identity)
            before=state['state_digest'];self.validator(state)
            if state['state_digest']!=before or training_state_digest({k:v for k,v in state.items() if k!='state_digest'})!=before:raise ValueError('Validation callback changed native state')
        except (TypeError,KeyError,AttributeError,IndexError,RuntimeError) as exc:raise ValueError('Malformed native complete state') from exc
        if self.kind=='TXGNN' and state['transition'] is not None:
            transition=state['transition'];source=self._header(transition['pretrain_commit_path'])
            # Reject a cycle/wrong-stage link before recursively validating its
            # payload. The complete pretrain source remains permanently stored.
            if source['cursor']['stage']!='PRETRAIN' or source['cursor']['pretrain_batch']!=state['scope']['planned_pretrain_batches'] or source['cursor']['finetune_updates']!=0:raise ValueError('Tx transition source is not completed pretrain')
            source=self.verify(transition['pretrain_commit_path'])
            if source['state_digest']!=transition['pretrain_state_digest'] or source['commit_digest']!=transition['pretrain_commit_digest'] or source['identity']!=self.native_identity:raise ValueError('Tx transition source differs')

    def _header(self,path):
        p=self._inside(path);r=read(p)
        if r.get('schema')!='RANKING_COMPARISON_NATIVE_COMMIT_V1' or r.get('run_identity')!=self.identity or r.get('kind')!=self.kind or r.get('checkpoint_commit_path')!=str(p) or r.get('identity')!=self.native_identity or r.get('source_bindings')!=self.bindings:raise ValueError('Native commit identity differs')
        if digest({k:v for k,v in r.items() if k!='commit_digest'})!=r['commit_digest']:raise ValueError('Native COMMIT corrupted')
        self._inside(r['checkpoint_path']);return r

    def load(self,path):
        r=self._header(path);p=self._inside(r['checkpoint_path'])
        if p.stat().st_size!=r['checkpoint_size'] or sha(p)!=r['checkpoint_sha256']:raise ValueError('Native payload corrupted')
        codec=r.get('storage_codec','torch-raw')
        if codec=='torch-raw':state=torch.load(p,map_location='cpu',weights_only=True)
        elif codec=='gzip-1' and self.kind=='TXGNN':
            size=r.get('uncompressed_size')
            if type(size) is not int or size<=0:raise ValueError('Compressed checkpoint size missing')
            from .runtime_budget import disk_guard
            disk_guard(self.root,size+10_000_000)
            # Random-access torch ZIP loading must not repeatedly seek a gzip
            # stream. One bounded temporary file is closed before validation.
            with tempfile.TemporaryFile(mode='w+b',dir=self.root,prefix='native_read_',suffix='.partial') as temporary:
                total=0
                with gzip.open(p,'rb') as compressed:
                    while True:
                        block=compressed.read(8*1024*1024)
                        if not block:break
                        total+=len(block)
                        if total>size:raise ValueError('Compressed checkpoint exceeds declared size')
                        temporary.write(block)
                if total!=size:raise ValueError('Compressed checkpoint truncated')
                temporary.seek(0);state=torch.load(temporary,map_location='cpu',weights_only=True)
        else:raise ValueError('Unknown native checkpoint codec')
        self._validate(state)
        if state['state_digest']!=r['state_digest'] or state['cursor']!=r['cursor']:raise ValueError('Native receipt/state differs')
        return state

    def verify(self,path):
        self.load(path);return self._header(path)

    def commit(self,name,state,*,fault=None,tags=()):
        if not isinstance(name,str) or not name or len(name)>96 or any(not(c.isascii() and (c.isalnum() or c in '_-')) for c in name):raise ValueError('Safe native checkpoint name required')
        if fault not in (None,'after_payload') or (fault and self.identity['mode']!='TEST_ONLY'):raise PermissionError('Only bounded TEST_ONLY interruption allowed')
        state=_snapshot(state);self._validate(state)
        from .runtime_budget import disk_guard
        from .resource_guard import checkpoint_required_bytes
        plan=gzip_plan(state) if getattr(self,'kind',None)=='TXGNN' else None
        estimated=(max(plan['uncompressed_size'],plan['compressed_size'])+10_000_000 if plan else checkpoint_required_bytes(state))
        disk_guard(self.root,estimated)
        prior=[self._header(p) for p in self.root.glob('native_*.COMMIT.json')]
        same=[r for r in prior if r['native_name']==name]
        if same:
            if len(same)!=1 or same[0]['state_digest']!=state['state_digest']:raise FileExistsError('Immutable native named commit collision')
            return self.verify(same[0]['checkpoint_commit_path'])
        sequence=max((r['publication_sequence'] for r in prior),default=0)+1
        stem=f'native_{sequence:08d}.{uuid.uuid4().hex}'
        payload=self.root/(stem+'.state.pt');partial=self.root/(stem+'.state.partial');marker=self.root/(stem+'.COMMIT.json')
        with partial.open('xb') as f:
            if plan:
                with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=1,mtime=0) as compressed:torch.save(state,compressed)
            else:torch.save(state,f)
            f.flush();os.fsync(f.fileno())
        if plan and partial.stat().st_size!=plan['compressed_size']:raise ValueError('Compressed write size differs from prewrite plan')
        os.rename(partial,payload)
        attempt=dict(schema='RANKING_COMPARISON_NATIVE_ATTEMPT_V1',path=str(payload),sha256=sha(payload),bytes=payload.stat().st_size,native_name=name,publication_sequence=sequence)
        write_new(self.root/(stem+'.ATTEMPT.json'),attempt)
        if fault:raise RuntimeError('TEST_ONLY interrupted native publication')
        r=dict(schema='RANKING_COMPARISON_NATIVE_COMMIT_V1',status='COMMITTED',kind=self.kind,run_identity=self.identity,
            identity=self.native_identity,native_name=name,publication_sequence=sequence,source_bindings=self.bindings,
            checkpoint_path=str(payload),checkpoint_commit_path=str(marker),checkpoint_sha256=attempt['sha256'],
            checkpoint_size=attempt['bytes'],state_digest=state['state_digest'],cursor=state['cursor'],retention='RETAIN_ALL_NATIVE_COMPATIBILITY' if self.retain_all else 'RANKING_COMPARISON_LATEST_PREVIOUS_AND_PINNED_V1',tags=sorted(set(tags)),
            stage_source_anchor=self.kind=='TXGNN' and name=='pretrain_complete')
        if plan:r.update(storage_codec=plan['storage_codec'],uncompressed_size=plan['uncompressed_size'])
        r['commit_digest']=digest(r);marker_partial=self.root/(stem+'.commit.partial')
        write_new(marker_partial,r);os.rename(marker_partial,marker)
        return self.verify(marker)

    def latest(self):
        failures=[]
        for p in sorted(self.root.glob('native_*.COMMIT.json'),reverse=True):
            try:
                receipt=self.verify(p)
                if failures:append(self.root/'RECOVERY.jsonl',dict(event='NATIVE_FALLBACK',anchor=str(p),rejected=failures))
                return receipt
            except (ValueError,OSError,KeyError,RuntimeError,EOFError,pickle.UnpicklingError) as exc:failures.append(dict(path=str(p),error=str(exc)))
        raise ValueError('No valid native recovery anchor: '+repr(failures))

    def collect(self,*,protected=()):
        """HGT-only bounded retention; selected best is embedded in both anchors.

        Explicit tags/pins protect endpoint and externally referenced milestones.
        No historical root is accepted by the constructor. Failure/orphan files
        are never candidates. Existing reclaimed headers remain audit-only.
        """
        records=[self._header(p) for p in self.root.glob('native_*.COMMIT.json')]
        records.sort(key=lambda r:r['publication_sequence'])
        live=[r for r in records if Path(r['checkpoint_path']).exists()]
        for r in live:self.verify(r['checkpoint_commit_path'])
        if self.retain_all:return []
        if len(live)<3:return []
        pins={str(self._inside(p)) for p in protected}
        if not pins<={r['checkpoint_commit_path'] for r in live}:raise ValueError('Explicit dependency pin is unavailable')
        anchors=live[-2:]
        keep=pins|{r['checkpoint_commit_path'] for r in anchors}|{r['checkpoint_commit_path'] for r in live if r.get('tags') or r['stage_source_anchor']}
        reclaimed=[]
        for r in live:
            if r['checkpoint_commit_path'] in keep:continue
            # Every required live state was validated before any deletion.
            p=self._inside(r['checkpoint_path'])
            event=dict(event='RECLAIM_INTENT',payload=str(p),sha256=r['checkpoint_sha256'],bytes=r['checkpoint_size'],
                reason='two newer complete native receiver anchors; untagged and unpinned',
                replacement_anchors=[x['checkpoint_commit_path'] for x in anchors],arbitrary_historic_resume=False)
            append(self.root/'RECLAIM.jsonl',event);p.unlink()
            append(self.root/'RECLAIM.jsonl',dict(event,event='RECLAIM_COMPLETE'));reclaimed.append(event)
        return reclaimed


def load_private_native_module(store):
    """Load unchanged native source in its own module dictionary.

    Tx transition/restore resolve verify_tx_commit through that private dictionary.
    No global historical module, source file, grant or runtime model is changed.
    The old run_formal entry remains unsuitable for a RankingComparison grant.
    """
    if not isinstance(store,NativeCheckpointStore):raise TypeError('Native store required')
    if store.bindings!=source_bindings(store.kind):raise ValueError('Native source changed')
    path=ROOT/'src/evi_hgt/ranking_protocol'/KINDS[store.kind]
    name='evi_hgt.ranking_protocol._ranking_comparison_private_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:sys.modules.pop(name,None);raise
    if store.kind=='TXGNN':module.verify_tx_commit=store.verify
    module._ranking_comparison_native_store=store
    return module


def restore_native(store,trainer,commit_path,private_module):
    """Actual read/restore adapter; caller must construct the authorized receiver."""
    if getattr(private_module,'_ranking_comparison_native_store',None) is not store:raise ValueError('Private native module/store binding differs')
    state=store.load(commit_path)
    if store.kind=='TXGNN':
        if type(trainer) is not private_module.TxTwoStageTrainer:raise ValueError('Tx receiver must use private verifier module')
        trainer.restore(state)
    else:
        if type(trainer) is not private_module.SingleLRTrainer:raise ValueError('HGTDR receiver must use bound private module')
        private_module.restore_native_state(state,trainer,repeat=store.native_identity['repeat'],fold=store.native_identity['fold'])
    return dict(commit=store.verify(commit_path),restored_native_state_digest=state['state_digest'])
