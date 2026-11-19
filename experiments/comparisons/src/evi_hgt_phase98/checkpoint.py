"""New-run complete-state commits with auditable payload retention.

Readers explicitly support reclaimed intermediate payloads. Receipts remain;
arbitrary historic-step recovery is not promised. Never operates on old roots.
Attempt payload names are resolved through new COMMIT receipts. Historical
readers/roots are unchanged; compatibility with their cursor schemas is not
claimed. Collection requires all extant committed payloads to validate, and
failed attempt bytes/receipts are retained indefinitely for explicit review.
"""
import copy,math,os,pickle,random,uuid
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,OUT,canonical,digest,read,sha,write_new,append
from evi_hgt.phase9.production_state import _snapshot,_finite,_rng,_validate_rng,training_state_digest,_optimizer_recipe,_validate_optimizer,_scheduler_recipe

SCHEMA='PHASE98_FULL_STATE_V1'
REQUIRED={'schema','identity','step','model','optimizer','scheduler','scheduler_recipe','rng','cursor','sampler','module_modes','gradients','parameter_requires_grad','optimizer_recipe','metadata','state_digest'}
def seal_state(state):
    state=_snapshot(state);_finite(state)
    state['state_digest']=training_state_digest({k:v for k,v in state.items() if k!='state_digest'})
    return state

def _check_optimizer(state):
    """Validate raw Adam/AdamW state without constructing a model or optimizer."""
    recipe=state['optimizer_recipe'];saved=state['optimizer'];step=state['step']
    if not isinstance(recipe,dict) or set(recipe)!={'class','groups'} or recipe['class'] not in ('torch.optim.adam.Adam','torch.optim.adamw.AdamW'):
        raise ValueError('Unsupported optimizer recipe')
    if not isinstance(saved,dict) or set(saved)!={'state','param_groups'} or not isinstance(saved['state'],dict) or not isinstance(saved['param_groups'],list) or not isinstance(recipe['groups'],list) or not recipe['groups'] or len(recipe['groups'])!=len(saved['param_groups']):
        raise ValueError('Malformed optimizer state/groups')
    names=[];ids=[];amsgrad=[]
    for group,declared in zip(saved['param_groups'],recipe['groups']):
        if not isinstance(group,dict) or not {'params','lr'}<=group.keys() or not isinstance(group['params'],list) or not isinstance(declared,dict) or set(declared)!={'parameter_names','options'} or not isinstance(declared['parameter_names'],list):
            raise ValueError('Malformed optimizer parameter mapping')
        options={k:v for k,v in group.items() if k not in ('params','lr','initial_lr')}
        options['base_lr']=group.get('initial_lr',group['lr']) if state['scheduler'] is not None else group['lr']
        if options!=declared['options'] or len(group['params'])!=len(declared['parameter_names']):raise ValueError('Optimizer recipe/group differs')
        if any(type(group.get(k)) not in (int,float) or group[k]<0 for k in ('lr','eps','weight_decay')) or type(group.get('amsgrad')) is not bool:
            raise ValueError('Invalid optimizer numerical options')
        betas=group.get('betas')
        if not isinstance(betas,(list,tuple)) or len(betas)!=2 or any(type(v) not in (int,float) or not 0<=v<1 for v in betas):raise ValueError('Invalid Adam betas')
        names.extend(declared['parameter_names']);ids.extend(group['params']);amsgrad.extend([group['amsgrad']]*len(group['params']))
    if not names or len(set(names))!=len(names) or any(type(n) is not str or n not in state['parameter_requires_grad'] for n in names) or any(type(i) is not int for i in ids) or ids!=list(range(len(names))):
        raise ValueError('Optimizer parameter identity/order differs')
    moments=saved['state'];maximum=0
    if (step==0)!=(len(moments)==0):raise ValueError('Optimizer cursor/moments differ')
    for index,values in moments.items():
        if type(index) is not int or not 0<=index<len(names) or not isinstance(values,dict):raise ValueError('Invalid Adam state index')
        required={'step','exp_avg','exp_avg_sq'}|({'max_exp_avg_sq'} if amsgrad[index] else set())
        if set(values)!=required:raise ValueError('Invalid Adam moment fields')
        counter=values['step']
        if isinstance(counter,torch.Tensor):
            if counter.ndim!=0 or counter.dtype==torch.bool:raise ValueError('Invalid Adam step tensor')
            counter=counter.item()
        if type(counter) not in (int,float) or not math.isfinite(counter) or int(counter)!=counter or not 1<=counter<=step:raise ValueError('Invalid Adam step')
        maximum=max(maximum,int(counter));parameter=state['model'][names[index]]
        for key in required-{'step'}:
            value=values[key]
            if not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=parameter.dtype:raise ValueError('Adam moment tensor schema differs')
            if key in ('exp_avg_sq','max_exp_avg_sq') and bool((value<0).any()):raise ValueError('Negative Adam squared moment')
    if maximum!=step:raise ValueError('Optimizer cursor overreports actual updates')
    if state['scheduler'] is not None and '_last_lr' in state['scheduler']:
        if state['scheduler']['_last_lr']!=[g['lr'] for g in saved['param_groups']]:raise ValueError('Scheduler/optimizer learning rates differ')

def _check_state(state,identity):
    if not isinstance(state,dict) or set(state)!=REQUIRED or state['schema']!=SCHEMA or state['identity']!=identity:raise ValueError('Full-state schema/identity differs')
    if type(state['step']) is not int or state['step']<0:raise ValueError('Invalid cursor')
    _finite(state)
    if training_state_digest({k:v for k,v in state.items() if k!='state_digest'})!=state['state_digest']:raise ValueError('State digest differs')
    for key in ('identity','cursor','sampler','metadata','model','module_modes','gradients','parameter_requires_grad'):
        if not isinstance(state[key],dict):raise ValueError('Expected state mapping: '+key)
    if type(state['cursor'].get('actual_optimizer_updates')) is not int or state['cursor']['actual_optimizer_updates']!=state['step']:raise ValueError('Cursor and committed step differ')
    if not state['sampler']:raise ValueError('Missing sampler state/identity')
    for key in ('identity','cursor','sampler','metadata'):canonical(state[key])
    if any(not isinstance(v,torch.Tensor) or v.layout!=torch.strided for v in state['model'].values()):raise ValueError('Model must contain dense tensors')
    modes=state['module_modes'];parameters=state['parameter_requires_grad'];gradients=state['gradients']
    if '' not in modes or any(type(n) is not str or type(v) is not bool for n,v in modes.items()):raise ValueError('Invalid module modes')
    if not parameters or gradients.keys()!=parameters.keys() or any(type(n) is not str or type(v) is not bool or n not in state['model'] for n,v in parameters.items()):raise ValueError('Invalid parameter/gradient mapping')
    for name,value in gradients.items():
        parameter=state['model'][name]
        if value is not None and (not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=parameter.dtype):raise ValueError('Gradient tensor schema differs')
    if (state['scheduler'] is None)!=(state['scheduler_recipe'] is None):raise ValueError('Scheduler presence differs')
    if state['scheduler'] is not None:
        recipe=state['scheduler_recipe']
        dynamic={'last_epoch','_step_count','_last_lr','_get_lr_called_within_step','_is_initial','best','num_bad_epochs','cooldown_counter'}
        if not isinstance(state['scheduler'],dict) or not isinstance(recipe,dict) or set(recipe)!={'class','config'} or not isinstance(recipe['class'],str) or recipe['config']!={k:v for k,v in state['scheduler'].items() if k not in dynamic}:raise ValueError('Malformed scheduler recipe/state')
    _validate_rng(state['rng']);_check_optimizer(state)

def check_state(state,identity):
    try:_check_state(state,identity)
    except (TypeError,KeyError,AttributeError,IndexError,OverflowError,RuntimeError) as exc:
        raise ValueError('Malformed complete training state: '+str(exc)) from exc

def _check_scheduler_owner(optimizer,scheduler):
    if scheduler is not None and (not hasattr(scheduler,'optimizer') or scheduler.optimizer is not optimizer):raise ValueError('Scheduler optimizer ownership differs')
def capture(model,optimizer,identity,step,*,cursor,sampler,metadata,scheduler=None):
    _check_scheduler_owner(optimizer,scheduler)
    state=dict(schema=SCHEMA,identity=identity,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict() if scheduler else None,scheduler_recipe=_scheduler_recipe(scheduler),rng=_rng(),cursor=cursor,sampler=sampler,metadata=metadata,module_modes={n:m.training for n,m in model.named_modules()},gradients={n:p.grad for n,p in model.named_parameters()},parameter_requires_grad={n:p.requires_grad for n,p in model.named_parameters()},optimizer_recipe=_optimizer_recipe(model,optimizer,scheduler))
    state=seal_state(state);check_state(state,identity);return state
def restore(state,model,optimizer,identity,*,metadata,scheduler=None):
    _check_scheduler_owner(optimizer,scheduler)
    check_state(state,identity)
    if state['metadata']!=metadata:raise ValueError('Metadata/implementation changed')
    current=model.state_dict()
    if current.keys()!=state['model'].keys() or any(current[k].shape!=v.shape or current[k].dtype!=v.dtype for k,v in state['model'].items()):raise ValueError('Tensor layout differs')
    if state['optimizer_recipe']!=_optimizer_recipe(model,optimizer,scheduler):raise ValueError('Optimizer/scheduler recipe differs')
    if (scheduler is None)!=(state['scheduler'] is None):raise ValueError('Scheduler presence differs')
    if state['scheduler_recipe']!=_scheduler_recipe(scheduler):raise ValueError('Scheduler class/configuration differs')
    if scheduler:copy.deepcopy(scheduler).load_state_dict(copy.deepcopy(state['scheduler']))
    modules=dict(model.named_modules());parameters=dict(model.named_parameters())
    if state['module_modes'].keys()!=modules.keys() or state['gradients'].keys()!=parameters.keys():raise ValueError('Mode/gradient keys differ')
    if state['parameter_requires_grad']!={n:p.requires_grad for n,p in parameters.items()}:raise ValueError('Frozen parameter contract differs')
    for n,g in state['gradients'].items():
        if g is not None and (g.shape!=parameters[n].shape or g.dtype!=parameters[n].dtype):raise ValueError('Gradient schema differs')
    numpy_state=_validate_rng(state['rng'])
    # Validate the complete optimizer before any receiver mutation.
    _validate_optimizer(dict(optimizer=state['optimizer'],optimizer_recipe=state['optimizer_recipe'],cursor={'actual_optimizer_updates':state['step']},scheduler=state['scheduler']),model,optimizer,scheduler)
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if scheduler:scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
    for n,m in modules.items():m.training=state['module_modes'][n]
    for n,p in parameters.items():p.grad=None if state['gradients'][n] is None else state['gradients'][n].to(p.device).clone()
    random.setstate(state['rng']['python']);np.random.set_state(numpy_state);torch.set_rng_state(state['rng']['cpu'].cpu())
    if state['rng']['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['rng']['cuda']])
    return copy.deepcopy(dict(step=state['step'],cursor=state['cursor'],sampler=state['sampler']))

class Store:
    def __init__(self,root,identity,*,execution_capability=None):
        self.root=Path(root).resolve();self.identity=copy.deepcopy(identity)
        allowed_roots=[(OUT/'TEST_ONLY').resolve(),(ROOT/'outputs/phase9/phase98/part2_1/TEST_ONLY').resolve()] if identity.get('mode')=='TEST_ONLY' else [(ROOT/'outputs/phase9/phase98/runs').resolve()]
        if identity.get('mode') not in ('TEST_ONLY','PHASE98_FORMAL') or not any(self.root!=a and self.root.is_relative_to(a) for a in allowed_roots):raise PermissionError('Only isolated new Phase98 run roots')
        if identity.get('mode')=='PHASE98_FORMAL':
            from .authorization import ExecutionCapability
            if not isinstance(execution_capability,ExecutionCapability):raise PermissionError('Formal storage creation needs future Phase98 launch authorization')
            execution_capability.permits(identity,self.root)
        for k in ('run_id','method','cell','source_digest','data_digest'):
            if not identity.get(k):raise ValueError('Missing identity '+k)
        self.root.mkdir(parents=True,exist_ok=True)
    def _inside(self,path):
        p=Path(path).resolve()
        if p.parent!=self.root:raise PermissionError('Store path escape')
        return p
    def _preserve_failure(self,path,reason):
        p=self._inside(path)
        record=dict(schema='PHASE98_FAILED_ATTEMPT_V1',identity=self.identity,path=str(p),sha256=sha(p),bytes=p.stat().st_size,reason=reason)
        receipt=self.root/('failed_attempt_'+digest(record)+'.json')
        if not receipt.exists():write_new(receipt,record)
        elif read(receipt)!=record:raise ValueError('Failure evidence receipt differs')
    def _preserve_incomplete(self):
        known=set()
        for marker in self.root.glob('step_*.COMMIT.json'):
            try:known.add(Path(self.header(marker)['payload_path']))
            except (ValueError,OSError,KeyError,TypeError):self._preserve_failure(marker,'invalid COMMIT; retained unchanged')
        for p in list(self.root.glob('*.partial'))+[p for p in self.root.glob('*.state.pt') if p not in known]:
            self._preserve_failure(p,'uncommitted attempt; retained unchanged')
    def commit(self,step,state,*,tags=(),fault=None):
        check_state(state,self.identity)
        from .runtime_budget import disk_guard
        def tensor_bytes(v):
            if isinstance(v,torch.Tensor):return v.numel()*v.element_size()
            if isinstance(v,dict):return sum(tensor_bytes(x) for x in v.values())
            if isinstance(v,(tuple,list)):return sum(tensor_bytes(x) for x in v)
            return 0
        disk_guard(self.root,required_bytes=2*tensor_bytes(state)+10_000_000)
        if state['step']!=step:raise ValueError('Claimed cursor differs')
        if fault and self.identity['mode']!='TEST_ONLY':raise PermissionError('Fault injection only TEST_ONLY')
        if fault not in (None,'after_partial','after_payload'):raise ValueError('Unknown test interruption point')
        prefix=f'step_{step:08d}'
        for prior in self.root.glob(prefix+'*.COMMIT.json'):
            try:self.load(prior)
            except (ValueError,OSError,KeyError,RuntimeError,pickle.UnpicklingError):
                self._preserve_failure(prior,'invalid committed attempt; retained unchanged')
                continue
            raise FileExistsError('Immutable valid publication collision')
        self._preserve_incomplete()
        name=prefix+'.'+uuid.uuid4().hex;payload=self.root/(name+'.state.pt');marker=self.root/(name+'.COMMIT.json')
        partial=self.root/(name+'.state.partial')
        with partial.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        if fault=='after_partial':raise RuntimeError('TEST_ONLY interrupted before publication')
        os.rename(partial,payload)
        if fault=='after_payload':raise RuntimeError('TEST_ONLY interrupted before COMMIT')
        record=dict(schema='PHASE98_COMMIT_V1',identity=self.identity,step=step,payload_path=str(payload),payload_sha256=sha(payload),payload_bytes=payload.stat().st_size,state_digest=state['state_digest'],tags=sorted(set(tags)),commit_path=str(marker))
        record['commit_digest']=digest(record)
        # The marker itself is also atomic; a torn marker is never discoverable
        # as a COMMIT. Unique attempt names preserve every failed publication.
        marker_partial=self.root/(name+'.commit.partial');write_new(marker_partial,record);os.rename(marker_partial,marker)
        self.load(marker) # durable readable anchor before retention is considered
        return record
    def header(self,path):
        p=self._inside(path);r=read(p)
        if r.get('schema')!='PHASE98_COMMIT_V1' or r.get('identity')!=self.identity or r.get('commit_path')!=str(p):raise ValueError('Wrong commit identity/kind')
        if digest({k:v for k,v in r.items() if k!='commit_digest'})!=r['commit_digest']:raise ValueError('COMMIT corrupted')
        self._inside(r['payload_path']);return r
    def load(self,path):
        r=self.header(path);p=self._inside(r['payload_path'])
        if not p.exists():raise ValueError('Payload unavailable/reclaimed; not an arbitrary-step recovery anchor')
        if p.stat().st_size!=r['payload_bytes'] or sha(p)!=r['payload_sha256']:raise ValueError('Payload corrupted')
        s=torch.load(p,map_location='cpu',weights_only=True);check_state(s,self.identity)
        if s['step']!=r['step'] or s['state_digest']!=r['state_digest']:raise ValueError('Commit cursor/state differs')
        return s
    def latest_valid(self):
        failures=[]
        for p in sorted(self.root.glob('step_*.COMMIT.json'),reverse=True):
            try:
                s=self.load(p)
                if failures:append(self.root/'RECOVERY.jsonl',dict(event='FALLBACK_VALIDATED',anchor=str(p),rejected=failures))
                return s
            except (ValueError,OSError,KeyError,RuntimeError,pickle.UnpicklingError) as e:failures.append(dict(commit=str(p),error=str(e)))
        raise ValueError('No valid complete recovery anchor: '+repr(failures))
    def save_selected(self,weights,*,source_commit,selection,source_path=('model',)):
        r=self.header(source_commit);source=self.load(source_commit)
        for key in source_path:source=source[key]
        if training_state_digest(weights)!=training_state_digest(source):raise ValueError('Selected weights not from claimed source field')
        state={'schema':'PHASE98_SELECTED_WEIGHTS_V1','weights':_snapshot(weights),'identity':self.identity,'source_commit':r['commit_path'],'selection':selection,'can_resume_training':False}
        tag=digest(dict(source=r['commit_digest'],selection=selection))[:16];p=self.root/f'selected_{tag}.weights.pt'
        if not p.exists():
            from .resource_guard import checkpoint_required_bytes
            from .runtime_budget import disk_guard
            disk_guard(p,checkpoint_required_bytes(state))
            partial=self.root/f'selected_{tag}.{uuid.uuid4().hex}.weights.partial'
            with partial.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
            os.rename(partial,p)
        receipt=self.root/f'selected_{tag}.json'
        saved=torch.load(p,map_location='cpu',weights_only=True)
        if not isinstance(saved,dict) or set(saved)!=set(state) or any(saved.get(k)!=state[k] for k in state if k!='weights'):raise ValueError('Selected saved identity differs')
        if training_state_digest(saved['weights'])!=training_state_digest(weights):raise ValueError('Selected persistence mismatch')
        record=dict(schema='PHASE98_SELECTED_WEIGHTS_RECEIPT_V1',path=str(p),sha256=sha(p),bytes=p.stat().st_size,source_commit=str(source_commit),source_path=list(source_path),source_step=r['step'],selection=selection,identity=self.identity,weights_digest=training_state_digest(weights),can_resume_training=False)
        if receipt.exists():
            if read(receipt)!=record:raise ValueError('Existing selected receipt differs')
        else:
            partial=self.root/f'selected_{tag}.{uuid.uuid4().hex}.receipt.partial';write_new(partial,record);os.rename(partial,receipt)
        return receipt
    def verify_selected(self,path,*,selection=None,source_step=None):
        record=read(self._inside(path))
        if record.get('schema')!='PHASE98_SELECTED_WEIGHTS_RECEIPT_V1' or record.get('identity')!=self.identity or record.get('can_resume_training') is not False:raise ValueError('Selected receipt identity differs')
        payload=self._inside(record['path'])
        if payload.stat().st_size!=record['bytes'] or sha(payload)!=record['sha256']:raise ValueError('Selected payload bytes differ')
        saved=torch.load(payload,map_location='cpu',weights_only=True)
        if saved.get('identity')!=self.identity or saved.get('selection')!=record['selection'] or saved.get('source_commit')!=record['source_commit']:raise ValueError('Selected state/receipt differs')
        if training_state_digest(saved['weights'])!=record['weights_digest']:raise ValueError('Selected tensors differ')
        commit=self.header(record['source_commit'])
        if commit['step']!=record['source_step']:raise ValueError('Selected source cursor differs')
        if selection is not None and selection!=record['selection']:raise ValueError('Selected rule differs')
        if source_step is not None and source_step!=record['source_step']:raise ValueError('Selected endpoint differs')
        return record
    def collect(self):
        records=[self.header(p) for p in self.root.glob('step_*.COMMIT.json')];records.sort(key=lambda r:r['step'])
        if len(records)<3:return []
        # Orphans are retained and hashed, never deletion candidates. Validate
        # every extant committed payload before deleting any older valid one.
        self._preserve_incomplete()
        for r in records:
            if Path(r['payload_path']).exists():self.load(r['commit_path'])
        anchors=records[-2:]
        for r in anchors:self.load(r['commit_path'])
        selections=sorted([read(p) for p in self.root.glob('selected_*.json')],key=lambda r:r['source_step'])
        # Current selected weights have their own durable verified payload. The
        # complete source commit remains in the audit chain, not a resume promise.
        for item in selections[-1:]:
            sp=self._inside(item['path'])
            if sha(sp)!=item['sha256']:raise ValueError('Current selected payload corrupted')
            saved=torch.load(sp,map_location='cpu',weights_only=True)
            if training_state_digest(saved['weights'])!=item['weights_digest']:raise ValueError('Selected weight identity differs')
        protected={r['commit_path'] for r in anchors}|{r['commit_path'] for r in records if r['tags']}
        reclaimed=[]
        for r in records:
            p=self._inside(r['payload_path'])
            if r['commit_path'] in protected or not p.exists():continue
            self.load(r['commit_path'])
            event=dict(event='RECLAIM_INTENT',payload=str(p),sha256=r['payload_sha256'],bytes=r['payload_bytes'],reason='newer two complete anchors validated; not tagged/selected',replacement_anchors=[x['commit_path'] for x in anchors],arbitrary_historic_resume=False)
            append(self.root/'RECLAIM.jsonl',event)
            # Exact checked file only; no recursive or historical deletion.
            p.unlink();append(self.root/'RECLAIM.jsonl',dict(event='RECLAIM_COMPLETE',payload=str(p),sha256=r['payload_sha256']))
            reclaimed.append(event)
        for old in selections[:-1]:
            p=self._inside(old['path'])
            if not p.exists():continue
            if sha(p)!=old['sha256']:raise ValueError('Old selected payload corrupted; preserve failure evidence')
            event=dict(event='RECLAIM_INTENT',payload=str(p),sha256=old['sha256'],reason='superseded selected weights; latest selected payload independently verified',replacement_selected=selections[-1]['path'],arbitrary_historic_resume=False)
            append(self.root/'RECLAIM.jsonl',event);p.unlink();append(self.root/'RECLAIM.jsonl',dict(event='RECLAIM_COMPLETE',payload=str(p),sha256=old['sha256']));reclaimed.append(event)
        return reclaimed
