"""Single RankingComparison execution path for bounded engineering and future approved G3."""
import argparse,json,random,time,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .common import ROOT,OUT,read,sha,digest,write_new,append
from .runtime_budget import OUT21,RuntimeBudget,disk_guard
from .checkpoint import Store,capture,restore
from evi_hgt.ranking_protocol.production_state import _snapshot,_rng,training_state_digest

def sources():
    prior=read(OUT/'SOURCE_CLOSURE.json')['local_source_sha256']
    paths=({Path(p) for p in prior}|set((ROOT/'src/evi_hgt_ranking_comparison').glob('*.py'))|
        set((ROOT/'scripts/ranking_comparison').glob('*.py'))|set((ROOT/'scripts/ranking_comparison').glob('*.ps1'))|
        set((ROOT/'tests/ranking_comparison').glob('*.py')))
    return {str(p):sha(p) for p in sorted(paths)}

def seed_streams(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def stage_state(built,opt,*,cursor=None,sampler=None,stage=None):
    from .recovery_diagnostics import snapshot_runtime
    return snapshot_runtime(built.model,opt,cursor=cursor,sampler=sampler,stage_state=stage)

def verify_complete(task,capability):
    root=Path(task['output_root']).resolve();receipt=read(root/'COMPLETE.json')
    identity={k:task[k] for k in ('run_id','method','cell','source_digest','data_digest')};identity['mode']='RANKING_COMPARISON_FORMAL'
    capability.permits_development(identity,root)
    if receipt.get('status')!='COMPLETE' or receipt.get('run_id')!=task['run_id'] or receipt.get('identity')!=identity:raise ValueError('Completion identity/status differs')
    def checked(path,expected):
        p=Path(path).resolve()
        if not p.is_relative_to(root) or sha(p)!=expected:raise ValueError('Completion artifact unavailable or changed')
        return p
    if task['method'] in ('HGTDR_ADAPTED','TXGNN_ADAPTED'):
        checked(receipt['selected_weights_path'],receipt['selected_weights_sha256'])
        endpoint=receipt['endpoint']
        if endpoint['run_identity']!=identity:raise ValueError('Native endpoint identity differs')
        checked(endpoint['checkpoint_path'],endpoint['checkpoint_sha256'])
        for artifact in receipt['evaluations']:checked(artifact['path'],artifact['sha256'])
    elif task['method']=='TRAIN_VISIBLE_POPULARITY':checked(receipt['metrics'],receipt['metrics_sha256'])
    else:
        selected=checked(receipt['selected'],receipt['selected_receipt_sha256']);checked(receipt['metrics'],receipt['metrics_sha256'])
        Store(root,identity,execution_capability=capability).verify_selected(selected,selection={'rule':'FIXED_3G'},source_step=task['optimizer_updates'])
    return receipt

class Engine:
    def __init__(self,cell,method,*,mode,run_id,source_digest,data_digest,budget=None,capability=None,test_seed=None,root=None,preparation='original',shared_inputs=None,task=None):
        self.mode=mode;self.method=method;self.budget=budget;self.preparation=preparation
        self.root=Path(root);self.identity=dict(mode=mode,run_id=run_id,method=method,cell=cell,source_digest=source_digest,data_digest=data_digest)
        if mode=='TEST_ONLY':
            if not isinstance(budget,RuntimeBudget) or budget.started is None:raise PermissionError('TEST_ONLY requires active registered shared ledger')
            if cell!=budget.spec['cell'] or method not in budget.spec['methods']:raise PermissionError('Unregistered method/cell')
        else:
            from .authorization import ExecutionCapability
            if mode!='RANKING_COMPARISON_FORMAL' or not isinstance(capability,ExecutionCapability):raise PermissionError('Missing issued G3 capability')
            capability.permits(self.identity,self.root)
        disk_guard(self.root,2_000_000_000)
        from .data import TrainingInputs,load_training_graph
        from .training import SealedTrainingPlan
        from evi_hgt_feature_baseline.pretrained import PretrainedStore
        from .runtime_console import RuntimeConsole
        self.console=RuntimeConsole(method,cell,mode);t=time.monotonic()
        with self.console.preparing():
            if shared_inputs is None:
                self.data=TrainingInputs().cell(cell)
                seal=next(c for c in read(OUT/'sampling/ALL_PLANS_SEALED.json')['cells'] if c['cell']==cell)
                self.plan=SealedTrainingPlan(self.data,seal)
                self.corpus,self.index,self.assembler=load_training_graph(self.data)
                self.features=PretrainedStore()
                self.shared_inputs=dict(data=self.data,plan=self.plan,corpus=self.corpus,index=self.index,assembler=self.assembler,features=self.features)
            else:
                if set(shared_inputs)!={'data','plan','corpus','index','assembler','features'}:raise ValueError('Raw input reuse fields differ')
                if mode!='TEST_ONLY' or shared_inputs['data'].cell_id!=cell:raise ValueError('Only same-cell engineering raw inputs may be reused')
                for key,value in shared_inputs.items():setattr(self,key,value)
                seal=self.plan.seal;self.shared_inputs=shared_inputs
        self.cold_seconds=time.monotonic()-t;self.reused_raw_inputs=shared_inputs is not None
        if mode!='TEST_ONLY':
            if task is None:raise PermissionError('Bound startup task required before model construction')
            from .startup import verify_task_inputs
            verify_task_inputs(task,self.data,self.corpus,seal=seal,feature_metadata=self.features.metadata)
        self.seed=self.data.training_seed if test_seed is None else test_seed
        if mode!='TEST_ONLY' and self.seed!=self.data.training_seed:raise ValueError('Formal seed mismatch')
        if method=='FULL':
            from .full import build_full
            self.built=build_full(cell,self.features)
            self.seed=self.data.training_seed
        else:
            from .models import build_matched_pair
            self.built=build_matched_pair(self.seed,self.features,self.index.edge_catalog)[method]
        from .models import make_optimizer
        self.built.model.to('cuda');self.built.model.train();self.opt=make_optimizer(self.built)
        seed_streams(self.seed);self.logical=0;self.next_step=1
        self.metadata=dict(cell=cell,method=method,seed=self.seed,source_digest=source_digest,
            corpus_digest=self.corpus['digest'],mask_digest=seal['mask_digest'],
            plan_digest=seal['plan_digest'],feature_digest=self.features.metadata['feature_manifest_sha256'],
            preparation=preparation,
            numerical_policy=__import__('evi_hgt_ranking_comparison.numerical_policy',fromlist=['numerical_policy_metadata']).numerical_policy_metadata(method))
        self.store=Store(self.root,self.identity,execution_capability=capability)
    def state(self):
        return capture(self.built.model,self.opt,self.identity,self.logical,
            cursor=dict(actual_optimizer_updates=self.logical,next_source_step=self.next_step),
            sampler=dict(plan_digest=self.plan.seal['plan_digest'],next_source_step=self.next_step),metadata=self.metadata)
    def restore(self,state):
        restored=restore(state,self.built.model,self.opt,self.identity,metadata=self.metadata)
        self.logical=restored['step'];self.next_step=restored['cursor']['next_source_step']
        if restored['sampler']['next_source_step']!=self.next_step:raise ValueError('Next sample cursor mismatch')
    def trace_state(self):
        return stage_state(self.built,self.opt,cursor={'logical':self.logical,'next_source_step':self.next_step},
            sampler={'plan_digest':self.plan.seal['plan_digest']},stage={'preparation':self.preparation,'method':self.method})
    def update(self,step=None,*,replay=False,trace_path=None,preparation=None):
        from .numerical_policy import method_numerical_policy
        with method_numerical_policy(self.method):
            return self._update(step,replay=replay,trace_path=trace_path,preparation=preparation)
    def _update(self,step=None,*,replay=False,trace_path=None,preparation=None):
        from .training import prepare_step,reduce_step
        spec=self.plan.step(self.next_step if step is None else step)
        if self.mode!='TEST_ONLY' and spec.step!=self.next_step:raise ValueError('Formal sequence cannot jump')
        mode=preparation or self.preparation
        traces={};t=time.monotonic()
        if trace_path:traces['before_prepare']=self.trace_state()
        if mode=='original':prepared=prepare_step(self.assembler,spec,device='cuda')
        else:
            from .input_preparation import prepare_optimized
            prepared=prepare_optimized(self.method,self.assembler,spec,device='cuda')
        prep=time.monotonic()-t;graphs=prepared.graphs+prepared.extras
        if trace_path:
            traces['input']=_snapshot(dict(spec=spec.metadata(self.method),batches=[asdict(g.batch.model_batch) for g in graphs]))
            if self.method!='FULL':
                from .input_preparation import consumed_inputs
                traces['consumed_input']=_snapshot(consumed_inputs(prepared,self.method))
            traces['after_prepare']=self.trace_state()
        self.opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic();pred=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for i,g in enumerate(graphs):
                fn=lambda g=g:self.built.forward(g.batch.model_batch,path_batch=g.batch.path_batch)
                v=self.budget.forward(fn,1,label=f'{self.method}/{spec.step}/{i}') if self.budget else fn()
                if not (torch.isfinite(v.support_logit).all() and torch.isfinite(v.unsuitability_logit).all()):raise FloatingPointError('Nonfinite output')
                pred.append(v)
            risk=reduce_step(prepared,pred[:len(prepared.graphs)],pred[len(prepared.graphs):])
        torch.cuda.synchronize();fwd=time.monotonic()-t
        if trace_path:traces['forward']=_snapshot(dict(logits=[(p.support_logit,p.unsuitability_logit) for p in pred],loss=risk.total,rng=_rng()))
        t=time.monotonic();risk.total.backward();torch.cuda.synchronize();back=time.monotonic()-t
        if trace_path:traces['backward']=self.trace_state()
        norm=torch.nn.utils.clip_grad_norm_(self.built.model.parameters(),1.,error_if_nonfinite=True)
        if trace_path:traces['after_clip']=self.trace_state()
        t=time.monotonic()
        if self.budget:self.budget.step(self.opt,label=f'{self.method}/{spec.step}',replay=replay)
        else:self.opt.step()
        torch.cuda.synchronize();optsec=time.monotonic()-t
        self.logical+=1;self.next_step=spec.step+1
        if trace_path:
            traces['after_optimizer']=self.trace_state()
            Path(trace_path).parent.mkdir(parents=True,exist_ok=True)
            with Path(trace_path).open('xb') as f:torch.save(traces,f)
        row=dict(method=self.method,source_step=spec.step,logical_update=self.logical,stage=spec.validate()['stage'],
            preparation=mode,prepare_seconds=prep,forward_seconds=fwd,backward_seconds=back,optimizer_seconds=optsec,
            loss=float(risk.total.detach()),gradient_norm=float(norm),query_instances=len(graphs),replay=replay,
            ordered_query_pair_digest=spec.identity['ordered_query_pair_digest'],
            allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved())
        append(self.root/'events.jsonl',row);self.console.update(row,self.plan.G)
        return row
    def close(self):
        self.console.close()
        for handle in getattr(self.built,'handles',[]):handle.remove()
    def commit(self,tags=()):
        t=time.monotonic();c=self.store.commit(self.logical,self.state(),tags=tags)
        append(self.root/'events.jsonl',dict(event='COMMIT',step=self.logical,seconds=time.monotonic()-t,bytes=c['payload_bytes']))
        return c

def formal(manifest_path,grant_path,approved_digest):
    from .authorization import require_formal_execution
    manifest=read(manifest_path);grant=read(grant_path) if grant_path else None
    capability=require_formal_execution(manifest,grant,'G3',approved_grant_digest=approved_digest)
    # Identity verification is before any data/model construction.
    from .startup import verify_launch_manifest
    verify_launch_manifest(manifest)
    from .resource_guard import configure,check_queue_capacity
    configure(manifest['resource_limits'])
    check_queue_capacity()
    from .safety import snapshot
    snapshot()
    lock=ROOT/'outputs/ranking_protocol/ranking_comparison/runtime_worker.lock';token=__import__('uuid').uuid4().hex
    write_new(lock,dict(pid=__import__('os').getpid(),token=token))
    try:
        for task_index,task in enumerate(manifest['training_tasks']):
            dest=Path(task['output_root'])
            if (dest/'COMPLETE.json').exists():
                verify_complete(task,capability)
                continue
            disk_guard(dest,manifest['resource_limits']['next_task_required_bytes'])
            if task['method'] not in ('FULL','PAIR_MLP_D1_MATCHED','HGT_D1_MATCHED'):
                from .native_worker import launch_formal
                launch_formal(manifest_path,grant_path,approved_digest,task_index)
                verify_complete(task,capability)
                continue
            engine=Engine(task['cell'],task['method'],mode='RANKING_COMPARISON_FORMAL',run_id=task['run_id'],
                source_digest=task['source_digest'],data_digest=task['data_digest'],root=dest,capability=capability,
                preparation=task.get('preparation','original'),task=task)
            if list(dest.glob('*.COMMIT.json')):engine.restore(engine.store.latest_valid())
            while engine.next_step<=3*engine.plan.G:
                engine.update()
                step=engine.logical;G=engine.plan.G;tag=['milestone'] if step in (G,6*G//4,2*G,3*G) else []
                if step%256==0 or tag:
                    commit=engine.commit(tags=tag)
                    if task.get('retention')=='QUALIFIED_ROLLING':engine.store.collect()
            endpoint=engine.store.latest_valid()
            if endpoint['step']!=3*engine.plan.G:raise ValueError('Endpoint recovery cursor differs')
            endpoint_commits=[engine.store.header(p) for p in dest.glob(f'step_{endpoint["step"]:08d}*.COMMIT.json')]
            commit=next(r for r in endpoint_commits if r['state_digest']==endpoint['state_digest'] and Path(r['payload_path']).exists())
            existing=list(dest.glob('selected_*.json'))
            selected=existing[-1] if existing else engine.store.save_selected(engine.built.model.state_dict(),source_commit=commit['commit_path'],selection={'rule':'FIXED_3G'})
            engine.store.verify_selected(selected,selection={'rule':'FIXED_3G'},source_step=3*engine.plan.G)
            from .runtime_evaluation import evaluate_development
            metrics=evaluate_development(engine,capability,task)
            write_new(dest/'COMPLETE.json',dict(run_id=task['run_id'],identity=engine.identity,selected=str(selected),selected_receipt_sha256=sha(selected),metrics=metrics,metrics_sha256=sha(metrics),status='COMPLETE'))
            verify_complete(task,capability)
            engine.close();del engine;__import__('gc').collect();torch.cuda.empty_cache()
        for task in manifest.get('development_tasks',[]):
            if task['method']!='TRAIN_VISIBLE_POPULARITY':raise ValueError('Unknown non-training development task')
            dest=Path(task['output_root']);identity={k:task[k] for k in ('run_id','method','cell','source_digest','data_digest')};identity['mode']='RANKING_COMPARISON_FORMAL'
            capability.permits_development(identity,dest)
            if (dest/'COMPLETE.json').exists():verify_complete(task,capability);continue
            disk_guard(dest,manifest['resource_limits']['next_task_required_bytes']);dest.mkdir(parents=True,exist_ok=True)
            from .data import TrainingInputs
            from .corpus import verify_manifest
            from .startup import verify_task_inputs
            from .runtime_evaluation import score_popularity
            data=TrainingInputs().cell(task['cell']);corpus=verify_manifest(data);verify_task_inputs(task,data,corpus)
            metrics=score_popularity(data,capability,identity,dest,task['counts_ref']);path=dest/'metrics.json'
            if path.exists():
                if read(path)!=metrics:raise ValueError('Popularity metric cache differs')
            else:write_new(path,metrics)
            from .console import Console
            console=Console(context='AUTHORIZED_RUNTIME');console.evaluation_result(task['method'],0,metrics);console.close()
            write_new(dest/'COMPLETE.json',dict(run_id=task['run_id'],identity=identity,status='COMPLETE',metrics=str(path),metrics_sha256=sha(path),optimizer_updates=0,model_forwards=0))
            verify_complete(task,capability)
    except BaseException as exc:
        failure=ROOT/'outputs/ranking_protocol/ranking_comparison/runtime_failures'/(__import__('uuid').uuid4().hex+'.json')
        failure.parent.mkdir(parents=True,exist_ok=True)
        write_new(failure,dict(status='FAILED_STOPPED',error=str(exc),traceback=traceback.format_exc(),task=task if 'task' in locals() else None))
        print('正式运行安全停止；完整日志：',failure,flush=True)
        raise
    finally:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('formal');f.add_argument('--manifest',required=True);f.add_argument('--grant');f.add_argument('--approved-grant-digest')
    q=sub.add_parser('test-only');q.add_argument('--registration',required=True)
    a=parser.parse_args()
    if a.command=='formal':return formal(a.manifest,a.grant,a.approved_grant_digest)
    registration=read(a.registration)
    if registration.get('family') in ('native','native22','native23','native24'):
        from .native_worker import launch_test_only
        return launch_test_only(a.registration)
    from .runtime_qualification import execute
    return execute(registration)

if __name__=='__main__':main()
