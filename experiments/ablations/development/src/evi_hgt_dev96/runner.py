"""A fixed current-cell arm; fresh initialization and exact paired schedule."""
import json
import os
import time
import traceback
import psutil
import torch
from evi_hgt.phase9.runner import numeric_backend
from evi_hgt.phase9.state import seed_streams,tensor_digest
from evi_hgt.phase9.production_state import _rng,training_state_digest
from evi_hgt_dev94.pretrained import PretrainedStore
from evi_hgt_dev94_recovery.runner import Events,prepare
from .protocol import *
from .data import load_data
from .model import build_arm,make_optimizer
from .sampling import freeze_schedule,conditional_plan
from .training import execute_update
from .checkpoint import Store,capture,restore,verify_commit
from .diagnostics import diagnose
from .evaluation import evaluate,verify_result
from .safety import no_conflicts,old_hold


def logged_updates(cell,arm):
    p=OUT/cell/arm/'updates.jsonl'
    return [json.loads(r) for r in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []


def verify_receipt(r,ident):
    cell=ident['cell'];s=cell_spec(cell);arm=ident['arm'];root=OUT/cell/arm
    if (r.get('status')!='COMPLETE_VERIFIED_DEVELOPMENT' or r.get('identity')!=ident
        or r.get('logical_optimizer_updates')!=s['total'] or set(r.get('evaluations',{}))!=set(map(str,s['milestones']))):
        raise ValueError('Incomplete DEV96 cell trajectory')
    for n,e in r['evaluations'].items():verify_result(cell,e,ident,int(n))
    commits=[verify_commit(p,ident) for p in sorted((root/'checkpoints').glob('*.COMMIT.json'))]
    if [c['step'] for c in commits]!=[n for n in range(1,s['total']+1) if checkpoint_due(cell,n)]:raise ValueError('Checkpoint boundaries differ')
    rows=logged_updates(cell,arm);latest={x['optimizer_step']:x for x in rows}
    if sorted(latest)!=list(range(1,s['total']+1)) or len(rows)!=r['physical_optimizer_updates']:raise ValueError('Update count differs')
    if any(x['cell']!=cell or x['loss_id']!=ident['loss_id'] or x['manifest_sha256']!=ident['manifest_sha256'] or not x['finite'] for x in rows):
        raise ValueError('Update identity/finiteness differs')
    if not r['conditional_plan_verified']:raise ValueError('Conditional plan unverified')
    return r


def run_arm(cell,arm,*,resume=False):
    if arm not in ARMS:raise PermissionError('Only DEV96 E0/E1')
    no_conflicts();old_hold();m=validate_manifest();s=cell_spec(cell);cm=m['cells'][cell];ident=identity(m,cell,arm)
    out=OUT/cell/arm;out.mkdir(parents=True,exist_ok=True);ui=Events();step=0;last_commit=None
    if (out/'receipt.json').exists():
        if not resume:raise FileExistsError('Trajectory already complete')
        return verify_receipt(read(out/'receipt.json'),ident)
    if (out/'IN_FLIGHT_UPDATE.json').exists():raise RuntimeError('Unresolved optimizer boundary; no automatic replay')
    if (out/'started.json').exists() and not resume:raise FileExistsError('Already started; explicit resume required')
    start=time.monotonic();attempt=str(time.time_ns());evaluations={};plan=None;records=[]
    control=dict(clipped_updates=0,zero_gradient_steps=0,training_seconds=0.,query_preparation_seconds=0.,
        forward_seconds=0.,backward_optimizer_seconds=0.,diagnostics={},evaluations={},sampling_preparation_seconds=0.)
    try:
        ui.stage('DEV96 数据准备',cell+'/'+arm+'；当前cell完整3G，独立fresh模型')
        numeric_backend();data,corpus=load_data(cell);groups,stream,schedule,cache=freeze_schedule(data,corpus)
        if dict(data.bindings,corpus_digest=corpus['digest'],schedule_digest=schedule['digest'])!=cm['data_bindings']:
            raise ValueError('Current-cell registered data/graph/schedule differs')
        index,assembler,index_seconds=prepare(data,corpus,ui);pre=PretrainedStore()
        if pre.metadata!=m['pretrained_identity']:raise ValueError('Pretrained input identity changed')
        seed_streams(s['seed']);rng=training_state_digest(_rng());built=build_arm(cell,pre)
        if training_state_digest(_rng())!=rng:raise ValueError('Construction changed RNG')
        initial=tensor_digest(built.model.state_dict())
        if initial!=cm['initial_tensor_sha256'] or built.metadata()!=cm['initial_metadata']:raise ValueError('Fresh cell D1 identity changed')
        built.model.cuda().train();optimizer=make_optimizer(built)
        if optimizer.state:raise ValueError('Fresh AdamW has state')
        store=Store(out/'checkpoints',ident);latest=store.latest()
        if latest:
            if not resume:raise FileExistsError('Existing checkpoint requires explicit same-arm resume')
            state=store.load(latest);r=restore(state,built.model,optimizer,ident,built.metadata())
            step=r['step'];control=r['control'];last_commit=verify_commit(latest,ident);del state
        elif resume and (out/'started.json').exists():raise RuntimeError('No complete checkpoint; no automatic restart')
        def bind_conditional():
            nonlocal plan,records
            t=time.monotonic();anchor=OUT/cell/ARMS[0]/'checkpoints'/f"step_{s['G']:08d}.COMMIT.json"
            plan=conditional_plan(data,groups,schedule,cell=cell,origin_commit=anchor)
            records=plan['legacy']['records'];control['sampling_preparation_seconds']+=time.monotonic()-t
            write(out/'paired_plan_binding.json',dict(cell=cell,identity=ident,plan_digest=plan['digest'],
                origin_sha256=plan['origin_sha256'],records=len(records),bound_before_first_evaluation=True))
        if arm==ARMS[1] or step>=s['G']:bind_conditional()
        startup=dict(identity=ident,cell=cell,arm=arm,seed=s['seed'],G=s['G'],total=s['total'],fresh=True,
            model='D1_PRETRAINED_EARLY',initial_tensor_sha256=initial,initial_metadata=built.metadata(),initial_rng_sha256=rng,
            optimizer_initial_state_entries=0,trained_weights_loaded=False,test_weights_loaded=False,
            schedule_digest=schedule['digest'],corpus_digest=corpus['digest'],index_preparation_seconds=index_seconds,
            pre_training_preparation_seconds=time.monotonic()-start,pid=os.getpid(),attempt=attempt,
            queue=[list(q) for q in QUEUE],mode='DEVELOPMENT')
        if not (out/'started.json').exists():write(out/'started.json',startup)
        else:write(out/('resume_'+attempt+'.json'),dict(step=step,identity=ident,checkpoint=last_commit))
        previous={r['optimizer_step']:r for r in logged_updates(cell,arm)}
        def diagnostics_at(n):
            p=out/'diagnostics'/f'step_{n:08d}.json';weight=tensor_digest(built.model.state_dict())
            if p.exists():
                d=read(p)
                if d['identity']!=ident or d['model_weight_sha256']!=weight:raise ValueError('Diagnostic identity differs')
            else:
                ui.stage('当前cell固定训练诊断',cell+'/'+arm+f' step={n}；hash面板，非开发评价')
                d=diagnose(cell,assembler,built,optimizer,groups,records,n,arm=arm,panels=cm['diagnostic_panels'],ui=ui)
                d.update(identity=ident,model_weight_sha256=weight,actual_training_clipped_updates=control['clipped_updates'],
                         actual_training_finite_zero_steps=control['zero_gradient_steps'])
                write(p,d)
            control['diagnostics'][str(n)]=dict(path=str(p),sha256=file_sha(p),seconds=d['seconds'])
        def milestone(n):
            if plan is None:raise RuntimeError('Conditional plan must bind before first evaluation')
            validate_manifest();diagnostics_at(n)
            write(out/'status.json',dict(status='EVALUATION_PENDING',step=n,identity=ident),immutable=False)
            e=evaluate(cell,data,assembler,built,optimizer,n,last_commit,ident,ui)
            evaluations[str(n)]=e;control['evaluations'][str(n)]=dict(seconds=e['seconds'])
        if step==0:diagnostics_at(0)
        elif step in s['milestones']:milestone(step)
        for n in s['milestones']:
            p=out/'evaluation'/f'step_{n:08d}'/'evaluation.json'
            if p.exists():evaluations[str(n)]=verify_result(cell,read(p),ident,n)
        label=cell+'/'+arm;ui.training_start(label,step,s['total'],s['G'])
        while step<s['total']:
            if (ROOT/'outputs/phase9/phase92b/phase9_device_queue.lock').exists():raise RuntimeError('Old G3 lock appeared')
            member=stream[step];record=records[step-s['G']] if step>=s['G'] else None
            rh=digest(record) if record is not None else None;sampling=plan['digest'] if record is not None else schedule['digest']
            if step+1 in previous and (previous[step+1]['training_member_sha256']!=member['member_sha256'] or previous[step+1]['record_sha256']!=rh):
                raise ValueError('Resume sample identity differs')
            write(out/'IN_FLIGHT_UPDATE.json',dict(step=step+1,attempt=attempt,identity=ident,member=member),immutable=False)
            result=execute_update(assembler,built,optimizer,groups[step],record,arm=arm);step+=1
            for k,r in [('training_seconds','seconds'),('query_preparation_seconds','preparation_seconds'),('forward_seconds','forward_seconds'),('backward_optimizer_seconds','backward_optimizer_seconds')]:control[k]+=result[r]
            control['clipped_updates']+=int(result['clipped']);control['zero_gradient_steps']+=int(result['zero_gradient_step'])
            row=dict(cell=cell,arm=arm,loss_id=LOSSES[arm],mode='DEVELOPMENT',pid=os.getpid(),attempt=attempt,
                optimizer_step=step,planned_optimizer_steps=s['total'],seed=s['seed'],G=s['G'],epoch=(step-1)//s['G']+1,
                stage='GLOBAL' if step<=s['G'] else 'CONDITIONAL',replayed_existing_step=step in previous,
                training_member_sha256=member['member_sha256'],record_sha256=rh,sampling_sha256=sampling,
                manifest_sha256=ident['manifest_sha256'],CUDA_allocated=torch.cuda.memory_allocated(),CUDA_reserved=torch.cuda.memory_reserved(),
                CPU_RSS=psutil.Process().memory_info().rss,**result)
            journal(out/'updates.jsonl',row);(out/'IN_FLIGHT_UPDATE.json').unlink();ui.update(dict(row,arm=label))
            if step==1:
                write(out/'FIRST_UPDATE.json',dict(**row,real_new_optimizer_update=True,test_only=False,
                    initial_tensor_sha256=initial,queued_following_trajectories=[list(q) for q in QUEUE[QUEUE.index((cell,arm))+1:]],log_path=str(out/'updates.jsonl')))
            if checkpoint_due(cell,step):
                validate_manifest();state=capture(built.model,optimizer,ident,step,control,built.metadata())
                last_commit=store.commit(step,state);del state;ui.checkpoint(step)
            if step==s['G'] and plan is None:bind_conditional()
            if step in s['milestones']:
                milestone(step)
                if step<s['total']:ui.training_start(label,step,s['total'],s['G'])
        physical=len(logged_updates(cell,arm));receipt=dict(status='COMPLETE_VERIFIED_DEVELOPMENT',identity=ident,cell=cell,arm=arm,
            logical_optimizer_updates=step,physical_optimizer_updates=physical,replayed_updates=physical-step,evaluations=evaluations,
            control=control,conditional_plan_verified=True,conditional_plan_digest=plan['digest'],
            preparation_seconds=startup['pre_training_preparation_seconds'],worker_wall_seconds=time.monotonic()-start,
            last_checkpoint=last_commit,old_G3='HELD',outer='UNSCORED',new_final='SEALED')
        verify_receipt(receipt,ident);write(out/'receipt.json',receipt)
        write(out/'status.json',dict(status=receipt['status'],step=step),immutable=False)
        ui.stage('本轨迹开发训练及评价完成',label+'；继续固定本地队列')
        return receipt
    except BaseException as exc:
        (out/'traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        write(out/('FAILED_'+attempt+'.json'),dict(status='PARTIAL_FAILED',identity=ident,step=step,
            optimizer_boundary_unknown=(out/'IN_FLIGHT_UPDATE.json').exists(),error=repr(exc),wall_seconds=time.monotonic()-start,control=control))
        ui.failure('DEV96已停止：'+str(exc),str(out/'traceback.log'));raise
