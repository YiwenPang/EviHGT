"""Fixed M4 two-stage driver. Exact independent G3 authority precedes data/model I/O."""
from pathlib import Path
import argparse
import copy
import json
import math
import os
import re
import time

from .identity import ROOT,G1_SHA256,file_sha,digest,frozen_science
from .txgnn_protocol import InnerSelection,SELECTION_EPOCHS,pretrain_commit_due,require_endpoint


class UpdateJournal:
    """Small fsync ledger outside rollback state: known replay work is never lost."""
    def __init__(self,path):
        self.path=Path(path)
        self.rows=[]
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                self.rows.append(json.loads(line))
        self._validate()

    def _validate(self):
        starts={};ends=set()
        for row in self.rows:
            if row.get('event')=='START':
                if (set(row)!={'event','attempt','stage','logical_step'} or type(row['attempt']) is not int
                    or row['attempt']!=len(starts)+1 or row['stage'] not in {'PRETRAIN','FINETUNE'}
                    or type(row['logical_step']) is not int or row['logical_step']<=0):raise ValueError('Invalid update journal START')
                starts[row['attempt']]=row
            elif row.get('event')=='END':
                if (set(row)!={'event','attempt','completed','failed','elapsed_seconds'}
                    or type(row['attempt']) is not int or row['attempt'] not in starts or row['attempt'] in ends
                    or type(row['completed']) is not bool or type(row['failed']) is not bool
                    or (not row['completed'] and not row['failed'])
                    or type(row['elapsed_seconds']) not in (int,float)
                    or not math.isfinite(row['elapsed_seconds']) or row['elapsed_seconds']<0):raise ValueError('Invalid update journal END')
                ends.add(row['attempt'])
            else:raise ValueError('Invalid update journal event')
        return starts,ends

    def _append(self,row):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
        self.rows.append(row)

    def begin(self,stage,logical_step):
        if stage not in {'PRETRAIN','FINETUNE'} or type(logical_step) is not int or logical_step<=0:
            raise ValueError('Invalid update stage/logical step')
        starts,_=self._validate();token=len(starts)+1
        self._append(dict(event='START',attempt=token,stage=stage,logical_step=logical_step))
        return token

    def end(self,token,*,completed,elapsed_seconds,failed=False):
        starts,ends=self._validate()
        if type(token) is not int or token not in starts or token in ends:raise ValueError('Duplicate or unknown update attempt')
        if (type(completed) is not bool or type(failed) is not bool or (not completed and not failed)
            or type(elapsed_seconds) not in (int,float) or not math.isfinite(elapsed_seconds) or elapsed_seconds<0):
            raise ValueError('Finite nonnegative measured attempt duration and result required')
        self._append(dict(event='END',attempt=token,completed=completed,failed=failed,elapsed_seconds=float(elapsed_seconds)))

    def stats(self):
        starts,ends=self._validate();seen=set();completed=0;replayed=0
        successful_seconds=[];failed_seconds=[]
        for row in self.rows:
            if row['event']=='END':
                (failed_seconds if row['failed'] else successful_seconds).append(row['elapsed_seconds'])
                if row['completed']:
                    start=starts[row['attempt']];key=(start['stage'],start['logical_step'])
                    completed+=1;replayed+=int(key in seen);seen.add(key)
        incomplete=len(set(starts)-ends)
        return dict(physical_completed_updates=completed,replayed_completed_updates=replayed,
            failed_measured_attempts=len(failed_seconds), incomplete_attempts=incomplete,
            known_successful_attempt_seconds=math.fsum(successful_seconds),
            known_failed_attempt_seconds=math.fsum(failed_seconds),
            known_physical_training_seconds=math.fsum(successful_seconds+failed_seconds),
            unmeasured_attempt_seconds=None if incomplete else 0.,training_time_complete=not incomplete)

    def require_complete_logical_domain(self,pretrain_batches,finetune_epochs):
        starts,_=self._validate()
        allowed={(stage,step) for stage,total in (('PRETRAIN',pretrain_batches),('FINETUNE',finetune_epochs))
                 for step in range(1,total+1)}
        actual={(starts[row['attempt']]['stage'],starts[row['attempt']]['logical_step'])
                for row in self.rows if row['event']=='END' and row['completed']}
        if actual!=allowed or any((row['stage'],row['logical_step']) not in allowed for row in starts.values()):
            raise ValueError('Update journal does not bind the complete fixed logical domain')


def _training_time_status(stats):
    return 'COMPLETE_MEASURED' if stats['training_time_complete'] else 'LOWER_BOUND_WITH_UNMEASURED_INTERRUPTED_ATTEMPTS'


def run_fixed_control(trainer,*,commit,evaluate_inner,snapshot_weights):
    """Native fixed budget control; synthetic schedule tests perform zero optimizer work.

    Public real entry is run_formal. TEST_ONLY trainers independently reject
    full budgets; callbacks do not create any evaluation authority.
    """
    control=trainer.external_control
    control.setdefault('selection',InnerSelection().state())
    control.setdefault('best_model',None)
    selector=InnerSelection.from_state(control['selection'])
    if trainer.stage=='PRETRAIN':
        while trainer.pretrain_batch<trainer.planned_pretrain_batches:
            trainer.pretrain_step()
            if pretrain_commit_due(trainer.pretrain_batch,trainer.planned_pretrain_batches) and trainer.pretrain_batch<trainer.planned_pretrain_batches:
                commit(f'pretrain_{trainer.pretrain_batch:05d}')
        source=commit('pretrain_complete')
        trainer.transition(source)
        commit('finetune_start')
    if trainer.stage!='FINETUNE':raise ValueError('Invalid two-stage control state')
    def pending_selection():
        epoch=trainer.finetune_epoch
        completed=len(selector.history)
        due=epoch//25
        if due==completed:return
        if epoch not in SELECTION_EPOCHS or due!=completed+1:
            raise ValueError('Missing or extra fixed inner selection opportunity')
        metrics=evaluate_inner(epoch)
        if selector.observe(epoch,metrics):control['best_model']=snapshot_weights()
        control['selection']=selector.state()
        commit(f'finetune_{epoch:04d}_inner_complete')
    pending_selection()
    while trainer.finetune_epoch<500:
        trainer.finetune_step()
        if trainer.finetune_epoch in SELECTION_EPOCHS:
            commit(f'finetune_{trainer.finetune_epoch:04d}_pre_inner')
            pending_selection()
    require_endpoint(pretrain_batches=trainer.pretrain_batch,planned_batches=trainer.planned_pretrain_batches,
        finetune_epoch=trainer.finetune_epoch,selection_epochs=[x['epoch'] for x in selector.history],
        decoder_resets=trainer.decoder_resets)
    return selector.state()


def _permit(permit,data=None):
    from .execution import verify_permit
    r,f=(data.repeat_index,data.cell['fold']) if data is not None else (0,1)
    return verify_permit(permit,'M4',r,f)


def panel_rows(data,name):
    if name not in {'validation','train_probe','auxiliary_validation'}:
        raise PermissionError('M4 exposes only the separately authorized G3 development panels')
    if name=='auxiliary_validation':
        rows=[dict(x,role='P') for x in data.cell[name]]+[dict(x) for x in data.cell['validation'] if x['role']=='U']
        relation='contraindication'
    else:
        rows=[dict(x) for x in data.cell[name]]
        relation='indication'
    rows=sorted(rows,key=lambda r:(r['drug_id'],r['disease_id']))
    if len({(r['drug_id'],r['disease_id']) for r in rows})!=len(rows):raise ValueError('Duplicate legal panel members')
    return rows,relation


def score_authorized_panel(trainer,data,name,permit):
    _permit(permit,data)
    import numpy as np
    import torch
    from evi_hgt.phase85_part2b.runtime import inner_metrics
    data.enable_development(permit,'M4')
    rows,relation=panel_rows(data,name)
    for row in rows:
        source='validation' if name=='auxiliary_validation' and row['role']=='U' else name
        data.assert_score_domain((row['drug_id'],row['disease_id']),source)
    native_data=trainer.data
    pairs=torch.tensor([[native_data.drug_local[r['drug_id']] for r in rows],
                        [native_data.disease_local[r['disease_id']] for r in rows]],dtype=torch.int64)
    graph=trainer.native.dda_graph(native_data.graph,pairs,pairs)
    etype=('drug',relation,'disease')
    from .production_state import _rng
    from .txgnn_recovery import _restore_rng
    modes={name:module.training for name,module in trainer.model.named_modules()}
    rng=_rng()
    try:
        trainer.model.eval()
        with torch.inference_mode():
            logits=trainer.adapter.score_graph(graph,etypes=[etype])[etype].cpu().tolist()
    finally:
        for name,module in trainer.model.named_modules():module.training=modes[name]
        _restore_rng(rng)
    if len(logits)!=len(rows) or not all(np.isfinite(logits)):raise ValueError('Nonfinite/incorrect native panel logits')
    for row,logit in zip(rows,logits):row['support_logit']=float(logit)
    # The unchanged common evaluator needs this NumPy1 spelling compatibility.
    if not hasattr(np,'trapezoid'):np.trapezoid=np.trapz
    metrics=inner_metrics(rows)
    metrics['channel']='unsuitability' if name=='auxiliary_validation' else 'support'
    return dict(rows=rows,metrics=metrics,raw_relation=relation)


def checkpoint_rank(receipt):
    c=receipt['cursor'];name=Path(receipt['checkpoint_commit_path']).name.removesuffix('.COMMIT.json')
    pt,ft=c['pretrain_batch'],c['finetune_epoch']
    if c['actual_optimizer_updates']!=pt+ft:raise ValueError('M4 committed update accounting differs')
    if name=='fresh_start' and c['stage']=='PRETRAIN' and pt==ft==0:rank=0
    elif name=='pretrain_complete' and c['stage']=='PRETRAIN' and pt>0 and ft==0:rank=1
    elif re.fullmatch(r'pretrain_\d{5}',name) and int(name[9:])==pt and c['stage']=='PRETRAIN' and ft==0:rank=0
    elif name=='finetune_start' and c['stage']=='FINETUNE' and ft==0:rank=2
    elif name==f'finetune_{ft:04d}_pre_inner' and c['stage']=='FINETUNE' and ft in SELECTION_EPOCHS:rank=3
    elif name==f'finetune_{ft:04d}_inner_complete' and c['stage']=='FINETUNE' and ft in SELECTION_EPOCHS:rank=4
    elif name=='endpoint_complete' and c['stage']=='FINETUNE' and ft==500:rank=5
    else:raise ValueError('M4 checkpoint name and stage cursor disagree')
    return c['actual_optimizer_updates'],rank


def exact_output(output,permit,repeat,fold):
    output=Path(output).resolve()
    if output!=Path(permit['output_root']).resolve()/f'r{repeat}_f{fold}'/'M4':
        raise PermissionError('M4 output must exactly match approved method/cell directory')
    return output


def _latest(store):
    from .txgnn_recovery import verify_tx_commit
    rows=[verify_tx_commit(p) for p in store.root.glob('*.COMMIT.json')]
    if not rows:raise FileNotFoundError('No complete same-arm M4 commit to resume')
    return max(rows,key=checkpoint_rank)


def run_formal(repeat,fold,output,execution_manifest,*,resume=False):
    from .execution import validate_execution_manifest
    permit=validate_execution_manifest(execution_manifest,'M4',repeat,fold)
    from .m4_amendment import verify_m4_amendment
    amendment=verify_m4_amendment(permit)
    output=exact_output(output,permit,repeat,fold)
    from .txgnn_recovery import TxTwoStageTrainer,TxAtomicStore,verify_tx_commit
    final=output/'receipt.json'
    if final.exists():
        receipt=json.loads(final.read_text(encoding='utf-8'))
        verify_trajectory_receipt(receipt,permit,repeat,fold,output)
        return receipt
    import torch
    from .real_data import RealInputs,build_fold_corpus
    from .txgnn_data import prepare_cell,stream_factory_for_cell,runtime_identity
    from .development_cache import once_panel
    from .production_state import _snapshot
    torch.set_num_threads(8)
    start=time.perf_counter()
    data=RealInputs().cell(repeat,fold)
    corpus=build_fold_corpus(data,permit=permit)
    native=prepare_cell(data,corpus,mode='FORMAL',permit=permit)
    identity=runtime_identity(native)
    identity.update(amendment_sha256=permit['amendment_sha256'],execution_manifest_sha256=permit['_manifest_sha256'])
    identity['runner_sha256']=file_sha(__file__)
    trainer=TxTwoStageTrainer(native,stream_factory_for_cell(native,data),identity,
                             mode='FORMAL',execution_permit=permit)
    store=TxAtomicStore(output/'checkpoints',mode='FORMAL',execution_permit=permit)
    last=None
    if resume:
        last=_latest(store) if resume is True else verify_tx_commit(resume)
        trainer.restore(store.load(last['checkpoint_commit_path']))
    elif any(store.root.iterdir()):raise FileExistsError('M4 fresh execution cannot reuse existing trajectory files')
    prep_seconds=time.perf_counter()-start
    control=trainer.external_control
    control.setdefault('selection',InnerSelection().state());control.setdefault('best_model',None)
    control.setdefault('evaluation_seconds',0.0)
    control.setdefault('evaluation_forwards',0)
    control.setdefault('training_seconds',0.0)
    control.setdefault('endpoint_complete',False)
    journal=UpdateJournal(output/'optimizer_updates.jsonl')
    def commit(name):
        nonlocal last
        last=store.commit(name,trainer.snapshot())
        print(json.dumps(dict(CURRENT_METHOD='M4',REPEAT=repeat,FOLD=fold,
            STAGE=trainer.stage,OPTIMIZER_STEP=trainer.cursor['actual_optimizer_updates'],
            PLANNED_STEPS=trainer.planned_pretrain_batches+500,ELAPSED_TIME=time.perf_counter()-start,
            TRAINING_LOSS=(trainer.last_step or {}).get('loss'),FINITE_STATUS=True,
            LAST_VERIFIED_CHECKPOINT=last['checkpoint_commit_path'])),flush=True)
        return last
    def cached(name,token,epoch):
        if last is None:raise ValueError('Development scoring requires committed model state')
        from .state import tensor_digest
        bindings=dict(method='M4',repeat=repeat,fold=fold,epoch=epoch,domain=name,
            g1_sha256=G1_SHA256,amendment_sha256=permit['amendment_sha256'],
            execution_manifest_sha256=permit['_manifest_sha256'],
            members=data.contract.bindings['membership_digests'],
            weight_digest=tensor_digest(trainer.model.state_dict()),
            training_commit_sha256=last['checkpoint_sha256'],runner_sha256=file_sha(__file__))
        before=dict(seconds=control['evaluation_seconds'],forwards=control['evaluation_forwards'])
        def producer():
            tick=time.perf_counter();result=score_authorized_panel(trainer,data,name,permit)
            result.update(counters_before=before,counters_after=dict(seconds=before['seconds']+time.perf_counter()-tick,forwards=before['forwards']+1))
            return result
        result=once_panel(output/'development_cache'/token,name,bindings,producer)
        current=dict(seconds=control['evaluation_seconds'],forwards=control['evaluation_forwards'])
        if current==result['counters_before']:
            control['evaluation_seconds']=result['counters_after']['seconds']
            control['evaluation_forwards']=result['counters_after']['forwards']
        elif current!=result['counters_after']:raise ValueError('Cached M4 evaluation work/cursor differs')
        return result
    # Timers measure training calls including sampling. The fsynced update journal
    # retains measured failures/replays outside rollback state; hard-kill time is unknown.
    for method in ('pretrain_step','finetune_step'):
        fn=getattr(trainer,method)
        def measured(fn=fn):
            stage=trainer.stage;previous=trainer.cursor['actual_optimizer_updates']
            logical=trainer.pretrain_batch+1 if stage=='PRETRAIN' else trainer.finetune_epoch+1
            token=journal.begin(stage,logical);tick=time.perf_counter()
            try:
                value=fn()
            except BaseException:
                journal.end(token,completed=trainer.cursor['actual_optimizer_updates']==previous+1,
                    failed=True,elapsed_seconds=time.perf_counter()-tick)
                control['physical_work']=journal.stats()
                control['optimizer_journal_sha256']=file_sha(journal.path)
                raise
            elapsed=time.perf_counter()-tick
            journal.end(token,completed=True,elapsed_seconds=elapsed)
            control['training_seconds']+=elapsed
            control['physical_work']=journal.stats()
            control['optimizer_journal_sha256']=file_sha(journal.path)
            print(json.dumps(dict(CURRENT_METHOD='M4',REPEAT=repeat,FOLD=fold,STAGE=stage,
                OPTIMIZER_STEP=trainer.cursor['actual_optimizer_updates'],PLANNED_STEPS=trainer.planned_pretrain_batches+500,
                TRAINING_LOSS=value['loss'],FINITE_STATUS=True,ELAPSED_TIME=time.perf_counter()-start,
                LAST_VERIFIED_CHECKPOINT=last['checkpoint_commit_path'] if last else None)),flush=True)
            return value
        setattr(trainer,method,measured)
    try:
        if last is None:commit('fresh_start')
        run_fixed_control(trainer,commit=commit,
            evaluate_inner=lambda epoch:cached('validation',f'epoch_{epoch:04d}',epoch)['metrics'],
            snapshot_weights=lambda:_snapshot(trainer.model.state_dict()))
        if not control['endpoint_complete']:
            current=_snapshot(trainer.model.state_dict())
            trainer.model.load_state_dict(control['best_model'],strict=True)
            try:
                epoch=control['selection']['best']['epoch']
                endpoints={name:cached(name,'endpoint_selected',epoch)
                           for name in ('train_probe','auxiliary_validation')}
            finally:trainer.model.load_state_dict(current,strict=True)
            control['endpoint_metrics']={name:value['metrics'] for name,value in endpoints.items()}
            control['endpoint_complete']=True
            commit('endpoint_complete')
        result=trajectory_receipt(trainer,last,permit,output,prep_seconds,time.perf_counter()-start)
        output.mkdir(parents=True,exist_ok=True)
        temporary=final.with_suffix('.json.partial')
        temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(final)
        return result
    except BaseException as exc:
        output.mkdir(parents=True,exist_ok=True)
        failure=dict(status='FAILED',method='M4',repeat=repeat,fold=fold,error=repr(exc),
            cursor=trainer.cursor,last_verified_checkpoint=last,physical_work=journal.stats(),
            optimizer_journal_sha256=file_sha(journal.path) if journal.path.exists() else None,
            G3_scope_only=True,G4=False,G5=False)
        (output/'FAILED.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        raise


def trajectory_receipt(trainer,checkpoint,permit,output,preparation_seconds,wall_seconds):
    control=trainer.external_control
    selection=InnerSelection.from_state(control['selection'])
    steps=require_endpoint(pretrain_batches=trainer.pretrain_batch,planned_batches=trainer.planned_pretrain_batches,
        finetune_epoch=trainer.finetune_epoch,selection_epochs=[r['epoch'] for r in selection.history],decoder_resets=trainer.decoder_resets)
    if not control['endpoint_complete'] or control['best_model'] is None:raise ValueError('M4 endpoint incomplete')
    journal=UpdateJournal(Path(output)/'optimizer_updates.jsonl')
    journal.require_complete_logical_domain(trainer.pretrain_batch,500)
    physical=journal.stats()
    if control.get('physical_work')!=physical or control.get('optimizer_journal_sha256')!=file_sha(journal.path):
        raise ValueError('M4 committed work counters and physical journal differ')
    return dict(schema='PHASE9_TRAJECTORY_RECEIPT_V1',mode='FORMAL',status='COMPLETE_VERIFIED',method='M4',
        repeat=trainer.identity['repeat'],fold=trainer.identity['fold'],g1_sha256=G1_SHA256,
        amendment_sha256=permit['amendment_sha256'],execution_manifest_sha256=permit['_manifest_sha256'],
        actual_optimizer_updates=steps,planned_optimizer_updates=steps,pretrain_completed_epochs=1,
        pretrain_batches=trainer.pretrain_batch,planned_pretrain_batches=trainer.planned_pretrain_batches,
        completed_epochs=500,selection_epochs=[r['epoch'] for r in selection.history],decoder_resets=1,
        development_metrics_authorized_complete=True,G4=False,G5=False,
        checkpoint_commit_path=checkpoint['checkpoint_commit_path'],checkpoint_path=checkpoint['checkpoint_path'],
        checkpoint_sha256=checkpoint['checkpoint_sha256'],
        authorized_development_metrics=dict(best_inner=selection.best,**control['endpoint_metrics']),
        actual_training_seconds=physical['known_physical_training_seconds'],
        training_time_status=_training_time_status(physical),
        committed_trajectory_training_seconds=control['training_seconds'],
        actual_evaluation_seconds=control['evaluation_seconds'],physical_work=physical,
        optimizer_journal_path=str(journal.path.resolve()),optimizer_journal_sha256=file_sha(journal.path),
        development_artifact_sha256={str(p.relative_to(output)):file_sha(p)
            for p in sorted((Path(output)/'development_cache').rglob('*.json'))},
        actual_preparation_seconds=preparation_seconds,current_invocation_wall_seconds=wall_seconds,
        final_report_absolute_path=str(Path(output)/'receipt.json'),outer_scored=False,final_opened=False)


def verify_trajectory_receipt(receipt,permit,repeat,fold,output):
    from .txgnn_recovery import verify_tx_commit,TxAtomicStore
    output=exact_output(output,permit,repeat,fold)
    for name,value in dict(method='M4',repeat=repeat,fold=fold,g1_sha256=G1_SHA256,mode='FORMAL',
            status='COMPLETE_VERIFIED',amendment_sha256=permit['amendment_sha256'],
            execution_manifest_sha256=permit['_manifest_sha256'],G4=False,G5=False,
            development_metrics_authorized_complete=True,
            final_report_absolute_path=str(Path(output).resolve()/'receipt.json')).items():
        if receipt.get(name)!=value:raise ValueError('M4 final receipt identity/gate differs: '+name)
    checkpoint=verify_tx_commit(receipt['checkpoint_commit_path'])
    store=TxAtomicStore(Path(checkpoint['checkpoint_path']).parent,mode='FORMAL',execution_permit=permit)
    state=store.load(receipt['checkpoint_commit_path'])
    from . import txgnn_recovery as recovery
    from . import production_state as state_helpers
    from .txgnn_adapter import verify_native_sources
    import dgl
    import torch
    actual_source=dict(native=verify_native_sources(),recovery=file_sha(recovery.__file__),
        state_helpers=file_sha(state_helpers.__file__),
        dgl_loader=file_sha(Path(dgl.__file__).parent/'dataloading/dataloader.py'),
        torch_loader=file_sha(Path(torch.__file__).parent/'utils/data/dataloader.py'))
    if state['source']!=actual_source:raise ValueError('Completed M4 training implementation differs')
    cell=next(c for c in frozen_science()['cells'] if (c['repeat_index'],c['fold_index'])==(repeat,fold))
    expected_identity=dict(method='M4',repeat=repeat,fold=fold,seed=cell['training_seed'],cell_digest=cell['cell_digest'],
        g1_sha256=G1_SHA256,mode='FORMAL',amendment_sha256=permit['amendment_sha256'],
        execution_manifest_sha256=permit['_manifest_sha256'],runner_sha256=file_sha(__file__))
    if any(state['identity'].get(k)!=v for k,v in expected_identity.items()):raise ValueError('M4 final state belongs to another cell/execution')
    if (checkpoint['checkpoint_sha256']!=receipt['checkpoint_sha256']
        or Path(checkpoint['checkpoint_path']).parent!=output/'checkpoints'
        or checkpoint['checkpoint_path']!=receipt['checkpoint_path']):raise ValueError('M4 receipt/commit path or content differs')
    cursor=state['cursor'];control=state['external_control']
    selector=InnerSelection.from_state(control['selection'])
    steps=require_endpoint(pretrain_batches=cursor['pretrain_batch'],planned_batches=state['scope']['planned_pretrain_batches'],
        finetune_epoch=cursor['finetune_epoch'],selection_epochs=[r['epoch'] for r in selector.history],
        decoder_resets=state['transition']['decoder_reset_count'])
    if (receipt['actual_optimizer_updates']!=steps or cursor['actual_optimizer_updates']!=steps
        or receipt['selection_epochs']!=list(SELECTION_EPOCHS) or not control['endpoint_complete']):
        raise ValueError('M4 receipt lacks complete two-stage and development work')
    expected_metrics=dict(best_inner=selector.best,**control['endpoint_metrics'])
    if (receipt['planned_optimizer_updates']!=steps or receipt['pretrain_completed_epochs']!=1
        or receipt['completed_epochs']!=500 or receipt['decoder_resets']!=1
        or receipt['pretrain_batches']!=cursor['pretrain_batch']
        or receipt['planned_pretrain_batches']!=state['scope']['planned_pretrain_batches']
        or receipt.get('outer_scored') is not False or receipt.get('final_opened') is not False
        or control.get('evaluation_forwards')!=22 or control.get('best_model') is None
        or receipt['authorized_development_metrics']!=expected_metrics
        or receipt['committed_trajectory_training_seconds']!=control['training_seconds']
        or receipt['actual_evaluation_seconds']!=control['evaluation_seconds']):
        raise ValueError('M4 receipt counters/metrics differ from committed state')
    journal_path=output/'optimizer_updates.jsonl'
    journal=UpdateJournal(journal_path)
    journal.require_complete_logical_domain(cursor['pretrain_batch'],cursor['finetune_epoch'])
    physical=journal.stats()
    if (receipt.get('physical_work')!=physical or control.get('physical_work')!=physical
        or receipt.get('optimizer_journal_path')!=str(journal_path.resolve())
        or receipt.get('optimizer_journal_sha256')!=file_sha(journal_path)
        or control.get('optimizer_journal_sha256')!=file_sha(journal_path)
        or receipt.get('actual_training_seconds')!=physical['known_physical_training_seconds']
        or receipt.get('training_time_status')!=_training_time_status(physical)):
        raise ValueError('M4 receipt physical counters/metrics or journal identity differ')
    artifacts=receipt.get('development_artifact_sha256',{})
    required={f'development_cache/epoch_{e:04d}/validation.{suffix}.json'
              for e in SELECTION_EPOCHS for suffix in ('COMPLETE','ROWS')}
    required|={f'development_cache/endpoint_selected/{name}.{suffix}.json'
               for name in ('train_probe','auxiliary_validation') for suffix in ('COMPLETE','ROWS')}
    normalized={str(Path(k)).replace('\\','/') for k in artifacts}
    if not required<=normalized:raise ValueError('M4 complete development artifacts missing')
    for key,sha in artifacts.items():
        path=(output/key).resolve()
        if not path.is_relative_to(output/'development_cache') or file_sha(path)!=sha:
            raise ValueError('M4 development artifact identity differs')
    return checkpoint


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',required=True);parser.add_argument('--method',required=True)
    parser.add_argument('--repeat',type=int,required=True);parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--execution-manifest',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.mode!='FORMAL' or args.method!='M4':raise PermissionError('This entry only serves independently authorized formal M4')
    return run_formal(args.repeat,args.fold,args.output,args.execution_manifest,resume=args.resume)

if __name__=='__main__':main()
