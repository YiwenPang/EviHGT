"""A private DEV96 driver instance with narrowly replaced dependencies.

No edit/global monkeypatch of the old module. The original driver owns loops,
finite-zero AdamW, atomic boundary accounting, and checkpoint cadence.
"""
import copy
import importlib.util
from pathlib import Path
from evi_hgt_dev96 import runner as original
from evi_hgt_dev96.training import execute_update as original_update
from evi_hgt_dev96.diagnostics import diagnose as original_diagnose
from evi_hgt_dev94_recovery.runner import Events as OriginalEvents
from . import protocol as p
from . import checkpoint as cp
from . import sampling,evaluation,model
from .safety import no_conflicts


def execute_update(assembler,built,optimizer,queries,record=None,*,arm,**kwargs):
    if arm not in p.ARMS:raise PermissionError('Only DEV97 ablation updates')
    return original_update(assembler,built,optimizer,queries,record,arm=p.REFERENCE_ARM,**kwargs)


def logged_updates(cell,arm):
    path=p.OUT/cell/arm/'updates.jsonl'
    import json
    return [json.loads(line) for line in path.read_text('utf-8').splitlines()] if path.exists() else []


PAIR_KEYS=('training_member_sha256','record_sha256','sampling_sha256','stage','queries','original_queries',
           'conditional_extra_U_queries','support_pairs','contra_pairs')


def verify_actual_pairing(cell,arm,*,complete=True):
    reference={r['optimizer_step']:r for r in original.logged_updates(cell,p.REFERENCE_ARM)}
    rows=logged_updates(cell,arm);latest={r['optimizer_step']:r for r in rows}
    if complete and sorted(latest)!=list(range(1,p.cell_spec(cell)['total']+1)):raise ValueError('Incomplete step stream')
    for step,row in latest.items():
        if any(row[k]!=reference[step][k] for k in PAIR_KEYS):raise ValueError(f'Actual REF pairing differs at {cell}/{arm}/{step}')
    return dict(status='ALL_STEPS_ACTUALLY_PAIRED' if complete else 'RECORDED_PREFIX_PAIRED',steps=len(latest),
        keys=list(PAIR_KEYS),reference='DEV96/E1',conditional_origin='same-cell DEV96/E0 genuine G SHA; external fixed plan')


def facility(cell,arm,resume=False):
    spec=importlib.util.spec_from_file_location('evi_hgt_dev96._dev97_driver_'+cell+'_'+arm,original.__file__)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for k in ('OUT','ARMS','LOSSES','QUEUE','write','journal','identity','checkpoint_due','cell_spec','read','digest','file_sha'):
        setattr(m,k,getattr(p,k))
    def manifest():
        result=copy.deepcopy(p.validate_manifest())
        result['cells'][cell]['initial_metadata'].update(dev97_ablation=arm,intervention_boundary='PHASE92_M2' if arm==p.ARMS[0] else 'PHASE92_M3')
        return result
    m.validate_manifest=manifest
    m.build_arm=lambda c,pre:model.build_arm(c,pre,arm)
    m.freeze_schedule=sampling.freeze_schedule;m.conditional_plan=sampling.conditional_plan
    m.execute_update=execute_update;m.Store=cp.Store;m.capture=cp.capture;m.restore=cp.restore;m.verify_commit=cp.verify_commit
    m.no_conflicts=no_conflicts
    m.evaluate=evaluation.evaluate;m.verify_result=evaluation.verify_result
    def diagnose(c,assembler,built,opt,groups,records,step,*,arm,panels=None,ui=None):
        r=original_diagnose(c,assembler,built,opt,groups,records,step,arm=p.REFERENCE_ARM,panels=panels,ui=ui)
        r['dev97_ablation']=arm
        return r
    m.diagnose=diagnose
    verify=m.verify_receipt
    def receipt(r,ident):
        result=verify(r,ident);verify_actual_pairing(cell,arm);return result
    m.verify_receipt=receipt
    class Events(OriginalEvents):
        def stage(self,title,detail):
            import json
            title=title.replace('DEV96','DEV97')
            detail=detail.replace('独立fresh模型','初始来源fresh；当前动作'+('完整checkpoint恢复' if resume else '从未训练状态开始'))
            print(json.dumps(dict(event='stage',args=[title,detail]),ensure_ascii=False),flush=True)
    m.Events=Events
    return m


def run_arm(cell,arm,resume=False):return facility(cell,arm,resume).run_arm(cell,arm,resume=resume)
def verify_receipt(r,ident):return facility(ident['cell'],ident['arm']).verify_receipt(r,ident)
