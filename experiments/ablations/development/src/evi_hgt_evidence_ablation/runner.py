"""A private PAIRED_LOSSES driver instance with narrowly replaced dependencies.

No edit/global monkeypatch of the old module. The original driver owns loops,
finite-zero AdamW, atomic boundary accounting, and checkpoint cadence.
"""
from ablation_protocol_ids import (
    CONTENT_INTERVENTION_BOUNDARY, EVIDENCE_INTERVENTION_FIELD, QUALITY_INTERVENTION_BOUNDARY,
)
import ablation_artifacts as artifacts
import copy
import importlib.util
from pathlib import Path
from evi_hgt_paired_losses import runner as original
from evi_hgt_paired_losses.training import execute_update as original_update
from evi_hgt_paired_losses.diagnostics import diagnose as original_diagnose
from evi_hgt_feature_baseline_recovery.runner import Events as OriginalEvents
from . import protocol as p
from . import checkpoint as cp
from . import sampling,evaluation,model
from .safety import no_conflicts


def execute_update(assembler,built,optimizer,queries,record=None,*,arm,**kwargs):
    if arm not in p.ARMS:raise PermissionError('Only EVIDENCE_ABLATION ablation updates')
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
        keys=list(PAIR_KEYS),reference=artifacts.PAIRED_REFERENCE,conditional_origin=artifacts.EVIDENCE_CONDITIONAL_ORIGIN)


def facility(cell,arm,resume=False):
    spec=importlib.util.spec_from_file_location(f"{original.__package__}._evidence_runner_{cell}_{arm}",original.__file__)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for k in ('OUT','ARMS','LOSSES','QUEUE','write','journal','identity','checkpoint_due','cell_spec','read','digest','file_sha'):
        setattr(m,k,getattr(p,k))
    def manifest():
        result=copy.deepcopy(p.validate_manifest())
        result['cells'][cell]['initial_metadata'].update({
            EVIDENCE_INTERVENTION_FIELD: arm,
            'intervention_boundary': CONTENT_INTERVENTION_BOUNDARY if arm == p.ARMS[0] else QUALITY_INTERVENTION_BOUNDARY,
        })
        return result
    m.validate_manifest=manifest
    m.build_arm=lambda c,pre:model.build_arm(c,pre,arm)
    m.freeze_schedule=sampling.freeze_schedule;m.conditional_plan=sampling.conditional_plan
    m.execute_update=execute_update;m.Store=cp.Store;m.capture=cp.capture;m.restore=cp.restore;m.verify_commit=cp.verify_commit
    m.no_conflicts=no_conflicts
    m.evaluate=evaluation.evaluate;m.verify_result=evaluation.verify_result
    def diagnose(c,assembler,built,opt,groups,records,step,*,arm,panels=None,ui=None):
        r=original_diagnose(c,assembler,built,opt,groups,records,step,arm=p.REFERENCE_ARM,panels=panels,ui=ui)
        r[EVIDENCE_INTERVENTION_FIELD]=arm
        return r
    m.diagnose=diagnose
    verify=m.verify_receipt
    def receipt(r,ident):
        result=verify(r,ident);verify_actual_pairing(cell,arm);return result
    m.verify_receipt=receipt
    class Events(OriginalEvents):
        def stage(self,title,detail):
            import json
            title=title.replace('配对损失实验','证据消融实验')
            detail=detail.replace('独立初始化模型','从完整检查点恢复' if resume else '从初始权重开始训练')
            print(json.dumps(dict(event='stage',args=[title,detail]),ensure_ascii=False),flush=True)
    m.Events=Events
    return m


def run_arm(cell,arm,resume=False):return facility(cell,arm,resume).run_arm(cell,arm,resume=resume)
def verify_receipt(r,ident):return facility(ident['cell'],ident['arm']).verify_receipt(r,ident)
