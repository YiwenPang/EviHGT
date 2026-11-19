"""Read actual DEV96 cell schedules and E0-G external plans, never resample U."""
from dataclasses import asdict
from pathlib import Path
from evi_hgt.phase85_part3a_r2.conditional import group_digest,validate_pairing_plan
from evi_hgt_dev96 import protocol as old
from evi_hgt_dev96.sampling import conditional_plan as historical_plan
from .protocol import *


def flatten_cycles(data):
    return [g for cycle in (1,2,3) for g in data.epoch_groups(cycle)]


def freeze_schedule(data,corpus):
    cell=data.cell_id;cm=read(old.MANIFEST)['cells'][cell];s=cell_spec(cell)
    schedule=read(cm['schedule_path']);groups=flatten_cycles(data)
    stream=[dict(step=i+1,member_sha256=digest([asdict(q) for q in g])) for i,g in enumerate(groups)]
    if (len(groups)!=s['total'] or group_digest(groups)!=schedule['group_digest'] or stream!=schedule['member_stream']
        or corpus['digest']!=cm['corpus_digest'] or schedule['digest']!=cm['schedule_digest']):raise ValueError('Cell actual training schedule changed')
    return groups,stream,schedule,Path(cm['schedule_path']).parent


def conditional_plan(data,groups,schedule,*,cell,origin_commit=None):
    # The caller's new-arm G is deliberately not a sampling origin.
    anchor=old.OUT/cell/old.ARMS[0]/'checkpoints'/f"step_{cell_spec(cell)['G']:08d}.COMMIT.json"
    path=Path(read(old.MANIFEST)['cells'][cell]['schedule_path']).parent/'conditional_v1/plan.json'
    if not path.exists():raise FileNotFoundError('Actual DEV96 external conditional plan unavailable')
    before=file_sha(path)
    plan=historical_plan(data,groups,schedule,cell=cell,origin_commit=anchor)
    if file_sha(path)!=before:raise ValueError('Historical plan bytes changed')
    expected=read(HISTORY/cell/REFERENCE_ARM/'paired_plan_binding.json')
    if plan['digest']!=expected['plan_digest'] or plan['origin_sha256']!=expected['origin_sha256']:raise ValueError('Reference sampling differs')
    return plan
