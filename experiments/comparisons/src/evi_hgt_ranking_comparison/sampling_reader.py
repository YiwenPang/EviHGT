"""Read only committed plan members, with exact cell/step binding checks."""
from dataclasses import asdict
import json
from pathlib import Path
import polars as pl
from evi_hgt.graph_inputs.common import digest,verify
from evi_hgt.training.supervision import make_unlabeled_query_supervision
from .common import OUT,read,sha
from .data import objects_in_array


def load_sealed_group(data,step):
    if type(step) is not int or not 1<=step<=3*data.budget()['G_rf']:
        raise ValueError('Exact registered one-based training step required')
    all_seal=verify(read(OUT/'sampling/ALL_PLANS_SEALED.json'))
    if all_seal['status']!='SEALED_15_CELLS':
        raise PermissionError('All 15 plans must be sealed before model tests')
    rows=[x for x in all_seal['cells'] if x['cell']==data.cell_id]
    if len(rows)!=1: raise ValueError('Unknown sealed cell')
    seal=rows[0]
    if seal['seed']!=data.training_seed or seal['mask_digest']!=data.bindings['fold_mask_digest']:
        raise ValueError('Cross-cell sampling identity')
    for kind in ('schedule','plan'):
        if sha(seal[kind+'_path'])!=seal[kind+'_sha256']:
            raise ValueError('Sealed sampling bytes differ')
    schedule=verify(read(seal['schedule_path']))
    if schedule['identity']['data_bindings']!=data.bindings or sha(schedule['groups_path'])!=schedule['groups_sha256']:
        raise ValueError('Sealed schedule/input binding differs')
    members=pl.scan_parquet(schedule['groups_path']).filter(pl.col('step')==step).sort('position').collect().to_dicts()
    group=[]
    for position,row in enumerate(members):
        if row['position']!=position: raise ValueError('Intragroup position differs')
        pair=row['drug_id'],row['disease_id'];role=row['role']
        if role not in ('support','unsuitability','U'):raise ValueError('Unknown training member role')
        data.pairs.assert_optimizer_pair(pair,'U' if role=='U' else 'P')
        group.append(make_unlabeled_query_supervision(*pair) if role=='U' else data.positive(pair,role))
    group=tuple(group);member_sha=digest([asdict(q) for q in group])
    if not group or member_sha!=schedule['member_stream'][step-1]['member_sha256']:
        raise ValueError('Sealed group members/order differ')
    record=None;G=seal['G']
    if step>G:
        target=step-G-1
        for index,raw in enumerate(objects_in_array(seal['plan_path'],'records')):
            if index==target:
                record=json.loads(raw);break
        if record is None or record['logical_step']!=step or record['group_index']!=target:
            raise ValueError('Conditional record cursor differs')
        for pairs in record['disease_unlabeled_pairs'].values():
            for pair in pairs:data.pairs.assert_optimizer_pair(pair,'U')
        for pair in record['extra_pairs']:data.pairs.assert_optimizer_pair(pair,'U')
    identity=dict(cell=data.cell_id,step=step,G=G,member_sha256=member_sha,
        schedule_sha256=seal['schedule_sha256'],plan_sha256=seal['plan_sha256'],plan_digest=seal['plan_digest'],
        mask_digest=seal['mask_digest'],origin_kind=seal['origin_kind'],origin_sha256=seal['origin_sha256'])
    return group,record,identity
