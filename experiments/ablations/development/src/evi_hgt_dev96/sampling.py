"""Three original complete P cycles; real E0 G digest anchors paired U."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from evi_hgt.phase85_part3a_r2.conditional import group_parts,group_digest,build_pairing_plan,validate_pairing_plan
from evi_hgt.phase9.runner import _plan_bindings
from evi_hgt.phase9.production_state import _rng,training_state_digest
from evi_hgt.phase85_v2.common import bound,verify
from .protocol import *


def freeze_schedule(data,corpus):
    cell=data.cell_id;s=cell_spec(cell);c=frozen_cell(cell);before=training_state_digest(_rng())
    groups=[];cycles=[]
    for cycle in (1,2,3):
        part=data.epoch_groups(cycle)
        if len(part)!=s['G']:raise ValueError('Incomplete cell cycle')
        ps=Counter();pc=Counter()
        for g in part:
            a,b,u=group_parts(g);ps.update(a);pc.update(b)
            for p in a+b:data.pairs.assert_optimizer_pair(p,'P')
            data.pairs.assert_optimizer_pair(u,'U')
        if set(ps)!=data.support or set(pc)!=data.unsuitability or set(ps.values())!={1} or set(pc.values())!={1}:
            raise ValueError('P cycle is not exact complete coverage')
        if cycle==1 and group_digest(part)!=c['global_cycle_group_digest']:raise ValueError('Global cycle order differs')
        if cycle==2 and group_digest(part[:c['budget']['suffix_groups']])!=c['suffix_group_digest']:raise ValueError('Original suffix order differs')
        cycles.append(dict(cycle=cycle,groups=len(part),support_P=len(ps),contra_P=len(pc),group_digest=group_digest(part),
                           positive_coverage='EXACTLY_ONCE_EACH_CHANNEL'))
        groups.extend(part)
    stream=[dict(step=i+1,member_sha256=digest([asdict(q) for q in g])) for i,g in enumerate(groups)]
    schedule=bound(dict(schema='DEV96_THREE_FULL_CYCLES_V1',cell=cell,spec=s,data_bindings=data.bindings,
        corpus_digest=corpus['digest'],mask_digest=data.bindings['fold_mask_digest'],cycles=cycles,
        group_digest=group_digest(groups),member_stream=stream,order='original epoch_groups(1/2/3); original intragroup order'))
    root=CACHE/cell/digest(dict(data=data.bindings,corpus=corpus['digest'],mask=data.bindings['fold_mask_digest']))
    write(root/'schedule.json',schedule,derived=True)
    if before!=training_state_digest(_rng()):raise RuntimeError('Schedule polluted training RNG')
    return groups,stream,schedule,root


def conditional_plan(data,groups,schedule,*,cell,origin_commit=None,test_origin=None,test_root=None):
    s=cell_spec(cell)
    if test_origin is not None:
        if origin_commit is not None or test_root is None or not Path(test_root).resolve().is_relative_to((OUT/'TEST_ONLY').resolve()):
            raise PermissionError('Synthetic origin only under isolated DEV96 TEST_ONLY')
        mode='TEST_ONLY';origin=test_origin;root=Path(test_root)
    else:
        if origin_commit is None:raise PermissionError('Real current-cell E0 G COMMIT is required')
        from .checkpoint import verify_commit
        commit=verify_commit(origin_commit)
        ident=identity(validate_manifest(),cell,ARMS[0])
        if commit['identity']!=ident or commit['step']!=s['G']:raise ValueError('Conditional anchor must be current-cell genuine E0 G')
        origin=commit['checkpoint_sha256'];mode='DEVELOPMENT'
        root=CACHE/cell/digest(dict(data=data.bindings,corpus=schedule['corpus_digest'],mask=data.bindings['fold_mask_digest']))/'conditional_v1'
    before=training_state_digest(_rng())
    bindings=dict(_plan_bindings(data,origin),dev96_namespace='DEV96_E0_G_SHARED_CONDITIONAL_V1',cell=cell,
        corpus_digest=schedule['corpus_digest'],schedule_digest=schedule['digest'],cycles=[2,3],record_order='cycle_then_original_group')
    p=root/'plan.json'
    if p.exists():
        result=verify(read(p))
        if (result['cell']!=cell or result['mode']!=mode or result['origin_sha256']!=origin
            or result['data_bindings']!=data.bindings or result['schedule_digest']!=schedule['digest']):raise ValueError('Cross-cell/origin/mode sampling plan')
        validate_pairing_plan(result['legacy'],data,groups[s['G']:],bindings=bindings)
    else:
        legacy=build_pairing_plan(data,groups[s['G']:],warmup_steps=s['G'],bindings=bindings)
        if not legacy['coverage_gate_pass']:raise ValueError('Original conditional coverage gate failed')
        result=bound(dict(schema='DEV96_SHARED_CONDITIONAL_V1',cell=cell,mode=mode,origin_sha256=origin,
            origin_commit=str(Path(origin_commit).resolve()) if origin_commit else None,
            data_bindings=data.bindings,schedule_digest=schedule['digest'],legacy=legacy,
            origin_use='E0_G_SHA_ONLY_PRIVATE_UNIFORM_SAMPLING_NO_SCORES_WEIGHTS_OPTIMIZER_OR_VALIDATION'))
        write(p,result,derived=(mode=='DEVELOPMENT'))
    if before!=training_state_digest(_rng()):raise RuntimeError('Conditional sampling polluted training RNG')
    return result
