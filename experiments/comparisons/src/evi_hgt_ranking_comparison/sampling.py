"""Freeze real, score-blind three-cycle plans before any model execution."""
from .protocol_ids import PLAN_ORIGIN_SCHEMA, CONDITIONAL_BINDINGS_SCHEMA
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import os
import random
import time

import polars as pl
from evi_hgt.graph_inputs.common import digest, bound, verify
from evi_hgt.conditional_sampling.conditional import group_digest, group_parts
from .common import ROOT, OUT, read, sha, write_new
from .data import TrainingInputs

CACHE=ROOT/'data/cache/ranking_comparison'
EXISTING=('R0F0','R0F1','R1F0')


def publish(path, value):
    path=Path(path)
    if not path.resolve().is_relative_to(CACHE.resolve()) and not path.resolve().is_relative_to((OUT/'sampling').resolve()):
        raise PermissionError('RankingComparison sampling output namespace required')
    if path.exists():
        if read(path)!=value:
            raise ValueError('Immutable sampling artifact differs: '+str(path))
        return path
    return write_new(path,value)


def static_origin(*, cell, training_seed, membership_digests, mask_digest, graph_parent_identity):
    payload=dict(namespace=PLAN_ORIGIN_SCHEMA,cell=cell,original_training_seed=training_seed,
        membership_digests=membership_digests,fold_mask_digest=mask_digest,graph_parent_identity=graph_parent_identity)
    return dict(origin_kind='STATIC_METADATA',origin_sha256=digest(payload),payload=payload,
        canonical_serialization='UTF8 JSON sort_keys=True ensure_ascii=True separators=(comma,colon) allow_nan=False',
        checkpoint_or_model_used=False)


def uniform_legal_pair(drugs, disease, excluded, seed):
    legal=tuple(drug for drug in drugs if drug not in excluded)
    return (legal[random.Random(seed).randrange(len(legal))],disease) if legal else None


def conditional_records(data, groups, *, warmup_steps, bindings):
    drugs=tuple(data.pairs.universe.eligible_drugs)
    drug_set=set(drugs); disease_set=set(data.pairs.universe.eligible_diseases)
    wanted={p[1] for g in groups for p in group_parts(g)[0]}
    forbidden=set(data.pairs.universe.observed_pairs)|data.pairs.gate.formal|data.pairs.gate.blocked|data.pairs.gate.positive
    excluded=defaultdict(set)
    for drug,disease in forbidden:
        if drug in drug_set and disease in wanted:
            excluded[disease].add(drug)
    # Precomputed exact complements preserve the original eligible-drug rank order.
    legal={d:tuple(x for x in drugs if x not in excluded[d]) if d in disease_set else () for d in wanted}
    rows=[]; occurrences=covered=extras_count=0
    for index,group in enumerate(groups):
        support,contra,original=group_parts(group)
        for pair in support+contra: data.pairs.assert_optimizer_pair(pair,'P')
        data.pairs.assert_optimizer_pair(original,'U')
        local={};seeds={};statuses={};extras=[]
        original_pairs={(q.drug_id,q.disease_id) for q in group}
        for disease,count in Counter(p[1] for p in support).items():
            occurrences+=count
            seed=int(digest(['PART3A_R2_CONDITIONAL_UNIFORM_V1',bindings,index,disease])[:16],16)
            seeds[disease]=seed
            if not legal[disease]:
                local[disease]=[];statuses[disease]='NO_LEGAL_U_ORIGINAL_PAIRING_FALLBACK'
                continue
            pair=(legal[disease][random.Random(seed).randrange(len(legal[disease]))],disease)
            data.pairs.assert_optimizer_pair(pair,'U')
            local[disease]=[list(pair)];statuses[disease]='DISEASE_CONDITIONAL_UNIFORM';covered+=count
            if pair not in original_pairs and pair not in extras: extras.append(pair)
        extras_count+=len(extras)
        rows.append(dict(group_index=index,logical_step=warmup_steps+index+1,original_global_u=list(original),
            support_positive_pairs=[list(p) for p in support],contra_positive_pairs=[list(p) for p in contra],
            original_support_u_count=1,disease_unlabeled_pairs=local,draw_seed_by_disease=seeds,
            disease_status=statuses,extra_pairs=[list(p) for p in extras]))
    return rows,dict(support_positive_occurrences=occurrences,conditional_positive_occurrences=covered,
        fallback_positive_occurrences=occurrences-covered,support_occurrence_coverage=covered/occurrences,
        coverage_gate_pass=covered/occurrences>=.95,extra_forward_count=extras_count,
        original_suffix_forward_count=sum(map(len,groups)),legal_drug_counts={d:len(v) for d,v in legal.items()})


def historical_paths(cell):
    if cell=='R0F0':
        return ROOT/'data/processed/feature_baseline/schedule_v1/schedule.json',ROOT/'data/processed/feature_baseline/conditional_v1/plan.json'
    manifest=read(ROOT/'outputs/ranking_protocol/paired_losses/PAIRED_LOSSES_EXECUTION_MANIFEST.json')
    path=Path(manifest['cells'][cell]['schedule_path'])
    return path,path.parent/'conditional_v1/plan.json'


def graph_parent():
    clean=ROOT/'data/processed/inner_validation/LEAKAGE_SAFE_OT_DISEASE_GENE_V1/manifest.json'
    base=ROOT/'data/processed/inner_validation/pilot_adapter_v1/manifest.json'
    adapter=ROOT/'outputs/message_direction/revision_r2/direction_adapter.py'
    return dict(clean_graph_digest=verify(read(clean))['digest'],clean_manifest_sha256=sha(clean),
        certified_unlabeled_parent_digest=verify(read(base))['digest'],parent_manifest_sha256=sha(base),
        canonical_message_adapter_sha256=sha(adapter),cell_mask_separately_bound=True)


def freeze_cell(data, parent):
    started=time.monotonic(); state=random.getstate(); G=data.budget()['G_rf']
    groups=[];cycles=[];rows=[];stream=[]
    for cycle in (1,2,3):
        part=data.epoch_groups(cycle)
        support=Counter(p for g in part for p in group_parts(g)[0]);contra=Counter(p for g in part for p in group_parts(g)[1])
        if len(part)!=G or set(support)!=data.support or set(contra)!=data.unsuitability or set(support.values())!={1} or set(contra.values())!={1}:
            raise ValueError('Incomplete original positive cycle')
        gd=group_digest(part)
        if cycle==1 and gd!=data.contract.bindings['global_cycle_group_digest']:
            raise ValueError('Original global cycle differs')
        if cycle==2 and group_digest(part[:data.budget()['suffix_groups']])!=data.contract.bindings['suffix_group_digest']:
            raise ValueError('Original suffix order differs')
        cycles.append(dict(cycle=cycle,groups=G,group_digest=gd,support_P=len(support),contra_P=len(contra),positive_coverage='EXACTLY_ONCE'))
        for group in part:
            step=len(stream)+1; member=digest([asdict(q) for q in group]);stream.append(dict(step=step,member_sha256=member))
            for position,q in enumerate(group):
                role='U' if q.observed_relation is None else 'support' if q.support.observed_mask else 'unsuitability'
                data.pairs.assert_optimizer_pair((q.drug_id,q.disease_id),'U' if role=='U' else 'P')
                rows.append(dict(step=step,cycle=cycle,position=position,role=role,drug_id=q.drug_id,disease_id=q.disease_id,member_sha256=member))
        groups.extend(part)
    ref=None
    if data.cell_id in EXISTING:
        sp,pp=historical_paths(data.cell_id); old_schedule=verify(read(sp));old_plan=verify(read(pp))
        if data.cell_id=='R0F0':
            if any(cycles[i]['group_digest']!=old_schedule['cycles'][i]['group_digest'] for i in range(3)):
                raise ValueError('Historical FEATURE_BASELINE cycles differ')
        elif old_schedule['member_stream']!=stream or old_schedule['group_digest']!=group_digest(groups):
            raise ValueError('Historical PAIRED_LOSSES members/order differ')
        bindings=old_plan['legacy']['bindings']
        records,counts=conditional_records(data,groups[G:],warmup_steps=G,bindings=bindings)
        if records!=old_plan['legacy']['records'][:2*G]:
            raise ValueError('Historical conditional draw/order differs')
        origin=dict(origin_kind='EXISTING_REGISTERED_DEV_PLAN',origin_sha256=old_plan['origin_sha256'],checkpoint_or_model_used=False)
        ref=dict(schedule_path=str(sp),schedule_sha256=sha(sp),plan_path=str(pp),plan_sha256=sha(pp),
            original_plan_digest=old_plan['digest'],record_slice=[0,2*G],source_files_unchanged=True)
    else:
        origin=static_origin(cell=data.cell_id,training_seed=data.training_seed,
            membership_digests=data.contract.bindings['membership_digests'],mask_digest=data.bindings['fold_mask_digest'],graph_parent_identity=parent)
        bindings=dict(schema=CONDITIONAL_BINDINGS_SCHEMA,cell=data.cell_id,
            fold_manifest_digest=data.bindings['fold_manifest_digest'],repeat_key=data.cell['repeat_key'],fold=data.cell['fold'],
            training_seed=data.training_seed,training_u_manifest_digest=data.u['digest'],exclusion_digest=data.bindings['fold_mask_digest'],
            origin_kind='STATIC_METADATA',origin_sha256=origin['origin_sha256'])
        records,counts=conditional_records(data,groups[G:],warmup_steps=G,bindings=bindings)
    if not counts['coverage_gate_pass'] or random.getstate()!=state:
        raise ValueError('Coverage/private RNG gate failed')
    identity=dict(cell=data.cell_id,data_bindings=data.bindings,origin=origin,graph_parent_identity=parent)
    dest=CACHE/data.cell_id/digest(identity)
    dest.mkdir(parents=True,exist_ok=True)
    gp=dest/'global_groups.parquet'
    if not gp.exists():
        tmp=gp.with_suffix('.parquet.partial');pl.DataFrame(rows).write_parquet(tmp);os.replace(tmp,gp)
    else:
        if pl.read_parquet(gp).to_dicts()!=rows: raise ValueError('Existing group rows differ')
    schedule=bound(dict(schema='RANKING_COMPARISON_THREE_GLOBAL_CYCLES_V1',identity=identity,G=G,total=3*G,
        group_digest=group_digest(groups),cycles=cycles,member_stream=stream,groups_path=str(gp),groups_sha256=sha(gp)))
    publish(dest/'schedule.json',schedule)
    plan=bound(dict(schema='RANKING_COMPARISON_SHARED_CONDITIONAL_PLAN_V1',cell=data.cell_id,origin=origin,bindings=bindings,
        schedule_digest=schedule['digest'],source_group_digest=group_digest(groups[G:]),warmup_steps=G,suffix_groups=2*G,
        historical_reference=ref,records=records,counts=counts,record_digest=digest(records),
        uniformity='private random.Random(seed).randrange(Nlegal), exact eligible-drug-order complement; with replacement; no hash modulo',
        seed_rule='int(SHA256(canonical_json([PART3A_R2_CONDITIONAL_UNIFORM_V1,bindings,zero_based_suffix_group_index,disease_id]))[:16],16)',
        support_conditional_cycles=[2,3],contra_original_global_unchanged=True,U_is_confirmed_negative=False,
        no_legal_U_fallback='retain original global U and every positive',shared_by=['FULL','PAIR_MLP_D1_MATCHED','HGT_D1_MATCHED']))
    publish(dest/'conditional_plan.json',plan)
    receipt=bound(dict(schema='RANKING_COMPARISON_CELL_PLAN_SEAL_V1',status='SEALED',cell=data.cell_id,seed=data.training_seed,G=G,
        schedule_path=str(dest/'schedule.json'),schedule_sha256=sha(dest/'schedule.json'),schedule_digest=schedule['digest'],
        plan_path=str(dest/'conditional_plan.json'),plan_sha256=sha(dest/'conditional_plan.json'),plan_digest=plan['digest'],
        origin_kind=origin['origin_kind'],origin_sha256=origin['origin_sha256'],historical_reference=ref,
        mask_digest=data.bindings['fold_mask_digest'],training_membership_digests={n:digest(data.cell[n]) for n in ('training','auxiliary_training')},
        cycles=cycles,counts=counts,projection_audit=data.projection_audit,model_forwards=0,optimizer_updates=0,
        elapsed_seconds=time.monotonic()-started,training_rng_unchanged=True))
    target=OUT/'sampling'/f'{data.cell_id}.json'
    if target.exists():
        old=verify(read(target))
        if any(old[k]!=receipt[k] for k in ('schedule_sha256','plan_sha256','origin_sha256')): raise ValueError('Cell seal changed')
        return old
    publish(target,receipt)
    return receipt


def freeze_all():
    target=OUT/'sampling/ALL_PLANS_SEALED.json'
    if target.exists():
        result=verify(read(target))
        for row in result['cells']:
            for kind in ('schedule','plan'):
                if sha(row[kind+'_path'])!=row[kind+'_sha256']: raise ValueError('Sealed artifact bytes differ')
        return result
    inputs=TrainingInputs();parent=graph_parent();rows=[]
    for r in range(3):
        for f in range(5):
            cell=f'R{r}F{f}'
            rows.append(freeze_cell(inputs.cell(cell),parent))
            print(f'Sealed sampling plan for {cell}: {rows[-1]["origin_kind"]}. Forward passes: 0. Optimizer updates: 0.',flush=True)
    result=bound(dict(schema='RANKING_COMPARISON_ALL_CELL_PLANS_SEALED_V1',status='SEALED_15_CELLS',cells=rows,
        existing_reference_count=3,static_metadata_count=12,formal_training_authorized=False,outer_authorized=False,final_authorized=False,
        source_sha256={str(ROOT/'src/evi_hgt_ranking_comparison'/n):sha(ROOT/'src/evi_hgt_ranking_comparison'/n) for n in ('data.py','sampling.py')},
        total_global_updates=sum(3*r['G'] for r in rows),total_conditional_records=sum(2*r['G'] for r in rows),
        shared_methods=['FULL','PAIR_MLP_D1_MATCHED','HGT_D1_MATCHED'],all_plans_before_model_execution=True))
    publish(target,result)
    return result
