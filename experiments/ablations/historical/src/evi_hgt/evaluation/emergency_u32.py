"""Post-launch/pre-results U32 amendment: identities, counts and budget only.

The unchanged eligible universe and HMAC-SHA256 family are reused. No model,
prediction, scientific evaluator or result store is accessed by this module.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import hmac
import json
import math

from evi_hgt.evaluation.formal_redesign import (
    FORMAL_RUN_ID,canonical_cells,comparison_protocol,digest,require)

PROTOCOL_ID='STANDARD_SAMPLED_PU_COMPARISON_HMAC_SHA256_U32_V2'
U_LIMIT=32


def protocol():
    value=comparison_protocol()
    value.update(protocol_version=PROTOCOL_ID,u_limit=U_LIMIT,ranking_k=[5,10],internal_ranking_k=[5,10],
        ranking_emphasis='SECONDARY',primary_emphasis='DISEASE_WISE_P_VS_U_ROC_AUC_AND_AP',
        excluded_sampled_reporting_k=[20,50])
    return value


def u_key(disease,drug):
    message=json.dumps([disease,drug],ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return hmac.digest(PROTOCOL_ID.encode('utf-8'),message,'sha256'),drug


def seal(value,field):return {**value,field:digest(value)}


def build_manifest(candidate_stream,*,source_binding,amendment_id,old_manifest_digest,progress=False):
    from tqdm import tqdm
    rows=[];seen=set()
    for source in tqdm(candidate_stream,desc='[U32_MANIFEST][RUNNING]',unit='disease',
            disable=not progress,dynamic_ncols=True,leave=True,mininterval=2):
        require((source.split_kind,source.partition,source.channel)==('standard','test','support'),
            'U32 requires original Standard Formal Policy B support candidates')
        d=source.disease_id;require(d not in seen,'duplicate disease');seen.add(d)
        eligible=set(source.candidate_drug_ids);p=sorted(set(source.held_out_relevant_drug_ids))
        require(len(eligible)==len(source.candidate_drug_ids) and len(p)==len(source.held_out_relevant_drug_ids),
            'duplicate eligible candidate or positive')
        require(set(p)<=eligible,'positive outside immutable eligible universe')
        u=sorted(sorted(eligible-set(p),key=lambda drug:u_key(d,drug))[:U_LIMIT]);ids=sorted(set(p)|set(u))
        row=dict(protocol_version=PROTOCOL_ID,disease_id=d,positive_drug_ids=p,u_drug_ids=u,candidate_drug_ids=ids,
            positive_count=len(p),u_count=len(u),total_candidate_count=len(ids),eligible_candidate_count=len(eligible),
            eligible_membership_digest=digest(sorted(eligible)),source_candidate_membership_digest=source.candidate_membership_digest,
            source_definition_digest=source.definition_digest,positive_digest=digest(p),u_digest=digest(u),
            candidate_order_digest=digest(ids))
        rows.append(seal(row,'candidate_digest'))
    rows.sort(key=lambda row:row['disease_id']);require(bool(rows),'empty disease manifest')
    spec=protocol()
    return seal(dict(format='EVIHGT_COMPARISON_MANIFEST_U32_V2',amendment_id=amendment_id,
        protocol=spec,protocol_digest=digest(spec),source_binding=source_binding,
        old_manifest_digest=old_manifest_digest,
        supersession=dict(relationship='SUPERSEDES_LAYER1_CANDIDATES_AND_SAMPLED_REPORTING_ONLY',
            previous_global_manifest_digest=old_manifest_digest,old_artifacts_preserved=True,
            old_partial_results_reusable=False),
        disease_set_digest=digest([r['disease_id'] for r in rows]),diseases=rows,
        totals=dict(diseases=len(rows),positives=sum(r['positive_count'] for r in rows),
            u=sum(r['u_count'] for r in rows),pairs=sum(r['total_candidate_count'] for r in rows))),
        'global_manifest_digest')


def validate_manifest(manifest,expected_digest=None):
    actual=manifest['global_manifest_digest']
    if expected_digest is not None:require(actual==expected_digest,'external manifest digest changed')
    require(digest({k:v for k,v in manifest.items() if k!='global_manifest_digest'})==actual,'manifest payload changed')
    require(manifest['protocol']==protocol() and manifest['protocol_digest']==digest(protocol()),'U32 protocol changed')
    require(manifest['supersession']==dict(relationship='SUPERSEDES_LAYER1_CANDIDATES_AND_SAMPLED_REPORTING_ONLY',
        previous_global_manifest_digest=manifest['old_manifest_digest'],old_artifacts_preserved=True,
        old_partial_results_reusable=False),'supersession changed')
    ids=[r['disease_id'] for r in manifest['diseases']]
    require(ids==sorted(set(ids)) and digest(ids)==manifest['disease_set_digest'],'disease set/order changed')
    for r in manifest['diseases']:
        require(r['candidate_digest']==digest({k:v for k,v in r.items() if k!='candidate_digest'}),'candidate digest changed')
        p,u,c=(r[k] for k in ('positive_drug_ids','u_drug_ids','candidate_drug_ids'))
        require(p==sorted(set(p)) and u==sorted(set(u)) and not set(p)&set(u),'P/U membership invalid')
        require(c==sorted(set(p)|set(u)),'candidate order changed')
        require((len(p),len(u),len(c))==(r['positive_count'],r['u_count'],r['total_candidate_count']),'counts changed')
        require(len(u)==min(U_LIMIT,r['eligible_candidate_count']-len(p)),'U quota changed')
        require((digest(p),digest(u),digest(c))==(r['positive_digest'],r['u_digest'],r['candidate_order_digest']),
            'membership/order hashes changed')
    require(manifest['totals']==dict(diseases=len(ids),positives=sum(r['positive_count'] for r in manifest['diseases']),
        u=sum(r['u_count'] for r in manifest['diseases']),pairs=sum(r['total_candidate_count'] for r in manifest['diseases'])),
        'manifest totals changed')


def build_layer1_plan(manifest,old_plan,amendment_id):
    validate_manifest(manifest);cells=deepcopy(canonical_cells(old_plan['cells']))
    require(old_plan['formal_run_id']==FORMAL_RUN_ID and old_plan['formal_test_count']==1,'Formal identity changed')
    require(manifest['source_binding']==old_plan['data_binding'],'source data binding differs')
    require(manifest['amendment_id']==amendment_id,'amendment binding differs')
    plan=deepcopy(old_plan);spec=deepcopy(old_plan['protocol']);spec['layer1']=protocol()
    plan.update(format='EVIHGT_LAYER1_COMPARISON_U32_FROZEN_V1',amendment_id=amendment_id,
        readiness='LAYER1_FROZEN; AWAITING_LAYER2_COLD_SCIENTIFIC_REDESIGN; EXECUTION_FORBIDDEN',
        execution_authorized=False,cells=cells,checkpoint_binding_digest=digest(cells),protocol=spec,
        protocol_digest=digest(spec),comparison_manifest_digest=manifest['global_manifest_digest'],
        output_root='outputs/phase8_formal/evaluation/'+amendment_id.lower()+'_layer1_u32_v2',
        old_output_root=old_plan['output_root'],old_manifest_digest=manifest['old_manifest_digest'],
        deferred_units=deepcopy([u for u in old_plan['units'] if u['layer']!='A']),
        deferred_status='AWAITING_SCIENTIFIC_REDESIGN; DO_NOT_EXECUTE',
        blindness=dict(scientific_results_inspected=False,top_k_inspected=False,model_comparison_inspected=False,
            old_formal_scoring_started=True,old_attempt_interrupted=True),
        old_partial_score_reuse=False,automatic_retry_allowed=False)
    plan['units']=[dict(unit_id='A_'+cell['label'],layer='A',split='standard',label=cell['label'],
        model_variant=cell['model_variant'],comparison_manifest_digest=manifest['global_manifest_digest'],
        benchmark='SAMPLED COMPARISON BENCHMARK',pairs=manifest['totals']['pairs'],diseases=manifest['totals']['diseases'],
        ranking_k=[5,10],primary_metrics=['ROC_AUC_P_VS_U','AP_P_VS_U'],ranking_emphasis='SECONDARY') for cell in cells[:6]]
    return plan


def validate_resume_binding(current,previous):
    require(digest(current)==digest(previous),'resume requires identical complete plan/protocol/manifest/checkpoints/output binding')
    require(current['output_root']!=current.get('old_output_root'),'old interrupted output root forbidden')


def engineering_budget(pairs,rate,startup_seconds,margin=1.15,budget_seconds=10800):
    require(isinstance(pairs,int) and pairs>=0,'invalid pair count')
    require(all(math.isfinite(v) for v in (rate,startup_seconds,margin,budget_seconds)) and
        rate>0 and startup_seconds>=0 and margin>=1 and budget_seconds>0,'invalid engineering budget inputs')
    scoring=pairs/rate;seconds=startup_seconds+scoring*margin
    return dict(pairs=pairs,rate_pairs_per_second=rate,startup_seconds=startup_seconds,scoring_seconds=scoring,
        margin=margin,budget_seconds=budget_seconds,estimated_seconds=seconds,
        physical_seconds_without_margin=startup_seconds+scoring,
        capacity_pairs=max(0,math.floor((budget_seconds-startup_seconds)*rate/margin)),
        gate='PASS' if seconds<=budget_seconds else 'FAIL')


def validate_interruption(receipt):
    require(receipt['state']=='INTERRUPTED' and receipt['process_stopped'] is True,'old process not safely interrupted')
    require((receipt['reason'],receipt['reason_detail'])==('COMPUTATIONAL_FEASIBILITY',
        'OBSERVED_RUNTIME_EXCEEDS_MASTER_PAPER_BUDGET'),'interruption reason differs')
    require(receipt['automatic_resume_forbidden'] is True and receipt['partial_artifact_disposition']==
        'PARTIAL / INTERRUPTED / NOT FOR SCIENTIFIC ANALYSIS','partial artifact disposition unsafe')
    for field in ('scientific_artifacts_read','scientific_metrics_inspected','scientific_values_printed',
        'top_k_inspected','model_comparison_inspected'):require(receipt[field] is False,'scientific blindness changed: '+field)
    require(datetime.fromisoformat(receipt['stop_requested_utc'])<=datetime.fromisoformat(receipt['stop_completed_utc']),
        'stop timestamps inconsistent')
    s=receipt['operational_final'];before=receipt['operational_snapshot']
    require(s['state']=='INTERRUPTED' and s['formal_run_id']==FORMAL_RUN_ID and s['formal_test_count']==1 and
        s['graph_batch_size']==1 and s['stage']=='A','old Formal interruption state inconsistent')
    require(0<=s['pairs_committed']<=s['pairs_processed']<s['pairs_total'] and
        math.isfinite(s['pairs_per_second']) and s['pairs_per_second']>0 and s['elapsed_seconds']>0,'engineering counters invalid')
    for field in ('formal_run_id','formal_test_count','protocol_digest','comparison_manifest_digest',
        'plan_sha256','checkpoint_binding_digest','source_data_identity'):require(s[field]==before[field],'stop binding changed')
