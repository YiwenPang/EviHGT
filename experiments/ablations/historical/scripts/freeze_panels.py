"""counts-only V2 amendment/freezing and conditional execution grant.

This script never scores. Independent overnight execution is a separate entry.
No old scientific shard/result/transcript is opened.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from historical_run import run_original
    raise SystemExit(run_original(Path(__file__)))

import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from evi_hgt.evaluation.formal_redesign import (
    FORMAL_RUN_ID,digest,file_sha,read_json,require,exclusive_json,next_amendment,verify_history,
)
from evi_hgt.evaluation import evaluable_protocol as protocol
from evi_hgt.evaluation.final_protocol import budget
from scripts import holdout_evaluation_final_freeze as previous

OUT=Path('outputs/phase842_evaluable')
DERIVED=Path('data/processed/formal_evaluation_evaluable_v2')
V1_FREEZE=Path('outputs/phase842_final/freeze.json')
V1_FREEZE_SHA='d4a8d67d41df1192dbaf8bb54a1fcbbbc9c3de8f3cd5ab40c0cb03d70bda316e'
V1_PLAN_DIGEST='fbfa838e9daba58757a597eb7cb8a167fd04c2551549d9026036c69887c9b62f'
PINS={'A':'17d4278d4de3007fcf32d433b6931887ad4dbc2860df9ba3ede46dcecb878c54',
    'C':'5fdd1363d6e2fc7e276b61ed248b368873dfb7673a65d6976e0949bcae6095ed',
    'D':'1b65a0a9e2109345e2d89b812eeb28c59eb7de4698f32f008ec0ffa278513bca'}
V1_MANIFEST='267124a85d8dae4862edd872f64e67d375c691f4ec174631ae3bc3567d4f43e3'
SOURCES=['src/evi_hgt/evaluation/evaluable_protocol.py','scripts/phase842_evaluable_freeze.py',
    'tests/test_phase842_evaluable_protocol.py','src/evi_hgt/evaluation/overnight_runtime.py',
    'tests/test_phase842_overnight_runtime.py','scripts/phase842_overnight_execution.py',
    'scripts/start_phase842_overnight_formal.ps1','scripts/phase842_overnight_console.ps1',
    'scripts/confirm_phase842_overnight_launch.ps1','tests/test_phase842_overnight_launcher.py']


def reject_v1_execution_entry(root,plan):
    require(not (Path(root)/plan['output_root']).exists(),'Layer2 V1 execution entry exists; stop without scientific artifact inspection')


def verify_unchanged(plan,prior):
    for stage in ('A','C','D'):
        require([u for u in plan['units'] if u['layer']==stage]==[u for u in prior['units'] if u['layer']==stage],
            'unchanged Stage '+stage+' execution unit mutated')
        require(plan['manifest_bindings'][stage]==prior['manifest_bindings'][stage],'unchanged Stage '+stage+' digest mutated')
    for key in ('cells','checkpoint_binding_digest','data_binding','graph_batch_size','representation_sha256','cold_policy_digest',
            'precision','deterministic_algorithms','training_protocol'):
        if key in prior:require(plan[key]==prior[key],'frozen training/graph/runtime binding changed: '+key)
    for key in ('layer1','cold_drug','cold_disease'):
        require(plan['protocol'][key]==prior['protocol'][key],'unchanged protocol mutated: '+key)


def prior_gate():
    _,original,stop,history,_,layer1,_=previous.prior_gate()
    require(file_sha(ROOT/V1_FREEZE)==V1_FREEZE_SHA,'Part3 final freeze changed')
    f=read_json(ROOT/V1_FREEZE)
    for p,s in [('amendment_path','amendment_sha256'),('history_event_path','history_event_sha256'),('plan_path','plan_sha256')]:
        require(file_sha(ROOT/f[p])==f[s],'Part3 frozen binding changed: '+p)
    for mapping in ('source_sha256','artifact_sha256','history_hashes'):
        for p,sha in f[mapping].items():require(file_sha(ROOT/p)==sha,'Part3 immutable source/evidence/history changed: '+p)
    require(all(history.get(p)==sha for p,sha in f['history_hashes'].items()),'Part3 history prefix mismatch')
    record,manifests,plan=previous.load_draft()
    require(digest(plan)==V1_PLAN_DIGEST and f['execution_plan_digest']==V1_PLAN_DIGEST,'Part3 V1 plan digest changed')
    require(manifests['standard']['global_manifest_digest']==V1_MANIFEST,'Layer2 V1 manifest changed')
    reject_v1_execution_entry(ROOT,plan)
    for stage in PINS:require(plan['manifest_bindings'][stage]==PINS[stage],'unchanged manifest pin mismatch: '+stage)
    require(layer1['global_manifest_digest']==PINS['A'],'Layer1 immutable digest changed')
    require(manifests['cold_drug']['totals']==dict(queries=1313,positives=3965,u=84032,pairs=87997) and
        manifests['cold_disease']['totals']==dict(queries=232,positives=4164,u=14848,pairs=19012),'Cold census changed')
    return f,plan,layer1,manifests,history,original,record


def construct(original,layer1,amendment):
    from evi_hgt.evaluation.candidates import iter_disease_candidates
    source,frames,drugs=previous.source_frames(original)
    stream=iter_disease_candidates(source,frames['standard'],split_kind='standard',partition='test',channel='support',formal_drug_ids=drugs)
    checked=previous.verified_source(stream,original['full_manifests']['standard'])
    cell=next(c for c in original['cells'] if c['split']=='standard' and c['model_variant']=='FULL_EVIHGT')
    manifest=protocol.build_manifest(layer1,checked,source_binding=original['data_binding'],checkpoint=cell,
        amendment_id=amendment,superseded_digest=V1_MANIFEST,progress=True)
    census=manifest['selection']
    require(census['N_all']==1354 and census['N_zero_positive']+census['N_retrieval_evaluable']==1354,'Formal label census inconsistent')
    return manifest


def estimate(manifest,prior_record):
    from copy import deepcopy
    values=deepcopy(prior_record['estimates']);evidence=values['B']
    revised=budget(manifest['totals']['pairs'],evidence['rate_pairs_per_second'],evidence['startup_seconds'])
    revised.update(measured_startup_seconds=evidence['measured_startup_seconds'],preflight_allowance_seconds=50,
        finalization_allowance_seconds=60,evidence_confidence=evidence['evidence_confidence'],measured_rate_reused=True,
        evidence_source_sha256=evidence['evidence_source_sha256'],remaining_margin_below_3h_seconds=10800-revised['estimated_seconds'])
    values['B']=revised
    return values


def build():
    print('[PHASE][START] Retrieval-evaluable V2 labels/candidate freeze draft; no scoring',flush=True)
    require(not (ROOT/OUT/'draft.json').exists(),'V2 draft exists; rebuild/preflight only')
    f,prior,layer1,manifests,history,original,prior_record=prior_gate();amendment,prior_amendments=next_amendment(ROOT)
    revised=construct(original,layer1,amendment);plan=protocol.build_plan(prior,revised,amendment_id=amendment)
    verify_unchanged(plan,prior);estimates=estimate(revised,prior_record)
    for stage,r in estimates.items():
        print(f'[PHASE][GATE] Stage {stage} 3h = {r["gate"]}; estimated_s={r["estimated_seconds"]:.3f}',flush=True)
    require(all(r['gate']=='PASS' and r['estimated_seconds']<10800 for r in estimates.values()),'runtime budget failed; no scientific fallback')
    path=DERIVED/'standard_manifest.json';exclusive_json(ROOT/path,revised)
    manifests_binding={'standard':dict(path=path.as_posix(),sha256=file_sha(ROOT/path),manifest_digest=revised['global_manifest_digest']),
        'cold_drug':f['manifests']['cold_drug'],'cold_disease':f['manifests']['cold_disease']}
    plan_path=OUT/'execution_plan.json';exclusive_json(ROOT/plan_path,plan)
    record=dict(amendment_id=amendment,history_count=len(history),history_tail_sha256=list(history.values())[-1],
        previous_amendment_hashes={p.relative_to(ROOT).as_posix():file_sha(p) for _,p in prior_amendments},
        prior_freeze_path=V1_FREEZE.as_posix(),prior_freeze_sha256=V1_FREEZE_SHA,
        layer1_manifest_path=f['layer1_manifest_path'],layer1_manifest_sha256=f['layer1_manifest_sha256'],
        layer1_manifest_digest=PINS['A'],manifests=manifests_binding,plan_path=plan_path.as_posix(),
        plan_sha256=file_sha(ROOT/plan_path),execution_plan_digest=digest(plan),estimates=estimates,
        expected_total_forwards=plan['total_model_forwards'],execution_authorized=False,scientific_results_inspected=False)
    exclusive_json(ROOT/OUT/'draft.json',record)
    print('[PHASE][DONE] evaluability census='+str({k:revised['selection'][k] for k in ('N_all','N_zero_positive','N_retrieval_evaluable')}),flush=True)
    print('[PHASE][DONE] Layer2V2 counts='+str(revised['totals'])+'; digest='+revised['global_manifest_digest'],flush=True)


def load_draft():
    record=read_json(ROOT/OUT/'draft.json');manifests={}
    for split,b in record['manifests'].items():
        require(file_sha(ROOT/b['path'])==b['sha256'],'frozen manifest bytes changed: '+split)
        value=read_json(ROOT/b['path'])
        if split=='standard':protocol.validate_manifest(value,expected_digest=b['manifest_digest'])
        else:previous.validate_manifest(value,expected_digest=b['manifest_digest'])
        manifests[split]=value
    require(file_sha(ROOT/record['plan_path'])==record['plan_sha256'],'V2 plan file changed')
    plan=read_json(ROOT/record['plan_path']);protocol.validate_plan(plan,expected_digest=record['execution_plan_digest'])
    return record,manifests,plan


def rebuild():
    print('[PHASE][START] Separate-process V2 actual-source rebuild; no scoring',flush=True)
    f,prior,layer1,manifests,history,original,prior_record=prior_gate();record,current,plan=load_draft()
    rebuilt=construct(original,layer1,record['amendment_id'])
    require(rebuilt==current['standard'],'V2 cross-process actual-source rebuild differs')
    expected=protocol.build_plan(prior,rebuilt,amendment_id=record['amendment_id'])
    require(expected==plan and estimate(rebuilt,prior_record)==record['estimates'],'V2 plan/runtime count changed')
    verify_unchanged(plan,prior)
    exclusive_json(ROOT/OUT/'repeat_build_verification.json',dict(status='PASS',separate_process=True,
        v2_manifest_digest=rebuilt['global_manifest_digest'],execution_plan_digest=digest(plan),unchanged_A_C_D=True,
        original_full_eligibility_and_positive_membership_unchanged=True,scientific_results_inspected=False))
    print('[PHASE][GATE] V2_CROSS_PROCESS_REBUILD = PASS; A/C/D UNCHANGED',flush=True)


def freeze():
    print('[PHASE][START] Append-only retrieval-evaluability amendment freeze',flush=True)
    f,prior,layer1,manifests,history,original,prior_record=prior_gate();record,current,plan=load_draft()
    next_id,_=next_amendment(ROOT)
    require(next_id==record['amendment_id'] and len(history)==record['history_count'] and
        list(history.values())[-1]==record['history_tail_sha256'],'monotonic amendment position changed')
    for p,sha in record['previous_amendment_hashes'].items():require(file_sha(ROOT/p)==sha,'previous amendment changed')
    repeat=read_json(ROOT/OUT/'repeat_build_verification.json')
    require(repeat['status']=='PASS' and repeat['execution_plan_digest']==record['execution_plan_digest'],'actual-source repeat verification missing')
    verify_unchanged(plan,prior)
    require(estimate(current['standard'],prior_record)==record['estimates'] and all(r['gate']=='PASS' for r in record['estimates'].values()),'3-hour gates changed')
    suites=ET.parse(ROOT/OUT/'tests_final.xml').getroot().findall('testsuite')
    require(suites and all(int(s.get('failures','0'))+int(s.get('errors','0'))==0 for s in suites),'targeted/relevant tests failed')
    cases=[c for s in suites for c in s.findall('testcase')];counts={name:sum(c.get('classname')=='tests.'+name for c in cases)
        for name in ('test_phase842_evaluable_protocol','test_phase842_overnight_runtime','test_phase842_overnight_launcher')}
    require(counts['test_phase842_evaluable_protocol']>=30 and counts['test_phase842_overnight_runtime']>=10 and
        counts['test_phase842_overnight_launcher']>=5,'required evaluability/orchestrator/launcher tests absent')
    parser_evidence=read_json(ROOT/OUT/'powershell_parse.json')
    require(parser_evidence['status']=='PASS' and len(parser_evidence['files'])==3 and
        all(r['errors']==0 and file_sha(ROOT/r['path'])==r['sha256'] for r in parser_evidence['files']),'PowerShell parse gate failed/stale')
    source_sha={p:file_sha(ROOT/p) for p in SOURCES}
    now=datetime.now(timezone.utc).isoformat();manifest=current['standard']
    amendment_path=Path('outputs/phase8_4_2')/(next_id+'.json')
    payload=dict(amendment_id=next_id,amendment_type='PRE-EXECUTION / PRE-RESULTS RETRIEVAL-EVALUABILITY AMENDMENT',
        timestamp_utc=now,formal_run_id=FORMAL_RUN_ID,formal_test_count=1,old_u1024_state='INTERRUPTED / CLOSED',
        v1_scientific_scoring_executed=False,v1_output_root_absent=True,v1_zero_positive_disease='EFO_0009425',
        v1_zero_positive_count=0,reason='Zero-positive query cannot define supervised retrieval metrics; require eligible support positive_count>=1',
        basis='METRIC_EVALUABILITY / EVALUATION_DEFINABILITY REQUIREMENT ONLY; NOT PERFORMANCE FILTERING',
        selection_used_formal_performance=False,selection_used_validation=False,selection_used_top_k=False,
        selection_used_model_winner=False,selection_used_manual_disease_preference=False,
        scientific_results_inspected=False,scientific_metrics_printed=False,top_k_inspected=False,model_comparison_inspected=False,
        layer1_unchanged=True,cold_unchanged=True,training_model_checkpoint_changed=False,exact_arithmetic_changed=False,graph_batch_size=1,
        previous_amendment_hashes=record['previous_amendment_hashes'],previous_history_tail_sha256=record['history_tail_sha256'],
        v1_manifest_digest=V1_MANIFEST,v1_disposition='SUPERSEDED BEFORE SCIENTIFIC EXECUTION',v1_resume_into_v2=False,
        evaluability_census=manifest['selection'],layer2_protocol=manifest['protocol'],manifests=record['manifests'],
        unchanged_manifest_pins=PINS,source_data_identity=plan['data_binding'],checkpoint_binding_digest=plan['checkpoint_binding_digest'],
        indexed_representation_identity=plan['representation_sha256'],cold_policy_digest=plan['cold_policy_digest'],
        execution_plan_path=record['plan_path'],execution_plan_sha256=record['plan_sha256'],execution_plan_digest=record['execution_plan_digest'],
        runtime_estimates=record['estimates'],expected_total_forwards=plan['total_model_forwards'],source_sha256=source_sha,
        test_evidence=dict(targeted=counts,relevant_total=len(cases),failures=0,path=(OUT/'tests_final.xml').as_posix(),sha256=file_sha(ROOT/OUT/'tests_final.xml')),
        user_conditional_authorization='Current Phase8.4.2 Part4 explicitly authorizes one visible Limited InteractiveToken PowerShell and automatic A->B->C->D only after ALL gates PASS',
        execution_authorized=False,status='FROZEN; CONDITIONAL_LAUNCH_REQUIRES_PRE_LAUNCH_PASS')
    exclusive_json(ROOT/amendment_path,payload)
    event_path=Path('outputs/phase8_4_1/amendment_history')/f'{len(history)+1:04d}_RETRIEVAL_EVALUABILITY_V2_FROZEN.json'
    exclusive_json(ROOT/event_path,dict(amendment_id=next_id,formal_run_id=FORMAL_RUN_ID,formal_test_count=1,
        event='RETRIEVAL_EVALUABILITY_V2_FROZEN',timestamp_utc=now,previous_event_sha256=record['history_tail_sha256'],
        payload=dict(amendment_path=amendment_path.as_posix(),amendment_sha256=file_sha(ROOT/amendment_path),execution_plan_digest=record['execution_plan_digest'])))
    value=dict(format='EVIHGT_FINAL_EVALUABLE_FREEZE_V2',formal_run_id=FORMAL_RUN_ID,formal_test_count=1,
        amendment_id=next_id,amendment_path=amendment_path.as_posix(),amendment_sha256=file_sha(ROOT/amendment_path),
        history_event_path=event_path.as_posix(),history_event_sha256=file_sha(ROOT/event_path),history_hashes=verify_history(ROOT),
        prior_freeze_path=V1_FREEZE.as_posix(),prior_freeze_sha256=V1_FREEZE_SHA,
        layer1_manifest_path=f['layer1_manifest_path'],layer1_manifest_sha256=f['layer1_manifest_sha256'],layer1_manifest_digest=PINS['A'],
        manifests=record['manifests'],plan_path=record['plan_path'],plan_sha256=record['plan_sha256'],execution_plan_digest=record['execution_plan_digest'],
        source_sha256=source_sha,artifact_sha256={(OUT/n).as_posix():file_sha(ROOT/OUT/n)
            for n in ('draft.json','repeat_build_verification.json','tests_final.xml','powershell_parse.json')},
        execution_authorized=False,scientific_results_inspected=False)
    exclusive_json(ROOT/OUT/'freeze.json',value)
    print('[PHASE][GATE] RETRIEVAL_EVALUABILITY_AMENDMENT = PASS; '+next_id+'; event='+event_path.name,flush=True)


def preflight():
    print('[PHASE][START] Final V2 bounded read-only pre-launch; no scientific artifact access',flush=True)
    oldf,prior,layer1,oldmanifests,history,original,prior_record=prior_gate()
    f=read_json(ROOT/OUT/'freeze.json');record,manifests,plan=load_draft()
    require(f['formal_run_id']==FORMAL_RUN_ID and f['formal_test_count']==1 and f['execution_authorized'] is False,'Formal state changed')
    require(history==f['history_hashes'] and f['prior_freeze_sha256']==V1_FREEZE_SHA,'history/freeze chain changed')
    for p,s in [('amendment_path','amendment_sha256'),('history_event_path','history_event_sha256'),('plan_path','plan_sha256')]:
        require(file_sha(ROOT/f[p])==f[s],'V2 frozen commitment changed: '+p)
    for mapping in ('source_sha256','artifact_sha256'):
        for p,sha in f[mapping].items():require(file_sha(ROOT/p)==sha,'V2 frozen source/test/evidence changed: '+p)
    require(f['manifests']==record['manifests'] and f['execution_plan_digest']==record['execution_plan_digest'],'freeze/manifest plan mismatch')
    expected=protocol.build_plan(prior,manifests['standard'],amendment_id=f['amendment_id'])
    require(expected==plan and estimate(manifests['standard'],prior_record)==record['estimates'],'V2 plan/protocol/budget drift')
    verify_unchanged(plan,prior)
    for stage,pin in PINS.items():require(plan['manifest_bindings'][stage]==pin,'unchanged stage pin differs: '+stage)
    require(all(r['gate']=='PASS' and r['estimated_seconds']<10800 for r in record['estimates'].values()),'3-hour budget failed')
    f={**f,'freeze_sha256':file_sha(ROOT/OUT/'freeze.json')}
    print('[PHASE][GATE] HISTORY_BINDINGS_A_C_D_UNCHANGED_AND_V2_ISOLATION = PASS',flush=True)
    return f,plan,layer1,manifests


def authorization_payload(f,plan):
    return dict(authorized=True,action='FINAL_OVERNIGHT_EVALUABLE_FORMAL_EXECUTION',formal_run_id=FORMAL_RUN_ID,formal_test_count=1,
        amendment_id=f['amendment_id'],plan_sha256=f['plan_sha256'],plan_digest=f['execution_plan_digest'],
        execution_plan_digest=f['execution_plan_digest'],freeze_sha256=f['freeze_sha256'],manifest_bindings=plan['manifest_bindings'],
        checkpoint_binding_digest=plan['checkpoint_binding_digest'],source_binding_digest=digest(plan['data_binding']),
        units=[u['unit_id'] for u in plan['units']],execution_order=['A','B','C','D'],scientific_result_visibility=False,automatic_retry=False,
        issued_utc=datetime.now(timezone.utc).isoformat(),
        user_authorization_source='Human Phase8.4.2 Part4: FINAL FORMAL EXECUTION YES; one visible Limited PowerShell; automatic A->B->C->D YES; scientific visibility NO; all pre-launch gates required PASS')


def authorize():
    f,plan,layer1,manifests=preflight();record,_,_=load_draft()
    require(not (ROOT/plan['output_root']).exists(),'fresh V2 output already exists; no automatic repeat')
    report=dict(formal_test_count=1,formal_run_id=FORMAL_RUN_ID,amendment_id=f['amendment_id'],
        manifest_bindings=plan['manifest_bindings'],plan_sha256=f['plan_sha256'],execution_plan_digest=f['execution_plan_digest'],
        checkpoint_binding_digest=plan['checkpoint_binding_digest'],source_data_identity=plan['data_binding'],
        representation_sha256=plan['representation_sha256'],cold_policy_digest=plan['cold_policy_digest'],graph_batch_size=1,
        layer2_selection=manifests['standard']['selection'],expected_total_forwards=plan['total_model_forwards'],
        stages={s:dict(expected_forwards=sum(u['pairs'] for u in plan['units'] if u['layer']==s),
            pairs_per_model=next(u['pairs'] for u in plan['units'] if u['layer']==s),estimated_seconds=record['estimates'][s]['estimated_seconds'],
            runtime_gate=record['estimates'][s]['gate']) for s in ('A','B','C','D')},
        history_integrity='PASS',cross_protocol_isolation='PASS',orchestrator_tests='PASS',all_tests='PASS',gate='PASS')
    exclusive_json(ROOT/OUT/'pre_launch_report.json',report)
    exclusive_json(ROOT/OUT/'execution_authorization.json',authorization_payload(f,plan))
    for key in ('formal_test_count','formal_run_id','amendment_id','manifest_bindings','execution_plan_digest','stages',
        'expected_total_forwards','checkpoint_binding_digest','history_integrity','cross_protocol_isolation','orchestrator_tests','all_tests'):
        print('[PHASE][RUNNING] '+key+'='+str(report[key]),flush=True)
    print('[PHASE][GATE] FINAL_PRE_LAUNCH = PASS',flush=True)
