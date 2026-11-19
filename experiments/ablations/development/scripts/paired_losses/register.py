"""Freeze accepted PAIRED_LOSSES sources/inputs, without evaluating any real panels."""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from historical_run import run_original
    raise SystemExit(run_original(Path(__file__)))

from pathlib import Path
import math
import shutil
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import ablation_records as records
from ablation_protocol_ids import PAIRED_NAMESPACE
from evi_hgt_paired_losses.protocol import *
from evi_hgt_paired_losses.safety import no_conflicts,old_hold


def main():
    no_conflicts();hold=old_hold()
    if MANIFEST.exists():validate_manifest();return
    real=read(OUT/'TEST_ONLY/REAL_ACCEPTANCE.json')
    if real['status']!='REAL_GPU_TEST_ONLY_ACCEPTANCE_PASS' or real['GPU_TEST_ONLY_optimizer_updates']!=12 or real['evaluation_forwards']!=0:
        raise ValueError('Two-cell real acceptance incomplete')
    for p,h in real['source_sha256'].items():
        if file_sha(p)!=h:raise ValueError('GPU-accepted implementation changed: '+p)
    # Revalidate final sampler wiring after moving development-derived records
    # to data/cache. No graph index, network forward or optimizer is constructed.
    from evi_hgt.ranking_protocol.real_data import RealInputs
    from evi_hgt_paired_losses.data import load_data
    from evi_hgt_paired_losses.sampling import freeze_schedule,conditional_plan
    shared=RealInputs();sampling_checks={}
    for cell in CELLS:
        data,corpus=load_data(cell,shared);groups,stream,schedule,cache=freeze_schedule(data,corpus)
        root=Path(real['test_directory'])/cell/'sampling';prior=read(root/'plan.json')
        plan=conditional_plan(data,groups,schedule,cell=cell,test_origin=prior['origin_sha256'],test_root=root)
        if plan!=prior or schedule['digest']!=real['cells'][cell]['schedule_digest']:raise ValueError('Final real-data sampling differs')
        sampling_checks[cell]=dict(schedule_digest=schedule['digest'],conditional_digest=plan['digest'],
            members=len(stream),conditional_records=len(plan['legacy']['records']),final_source_revalidated=True,
            development_cache_root=str(cache/'conditional_v1'))
    del shared,data,groups,stream,plan
    junit=OUT/'TEST_ONLY/final_tests.xml';suite=ET.parse(junit);cases=suite.findall('.//testcase')
    if not cases or any(suite.findall('.//'+n) for n in ('failure','error','skipped')):raise ValueError('Targeted tests absent/failed/skipped')
    attempts=[];gpu=0
    for p in (OUT/'TEST_ONLY').glob('real_*/STARTED.json'):
        log=p.parent/'updates.jsonl';n=len(log.read_text(encoding='utf-8').splitlines()) if log.exists() else 0;gpu+=n
        attempts.append(dict(path=str(p.parent),GPU_updates=n,status='FAILED' if (p.parent/'FAILED.json').exists() else 'COMPLETE' if (p.parent/'COMPLETE.json').exists() else 'INCOMPLETE'))
    cpu=sum(json.loads(x)['CPU_updates'] for x in (OUT/'TEST_ONLY/unit_updates.jsonl').read_text(encoding='utf-8').splitlines())
    tests=dict(status='TARGETED_ACCEPTANCE_PASS',unit_tests=len(cases),test_names=[c.attrib['name'] for c in cases],
        junit_path=str(junit),junit_sha256=file_sha(junit),real_acceptance=str(OUT/'TEST_ONLY/REAL_ACCEPTANCE.json'),
        real_acceptance_sha256=file_sha(OUT/'TEST_ONLY/REAL_ACCEPTANCE.json'),actual_CPU_TEST_ONLY_optimizer_updates=cpu,
        actual_GPU_TEST_ONLY_optimizer_updates=gpu,development_updates_before_launch=0,evaluation_forwards=0,attempts=attempts,
        earlier_failures=[
            records.PAIRED_INITIAL_TEST_NOTE,
            'First GPU inventory rejected Realtek RtkUWP desktop audio UI as unknown. Actual image path established it as desktop graphics; no test or development update.',
            'First real graph attempt used nonexistent tensor batch .facts attribute. Corrected to existing canonical path_source facts; regression rejects query/reverse/equivalent/protected IDs. Stopped before optimizer update.',
        ],
        final_sampling_revalidation=sampling_checks,
        assertions=dict(two_frozen_cells_members_seed_G_budget=True,all_three_P_cycles_complete=True,paired_initial_tensor_schema_rng=True,
            actual_training_graphs_paired=True,FP32_BF16_dual_head_finite=True,**{records.REFERENCE_LOSS_CHECK_FIELD: True},
            legal_U_and_protected_exclusion=True,conditional_plan_deterministic_shared=True,
            synthetic_origin_never_development=True,same_arm_full_state_restore=True,cross_cell_loss_refused_before_mutation=True,
            evaluation_cache_cell_head_checkpoint_bound=True,diagnostics_state_RNG_protected=True,outer_final_denied=True,
            actual_PowerShell51_BOM_parse=True,finite_zero_AdamW_executes=True),old_G3=hold,outer='UNSCORED',new_final='SEALED')
    write(OUT/'TARGETED_TEST_RESULTS.json',tests)
    history_paths=[ROOT/'outputs/phase85_part3b1/PART3B1_REPORT.md',ROOT/'outputs/phase9/phase92b/REAL_DATA_PREFLIGHT_REPORT.md']
    history={
        'R0F1':dict(verified_uses=['Phase85 Part3B1: training and inner development selection completed; seed342565966, endpoint10270, original global+conditional recipe.',
                                   'Phase92b: bounded real training TEST_ONLY smoke/mechanism/conditional tests; no outer scoring.'],
                     source_reports=[str(p) for p in history_paths],old_G3_completed_here=0,
                     complete_lifetime_development_inventory='NOT_YET_VERIFIED; no Phase1–8 re-audit performed'),
        'R1F0':dict(verified_uses=['Phase92b: bounded real TEST_ONLY smoke, same-arm recovery, actual driver entries, mechanism and shared sampling checks.'],
                     source_reports=[str(history_paths[1])],old_G3_completed_here=0,
                     no_prior_long_run_claim='NOT_ASSERTED',complete_lifetime_development_inventory='NOT_YET_VERIFIED; no Phase1–8 re-audit performed'),
        'pool_overlap':'Same original development pool; role overlap across folds/repeats, not independent new datasets.',
    }
    paths=[OUT/records.PAIRED_PLAN,OUT/'TARGETED_TEST_RESULTS.json',OUT/'TEST_ONLY/REAL_ACCEPTANCE.json',junit,
        ROOT/'AGENTS.md',ROOT/records.LOSS_OUTPUT/records.LOSS_PLAN,ROOT/records.LOSS_OUTPUT/'LOSS_CONTRACT.md',
        ROOT/records.LOSS_OUTPUT/'TARGETED_TEST_RESULTS.json',ROOT/records.LOSS_OUTPUT/records.LOSS_JSON,ROOT/records.LOSS_OUTPUT/'results-audit.md',
        ROOT/'outputs/phase9/phase92/G1_FROZEN_SCIENCE_PROTOCOL.md',ROOT/'outputs/phase85_part3b1/cv_15_cell_static_census.json',
        ROOT/'outputs/phase85_part3b1/COMMON_DATA_CONTRACT.json',ROOT/records.FEATURE_OUTPUT/'OLD_G3_HOLD_DECISION.json',*history_paths]
    artifacts={str(p.resolve()):file_sha(p) for p in paths};cells={}
    census=read(ROOT/'outputs/phase85_part3b1/cv_15_cell_static_census.json')
    for cell,c in real['cells'].items():
        frozen=frozen_cell(cell)
        static=next(x for x in census['cells'] if (x['repeat_index'],x['fold_index'])==(c['spec']['repeat'],c['spec']['fold']))
        if static['budget']!=frozen['budget'] or static['budget']['G_rf']!=c['spec']['G'] or static['budget']['B_rf']!=c['spec']['milestones'][0]:
            raise ValueError('Actual frozen census budget disagrees; no member/seed adjustment allowed')
        if c['spec']!=cell_spec(cell):raise ValueError('Accepted cell spec differs')
        cells[cell]={k:v for k,v in c.items() if k not in ('arms','graph_inputs_sha256')}
        artifacts.update(c['graph_inputs_sha256'])
        for p in (Path(c['corpus_manifest']),Path(c['schedule_path'])):artifacts[str(p.resolve())]=file_sha(p)
    sources=source_hashes();sources.update(real['pretrained_identity']['generator_source_sha256'])
    artifacts.update(real['pretrained_identity']['source_artifact_sha256'])
    per={cell:sum(checkpoint_due(cell,n) for n in range(1,cell_spec(cell)['total']+1)) for cell in CELLS}
    sizes={cell:max(a['checkpoint']['checkpoint_size'] for a in c['arms']) for cell,c in real['cells'].items()}
    estimated=sum(2*per[cell]*sizes[cell] for cell in CELLS)
    required=math.ceil(estimated*1.30)+4*(1<<30)
    if shutil.disk_usage(OUT).free<required:raise OSError('Insufficient disk for the four PAIRED_LOSSES trajectories')
    old=read(ROOT/records.LOSS_OUTPUT/records.LOSS_MANIFEST)
    m=dict(schema=records.PAIRED_SCHEMA,mode='DEVELOPMENT',authorization=records.PAIRED_AUTHORIZATION,
        cells_order=list(CELLS),cells=cells,queue=[list(q) for q in QUEUE],arms=list(ARMS),losses=LOSSES,total_logical_updates=82482,
        double_channel_evaluation_points=8,evaluation_domains=list(DOMAINS),old_G3='HELD',outer='UNSCORED',new_final='SEALED',
        G4=False,G5=False,old_Formal_access=1,new_Final_access=0,source_sha256=sources,artifact_sha256=artifacts,
        pretrained_identity=real['pretrained_identity'],optimizer=old['optimizer'],graph=old['graph'],paths=old['paths'],numeric_flags=old['numeric_flags'],
        historical_development_use=history,primary_endpoint='Per-cell 3G support disease-macro AUC E1−E0; AP key auxiliary',
        conditional_source='Each cell genuine E0 G complete COMMIT SHA; private uniform complement; no predictions/weights/optimizer/validation for sampling',
        conditional_namespace=PAIRED_NAMESPACE,record_order='cycle2 then cycle3, original intragroup sequence; full P coverage each cycle',
        diagnostic_panels=records.PAIRED_PANEL_NOTE,
        checkpoints=dict(every=1024,also='G and fixed1.5G/3G',per_arm=per,complete_state=True,estimated_bytes=estimated,test_checkpoint_bytes=sizes),
        required_free_bytes=required,disk_free_at_registration=shutil.disk_usage(OUT).free,
        execution=dict(GPU_workers=1,automatic_training_retry=False,automatic_next_phase=False,score_conditioned_stopping=False),
        final_markdown_report=str(OUT/records.PAIRED_MARKDOWN),final_json_report=str(OUT/records.PAIRED_JSON))
    write(MANIFEST,m);validate_manifest()
    print(f'PAIRED_LOSSES_PRELAUNCH_REGISTERED；{len(cases)}项针对性测试；GPU TEST_ONLY={gpu}；开发=0；预留磁盘={required/(1<<30):.2f}GiB',flush=True)
