"""Freeze only after all bounded gates pass; no training/evaluation performed."""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from historical_run import run_original
    raise SystemExit(run_original(Path(__file__)))

from pathlib import Path
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import ablation_records as records
from evi_hgt_evidence_ablation.protocol import *
from evi_hgt_evidence_ablation.safety import no_conflicts,old_hold
from evi_hgt_paired_losses.protocol import MANIFEST as OLD_MANIFEST,validate_manifest as old_manifest


def main():
    no_conflicts();hold=old_hold();old=old_manifest()
    if MANIFEST.exists():validate_manifest();return
    real=read(OUT/'TEST_ONLY/REAL_ACCEPTANCE.json');console=read(OUT/'TEST_ONLY/CONSOLE_CHAIN_ACCEPTANCE.json')
    if real['status']!='REAL_GPU_TEST_ONLY_ACCEPTANCE_PASS' or real['GPU_TEST_ONLY_optimizer_updates']!=16 or real['evaluation_forwards']!=0:
        raise ValueError('Two-cell real GPU acceptance incomplete')
    if console['status']!='ACTUAL_POWERSHELL_RELAY_MULTILINE_PASS':raise ValueError('Actual console chain not accepted')
    junit=OUT/'TEST_ONLY/final_tests.xml';tree=ET.parse(junit);cases=tree.findall('.//testcase')
    if not cases or any(tree.findall('.//'+k) for k in ('failure','error','skipped')):raise ValueError('Targeted tests absent/fail/skipped')
    ref=read(OUT/'REFERENCE_REUSE_CHECK.json')
    for p,h in ref['read_only_sha256'].items():
        if file_sha(p)!=h:raise ValueError('Historical reference bytes changed: '+p)
    pending=OUT/'TEST_ONLY/REFERENCE_REUSE_CHECK.initial.json'
    if not pending.exists():write(pending,ref)
    for cell in CELLS:
        rc=real['cells'][cell];cm=old['cells'][cell]
        if (rc['initial_tensor_sha256']!=ref['cells'][cell]['initial_tensor_sha256'] or rc['initial_rng_sha256']!=ref['cells'][cell]['initial_rng_sha256']
            or rc['conditional_plan_digest']!=ref['cells'][cell]['conditional_plan_digest']):raise ValueError('Actual pairing/ref start differs')
        ref['cells'][cell]['runtime_equivalence_gate']='PASS'
        ref['cells'][cell]['runtime_acceptance']=str(OUT/'TEST_ONLY/REAL_ACCEPTANCE.json')
        ref['cells'][cell]['runtime_max_error']=rc['off_wrapper_max_absolute_error']
    ref['status']='REFERENCE_REUSE_ELIGIBLE_BOTH_CELLS';write(OUT/'REFERENCE_REUSE_CHECK.json',ref,immutable=False)
    tests=dict(status='TARGETED_ACCEPTANCE_PASS',unit_tests=len(cases),test_names=[c.attrib['name'] for c in cases],
        junit_path=str(junit),junit_sha256=file_sha(junit),development_updates_before_launch=0,evaluation_forwards=0,
        actual_CPU_TEST_ONLY_optimizer_updates=sum(json.loads(x)['CPU_updates'] for x in (OUT/'TEST_ONLY/unit_updates.jsonl').read_text('utf-8').splitlines()),
        actual_GPU_TEST_ONLY_optimizer_updates=sum(json.loads(x).get('GPU_updates',0) for p in (OUT/'TEST_ONLY').glob('real_*/updates.jsonl') for x in p.read_text('utf-8').splitlines()),
        real_acceptance=str(OUT/'TEST_ONLY/REAL_ACCEPTANCE.json'),console_chain=str(OUT/'TEST_ONLY/CONSOLE_CHAIN_ACCEPTANCE.json'),
        diagnostic_counts_note='Counts include failed TEST_ONLY attempts and repeated CPU verification; never enter development start.',
        failures_retained=['Initial RED missing modules; tuple/list flattening; package-relative private module naming; raw path trailing-slash syntax.',
            'Threaded CPU reduction was not strictly deterministic; strict fixture uses single thread only. Lazy CUDA initialization was settled before RNG snapshots. Production flags unchanged.',
            'First actual GPU gate blocked FlightSimulator and unknown desktop contexts before updates. User closed game; exact installed desktop paths separately verified.',
            'Initial console inspector used inherited redirected standard handle; read-only CONOUT$ handle corrected. CUA does not expose terminal windows; actual visible Win32 console buffer used.'],
        old_G3=hold,outer='UNSCORED',new_final='SEALED')
    write(OUT/'TARGETED_TEST_RESULTS.json',tests)
    per={c:sum(checkpoint_due(c,n) for n in range(1,cell_spec(c)['total']+1)) for c in CELLS}
    sizes={c:max([real['cells'][c]['arms'][a]['checkpoint']['checkpoint_size'] for a in ARMS]+
                 [read(HISTORY/c/REFERENCE_ARM/'checkpoints'/f'step_{cell_spec(c)["total"]:08d}.COMMIT.json')['checkpoint_size']]) for c in CELLS}
    estimated=sum(2*per[c]*sizes[c] for c in CELLS);required=math.ceil(estimated*1.30)+4*(1<<30)
    if shutil.disk_usage(OUT).free<required:raise OSError('Four actual EVIDENCE_ABLATION trajectories disk capacity insufficient')
    artifacts={**old['artifact_sha256'],**ref['read_only_sha256']}
    for path in (OUT/records.EVIDENCE_PLAN,OUT/'REFERENCE_REUSE_CHECK.json',OUT/'ABLATION_AND_CONSOLE_ACCEPTANCE.md',
                 OUT/'TARGETED_TEST_RESULTS.json',OUT/'TEST_ONLY/REAL_ACCEPTANCE.json',OUT/'TEST_ONLY/CONSOLE_CHAIN_ACCEPTANCE.json',
                 junit,ROOT/records.EVIDENCE_CONSOLE_RULES,OLD_MANIFEST):artifacts[str(path.resolve())]=file_sha(path)
    m=dict(schema=records.EVIDENCE_SCHEMA,mode='DEVELOPMENT',authorization=records.EVIDENCE_AUTHORIZATION,
        cells={c:old['cells'][c] for c in CELLS},cells_order=list(CELLS),queue=[list(q) for q in QUEUE],arms=list(ARMS),losses=LOSSES,
        total_logical_updates=82482,double_channel_evaluation_points=8,reference_models_reused=2,
        reference_arm=REFERENCE_ARM,reference_manifest_sha256=file_sha(OLD_MANIFEST),source_sha256=sources(),artifact_sha256=artifacts,
        pretrained_identity=old['pretrained_identity'],optimizer=old['optimizer'],numeric_flags=old['numeric_flags'],graph=old['graph'],paths=old['paths'],
        checkpoint_counts=per,checkpoint_size_basis=sizes,estimated_checkpoint_bytes=estimated,required_free_bytes=required,
        disk_free_at_registration=shutil.disk_usage(OUT).free,conditional_source=records.REFERENCE_PLAN_SOURCE,
        primary_endpoint='Per-cell3G REF_FULL_MINUS_ABLATION; all heads/metrics/milestones retained',
        old_G3='HELD',outer='UNSCORED',new_final='SEALED',G4=False,G5=False,old_Formal_access=1,new_Final_access=0,
        execution=dict(GPU_workers=1,automatic_training_retry=False,automatic_next_phase=False),
        final_markdown_report=str(OUT/records.EVIDENCE_MARKDOWN),final_json_report=str(OUT/records.EVIDENCE_JSON))
    write(MANIFEST,m);validate_manifest()
    print(f'EVIDENCE_ABLATION_PRELAUNCH_ACCEPTED: {len(cases)} targeted tests. Reserved {required/2**30:.2f} GiB.',flush=True)
