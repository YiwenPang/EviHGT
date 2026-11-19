"""Freeze the implemented, accepted LOSS_COMPARISON contract; no new model or training."""

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
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import ablation_records as records
from evi_hgt_loss_comparison.protocol import *
from evi_hgt_loss_comparison.safety import no_conflicts,old_hold


def main():
    no_conflicts();hold=old_hold()
    if MANIFEST.exists():validate_manifest();return
    real=read(OUT/"TEST_ONLY/REAL_ACCEPTANCE.json")
    if real["status"]!="REAL_GPU_TEST_ONLY_ACCEPTANCE_PASS" or real["GPU_TEST_ONLY_optimizer_updates"]!=6:
        raise ValueError("Real TEST_ONLY acceptance incomplete")
    for p,h in real["source_sha256"].items():
        if file_sha(p)!=h:raise ValueError("GPU-accepted computation changed: "+p)
    suite=ET.parse(OUT/"TEST_ONLY/final_tests.xml")
    cases=suite.findall(".//testcase")
    if not cases or suite.findall(".//failure") or suite.findall(".//error") or suite.findall(".//skipped"):
        raise ValueError("Targeted tests failed, skipped or absent")
    if read(OUT/records.FEATURE_REVIEW)["status"]!="VERIFIED_READ_ONLY":raise ValueError("FEATURE_BASELINE review incomplete")
    unit_log=OUT/"TEST_ONLY/unit_optimizer_updates.jsonl"
    cpu=2+sum(json.loads(x)["cpu_optimizer_updates"] for x in unit_log.read_text(encoding="utf-8").splitlines())
    gpu=0;attempts=[]
    for p in (OUT/"TEST_ONLY").glob("real_*/STARTED.json"):
        log=p.parent/"updates.jsonl"
        count=len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0
        gpu+=count;attempts.append(dict(path=str(p.parent),recorded_GPU_optimizer_updates=count))
    tests=dict(status="TARGETED_ACCEPTANCE_PASS",unit_tests=len(cases),test_names=[c.attrib["name"] for c in cases],
        junit_path=str(OUT/"TEST_ONLY/final_tests.xml"),junit_sha256=file_sha(OUT/"TEST_ONLY/final_tests.xml"),
        actual_CPU_TEST_ONLY_optimizer_updates=cpu,actual_GPU_TEST_ONLY_optimizer_updates=gpu,
        total_TEST_ONLY_optimizer_updates=cpu+gpu,development_optimizer_updates_before_launch=0,
        real_acceptance=str(OUT/"TEST_ONLY/REAL_ACCEPTANCE.json"),real_acceptance_sha256=file_sha(OUT/"TEST_ONLY/REAL_ACCEPTANCE.json"),
        real_attempts=attempts,earlier_failures=[
          records.LOSS_INITIAL_TEST_NOTE,
          "One pytest default temporary-directory permission error; moved test artifacts to TEST_ONLY, no global permission change",
          "Corpus wrapper omitted selected corpus_root; identity gate stopped before GPU optimizer updates; regression test added and passed",
          "An acceptance assertion required bitwise equality across independent CUDA forwards; exact loss equality is now tested on identical real predictions. That earlier numerical difference remains UNKNOWN. No model/backend/optimizer change.",
          "Initial visible PowerShell 5.1 launch rejected UTF-8 without BOM. No worker or development update started. Original manifest, launch receipt and script bytes preserved under prelaunch_attempt1; BOM-only repair passes the actual 5.1 parser."],
        cpu_accounting_note="2 earlier CPU checkpoint-test updates preceded the durable unit journal; all subsequent invocations journal each update.",
        mathematical_tolerances=dict(gradcheck_dtype="float64",eps=1e-6,atol=1e-8,rtol=1e-5,
            old_gradient_FP64_reference_rtol=1e-5,old_gradient_atol=2**-149,group_atol=1e-7,group_rtol=1e-6),
        assertions=dict(real_FP32_BF16_dual_head_finite=True,old_objective_unchanged=True,
          initialization_and_private_rng_match=True,all_20568_step_members_match=True,conditional_13712_records_reproduced=True,
          original_graph_and_reverse_equivalent_mask_audits=True,complete_state_same_arm_restore=True,cross_loss_rejected=True,
          finite_zero_allowed=True,diagnostic_state_protected=True,outer_final_rejected=True,
          complete_eval_cache_no_forward=True,partial_attempts_not_merged=True,chinese_console=True),
        old_G3=hold,outer="UNSCORED",new_final="SEALED")
    write(OUT/"TARGETED_TEST_RESULTS.json",tests)
    original=read(OLD/records.FEATURE_MANIFEST_NAME)
    historical_start=read(RECOVERY/"D1_PRETRAINED_EARLY/started.json")
    if real["initial_metadata"]!=historical_start["metadata"]:
        raise ValueError("Fresh D1 architecture/input identity differs from actual FEATURE_BASELINE D1")
    schedule=read(SCHEDULE)
    paths=[SCHEDULE,PAIRING,Path(schedule["groups_path"]),Path(real["corpus_manifest"]),
        OLD/records.FEATURE_MANIFEST_NAME,RECOVERY/"RECOVERY_EXECUTION_MANIFEST.json",
        RECOVERY/"D1_PRETRAINED_EARLY/started.json",
        OLD/"OLD_G3_HOLD_DECISION.json",OUT/records.FEATURE_REVIEW,OUT/"TEST_ONLY/REAL_ACCEPTANCE.json",
        OUT/"TARGETED_TEST_RESULTS.json",OUT/"LOSS_CONTRACT.md",OUT/records.LOSS_PLAN]
    artifacts={str(p.resolve()):file_sha(p) for p in paths}
    artifacts.update(real["pretrained_identity"]["source_artifact_sha256"])
    sources=source_hashes();sources.update(real["pretrained_identity"]["generator_source_sha256"])
    count=sum(1 for i in range(1,TOTAL+1) if checkpoint_due(i))
    required=math.ceil(2*count*max(r["checkpoint"]["checkpoint_size"] for r in real["arms"])*1.25)+10*(1<<30)
    if shutil.disk_usage(OUT).free<required:raise OSError("Insufficient storage for full-state checkpoints")
    m=dict(schema=records.LOSS_SCHEMA,mode="DEVELOPMENT",arms=list(ARMS),losses=LOSSES,
        updates_per_arm=TOTAL,total_logical_updates=TOTAL*2,milestones=list(MILESTONES),
        evaluation_domains=list(DOMAINS),repeat=0,fold=0,seed=SEED,G=G,G3=False,G4=False,G5=False,
        old_G3="HELD",outer="UNSCORED",new_final="SEALED",old_Formal_access=1,new_Final_access=0,
        source_sha256=sources,artifact_sha256=artifacts,
        data_bindings=real["data_bindings"],corpus_digest=real["corpus_digest"],schedule_digest=real["schedule_digest"],
        **{k:real["paired_sampling"][k] for k in ("paired_stream_sha256","pairing_sha256","schedule_sha256","conditional_prefix_sha256")},
        sampling_protocol=real["paired_sampling"],diagnostic_panels=real["diagnostic_panels"],
        initial_tensor_sha256=real["initial_tensor_sha256"],initial_metadata=real["initial_metadata"],
        model_parameter_count=real["model_parameter_count"],model_schema=real["model_schema"],
        pretrained_identity=real["pretrained_identity"],optimizer=original["optimizer"],graph=original["graph"],paths=original["paths"],
        numeric_flags=dict(threads=4,matmul_allow_tf32=False,cudnn_allow_tf32=True,deterministic_algorithms=False,
                           forward="BF16",loss="FP32",tau=1.,CUDA_bitwise_claim=False),
        primary_comparison="3G E1-E0 support disease-macro ROC-AUC",
        diagnostics=dict(steps=[0,10284,20568],low_derivative=1e-6,cosine_low_norm=1e-12,reduction="FP64"),
        checkpoints=dict(every=1024,additional=list(MILESTONES),per_arm=count,complete_state=True),
        required_free_bytes=required,execution=dict(GPU_workers=1,order=list(ARMS),
            automatic_next_phase=False,automatic_training_retry=False,score_conditioned_stopping=False),
        final_markdown_report=str(OUT/records.LOSS_MARKDOWN),final_json_report=str(OUT/records.LOSS_JSON))
    write(MANIFEST,m);validate_manifest()
    print(f"LOSS_COMPARISON_PRELAUNCH_REGISTERED；{len(cases)} 项针对性测试通过；GPU TEST_ONLY={gpu}；DEVELOPMENT=0",flush=True)
