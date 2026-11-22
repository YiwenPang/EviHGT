"""One arm, exactly three cycles, with frozen pairing and full same-arm recovery."""
import ablation_artifacts as artifacts
import json
import os
import time
import traceback
import torch
import psutil

from evi_hgt.ranking_protocol.runner import numeric_backend
from evi_hgt.ranking_protocol.state import seed_streams,tensor_digest
from evi_hgt.ranking_protocol.production_state import _rng,training_state_digest
from evi_hgt_feature_baseline.model import build_arm,make_optimizer
from evi_hgt_feature_baseline.pretrained import PretrainedStore
from evi_hgt_feature_baseline_recovery.runner import Events,prepare
from .protocol import *
from .data import load_data,load_paired
from .training import execute_update
from .checkpoint import Store,capture,restore,verify_commit
from .diagnostics import diagnose
from .evaluation import evaluate,verify_result
from .safety import no_conflicts


def logged_updates(arm):
    p=OUT/arm/"updates.jsonl"
    return [json.loads(r) for r in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def verify_receipt(receipt,ident):
    if (receipt.get("status")!="COMPLETE_VERIFIED_DEVELOPMENT" or receipt["identity"]!=ident
        or receipt["logical_optimizer_updates"]!=TOTAL or set(receipt["evaluations"])!=set(map(str,MILESTONES))):
        raise ValueError("Incomplete LOSS_COMPARISON trajectory")
    for n,e in receipt["evaluations"].items(): verify_result(e,ident,int(n))
    commits=[verify_commit(p,ident) for p in sorted((OUT/ident["arm"]/"checkpoints").glob("*.COMMIT.json"))]
    if [c["step"] for c in commits]!=[n for n in range(1,TOTAL+1) if checkpoint_due(n)]:
        raise ValueError("Missing or additional checkpoint boundaries")
    rows=logged_updates(ident["arm"])
    latest={r["optimizer_step"]:r for r in rows}
    if sorted(latest)!=list(range(1,TOTAL+1)) or len(rows)!=receipt["physical_optimizer_updates"]:
        raise ValueError("Update journal count mismatch")
    if any(r["manifest_sha256"]!=ident["manifest_sha256"] or r["loss_id"]!=ident["loss_id"] or not r["finite"] for r in rows):
        raise ValueError("Update identity or finiteness mismatch")
    return receipt


def run_arm(arm,*,resume=False):
    if arm not in ARMS:raise PermissionError("Only the two LOSS_COMPARISON arms")
    no_conflicts();m=validate_manifest();ident=identity(m,arm)
    out=OUT/arm;out.mkdir(parents=True,exist_ok=True);ui=Events()
    if (out/"receipt.json").exists():
        if not resume: raise FileExistsError("Arm already complete")
        return verify_receipt(read(out/"receipt.json"),ident)
    if (out/"IN_FLIGHT_UPDATE.json").exists(): raise RuntimeError("Unresolved optimizer boundary; no automatic replay")
    if (out/"started.json").exists() and not resume: raise FileExistsError("Arm already started; explicit resume required")
    step=0;last_commit=None;start=time.monotonic();evaluations={}
    control=dict(clipped_updates=0,zero_gradient_steps=0,training_seconds=0.,query_preparation_seconds=0.,
                 forward_seconds=0.,backward_optimizer_seconds=0.,diagnostics={},evaluations={})
    attempt=str(time.time_ns())
    try:
        ui.stage("Preparing loss comparison inputs",arm+". Uses the first three sampling cycles of the R0F0 feature baseline.")
        numeric_backend();data,corpus=load_data()
        groups,records,stream,paired,schedule=load_paired(data)
        if (corpus["digest"]!=m["corpus_digest"] or paired["paired_stream_sha256"]!=m["paired_stream_sha256"]
            or paired["pairing_sha256"]!=m["pairing_sha256"]):
            raise ValueError("Registered data/sampling identity differs")
        index,assembler,index_seconds=prepare(data,corpus,ui)
        pre=PretrainedStore()
        if pre.metadata!=m["pretrained_identity"]:raise ValueError("Pretrained features changed")
        seed_streams(SEED);rng_before=training_state_digest(_rng())
        built=build_arm("D1_PRETRAINED_EARLY",pre)
        if training_state_digest(_rng())!=rng_before: raise ValueError("Construction advanced training RNG")
        initial=tensor_digest(built.model.state_dict())
        if initial!=m["initial_tensor_sha256"] or built.metadata()!=m["initial_metadata"]:
            raise ValueError("Fresh D1 tensor/model identity differs")
        built.model.cuda().train();optimizer=make_optimizer(built)
        if optimizer.state:raise ValueError("Fresh AdamW contains state")
        store=Store(out/"checkpoints",ident);latest=store.latest()
        if latest:
            if not resume: raise FileExistsError("Existing checkpoints require same-arm resume")
            state=store.load(latest);restored=restore(state,built.model,optimizer,ident,built.metadata())
            step=restored["step"];control=restored["control"];last_commit=verify_commit(latest,ident);del state
        elif resume and (out/"started.json").exists():
            raise RuntimeError("Started arm has no complete checkpoint; no automatic fresh restart")
        startup=dict(identity=ident,arm=arm,model="D1_PRETRAINED_EARLY",fresh=True,
            initial_tensor_sha256=initial,initial_metadata=built.metadata(),initial_rng_sha256=rng_before,
            optimizer_initial_state_entries=0,trained_weights_loaded=False,test_weights_loaded=False,
            sampling=paired,index_preparation_seconds=index_seconds,pid=os.getpid(),attempt=attempt,
            pre_training_preparation_seconds=time.monotonic()-start,
            queue=list(ARMS),mode="DEVELOPMENT")
        if not (out/"started.json").exists():write(out/"started.json",startup)
        else:write(out/("resume_"+attempt+".json"),dict(step=step,identity=ident,checkpoint=last_commit))
        previous={r["optimizer_step"]:r for r in logged_updates(arm)}

        def diagnostics_at(n):
            p=out/"diagnostics"/f"step_{n:08d}.json"
            weight=tensor_digest(built.model.state_dict())
            if p.exists():
                d=read(p)
                if d["identity"]!=ident or d["model_weight_sha256"]!=weight:
                    raise ValueError("Diagnostic checkpoint mismatch")
            else:
                ui.stage("Fixed training diagnostics",f"{arm} step={n}. Original 20 queries and 32 groups each from the global and conditional samplers.")
                d=diagnose(assembler,built,optimizer,groups,records,n,arm=arm,panels=m["diagnostic_panels"],ui=ui)
                d.update(identity=ident,model_weight_sha256=weight,
                         actual_training_clipped_updates=control["clipped_updates"],
                         actual_training_finite_zero_steps=control["zero_gradient_steps"])
                write(p,d)
            control["diagnostics"][str(n)]=dict(path=str(p),sha256=file_sha(p),seconds=d["seconds"])

        def milestone(n):
            validate_manifest();diagnostics_at(n)
            write(out/"status.json",dict(status="EVALUATION_PENDING",step=n,identity=ident),immutable=False)
            e=evaluate(data,assembler,built,optimizer,n,last_commit,ident,ui)
            evaluations[str(n)]=e;control["evaluations"][str(n)]=dict(seconds=e["seconds"])

        if step==0:diagnostics_at(0)
        elif step in MILESTONES:milestone(step)
        for n in MILESTONES:
            p=out/"evaluation"/f"step_{n:08d}"/"evaluation.json"
            if p.exists():evaluations[str(n)]=verify_result(read(p),ident,n)
        ui.training_start(arm,step,TOTAL,G)
        while step<TOTAL:
            if (ROOT/artifacts.PILOT_DEVICE_LOCK).exists():
                raise RuntimeError("Old G3 queue lock appeared")
            member=stream[step];record=records[step-G] if step>=G else None
            if step+1 in previous:
                old=previous[step+1]
                if old["training_member_sha256"]!=member["member_sha256"] or old["record_sha256"]!=member["record_sha256"]:
                    raise ValueError("Recovery sample identity mismatch")
            write(out/"IN_FLIGHT_UPDATE.json",dict(step=step+1,attempt=attempt,identity=ident,member=member),immutable=False)
            result=execute_update(assembler,built,optimizer,groups[step],record,arm=arm)
            step+=1
            for k,r in (("training_seconds","seconds"),("query_preparation_seconds","preparation_seconds"),
                        ("forward_seconds","forward_seconds"),("backward_optimizer_seconds","backward_optimizer_seconds")):
                control[k]+=result[r]
            control["clipped_updates"]+=int(result["clipped"])
            control["zero_gradient_steps"]+=int(result["zero_gradient_step"])
            row=dict(arm=arm,loss_id=LOSSES[arm],mode="DEVELOPMENT",pid=os.getpid(),attempt=attempt,
                optimizer_step=step,planned_optimizer_steps=TOTAL,epoch=(step-1)//G+1,
                stage="GLOBAL" if step<=G else "CONDITIONAL",replayed_existing_step=step in previous,
                training_member_sha256=member["member_sha256"],record_sha256=member["record_sha256"],
                manifest_sha256=file_sha(MANIFEST),sampling_sha256=m["paired_stream_sha256"],
                CUDA_allocated=torch.cuda.memory_allocated(),CUDA_reserved=torch.cuda.memory_reserved(),
                CPU_RSS=psutil.Process().memory_info().rss,**result)
            journal(out/"updates.jsonl",row);(out/"IN_FLIGHT_UPDATE.json").unlink();ui.update(row)
            if step==1:
                write(out/"FIRST_UPDATE.json",dict(**row,real_new_optimizer_update=True,test_only=False,
                    initial_tensor_sha256=initial,queued_following_arms=list(ARMS[1:]) if arm==ARMS[0] else [],
                    log_path=str(out/"updates.jsonl")))
            if checkpoint_due(step):
                validate_manifest()
                state=capture(built.model,optimizer,ident,step,control,built.metadata())
                last_commit=store.commit(step,state);del state;ui.checkpoint(step)
            if step in MILESTONES:
                milestone(step)
                if step<TOTAL:ui.training_start(arm,step,TOTAL,G)
        physical=len(logged_updates(arm))
        receipt=dict(status="COMPLETE_VERIFIED_DEVELOPMENT",identity=ident,arm=arm,
            logical_optimizer_updates=step,physical_optimizer_updates=physical,replayed_updates=physical-step,
            evaluations=evaluations,control=control,preparation_seconds=startup["pre_training_preparation_seconds"],
            worker_wall_seconds=time.monotonic()-start,last_checkpoint=last_commit,
            old_G3="HELD",outer="UNSCORED",new_final="SEALED")
        verify_receipt(receipt,ident);write(out/"receipt.json",receipt)
        write(out/"status.json",dict(status="COMPLETE_VERIFIED_DEVELOPMENT",step=step),immutable=False)
        ui.stage("Development run complete",arm+". Both scheduled evaluations are complete.")
        return receipt
    except BaseException as exc:
        detail=traceback.format_exc();(out/"traceback.log").write_text(detail,encoding="utf-8")
        write(out/("FAILED_"+attempt+".json"),dict(status="PARTIAL_FAILED",identity=ident,step=step,
            optimizer_boundary_unknown=(out/"IN_FLIGHT_UPDATE.json").exists(),error=repr(exc),
            wall_seconds=time.monotonic()-start,control=control,evaluations=evaluations))
        ui.failure("Loss comparison stopped: "+str(exc),str(out/"traceback.log"));raise
