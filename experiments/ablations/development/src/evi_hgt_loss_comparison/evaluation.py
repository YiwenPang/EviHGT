"""Two frozen panels, bound complete caches, and separate immutable retry attempts."""
import math
from pathlib import Path
import time

import torch
from evi_hgt.ranking_protocol.state import tensor_digest
from evi_hgt.training.supervision import make_unlabeled_query_supervision
from evi_hgt_feature_baseline.evaluation import metrics
from .protocol import OUT, DOMAINS, check_evaluation, read, write, journal, digest, file_sha, validate_manifest, identity
from .checkpoint import Store, verify_commit
from .diagnostics import isolated


def validate_rows(rows,members):
    if len(rows)!=len(members): raise ValueError("Incomplete panel membership")
    for row,member in zip(rows,members,strict=True):
        if {k:v for k,v in row.items() if k not in ("support_logit","unsuitability_logit")}!=member:
            raise ValueError("Panel candidate/order differs")
        if not all(math.isfinite(row[k]) for k in ("support_logit","unsuitability_logit")):
            raise FloatingPointError("Nonfinite evaluation logit")


def cached_panel(root,domain,bindings,members,predict,ui=None):
    check_evaluation(bindings["step"],domain)
    root=Path(root).resolve()
    if not root.is_relative_to(OUT.resolve()): raise PermissionError("LOSS_COMPARISON panel namespace")
    complete=root/(domain+".COMPLETE.json")
    member_hash=digest(members)
    if complete.exists():
        c=read(complete)
        if c["bindings"]!=bindings or c["members_sha256"]!=member_hash or c["status"]!="COMPLETE":
            raise ValueError("Cached panel binding changed")
        p=Path(c["rows_path"])
        if not p.resolve().is_relative_to(root) or file_sha(p)!=c["rows_sha256"]:
            raise ValueError("Cached panel rows changed")
        rows=read(p); validate_rows(rows,members)
        return rows
    attempts=sorted(root.glob(domain+".attempt_*"))
    for old in attempts:
        s=read(old/"STARTED.json")
        if s["bindings"]!=bindings or s["members_sha256"]!=member_hash:
            raise ValueError("Partial attempt belongs to another checkpoint or candidate panel")
    attempt=root/f"{domain}.attempt_{len(attempts)+1:03d}"
    write(attempt/"STARTED.json",dict(status="STARTED",bindings=bindings,members_sha256=member_hash,
          prior_attempts=[str(p) for p in attempts],policy="FULL_SAME_CHECKPOINT_REEVALUATION_NO_PARTIAL_ROW_MERGE"))
    rows=[]; start=time.monotonic()
    try:
        for i,member in enumerate(members):
            journal(attempt/"forwards.jsonl",dict(event="BEGIN",index=i))
            t=time.monotonic(); row=predict(member)
            validate_rows([row],[member]); rows.append(row)
            journal(attempt/"forwards.jsonl",dict(event="DONE",index=i,seconds=time.monotonic()-t))
            if ui is not None: ui.evaluation_update(i+1)
        rows_path=attempt/"ROWS.json"
        write(rows_path,rows)
        write(complete,dict(status="COMPLETE",bindings=bindings,members_sha256=member_hash,
            rows_path=str(rows_path),rows_sha256=file_sha(rows_path),count=len(rows),seconds=time.monotonic()-start,
            accepted_attempt=str(attempt),prior_attempts=[str(p) for p in attempts]))
        return rows
    except BaseException as exc:
        write(attempt/"FAILED.json",dict(status="PARTIAL_FAILED",completed_forwards=len(rows),
              seconds=time.monotonic()-start,error=repr(exc)))
        raise


def panel_cost(root):
    completed=begun=0;seconds=0.; attempts=0; unknown=0
    for p in Path(root).glob("*.attempt_*"):
        attempts+=1; events=[]
        if (p/"forwards.jsonl").exists():
            import json
            events=[json.loads(x) for x in (p/"forwards.jsonl").read_text(encoding="utf-8").splitlines()]
        a=sum(r["event"]=="BEGIN" for r in events); b=sum(r["event"]=="DONE" for r in events)
        begun+=a;completed+=b;unknown+=a-b
        seconds+=sum(r.get("seconds",0.) for r in events if r["event"]=="DONE")
    return dict(attempts=attempts,recorded_completed_forwards=completed,begun_forwards=begun,
                in_flight_forward_outcomes_unknown=unknown,recorded_forward_seconds=seconds)


def metric_pair(support,aux):
    contra=[dict(r,support_logit=r["unsuitability_logit"],role="P") for r in aux]
    contra += [dict(r,support_logit=r["unsuitability_logit"]) for r in support if r["role"]=="U"]
    return dict(support=metrics(support),contra=metrics(contra))


def verify_result(result,ident,step):
    check_evaluation(step,"validation")
    root=OUT/ident["arm"]/"evaluation"/f"step_{step:08d}"
    if (result.get("status")!="COMPLETE_VERIFIED_DEVELOPMENT_EVALUATION" or result.get("arm")!=ident["arm"]
        or result.get("step")!=step or result["bindings"]["identity"]!=ident):
        raise ValueError("Evaluation status or identity mismatch")
    c=verify_commit(result["checkpoint_commit_path"],ident)
    if c["step"]!=step or c["checkpoint_sha256"]!=result["bindings"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint binding mismatch")
    rows={}
    for domain,key in zip(DOMAINS,("validation_members","auxiliary_members"),strict=True):
        complete=read(root/(domain+".COMPLETE.json")); p=Path(complete["rows_path"])
        if (complete["bindings"]!=result["bindings"] or complete["members_sha256"]!=result["bindings"][key]
            or complete["status"]!="COMPLETE" or not p.resolve().is_relative_to(root.resolve())
            or file_sha(p)!=complete["rows_sha256"]):
            raise ValueError("Incomplete or mismatched panel cache")
        rows[domain]=read(p)
        members=[{k:v for k,v in r.items() if k not in ("support_logit","unsuitability_logit")} for r in rows[domain]]
        if digest(members)!=complete["members_sha256"] or len(members)!=complete["count"]:
            raise ValueError("Panel member digest/count mismatch")
        validate_rows(rows[domain],members)
    if metric_pair(rows[DOMAINS[0]],rows[DOMAINS[1]])!=result["metrics"]:
        raise ValueError("Saved raw-score metrics do not reproduce")
    return result


def evaluate(data,assembler,built,optimizer,step,commit,ident,ui):
    check_evaluation(step,"validation")
    m=validate_manifest()
    if ident!=identity(m,ident["arm"]) or commit["identity"]!=ident or commit["step"]!=step:
        raise PermissionError("Unregistered development checkpoint")
    checkpoint_root=OUT/ident["arm"]/"checkpoints"
    state=Store(checkpoint_root,ident).load(commit["checkpoint_commit_path"])
    weights=tensor_digest(built.model.state_dict())
    if tensor_digest(state["model"])!=weights: raise ValueError("Model differs from checkpoint")
    del state
    if dict(data.bindings,corpus_digest=m["corpus_digest"],schedule_digest=m["schedule_digest"])!=ident["data_bindings"]:
        raise ValueError("Evaluation data differs")
    root=OUT/ident["arm"]/"evaluation"/f"step_{step:08d}"
    bindings=dict(identity=ident,step=step,checkpoint_sha256=commit["checkpoint_sha256"],weight_sha256=weights,
        validation_members=digest(data.cell[DOMAINS[0]]),auxiliary_members=digest(data.cell[DOMAINS[1]]),
        mask_sha256=data.bindings["fold_mask_digest"])
    if (root/"evaluation.json").exists():
        result=read(root/"evaluation.json")
        if result["bindings"]!=bindings: raise ValueError("Evaluation cache identity differs")
        return verify_result(result,ident,step)
    start=time.monotonic(); rows={}
    with isolated(built,optimizer,assembler),data.evaluation_scope(step):
        for domain in DOMAINS:
            ui.evaluation_start(ident["arm"],step,domain,len(data.cell[domain]))
            def predict(row):
                q=make_unlabeled_query_supervision(row["drug_id"],row["disease_id"])
                g=assembler.assemble(q,domain=domain,device="cuda")
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                    p=built.forward(g.batch.model_batch,path_batch=g.batch.path_batch)
                return dict(row,support_logit=float(p.support_logit[0]),unsuitability_logit=float(p.unsuitability_logit[0]))
            rows[domain]=cached_panel(root,domain,bindings,data.cell[domain],predict,ui)
    result=dict(status="COMPLETE_VERIFIED_DEVELOPMENT_EVALUATION",arm=ident["arm"],step=step,bindings=bindings,
        metrics=metric_pair(rows[DOMAINS[0]],rows[DOMAINS[1]]),checkpoint_commit_path=commit["checkpoint_commit_path"],
        seconds=sum(read(root/(d+".COMPLETE.json"))["seconds"] for d in DOMAINS),
        current_invocation_seconds=time.monotonic()-start,costs=panel_cost(root),
        outer="UNSCORED",new_final="SEALED")
    write(root/"evaluation.json",result)
    verify_result(result,ident,step); ui.evaluation_result(ident["arm"],step,result["metrics"])
    return result
