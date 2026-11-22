"""Local final report: all preregistered points, raw-score verification, no scoring."""
import csv
import json
from pathlib import Path
import ablation_records as records
from .protocol import *
from .runner import logged_updates,verify_receipt
from .evaluation import verify_result,panel_cost


def paired(a,b):
    if a["candidate_identity_sha256"]!=b["candidate_identity_sha256"]:
        raise ValueError("Comparison candidate identity differs")
    left={r["disease_id"]:r for r in a["disease_metrics"]}
    right={r["disease_id"]:r for r in b["disease_metrics"]}
    if left.keys()!=right.keys() or a["skipped_disease_details"]!=b["skipped_disease_details"]:
        raise ValueError("Comparison disease/skip identity differs")
    rows=[]
    for d in sorted(left):
        x,y=left[d],right[d]
        if any(x[k]!=y[k] for k in ("P","U","N","effective_k_5","effective_k_10")):
            raise ValueError("Comparison candidate census differs")
        rows.append(dict(disease_id=d,**{k:x[k] for k in ("P","U","N","effective_k_5","effective_k_10")},
            E0_auc=x["auc"],E1_auc=y["auc"],delta_auc=y["auc"]-x["auc"],
            E0_ap=x["ap"],E1_ap=y["ap"],delta_ap=y["ap"]-x["ap"],
            ranking_deltas={k:y[k]-x[k] for k in x if "@" in k and isinstance(x[k],(float,int))}))
    return dict(delta_auc=b["macro_auc"]-a["macro_auc"],delta_ap=b["macro_ap"]-a["macro_ap"],
                ranking_deltas={k:b["ranking"][k]-a["ranking"][k] for k in a["ranking"]},per_disease=rows)


def diagnostic_summary(value):
    import numpy as np
    result={}
    for panel,indices in value["panels"].items():
        rs=[r for r in value["records"] if r["group_index"] in indices]
        result[panel]={}
        for head in ("support","contra"):
            channels=[r["channels"][head] for r in rs]
            shape={name:dict(low_derivative=sum(c[name]["low_derivative"] for c in channels),
                            violated_low_derivative=sum(c[name]["violated_low_derivative"] for c in channels))
                   for name in ("sigmoid","softplus")}
            result[panel][head]=dict(groups=len(rs),pairs=sum(c["pairs"] for c in channels),
                positive=sum(c["positive"] for c in channels),negative=sum(c["negative"] for c in channels),
                zero=sum(c["zero"] for c in channels),shape_on_identical_raw_scores=shape,
                gradient_norm_quantiles=np.quantile([r["head_gradients"][head+"_norm"] for r in rs],[0,.25,.5,.75,1]).tolist(),
                low_norm_cosines=sum(r["shared_gradients"]["status"]=="UNDEFINED/LOW_NORM" for r in rs))
    return result


def summarize(*,failed=False,ui=None):
    m=validate_manifest();trajectories={};points=[];comparisons={};diagnostics={}
    issues=[dict(path=str(p),**read(p)) for p in OUT.glob("prelaunch_attempt*/FAILURE.json")]
    train_rows=[];logical=physical=complete=0
    costs=dict(training_seconds=0.,query_preparation_seconds=0.,forward_seconds=0.,
               backward_optimizer_seconds=0.,evaluation_seconds=0.,diagnostic_seconds=0.,initial_preparation_seconds=0.)
    for arm in ARMS:
        ident=identity(m,arm);root=OUT/arm
        updates=logged_updates(arm);train_rows.extend(updates)
        count=max((r["optimizer_step"] for r in updates),default=0);logical+=count;physical+=len(updates)
        for r in updates:
            for k,source in (("training_seconds","seconds"),("query_preparation_seconds","preparation_seconds"),
                             ("forward_seconds","forward_seconds"),("backward_optimizer_seconds","backward_optimizer_seconds")):
                costs[k]+=r[source]
        state="NOT_STARTED" if not (root/"started.json").exists() else "DEVELOPMENT_RUNNING"
        if (root/"receipt.json").exists():
            receipt=verify_receipt(read(root/"receipt.json"),ident);complete+=1;state=receipt["status"]
        elif list(root.glob("FAILED_*.json")):state="PARTIAL_FAILED"
        elif count==TOTAL:state="EVALUATION_PENDING"
        if (root/"started.json").exists():
            initial=read(root/"started.json")
            costs["initial_preparation_seconds"]+=initial.get("pre_training_preparation_seconds",initial["index_preparation_seconds"])
        trajectories[arm]=dict(status=state,logical_updates=count,physical_updates=len(updates),
            replayed_updates=len(updates)-count,clipped_updates=sum(int(r["clipped"]) for r in updates),
            finite_zero_gradient_updates=sum(int(r["zero_gradient_step"]) for r in updates),
            clipping_frequency=sum(int(r["clipped"]) for r in updates)/len(updates) if updates else None,
            errors=[str(p) for p in root.glob("FAILED_*.json")])
        for step in MILESTONES:
            p=root/"evaluation"/f"step_{step:08d}"/"evaluation.json"
            metrics_value=None
            if p.exists():
                e=verify_result(read(p),ident,step);metrics_value=e["metrics"]
            pc=panel_cost(p.parent);costs["evaluation_seconds"]+=pc["recorded_forward_seconds"]
            points.append(dict(arm=arm,step=step,status="AVAILABLE" if metrics_value else "UNAVAILABLE",
                               metrics=metrics_value,costs=pc))
        for p in sorted((root/"diagnostics").glob("*.json")):
            d=read(p)
            if d["identity"]!=ident or not d["training_state_unchanged"]:raise ValueError("Diagnostic identity changed")
            costs["diagnostic_seconds"]+=d["seconds"]
            diagnostics[arm+":"+str(d["step"])]=dict(path=str(p),sha256=file_sha(p),summary=diagnostic_summary(d))
    for step in MILESTONES:
        selected=[p for p in points if p["step"]==step]
        if all(p["metrics"] is not None for p in selected):
            comparisons[str(step)]={c:paired(selected[0]["metrics"][c],selected[1]["metrics"][c]) for c in ("support","contra")}
    # Require matching actual per-step plan, not seed-only pairing.
    if complete==2:
        a=logged_updates(ARMS[0]);b=logged_updates(ARMS[1])
        normalize=lambda rows:{r["optimizer_step"]:(r["training_member_sha256"],r["record_sha256"],r["sampling_sha256"]) for r in rows}
        if normalize(a)!=normalize(b):raise ValueError("Two trajectories did not consume identical plans")
        starts=[read(OUT/x/"started.json") for x in ARMS]
        for key in ("initial_tensor_sha256","initial_metadata","initial_rng_sha256"):
            if starts[0][key]!=starts[1][key]:raise ValueError("Paired initialization differs")
    status=records.LOSS_COMPLETE if complete==2 else "PARTIAL_FAILED" if failed else "DEVELOPMENT_RUNNING"
    if not failed and complete==0 and all(x["status"]=="NOT_STARTED" for x in trajectories.values()):
        status="REGISTERED_NOT_STARTED"
    review=read(OUT/records.FEATURE_REVIEW)
    history=[r for r in review["milestones"] if r["arm"]=="D1_PRETRAINED_EARLY" and r["step"] in MILESTONES]
    historical_checks=dict(shared_model_factory_and_input_sources=True,common_initial_tensor_verified=True,
         sampling_origin_and_prefix="EXACT_REUSE",candidate_identity="BOUND_ORIGINAL_R0F0_PANELS",
         unexplained_numerical_difference_cause="UNKNOWN; no generic attribution to GPU randomness",
         historical_D1_is_not_contemporaneous_control=True)
    result=dict(status=status,mode="DEVELOPMENT_ADAPTIVE",complete_development_trajectories=complete,
        planned_trajectories=2,logical_optimizer_updates=logical,physical_optimizer_updates=physical,
        replayed_updates=physical-logical,trajectories=trajectories,milestones=points,
        primary_comparison="3G E1-E0 support disease-macro ROC-AUC",comparisons=comparisons,
        diagnostics=diagnostics,costs=costs,historical_D1=history,historical_identity_checks=historical_checks,
        **{records.FEATURE_LIMITATIONS_FIELD: review["d2_prior_physical_forward_total_and_incomplete_seconds"]},
        errors=issues,old_G3="HELD",outer="UNSCORED",new_final="SEALED",
        final_markdown_report=str(OUT/records.LOSS_MARKDOWN),final_json_report=str(OUT/records.LOSS_JSON),
        scientific_limits=["U is unknown, not confirmed negative","No unbiased PU risk or symmetry claim for E1",
          "No direct cross-loss value ranking","Local diagnostics do not estimate full training prevalence",
          "Single adaptive cell: no significance, stable improvement, SOTA, independent generalization or clinical accuracy"])
    write(OUT/records.LOSS_JSON,result,immutable=False)
    with (OUT/"learning_curves.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=["arm","step","loss","support_loss","contra_loss","gradient_norm_before_clip","clipped","zero_gradient_step",
                "support_auc","support_ap","contra_auc","contra_ap"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        scores={(p["arm"],p["step"]):p["metrics"] for p in points if p["metrics"]}
        for r in train_rows:
            row={k:r.get(k) for k in fields};row["step"]=r["optimizer_step"]
            met=scores.get((r["arm"],r["optimizer_step"]))
            if met:
                for c in ("support","contra"):
                    row[c+"_auc"]=met[c]["macro_auc"];row[c+"_ap"]=met[c]["macro_ap"]
            writer.writerow(row)
    lines=["# Paired loss comparison on R0F0","",f"**{records.display_status(status)}**","",
        "The adaptive development comparison uses E0 as the contemporaneous control. D1 is a historical reference. The ranking metrics compare recorded positives with unlabeled candidates.",
        "", "| Arm | Updates | Status | Support AUC | Support AP | Contra AUC | Contra AP |",
        "|---|---:|---|---:|---:|---:|---:|"]
    fmt=lambda x:"Unavailable" if x is None else f"{x:.6f}"
    for p in points:
        values=[fmt(p["metrics"][c][k]) if p["metrics"] else "Unavailable" for c in ("support","contra") for k in ("macro_auc","macro_ap")]
        lines.append(f"| {p['arm']} | {p['step']} | {p['status']} | "+" | ".join(values)+" |")
    lines += ["","| Arm/update | Task | P/U | Evaluable/skipped diseases | P@5/10 | R@5/10 | Hits@5/10 | NDCG@5/10 |",
              "|---|---|---|---|---|---|---|---|"]
    for p in points:
        if not p["metrics"]:continue
        for c,met in p["metrics"].items():
            r=met["ranking"];scores=[" / ".join(fmt(r[f"{name}@{k}"]) for k in (5,10)) for name in ("P","R","Hits","NDCG")]
            lines.append(f"| {p['arm']}/{p['step']} | {c} | {met['P']}/{met['U']} | {met['evaluable_diseases']}/{met['skipped_diseases']} | "+" | ".join(scores)+" |")
    lines += ["","The JSON records per-disease P/U/N, effective k, skip reasons, top-k scores and every E1 minus E0 difference. The evaluator uses the original tie and skip rules and caps k at N.",
              "","## Prespecified questions","",
              "1. Objectives. Let m = score_P minus score_U and tau = 1. E0 is sigmoid(−m), with stable derivative −exp(−|m|)/(1+exp(−|m|))². E1 is half-softplus(−m), with derivative −0.5 sigmoid(−m). Both have derivative −0.25 at zero and use the same pairing and reduction."]
    final=comparisons.get(str(TOTAL))
    lines.append("2. Small gradients on misordered pairs. The table covers the original 20 queries and two additional fixed training panels. It also evaluates both derivatives on the same scores to isolate the loss shape.")
    lines += ["","| Arm/update | Panel | Task | Pairs/negative margin | Small sigmoid gradients on misordered pairs | Small softplus gradients on misordered pairs |",
              "|---|---|---|---|---:|---:|"]
    for key,d in diagnostics.items():
        for panel,heads in d["summary"].items():
            for c,x in heads.items():
                shape=x["shape_on_identical_raw_scores"]
                lines.append(f"| {key} | {panel} | {c} | {x['pairs']}/{x['negative']} | {shape['sigmoid']['violated_low_derivative']} | {shape['softplus']['violated_low_derivative']} |")
    lines += [""]
    for panel in ("legacy20","global32","conditional32"):
        a=diagnostics.get(ARMS[0]+":"+str(TOTAL),{}).get("summary",{}).get(panel)
        b=diagnostics.get(ARMS[1]+":"+str(TOTAL),{}).get("summary",{}).get(panel)
        if a is not None and b is not None:
            for c in ("support","contra"):
                x=a[c]["shape_on_identical_raw_scores"]["sigmoid"]["violated_low_derivative"]
                y=b[c]["shape_on_identical_raw_scores"]["softplus"]["violated_low_derivative"]
                lines.append(f"3G {panel}/{c}: small gradients on misordered pairs under each arm's objective. E0={x}/{a[c]['pairs']}, E1={y}/{b[c]['pairs']}, difference={y-x:+d}.")
    if final:
        lines += [f"3. At 3G, support differences (E1 minus E0): AUC {final['support']['delta_auc']:+.8f}, AP {final['support']['delta_ap']:+.8f}. Scores use the complete paired development panel.",
                  f"4. At 3G, contraindication differences (E1 minus E0): AUC {final['contra']['delta_auc']:+.8f}, AP {final['contra']['delta_ap']:+.8f}."]
    else:lines += ["3. The paired support endpoint is incomplete.","4. The paired contraindication endpoint is incomplete."]
    temporal=[]
    for arm in ARMS:
        early=next(p for p in points if p["arm"]==arm and p["step"]==MILESTONES[0])
        late=next(p for p in points if p["arm"]==arm and p["step"]==MILESTONES[1])
        if early["metrics"] and late["metrics"]:
            for c in ("support","contra"):
                delta={k:late["metrics"][c][k]-early["metrics"][c][k] for k in ("macro_auc","macro_ap")}
                temporal.append(f"{arm} {c} 1.5G→3G：ΔAUC={delta['macro_auc']:+.8f}，ΔAP={delta['macro_ap']:+.8f}。")
    lines += ["5. Clipping, finite zero gradients and measured costs.",
              "```json",json.dumps(dict(trajectories=trajectories,costs=costs,prelaunch_failures=issues),ensure_ascii=False,indent=2),"```",
              "Each evaluation attempt records duration independently. Attempts with no completion record have unknown forward-pass counts. Timings distinguish training, group preparation, forward passes, backward passes and optimizer steps, indexing, evaluation and diagnostics.",
              "6. Training fit and development ranking. Negative margins can decrease without an improvement in development ranks. The two training losses have different scales. Score differences are interpreted on their common development candidates.",
              *temporal,
              "7. The training diagnostics cover small registered panels. AUC, AP and top-k metrics use the complete R0F0 development panel. Cosine similarity for low-norm vectors is recorded as UNDEFINED/LOW_NORM.",
              "8. E0 is bounded and has a constant symmetric sum. E1 retains gradients on misordered pairs and can increase both the penalty on latent positives and gradient clipping. The study evaluates half-softplus as an empirical P/U objective.",
              "9. The endpoint scores, per-disease changes and measured costs together inform further validation. The estimates are conditional on one adaptive development cell.",
              "","## Historical reference","",
              "D1 feature-baseline reference: "+json.dumps(history,ensure_ascii=False),
              "When E0 differs from the D1 reference, check the factory, source inputs, initial tensors, sampling plan and candidate identities. Unresolved causes are recorded as UNKNOWN.",
              "From 3G to 6G, D0 support AUC/AP rose and contraindication AUC/AP fell. The run replayed 51 updates and bitwise equivalence was unverified. The D2 completion was verified. The number and duration of the original partial forward passes are unknown.",
              "","OLD G3 = HELD. OUTER = UNSCORED. NEW FINAL = SEALED."]
    if final:
        lines += ["",("The 3G support AUC difference is positive on this development panel."
                        if final["support"]["delta_auc"]>0 else
                        "The 3G support AUC did not increase on this development panel. The gradient diagnostic measures a separate property.")]
    (OUT/records.LOSS_MARKDOWN).write_text("\n".join(lines)+"\n",encoding="utf-8")
    available=[p for p in points if p["metrics"]]
    if available:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
        for i,c in enumerate(("support","contra")):
            for j,k in enumerate(("macro_auc","macro_ap")):
                ax=axes[i,j]
                for arm in ARMS:
                    rs=[p for p in available if p["arm"]==arm]
                    if rs: ax.plot([p["step"]/G for p in rs],[p["metrics"][c][k] for p in rs],marker="o",label=arm.split("_")[0])
                ax.set(title=f"{c} {k}",xlabel="Training budget (G)",ylabel=k);ax.set_xticks([1.5,3]);ax.legend()
        fig.suptitle("Single-fold loss comparison: registered development milestones")
        fig.savefig(OUT/"auc_ap_curves.png",dpi=160);plt.close(fig)
    if ui:
        ui.close();ui.stage("FINAL STATUS",records.display_status(status))
        ui._write(f"COMPLETE DEVELOPMENT TRAJECTORIES / 2 = {complete}/2")
        ui._write(f"ACTUAL LOGICAL / PHYSICAL UPDATES = {logical} / {physical}")
        for p in points:
            met=p["metrics"]
            ui._write(f"{p['arm']} step={p['step']} "+("UNAVAILABLE" if met is None else
                "；".join(f"{c} AUC/AP={met[c]['macro_auc']:.6f}/{met[c]['macro_ap']:.6f}" for c in ("support","contra"))))
        ui._write("Measured training, preparation, evaluation and diagnostic times: "+str(costs))
        ui._write("Errors, recovery and replay: "+str(dict(trajectories=trajectories,prelaunch_failures=issues)))
        ui._write("OLD G3 = HELD\nOUTER = UNSCORED\nNEW FINAL = SEALED")
        ui._write(str(OUT/records.LOSS_MARKDOWN));ui._write(str(OUT/records.LOSS_JSON))
    return result
