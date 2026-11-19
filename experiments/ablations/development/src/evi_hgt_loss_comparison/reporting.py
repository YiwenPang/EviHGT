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
    lines=["# 单折配对损失比较","",f"**{records.display_status(status)}**","",
        "DEVELOPMENT / ADAPTIVE。同期 E0 为主要对照；旧 D1 仅为历史参照。U 是未知，排名指标不是临床分类准确率。",
        "", "| 臂 | 步数 | 状态 | Support AUC | Support AP | Contra AUC | Contra AP |",
        "|---|---:|---|---:|---:|---:|---:|"]
    fmt=lambda x:"不可用" if x is None else f"{x:.6f}"
    for p in points:
        values=[fmt(p["metrics"][c][k]) if p["metrics"] else "不可用" for c in ("support","contra") for k in ("macro_auc","macro_ap")]
        lines.append(f"| {p['arm']} | {p['step']} | {p['status']} | "+" | ".join(values)+" |")
    lines += ["","| 臂/步 | 通道 | P/U | 可评/跳过疾病 | P@5/10 | R@5/10 | Hits@5/10 | NDCG@5/10 |",
              "|---|---|---|---|---|---|---|---|"]
    for p in points:
        if not p["metrics"]:continue
        for c,met in p["metrics"].items():
            r=met["ranking"];scores=[" / ".join(fmt(r[f"{name}@{k}"]) for k in (5,10)) for name in ("P","R","Hits","NDCG")]
            lines.append(f"| {p['arm']}/{p['step']} | {c} | {met['P']}/{met['U']} | {met['evaluable_diseases']}/{met['skipped_diseases']} | "+" | ".join(scores)+" |")
    lines += ["","逐病 P/U/N、effective_k、跳过原因、Top-K 和每病 E1−E0 完整差值保存在 JSON。沿用原 tie/skip 和 effective_k=min(k,N)。",
              "","## 预登记九问","",
              "1. 数学区别：m=score_P−score_U，tau=1。E0 为 sigmoid(−m)，稳定解析一阶导数 −exp(−|m|)/(1+exp(−|m|))²；E1 为 0.5 softplus(−m)，正常 autograd 导数 −0.5 sigmoid(−m)。零点均 −0.25；所有配对及归约不变。"]
    final=comparisons.get(str(TOTAL))
    lines.append("2. 错误方向低梯度：下表按原 20 查询和新增两个固定训练面板分别列示；每份诊断还在相同 raw scores 上计算两种导数，避免将网络差异混入纯损失形状比较。")
    lines += ["","| 臂/步 | 面板 | 头 | pair/负 margin | sigmoid 错方向低导数 | softplus 错方向低导数 |",
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
                lines.append(f"3G {panel}/{c}，各自实际目标的错方向低导数 pair：E0={x}/{a[c]['pairs']}，E1={y}/{b[c]['pairs']}，差值={y-x:+d}。仅代表该固定局部面板。")
    if final:
        lines += [f"3. 3G 支持 E1−E0：AUC {final['support']['delta_auc']:+.8f}，AP {final['support']['delta_ap']:+.8f}。这只表示此完整开发面板的配对差值；不证明变化由解决饱和导致。",
                  f"4. 3G 反证 E1−E0：AUC {final['contra']['delta_auc']:+.8f}，AP {final['contra']['delta_ap']:+.8f}。保留全部正向与负向变化。"]
    else:lines += ["3. 支持是否提升：终点配对结果未全部完成，UNRESOLVED。","4. 反证是否改善或退化：终点配对结果未全部完成，UNRESOLVED。"]
    temporal=[]
    for arm in ARMS:
        early=next(p for p in points if p["arm"]==arm and p["step"]==MILESTONES[0])
        late=next(p for p in points if p["arm"]==arm and p["step"]==MILESTONES[1])
        if early["metrics"] and late["metrics"]:
            for c in ("support","contra"):
                delta={k:late["metrics"][c][k]-early["metrics"][c][k] for k in ("macro_auc","macro_ap")}
                temporal.append(f"{arm} {c} 1.5G→3G：ΔAUC={delta['macro_auc']:+.8f}，ΔAP={delta['macro_ap']:+.8f}。")
    lines += ["5. clipping、有限零梯度和成本：",
              "```json",json.dumps(dict(trajectories=trajectories,costs=costs,prelaunch_failures=issues),ensure_ascii=False,indent=2),"```",
              "评价成本逐 attempt 记录；BEGIN 没有 DONE 的在途 forward 不假定为 0。训练、组准备、forward、backward/optimizer、索引准备、评价、诊断独立计时。",
              "6. 过拟合 P/U 代理：训练负 margin 减少而完整开发排名不改善，或 1.5G→3G 开发排名下降，可作为局部警示，不能据此确认临床错误或 U 污染因果。两种训练 loss 尺度不同，不据 loss 大小判优。",
              *temporal,
              "7. 诊断仅覆盖预登记小型训练面板；AUC/AP/Top-K 来自完整、身份一致的 R0F0 开发候选面板。低范数 cosine 标 UNDEFINED/LOW_NORM。",
              "8. E1 不满足常数对称和，不是经证明无偏的 PU 风险，也不是 nnPU。E0 的有界性和对称性可能具有污染鲁棒性价值；E1 对错序保持梯度，但可能加重潜在正例惩罚和 clipping。",
              "9. 是否值得继续验证：只有综合终点 support/contra、各疾病变化、优化与资源代价后才能提出后续验证建议；本轮不自动扩展。单 cell 自适应结果不建立显著性、稳定性、SOTA、独立泛化或机制因果。",
              "","## 历史参照与限制",
              "特征基线 D1 对应点："+json.dumps(history,ensure_ascii=False),
              "E0 如与历史 D1 不同，先核对同一 D1 factory、输入源码、初始 tensor、冻结采样和候选身份；未定位的数值差异原因标 UNKNOWN，不笼统归为 GPU 随机性。",
              "特征基线 D0 从 3G 到 6G 支持 AUC/AP 上升、反证 AUC/AP 下降；51 次重放和非逐位一致限制保留。D2 补齐已验证，但原未完成评价的实际总 forward/耗时未知。",
              "","OLD G3 = HELD；OUTER = UNSCORED；NEW FINAL = SEALED。没有自动下一阶段。"]
    if final:
        lines += ["",("3G 主要 support AUC 差值为正，可作为后续独立复核的开发线索；仍须结合 AP、反证和资源代价，不能确认机制或泛化收益。"
                        if final["support"]["delta_auc"]>0 else
                        "3G 主要 support AUC 未提高；本轮没有提供该主要指标获益的证据，低导数减少本身不足以支持继续扩展实验。")]
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
        ui._write("实际训练/准备/评价/诊断耗时："+str(costs))
        ui._write("错误、恢复与重放记录："+str(dict(trajectories=trajectories,prelaunch_failures=issues)))
        ui._write("OLD G3 = HELD\nOUTER = UNSCORED\nNEW FINAL = SEALED")
        ui._write(str(OUT/records.LOSS_MARKDOWN));ui._write(str(OUT/records.LOSS_JSON))
    return result
