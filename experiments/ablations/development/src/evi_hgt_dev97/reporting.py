"""All authorized comparisons, REF_FULL minus ablation, without new forwards."""
import csv
import math
from pathlib import Path
from evi_hgt_dev96.reporting import compare as legacy_compare
from .protocol import *
from .runner import logged_updates,verify_receipt,verify_actual_pairing
from .evaluation import verify_result,facility

LIMITS=[
    '同一开发池的两个既有cell，角色重叠；不建立独立泛化、统计显著或优于HGTDR/TxGNN的结论。',
    'EP只截断显式evidence pooled64和最终provenance16两个出口；properties、数量、polarity、availability、SideInfo和D1均保留。',
    'EP的pool、s/c和质量随干预重新学习；不是所有证据对无证据，也不是Evidence与Provenance各自因果贡献。',
    'QUALITY只识别显式动态fact质量context；pool/属性仍含代理，不证明四维展开优于直接s/c，NA不等于U。',
    '名义参数相同但有效容量、输入秩、梯度和AdamW活跃状态不同；两消融不是完全正交因果分解。',
    'softplus为经验P/U代理，不是nnPU或已证明无偏风险；U是未知，contra不是support补集；潜在四值不是临床真值。',
    'available覆盖及诊断仅来自登记训练面板；结构可达和局部logit变化不等于性能增益。上游预训练目标暴露NOT_VERIFIED。',
    '未增加cell、seed、epoch、联合删除臂、疾病末端残差或超参搜索；旧G3 HELD、outer UNSCORED、new final SEALED。',
]


def compare(reference,ablation):
    r=legacy_compare(ablation,reference)  # legacy b-a; here FULL-ablation
    for row in r['per_disease']:
        for metric in ('auc','ap'):
            row['ABLATION_'+metric]=row.pop('E0_'+metric);row['REF_FULL_'+metric]=row.pop('E1_'+metric)
    r['direction']='REF_FULL_MINUS_ABLATION';r['positive_means']='FULL_BETTER';return r


def summarize(failed=False,ui=None):
    m=validate_manifest();reuse=read(OUT/'REFERENCE_REUSE_CHECK.json');trajectories={};comparisons={};points=[];completed=0
    logical=physical=replay=0;unknown_boundaries=[]
    diagnostics={}
    costs=dict(training_seconds=0.,evaluation_completed_forward_seconds=0.,diagnostic_seconds=0.,preparation_seconds=0.,
               completed_evaluation_forwards=0,evaluation_forward_outcomes_unknown=0)
    for cell,arm in QUEUE:
        root=OUT/cell/arm;ident=identity(m,cell,arm);rows=logged_updates(cell,arm);n=max((r['optimizer_step'] for r in rows),default=0)
        logical+=n;physical+=len(rows);replay+=len(rows)-n
        state='NOT_STARTED'
        if (root/'started.json').exists():
            st=read(root/'started.json');state='DEVELOPMENT_RUNNING';costs['preparation_seconds']+=st['pre_training_preparation_seconds']
            if st['initial_tensor_sha256']!=m['cells'][cell]['initial_tensor_sha256'] or st['initial_rng_sha256']!=m['cells'][cell]['initial_rng_sha256']:
                raise ValueError('Common untrained start/RNG differs')
        if (root/'receipt.json').exists():verify_receipt(read(root/'receipt.json'),ident);completed+=1;state='COMPLETE_VERIFIED_DEVELOPMENT'
        elif list(root.glob('FAILED_*.json')):state='PARTIAL_FAILED'
        if (root/'IN_FLIGHT_UPDATE.json').exists():unknown_boundaries.append(dict(cell=cell,arm=arm,boundary=read(root/'IN_FLIGHT_UPDATE.json')))
        costs['training_seconds']+=sum(r['seconds'] for r in rows)
        trajectories[cell+'/'+arm]=dict(status=state,logical_updates=n,known_physical_updates=len(rows),known_replay=len(rows)-n,
            physical_exact='UNRESOLVED' if (root/'IN_FLIGHT_UPDATE.json').exists() else len(rows),
            pairing=verify_actual_pairing(cell,arm,complete=state=='COMPLETE_VERIFIED_DEVELOPMENT'))
        for step in cell_spec(cell)['milestones']:
            path=root/'evaluation'/f'step_{step:08d}'/'evaluation.json';metrics=None
            pc=facility(cell).panel_cost(path.parent)
            costs['evaluation_completed_forward_seconds']+=pc['recorded_forward_seconds'];costs['completed_evaluation_forwards']+=pc['recorded_completed_forwards']
            costs['evaluation_forward_outcomes_unknown']+=pc['in_flight_forward_outcomes_unknown']
            if path.exists():
                e=verify_result(cell,read(path),ident,step);metrics=e['metrics']
                for head in ('support','contra'):
                    c=compare(reuse['cells'][cell]['metrics'][str(step)][head],metrics[head])
                    comparisons[f'{cell}/{arm}/{step}/{head}']=dict(cell=cell,ablation=arm,step=step,primary_endpoint=step==cell_spec(cell)['total'],head=head,**c)
            points.append(dict(cell=cell,arm=arm,step=step,status='AVAILABLE' if metrics else 'UNAVAILABLE',metrics=metrics,costs=pc))
        for path in (root/'diagnostics').glob('step_*.json'):
            diagnostic=read(path);costs['diagnostic_seconds']+=diagnostic['seconds']
            es=[x for record in diagnostic['records'] for x in record['graph_exposures']]
            diagnostics[f'{cell}/{arm}/{diagnostic["step"]}']=dict(groups=len(diagnostic['records']),queries=len(es),
                **{k:sum(e[k] for e in es) for k in ('facts','messages','records','available_facts','NA_facts','available_messages','path_count','path_steps','available_path_steps')},
                query_has_available=sum(e['query_has_available'] for e in es),queries_without_paths=sum(e['path_count']==0 for e in es),
                denominator='Sum of actual per-query graph fact/message slots; not global unique facts or configured budgets',
                state_unchanged=diagnostic['training_state_unchanged'])
    complete=completed==4 and all(p['status']=='AVAILABLE' for p in points) and not unknown_boundaries
    status='DEV97_COMPLETE_VERIFIED_DEVELOPMENT' if complete else 'PARTIAL_FAILED' if failed else 'DEVELOPMENT_RUNNING'
    actual=dict(logical=logical,known_physical_lower=physical,known_replay=replay,unknown_range=[physical,physical+len(unknown_boundaries)],
                exact=physical if not unknown_boundaries else 'UNRESOLVED',unknown_boundaries=unknown_boundaries)
    answers={}
    for arm,label in zip(ARMS,('explicit_Evidence_Provenance_increment','dynamic_fact_quality_context_increment')):
        endpoints=[v for v in comparisons.values() if v['ablation']==arm and v['primary_endpoint']]
        answers[label]=dict(status='DESCRIPTIVE_COMPLETE' if len(endpoints)==4 else 'PENDING',
            endpoint_deltas={v['cell']+'/'+v['head']:{k:v[k] for k in ('delta_auc','delta_ap','ranking_deltas','direction_counts','candidate_tiers')} for v in endpoints},
            interpretation='Positive FULL−ablation means FULL better; mixed/zero/negative values retained. No superiority assumption.')
    signs=lambda value:'FULL_BETTER' if value>1e-12 else 'ABLATION_BETTER' if value< -1e-12 else 'TIED'
    directions={}
    for arm in ARMS:
        directions[arm]={}
        for head in ('support','contra'):
            rows=[v for v in comparisons.values() if v['ablation']==arm and v['head']==head and v['primary_endpoint']]
            directions[arm][head]={metric:dict(per_cell={v['cell']:signs(v['delta_'+metric]) for v in rows},
                both_cells_same_direction=(len(rows)==2 and len({signs(v['delta_'+metric]) for v in rows})==1)) for metric in ('auc','ap')}
    answers['head_and_cell_direction_consistency']=directions
    answers['disease_and_candidate_degradations']='All negative FULL−ablation values, exact per-disease IDs and candidate tiers retained in comparisons; negative means the FULL system degrades.'
    answers['quality_exposure_and_capacity']=dict(actual_new_diagnostics=diagnostics,limits=LIMITS[1:5],
        warning='Insufficient available exposure or zero available path steps limits observation; no effect is not proof all evidence reasoning lacks value.')
    answers['paper_claim_scope']='Only describe measured conditional increments on the observed cells/channels. Positive directions across two cells strengthen development evidence only; mixed/negative/zero directions require narrowing. No causal orthogonality, statistical significance or external superiority.'
    answers['future_formal_questions']='Independent evaluation; dependency-aware inference; separate Evidence and Provenance ablations; four masses versus s/c; formal HGTDR/TxGNN system comparison. None automatically authorized.'
    result=dict(status=status,authorization=m['authorization'],mode='DEVELOPMENT',reference_models_reused=2,new_development_trajectories_complete=completed,
        planned_new_trajectories=4,actual_updates=actual,trajectories=trajectories,milestones=points,comparisons=comparisons,answers=answers,
        new_costs=costs,diagnostic_exposure=diagnostics,historical_reference_costs={c:reuse['cells'][c]['costs'] for c in CELLS},
        reference_training_updates_charged_to_DEV97=0,reference_evaluation_forwards_charged_to_DEV97=0,
        historical_DEV96=reuse['DEV96'],scientific_limits=LIMITS,old_G3='HELD',outer='UNSCORED',new_final='SEALED',
        final_markdown_report=str(OUT/'DEV97_RESULTS.md'),final_json_report=str(OUT/'DEV97_RESULTS.json'))
    write(OUT/'DEV97_RESULTS.json',result,immutable=False)
    lines=['# DEV97 Core Evidence Ablations', '', status,'',f'只读参照2条；新增消融完成{completed}/4。',
        f'逻辑更新{logical}；已知物理下界{physical}；已知重放{replay}；物理范围{actual["unknown_range"]}；精确值{actual["exact"]}。',
        'DEV96：82482逻辑；已知物理下界82598、上界82599；精确值UNRESOLVED；已知重放116；历史7285边界UNKNOWN、旧中断原因UNRESOLVED。','',
        '固定3G为主要终点；1.5G完整保留。差值始终 REF_FULL − ablation；正值完整模型较好。','',
        '|cell/arm/step|head|FULL AUC/AP|消融AUC/AP|ΔAUC|ΔAP|ΔP/R/Hits/NDCG@5/10|','|---|---|---|---|---|---|---|']
    csv_rows=[]
    for c in comparisons.values():
        ref=reuse['cells'][c['cell']]['metrics'][str(c['step'])][c['head']]
        point=next(p for p in points if (p['cell'],p['arm'],p['step'])==(c['cell'],c['ablation'],c['step']))
        a=point['metrics'][c['head']]
        lines.append(f'|{c["cell"]}/{c["ablation"]}/{c["step"]}|{c["head"]}|{ref["macro_auc"]:.9f}/{ref["macro_ap"]:.9f}|{a["macro_auc"]:.9f}/{a["macro_ap"]:.9f}|{c["delta_auc"]:+.9f}|{c["delta_ap"]:+.9f}|{c["ranking_deltas"]}|')
        for metric,delta in [('AUC',c['delta_auc']),('AP',c['delta_ap']),*c['ranking_deltas'].items()]:
            csv_rows.append(dict(cell=c['cell'],ablation=c['ablation'],step=c['step'],primary_endpoint=c['primary_endpoint'],head=c['head'],metric=metric,delta_FULL_minus_ablation=delta))
        lines+=['',f'逐病及候选分层 {c["cell"]}/{c["ablation"]}/{c["step"]}/{c["head"]}：',
                str(c['direction_counts']),str(c['candidate_tiers']),f'可评疾病={a["evaluable_diseases"]}；跳过={a["skipped_diseases"]}。']
    lines+=['','完整逐病改善/退化/持平、原始P/U/N、effective_k、全部TopK及skip明细在JSON。',
        '','训练/评价/诊断新成本：'+str(costs),'历史参照成本单列JSON，不计为新增。','实际训练面板质量暴露：'+str(diagnostics),'',*LIMITS,'',
        '论文可加强或收窄的范围：只按每格每通道固定终点的实际差值讨论两条显式路径的条件增量；方向不一致则收窄为cell/通道依赖。',
        '仍需另行正式授权：独立测试、依赖感知统计推断、Evidence/Provenance分别删除、四维对s/c、正式HGTDR/TxGNN对比。',
        '本轮未自动扩展。OLD G3=HELD；OUTER=UNSCORED；NEW FINAL=SEALED。']
    (OUT/'DEV97_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (OUT/'paired_ablation_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['cell','ablation','step','primary_endpoint','head','metric','delta_FULL_minus_ablation'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(csv_rows)
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,cell in zip(axes,CELLS):
            selected=[v for v in comparisons.values() if v['cell']==cell and v['primary_endpoint']]
            labels=[v['ablation']+'/'+v['head'] for v in selected]
            ax.bar(range(len(selected)),[v['delta_auc'] for v in selected]);ax.axhline(0,color='black',linewidth=.6)
            ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right',fontsize=8);ax.set_title(cell+' 3G FULL - ablation');ax.set_ylabel('ROC-AUC delta')
        fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/paired_3G_AUC.png',dpi=180);plt.close(fig)
    if ui:
        ui.close();ui._write('FINAL STATUS = '+status);ui._write('REFERENCE MODELS REUSED = 2')
        ui._write(f'NEW DEVELOPMENT TRAJECTORIES COMPLETE = {completed} / 4');ui._write('ACTUAL LOGICAL UPDATES = '+str(logical))
        ui._write('KNOWN PHYSICAL UPDATES / REPLAY / UNKNOWN RANGE = '+str((physical,replay,actual['unknown_range'])))
        for c in comparisons.values():ui._write(f'{c["cell"]}/{c["ablation"]}/{c["step"]}/{c["head"]} FULL−ablation AUC={c["delta_auc"]:+.9f} AP={c["delta_ap"]:+.9f} '+str(c['ranking_deltas']))
        ui._write('训练/评价/诊断实际成本 = '+str(costs));ui._write('OLD G3 = HELD\nOUTER = UNSCORED\nNEW FINAL = SEALED')
        ui._write(str(OUT/'DEV97_RESULTS.md'));ui._write(str(OUT/'DEV97_RESULTS.json'))
    return result
