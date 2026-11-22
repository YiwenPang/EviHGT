"""All authorized comparisons, REF_FULL minus ablation, without new forwards."""
import csv
import math
from pathlib import Path
from evi_hgt_paired_losses.reporting import compare as legacy_compare
import ablation_records as records
from .protocol import *
from .runner import logged_updates,verify_receipt,verify_actual_pairing
from .evaluation import verify_result,facility

LIMITS=[
    'Both cells come from the same development pool, with overlapping data roles. The comparison estimates changes within these cells.',
    'EP zeros the pooled 64-dimensional evidence vector and the final 16-dimensional provenance vector. Properties, counts, polarity, availability, SideInfo and D1 remain available.',
    'The EP intervention retrains pooling, support and refutation strengths, and quality estimates. It estimates the joint contribution of the two zeroed outputs.',
    'QUALITY fixes the explicit fact-quality context while retaining pooled evidence and attributes. Not-applicable and unknown are distinct states.',
    'The interventions have equal nominal parameter counts. Effective input rank, gradient flow and active optimizer states differ and can interact.',
    'Half-softplus is an empirical P/U objective. U is unlabeled, and contraindication scores have their own head. The four latent fact states describe model representations.',
    'Training-panel coverage and local logit changes are diagnostic. Pretraining target exposure remains unverified.',
    'The registered study covers its two cells and original training budget. The outer and final panels were held aside.',
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
    status=records.EVIDENCE_COMPLETE if complete else 'PARTIAL_FAILED' if failed else 'DEVELOPMENT_RUNNING'
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
        **{records.REFERENCE_TRAINING_FIELD: 0, records.REFERENCE_EVALUATION_FIELD: 0,
           records.PAIRED_HISTORY_FIELD: reuse[records.PAIRED_REFERENCE_FIELD]},scientific_limits=LIMITS,old_G3='HELD',outer='UNSCORED',new_final='SEALED',
        final_markdown_report=str(OUT/records.EVIDENCE_MARKDOWN),final_json_report=str(OUT/records.EVIDENCE_JSON))
    write(OUT/records.EVIDENCE_JSON,result,immutable=False)
    lines=['# Core evidence ablations', '', records.display_status(status),'',f'Two FULL references reused. Completed interventions: {completed}/4.',
        f'Logical updates: {logical}. Known physical lower bound: {physical}. Replayed: {replay}. Physical range: {actual["unknown_range"]}. Exact: {actual["exact"]}.',
        'The paired-loss comparison recorded 82,482 logical updates and 116 known replays. The physical total is between 82,598 and 82,599 because the outcome of attempt 7285 is unresolved.','',
        'The primary endpoint is 3G. Results at 1.5G are also reported. Each difference is FULL minus the ablation.','',
        '|cell/arm/step|head|FULL AUC/AP|Ablation AUC/AP|ΔAUC|ΔAP|ΔP/R/Hits/NDCG@5/10|','|---|---|---|---|---|---|---|']
    csv_rows=[]
    for c in comparisons.values():
        ref=reuse['cells'][c['cell']]['metrics'][str(c['step'])][c['head']]
        point=next(p for p in points if (p['cell'],p['arm'],p['step'])==(c['cell'],c['ablation'],c['step']))
        a=point['metrics'][c['head']]
        lines.append(f'|{c["cell"]}/{c["ablation"]}/{c["step"]}|{c["head"]}|{ref["macro_auc"]:.9f}/{ref["macro_ap"]:.9f}|{a["macro_auc"]:.9f}/{a["macro_ap"]:.9f}|{c["delta_auc"]:+.9f}|{c["delta_ap"]:+.9f}|{c["ranking_deltas"]}|')
        for metric,delta in [('AUC',c['delta_auc']),('AP',c['delta_ap']),*c['ranking_deltas'].items()]:
            csv_rows.append(dict(cell=c['cell'],ablation=c['ablation'],step=c['step'],primary_endpoint=c['primary_endpoint'],head=c['head'],metric=metric,delta_FULL_minus_ablation=delta))
        lines+=['',f'Per-disease and candidate-stratum results for {c["cell"]}/{c["ablation"]}/{c["step"]}/{c["head"]}',
                str(c['direction_counts']),str(c['candidate_tiers']),f'Evaluable diseases: {a["evaluable_diseases"]}. Skipped: {a["skipped_diseases"]}.']
    lines+=['','See the JSON for per-disease changes, P/U/N counts, effective k, top-k metrics and reasons for skipping a disease.',
        '','Training, evaluation and diagnostic costs: '+str(costs),'The JSON lists costs for reused references separately.','Quality diagnostics on the training panels: '+str(diagnostics),'',*LIMITS,'',
        'Endpoint differences describe the conditional contribution of each explicit intervention by cell and task.',
        'Independent test sets and individual evidence and provenance interventions would answer separate questions.',
        'OLD G3=HELD. OUTER=UNSCORED. NEW FINAL=SEALED.']
    (OUT/records.EVIDENCE_MARKDOWN).write_text('\n'.join(lines)+'\n',encoding='utf-8')
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
        ui.close();ui._write('FINAL STATUS = '+records.display_status(status));ui._write('REFERENCE MODELS REUSED = 2')
        ui._write(f'NEW DEVELOPMENT TRAJECTORIES COMPLETE = {completed} / 4');ui._write('ACTUAL LOGICAL UPDATES = '+str(logical))
        ui._write('KNOWN PHYSICAL UPDATES / REPLAY / UNKNOWN RANGE = '+str((physical,replay,actual['unknown_range'])))
        for c in comparisons.values():ui._write(f'{c["cell"]}/{c["ablation"]}/{c["step"]}/{c["head"]} FULL−ablation AUC={c["delta_auc"]:+.9f} AP={c["delta_ap"]:+.9f} '+str(c['ranking_deltas']))
        ui._write('Measured training, evaluation and diagnostic costs: '+str(costs));ui._write('OLD G3 = HELD\nOUTER = UNSCORED\nNEW FINAL = SEALED')
        ui._write(str(OUT/records.EVIDENCE_MARKDOWN));ui._write(str(OUT/records.EVIDENCE_JSON))
    return result
