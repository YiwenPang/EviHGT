"""Local, descriptive all-cell report; reads authorized caches, never forwards."""
import csv
import math
import statistics
from pathlib import Path
from evi_hgt_loss_comparison.reporting import paired,diagnostic_summary
import ablation_records as records
from .protocol import *
from .runner import logged_updates,verify_receipt
from .evaluation import facility,verify_result

TIERS=(('2–5',2,5),('6–10',6,10),('11–32',11,32),('≥33',33,math.inf))
LIMITS=[
    'The development cells come from the same pool and have overlapping data roles.',
    'The split and its assigned seed change together. Per-disease counts describe the evaluation panel.',
    'R0F0 informed loss selection. The history records the two additional cells and the earlier TEST_ONLY use.',
    'U is unlabeled. Half-softplus is an empirical P/U objective.',
    'Diagnostics cover the small registered training panels.',
    'The CUDA flags allow bitwise differences. Pretraining target exposure remains unverified.',
    'Each cell uses the digest of its E0 checkpoint at G as the shared random source for the paired comparison.',
]


def counts(values):
    return dict(improved=sum(v>1e-12 for v in values),degraded=sum(v< -1e-12 for v in values),
                tied=sum(abs(v)<=1e-12 for v in values),mean=statistics.fmean(values) if values else None)


def compare(a,b):
    r=paired(a,b)
    r['direction_counts']={k:counts([p['delta_'+k] for p in r['per_disease']]) for k in ('auc','ap')}
    r['topk_direction_counts']={k:counts([p['ranking_deltas'][k] for p in r['per_disease']]) for k in r['ranking_deltas']}
    tiers={}
    for name,lo,hi in TIERS:
        rows=[p for p in r['per_disease'] if lo<=p['N']<=hi]
        tiers[name]=dict(diseases=len(rows),P=sum(p['P'] for p in rows),U=sum(p['U'] for p in rows),
            delta_auc=statistics.fmean(p['delta_auc'] for p in rows) if rows else None,
            delta_ap=statistics.fmean(p['delta_ap'] for p in rows) if rows else None,
            ranking_deltas={k:statistics.fmean(p['ranking_deltas'][k] for p in rows) if rows else None for k in r['ranking_deltas']},
            direction_counts={k:counts([p['delta_'+k] for p in rows]) for k in ('auc','ap')},
            scope='DESCRIPTIVE_SMALL_GROUP' if len(rows)<20 else 'DESCRIPTIVE')
    r['candidate_tiers']=tiers
    return r


def verify_pairing(cell,m):
    starts=[read(OUT/cell/a/'started.json') for a in ARMS]
    for key in ('initial_tensor_sha256','initial_metadata','initial_rng_sha256','schedule_digest','corpus_digest'):
        if starts[0][key]!=starts[1][key]:raise ValueError('Same-cell paired initialization/input differs: '+key)
    for st in starts:
        if not st['fresh'] or st['trained_weights_loaded'] or st['test_weights_loaded'] or st['optimizer_initial_state_entries']!=0:
            raise ValueError('Not an independent fresh paired arm')
    rows=[logged_updates(cell,a) for a in ARMS]
    keys=('training_member_sha256','record_sha256','sampling_sha256','stage','queries','original_queries',
          'conditional_extra_U_queries','support_pairs','contra_pairs')
    normalize=lambda rs:{r['optimizer_step']:tuple(r[k] for k in keys) for r in rs}
    if normalize(rows[0])!=normalize(rows[1]):raise ValueError('Actual per-step E0/E1 records differ')
    binding=[read(OUT/cell/a/'paired_plan_binding.json') for a in ARMS]
    if binding[0]['plan_digest']!=binding[1]['plan_digest']:raise ValueError('Shared conditional plan differs')
    # Recreate all original members and conditional U from the actual E0 G anchor.
    from .data import load_data
    from .sampling import freeze_schedule,conditional_plan
    data,corpus=load_data(cell);groups,stream,schedule,cache=freeze_schedule(data,corpus);s=cell_spec(cell)
    anchor=OUT/cell/ARMS[0]/'checkpoints'/f"step_{s['G']:08d}.COMMIT.json"
    plan=conditional_plan(data,groups,schedule,cell=cell,origin_commit=anchor)
    expected={r['step']:(r['member_sha256'],digest(plan['legacy']['records'][r['step']-s['G']-1]) if r['step']>s['G'] else None,
                        plan['digest'] if r['step']>s['G'] else schedule['digest']) for r in stream}
    for rs in rows:
        if {r['optimizer_step']:(r[keys[0]],r[keys[1]],r[keys[2]]) for r in rs}!=expected:raise ValueError('Actual samples differ from complete legal plan')
    return dict(status='ALL_STEPS_ACTUALLY_PAIRED',steps=s['total'],initial_tensor_schema_rng_equal=True,
                source_E0_G_sha256=plan['origin_sha256'],conditional_plan_digest=plan['digest'],all_members_records_order_fallback_equal=True)


def summarize(*,failed=False,ui=None):
    m=validate_manifest();trajectories={};points=[];diagnostics={};comparisons={};pairing={};updates_all=[];complete=0
    costs={k:0. for k in ('training_seconds','query_preparation_seconds','forward_seconds','backward_optimizer_seconds',
                          'evaluation_seconds','evaluation_recorded_forward_seconds','diagnostic_seconds','initial_preparation_seconds','sampling_preparation_seconds')}
    for cell,arm in QUEUE:
        ident=identity(m,cell,arm);s=cell_spec(cell);root=OUT/cell/arm;key=cell+'/'+arm
        rows=logged_updates(cell,arm);updates_all.extend(rows);last=max((r['optimizer_step'] for r in rows),default=0)
        state='NOT_STARTED' if not (root/'started.json').exists() else 'DEVELOPMENT_RUNNING'
        if (root/'receipt.json').exists():
            receipt=verify_receipt(read(root/'receipt.json'),ident);complete+=1;state=receipt['status']
            costs['sampling_preparation_seconds']+=receipt['control']['sampling_preparation_seconds']
        elif list(root.glob('FAILED_*.json')):state='PARTIAL_FAILED'
        for target,src in [('training_seconds','seconds'),('query_preparation_seconds','preparation_seconds'),('forward_seconds','forward_seconds'),('backward_optimizer_seconds','backward_optimizer_seconds')]:
            costs[target]+=sum(r[src] for r in rows)
        if (root/'started.json').exists():costs['initial_preparation_seconds']+=read(root/'started.json')['pre_training_preparation_seconds']
        norms=[r['gradient_norm_before_clip'] for r in rows]
        trajectories[key]=dict(status=state,logical_updates=last,physical_updates=len(rows),replayed_updates=len(rows)-last,
            clipped_updates=sum(r['clipped'] for r in rows),clipping_frequency=sum(r['clipped'] for r in rows)/len(rows) if rows else None,
            finite_zero_gradient_updates=sum(r['zero_gradient_step'] for r in rows),
            gradient_norm_quantiles=quantiles(norms),CUDA_max_recorded_allocated=max((r['CUDA_allocated'] for r in rows),default=0),
            CUDA_max_recorded_reserved=max((r['CUDA_reserved'] for r in rows),default=0),
            training_seconds=sum(r['seconds'] for r in rows),
            optimizer_boundary_unknown=(root/'IN_FLIGHT_UPDATE.json').exists(),
            failures=[dict(path=str(p),**read(p)) for p in root.glob('FAILED_*.json')],
            recoveries=[dict(path=str(p),**read(p)) for p in root.glob('resume_*.json')])
        for step in s['milestones']:
            p=root/'evaluation'/f'step_{step:08d}'/'evaluation.json';metrics=None
            if p.exists():metrics=verify_result(cell,read(p),ident,step)['metrics']
            cost=facility(cell).panel_cost(p.parent);costs['evaluation_recorded_forward_seconds']+=cost['recorded_forward_seconds']
            # Completed panel wall time plus retained failed-attempt wall time.
            costs['evaluation_seconds']+=sum(read(q)['seconds'] for q in p.parent.glob('*.COMPLETE.json'))
            costs['evaluation_seconds']+=sum(read(q)['seconds'] for q in p.parent.glob('*.attempt_*/FAILED.json'))
            points.append(dict(cell=cell,arm=arm,step=step,stage='1.5G' if step==s['milestones'][0] else '3G',
                               status='AVAILABLE' if metrics else 'UNAVAILABLE',metrics=metrics,costs=cost))
        for p in sorted((root/'diagnostics').glob('*.json')):
            d=read(p)
            if d['identity']!=ident or not d['training_state_unchanged']:raise ValueError('Diagnostic identity differs')
            costs['diagnostic_seconds']+=d['seconds']
            summary=diagnostic_summary(d)
            for panel,indices in d['panels'].items():
                for h in ('support','contra'):
                    margins=[v for row in d['records'] if row['group_index'] in indices for v in row['channels'][h]['margins']]
                    summary[panel][h]['margin_quantiles']=quantiles(margins)
            diagnostics[key+':'+str(d['step'])]=dict(path=str(p),sha256=file_sha(p),summary=summary,
                preparation_seconds=d['preparation_seconds'],forward_seconds=d['forward_seconds'],backward_seconds=d['backward_seconds'])
    for cell in CELLS:
        comparisons[cell]={}
        for stage in ('1.5G','3G'):
            ps=[p for p in points if p['cell']==cell and p['stage']==stage]
            if all(p['metrics'] is not None for p in ps):comparisons[cell][stage]={h:compare(ps[0]['metrics'][h],ps[1]['metrics'][h]) for h in ('support','contra')}
        if all(trajectories[cell+'/'+a]['status']=='COMPLETE_VERIFIED_DEVELOPMENT' for a in ARMS):pairing[cell]=verify_pairing(cell,m)
    status=records.PAIRED_COMPLETE if complete==4 and len(pairing)==2 else 'PARTIAL_FAILED' if failed else 'DEVELOPMENT_RUNNING'
    descriptive=None
    if all('3G' in comparisons[c] for c in CELLS):
        descriptive={h:dict(delta_auc=statistics.fmean(comparisons[c]['3G'][h]['delta_auc'] for c in CELLS),
            delta_ap=statistics.fmean(comparisons[c]['3G'][h]['delta_ap'] for c in CELLS),
            ranking_deltas={k:statistics.fmean(comparisons[c]['3G'][h]['ranking_deltas'][k] for c in CELLS) for k in comparisons['R0F1']['3G'][h]['ranking_deltas']}) for h in ('support','contra')}
    history=read(ROOT/records.LOSS_OUTPUT/records.LOSS_JSON)
    discovery=dict(cell='R0F0',role='DISCOVERY_REFERENCE_SEPARATE_FROM_NEW_CELLS',
        milestones=[dict(arm=x['arm'],step=x['step'],metrics={h:{k:v for k,v in met.items() if k in ('macro_auc','macro_ap','ranking','P','U','evaluable_diseases','skipped_diseases')} for h,met in x['metrics'].items()}) for x in history['milestones'] if x['metrics']],
        comparisons={n:{h:{k:v for k,v in c.items() if k!='per_disease'} for h,c in heads.items()} for n,heads in history['comparisons'].items()})
    logical=sum(t['logical_updates'] for t in trajectories.values());physical=len(updates_all)
    result=dict(status=status,mode='DEVELOPMENT',complete_development_trajectories=complete,planned_trajectories=4,
        logical_optimizer_updates=logical,physical_optimizer_updates=physical,replayed_updates=physical-logical,
        trajectories=trajectories,milestones=points,comparisons=comparisons,new_two_cell_equal_weight_descriptive=descriptive,
        actual_pairing=pairing,diagnostics=diagnostics,costs=costs,
        costs_note='query_preparation is internal to training; never add it twice. Evaluation wall seconds includes complete panels and durable failed attempts. Recorded forward seconds is internal; begun incomplete outcomes remain unknown. Preparation/conditional binding timers may overlap and are not added to worker wall seconds.',
        discovery_R0F0=discovery,historical_development_use=m['historical_development_use'],scientific_limits=LIMITS,
        old_G3='HELD',outer='UNSCORED',new_final='SEALED',old_Formal_access=1,new_Final_access=0,
        final_markdown_report=str(OUT/records.PAIRED_MARKDOWN),final_json_report=str(OUT/records.PAIRED_JSON))
    result['answers']=answers(result)
    write(OUT/records.PAIRED_JSON,result,immutable=False);curves(updates_all,points);markdown(result)
    if complete:plots(result)
    if ui:
        ui.stage('FINAL STATUS',records.display_status(status));ui._write(f'COMPLETE DEVELOPMENT TRAJECTORIES = {complete} / 4')
        ui._write(f'ACTUAL LOGICAL / PHYSICAL UPDATES = {logical} / {physical}；REPLAY = {physical-logical}')
        for p in points:
            if p['metrics']:ui.evaluation_result(p['cell']+'/'+p['arm'],p['step'],p['metrics'])
            else:ui._write(f"{p['cell']}/{p['arm']} {p['stage']}: metrics unavailable")
        for cell in CELLS:
            for h,v in comparisons[cell].get('3G',{}).items():ui._write(f"3G {cell} {h} E1−E0 AUC={v['delta_auc']:+.6f} AP={v['delta_ap']:+.6f} Top-K={v['ranking_deltas']}")
        ui._write('Descriptive mean across two cells: '+str(descriptive))
        ui._write('Measured training, evaluation and diagnostic times: '+str(costs))
        ui._write('Failures, recovery and replay: '+str({k:{f:v[f] for f in ('failures','recoveries','replayed_updates','optimizer_boundary_unknown')} for k,v in trajectories.items()}))
        for line in ('OLD G3 = HELD','OUTER = UNSCORED','NEW FINAL = SEALED',str(OUT/records.PAIRED_MARKDOWN),str(OUT/records.PAIRED_JSON)):ui._write(line)
    return result


def quantiles(xs):
    if not xs:return None
    import numpy as np
    return dict(zip(('min','q25','median','q75','q90','q99','max'),np.quantile(xs,[0,.25,.5,.75,.9,.99,1]).tolist()))


def answers(r):
    if r['new_two_cell_equal_weight_descriptive'] is None:return ['Training or dual-task evaluation is incomplete.']*8
    c=r['comparisons'];good=all(c[x]['3G']['support']['delta_auc']>0 and c[x]['3G']['support']['delta_ap']>0 for x in CELLS)
    bad=all(c[x]['3G']['support']['delta_auc']<0 and c[x]['3G']['support']['delta_ap']<0 for x in CELLS)
    decision='Support AUC and AP increased in both cells, supporting further study of half-softplus.' if good else 'Support scores decreased in both cells. The R0F0 result needs further study.' if bad else 'The direction of the support difference varies across the two cells.'
    late_drop=any(next(p for p in r['milestones'] if p['cell']==cell and p['arm']==ARMS[1] and p['stage']=='3G')['metrics']['support']['macro_auc']<next(p for p in r['milestones'] if p['cell']==cell and p['arm']==ARMS[1] and p['stage']=='1.5G')['metrics']['support']['macro_auc'] for cell in CELLS)
    ap_top=[];contra=[];tiers=[];diagnostic=[];resource=[]
    for cell in CELLS:
        s=c[cell]['3G']['support'];t=c[cell]['3G']['contra']
        neg=[k for k,v in s['ranking_deltas'].items() if v< -1e-12]
        ap_top.append(f"{cell} support ΔAP={s['delta_ap']:+.6f}. Lower top-k metrics: {neg or 'none'}.")
        e=[next(p for p in r['milestones'] if p['cell']==cell and p['arm']==ARMS[1] and p['stage']==stage)['metrics']['contra'] for stage in ('1.5G','3G')]
        contra.append(f"{cell} contraindication ΔAUC/AP={t['delta_auc']:+.6f}/{t['delta_ap']:+.6f}. Lower top-k metrics: {[k for k,v in t['ranking_deltas'].items() if v< -1e-12] or 'none'}. E1 change from 1.5G to 3G: ΔAUC/AP={e[1]['macro_auc']-e[0]['macro_auc']:+.6f}/{e[1]['macro_ap']-e[0]['macro_ap']:+.6f}.")
        for h in ('support','contra'):
            desc=[]
            for tier,v in c[cell]['3G'][h]['candidate_tiers'].items():
                desc.append(tier+' is empty' if not v['diseases'] else f"{tier} (n={v['diseases']}) ΔAUC/AP={v['delta_auc']:+.6f}/{v['delta_ap']:+.6f}. Lower top-k metrics: {[k for k,x in v['ranking_deltas'].items() if x< -1e-12] or 'none'}.")
            tiers.append(cell+'/'+h+'：'+'；'.join(desc))
        for panel in ('global32','conditional32'):
            for h in ('support','contra'):
                ds=[r['diagnostics'][cell+'/'+a+':'+str(cell_spec(cell)['total'])]['summary'][panel][h] for a in ARMS]
                lows=[ds[i]['shape_on_identical_raw_scores'][loss]['violated_low_derivative'] for i,loss in enumerate(('sigmoid','softplus'))]
                diagnostic.append(f"{cell}/{panel}/{h}: misordered pairs {ds[0]['negative']} to {ds[1]['negative']}. Small gradients on misordered pairs {lows[0]} to {lows[1]}.")
        ts=[r['trajectories'][cell+'/'+a] for a in ARMS]
        resource.append(f"{cell} E0/E1 clipping={ts[0]['clipping_frequency']:.2%}/{ts[1]['clipping_frequency']:.2%}, training={ts[0]['training_seconds']/3600:.3f}/{ts[1]['training_seconds']/3600:.3f} h, finite zero gradients={ts[0]['finite_zero_gradient_updates']}/{ts[1]['finite_zero_gradient_updates']}.")
    return [decision,
        ' '.join(ap_top)+' The table reports all top-k differences.',
        ' '.join(contra),
        ' '.join(tiers)+' Candidate-size boundaries were prespecified.',
        ' '.join(diagnostic)+' The JSON reports margins and gradient summaries for both tasks.',
        ' '.join(resource)+' See the resource table for gradient quantiles and memory use.',
        'The development estimates are conditional on these two split and seed combinations.',
        ('Inspect the score trajectory under the fixed training budget.' if late_drop else 'Evaluate the effect of the evidence interventions.') if good else 'Check sensitivity to the split and assigned seed, and inspect the contraindication and candidate-stratum results.']


def markdown(r):
    fmt=lambda v:'Unavailable' if v is None else f'{v:.6f}'
    lines=['# Paired loss comparison across splits','',records.display_status(r['status']),'',f"Completed {r['complete_development_trajectories']}/4. Logical/physical updates: {r['logical_optimizer_updates']}/{r['physical_optimizer_updates']}. Replayed: {r['replayed_updates']}.",'',
        '| Cell | Loss | Endpoint/update | Support AUC | AP | Contra AUC | AP |','|---|---|---|---:|---:|---:|---:|']
    for p in r['milestones']:
        vals=[fmt(p['metrics'][h][k]) if p['metrics'] else 'Unavailable' for h in ('support','contra') for k in ('macro_auc','macro_ap')]
        lines.append(f"| {p['cell']} | {p['arm']} | {p['stage']}/{p['step']} | "+' | '.join(vals)+' |')
    lines+=['','| Cell/loss/endpoint | Task | Total P/U | Evaluable P/U | Evaluable/skipped diseases | P@5/10 | R@5/10 | Hits@5/10 | NDCG@5/10 |','|---|---|---|---|---|---|---|---|---|']
    for p in r['milestones']:
        if not p['metrics']:continue
        for h,v in p['metrics'].items():
            pu=[sum(d[k] for d in v['disease_metrics']) for k in ('P','U')]
            top=[' / '.join(fmt(v['ranking'][f'{k}@{n}']) for n in (5,10)) for k in ('P','R','Hits','NDCG')]
            lines.append(f"| {p['cell']}/{p['arm']}/{p['stage']} | {h} | {v['P']}/{v['U']} | {pu[0]}/{pu[1]} | {v['evaluable_diseases']}/{v['skipped_diseases']} | "+' | '.join(top)+' |')
    lines+=['','The JSON lists per-disease P/U/N, effective k, skip reasons and all top-k values. AP groups tied scores by threshold.','',
        '| Cell | Endpoint | Task | ΔAUC | ΔAP | AUC higher/lower/tied | AP higher/lower/tied | ΔTop-K |','|---|---|---|---:|---:|---|---|---|']
    for cell,stages in r['comparisons'].items():
        for stage,heads in stages.items():
            for h,c in heads.items():
                count=lambda k:'/'.join(str(c['direction_counts'][k][x]) for x in ('improved','degraded','tied'))
                lines.append(f"| {cell} | {stage} | {h} | {c['delta_auc']:+.6f} | {c['delta_ap']:+.6f} | {count('auc')} | {count('ap')} | {c['ranking_deltas']} |")
                lines+=['','| Candidates N | Diseases | P/U | ΔAUC | ΔAP | ΔTop-K |','|---|---:|---|---:|---:|---|']
                for tier,t in c['candidate_tiers'].items():lines.append(f"| {cell}/{stage}/{h}/{tier} | {t['diseases']} | {t['P']}/{t['U']} | {fmt(t['delta_auc'])} | {fmt(t['delta_ap'])} | {t['ranking_deltas']} |")
                lines+=['','The per_disease JSON contains every E1 minus E0 difference. The tie tolerance is 1e-12.','']
    lines+=['','Descriptive mean difference across the two added cells:',str(r['new_two_cell_equal_weight_descriptive']),'',
        '| Run | Clipping frequency | Finite zero gradients | Training (s) | Gradient quantiles | Peak CUDA reserved |','|---|---:|---:|---:|---|---:|']
    for k,t in r['trajectories'].items():lines.append(f"| {k} | {fmt(t['clipping_frequency'])} | {t['finite_zero_gradient_updates']} | {t['training_seconds']:.3f} | {t['gradient_norm_quantiles']} | {t['CUDA_max_recorded_reserved']} |")
    lines+=['','| Fixed training diagnostic | Panel | Task | Pairs | Misordered pairs | Small gradients on misordered pairs | LOW_NORM cosine |','|---|---|---|---:|---:|---:|---:|']
    for k,d in r['diagnostics'].items():
        loss='softplus' if ARMS[1] in k else 'sigmoid'
        for panel,heads in d['summary'].items():
            for h,v in heads.items():lines.append(f"| {k} | {panel} | {h} | {v['pairs']} | {v['negative']} | {v['shape_on_identical_raw_scores'][loss]['violated_low_derivative']} | {v['low_norm_cosines']} |")
    lines+=['','Measured training, evaluation, diagnostics and preparation times:',str(r['costs']),'',r['costs_note'],'',
        'The trajectories section of the JSON records failures, recovery, replay and unresolved optimizer state. Incomplete evaluations are marked UNAVAILABLE.','',
        '## Eight prespecified questions','']
    for i,a in enumerate(r['answers'],1):lines.append(f'{i}. {a}')
    lines+=['','## R0F0 development reference','']
    for p in r['discovery_R0F0']['milestones']:
        if p.get('metrics'):lines.append(f"- Paired loss comparison {p['arm']} step={p['step']}: support AUC/AP={p['metrics']['support']['macro_auc']:.6f}/{p['metrics']['support']['macro_ap']:.6f}, contra={p['metrics']['contra']['macro_auc']:.6f}/{p['metrics']['contra']['macro_ap']:.6f}.")
    lines+=['','## Prior development use','',str(r['historical_development_use']),'']+['- '+s for s in LIMITS]
    lines+=['','OLD G3 = HELD / 4 OF 75 COMPLETE','OUTER = UNSCORED','NEW FINAL = SEALED','',r['final_json_report']]
    (OUT/records.PAIRED_MARKDOWN).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def curves(rows,points):
    fields=['cell','arm','step','stage','loss','support_loss','contra_loss','gradient_norm_before_clip','clipped','zero_gradient_step',
            'seconds','preparation_seconds','forward_seconds','backward_optimizer_seconds','support_auc','support_ap','contra_auc','contra_ap']
    score={(p['cell'],p['arm'],p['step']):p['metrics'] for p in points}
    with (OUT/'learning_curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            row={k:r.get(k) for k in fields};row['step']=r['optimizer_step'];met=score.get((r['cell'],r['arm'],r['optimizer_step']))
            if met:
                for h in ('support','contra'):
                    for k in ('auc','ap'):row[h+'_'+k]=met[h]['macro_'+k]
            w.writerow(row)


def plots(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for cell in CELLS:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for ax,h in zip(axes,('support','contra')):
            for a in ARMS:
                p=[x for x in r['milestones'] if x['cell']==cell and x['arm']==a and x['metrics']]
                for metric,style in [('auc','-o'),('ap','--s')]:
                    ax.plot([x['step']/cell_spec(cell)['G'] for x in p],[x['metrics'][h]['macro_'+metric] for x in p],style,label=a[:2]+' '+metric.upper())
            ax.set(xlabel='G',ylabel='Development metric',title=cell+' '+h);ax.legend(fontsize=8)
        fig.savefig(dest/(cell+'_learning_curves.png'),dpi=180);plt.close(fig)
    if r['new_two_cell_equal_weight_descriptive']:
        fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
        labels=[];values=[]
        for cell in CELLS:
            for h in ('support','contra'):
                for metric in ('auc','ap'):labels.append(cell+' '+h+' '+metric.upper());values.append(r['comparisons'][cell]['3G'][h]['delta_'+metric])
        ax.bar(range(len(values)),values,color=['#15803d' if v>=0 else '#b91c1c' for v in values]);ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(range(len(values)),labels,rotation=60,ha='right',fontsize=8);ax.set_ylabel('3G E1 − E0 (descriptive)')
        fig.savefig(dest/'paired_3G_deltas.png',dpi=180);plt.close(fig)
