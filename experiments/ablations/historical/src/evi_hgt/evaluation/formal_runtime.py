"""Evaluation-only full-candidate runtime. Data entry requires separate authorization.

One bounded graph remains pair-specific; immutable encodings and standard-model
input assembly are reused. Synthetic preflight never enters the real-data loader.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from evi_hgt.evaluation.candidates import iter_disease_candidates
from evi_hgt.evaluation.protocol import (
    EvaluationProtocolBinding, RankingCandidateProtocol, TRAINING_KNOWN_RELATIONS, RANKING_K,
)
from evi_hgt.evaluation.ranking import evaluate_disease_ranking

CHANNELS = ('support', 'unsuitability')
METRICS = tuple(f'{name}@{k}' for name in ('P','R','Hits','NDCG') for k in RANKING_K)
SECONDARY = ('Support ROC-AUC','Support AUPR','Unsuitability ROC-AUC','Unsuitability AUPR')
COLUMNS = ('MODEL','SPLIT','CHANNEL','STATUS', *METRICS, *SECONDARY,
           'EVALUABLE_QUERIES','SKIPPED_QUERIES','SECONDARY_EVALUABLE_QUERIES','REASON','LOG PATH')
U_DEFINITION = 'CANDIDATES_MINUS_CHANNEL_HELD_OUT_RELEVANCE'
FROZEN_FORMAL_MATRIX = (
    ('FULL','FULL_EVIHGT','standard'),
    ('BASE','BASE_HGT','standard'),
    ('NO_EVIDENCE_CONTEXT','NO_EVIDENCE_CONTEXT','standard'),
    ('NO_FOUR_VALUED_LOGIC','NO_FOUR_VALUED_LOGIC','standard'),
    ('NO_PATH_REASONING','NO_PATH_REASONING','standard'),
    ('NO_LOGIC_MASS_CONDITIONING','NO_LOGIC_MASS_CONDITIONING','standard'),
    ('COLD_DRUG','FULL_EVIHGT','cold_drug'),
    ('COLD_DISEASE','FULL_EVIHGT','cold_disease'),
)


class SharedInvariantError(RuntimeError):
    """Abort the entire pipeline; no automatic retry."""


class ModelEvaluationError(RuntimeError):
    """An isolated forward failure (e.g. OOM); other frozen cells may continue."""


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    content=json.dumps(value, indent=2, allow_nan=False) + '\n'
    with temporary.open('w',encoding='utf-8') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _commit_first_access_ledger(path, ledger):
    """Called under the kernel lease; a failed replace cannot precede test access."""
    if Path(path).exists():
        raise PermissionError('First formal test run already entered')
    atomic_json(path,ledger)


def _phase840_formal_run_identity(first_access_path, expected_binding):
    """Bind a future full execution to the irreversible Phase 8.4.0 entry."""
    ledger=read_json(first_access_path)
    if (ledger.get('formal_test_count')!=1 or ledger.get('formal_test_accessed') is not True
            or not ledger.get('formal_run_id')):
        raise ValueError('first formal access ledger missing or invalid')
    actual=ledger.get('binding',{})
    if any(actual.get(key)!=value for key,value in expected_binding.items()):
        raise ValueError('first formal access binding mismatch')
    return ledger['formal_run_id']


def secondary_metrics(scores, relevant):
    """Frozen disease-wise P versus remaining candidates; never true negatives."""
    from evi_hgt.evaluation.validation_diagnostics import _threshold_curve
    import numpy as np
    p = np.asarray([v for drug,v in scores.items() if drug in relevant], dtype=float)
    u = np.asarray([v for drug,v in scores.items() if drug not in relevant], dtype=float)
    if not p.size or not u.size:
        return {'roc_auc':None,'average_precision':None,'reason':'NO_P_OR_NO_PROTOCOL_U', 'roc':[], 'pr':[]}
    roc,pr,auc,ap = _threshold_curve(p,u)
    # Curves are stored at fixed x positions for macro figure display only;
    # metric values above use every tied threshold, not this display grid.
    grid = np.linspace(0,1,101)
    return {'roc_auc':auc,'average_precision':ap,'reason':'',
            'roc':np.interp(grid,[p['fpr_proxy'] for p in roc],[p['tpr_observed_p'] for p in roc]).tolist(),
            'pr':np.interp(grid,[p['recall_observed_p'] for p in pr],[p['precision_p_vs_u'] for p in pr]).tolist()}


def evaluate_cell(supervision, split_frame, *, formal_drug_ids, split_kind, partition,
                  score_pair, output, identity, model, fixture=False, write_result=True):
    if partition != 'test' and not fixture:
        raise SharedInvariantError('Real validation replay is not a formal runtime mode')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    identity = dict(identity, model=model, split=split_kind, partition=partition,
                    fixture=fixture, output_schema='FORMAL_RETRIEVAL_V1')
    identity_path = output/'identity.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise SharedInvariantError('resume identity mismatch')
    atomic_json(identity_path,identity)
    started=time.monotonic(); queries=[]
    def compact(payload):
        return {'results':{c:{'ranking':{k:v for k,v in item['ranking'].items() if k!='ranked_drug_ids'},
                               'secondary':item['secondary']} for c,item in payload['results'].items()}}
    streams = [iter_disease_candidates(supervision,split_frame,split_kind=split_kind,
               partition=partition,channel=c,formal_drug_ids=formal_drug_ids) for c in CHANNELS]
    try:
        for support,counter in zip(*streams, strict=True):
            if (support.disease_id != counter.disease_id or
                    support.candidate_membership_digest != counter.candidate_membership_digest):
                raise SharedInvariantError('channel candidate membership mismatch')
            expected=dict(identity=identity, disease=support.disease_id,
                          definitions=[support.definition_digest,counter.definition_digest])
            path=output/'queries'/(digest(support.disease_id)+'.json')
            if path.exists():
                saved=read_json(path)
                if saved['sha256'] != digest(saved['payload']) or saved['payload']['binding'] != expected:
                    raise SharedInvariantError('query manifest corruption or resume identity mismatch')
                queries.append(compact(saved['payload'])); continue
            scores={c:{} for c in CHANNELS}
            for drug in support.candidate_drug_ids:
                values=score_pair(drug,support.disease_id)
                if len(values)!=2 or not all(math.isfinite(float(v)) for v in values):
                    raise ModelEvaluationError('nonfinite or invalid dual-head prediction')
                for c,v in zip(CHANNELS,values,strict=True): scores[c][drug]=float(v)
            results={}
            for candidates in (support,counter):
                channel=candidates.channel
                ranking=evaluate_disease_ranking(binding=EvaluationProtocolBinding(split_kind,partition,channel),
                    disease_id=candidates.disease_id,candidate_drug_ids=candidates.candidate_drug_ids,
                    scores=tuple(scores[channel].values()), relevant_drug_ids=candidates.held_out_relevant_drug_ids)
                results[channel]={'ranking':ranking.to_payload(),
                    'secondary':secondary_metrics(scores[channel],set(candidates.held_out_relevant_drug_ids))}
            payload={'binding':expected, 'candidate_ids':support.candidate_drug_ids,
                     'scores':{c:list(v.values()) for c,v in scores.items()},'results':results}
            atomic_json(path,{'payload':payload,'sha256':digest(payload)})
            # Retain summaries only in memory, never all drug-disease scores.
            queries.append(compact(payload))
            print(f'MODEL={model} SPLIT={split_kind} DISEASES_COMMITTED={len(queries)}',flush=True)
    except (ValueError,KeyError,TypeError) as exc:
        raise SharedInvariantError(f'candidate/metric/output invariant: {exc}') from exc
    if not queries:
        raise SharedInvariantError('empty query manifest')
    rows=[]; curves={}
    for channel in CHANNELS:
        results=[q['results'][channel] for q in queries]
        eligible=[q['ranking'] for q in results if q['ranking']['evaluable']]
        secondary=[q['secondary'] for q in results if q['secondary']['roc_auc'] is not None]
        row={key:None for key in COLUMNS}
        row.update(MODEL=model,SPLIT=split_kind,CHANNEL=channel,STATUS='COMPLETE',
                   EVALUABLE_QUERIES=len(eligible), SKIPPED_QUERIES=len(results)-len(eligible),
                   SECONDARY_EVALUABLE_QUERIES=len(secondary), REASON='' if eligible else 'NO_ELIGIBLE_POSITIVE_QUERY')
        mapping={'P':'known_positive_precision','R':'known_positive_recall','Hits':'hits_at_k','NDCG':'ndcg'}
        for short,field in mapping.items():
            for i,k in enumerate(RANKING_K):
                row[f'{short}@{k}']=statistics.fmean(q['metrics'][i][field] for q in eligible) if eligible else None
        title=channel.capitalize()
        row[title+' ROC-AUC']=statistics.fmean(s['roc_auc'] for s in secondary) if secondary else None
        row[title+' AUPR']=statistics.fmean(s['average_precision'] for s in secondary) if secondary else None
        if not secondary: row['REASON'] += ' NO_P_OR_NO_PROTOCOL_U_FOR_SECONDARY'
        curves[channel]={kind:[statistics.fmean(s[kind][i] for s in secondary) for i in range(101)]
                         if secondary else [] for kind in ('roc','pr')}
        rows.append(row)
    for row in rows:
        for other in rows:
            for key in SECONDARY:
                if other[key] is not None: row[key]=other[key]
    result={'model':model,'split':split_kind,'rows':rows,'curves':curves,'fixture':fixture,
            'identity':identity, 'wall_seconds_this_invocation':time.monotonic()-started,
            'secondary_u_definition':U_DEFINITION,'unlabeled_is_negative':False,
            'query_count':len(queries),'status':'COMPLETE'}
    result_path=output/'result.json'
    if result_path.exists():
        saved=read_json(result_path)
        if not _existing_result_matches(saved,result):
            raise SharedInvariantError('completed cell result changed on resume')
        return saved
    if write_result:
        atomic_json(result_path,result)
    return result


def failure_result(model,split,reason,log):
    rows=[]
    for channel in CHANNELS:
        row={k:None for k in COLUMNS}
        row.update(MODEL=model,SPLIT=split,CHANNEL=channel,STATUS='FAILED',REASON=reason)
        row['LOG PATH']=log; rows.append(row)
    return {'model':model,'split':split,'status':'FAILED','rows':rows,'curves':{}}


def write_abort_report(output,cells,completed,*,active_label,reason,fixture=False):
    """Durable failure accounting without rerunning a model or the plot writer."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    log=output/'ABORTED.log';log.write_text(reason,encoding='utf-8')
    rows=[]
    for cell in cells:
        existing=next((r for r in completed if (r['model'],r['split'])==(cell['model_variant'],cell['split'])),None)
        if existing:
            rows.extend(existing['rows']);continue
        missing=failure_result(cell['model_variant'],cell['split'],reason,str(log))
        for row in missing['rows']:
            if cell['label']!=active_label: row['STATUS']='NOT_RUN_ABORTED'
            print(f"MODEL={row['MODEL']} STATUS={row['STATUS']} CHANNEL={row['CHANNEL']}",flush=True)
            for metric in SECONDARY: print(metric+' = N/A',flush=True)
            print('REASON = '+reason,flush=True)
        rows.extend(missing['rows'])
    payload={'status':'ABORTED_SHARED_INVARIANT','reason':reason,'rows':rows,'fixture':fixture,
             'formal_test_count':0 if fixture else 1}
    atomic_json(output/'ABORTED.json',payload)
    atomic_json(output/'summary.json',payload)
    temporary=output/'summary.csv.tmp'
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS);writer.writeheader()
        writer.writerows({k:('N/A' if row[k] is None else row[k]) for k in COLUMNS} for row in rows)
    temporary.replace(output/'summary.csv')
    (output/'REPORT.md').write_text('# '+('SYNTHETIC ONLY — ' if fixture else '')+'ABORTED — shared invariant\n\n'+reason+'\n\n'+
        'Uncomputed metrics are N/A, not zero. See summary.csv and ABORTED.json.\n',encoding='utf-8')


def write_outputs(output,results,*,fixture=False,metadata=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[r for result in results for r in result['rows']]
    temp=output/'summary.csv.tmp'
    with temp.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS); writer.writeheader()
        writer.writerows({k:('N/A' if row[k] is None else row[k]) for k in COLUMNS} for row in rows)
    temp.replace(output/'summary.csv')
    atomic_json(output/'summary.json',{'fixture':fixture,'rows':rows,'unlabeled_is_negative':False})
    atomic_json(output/'run_metadata.json',dict(metadata or {}, fixture=fixture,
        output_schema='FORMAL_RETRIEVAL_V1',secondary_u_definition=U_DEFINITION,
        raw_logits_are_calibrated_probabilities=False, aggregation='MACRO_ACROSS_EVALUABLE_DISEASES'))
    lines=['# '+('SYNTHETIC PREFLIGHT ONLY' if fixture else 'Formal known-positive retrieval'),
           '', 'U != confirmed negative. Raw logits are not calibrated probabilities.',
           'Support and treatment-unsuitability are separate retrieval tasks. Contraindication != inefficacy.',
           '', '| '+' | '.join(COLUMNS)+' |','| '+' | '.join('---' for _ in COLUMNS)+' |']
    for row in rows:
        lines.append('| '+' | '.join('N/A' if row[k] is None else str(row[k]) for k in COLUMNS)+' |')
        print(f"MODEL = {row['MODEL']} / {row['SPLIT']} / {row['CHANNEL']} STATUS = {row['STATUS']}",flush=True)
        for key in SECONDARY: print(f'{key} = '+('N/A' if row[key] is None else f'{row[key]:.6f}'),flush=True)
        if row['REASON']: print('REASON = '+row['REASON'],flush=True)
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figures=output/'figures'; figures.mkdir(exist_ok=True)
    # Each split has its own figure; cold results never share a comparison panel.
    for split in sorted({r['SPLIT'] for r in rows}):
        for metric in ('P','R','Hits','NDCG'):
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for channel,ax in zip(CHANNELS,axes,strict=True):
                for row in rows:
                    if row['SPLIT']==split and row['CHANNEL']==channel and row[f'{metric}@5'] is not None:
                        ax.plot(RANKING_K,[row[f'{metric}@{k}'] for k in RANKING_K],marker='o',label=row['MODEL'])
                ax.set(title=channel,xlabel='K',ylabel=metric+'@K',ylim=(0,1.02));
                if ax.lines: ax.legend(fontsize=6)
            fig.suptitle(('SYNTHETIC ONLY | ' if fixture else '')+split)
            fig.tight_layout(); fig.savefig(figures/f'{split}_{metric}_at_k.png',dpi=130); plt.close(fig)
        for channel in CHANNELS:
            for kind in ('roc','pr'):
                fig,ax=plt.subplots(figsize=(5,4))
                for result in results:
                    curve=result.get('curves',{}).get(channel,{}).get(kind,[])
                    if result['split']==split and curve:
                        ax.plot(np.linspace(0,1,101),curve,label=result['model'])
                ax.set(title=('SYNTHETIC ONLY | ' if fixture else '')+f'{split} {channel} P-vs-U',
                       xlabel='FPR proxy' if kind=='roc' else 'Recall',ylabel='TPR' if kind=='roc' else 'Precision',ylim=(0,1.02))
                if ax.lines: ax.legend(fontsize=6)
                fig.tight_layout();fig.savefig(figures/f'{split}_{channel}_{kind}.png',dpi=130);plt.close(fig)


def validate_plan_bindings(plan, source_cells):
    if plan.get('readiness')!='READY' or plan.get('blockers') or any(c.get('readiness')!='READY' for c in plan['cells']):
        raise SharedInvariantError('formal matrix is blocked')
    matrix=tuple((cell['label'],cell['model_variant'],cell['split']) for cell in plan['cells'])
    if matrix!=FROZEN_FORMAL_MATRIX:
        raise SharedInvariantError('frozen eight-cell formal matrix mismatch')
    fields=('label','model_variant','split','run_id','checkpoint_sha256',
            'checkpoint_identity','cold_visibility_policy_digest')
    expected=[tuple(c[k] for k in fields) for c in source_cells]
    actual=[tuple(c[k] for k in fields) for c in plan['cells']]
    if not actual or actual!=expected or len({c['label'] for c in plan['cells']})!=len(actual):
        raise SharedInvariantError('checkpoint/cell mapping mismatch')


def _runtime_rates(root, plan):
    """Read the frozen planning coefficients before any sealed input is opened."""
    path = root/plan.get('runtime_estimate_path', '')
    try:
        estimate = read_json(path)
        rates = estimate['seconds_per_distinct_pair']
        values = {split: float(rates[key]) for split,key in
                  (('standard','standard_six'),('cold_drug','cold_drug'),('cold_disease','cold_disease'))}
        setup = float(estimate['setup']['planning_reserve_seconds'])
        if setup < 0 or any(not math.isfinite(v) or v <= 0 for v in values.values()) or not math.isfinite(setup):
            raise ValueError('nonfinite or nonpositive coefficient')
    except (OSError,ValueError,TypeError,KeyError) as exc:
        raise SharedInvariantError(f'frozen runtime estimate unavailable or invalid: {exc}') from exc
    return {'seconds_per_pair':values,'setup_seconds':setup,'source_sha256':file_sha(path)}


def _count_split_volume(supervision, split_frame, split_kind, drugs):
    """Count exact frozen membership using structural columns, without relevance."""
    import polars as pl
    protocol=RankingCandidateProtocol(split_kind,'test','support')
    source_cols={'drug_id','disease_id','raw_relation'}
    split_cols={'drug_id','disease_id','split_assignment','drug_partition','disease_partition'}
    if source_cols-set(supervision.columns) or split_cols-set(split_frame.columns):
        raise SharedInvariantError('candidate volume structural columns missing')
    source=supervision.select(sorted(source_cols));split=split_frame.select(sorted(split_cols))
    observed={str(v) for v in source['raw_relation'].unique().to_list()}
    if observed-set(TRAINING_KNOWN_RELATIONS):
        raise SharedInvariantError('candidate volume unexpected DDA relation')
    joined=source.join(split,on=['drug_id','disease_id'],how='inner',validate='1:1')
    if joined.height!=source.height or joined.height!=split.height:
        raise SharedInvariantError('candidate volume split binding mismatch')
    assignments={str(v) for v in joined['split_assignment'].unique().to_list() if v is not None}
    if not assignments<={'train','validation','test'}:
        raise SharedInvariantError('candidate volume unexpected split assignment')
    all_drugs=tuple(sorted({str(value) for value in drugs}))
    if protocol.candidate_drug_scope=='MATCHING_HELD_OUT_DRUG_PARTITION':
        scoped=set(str(v) for v in split_frame.filter(pl.col('drug_partition')=='test')['drug_id'].unique().to_list())
        candidate_drugs=tuple(v for v in all_drugs if v in scoped)
    else:
        candidate_drugs=all_drugs
    if not candidate_drugs:
        raise SharedInvariantError('candidate volume drug scope is empty')
    if protocol.evaluation_disease_scope=='MATCHING_HELD_OUT_DISEASE_PARTITION':
        diseases=tuple(sorted(str(v) for v in split_frame.filter(pl.col('disease_partition')=='test')
                              ['disease_id'].unique().to_list()))
    else:
        diseases=tuple(sorted(str(v) for v in split_frame.filter(pl.col('split_assignment')=='test')
                              ['disease_id'].unique().to_list()))
    if not diseases:
        raise SharedInvariantError('candidate volume disease scope is empty')
    excluded=joined.filter((pl.col('split_assignment')=='train') &
        pl.col('raw_relation').is_in(TRAINING_KNOWN_RELATIONS) &
        pl.col('drug_id').is_in(candidate_drugs))
    excluded_by_disease={str(disease):int(count) for disease,count in
        excluded.group_by('disease_id').agg(pl.col('drug_id').n_unique().alias('count'))
        .select('disease_id','count').iter_rows()}
    if any(excluded_by_disease.get(disease,0)>=len(candidate_drugs) for disease in diseases):
        raise SharedInvariantError('candidate volume contains an empty query candidate set')
    return len(diseases),sum(len(candidate_drugs)-excluded_by_disease.get(disease,0) for disease in diseases)


def _first_entry_projection(output, plan, rates, supervision, split_frames, drugs):
    """Count candidate membership after ledger, without reading relevance columns."""
    query_counts={};pairs={}
    for split in dict.fromkeys(cell['split'] for cell in plan['cells']):
        query_counts[split],pairs[split]=_count_split_volume(supervision,split_frames[split],split,drugs)
    projected={'query_counts':query_counts,'unique_pairs':pairs,
        'estimated_wall_seconds':rates['setup_seconds']+sum(
            pairs[split]*rates['seconds_per_pair'][split] for split in pairs),
        'rates':rates,'candidate_protocol':'FROZEN'}
    path=output/'runtime_projection.json'
    if path.exists():
        if read_json(path)!=projected:
            raise SharedInvariantError('first-entry runtime projection changed on resume')
    else:
        atomic_json(path,projected)
    for split,label in (('standard','STANDARD'),('cold_drug','COLD_DRUG'),('cold_disease','COLD_DISEASE')):
        print(f'{label} PAIRS = {pairs.get(split,0)}',flush=True)
    print(f'ESTIMATED WALL CLOCK = {projected["estimated_wall_seconds"] / 3600:.3f} hours '
          '(LOW-CONFIDENCE SYNTHETIC-RATE PROJECTION)',flush=True)
    return projected


def _shared_score_identity(root, plan, plan_sha, split, cells, formal_run_id=None):
    protocol_path=root/plan['candidate_protocol_path']
    code_config=[{'path':item['path'],'sha256':item['sha256']} for item in plan['protected_files']
                 if item['path'].startswith(('src/','scripts/','configs/'))]
    identity={'plan_sha256':plan_sha,'split':split,
            'split_sha256':plan['data_binding']['split_sha256'][split],
            'candidate_protocol_sha256':file_sha(protocol_path),
            'code_config_sha256':digest(code_config),
            'checkpoints':{cell['label']:{'run_id':cell['run_id'],
                'checkpoint_sha256':cell['checkpoint_sha256'],
                'checkpoint_identity':cell.get('checkpoint_identity')} for cell in cells}}
    if formal_run_id is not None:
        identity['formal_run_id']=formal_run_id
        identity['metric_config_sha256']=digest(read_json(protocol_path)['metrics'])
    return identity


def _existing_result_matches(saved, current):
    """Ignore invocation duration and separately attached log path only."""
    def stable(value):
        value=json.loads(json.dumps(value,allow_nan=False))
        value.pop('wall_seconds_this_invocation',None)
        for row in value['rows']:
            row['LOG PATH']=None
        return value
    return stable(saved)==stable(current)


def run_formal_evaluation(root, *, authorization_path, plan_path=None, resume=False):
    """Hold the existing kernel lease across first entry or explicit resume."""
    if authorization_path is None:
        raise PermissionError('Explicit first formal test authorization is required')
    root=Path(root).resolve()
    plan_path=Path(plan_path or root/'outputs/phase8_3_3/formal_evaluation_plan.json')
    auth=read_json(authorization_path);plan=read_json(plan_path)
    if (auth.get('authorized') is not True or auth.get('action')!='FIRST_FORMAL_TEST_RUN'
            or auth.get('plan_sha256')!=file_sha(plan_path) or not auth.get('user_authorization_source')
            or plan.get('readiness')!='READY'):
        raise PermissionError('Explicit authorization and immutable READY plan required')
    from evi_hgt.formal_execution import formal_execution_lease
    with formal_execution_lease(root/plan['output_root']):
        return _run_formal_evaluation_locked(root,authorization_path=authorization_path,
            plan_path=plan_path,resume=resume)


def _run_formal_evaluation_locked(root, *, authorization_path, plan_path=None, resume=False):
    """The ONLY real-data entry. Phase 8.3.1 must never pass authorization."""
    if authorization_path is None:
        raise PermissionError('Explicit first formal test authorization is required')
    root=Path(root).resolve()
    auth=read_json(authorization_path)
    plan_path=Path(plan_path or root/'outputs/phase8_3_3/formal_evaluation_plan.json')
    plan=read_json(plan_path)
    plan_sha=file_sha(plan_path)
    if (auth.get('authorized') is not True or auth.get('action')!='FIRST_FORMAL_TEST_RUN'
            or auth.get('plan_sha256')!=plan_sha
            or not auth.get('user_authorization_source') or plan.get('readiness')!='READY'):
        raise PermissionError('Explicit authorization and immutable READY plan required')
    if auth.get('cells') != [cell['label'] for cell in plan['cells']]:
        raise PermissionError('authorization must bind the entire frozen matrix')
    if not plan.get('protected_files'):
        raise SharedInvariantError('frozen input hashes missing')
    for record in plan['protected_files']:
        if file_sha(root/record['path']) != record['sha256']:
            raise SharedInvariantError('frozen code/config/checkpoint mapping drift: '+record['path'])
    rates=_runtime_rates(root,plan)
    output=root/plan['output_root'];output.mkdir(parents=True,exist_ok=True)
    ledger_path=output/'formal_test_access.json'
    if plan.get('phase')=='8.4.0':
        protocol_path=root/plan['candidate_protocol_path']
        code_config=[{'path':item['path'],'sha256':item['sha256']} for item in plan['protected_files']
                     if item['path'].startswith(('src/','scripts/','configs/'))]
        expected_first_binding={
            'formal_plan_sha256':plan_sha,
            'matrix_sha256':plan['scientific_cells_sha256'],
            'candidate_protocol_sha256':file_sha(protocol_path),
            'metric_config_sha256':digest(read_json(protocol_path)['metrics']),
            'evaluator_code_config_sha256':digest(code_config),
            'checkpoint_sha256':{c['label']:c['checkpoint_sha256'] for c in plan['cells']},
            'split_sha256':{k:plan['data_binding']['split_sha256'][k]
                            for k in ('standard','cold_drug','cold_disease')},
            'cold_policy_digest':read_json(protocol_path)['cold_visibility_digest'],
            'determinism':{'torch_deterministic_algorithms':True,
                           'CUBLAS_WORKSPACE_CONFIG':':4096:8','dtype':'BF16',
                           'graph_batch_size':1},
        }
        run_identity=_phase840_formal_run_identity(
            root/'outputs/phase8_4_0/FIRST_FORMAL_ACCESS_LEDGER.json',expected_first_binding)
    else:
        run_identity=digest({'plan_sha256':plan_sha,'authorization_sha256':file_sha(authorization_path),
                             'cells':[cell['label'] for cell in plan['cells']]})
    ledger={'formal_test_count':1,'formal_test_accessed':True,
            'plan_sha256':plan_sha,'run_identity':run_identity}
    if resume:
        if not ledger_path.exists():
            raise PermissionError('resume requires the existing first-entry ledger; fresh run forbidden')
        try:
            if read_json(ledger_path)!=ledger:
                raise SharedInvariantError('formal resume ledger identity mismatch')
        except (ValueError,KeyError) as exc:
            raise SharedInvariantError('formal resume ledger is corrupt') from exc
        completion_path=output/'formal_execution_complete.json'
        if completion_path.exists():
            completion=read_json(completion_path)
            expected_hashes={cell['label']:file_sha(output/'cells'/cell['label']/'result.json')
                             for cell in plan['cells']}
            if (completion.get('run_identity')!=run_identity
                    or completion.get('formal_test_count')!=1
                    or completion.get('cell_result_sha256')!=expected_hashes
                    or completion.get('summary_sha256')!=file_sha(output/'summary.json')
                    or completion.get('exit_code') not in (0,2)):
                raise SharedInvariantError('completed formal run identity or outputs changed')
            print('FROZEN FORMAL RUN ALREADY FINALIZED; NO TEST RE-ENTRY',flush=True)
            return completion['exit_code']
    elif ledger_path.exists():
        raise PermissionError('First formal test run already entered; no automatic second run or retry')
    import polars as pl
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.use_deterministic_algorithms(True)
    import gc
    import platform
    from scripts.phase8_validation_diagnostics import load_replay_context, no_optimizer_allowed
    from scripts.phase8_validation_runtime import build_diagnostic_components
    from evi_hgt.training.formal_checkpoint import formal_checkpoint_identity_from_execution, inspect_formal_checkpoint_file
    from evi_hgt.training.formal_driver import _assemble_production_query
    from evi_hgt.training.supervision import make_unlabeled_query_supervision
    from evi_hgt.utils.seed import set_global_seed
    from types import SimpleNamespace
    from evi_hgt.training.input_index import StaticInputIndex
    from evi_hgt.training.candidate_universe import CandidateUniverse
    from evi_hgt.training.batcher import Phase3QueryBatchAssembler
    from evi_hgt.data.cold_visibility import ColdVisibility
    from evi_hgt.evaluation.indexed_adapter import IndexedEvaluationAssembler,build_cold_index
    from evi_hgt.evaluation.shared_scoring import SharedScoreStore
    # All checkpoint identities are inspected before the seal is opened.
    source_cells=[]
    for cell in plan['cells']:
        _,binding,execution=load_replay_context(root/plan['diagnostic_config'],cell['label'])
        source_cells.append({'label':binding['label'],'model_variant':execution.run['model_variant'],
            'split':execution.run['split'],'run_id':execution.run['run_id'],
            'checkpoint_sha256':file_sha(root/binding['best_checkpoint_path']),
            'checkpoint_identity':formal_checkpoint_identity_from_execution(execution),
            'cold_visibility_policy_digest':execution.run.get('cold_visibility_policy_digest')})
        inspect_formal_checkpoint_file(root/binding['best_checkpoint_path'],expected_role='BEST',
            expected_identity=formal_checkpoint_identity_from_execution(execution),map_location='cpu')
    validate_plan_bindings(plan,source_cells)
    # Exclusive durable entry record: an interrupted opening also counts as access.
    if not resume:
        _commit_first_access_ledger(ledger_path,ledger)
    results=[];start=time.monotonic();active_label=None
    environment={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'seed':42,'precision':'bf16_cuda_fp32_cpu','query_graph_batch_size':1,
        'determinism':'TORCH_DETERMINISTIC_ALGORITHMS_CUBLAS_4096_8; SAME_DEVICE_BITWISE_PREFLIGHT',
        'optimization':'EXISTING_STATIC_INDEX_NATIVE_CSR_SHARED_STANDARD_INPUTS_V1',
        'torch_cpu_threads':torch.get_num_threads()}
    try:
        base=root/'data/processed/phase3/v1'
        # These files may contain sealed queries. Hash/read only AFTER the
        # authorized, durable first-access marker above, never during preflight.
        if file_sha(base/'supervision/dda_supervision.parquet')!=plan['data_binding']['supervision_sha256']:
            raise SharedInvariantError('supervision manifest corruption')
        for split in {c['split'] for c in plan['cells']}:
            if file_sha(base/'splits'/(split+'.parquet'))!=plan['data_binding']['split_sha256'][split]:
                raise SharedInvariantError('split/query manifest corruption: '+split)
        splits={c['split'] for c in plan['cells']}
        count_source=pl.read_parquet(base/'supervision/dda_supervision.parquet',
            columns=['drug_id','disease_id','raw_relation'])
        count_frames={s:pl.read_parquet(base/'splits'/(s+'.parquet'),
            columns=['drug_id','disease_id','split_assignment','drug_partition','disease_partition']) for s in splits}
        count_drugs=tuple(sorted(count_source['drug_id'].unique().to_list()))
        _first_entry_projection(output,plan,rates,count_source,count_frames,count_drugs)
        del count_source,count_frames,count_drugs
        supervision=pl.read_parquet(base/'supervision/dda_supervision.parquet')
        drugs=tuple(sorted(supervision['drug_id'].unique().to_list()))
        split_frames={s:pl.read_parquet(base/'splits'/(s+'.parquet')) for s in splits}
        # The historical input-index builder reads full bindings. It is ONLY
        # called here, after the authorized first-access ledger above.
        index=StaticInputIndex.build(root)
        atomic_json(output/'input_index_startup.json',index.memory_startup_audit())
        cold_visibility=cold_index=None
        for split in dict.fromkeys(c['split'] for c in plan['cells']):
            cells=[c for c in plan['cells'] if c['split']==split];runtimes={}
            active_label=cells[0]['label']
            with no_optimizer_allowed():
                for cell in cells:
                    label=cell['label'];set_global_seed(42)
                    _,binding,execution=load_replay_context(root/plan['diagnostic_config'],label)
                    if not runtimes:
                        graph=execution.run['formal_training_protocol']['graph']
                        scalar=Phase3QueryBatchAssembler(root,**graph,
                            cold_visibility_policy_digest=execution.run.get('cold_visibility_policy_digest'))
                        scalar._universes[split]=CandidateUniverse.from_frames(supervision,split_frames[split],split_kind=split)
                        scalar._split_digests[split]=plan['data_binding']['split_sha256'][split]
                        heldout=sorted((*scalar._universes[split].heldout_drugs,*scalar._universes[split].heldout_diseases))
                        scalar._heldout_global_ids[split]=frozenset(map(int,index.node_global_ids(heldout)))
                        if split!='standard' and cold_visibility is None:
                            cold_visibility=ColdVisibility(root,execution.run['cold_visibility_policy_digest'])
                            cold_index=build_cold_index(index,cold_visibility)
                            atomic_json(output/'cold_index_startup.json',{'seconds':cold_index.cold_startup_seconds,
                                'additional_encoded_bytes':cold_index.cold_extra_bytes,'policy_digest':execution.run['cold_visibility_policy_digest']})
                        assembler=IndexedEvaluationAssembler(scalar,index if split=='standard' else cold_index,
                            split_frames={split:split_frames[split]},cold_visibility=cold_visibility if split!='standard' else None)
                    elif execution.run['formal_training_protocol']['graph']!=graph:
                        raise SharedInvariantError('shared standard graph parameters differ')
                    if runtimes:
                        reference=next(iter(runtimes.values())).execution.run['formal_training_protocol']
                        if execution.run['formal_training_protocol'].get('paths')!=reference.get('paths'):
                            raise SharedInvariantError('shared standard path parameters differ')
                    components=build_diagnostic_components(execution,assembler=assembler)
                    inspection=inspect_formal_checkpoint_file(root/binding['best_checkpoint_path'],expected_role='BEST',
                        expected_identity=formal_checkpoint_identity_from_execution(execution),map_location='cpu')
                    components.model_variant.model.load_state_dict(inspection.payload['model_state_dict'],strict=True)
                    runtimes[label]=SimpleNamespace(execution=execution,components=components)
                    del inspection
                # FULL supplies the already-frozen path construction for shared
                # standard inputs. Each head/model still uses its own weights.
                leader=next((r for r in runtimes.values() if r.execution.run['model_variant']=='FULL_EVIHGT'),None)
                if leader is None:raise SharedInvariantError('shared group lacks frozen FULL graph/path adapter')
                def assemble(drug,disease):
                    return _assemble_production_query(leader,make_unlabeled_query_supervision(drug,disease),partition='test')
                def score_batch(label,batch):
                    runtime=runtimes[label];components=runtime.components;device=torch.device(components.device)
                    path=batch.path_batch if runtime.execution.run['architecture']['use_path_reasoning'] else None
                    try:
                        with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                            prediction=components.model_variant.model(batch.model_batch,path_batch=path)
                        return tuple(float(getattr(prediction,c+'_logit').detach().float().cpu().reshape(())) for c in CHANNELS)
                    except (torch.cuda.OutOfMemoryError,FloatingPointError) as exc:
                        raise ModelEvaluationError(str(exc)) from exc
                store=SharedScoreStore(output/'shared_raw_scores'/split,
                    _shared_score_identity(root,plan,plan_sha,split,cells,
                        formal_run_id=run_identity if plan.get('phase')=='8.4.0' else None),
                    [c['label'] for c in cells])
                streams=[iter_disease_candidates(supervision,split_frames[split],split_kind=split,partition='test',
                    channel=c,formal_drug_ids=drugs) for c in CHANNELS]
                store.compute(streams,assemble=assemble,score_batch=score_batch)
            # Query score matrices are compact disk commits; only one disease
            # is read back at a time by the existing frozen metric/writer code.
            runtimes.clear();components=leader=runtime=None;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            for cell in cells:
                label=cell['label'];active_label=label;log=output/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
                result_path=output/'cells'/label/'result.json';had_result=result_path.exists()
                if label in store.failures:
                    failure=store.failures[label];result=failure_result(cell['model_variant'],split,failure['reason'],failure['log'])
                    log.write_text('STATUS=FAILED\n'+failure['reason']+'\nLOG='+failure['log']+'\n',encoding='utf-8')
                else:
                    result=evaluate_cell(supervision,split_frames[split],formal_drug_ids=drugs,split_kind=split,
                        partition='test',score_pair=lambda d,s:store.score(label,d,s),output=output/'cells'/label,
                        identity={'plan_sha256':plan_sha,'checkpoint_sha256':cell['checkpoint_sha256']},
                        model=cell['model_variant'],write_result=False)
                    log.write_text('STATUS=COMPLETE\nSHARED_INPUT_ASSEMBLY=YES\n',encoding='utf-8')
                for row in result['rows']:row['LOG PATH']=str(log)
                if had_result:
                    if not _existing_result_matches(read_json(result_path),result):
                        raise SharedInvariantError('completed cell result changed on resume')
                else:
                    atomic_json(result_path,result)
                results.append(result)
        write_outputs(output,results,metadata={'plan_sha256':plan_sha,
            'formal_test_count':1,'wall_seconds':time.monotonic()-start,'batch_size':1,'environment':environment})
        exit_code=0 if all(result['status']=='COMPLETE' for result in results) else 2
        atomic_json(output/'formal_execution_complete.json',{
            'run_identity':run_identity,'formal_test_count':1,'exit_code':exit_code,
            'summary_sha256':file_sha(output/'summary.json'),
            'cell_result_sha256':{cell['label']:file_sha(output/'cells'/cell['label']/'result.json')
                                  for cell in plan['cells']}})
    except Exception:
        write_abort_report(output,plan['cells'],results,active_label=active_label,reason=traceback.format_exc())
        raise
    return exit_code
