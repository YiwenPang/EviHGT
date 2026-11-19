"""Read sealed predictions after authorized unblinding; never construct a model.

Original result files, manifests, history and checkpoints are read-only inputs.
Metric recomputation checks previously persisted predictions, not new inference.
All post-hoc uncertainty is descriptive and cannot alter the frozen protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import polars as pl
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from evi_hgt.evaluation.formal_runtime import digest, secondary_metrics
from evi_hgt.evaluation.ranking import evaluate_disease_ranking
from evi_hgt.evaluation.protocol import EvaluationProtocolBinding

RUN_ID='EVIHGT-FORMAL-ad110867a0e6f6db-a6fdc76814484f4bb228673646a03e82'
PLAN_DIGEST='507a9e7aaa9e0a825f1e1d98b37f0585f526d0fbda8d62796fa3811743e721d5'
MANIFESTS={
 'A':'17d4278d4de3007fcf32d433b6931887ad4dbc2860df9ba3ede46dcecb878c54',
 'B':'de3e6f94a6119ac7b59cbc96414748bc8660d8304c1b6f65f580890749954d08',
 'C':'5fdd1363d6e2fc7e276b61ed248b368873dfb7673a65d6976e0949bcae6095ed',
 'D':'1b65a0a9e2109345e2d89b812eeb28c59eb7de4698f32f008ec0ffa278513bca'}
MODELS=['FULL_EVIHGT','BASE_HGT','NO_EVIDENCE_CONTEXT','NO_FOUR_VALUED_LOGIC',
        'NO_PATH_REASONING','NO_LOGIC_MASS_CONDITIONING']
RANK_FIELDS={'P':'known_positive_precision','R':'known_positive_recall','Hits':'hits_at_k','NDCG':'ndcg'}
METRICS=['roc_auc','average_precision']+[f'{name}@{k}' for k in (5,10) for name in RANK_FIELDS]
GRID=np.linspace(0,1,101)


def require(condition,message):
    if not condition:raise AssertionError(message)


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def require_unblinding(root):
    folder=Path(root)/'outputs/phase843_results'
    marker=read_json(folder/'FIRST_RESULTS_UNBLINDING.json')
    require(marker.get('authorized') is True,'first results unblinding not authorized')
    require(marker.get('closure_integrity_status')=='PASS','closure gate not PASS')
    require(marker.get('closure_integrity_sha256')==sha(folder/'closure_integrity.json'),'closure audit changed')
    closure=read_json(folder/'closure_integrity.json')
    require(closure.get('status')=='PASS' and closure.get('formal_test_count')==1,'invalid closure state/count')
    for value in (marker,closure):
        require(value.get('formal_run_id')==RUN_ID,'Formal run binding mismatch')
        require(value.get('execution_plan_digest')==PLAN_DIGEST,'execution plan mismatch')
        require(value.get('manifest_bindings')==MANIFESTS,'manifest binding mismatch')
    require(bool(marker.get('timestamp_utc')),'unblinding timestamp absent')
    return marker,closure


def recompute(query_id,ids,duals,positive,ks,split):
    require(len(ids)==len(duals) and len(ids)==len(set(ids)),'raw score alignment changed')
    require(positive<=set(ids),'positive membership missing')
    values=np.asarray(duals,dtype=np.float64)
    require(values.shape==(len(ids),2) and np.isfinite(values).all(),'nonfinite/invalid dual logits')
    score={d:float(v) for d,v in zip(ids,values[:,0],strict=True)}
    rank=evaluate_disease_ranking(binding=EvaluationProtocolBinding(split,'test','support'),disease_id=query_id,
        candidate_drug_ids=ids,scores=tuple(score.values()),relevant_drug_ids=positive)
    diagnostic=secondary_metrics(score,positive)
    result={'roc_auc':diagnostic['roc_auc'],'average_precision':diagnostic['average_precision'],
            'evaluable':rank.evaluable,'skip_reason':diagnostic['reason'],
            'ranking_payload':[r.to_payload() for r in rank.metrics if r.requested_k in ks],
            'ordered_ids':list(rank.ranked_drug_ids),'query_id':query_id,'candidate_count':len(ids)}
    metrics={r['requested_k']:r for r in result['ranking_payload']}
    for k in ks:
        for short,field in RANK_FIELDS.items():result[f'{short}@{k}']=metrics[k][field] if rank.evaluable else None
    if diagnostic['roc_auc'] is None:return result,None,None
    return result,np.asarray(diagnostic['roc']),np.asarray(diagnostic['pr'])


def verify_query_metrics(saved,computed,ks):
    require(saved.get('state')=='COMPLETE','query result is partial')
    require(saved['disease_id']==computed['query_id'] and saved['pairs']==computed['candidate_count'],'query census mismatch')
    require(saved['evaluable']==computed['evaluable'],'ranking eligibility changed')
    require(saved['ranking']==computed['ranking_payload'],'ranking metrics differ from sealed logits')
    require(saved['secondary']==dict(roc_auc=computed['roc_auc'],average_precision=computed['average_precision'],
                                   reason=computed['skip_reason']),'AUC/AP differ from sealed logits')


def verify_topk(ids,scores,sealed):
    ranked=[d for d,_ in sorted(zip(ids,scores,strict=True),key=lambda item:(-item[1],item[0]))[:50]]
    require(ranked==sealed,'sealed Top50 disagrees with raw score/canonical tie order')
    return ranked


def assert_common_membership(rows,models):
    memberships={m:{r['query_id'] for r in rows if r['model']==m and r['evaluable']} for m in models}
    require(all(memberships[m]==memberships[models[0]] for m in models),'models have different evaluable disease membership')
    return sorted(memberships[models[0]])


def paired_bootstrap(deltas,*,seed=20261003,resamples=10000,chunk=128,progress=False):
    """Identical query resample indices across all metrics and comparisons."""
    values=np.asarray(deltas,dtype=np.float64)
    require(values.ndim==2 and len(values)>0 and np.isfinite(values).all(),'invalid paired delta matrix')
    rng=np.random.Generator(np.random.PCG64(seed));n=len(values)
    means=np.empty((resamples,values.shape[1]),dtype=np.float64)
    iterator=range(0,resamples,chunk)
    if progress:iterator=tqdm(iterator,total=(resamples+chunk-1)//chunk,desc='[ANALYSIS][RUNNING] paired bootstrap',
                              dynamic_ncols=True,leave=True,mininterval=2,unit='chunk')
    for start in iterator:
        end=min(start+chunk,resamples);indices=rng.integers(0,n,size=(end-start,n))
        means[start:end]=values[indices].mean(axis=1)
    return np.quantile(means,[.025,.975],axis=0,method='linear')


def name_lookup(root):
    names={};paths={}
    for kind in ('drug','disease'):
        path=Path(root)/'data/processed/phase3/v1/partitions/nodes'/f'{kind}.parquet'
        paths[kind]={'path':str(path.relative_to(root)).replace('\\','/'),'sha256':sha(path)}
        lookup={}
        for row in pl.read_parquet(path,columns=['id','properties']).iter_rows(named=True):
            properties=json.loads(row['properties'])
            lookup[row['id']]={'name':properties.get('name') or '',
                'sources':properties.get('sources'),'source_ids':properties.get('source_ids')}
        names[kind]=lookup
    return names,paths


def save_table(folder,name,rows,*,parquet=False):
    require(bool(rows),f'empty table {name}')
    frame=pl.DataFrame(rows,infer_schema_length=None)
    with (folder/(name+'.csv')).open('x',encoding='utf-8',newline='') as handle:frame.write_csv(handle)
    if parquet:
        path=folder/(name+'.parquet');require(not path.exists(),'derived table overwrite forbidden');frame.write_parquet(path)
    return frame


def paired_analysis(rows,folder):
    ids=assert_common_membership(rows,MODELS)
    lookup={(r['model'],r['query_id']):r for r in rows}
    columns=[];vectors=[];per_query=[]
    for other in MODELS[1:]:
        for metric in METRICS:
            d=np.array([lookup['FULL_EVIHGT',q][metric]-lookup[other,q][metric] for q in ids])
            columns.append((other,metric));vectors.append(d)
            per_query.extend(dict(comparator=other,metric=metric,query_id=q,delta=float(v)) for q,v in zip(ids,d,strict=True))
    matrix=np.stack(vectors,axis=1)
    intervals=paired_bootstrap(matrix,progress=True)
    summary=[]
    for i,(other,metric) in enumerate(columns):
        d=matrix[:,i]
        summary.append(dict(comparator=other,metric=metric,n_queries=len(ids),mean_delta=float(statistics.fmean(d)),
            median_delta=float(np.median(d)),full_better=int((d>0).sum()),full_equal=int((d==0).sum()),
            full_worse=int((d<0).sum()),ci95_low=float(intervals[0,i]),ci95_high=float(intervals[1,i]),
            seed=20261003,resamples=10000,uncertainty_role='POST_HOC_DESCRIPTIVE; NOT_PROTOCOL_SELECTION'))
    save_table(folder,'stage_a_paired_deltas',summary)
    save_table(folder,'stage_a_paired_query_deltas',per_query,parquet=True)
    return summary,ids


def analyze(root=ROOT):
    root=Path(root).resolve();marker,closure=require_unblinding(root)
    print('[ANALYSIS][GATE] UNBLINDING_AUTHORIZATION = PASS',flush=True)
    plan_path=root/'outputs/phase842_evaluable/execution_plan.json'
    require(sha(plan_path)==closure['plan_sha256'],'plan file changed after closure')
    plan=read_json(plan_path);execution=root/plan['output_root']
    freeze=read_json(root/'outputs/phase842_evaluable/freeze.json')
    paths={'A':freeze['layer1_manifest_path'],**{stage:freeze['manifests'][split]['path']
           for stage,split in [('B','standard'),('C','cold_drug'),('D','cold_disease')]}}
    manifests={stage:read_json(root/path) for stage,path in paths.items()}
    require({s:m['global_manifest_digest'] for s,m in manifests.items()}==MANIFESTS,'manifest identity changed')
    expected_hashes=closure['protected_artifact_sha256'];reads={}
    def scientific_read(path):
        relative=str(path.relative_to(root)).replace('\\','/')
        require(relative in expected_hashes,'scientific input not covered by closure seal')
        actual=sha(path);require(actual==expected_hashes[relative],'scientific input seal changed')
        reads[relative]=actual;value=read_json(path)
        if 'payload' in value:require(digest(value['payload'])==value['sha256'],'embedded payload seal mismatch')
        return value
    folder=root/'outputs/phase843_results/data'
    folder.mkdir(parents=True,exist_ok=False)
    names,name_sources=name_lookup(root)
    all_per_query={};all_summary={};curves=[];tie_rows=[];top50=[];contingencies=[];raw_verified=0
    for stage in 'ABCD':
        print(f'[ANALYSIS][START] Stage {stage}: sealed predictions and metric integrity',flush=True)
        manifest=manifests[stage];queries=manifest.get('diseases',manifest.get('queries'))
        units=[u for u in plan['units'] if u['layer']==stage]
        unit_results={u['label']:scientific_read(execution/'units'/u['unit_id']/'result.json') for u in units}
        accum={u['label']:{'roc':[],'pr':[]} for u in units};stage_rows=[]
        if stage=='B':
            sealed_top=scientific_read(execution/'units'/units[0]['unit_id']/'top_k.json')
            top_rows={r['disease_id']:r for r in sealed_top['queries']}
        for query in tqdm(queries,desc=f'[ANALYSIS][RUNNING] Stage {stage}',unit='query',dynamic_ncols=True,leave=True,mininterval=2):
            q=query['disease_id'];filename=digest(q)+'.json';ids=query['candidate_drug_ids'];positive=set(query['positive_drug_ids'])
            raw=scientific_read(execution/'scores'/stage/filename)['payload']
            require(raw['candidate_ids']==ids and not raw['failures'],'sealed candidate order/failure changed')
            require(set(raw['scores'])=={u['label'] for u in units},'sealed model set mismatch')
            require(raw['binding']['disease']==q,'sealed raw-score query mismatch')
            for unit in units:
                label=unit['label'];model=unit['model_variant'];ks=unit['ranking_k'];duals=raw['scores'][label]
                computed,roc,pr=recompute(q,ids,duals,positive,ks,unit['split'])
                saved=scientific_read(execution/'units'/unit['unit_id']/'queries'/filename)['payload']
                verify_query_metrics(saved,computed,ks);raw_verified+=1
                row=dict(stage=stage,model=model,query_id=q,disease_id=q,disease_name=names['disease'].get(q,{}).get('name',''),
                    positive_count=len(positive),u_count=len(ids)-len(positive),candidate_count=len(ids),
                    evaluable=computed['evaluable'],skip_reason=computed['skip_reason'],stratum=query.get('stratum'),
                    benchmark=unit['benchmark'],manifest_digest=MANIFESTS[stage],
                    roc_auc=computed['roc_auc'],average_precision=computed['average_precision'])
                for k in ks:
                    for short in RANK_FIELDS:row[f'{short}@{k}']=computed[f'{short}@{k}']
                stage_rows.append(row)
                if roc is not None:accum[label]['roc'].append(roc);accum[label]['pr'].append(pr)
                scores=np.asarray(duals,dtype=np.float64)[:,0];unique,counts=np.unique(scores,return_counts=True)
                tie_rows.append(dict(stage=stage,model=model,query_id=q,n_candidates=len(ids),
                    n_unique_support_logits=len(unique),constant_score=len(unique)==1,largest_tie_fraction=float(counts.max()/len(ids)),
                    positive_prevalence=len(positive)/len(ids),evaluable=computed['evaluable'],
                    support_logit_min=float(scores.min()),support_logit_max=float(scores.max())))
                for rec in computed['ranking_payload']:
                    contingencies.append(dict(stage=stage,model=model,query_id=q,k=rec['requested_k'],effective_k=rec['effective_k'],
                        retrieved_known_positive=rec['hit_count'],retrieved_unlabeled=rec['effective_k']-rec['hit_count'],
                        missed_known_positive=len(positive)-rec['hit_count'],
                        interpretation='RETRIEVAL_MEMBERSHIP_COUNTS; UNLABELED_NOT_CONFIRMED_NEGATIVE'))
                if stage=='B':
                    ordered=verify_topk(ids,scores,top_rows[q]['drug_ids']);score_lookup=dict(zip(ids,scores,strict=True))
                    for rank,d in enumerate(ordered,start=1):
                        name=names['drug'].get(d,{})
                        top50.append(dict(disease_id=q,disease_name=row['disease_name'],stratum=row['stratum'],rank=rank,
                            drug_id=d,drug_name=name.get('name',''),support_logit=float(score_lookup[d]),known_formal_positive=d in positive,
                            membership='KNOWN_FORMAL_POSITIVE' if d in positive else 'UNLABELED_CANDIDATE',
                            candidate_provenance=json.dumps({'sources':name.get('sources'),'source_ids':name.get('source_ids')},ensure_ascii=False,sort_keys=True),
                            candidate_count=len(ids),manifest_digest=MANIFESTS[stage]))
        summaries=[]
        for unit in units:
            label=unit['label'];model=unit['model_variant'];items=[r for r in stage_rows if r['model']==model]
            eligible=[r for r in items if r['roc_auc'] is not None];ranking=[r for r in items if r['evaluable']]
            result=unit_results[label];ks=unit['ranking_k'];metric_names=['roc_auc','average_precision']+[f'{n}@{k}' for k in ks for n in RANK_FIELDS]
            macro={metric:statistics.fmean(r[metric] for r in (eligible if metric in ('roc_auc','average_precision') else ranking))
                   for metric in metric_names}
            require(macro==result['macro'],'persisted macro differs from exact per-query aggregation')
            require(result['disease_count']==len(items) and result['pairs']==sum(r['candidate_count'] for r in items),'persisted census changed')
            require(result['secondary_evaluable']==len(eligible) and result['secondary_skipped']==len(items)-len(eligible),'persisted skip census changed')
            require(result['ranking_evaluable']==len(ranking),'persisted ranking evaluability changed')
            require(len(items)==unit['diseases'] and sum(r['candidate_count'] for r in items)==unit['pairs'],'plan query/pair census changed')
            skip=dict(Counter(r['skip_reason'] for r in items if r['roc_auc'] is None))
            summary=dict(stage=stage,model=model,benchmark=unit['benchmark'],n_total=len(items),n_evaluable=len(eligible),
                n_skipped=len(items)-len(eligible),ranking_evaluable=len(ranking),candidate_pairs=unit['pairs'],
                positives=sum(r['positive_count'] for r in items),unlabeled=sum(r['u_count'] for r in items),
                skip_reasons=json.dumps(skip,sort_keys=True),manifest_digest=MANIFESTS[stage],**macro)
            summaries.append(summary)
            for kind,metric in [('roc','roc_auc'),('pr','average_precision')]:
                # Match frozen display convention exactly, including repeated x at tied thresholds.
                arrays=accum[label][kind];mean=[statistics.fmean(float(a[i]) for a in arrays) for i in range(101)]
                curves.extend(dict(stage=stage,model=model,curve=kind,x=float(x),y=float(y),n_evaluable=len(eligible),
                    macro_metric=macro[metric]) for x,y in zip(GRID,mean,strict=True))
        all_per_query[stage]=stage_rows;all_summary[stage]=summaries
        if stage=='A':
            common=assert_common_membership(stage_rows,MODELS)
            require(len(common)==summaries[0]['n_evaluable'],'common membership census changed')
            save_table(folder,'stage_a_model_summary',summaries)
            save_table(folder,'stage_a_per_disease_metrics',stage_rows,parquet=True)
        elif stage=='B':
            save_table(folder,'stage_b_per_disease_metrics',stage_rows)
            save_table(folder,'stage_b_subset_summary',summaries)
            save_table(folder,'stage_b_top50_rankings',top50)
            save_table(folder,'stage_b_repurposing_candidate_shortlist',[r for r in top50 if r['rank']<=20])
        else:
            prefix='cold_drug' if stage=='C' else 'cold_disease'
            save_table(folder,prefix+'_summary',summaries)
            save_table(folder,prefix+'_per_query_metrics',stage_rows,parquet=True)
        print(f'[ANALYSIS][DONE] Stage {stage}: exact frozen-metric verification PASS; {len(stage_rows)} model-query rows',flush=True)
    paired,common=paired_analysis(all_per_query['A'],folder)
    save_table(folder,'macro_curves',curves)
    save_table(folder,'score_tie_audit',tie_rows,parquet=True)
    save_table(folder,'topk_retrieval_membership_counts',contingencies,parquet=True)
    tie_summary=[]
    for stage in 'ABCD':
        for summary in all_summary[stage]:
            items=[r for r in tie_rows if r['stage']==stage and r['model']==summary['model']]
            eligible=[r for r in items if r['evaluable']]
            tie_summary.append(dict(stage=stage,model=summary['model'],queries=len(items),
                constant_score_queries=sum(r['constant_score'] for r in items),constant_score_evaluable=sum(r['constant_score'] for r in eligible),
                n_unique_support_logits_min=min(r['n_unique_support_logits'] for r in items),
                n_unique_support_logits_median=float(np.median([r['n_unique_support_logits'] for r in items])),
                largest_tie_fraction_mean=statistics.fmean(r['largest_tie_fraction'] for r in items),
                positive_prevalence_macro_evaluable=statistics.fmean(r['positive_prevalence'] for r in eligible)))
    save_table(folder,'score_tie_summary',tie_summary)
    methods=dict(analysis_protocol='SEALED_FORMAL_RESULTS_ANALYSIS_V1',score='RAW_SUPPORT_LOGIT',
        metric_recomputation='EXACT_FROM_ALREADY_SEALED_PREDICTIONS; NO_MODEL_FORWARD',
        raw_prediction_artifacts_copied=False,macro='UNWEIGHTED_MEAN_OVER_EVALUABLE_DISEASE_QUERY_UNITS',
        skip='RANKING: NO_HELD_OUT_RELEVANCE; ROC/AP: NO_P_OR_NO_PROTOCOL_U; SKIPS_NOT_ZERO_FILLED',
        cold_unit='EXISTING_DISEASE_WISE_QUERY_UNITS_WITH_COLD_QUERY_SIDE_INFORMATION_V1',
        curve_grid=GRID.tolist(),curve_aggregation='FROZEN_SECONDARY_METRICS: TIED_THRESHOLDS; NUMPY_INTERP_EACH_QUERY; UNWEIGHTED_MACRO',
        roc_area='EXACT_EMPIRICAL_TIED_THRESHOLD_TRAPEZOID; NOT_AREA_UNDER_GRID_MACRO_DISPLAY',
        ap='NON_INTERPOLATED_SUM_DELTA_RECALL_TIMES_PRECISION; NOT_TRAPEZOIDAL_PR_AREA',
        curve_smoothing='NONE',tie_break='SCORE_DESCENDING_CANONICAL_DRUG_ID_ASCENDING',
        bootstrap=dict(seed=20261003,resamples=10000,generator='NUMPY_PCG64',unit='DISEASE_QUERY',
            identical_indices_across_models_and_metrics=True,ci='PERCENTILE_95; NUMPY_QUANTILE_LINEAR',
            role='POST_HOC_DESCRIPTIVE_UNCERTAINTY',training_seed_uncertainty_included=False),
        confusion_matrix='NOT_SCIENTIFICALLY_DEFINED',
        threshold_audit='NO_FROZEN_FORMAL_CLASSIFICATION_THRESHOLD; VALIDATION_LOGIT_ZERO_CONTINGENCY_IS_VALIDATION_ONLY',
        binary_ground_truth='UNAVAILABLE; U_IS_NOT_CONFIRMED_NEGATIVE',
        no_result_dependent_protocol_changes=True,name_sources=name_sources,
        retrieval_contingency='KNOWN_POSITIVE_RETRIEVED / UNLABELED_RETRIEVED / KNOWN_POSITIVE_MISSED; NOT_BINARY_CONFUSION_MATRIX',
        stage_b_shortlist='ALL_SIX_DISEASES_TOP20_WITH_P_AND_U_MEMBERSHIP; NOT_EXTERNALLY_VALIDATED')
    write_json(folder/'analysis_methods.json',methods)
    result=dict(status='COMPLETE',timestamp_utc=datetime.now(timezone.utc).isoformat(),formal_run_id=RUN_ID,formal_test_count=1,
        execution_plan_digest=PLAN_DIGEST,manifest_bindings=MANIFESTS,unblinding_timestamp=marker['timestamp_utc'],
        closure_integrity_sha256=sha(root/'outputs/phase843_results/closure_integrity.json'),
        stage_summaries=all_summary,stage_b_per_disease=all_per_query['B'],paired_comparisons=paired,
        score_tie_summary=tie_summary,stage_b_top_ranked_unlabeled=[next(r for r in top50 if r['disease_id']==q and not r['known_formal_positive'])
            for q in sorted({r['disease_id'] for r in top50})],
        common_stage_a_evaluable_queries=len(common),common_stage_a_evaluable_digest=digest(common),
        exact_raw_prediction_metric_checks=raw_verified,exact_metric_recomputation='PASS',
        model_forward_reruns=0,formal_rescoring=False,training=False,confusion_matrix='NOT_SCIENTIFICALLY_DEFINED')
    write_json(folder/'analysis_summary.json',result)
    write_json(folder/'sealed_scientific_input_inventory.json',reads)
    write_json(folder/'analysis_artifact_inventory.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    print(f'[ANALYSIS][GATE] EXACT_PERSISTED_METRIC_RECOMPUTATION = PASS; model-query checks={raw_verified}',flush=True)
    print('[ANALYSIS][DONE] data package committed; model forwards=0; Formal Test Count=1',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();analyze(args.root)
