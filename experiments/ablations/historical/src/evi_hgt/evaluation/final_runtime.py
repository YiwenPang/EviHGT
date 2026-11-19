"""Future authorized execution of the final frozen scope. Never called by freeze.

Candidate/reporting adapters only. Original scalar graph assembly, checkpoint
loader, dual forward, precision and query-boundary recovery are reused unchanged.
Real execution needs a new explicit grant bound to this final freeze, not the old
Amendment002 grant. This module's tests use synthetic scores only.
"""
from copy import deepcopy
from dataclasses import replace
import gc
import math
import os
from pathlib import Path
import statistics

from evi_hgt.evaluation.formal_redesign import FORMAL_RUN_ID,digest,exclusive_json,file_sha,read_json,require
from evi_hgt.evaluation.final_protocol import candidate_stream,validate_plan


def preflight(root):
    from scripts import phase842_final_freeze as freeze
    require(Path(root).resolve()==freeze.ROOT,'final scope must use its bound project root')
    freeze.preflight()
    f=read_json(Path(root)/freeze.OUT/'freeze.json');record,manifests,plan=freeze.load_draft()
    f['freeze_sha256']=file_sha(Path(root)/freeze.OUT/'freeze.json')
    layer1=read_json(Path(root)/f['layer1_manifest_path'])
    return f,plan,layer1,manifests


def safety_plan(plan,manifests):
    """IO-only projection for the existing recovery class; frozen plan stays intact."""
    projected=deepcopy(plan)
    projected['full_manifests']={s:dict(diseases=m['queries'],pairs=m['totals']['pairs'],queries=m['totals']['queries'],
        global_manifest_digest=m['global_manifest_digest']) for s,m in manifests.items()}
    return projected


def streams(manifest):
    if 'diseases' in manifest:
        from evi_hgt.evaluation.redesigned_runtime import comparison_candidates
        return comparison_candidates(manifest)
    return candidate_stream(manifest)


def evaluate_frozen(manifest,*,score_pair,model,output,identity):
    """Future scientific writer. Original metric functions, frozen scope/K only."""
    from evi_hgt.evaluation.formal_runtime import secondary_metrics,atomic_json
    from evi_hgt.evaluation.protocol import EvaluationProtocolBinding
    from evi_hgt.evaluation.ranking import evaluate_disease_ranking
    from tqdm import tqdm
    output=Path(output);spec=manifest['protocol'];ks=spec['ranking_k']
    sampled='diseases' in manifest
    benchmark='SAMPLED COMPARISON BENCHMARK' if sampled else spec['benchmark']
    rows=manifest['diseases'] if sampled else manifest['queries']
    identity={**identity,'model':model,'benchmark':benchmark,'manifest_digest':manifest['global_manifest_digest'],
        'manifest_protocol_digest':manifest['protocol_digest'],'ranking_k':ks}
    binding=EvaluationProtocolBinding(spec['split'],'test','support');records=[]
    for source in tqdm(streams(manifest),total=len(rows),desc='[FROZEN_RESULT_COMMIT][RUNNING]',unit='query',
            dynamic_ncols=True,leave=True,mininterval=2):
        path=output/'queries'/(digest(source.disease_id)+'.json')
        expected=dict(identity=identity,definition_digest=source.definition_digest,candidate_order_digest=digest(source.candidate_drug_ids))
        if path.exists():
            saved=read_json(path);payload=saved['payload']
            require(saved['sha256']==digest(payload) and payload['binding']==expected and
                payload['state']=='COMPLETE' and payload['pairs']==len(source.candidate_drug_ids),'query result recovery binding/count changed')
            records.append(payload);continue
        scores={}
        for drug in source.candidate_drug_ids:
            dual=score_pair(drug,source.disease_id)
            require(len(dual)==2 and all(math.isfinite(float(v)) for v in dual),'invalid unchanged dual prediction')
            scores[drug]=float(dual[0])
        ranking=evaluate_disease_ranking(binding=binding,disease_id=source.disease_id,candidate_drug_ids=source.candidate_drug_ids,
            scores=tuple(scores.values()),relevant_drug_ids=source.held_out_relevant_drug_ids)
        secondary=secondary_metrics(scores,set(source.held_out_relevant_drug_ids))
        payload=dict(binding=expected,disease_id=source.disease_id,state='COMPLETE',pairs=len(source.candidate_drug_ids),
            evaluable=ranking.evaluable,ranking=[r.to_payload() for r in ranking.metrics if r.requested_k in ks],
            secondary={k:secondary[k] for k in ('roc_auc','average_precision','reason')})
        atomic_json(path,dict(payload=payload,sha256=digest(payload)));records.append(payload)
    macro={};eligible=[r for r in records if r['evaluable']]
    secondary=[r['secondary'] for r in records if r['secondary']['roc_auc'] is not None]
    for key in ('roc_auc','average_precision'):
        macro[key]=statistics.fmean(s[key] for s in secondary) if secondary else None
    for name,field in [('P','known_positive_precision'),('R','known_positive_recall'),('Hits','hits_at_k'),('NDCG','ndcg')]:
        for i,k in enumerate(ks):macro[f'{name}@{k}']=statistics.fmean(r['ranking'][i][field] for r in eligible) if eligible else None
    require(len(records)==len(rows) and sum(r['pairs'] for r in records)==manifest['totals']['pairs'],'scientific result scope incomplete')
    result=dict(identity=identity,model=model,benchmark=benchmark,ranking_k=ks,primary_metrics=['roc_auc','average_precision'],
        macro=macro,unlabeled_is_negative=False,disease_count=len(records),pairs=sum(r['pairs'] for r in records),
        ranking_evaluable=len(eligible),secondary_evaluable=len(secondary),secondary_skipped=len(records)-len(secondary),
        aggregation=spec['aggregation'],population_wide_full_retrieval=False,status='COMPLETE')
    target=output/'result.json'
    if target.exists():require(read_json(target)==result,'immutable completed result changed')
    else:atomic_json(target,result)
    return result


def run(root,*,authorization_path=None,plan_path=None,resume=False):
    if authorization_path is None:raise PermissionError('Final freeze requires new separate user authorization')
    auth=read_json(authorization_path)
    if auth.get('authorized') is not True or auth.get('action')!='FINAL_FOUR_STAGE_FORMAL_EXECUTION':
        raise PermissionError('Old execution grants are superseded; final four-stage grant required')
    root=Path(root).resolve();f,plan,layer1,manifests=preflight(root)
    if not (auth.get('formal_run_id')==FORMAL_RUN_ID and auth.get('formal_test_count')==1 and
        auth.get('plan_sha256')==f['plan_sha256'] and auth.get('plan_digest')==f['execution_plan_digest'] and
        auth.get('freeze_sha256')==f['freeze_sha256'] and auth.get('manifest_bindings')==plan['manifest_bindings'] and
        auth.get('units')==[u['unit_id'] for u in plan['units']] and auth.get('user_authorization_source')):
        raise PermissionError('Final grant must bind complete plan/freeze/manifests/units and existing Formal state')
    selected=Path(plan_path) if plan_path else root/f['plan_path']
    if not selected.is_absolute():selected=root/selected
    require(file_sha(selected)==f['plan_sha256'],'selected final plan file differs')
    validate_plan(plan,expected_digest=f['execution_plan_digest'])
    from evi_hgt.formal_execution import formal_execution_lease
    from evi_hgt.evaluation.execution_safety import ExecutionSafety
    projected=safety_plan(plan,manifests)
    safety=ExecutionSafety(root,projected,f,comparison=layer1,extension_digest=f['execution_plan_digest'])
    safety.status_path=root/'outputs/phase842_final/execution_status.json'
    with formal_execution_lease(root/plan['output_root']),safety.installed():
        return _execute(root,projected,layer1,manifests,f,auth,safety,resume=resume)


def _execute(root,plan,layer1,manifests,f,auth,safety,*,resume):
    """Original strict graph/dual-forward path; only manifests and reporting differ."""
    output=root/plan['output_root'];marker=output/'final_execution_access.json'
    identity=dict(formal_run_id=FORMAL_RUN_ID,formal_test_count=1,plan_sha256=f['plan_sha256'],freeze_sha256=f['freeze_sha256'],
        checkpoint_binding_digest=plan['checkpoint_binding_digest'],source_data_identity=plan['data_binding'],
        protocol_digest=plan['protocol_digest'],manifest_bindings=plan['manifest_bindings'],
        units=[u['unit_id'] for u in plan['units']],graph_batch_size=1)
    if resume:require(marker.exists() and read_json(marker)==identity,'resume requires same final immutable execution entry')
    else:
        require(not marker.exists(),'final attempt already entered; fresh retries forbidden')
        safety.sealed_json(marker,identity)
    complete=output/'execution_complete.json'
    if complete.exists():
        saved=read_json(complete);require(saved['identity']==identity,'final completion identity changed')
        for p,sha in saved['result_hashes'].items():require(file_sha(output/p)==sha,'completed final result bytes changed')
        safety.validate_complete();return 0
    # All runtime imports occur after the new authorization and immutable entry.
    import polars as pl
    import torch
    from scripts.phase8_validation_diagnostics import no_optimizer_allowed
    from scripts.phase841_indexed_fact_columns import IndexedFactColumns
    from evi_hgt.training.input_index import StaticInputIndex
    from evi_hgt.training.formal_driver import _assemble_production_query
    from evi_hgt.training.supervision import make_unlabeled_query_supervision
    from evi_hgt.data.cold_visibility import ColdVisibility
    from evi_hgt.evaluation.indexed_adapter import build_cold_index
    from evi_hgt.evaluation.shared_scoring import SharedScoreStore
    from evi_hgt.evaluation.redesigned_runtime import load_group
    from tqdm import tqdm
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(),'frozen CUDA runtime unavailable')
    base=root/'data/processed/phase3/v1';supervision=pl.read_parquet(base/'supervision/dda_supervision.parquet')
    frames={s:pl.read_parquet(base/'splits'/f'{s}.parquet') for s in ('standard','cold_drug','cold_disease')}
    results=[];hashes={};cold_visibility=cold_index=None
    with no_optimizer_allowed():
        with safety.visible_stage('INDEX_STARTUP'):
            original=StaticInputIndex.build(root);columns=IndexedFactColumns.from_raw(original._fact_raw)
            index=replace(original,_fact_raw=columns)
        for stage,split,cells,manifest in [('A','standard',plan['cells'][:6],layer1),('B','standard',[plan['cells'][0]],manifests['standard']),
                ('C','cold_drug',[plan['cells'][6]],manifests['cold_drug']),('D','cold_disease',[plan['cells'][7]],manifests['cold_disease'])]:
            if split!='standard' and cold_index is None:
                with safety.visible_stage('COLD_INDEX_STARTUP'):
                    cold_visibility=ColdVisibility(root,plan['cold_policy_digest']);cold_index=build_cold_index(index,cold_visibility)
            with safety.visible_stage(stage+'_MODEL_SETUP'):
                runtimes=load_group(root,plan,cells,index if split=='standard' else cold_index,supervision,frames[split],
                    cold_visibility=cold_visibility if split!='standard' else None)
            leader=runtimes[cells[0]['label']]
            def assemble(drug,disease):
                return _assemble_production_query(leader,make_unlabeled_query_supervision(drug,disease),partition='test')
            def score_batch(label,batch):
                runtime=runtimes[label];components=runtime.components;device=torch.device(components.device)
                paths=batch.path_batch if runtime.execution.run['architecture']['use_path_reasoning'] else None
                with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    prediction=components.model_variant.model(batch.model_batch,path_batch=paths)
                return tuple(float(getattr(prediction,c+'_logit').detach().float().cpu().reshape(()))
                    for c in ('support','unsuitability'))
            group_identity={**identity,'group':stage,'split':split,'candidate_digest':manifest['global_manifest_digest']}
            store=SharedScoreStore(output/'scores'/stage,group_identity,[c['label'] for c in cells])
            with tqdm(total=manifest['totals']['pairs'],desc=f'[PHASE][RUNNING] Layer {stage}',unit='pair',
                    dynamic_ncols=True,leave=True,mininterval=5) as bar:
                def counted_assemble(drug,disease):
                    batch=assemble(drug,disease);bar.update(1);return batch
                store.compute([streams(manifest),streams(manifest)],assemble=counted_assemble,score_batch=score_batch)
            require(not store.failures,'model scoring failure; final amended attempt interrupted')
            runtimes.clear();leader=None;gc.collect();torch.cuda.empty_cache()
            for cell in cells:
                unit_id=stage+'_'+cell['label'];destination=output/'units'/unit_id
                result=evaluate_frozen(manifest,score_pair=lambda d,s:store.score(cell['label'],d,s),model=cell['model_variant'],
                    output=destination,identity={**group_identity,'checkpoint_identity':cell['checkpoint_identity'],
                        'checkpoint_sha256':cell['checkpoint_sha256']})
                results.append(dict(unit_id=unit_id,layer=stage,benchmark=result['benchmark'],result=result))
                path=destination/'result.json';hashes[path.relative_to(output).as_posix()]=file_sha(path)
            safety.finish_stage(stage)
            print(f'[PHASE][DONE] Layer {stage}; units={len(cells)}; scientific values sealed on disk',flush=True)
        del original,index,columns,cold_index,cold_visibility
    summary=output/'final_formal_summary.json';safety.sealed_json(summary,dict(identity=identity,units=results))
    hashes[summary.relative_to(output).as_posix()]=file_sha(summary)
    safety.sealed_json(complete,dict(identity=identity,result_hashes=hashes,status='COMPLETE'))
    print('[PHASE][GATE] FORMAL_EXECUTION_COMPLETE = PASS; scientific values remain undisplayed',flush=True)
    return 0


def dispatch_main(argv,*,project_root):
    """Scoped dispatch through unchanged main CLI; no file/source overrides."""
    from unittest.mock import patch
    import evi_hgt.evaluation.formal_redesign as protocol
    import evi_hgt.evaluation.redesigned_runtime as old
    from evi_hgt.evaluation.final_protocol import PLAN_FORMAT
    from evi_hgt.cli import main
    with patch.object(protocol,'PLAN_FORMAT',PLAN_FORMAT),patch.object(old,'run',run):
        return main(argv,project_root=project_root)
