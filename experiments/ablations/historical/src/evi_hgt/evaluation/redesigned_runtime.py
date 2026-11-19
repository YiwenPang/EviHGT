"""Three-layer evaluation entry. Real execution requires a separate explicit grant.

Freeze/preflight never call this scoring path. The original scalar graph batching,
dual forward, checkpoint loader, metric algorithms and Cold mask logic are reused.
"""
from contextlib import contextmanager
from dataclasses import replace
import gc
import math
import os
from pathlib import Path
import statistics
import threading
import time
from types import SimpleNamespace

from evi_hgt.evaluation.formal_redesign import (
    FORMAL_RUN_ID, PLAN_PATH, digest, exclusive_json, file_sha, read_json, require, validate_checkpoint_context,
)


def preflight(root):
    from evi_hgt.evaluation.redesign_freeze import preflight as verify
    return verify(root)


def run(root, *, authorization_path=None, plan_path=None, resume=False):
    if authorization_path is None:
        raise PermissionError('Separate user authorization for THREE_LAYER_FORMAL_EXECUTION required')
    # No model/data runtime import precedes this gate.
    auth=read_json(authorization_path)
    if auth.get('authorized') is not True or auth.get('action')!='THREE_LAYER_FORMAL_EXECUTION':
        raise PermissionError('This task freezes readiness only; future execution requires explicit authorization')
    frozen,plan,manifest=preflight(root)
    if (auth.get('formal_run_id')!=FORMAL_RUN_ID or auth.get('formal_test_count')!=1
            or auth.get('plan_sha256')!=frozen['plan_sha256'] or auth.get('freeze_sha256')!=frozen['freeze_sha256']
            or auth.get('units')!=[u['unit_id'] for u in plan['units']] or not auth.get('user_authorization_source')):
        raise PermissionError('Authorization must bind existing Formal run/count, frozen plan, freeze and all nine units')
    root=Path(root).resolve()
    selected=Path(plan_path) if plan_path is not None else root/PLAN_PATH
    if not selected.is_absolute(): selected=root/selected
    require(file_sha(selected)==frozen['plan_sha256'],'execution plan path/hash changed')
    from evi_hgt.formal_execution import formal_execution_lease
    with formal_execution_lease(root/plan['output_root']):
        return _execute(root,plan,manifest,frozen,auth,resume=resume)


def comparison_candidates(manifest):
    from evi_hgt.evaluation.candidates import DiseaseCandidateSet
    for row in manifest['diseases']:
        yield DiseaseCandidateSet(split_kind='standard',partition='test',channel='support',
            disease_id=row['disease_id'],candidate_drug_ids=tuple(row['candidate_drug_ids']),
            held_out_relevant_drug_ids=tuple(row['positive_drug_ids']),training_known_excluded_drug_ids=(),
            candidate_protocol_digest=manifest['protocol_digest'],
            candidate_membership_digest=row['candidate_order_digest'],definition_digest=row['candidate_digest'])


def evaluate_comparison(manifest, *, score_pair, model, output, identity):
    """Future result writer. Synthetic tests only in the implementation task.

    Primary AP/ROC and optional ranking use the original mathematical functions;
    @50 stays internal to that function and is omitted from sampled result tables.
    """
    from evi_hgt.evaluation.formal_runtime import atomic_json, secondary_metrics
    from evi_hgt.evaluation.protocol import EvaluationProtocolBinding
    from evi_hgt.evaluation.ranking import evaluate_disease_ranking
    from tqdm import tqdm
    output=Path(output); records=[];binding=EvaluationProtocolBinding('standard','test','support')
    identity={**identity,'model':model,'benchmark':'SAMPLED COMPARISON BENCHMARK',
        'manifest_digest':manifest['global_manifest_digest'],'protocol_digest':manifest['protocol_digest']}
    for source in tqdm(comparison_candidates(manifest),total=manifest['totals']['diseases'],
            desc=f'[COMPARISON:{model}][RUNNING]',unit='disease',dynamic_ncols=True,leave=True,mininterval=2):
        path=output/'queries'/(digest(source.disease_id)+'.json')
        expected=dict(identity=identity,definition_digest=source.definition_digest)
        if path.exists():
            saved=read_json(path)
            require(saved['sha256']==digest(saved['payload']) and saved['payload']['binding']==expected,
                'sampled result identity changed')
            records.append(saved['payload']);continue
        scores={}
        for drug in source.candidate_drug_ids:
            dual=score_pair(drug,source.disease_id)
            require(len(dual)==2 and all(math.isfinite(float(v)) for v in dual),'invalid dual prediction')
            scores[drug]=float(dual[0])
        ranking=evaluate_disease_ranking(binding=binding,disease_id=source.disease_id,
            candidate_drug_ids=source.candidate_drug_ids,scores=tuple(scores.values()),
            relevant_drug_ids=source.held_out_relevant_drug_ids)
        secondary=secondary_metrics(scores,set(source.held_out_relevant_drug_ids))
        payload=dict(binding=expected,disease_id=source.disease_id,evaluable=ranking.evaluable,
            ranking=[r.to_payload() for r in ranking.metrics if r.requested_k in (5,10,20)],
            secondary={k:secondary[k] for k in ('roc_auc','average_precision','reason')})
        atomic_json(path,dict(payload=payload,sha256=digest(payload)));records.append(payload)
    macro={}; eligible=[r for r in records if r['evaluable']]
    secondary=[r['secondary'] for r in records if r['secondary']['roc_auc'] is not None]
    for key in ('roc_auc','average_precision'):
        macro[key]=statistics.fmean(s[key] for s in secondary) if secondary else None
    for name,field in [('P','known_positive_precision'),('R','known_positive_recall'),('Hits','hits_at_k'),('NDCG','ndcg')]:
        for i,k in enumerate((5,10,20)):
            macro[f'{name}@{k}']=statistics.fmean(r['ranking'][i][field] for r in eligible) if eligible else None
    result=dict(identity=identity,model=model,benchmark='SAMPLED COMPARISON BENCHMARK',ranking_k=[5,10,20],
        primary_metrics=['roc_auc','average_precision'],macro=macro,unlabeled_is_negative=False,
        disease_count=len(records),ranking_evaluable=len(eligible),secondary_evaluable=len(secondary),
        secondary_skipped=len(records)-len(secondary),status='COMPLETE')
    target=output/'result.json'
    if target.exists():require(read_json(target)==result,'completed comparison result changed')
    else:atomic_json(target,result)
    return result


@contextmanager
def visible_stage(name):
    """Bounded-stage heartbeat, stopped on success/failure; no polling external jobs."""
    stopped=threading.Event();start=time.monotonic()
    def heartbeat():
        while not stopped.wait(30):print(f'[{name}][RUNNING] elapsed_seconds={time.monotonic()-start:.0f}',flush=True)
    print(f'[{name}][START]',flush=True);thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:yield
    finally:
        stopped.set();thread.join()
        print(f'[{name}][DONE] elapsed_seconds={time.monotonic()-start:.1f}',flush=True)


def load_group(root,plan,cells,index,supervision,frame,*,cold_visibility=None):
    from scripts.formal_study_validation_diagnostics import load_replay_context
    from scripts.formal_study_validation_runtime import build_diagnostic_components
    from evi_hgt.training.formal_checkpoint import formal_checkpoint_identity_from_execution, inspect_formal_checkpoint_file
    from evi_hgt.training.batcher import Phase3QueryBatchAssembler
    from evi_hgt.training.candidate_universe import CandidateUniverse
    from evi_hgt.evaluation.indexed_adapter import IndexedEvaluationAssembler
    from evi_hgt.utils.seed import set_global_seed
    runtimes={}; assembler=None; reference=None;split=cells[0]['split']
    for cell in cells:
        set_global_seed(42)
        _,binding,execution=load_replay_context(root/plan['diagnostic_config'],cell['label'])
        expected_identity=formal_checkpoint_identity_from_execution(execution)
        validate_checkpoint_context(cell,binding,execution,expected_identity)
        require(file_sha(root/binding['best_checkpoint_path'])==cell['checkpoint_sha256'],'checkpoint content changed')
        protocol=execution.run['formal_training_protocol']
        if reference is None:
            reference=protocol
            scalar=Phase3QueryBatchAssembler(root,**protocol['graph'],
                cold_visibility_policy_digest=execution.run.get('cold_visibility_policy_digest'))
            scalar._universes[split]=CandidateUniverse.from_frames(supervision,frame,split_kind=split)
            scalar._split_digests[split]=plan['data_binding']['split_sha256'][split]
            heldout=sorted((*scalar._universes[split].heldout_drugs,*scalar._universes[split].heldout_diseases))
            scalar._heldout_global_ids[split]=frozenset(map(int,index.node_global_ids(heldout)))
            assembler=IndexedEvaluationAssembler(scalar,index,split_frames={split:frame},cold_visibility=cold_visibility)
        else:
            require(protocol['graph']==reference['graph'] and protocol.get('paths')==reference.get('paths'),
                'shared graph/path configuration changed')
        components=build_diagnostic_components(execution,assembler=assembler)
        inspection=inspect_formal_checkpoint_file(root/binding['best_checkpoint_path'],expected_role='BEST',
            expected_identity=expected_identity,map_location='cpu')
        components.model_variant.model.load_state_dict(inspection.payload['model_state_dict'],strict=True)
        runtimes[cell['label']]=SimpleNamespace(execution=execution,components=components)
        del inspection
    return runtimes


def _execute(root,plan,manifest,frozen,auth,*,resume):
    """Authorized future execution only. No optimizer, validation replay or new access count."""
    output=root/plan['output_root'];marker=output/'three_layer_execution_access.json'
    identity=dict(formal_run_id=FORMAL_RUN_ID,formal_test_count=1,plan_sha256=frozen['plan_sha256'],
        freeze_sha256=frozen['freeze_sha256'],checkpoint_binding_digest=plan['checkpoint_binding_digest'],
        protocol_digest=plan['protocol_digest'],units=[u['unit_id'] for u in plan['units']],graph_batch_size=1)
    if resume:
        require(marker.exists() and read_json(marker)==identity,'resume requires same immutable execution entry')
    else:
        if marker.exists():raise PermissionError('Three-layer execution already entered; fresh retries forbidden')
        exclusive_json(marker,identity)
    complete=output/'execution_complete.json'
    if complete.exists():
        saved=read_json(complete);require(saved['identity']==identity,'completion identity changed')
        for path,sha in saved['result_hashes'].items():require(file_sha(output/path)==sha,'completed result changed')
        print('[EXECUTION][GATE] Already completed; no scoring re-entry',flush=True);return 0
    # Authorized entry is durably committed before any model/data runtime load.
    import polars as pl
    import torch
    from scripts.formal_study_validation_diagnostics import no_optimizer_allowed
    from scripts.indexed_evaluation_indexed_fact_columns import IndexedFactColumns
    from evi_hgt.training.input_index import StaticInputIndex
    from evi_hgt.training.formal_driver import _assemble_production_query
    from evi_hgt.training.supervision import make_unlabeled_query_supervision
    from evi_hgt.data.cold_visibility import ColdVisibility
    from evi_hgt.evaluation.indexed_adapter import build_cold_index
    from evi_hgt.evaluation.shared_scoring import SharedScoreStore
    from evi_hgt.evaluation.formal_runtime import evaluate_cell, atomic_json
    from evi_hgt.evaluation.candidates import iter_disease_candidates
    from tqdm import tqdm
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(),'frozen CUDA runtime unavailable')
    base=root/'data/processed/phase3/v1'
    supervision=pl.read_parquet(base/'supervision/dda_supervision.parquet')
    drugs=tuple(sorted(supervision['drug_id'].unique().to_list()))
    frames={s:pl.read_parquet(base/'splits'/f'{s}.parquet') for s in ('standard','cold_drug','cold_disease')}
    results=[];result_hashes={};cold_visibility=cold_index=None
    with no_optimizer_allowed():
        with visible_stage('INDEX_STARTUP'):
            original=StaticInputIndex.build(root)
            # Only _fact_raw is replaced. All IDs, encodings, masks and graph/path arithmetic remain unchanged.
            columns=IndexedFactColumns.from_raw(original._fact_raw)
            index=replace(original,_fact_raw=columns)
        for group,layer,split,cells in [('A','A','standard',plan['cells'][:6]),
                ('B','B','standard',[plan['cells'][0]]),('C','C','cold_drug',[plan['cells'][6]]),
                ('D','D','cold_disease',[plan['cells'][7]])]:
            if split!='standard' and cold_index is None:
                with visible_stage('COLD_INDEX_STARTUP'):
                    cold_visibility=ColdVisibility(root,plan['cold_policy_digest'])
                    cold_index=build_cold_index(index,cold_visibility)
            with visible_stage(group+'_MODEL_SETUP'):
                runtimes=load_group(root,plan,cells,index if split=='standard' else cold_index,
                    supervision,frames[split],cold_visibility=cold_visibility if split!='standard' else None)
            leader=runtimes[cells[0]['label']]
            def assemble(drug,disease):
                return _assemble_production_query(leader,make_unlabeled_query_supervision(drug,disease),partition='test')
            def score_batch(label,batch):
                runtime=runtimes[label];components=runtime.components;device=torch.device(components.device)
                paths=batch.path_batch if runtime.execution.run['architecture']['use_path_reasoning'] else None
                # Byte-for-byte arithmetic/dtype/terminal readback sequence from the accepted route.
                with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    prediction=components.model_variant.model(batch.model_batch,path_batch=paths)
                return tuple(float(getattr(prediction,c+'_logit').detach().float().cpu().reshape(()))
                    for c in ('support','unsuitability'))
            group_identity={**identity,'group':group,'split':split,
                'candidate_digest':manifest['global_manifest_digest'] if layer=='A' else plan['full_manifests'][split]['global_manifest_digest']}
            store=SharedScoreStore(output/'scores'/group,group_identity,[c['label'] for c in cells])
            def streams():
                if layer=='A':return [comparison_candidates(manifest),comparison_candidates(manifest)]
                return [iter_disease_candidates(supervision,frames[split],split_kind=split,partition='test',
                    channel=c,formal_drug_ids=drugs) for c in ('support','unsuitability')]
            total=manifest['totals']['pairs'] if layer=='A' else plan['full_manifests'][split]['pairs']
            with tqdm(total=total,desc=f'[{group}][RUNNING]',unit='pair',dynamic_ncols=True,leave=True,mininterval=5) as bar:
                def counted_assemble(drug,disease):
                    batch=assemble(drug,disease);bar.update(1);return batch
                store.compute(streams(),assemble=counted_assemble,score_batch=score_batch)
            require(not store.failures,'model scoring failure; abort whole amended run')
            runtimes.clear();leader=None;gc.collect();torch.cuda.empty_cache()
            for cell in cells:
                label=cell['label'];unit_id=group+'_'+label;destination=output/'units'/unit_id
                if layer=='A':
                    result=evaluate_comparison(manifest,score_pair=lambda d,s:store.score(label,d,s),
                        model=cell['model_variant'],output=destination,identity=group_identity)
                else:
                    result=evaluate_cell(supervision,frames[split],formal_drug_ids=drugs,split_kind=split,partition='test',
                        score_pair=lambda d,s:store.score(label,d,s),output=destination,identity=group_identity,model=cell['model_variant'])
                # Layer IDs remain explicit; sampled/full tables are never pooled.
                results.append(dict(unit_id=unit_id,layer=layer,benchmark='SAMPLED COMPARISON BENCHMARK' if layer=='A' else 'FULL CANDIDATE RETRIEVAL',result=result))
                path=destination/'result.json';result_hashes[path.relative_to(output).as_posix()]=file_sha(path)
            print(f'[{group}][DONE] Units completed={len(cells)}; scientific values stay in authorized result artifacts',flush=True)
        del original,index,columns,cold_index,cold_visibility
    summary=output/'three_layer_summary.json'
    if summary.exists():require(read_json(summary)==dict(identity=identity,units=results),'completed summary changed')
    else:atomic_json(summary,dict(identity=identity,units=results))
    result_hashes[summary.relative_to(output).as_posix()]=file_sha(summary)
    exclusive_json(complete,dict(identity=identity,result_hashes=result_hashes,status='COMPLETE'))
    print('[EXECUTION][GATE] All nine units complete',flush=True)
    return 0
