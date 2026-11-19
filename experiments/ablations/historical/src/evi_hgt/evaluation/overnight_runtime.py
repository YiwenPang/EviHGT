"""One authorized, exact, fail-closed A→B→C→D Formal execution.

This module changes execution state and sealed output only. Scalar graph assembly,
the dual model forward, precision, indexed representation and metric mathematics
are the accepted immutable implementations. Its tests use synthetic fixtures.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import time
from unittest.mock import patch

from evi_hgt.evaluation.formal_redesign import FORMAL_RUN_ID,digest,exclusive_json,file_sha,read_json,require
from evi_hgt.evaluation.execution_safety import ExecutionSafety,ExecutionSafetyError,check
from evi_hgt.evaluation.formal_runtime import atomic_json as operational_json
from evi_hgt.evaluation.final_runtime import safety_plan


def preflight(root):
    from scripts import phase842_evaluable_freeze as freeze
    require(Path(root).resolve()==freeze.ROOT,'evaluable scope requires its bound project root')
    return freeze.preflight()


def streams(manifest):
    if 'diseases' in manifest:
        from evi_hgt.evaluation.redesigned_runtime import comparison_candidates
        return comparison_candidates(manifest)
    if manifest['protocol']['split']=='standard':
        from evi_hgt.evaluation.evaluable_protocol import candidate_stream
    else:
        from evi_hgt.evaluation.final_protocol import candidate_stream
    return candidate_stream(manifest)


def evaluate_frozen(manifest,**kwargs):
    """Use the immutable metric writer; adapt frozen identities, never equations."""
    from evi_hgt.evaluation import final_runtime
    with patch.object(final_runtime,'streams',streams):
        return final_runtime.evaluate_frozen(manifest,**kwargs)


def seal_top_k(manifest,*,score_pair,path,identity,writer):
    """Persist Stage B canonical Top-50 from committed scores, without new forward."""
    records=[]
    for source in streams(manifest):
        scores={d:float(score_pair(d,source.disease_id)[0]) for d in source.candidate_drug_ids}
        ids=sorted(source.candidate_drug_ids,key=lambda d:(-scores[d],d))[:50]
        records.append(dict(disease_id=source.disease_id,drug_ids=ids,
                            candidate_order_digest=digest(source.candidate_drug_ids),
                            pairs=len(source.candidate_drug_ids)))
    check(len(records)==manifest['totals']['queries'] and sum(r['pairs'] for r in records)==manifest['totals']['pairs'],
          'sealed ranking census incomplete')
    writer(path,dict(identity=identity,state='COMPLETE',manifest_digest=manifest['global_manifest_digest'],
                    protocol_digest=manifest['protocol_digest'],k=50,tie_break='CANONICAL_DRUG_ID_ASCENDING',
                    queries=records,scientific_values_printed=False))


@contextmanager
def prevent_sleep(*,platform=None,set_state=None):
    """Thread-local Windows request, automatically reverted; no policy changes."""
    active=False;platform=os.name if platform is None else platform
    if platform=='nt':
        if set_state is None:
            import ctypes
            set_state=ctypes.windll.kernel32.SetThreadExecutionState
            set_state.argtypes=[ctypes.c_uint32];set_state.restype=ctypes.c_uint32
        active=bool(set_state(0x80000003))  # CONTINUOUS | SYSTEM_REQUIRED | DISPLAY_REQUIRED
    print('[PHASE][GATE] PROCESS_LEVEL_SLEEP_PREVENTION = '+('AVAILABLE' if active else 'UNAVAILABLE'),flush=True)
    try:yield active
    finally:
        if active:set_state(0x80000000)


class OvernightSafety(ExecutionSafety):
    """Extend existing query-atomic recovery with stricter result metadata seals."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.status_path=self.root/'outputs/phase842_evaluable/execution_status.json'
        self.base.update(manifest_digests=deepcopy(self.plan['manifest_bindings']),
                         manifest_bindings=deepcopy(self.plan['manifest_bindings']),
                         source_binding_digest=digest(self.plan['data_binding']),scientific_results_printed=False)

    def expected_result_identity(self,unit):
        cell=self.cells[unit['label']];stage=unit['layer']
        identity=dict(formal_run_id=self.plan['formal_run_id'],formal_test_count=self.plan['formal_test_count'],
            plan_sha256=self.base['plan_sha256'],freeze_sha256=self.base['freeze_sha256'],
            checkpoint_binding_digest=self.plan['checkpoint_binding_digest'],source_data_identity=self.plan['data_binding'],
            protocol_digest=self.plan['protocol_digest'],manifest_bindings=self.plan['manifest_bindings'],
            units=[u['unit_id'] for u in self.plan['units']],graph_batch_size=1,
            group=stage,split=unit['split'],candidate_digest=self.candidate_digest(stage),
            checkpoint_identity=cell['checkpoint_identity'],checkpoint_sha256=cell['checkpoint_sha256'],
            model=cell['model_variant'],benchmark=unit.get('benchmark','SYNTHETIC'),
            manifest_digest=self.candidate_digest(stage),manifest_protocol_digest=unit.get('protocol_digest',
                self.plan.get('protocol',{}).get('layer1',{}).get('protocol_digest','SYNTHETIC')),
            ranking_k=unit.get('ranking_k',[5,10]))
        return identity

    def validate_result(self,unit,value):
        check(value.get('status')=='COMPLETE' and value.get('disease_count',value.get('query_count'))==unit['diseases']
              and value.get('pairs')==unit['pairs'],'scientific result census/state mismatch')
        check(value.get('identity')==self.expected_result_identity(unit),'scientific result immutable identity mismatch')
        check(value.get('benchmark')==unit.get('benchmark','SYNTHETIC') and
              value.get('ranking_k')==unit.get('ranking_k',[5,10]),'scientific result reporting scope mismatch')

    def sealed_json(self,path,value):
        path=Path(path)
        unit=next((u for u in self.plan['units'] if u['unit_id']==path.parent.name),None)
        if path.name=='result.json' and unit:self.validate_result(unit,value)
        super().sealed_json(path,value)

    def unit_receipt(self,path):
        unit=next(u for u in self.plan['units'] if u['unit_id']==path.parent.name);cell=self.cells[unit['label']]
        value=read_json(path);self.validate_result(unit,value)
        query_hashes={p.relative_to(path.parent).as_posix():file_sha(p) for p in sorted((path.parent/'queries').glob('*.json'))}
        if not self.base['fixture']:check(len(query_hashes)==unit['diseases'],'query scientific artifact census incomplete')
        receipt={**self.base,'state':'COMPLETE','unit_id':unit['unit_id'],'stage':unit['layer'],
            'model_variant':cell['model_variant'],'checkpoint_sha256':cell['checkpoint_sha256'],
            'checkpoint_identity':cell['checkpoint_identity'],'candidate_manifest_digest':self.candidate_digest(unit['layer']),
            'queries':unit['diseases'],'pairs':unit['pairs'],'result_sha256':file_sha(path),
            'result_identity_digest':digest(self.expected_result_identity(unit)),
            'benchmark':unit.get('benchmark','SYNTHETIC'),'ranking_k':unit.get('ranking_k',[5,10]),
            'query_artifact_sha256':query_hashes}
        self.sealed_json(path.parent/'COMPLETE.json',receipt)


class OvernightChain:
    """Transitions are determined only by sealed stage metadata and byte hashes."""
    def __init__(self,safety,*,identity):
        self.safety=safety;self.identity=identity;self.output=safety.output
        check(safety.plan['execution_order']==list('ABCD'),'overnight execution order changed')
        self.order=tuple('ABCD')

    def receipt_path(self,stage):return self.output/'stages'/stage/'OVERNIGHT_COMPLETE.json'

    def verify(self,stage):
        safety=self.safety;path=self.receipt_path(stage);record=read_json(path)
        pairs,queries,models=safety.expected(stage)
        check(all(record.get(k)==v for k,v in safety.base.items()),'overnight stage immutable binding mismatch')
        check(record.get('identity')==self.identity and record.get('stage')==stage and record.get('state')=='COMPLETE'
              and (record.get('pairs'),record.get('queries'),record.get('forwards'))==(pairs,queries,pairs*models)
              and record.get('candidate_manifest_digest')==safety.candidate_digest(stage),'overnight stage census/binding mismatch')
        original=path.parent/'COMPLETE.json';check(file_sha(original)==record['stage_completion_sha256'],
                                                  'stage receipt bytes changed')
        saved=read_json(original)
        check(all(saved.get(k)==v for k,v in safety.base.items()) and saved.get('state')=='COMPLETE' and
              (saved.get('pairs'),saved.get('queries'),saved.get('forwards'))==(pairs,queries,pairs*models),
              'original stage receipt binding/counts changed')
        check(set(record['unit_receipt_sha256'])=={u['unit_id'] for u in safety.units(stage)},'stage unit set mismatch')
        for unit in safety.units(stage):
            unit_path=self.output/'units'/unit['unit_id']/'COMPLETE.json';receipt=read_json(unit_path)
            check(file_sha(unit_path)==record['unit_receipt_sha256'][unit['unit_id']]
                  ==saved['unit_receipt_sha256'][unit['unit_id']],'unit completion bytes changed')
            check(all(receipt.get(k)==v for k,v in safety.base.items()) and receipt.get('state')=='COMPLETE'
                  and receipt.get('pairs')==pairs and receipt.get('queries')==queries
                  and receipt.get('result_identity_digest')==digest(safety.expected_result_identity(unit)),
                  'unit completion binding/census mismatch')
            check(file_sha(unit_path.parent/'result.json')==receipt['result_sha256'],'scientific result bytes changed')
            for name,sha in receipt['query_artifact_sha256'].items():
                check(file_sha(unit_path.parent/name)==sha,'query scientific artifact bytes changed')
        for name,sha in record['extra_artifact_sha256'].items():
            check(file_sha(self.output/name)==sha,'additional sealed artifact bytes changed')
        raw_receipts=record['raw_score_receipt_sha256']
        if raw_receipts or not safety.base['fixture']:
            check(len(raw_receipts)==queries,'raw query completion census incomplete')
        raw_pairs=0
        for name,sha in raw_receipts.items():
            path=self.output/name;metadata=read_json(path)
            check(file_sha(path)==sha and all(metadata.get(k)==v for k,v in safety.base.items())
                  and metadata.get('state')=='COMPLETE' and metadata.get('stage')==stage
                  and metadata.get('candidate_manifest_digest')==safety.candidate_digest(stage)
                  and metadata.get('models')==[u['label'] for u in safety.units(stage)],
                  'raw query receipt immutable binding/count changed')
            check(file_sha(path.parent.parent/path.name)==metadata['shard_sha256'],'raw score shard bytes changed')
            raw_pairs+=metadata['pairs']
        if raw_receipts:check(raw_pairs==pairs,'raw pair completion census mismatch')
        if stage=='B' and not safety.base['fixture']:
            check(set(record['extra_artifact_sha256'])=={'units/'+safety.units(stage)[0]['unit_id']+'/top_k.json'},
                  'Stage B sealed ranking artifact absent')
        return record

    def commit(self,stage):
        safety=self.safety;pairs,queries,models=safety.expected(stage)
        original=self.output/'stages'/stage/'COMPLETE.json'
        check(original.exists(),'stage has not finalized its independent completion receipt')
        saved=read_json(original)
        extras={}
        raw_receipts={p.relative_to(self.output).as_posix():file_sha(p)
                      for p in sorted((self.output/'scores'/stage/'completion').glob('*.json'))}
        if stage=='B' and not safety.base['fixture']:
            p=self.output/'units'/safety.units(stage)[0]['unit_id']/'top_k.json'
            check(p.exists(),'Stage B ranking artifact finalization missing');extras[p.relative_to(self.output).as_posix()]=file_sha(p)
        receipt={**safety.base,'identity':self.identity,'state':'COMPLETE','stage':stage,
                 'pairs':pairs,'queries':queries,'forwards':pairs*models,
                 'candidate_manifest_digest':safety.candidate_digest(stage),'stage_completion_sha256':file_sha(original),
                 'unit_receipt_sha256':saved['unit_receipt_sha256'],'extra_artifact_sha256':extras,
                 'raw_score_receipt_sha256':raw_receipts}
        safety.sealed_json(self.receipt_path(stage),receipt);return self.verify(stage)

    def interrupted(self,stage,exc):
        safety=self.safety;safety.stage=stage;safety.status('INTERRUPTED',force=True)
        status=read_json(safety.status_path);stamp=datetime.now(timezone.utc).isoformat()
        record={**safety.base,'identity':self.identity,'state':'INTERRUPTED','stage':stage,
                'stop_timestamp':stamp,'reason':type(exc).__name__,'next_stage_started':False,
                'last_committed_query':status['diseases_committed'],'pairs_processed':status['pairs_processed'],
                'pairs_committed':status['pairs_committed'],'elapsed_seconds':status['elapsed_seconds']}
        safety.sealed_json(self.output/'stages'/stage/('INTERRUPTED_'+str(time.time_ns())+'.json'),record)
        status.update(failed_stage=stage,reason=type(exc).__name__,next_stage_started=False)
        operational_json(safety.status_path,status)
        print(f'[PHASE][GATE] Stage {stage} = INTERRUPTED; reason={type(exc).__name__}; next stage started=NO',flush=True)

    def run(self,execute_stage,*,verify_inputs=None):
        stage=None
        try:
            for stage in self.order:
                self.safety.stage=stage
                for previous in self.order[:self.order.index(stage)]:self.verify(previous)
                if self.receipt_path(stage).exists():
                    self.verify(stage);print(f'[PHASE][GATE] Stage {stage} = PASS; completed immutable stage reused',flush=True)
                    continue
                if verify_inputs:verify_inputs(stage)
                print(f'[PHASE][START] Stage {stage}',flush=True)
                execute_stage(stage);self.commit(stage)
                print(f'[PHASE][DONE] Stage {stage}; sealed output integrity verified',flush=True)
                print(f'[PHASE][GATE] Stage {stage} = PASS',flush=True)
            hashes={stage:file_sha(self.receipt_path(stage)) for stage in self.order}
            for stage in self.order:self.verify(stage)
            receipt={**self.safety.base,'identity':self.identity,'state':'COMPLETE','stage_receipt_sha256':hashes,
                     'total_model_forwards':sum(u['pairs'] for u in self.safety.plan['units']),
                     'scientific_results':'PERSISTED / SEALED / NOT PRINTED',
                     'top_k':'PERSISTED / SEALED / NOT PRINTED','readiness':'READY_FOR_FORMAL_RESULTS_UNBLINDING'}
            self.safety.sealed_json(self.output/'OVERNIGHT_COMPLETE.json',receipt)
            self.safety.status('COMPLETE',force=True)
            status=read_json(self.safety.status_path);status.update(readiness=receipt['readiness'],
                stages={stage:'COMPLETE / PASS' for stage in self.order},stage_receipt_sha256=hashes,
                total_model_forwards=receipt['total_model_forwards'],scientific_results='PERSISTED / SEALED / NOT PRINTED',
                top_k='PERSISTED / SEALED / NOT PRINTED')
            operational_json(self.safety.status_path,status)
            print('[PHASE][GATE] FORMAL EXECUTION COMPLETE; A PASS; B PASS; C PASS; D PASS',flush=True)
            print('READY FOR FORMAL RESULTS UNBLINDING',flush=True)
            return 0
        except BaseException as exc:
            if stage:self.interrupted(stage,exc)
            raise


def validate_authorization(auth,f,plan):
    check(auth.get('authorized') is True and auth.get('action')=='FINAL_OVERNIGHT_EVALUABLE_FORMAL_EXECUTION',
          'superseded execution grant forbidden')
    check(auth.get('formal_run_id')==FORMAL_RUN_ID and auth.get('formal_test_count')==1
          and auth.get('amendment_id')==plan['amendment_id'] and auth.get('plan_sha256')==f['plan_sha256']
          and auth.get('plan_digest')==auth.get('execution_plan_digest')==f['execution_plan_digest']
          and auth.get('freeze_sha256')==f['freeze_sha256'] and auth.get('manifest_bindings')==plan['manifest_bindings']
          and auth.get('units')==[u['unit_id'] for u in plan['units']]
          and auth.get('checkpoint_binding_digest')==plan['checkpoint_binding_digest']
          and auth.get('source_binding_digest')==digest(plan['data_binding'])
          and auth.get('execution_order')==list('ABCD') and auth.get('scientific_result_visibility') is False
          and auth.get('automatic_retry') is False and auth.get('issued_utc') and auth.get('user_authorization_source'),
          'overnight grant immutable bindings/visibility differ')


def run(root,*,authorization_path=None,plan_path=None,resume=False):
    if authorization_path is None:raise PermissionError('Overnight execution requires the explicit bound user grant')
    auth=read_json(authorization_path)
    if auth.get('authorized') is not True or auth.get('action')!='FINAL_OVERNIGHT_EVALUABLE_FORMAL_EXECUTION':
        raise PermissionError('Superseded grants cannot authorize evaluable overnight execution')
    root=Path(root).resolve();f,plan,layer1,manifests=preflight(root);validate_authorization(auth,f,plan)
    selected=Path(plan_path) if plan_path else root/f['plan_path']
    if not selected.is_absolute():selected=root/selected
    require(file_sha(selected)==f['plan_sha256'],'selected overnight plan file differs')
    from evi_hgt.evaluation.evaluable_protocol import validate_plan
    validate_plan(plan,expected_digest=f['execution_plan_digest'])
    from evi_hgt.formal_execution import formal_execution_lease
    projected=safety_plan(plan,manifests)
    # Every unit already commits its manifest protocol digest; Stage A retains V2 U32.
    for u in projected['units']:
        if u['layer']=='A':u['protocol_digest']=layer1['protocol_digest']
    safety=OvernightSafety(root,projected,f,comparison=layer1,extension_digest=f['execution_plan_digest'])
    with formal_execution_lease(root/plan['output_root']),prevent_sleep(),safety.installed():
        return _execute(root,projected,layer1,manifests,f,auth,safety,resume=resume)


def _execute(root,plan,layer1,manifests,f,auth,safety,*,resume):
    """The immutable scalar assembly/dual-forward AST is preserved exactly."""
    output=root/plan['output_root'];marker=output/'final_execution_access.json'
    identity=dict(formal_run_id=FORMAL_RUN_ID,formal_test_count=1,plan_sha256=f['plan_sha256'],freeze_sha256=f['freeze_sha256'],
        checkpoint_binding_digest=plan['checkpoint_binding_digest'],source_data_identity=plan['data_binding'],
        protocol_digest=plan['protocol_digest'],manifest_bindings=plan['manifest_bindings'],
        units=[u['unit_id'] for u in plan['units']],graph_batch_size=1)
    if resume:require(marker.exists() and read_json(marker)==identity,'resume requires same final immutable execution entry')
    else:
        require(not marker.exists(),'final attempt already entered; fresh retries forbidden')
        safety.sealed_json(marker,identity)
    chain=OvernightChain(safety,identity=identity)
    if (output/'OVERNIGHT_COMPLETE.json').exists():
        return chain.run(lambda stage:check(False,'complete execution cannot replay scoring'))
    # Imports that can construct models occur only after all authorization/freeze gates.
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
    cold_visibility=cold_index=None
    with no_optimizer_allowed():
        with safety.visible_stage('INDEX_STARTUP'):
            original=StaticInputIndex.build(root);columns=IndexedFactColumns.from_raw(original._fact_raw)
            index=replace(original,_fact_raw=columns)
        def execute_stage(stage):
            nonlocal cold_visibility,cold_index
            split={'A':'standard','B':'standard','C':'cold_drug','D':'cold_disease'}[stage]
            cells=[safety.cells[u['label']] for u in safety.units(stage)]
            manifest=layer1 if stage=='A' else manifests[split]
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
            with tqdm(total=manifest['totals']['pairs'],desc=f'[PHASE][RUNNING] Stage {stage}',unit='pair',
                    dynamic_ncols=True,leave=True,mininterval=5) as bar:
                def counted_assemble(drug,disease):
                    batch=assemble(drug,disease);bar.update(1);return batch
                store.compute([streams(manifest),streams(manifest)],assemble=counted_assemble,score_batch=score_batch)
            require(not store.failures,'model scoring failure prevents next stage')
            runtimes.clear();leader=None;gc.collect();torch.cuda.empty_cache()
            for cell in cells:
                unit_id=stage+'_'+cell['label'];destination=output/'units'/unit_id
                evaluate_frozen(manifest,score_pair=lambda d,s:store.score(cell['label'],d,s),model=cell['model_variant'],
                    output=destination,identity={**group_identity,'checkpoint_identity':cell['checkpoint_identity'],
                        'checkpoint_sha256':cell['checkpoint_sha256']})
                if stage=='B':
                    seal_top_k(manifest,score_pair=lambda d,s:store.score(cell['label'],d,s),path=destination/'top_k.json',
                               identity=safety.expected_result_identity(safety.units(stage)[0]),writer=safety.sealed_json)
            safety.finish_stage(stage)
        def verify_inputs(stage):
            frozen,loaded,comparison,candidates=preflight(root)
            require(frozen['freeze_sha256']==f['freeze_sha256'] and frozen['plan_sha256']==f['plan_sha256']
                    and loaded['manifest_bindings']==plan['manifest_bindings'],'stage-boundary frozen bindings changed')
        return chain.run(execute_stage,verify_inputs=verify_inputs)


def dispatch_main(argv,*,project_root):
    """Preserve the primary main CLI, with scoped final-format/runtime dispatch."""
    import evi_hgt.evaluation.formal_redesign as protocol
    import evi_hgt.evaluation.redesigned_runtime as old
    from evi_hgt.evaluation.evaluable_protocol import PLAN_FORMAT
    from evi_hgt.cli import main
    with patch.object(protocol,'PLAN_FORMAT',PLAN_FORMAT),patch.object(old,'run',run):
        return main(argv,project_root=project_root)
