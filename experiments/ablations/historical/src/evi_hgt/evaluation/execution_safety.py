"""Execution-state/IO extension for the immutable Formal evaluator.

No model, candidate, metric or arithmetic implementation is replaced. Completed
scientific shards keep their existing schemas. Separate receipts commit counts,
source/checkpoint bindings and completion; a mutable operational status contains
only counters. An interrupted disease is recomputed in its frozen order.
"""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch

from evi_hgt.evaluation.formal_redesign import digest, exclusive_json, file_sha, read_json
from evi_hgt.evaluation.formal_runtime import atomic_json as _operational_json


class ExecutionSafetyError(RuntimeError):
    pass


def check(condition, message):
    if not condition:raise ExecutionSafetyError(message)


class ExecutionSafety:
    def __init__(self,root,plan,frozen,*,comparison,extension_digest,fixture=False):
        self.root=Path(root);self.plan=plan;self.output=self.root/plan['output_root']
        self.status_path=self.root/'outputs/phase8_4_2/execution_status.json'
        self.base=dict(formal_run_id=plan['formal_run_id'],formal_test_count=plan['formal_test_count'],
            amendment_id=plan['amendment_id'],plan_sha256=frozen['plan_sha256'],
            freeze_sha256=frozen['freeze_sha256'],execution_extension_digest=extension_digest,
            protocol_digest=plan['protocol_digest'],checkpoint_binding_digest=plan['checkpoint_binding_digest'],
            source_data_identity=plan['data_binding'],comparison_manifest_digest=comparison['global_manifest_digest'],
            fixture=fixture,graph_batch_size=1)
        self.rows={'A':comparison['diseases']}
        self.rows.update({layer:plan['full_manifests'][split]['diseases'] for layer,split in
            [('B','standard'),('C','cold_drug'),('D','cold_disease')] if split in plan['full_manifests']})
        self.stage=None;self.stats={};self.started=time.monotonic();self.last_write=0;self.active_model=None
        self._status_lock=threading.Lock()
        self.cells={c['label']:c for c in plan['cells']}

    def units(self,stage):return [u for u in self.plan['units'] if u['layer']==stage]

    def expected(self,stage):
        units=self.units(stage);check(bool(units),'unrecognized execution stage')
        pairs=units[0]['pairs'];diseases=units[0]['diseases']
        check(all(u['pairs']==pairs and u['diseases']==diseases for u in units),'stage candidate scope differs across models')
        return pairs,diseases,len(units)

    def status(self,state='RUNNING',*,force=False,substage=None):
        now=time.monotonic()
        if not force and now-self.last_write<10:return
        self.last_write=now;stat=self.stats.get(self.stage,{})
        elapsed=now-stat.get('started',self.started);fresh=stat.get('fresh',0)
        rate=fresh/elapsed if elapsed>0 and fresh else None
        processed=stat.get('committed',0)+stat.get('live',0)
        payload={**self.base,'pid':os.getpid(),'parent_pid':os.getppid(),'state':state,'stage':self.stage,'substage':substage,
            'disease_index':stat.get('index',0),'diseases_committed':stat.get('diseases',0),
            'diseases_total':stat.get('disease_total',0),'pairs_processed':processed,
            'pairs_committed':stat.get('committed',0),'pairs_total':stat.get('pair_total',0),
            'model_forwards_accounted':processed*stat.get('models',0),
            'active_model_index':self.active_model,'models_serial_per_pair':stat.get('models',0),
            'elapsed_seconds':elapsed,'pairs_per_second':rate,
            'eta_seconds':(stat.get('pair_total',0)-processed)/rate if rate and fresh>=64 else None}
        with self._status_lock:_operational_json(self.status_path,payload)
        if state!='RUNNING' or self.stage is not None:
            print(f'[PHASE][RUNNING] state={state} layer={self.stage} model={self.active_model}/{stat.get("models",0)} '
                f'diseases={stat.get("diseases",0)}/{stat.get("disease_total",0)} '
                f'pairs={processed}/{stat.get("pair_total",0)} elapsed_s={elapsed:.1f} '
                f'pair_rate={rate:.3f}' if rate else
                f'[PHASE][RUNNING] state={state} layer={self.stage} elapsed_s={elapsed:.1f}',flush=True)

    @contextmanager
    def visible_stage(self,name):
        if name.endswith('_MODEL_SETUP'):
            incoming=name[0]
            if self.stage is not None and incoming!=self.stage:self.finish_stage(self.stage)
            self.stage=incoming
        stopped=threading.Event();started=time.monotonic()
        print(f'[{name}][START]',flush=True);self.status(force=True,substage=name)
        def heartbeat():
            while not stopped.wait(30):
                self.status(force=True,substage=name)
                print(f'[{name}][RUNNING] elapsed_s={time.monotonic()-started:.1f}',flush=True)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:yield
        except BaseException:
            self.status('INTERRUPTED',force=True,substage=name)
            print(f'[{name}][GATE] INTERRUPTED; no completion declared',flush=True);raise
        else:print(f'[{name}][DONE] elapsed_s={time.monotonic()-started:.1f}',flush=True)
        finally:stopped.set();thread.join()

    def sealed_json(self,path,value):
        path=Path(path)
        if path.name=='result.json' and path.parent.name in {u['unit_id'] for u in self.plan['units']}:
            unit=next(u for u in self.plan['units'] if u['unit_id']==path.parent.name)
            check(value.get('status')=='COMPLETE' and value.get('disease_count',value.get('query_count'))==unit['diseases'],
                'result query count/status is incomplete')
            stat=self.stats.get(unit['layer'],{})
            check(stat.get('committed')==unit['pairs'] and stat.get('diseases')==unit['diseases'],
                'result lacks verified complete score shards')
        if path.name=='execution_complete.json':
            for layer in dict.fromkeys(u['layer'] for u in self.plan['units']):self.finish_stage(layer)
            check(value.get('status')=='COMPLETE','Formal completion status invalid')
        if path.exists():check(digest(read_json(path))==digest(value),'immutable artifact would be overwritten')
        else:exclusive_json(path,value)
        if path.name=='result.json' and path.parent.name in {u['unit_id'] for u in self.plan['units']}:
            self.unit_receipt(path)
        if path.name=='execution_complete.json':self.validate_complete()

    def validate_complete(self):
        marker=self.output/'execution_complete.json'
        check(marker.exists() and read_json(marker).get('status')=='COMPLETE','Formal finalization missing')
        stages={}
        for stage in dict.fromkeys(u['layer'] for u in self.plan['units']):
            path=self.output/'stages'/stage/'COMPLETE.json';record=read_json(path)
            pairs,diseases,models=self.expected(stage)
            check(all(record.get(k)==v for k,v in self.base.items()),'stage completion binding changed')
            check((record.get('state'),record.get('queries'),record.get('pairs'),record.get('forwards'))==
                ('COMPLETE',diseases,pairs,pairs*models),'stage completion counts changed')
            for unit in self.units(stage):
                receipt=self.output/'units'/unit['unit_id']/'COMPLETE.json';r=read_json(receipt)
                check(file_sha(receipt)==record['unit_receipt_sha256'][unit['unit_id']],
                    'unit completion receipt changed')
                check(file_sha(receipt.parent/'result.json')==r['result_sha256'],'complete result bytes changed')
            stages[stage]=file_sha(path)
        payload={**self.base,'state':'COMPLETE','stage_receipt_sha256':stages,
            'execution_complete_sha256':file_sha(marker),'total_model_forwards':sum(u['pairs'] for u in self.plan['units'])}
        # If interrupted between the original final commit and this companion,
        # verified immutable stage/unit receipts reconstruct this IO-only seal.
        self.sealed_json(self.output/'COMPLETE.json',payload)

    def unit_receipt(self,path):
        unit=next(u for u in self.plan['units'] if u['unit_id']==path.parent.name);cell=self.cells[unit['label']]
        receipt={**self.base,'state':'COMPLETE','unit_id':unit['unit_id'],'stage':unit['layer'],
            'model_variant':cell['model_variant'],'checkpoint_sha256':cell['checkpoint_sha256'],
            'checkpoint_identity':cell['checkpoint_identity'],'candidate_manifest_digest':self.candidate_digest(unit['layer']),
            'queries':unit['diseases'],'pairs':unit['pairs'],'result_sha256':file_sha(path)}
        self.sealed_json(path.parent/'COMPLETE.json',receipt)

    def candidate_digest(self,stage):
        if stage=='A':return self.base['comparison_manifest_digest']
        split=self.units(stage)[0]['split'];return self.plan['full_manifests'][split]['global_manifest_digest']

    def finish_stage(self,stage):
        pairs,diseases,models=self.expected(stage);stat=self.stats.get(stage,{})
        check((stat.get('committed'),stat.get('diseases'))==(pairs,diseases),'stage scoring counts incomplete')
        receipts={}
        for unit in self.units(stage):
            path=self.output/'units'/unit['unit_id']/'result.json'
            check(path.exists(),'stage has missing unit result')
            result=read_json(path)
            check(result.get('status')=='COMPLETE' and result.get('disease_count',result.get('query_count'))==diseases,
                'stage result is partial')
            self.unit_receipt(path);receipts[unit['unit_id']]=file_sha(path.parent/'COMPLETE.json')
        receipt={**self.base,'state':'COMPLETE','stage':stage,'queries':diseases,'pairs':pairs,
            'forwards':pairs*models,'candidate_manifest_digest':self.candidate_digest(stage),'unit_receipt_sha256':receipts}
        self.sealed_json(self.output/'stages'/stage/'COMPLETE.json',receipt)
        print(f'[PHASE][GATE] STAGE_{stage}_COMPLETE = PASS; queries={diseases}; pairs={pairs}; forwards={pairs*models}',flush=True)

    def store_class(self,original):
        safety=self
        class SafeStore(original):
            def validate_shard(self,source):
                path=self.output/(digest(source.disease_id)+'.json');saved=read_json(path);payload=saved['payload']
                binding=dict(identity=self.identity,disease=source.disease_id,candidate_membership=source.candidate_membership_digest)
                check(saved['sha256']==digest(payload) and payload['binding']==binding,'score shard hash/binding mismatch')
                check(tuple(payload['candidate_ids'])==tuple(source.candidate_drug_ids),'score shard order changed')
                check(not payload['failures'] and set(payload['scores'])==set(self.labels),'score shard model set/failure changed')
                for rows in payload['scores'].values():
                    check(len(rows)==len(source.candidate_drug_ids),'score shard pair count incomplete')
                    check(all(len(r)==2 and all(math.isfinite(float(v)) for v in r) for r in rows),'score shard dual rows invalid')
                receipt={**safety.base,'state':'COMPLETE','stage':safety.stage,
                    'candidate_manifest_digest':safety.candidate_digest(safety.stage),
                    'candidate_order_digest':digest(source.candidate_drug_ids),'pairs':len(source.candidate_drug_ids),
                    'models':list(self.labels),'shard_sha256':file_sha(path)}
                safety.sealed_json(path.parent/'completion'/(path.stem+'.json'),receipt)
            def compute(self,candidate_streams,*,assemble,score_batch):
                stage=self.identity['group'];safety.stage=stage;pairs,diseases,models=safety.expected(stage)
                check(tuple(self.labels)==tuple(u['label'] for u in safety.units(stage)),'serial model order changed')
                safety.stats[stage]=dict(committed=0,live=0,fresh=0,diseases=0,index=0,
                    pair_total=pairs,disease_total=diseases,models=models,started=time.monotonic())
                stat=safety.stats[stage];expected_rows=safety.rows[stage]
                def verified_pairs():
                    for index,(support,counter) in enumerate(zip(*candidate_streams,strict=True)):
                        check(index<len(expected_rows),'extra disease in scoring stream');row=expected_rows[index]
                        check(support.disease_id==counter.disease_id==row['disease_id']
                            and support.candidate_drug_ids==counter.candidate_drug_ids
                            and digest(support.candidate_drug_ids)==row['candidate_order_digest'],
                            'frozen disease/candidate stream changed')
                        stat['index']=index+1;stat['live']=0;safety.active_model=None
                        path=self.output/(digest(support.disease_id)+'.json')
                        if path.exists():self.validate_shard(support)
                        yield support,counter
                        self.validate_shard(support)
                        stat['committed']+=len(support.candidate_drug_ids);stat['live']=0;stat['diseases']+=1
                        safety.status(force=True)
                    check(stat['diseases']==diseases and stat['committed']==pairs,'scoring query/pair census incomplete')
                # tee at query boundaries only; never prefetch an extra pair/graph.
                import itertools
                left,right=itertools.tee(verified_pairs(),2)
                def observed_score(label,batch):
                    safety.active_model=self.labels.index(label)+1
                    values=score_batch(label,batch)
                    check(len(values)==2 and all(math.isfinite(float(v)) for v in values),'nonfinite dual prediction')
                    if label==self.labels[-1]:
                        stat['live']+=1;stat['fresh']+=1
                        safety.status(force=stat['fresh']==1)
                    return values
                print(f'[PHASE][START] layer={stage}; serial_models={models}; pairs={pairs}; queries={diseases}',flush=True)
                try:
                    super().compute([(p[0] for p in left),(p[1] for p in right)],assemble=assemble,score_batch=observed_score)
                    check(not self.failures,'model failure prevents stage completion')
                    safety.status(force=True)
                except BaseException:
                    safety.status('INTERRUPTED',force=True);raise
        return SafeStore

    @contextmanager
    def installed(self):
        import evi_hgt.evaluation.formal_runtime as runtime
        import evi_hgt.evaluation.shared_scoring as scores
        import evi_hgt.evaluation.redesigned_runtime as redesigned
        safe_store=self.store_class(scores.SharedScoreStore)
        with patch.object(runtime,'atomic_json',self.sealed_json),patch.object(scores,'atomic_json',self.sealed_json),\
                patch.object(scores,'SharedScoreStore',safe_store),patch.object(redesigned,'visible_stage',self.visible_stage),\
                patch.object(redesigned,'exclusive_json',self.sealed_json):
            try:yield
            except BaseException:self.status('INTERRUPTED',force=True);raise
