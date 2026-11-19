"""G3 development-only scoring route. Capability is checked before projection."""
import math,random
import numpy as np
import torch
from .common import ROOT,OUT,read,write_new,sha,digest
from .authorization import ExecutionCapability
from evi_hgt.graph_inputs.common import digest as historical_digest

class DevelopmentView:
    def __init__(self,data,parts):
        self.data=data;self.parts=parts;self.cell={**data.cell,**parts}
        self.domains={k:{(r['drug_id'],r['disease_id']) for r in v} for k,v in parts.items()}
    def __getattr__(self,key):return getattr(self.data,key)
    def assert_score_domain(self,pair,domain):
        if domain not in ('validation','train_probe','auxiliary_validation') or tuple(pair) not in self.domains[domain] or tuple(pair) in self.data.frozen:raise PermissionError('G3 development membership violation')

def project_development(data,capability,identity,root):
    if not isinstance(capability,ExecutionCapability):raise PermissionError('G3 capability required before development read')
    capability.permits_development(identity,root)
    from .data import objects_in_array,field,reserved
    frozen=next(x for x in data.shared.part1['cells'] if x['cell']==data.cell_id)
    direct=ROOT/f'data/processed/query_model/repeat_{data.repeat_index}_fold_{data.cell["fold"]}/cell_manifest.json'
    if direct.exists():
        parts={n:[__import__('json').loads(x) for x in objects_in_array(direct,n)] for n in ('validation','train_probe','auxiliary_validation')}
    else:
        entry=next(x for x in data.shared.keys['selected_repeat_bindings'] if x['original_repeat_index']==data.repeat_index)
        path=ROOT/entry['fold_assignment_path']
        if sha(path)!=entry['fold_assignment_sha256']:raise ValueError('Assignment changed')
        key=historical_digest([data.cell['repeat_key'],data.cell['fold'],'inner'])
        parts=dict(validation=[],train_probe=[],auxiliary_validation=[]);domain_ids=set()
        for raw in objects_in_array(path,'assignments'):
            if field(raw,'fold',True)==data.cell['fold']:continue
            pair=field(raw,'drug_id'),field(raw,'disease_id')
            which='validation' if reserved(pair,key+'validation',.2) else 'train_probe' if reserved(pair,key+'probe',.05) else None
            if which:parts[which].append(dict(drug_id=pair[0],disease_id=pair[1],role=field(raw,'role')))
            else:domain_ids.add(pair)
        excluded=data.shared.panel_ids_for_exclusion-domain_ids
        for a,b in sorted(data.shared.aux_pairs-excluded):
            if reserved((a,b),key+'aux_validation',.2):parts['auxiliary_validation'].append(dict(drug_id=a,disease_id=b,raw_relation='CONTRAINDICATION'))
        for name in parts:parts[name].sort(key=lambda r:(r['drug_id'],r['disease_id']))
    for name,rows in parts.items():
        if historical_digest(rows)!=frozen['membership_digests'][name]:raise ValueError('Development membership/order changed '+name)
    return DevelopmentView(data,parts)

def evaluate_development(engine,capability,task):
    if engine.mode=='TEST_ONLY':return evaluate_training_fixture(engine,task)
    from evi_hgt.ranking_protocol.real_data import load_canonical_assembler
    from evi_hgt.training.supervision import make_unlabeled_query_supervision
    from evi_hgt.ranking_protocol.production_state import _rng,_validate_rng
    from .metrics import metric_pair
    view=project_development(engine.data,capability,engine.identity,engine.root)
    assembler=load_canonical_assembler(view,engine.corpus,engine.index)
    rng=_rng();modes={n:m.training for n,m in engine.built.model.named_modules()};results={}
    weights=__import__('evi_hgt_ranking_comparison.models',fromlist=['state_digest']).state_digest(engine.built.model.state_dict())
    target=engine.root/'evaluation';target.mkdir(exist_ok=True)
    try:
        engine.built.model.eval()
        for domain in ('validation','auxiliary_validation'):
            path=target/(domain+'.json')
            binding=dict(identity=engine.identity,weights_digest=weights,head='BOTH',mask=engine.metadata['mask_digest'],members_digest=historical_digest(view.parts[domain]))
            if path.exists():
                cached=read(path)
                if any(cached.get(k)!=v for k,v in binding.items()):raise ValueError('Development cache identity differs')
                members=[{k:v for k,v in r.items() if k not in ('support_logit','unsuitability_logit')} for r in cached['rows']]
                if members!=view.parts[domain]:raise ValueError('Development cached membership/order differs')
                results[domain]=cached['rows'];continue
            rows=[]
            for r in view.parts[domain]:
                q=make_unlabeled_query_supervision(r['drug_id'],r['disease_id'])
                g=assembler.assemble(q,domain=domain,device='cuda')
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):pred=engine.built.forward(g.batch.model_batch,path_batch=g.batch.path_batch)
                row=dict(r,support_logit=float(pred.support_logit.float().item()),unsuitability_logit=float(pred.unsuitability_logit.float().item()))
                if not all(math.isfinite(row[k]) for k in ('support_logit','unsuitability_logit')):raise FloatingPointError('Nonfinite development score')
                rows.append(row)
            results[domain]=rows
            write_new(path,dict(binding,rows=rows))
        metrics=metric_pair(results['validation'],results['auxiliary_validation'])
        if (target/'metrics.json').exists():
            if read(target/'metrics.json')!=metrics:raise ValueError('Development metric cache differs')
        else:write_new(target/'metrics.json',metrics)
        from .console import Console
        console=Console(context='AUTHORIZED_RUNTIME');console.evaluation_result(engine.method,engine.logical,metrics);console.close()
        return str(target/'metrics.json')
    finally:
        for n,m in engine.built.model.named_modules():m.training=modes[n]
        random.setstate(rng['python']);np.random.set_state(_validate_rng(rng));torch.set_rng_state(rng['cpu'])
        if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])

def score_popularity(data,capability,identity,root,counts_ref):
    """No candidate reads until G3 permission; uses sealed actual count payload."""
    view=project_development(data,capability,identity,root)
    if sha(counts_ref['counts_path'])!=counts_ref['counts_sha256']:raise ValueError('Popularity count bytes changed')
    counts=read(counts_ref['counts_path'])
    if counts['identity']['cell']!=data.cell_id or counts['identity']['mask_digest']!=data.bindings['fold_mask_digest']:raise ValueError('Popularity binding differs')
    from .popularity import Popularity
    from .metrics import metric_pair
    scorer=Popularity(data.cell_id,counts['support_counts'],counts['contra_counts'],counts['identity'])
    support=scorer.score_rows(view.parts['validation'],cell=data.cell_id)
    contra=scorer.score_rows(view.parts['auxiliary_validation'],cell=data.cell_id)
    return metric_pair(support,contra)

def evaluate_training_fixture(engine,task):
    """Same scoring entry, but only registered training queries and ledger calls."""
    from .runtime_budget import RuntimeBudget
    from .training import prepare_step
    from .numerical_policy import method_numerical_policy
    from .checkpoint import _rng,_validate_rng
    from .metrics import metric_pair
    from .recovery_diagnostics import compare_snapshots
    budget=engine.budget
    if not isinstance(budget,RuntimeBudget) or budget.started is None or budget.spec.get('test_evaluation_mode')!='REGISTERED_TRAINING_ONLY':raise PermissionError('Registered fixture scoring required')
    step=task['fixture_step']
    if step not in budget.spec['steps']:raise PermissionError('Unregistered fixture step')
    spec=engine.plan.step(step);before=engine.state();rng=_rng();modes={n:m.training for n,m in engine.built.model.named_modules()}
    support=[];contra=[];rows=[]
    try:
        prepared=prepare_step(engine.assembler,spec,device='cuda');engine.built.model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16),method_numerical_policy(engine.method):
            for query,graph in zip(spec.queries,prepared.graphs,strict=True):
                pred=budget.forward(lambda:engine.built.forward(graph.batch.model_batch,path_batch=graph.batch.path_batch),1,label=engine.method+'/TRAINING_FIXTURE_EVALUATION')
                row=dict(drug_id=query.drug_id,disease_id=query.disease_id,support_logit=float(pred.support_logit.float().item()),unsuitability_logit=float(pred.unsuitability_logit.float().item()))
                if query.observed_relation is None:support.append(dict(row,role='U'))
                elif query.support.observed_mask:support.append(dict(row,role='P'))
                if query.unsuitability.observed_mask:contra.append(dict(row,role='P'))
                rows.append(row)
        metrics=metric_pair(support,contra)
        for head in metrics:metrics[head]['scope']='TEST_ONLY_REGISTERED_TRAINING_FIXTURE_NOT_DEVELOPMENT'
    finally:
        for n,m in engine.built.model.named_modules():m.training=modes[n]
        random.setstate(rng['python']);np.random.set_state(_validate_rng(rng));torch.set_rng_state(rng['cpu'])
        if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])
    invariant=compare_snapshots(before,engine.state(),exact=True)
    if not invariant['pass_']:raise ValueError('Fixture evaluator changed training state')
    path=engine.root/'training_fixture_evaluation.json'
    write_new(path,dict(mode='TEST_ONLY',not_formal_development=True,source_step=step,
        fixture_content_digest=spec.content_digest,rows=rows,metrics=metrics,training_state_preserved=invariant))
    from .console import Console
    console=Console(context='TEST_ONLY');console.stage('训练夹具评分验收','不是正式开发集、outer或final结果')
    console.evaluation_result(engine.method,engine.logical,metrics);console.close()
    return str(path)
