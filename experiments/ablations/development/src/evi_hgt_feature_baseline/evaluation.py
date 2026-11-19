"""Exactly two approved DEVELOPMENT panels, three fixed endpoint opportunities."""
from collections import defaultdict
import time
from pathlib import Path

import torch
from evi_hgt.inner_validation.runtime import inner_metrics
from evi_hgt.ranking_protocol.state import tensor_digest
from evi_hgt.training.supervision import make_unlabeled_query_supervision
from .common import OUT,read,write,digest,file_sha
from .diagnostics import preserve_training_state
from .sampling import MILESTONES

def metrics(rows):
    result=inner_metrics(rows)
    for r in result['disease_metrics']:
        r.update(N=r['P']+r['U'],effective_k_5=min(5,r['P']+r['U']),effective_k_10=min(10,r['P']+r['U']))
    counts=defaultdict(lambda:{'P':0,'U':0})
    for r in rows:counts[r['disease_id']][r['role']]+=1
    result.update(P=sum(r['role']=='P' for r in rows),U=sum(r['role']=='U' for r in rows),
        skipped_disease_details=[dict(disease_id=d,**v,N=v['P']+v['U'],reason='NO_P' if not v['P'] else 'NO_U')
                                for d,v in sorted(counts.items()) if not v['P'] or not v['U']],
        candidate_identity_sha256=digest(sorted((r['drug_id'],r['disease_id'],r['role']) for r in rows)),
        AP_definition='threshold-group sum(delta recall * precision); not trapezoidal PR-AUC',
        Hits_interpretation='descriptive within fixed P/U candidate sets; not full-drug retrieval perfection')
    return result

def once_panel(root,domain,bindings,producer):
    if domain not in ('validation','auxiliary_validation'):raise PermissionError('FEATURE_BASELINE evaluation domain denied')
    root=Path(root);complete=root/(domain+'.COMPLETE.json');rows=root/(domain+'.ROWS.json');started=root/(domain+'.STARTED.json')
    if complete.exists():
        r=read(complete)
        if r['bindings']!=bindings or file_sha(rows)!=r['rows_sha256']:raise ValueError('Development panel cache mismatch')
        return read(rows)
    if rows.exists() or started.exists():raise RuntimeError('Partial evaluation; refusing automatic rescore')
    write(started,dict(bindings=bindings,status='STARTED'));t=time.monotonic()
    result=producer();write(rows,result)
    write(complete,dict(status='COMPLETE',bindings=bindings,rows_sha256=file_sha(rows),seconds=time.monotonic()-t))
    return result

def verify_result(result,identity,step):
    if (step not in MILESTONES or result.get('step')!=step or result.get('arm')!=identity['arm']
        or result.get('status')!='COMPLETE_VERIFIED_DEVELOPMENT_EVALUATION'
        or result.get('bindings',{}).get('identity')!=identity
        or set(result.get('rows_paths',{}))!={'validation','auxiliary_validation'}
        or set(result.get('metrics',{}))!={'support','contra'}):
        raise ValueError('Incomplete or cross-arm development milestone')
    expected_root=OUT/identity['arm']/'evaluation'/f'step_{step:08d}'
    expected_files={str((expected_root/(n+s)).resolve()) for n in ('validation','auxiliary_validation')
                    for s in ('.STARTED.json','.COMPLETE.json','.ROWS.json')}
    if set(result.get('artifact_sha256',{}))!=expected_files:raise ValueError('Incomplete development artifact closure')
    for p,h in result['artifact_sha256'].items():
        if file_sha(p)!=h:raise ValueError('Evaluation artifact bytes changed')
    for n in ('validation','auxiliary_validation'):
        rows=expected_root/(n+'.ROWS.json');complete=read(expected_root/(n+'.COMPLETE.json'))
        if (Path(result['rows_paths'][n]).resolve()!=rows.resolve() or complete['bindings']!=result['bindings']
            or complete['status']!='COMPLETE' or complete['rows_sha256']!=file_sha(rows)):
            raise ValueError('Development panel completion differs')
    from .checkpoint import verify_commit
    commit=verify_commit(result['checkpoint_commit_path'],identity)
    if commit['step']!=step or commit['checkpoint_sha256']!=result['bindings']['checkpoint_sha256']:
        raise ValueError('Evaluation did not bind its exact milestone checkpoint')
    # Recompute only metrics from already authorized cached logits, never rescore.
    support=read(result['rows_paths']['validation']);aux=read(result['rows_paths']['auxiliary_validation'])
    contra=[dict(r,support_logit=r['unsuitability_logit'],role='P') for r in aux]
    contra += [dict(r,support_logit=r['unsuitability_logit']) for r in support if r['role']=='U']
    if result['metrics']!={'support':metrics(support),'contra':metrics(contra)}:raise ValueError('Cached metrics do not reproduce')
    return result

def evaluate(data,assembler,built,optimizer,step,commit,identity,root):
    if identity.get('mode')!='DEVELOPMENT':
        raise PermissionError('Only registered DEVELOPMENT checkpoints authorize real scoring')
    if step not in MILESTONES or commit['step']!=step or commit['identity']!=identity:
        raise PermissionError('Wrong FEATURE_BASELINE checkpoint/milestone for evaluation')
    from .common import validate_manifest
    from .runner import identity_from_manifest
    from .checkpoint import Store
    if identity!=identity_from_manifest(validate_manifest(),identity['arm']):
        raise PermissionError('Evaluation requires the current registered FEATURE_BASELINE identity')
    if Path(root).resolve()!=(OUT/identity['arm']/'evaluation').resolve():
        raise PermissionError('Evaluation requires its registered arm directory')
    state=Store(OUT/identity['arm']/'checkpoints',identity).load(commit['checkpoint_commit_path'])
    if (state['step']!=step or tensor_digest(state['model'])!=tensor_digest(built.model.state_dict())):
        raise PermissionError('Evaluation model is not the exact committed milestone weight state')
    from .checkpoint import verify_commit
    if verify_commit(commit['checkpoint_commit_path'],identity)!=commit:
        raise PermissionError('Supplied checkpoint receipt differs from its verified commit')
    del state
    root=Path(root);out=root/f'step_{step:08d}'
    binding=dict(identity=identity,step=step,checkpoint_sha256=commit['checkpoint_sha256'],
        weight_sha256=tensor_digest(built.model.state_dict()),
        validation_members=digest(data.cell['validation']),auxiliary_members=digest(data.cell['auxiliary_validation']),
        mask_sha256=data.bindings['fold_mask_digest'])
    if (out/'evaluation.json').exists():
        r=read(out/'evaluation.json')
        if r['bindings']!=binding:raise ValueError('Evaluation identity differs')
        return verify_result(r,identity,step)
    t=time.monotonic();rows={}
    with preserve_training_state(built.model,optimizer),data.evaluation_scope(step):
        for domain in ('validation','auxiliary_validation'):
            def producer(domain=domain):
                result=[]
                for i,r in enumerate(data.cell[domain]):
                    q=make_unlabeled_query_supervision(r['drug_id'],r['disease_id'])
                    g=assembler.assemble(q,domain=domain,device='cuda')
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        p=built.forward(g.batch.model_batch,path_batch=g.batch.path_batch)
                    result.append(dict(r,support_logit=float(p.support_logit[0]),unsuitability_logit=float(p.unsuitability_logit[0])))
                    if i%256==0:print(f'DEVELOPMENT_EVALUATION {built.arm} {step} {domain} {i}/{len(data.cell[domain])}',flush=True)
                return result
            rows[domain]=once_panel(out,domain,binding,producer)
    contra=[dict(r,support_logit=r['unsuitability_logit'],role='P') for r in rows['auxiliary_validation']]
    contra += [dict(r,support_logit=r['unsuitability_logit']) for r in rows['validation'] if r['role']=='U']
    paths={n:str((out/(n+'.ROWS.json')).resolve()) for n in rows}
    result=dict(status='COMPLETE_VERIFIED_DEVELOPMENT_EVALUATION',arm=built.arm,step=step,bindings=binding,
        metrics={'support':metrics(rows['validation']),'contra':metrics(contra)},rows_paths=paths,
        checkpoint_commit_path=commit['checkpoint_commit_path'],
        seconds=sum(read(out/(n+'.COMPLETE.json'))['seconds'] for n in rows),
        current_invocation_seconds=time.monotonic()-t,
        cost_definition='Sum of durable completed panel wall times across all invocations; partial panel costs remain unknown',
        artifact_sha256={str(p.resolve()):file_sha(p) for p in out.glob('*.json')},
        scope='DEVELOPMENT_ADAPTIVE_NOT_FORMAL_NOT_OUTER_NOT_FINAL',outer='UNSCORED',new_final='SEALED')
    write(out/'evaluation.json',result)
    return result
