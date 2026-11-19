"""CPU-only checkpoint audit and selected-state binding; never constructs a model."""
from pathlib import Path
import torch
from evi_hgt.ranking_protocol.production_state import training_state_digest
from evi_hgt.graph_inputs.common import digest, bound
from .common import ROOT,OUT,read,sha,write_new


def tensor_schema(weights):
    if not isinstance(weights,dict) or not weights:
        raise ValueError('Nonempty tensor state required')
    rows={}
    for name,value in weights.items():
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu':
            raise ValueError('Only CPU tensor weights allowed')
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite checkpoint tensor')
        rows[name]=[list(value.shape),str(value.dtype)]
    return rows


def select_weights(state,method):
    if method=='FULL':
        return state['model'],dict(selected_weight_source='model',selection='FIXED_3G_ENDPOINT',
            selected_step=state['step'],recovery_step=state['step'])
    if method!='HGTDR_ADAPTED':
        raise ValueError('No other historical training credit')
    control=state['sampler']['native_control'];selection=control['selection']
    epoch=selection['best']['epoch']
    if epoch!=222:
        raise ValueError('Registered HGTDR selected epoch must be 222')
    if state['cursor']['completed_epochs']!=300 or len(selection['history'])!=300 or selection['opportunities']!=300:
        raise ValueError('HGTDR complete 300 epoch selection history required')
    if 'best_model' not in control:
        raise ValueError('Selected model state missing; latest recovery cannot substitute')
    return control['best_model'],dict(selected_weight_source='sampler.native_control.best_model',
        selection='INNER_MACRO_AUC_THEN_AP_THEN_EARLIER_EPOCH',selected_epoch=epoch,recovery_epoch=300,
        selection_opportunities=300)


def source_audit(source_map):
    result=[]
    for path,expected in source_map.items():
        p=Path(path)
        if not p.is_absolute(): p=ROOT/p
        actual=sha(p) if p.is_file() else None
        result.append(dict(path=str(p),expected_sha256=expected,actual_sha256=actual,match=actual==expected))
    return result


def verify_reuse_row(row, public):
    cp=Path(row['checkpoint_path']);commit=read(row['commit_path'])
    before=sha(cp)
    if before!=row['checkpoint_sha256'] or before!=commit['checkpoint_sha256'] or cp.stat().st_size!=commit['checkpoint_size']:
        raise ValueError('Historical checkpoint bytes or size changed')
    expected_commit=digest({k:v for k,v in commit.items() if k!='commit_digest'})
    if expected_commit!=commit['commit_digest'] or commit['status']!='COMMITTED':
        raise ValueError('Historical COMMIT integrity differs')
    state=torch.load(cp,map_location='cpu',weights_only=True)
    actual_state=training_state_digest({k:v for k,v in state.items() if k!='state_digest'})
    if actual_state!=state['state_digest'] or actual_state!=commit['state_digest'] or state['identity']!=commit['identity']:
        raise ValueError('Historical complete-state integrity differs')
    selected,selection=select_weights(state,row['method'])
    schema=tensor_schema(selected);recovery_schema=tensor_schema(state['model'])
    if schema!=recovery_schema:
        raise ValueError('Selected/recovery tensor schemas differ')
    if row['method']=='FULL':
        if state['step']!=public['full_3G_updates'] or state['identity']['loss_id']!='EMPIRICAL_PU_HALF_SOFTPLUS_V1':
            raise ValueError('FULL endpoint/loss differs')
        bindings=state['identity']['data_bindings']
        manifest=read(row['source_map_path'])
        source_rows=source_audit(manifest['source_sha256'])
        implementation=source_audit(state['implementation_sources'])
        configured=manifest.get('model_schema')
        if configured is not None and configured!=schema:
            raise ValueError('Historical registered model schema differs')
        metadata=state['metadata']
        if metadata['arm']!='D1_PRETRAINED_EARLY' or metadata['seed']!=public['seed']:
            raise ValueError('FULL D1 constructor metadata differs')
        if sum(selected[k].numel() for k in state['parameter_requires_grad'])!=3846610:
            raise ValueError('FULL parameter count differs')
        constructor=dict(module='evi_hgt_loss_comparison.model',arm='D1_PRETRAINED_EARLY',
            source_map_path=row['source_map_path'],new_wrapper='evi_hgt_ranking_comparison.reuse',architecture_changed=False)
        input_identity=metadata['pretrained']
    else:
        metadata=state['model_metadata'];bindings=metadata['data_bindings']
        manifest=read(commit['execution_manifest_path'])
        source_rows=source_audit(manifest['source_sha256'])
        implementation=source_audit({'src/evi_hgt/ranking_protocol/production_state.py':state['implementation_sha256']})
        source_rows+=source_audit({x['path']:x['sha256'] for x in metadata['native_sources']})
        if metadata['has_refutation_head'] is not False or metadata['config']['epochs']!=300:
            raise ValueError('HGTDR native head/budget differs')
        constructor=dict(module='evi_hgt.ranking_protocol.hgtdr_adapter',native_model='HGTDR',
            source_map_path=commit['execution_manifest_path'],new_wrapper='evi_hgt_ranking_comparison.reuse',architecture_changed=False)
        input_identity=dict(native_sources=metadata['native_sources'],config=metadata['config'],has_refutation_head=False)
    for key,expected in [('fold_manifest_digest',public['cell_digest']),('training_seed',public['seed']),('fold_mask_digest',public['mask_digest'])]:
        if bindings[key]!=expected: raise ValueError('Checkpoint cell binding differs: '+key)
    if not all(x['match'] for x in implementation):
        raise ValueError('Checkpoint implementation differs')
    # Existing constructor closure is recorded separately from this new wrapper.
    if not all(x['match'] for x in source_rows):
        raise ValueError('Historical source closure mismatch')
    selected_hash=training_state_digest(selected);recovery_hash=training_state_digest(state['model'])
    if sha(cp)!=before:
        raise ValueError('Read-only checkpoint changed during audit')
    result=bound(dict(schema='RANKING_COMPARISON_SELECTED_WEIGHT_REUSE_BINDING_V1',status='CPU_WEIGHT_AND_SOURCE_VERIFIED_NO_FORWARD',
        artifact_id=row['artifact_id'],method=row['method'],cell=row['cell'],checkpoint_path=str(cp),checkpoint_sha256=before,
        commit_path=row['commit_path'],commit_sha256=sha(row['commit_path']),complete_state_digest=actual_state,
        selection=selection,selected_tensor_sha256=selected_hash,recovery_tensor_sha256=recovery_hash,
        selected_equals_recovery=selected_hash==recovery_hash,tensor_schema_digest=digest(schema),tensor_entries=len(schema),
        tensor_values=sum(t.numel() for t in selected.values()),constructor_identity=constructor,input_identity=input_identity,
        data_bindings=bindings,source_verification=source_rows,implementation_verification=implementation,
        new_wrapper_sha256=sha(__file__),map_location='cpu',weights_only=True,model_constructed=False,model_forwards=0,
        optimizer_updates=0,old_files_unchanged=True,formal_training_authorized=False,outer_authorized=False,final_authorized=False))
    del state
    return result


def verify_all_reuse():
    part=read(ROOT/'outputs/ranking_protocol/ranking_comparison/part1/RANKING_COMPARISON_EVALUATION_PLAN.json')
    rows=[]
    for row in part['artifact_reuse']:
        if not row.get('training_credit'): continue
        public=next(x for x in part['cells'] if x['cell']==row['cell'])
        result=verify_reuse_row(row,public)
        dest=OUT/'reuse'/f"{row['method']}_{row['cell']}.json"
        if dest.exists():
            if read(dest)!=result: raise ValueError('Immutable reuse binding differs')
        else: write_new(dest,result)
        rows.append(result)
        print(f"复用核验 {row['method']} {row['cell']}：CPU权重与历史源码一致；前向 0。",flush=True)
    if len(rows)!=4: raise ValueError('Exactly four registered training credits required')
    result=bound(dict(schema='RANKING_COMPARISON_FOUR_REUSE_BINDINGS_V1',status='VERIFIED_NO_FORWARD',rows=rows,
        actual_training_credits=4,model_forwards=0,optimizer_updates=0))
    target=OUT/'reuse/REUSE_VERIFIED.json'
    if target.exists():
        if read(target)!=result: raise ValueError('Immutable reuse summary differs')
    else: write_new(target,result)
    return result
