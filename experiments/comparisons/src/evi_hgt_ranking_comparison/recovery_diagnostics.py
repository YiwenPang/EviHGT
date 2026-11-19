"""Lossless recovery diagnostics; never invokes a model, optimizer or sampler.

All captured tensors are detached CPU clones. Reports retain both inputs, the
first mismatch and full stage-wise differences without relaxing tolerance.
"""
from dataclasses import fields, is_dataclass
from pathlib import Path
import copy
import math
import os
import random
import uuid

import numpy as np
import torch
from .common import ROOT, digest, sha, write_new
from evi_hgt.ranking_protocol.production_state import _rng

ATOL=1e-5
RTOL=1e-4
STAGES=('before_prepare','input','after_prepare','forward','backward','after_clip','after_optimizer')


def independent_snapshot(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return torch.from_numpy(value.copy()).clone()
    if isinstance(value,np.generic):return value.item()
    if is_dataclass(value):return {f.name:independent_snapshot(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,dict):return {k:independent_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [independent_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(independent_snapshot(v) for v in value)
    if value is None or type(value) in (str,int,float,bool):return copy.deepcopy(value)
    raise TypeError('Unsupported diagnostic value: '+type(value).__name__)


def _tensor_locations(value,path='$'):
    result={}
    if isinstance(value,torch.Tensor):
        result[path]=dict(device=str(value.device),dtype=str(value.dtype),shape=list(value.shape),
            stride=list(value.stride()),storage_offset=value.storage_offset(),requires_grad=value.requires_grad)
    elif isinstance(value,dict):
        for key,item in value.items():result.update(_tensor_locations(item,path+'.'+str(key)))
    elif isinstance(value,(tuple,list)):
        for i,item in enumerate(value):result.update(_tensor_locations(item,path+'['+str(i)+']'))
    return result


def storage_alias_audit(value):
    """Inspect original tensor storage before independent clones erase aliases.

    Addresses are used only transiently to group storage; reports contain stable
    tensor paths. Intersecting non-dense views that cannot be proved disjoint or
    overlapping from their bounds fail closed, without enumerating large views.
    """
    groups={};count=0;internal=[];unresolved_internal=[]
    def collect(item,path='$'):
        nonlocal count
        if isinstance(item,torch.Tensor):
            count+=1
            if not item.numel():return
            if item.layout!=torch.strided or any(s<0 for s in item.stride()):
                unresolved_internal.append(path);return
            dimensions=sorted((s,n) for s,n in zip(item.stride(),item.shape) if n>1)
            span=1;separated=True
            for stride,size in dimensions:
                if stride<span:separated=False
                span+=(size-1)*stride
            if any(stride==0 for stride,size in dimensions) or item.numel()>span:
                internal.append(path)
            elif not separated:
                unresolved_internal.append(path)
            width=item.element_size();start=item.storage_offset()*width
            row=dict(path=path,start=start,stop=start+span*width,
                dense=separated and item.numel()==span)
            key=(str(item.device),item.untyped_storage().data_ptr())
            groups.setdefault(key,[]).append(row)
        elif isinstance(item,dict):
            for key,child in item.items():collect(child,path+'.'+str(key))
        elif isinstance(item,(tuple,list)):
            for i,child in enumerate(item):collect(child,path+'['+str(i)+']')
    collect(value)
    overlapping=[];unresolved=[];disjoint=0
    for members in groups.values():
        for i,left in enumerate(members):
            for right in members[i+1:]:
                if left['stop']<=right['start'] or right['stop']<=left['start']:
                    disjoint+=1;continue
                pair=dict(paths=[left['path'],right['path']],
                    relative_offset_bytes=right['start']-left['start'])
                if left['start']==right['start'] or (left['dense'] and right['dense']):
                    overlapping.append(pair)
                else:unresolved.append(pair)
    return dict(schema='ORIGINAL_TENSOR_STORAGE_ALIAS_V1',
        status='UNRESOLVED' if unresolved or unresolved_internal else 'COMPLETE',
        tensors_audited=count,overlapping_pairs=overlapping,
        internal_overlap_paths=internal,unresolved_pairs=unresolved,
        unresolved_internal_paths=unresolved_internal,
        observed_disjoint_storage_pairs=disjoint)


def snapshot_runtime(model,optimizer,*,inputs=None,cursor=None,sampler=None,
                     stage_state=None,generators=None,extra=None):
    """Introspect existing objects only. This does not construct or run a model."""
    params=list(model.named_parameters());names={id(p):n for n,p in params}
    groups=[];seen=[]
    for group in optimizer.param_groups:
        mapped=[]
        for p in group['params']:
            if id(p) not in names:raise ValueError('Optimizer parameter not owned by model')
            mapped.append(names[id(p)]);seen.append(id(p))
        groups.append(dict(names=mapped,options={k:v for k,v in group.items() if k!='params'}))
    if len(seen)!=len(set(seen)):raise ValueError('Optimizer duplicate parameter mapping')
    generators=generators or {}
    raw=dict(model=model.state_dict(),buffers=dict(model.named_buffers()),
        parameters_order=[n for n,_ in params],optimizer=optimizer.state_dict(),optimizer_named_groups=groups,
        gradients={n:p.grad for n,p in params},requires_grad={n:p.requires_grad for n,p in params},
        module_modes={n:m.training for n,m in model.named_modules()},rng=_rng(),
        generators={n:dict(device=str(g.device),state=g.get_state()) for n,g in generators.items()},
        cursor=cursor,sampler=sampler,stage_state=stage_state,inputs=inputs,extra=extra)
    metadata=_tensor_locations(raw)
    aliases=storage_alias_audit(raw)
    return dict(state=independent_snapshot(raw),tensor_locations=metadata,storage_aliases=aliases)


def _position(flat,shape):
    result=[]
    for size in reversed(shape):result.append(flat%size);flat//=size
    return list(reversed(result))


def compare_snapshots(a,b,*,atol=ATOL,rtol=RTOL,exact=False,max_failures=128):
    """Compare in saved insertion order, keeping the earliest failing path.

    Integer/RNG/cursor/schema/order are exact. Floating data may use the frozen
    tolerances unless exact=True (use for input and pre-step equality checks).
    """
    if atol!=ATOL or rtol!=RTOL:raise ValueError('Qualification tolerances cannot be changed')
    failures=[];failure_count=0;tensors=0;maximum=0.
    def fail(item):
        nonlocal failure_count
        failure_count+=1
        if len(failures)<max_failures:failures.append(item)
    def visit(x,y,path):
        nonlocal tensors,maximum
        if isinstance(x,torch.Tensor):
            tensors+=1
            if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:
                fail(dict(path=path,kind='TENSOR_SCHEMA',reference_shape=list(x.shape),reference_dtype=str(x.dtype),
                    actual_shape=list(y.shape) if isinstance(y,torch.Tensor) else None,
                    actual_dtype=str(y.dtype) if isinstance(y,torch.Tensor) else type(y).__name__));return
            xx=x.detach().cpu();yy=y.detach().cpu()
            if not xx.numel():return
            if xx.is_floating_point() or xx.is_complex():
                finite=torch.isfinite(xx)&torch.isfinite(yy)
                delta=(xx.to(torch.complex128) if xx.is_complex() else xx.double())-(yy.to(torch.complex128) if yy.is_complex() else yy.double())
                error=delta.abs();safe=torch.where(finite,error,torch.full_like(error,float('inf')))
                absmax=float(safe.max());maximum=max(maximum,absmax)
                relative=error/yy.abs().double().clamp_min(torch.finfo(torch.float64).tiny)
                passed=(xx==yy)&finite if exact else finite&(error <= atol+rtol*yy.abs())
                if not bool(passed.all()):
                    first=int((~passed).flatten().nonzero()[0]);worst=int(safe.flatten().argmax())
                    def value(z,i):
                        if z.is_complex():return str(complex(z.flatten()[i]))
                        v=float(z.flatten()[i]);return v if math.isfinite(v) else repr(v)
                    rel=float(torch.nan_to_num(relative,nan=float('inf'),posinf=float('inf')).max())
                    fail(dict(path=path,kind='FLOAT_TENSOR',shape=list(x.shape),dtype=str(x.dtype),
                        reference_device=str(x.device),actual_device=str(y.device),
                        max_abs_difference=absmax if math.isfinite(absmax) else 'NONFINITE',
                        max_relative_difference=rel if math.isfinite(rel) else 'INF',
                        mismatch_count=int((~passed).sum()),first_failed_index=_position(first,x.shape),
                        max_abs_index=_position(worst,x.shape),reference_at_first=value(xx,first),actual_at_first=value(yy,first)))
            elif not torch.equal(xx,yy):
                first=int((xx!=yy).flatten().nonzero()[0])
                fail(dict(path=path,kind='EXACT_TENSOR',shape=list(x.shape),dtype=str(x.dtype),
                    reference_device=str(x.device),actual_device=str(y.device),first_failed_index=_position(first,x.shape),
                    reference_at_first=xx.flatten()[first].item(),actual_at_first=yy.flatten()[first].item()))
        elif isinstance(x,dict):
            if not isinstance(y,dict) or list(x)!=list(y):
                fail(dict(path=path,kind='KEYS_OR_ORDER',reference_keys=list(map(str,x)),actual_keys=list(map(str,y)) if isinstance(y,dict) else None));return
            for key in x:
                if key!='state_digest':visit(x[key],y[key],path+'.'+str(key))
        elif isinstance(x,(tuple,list)):
            if type(x)!=type(y) or len(x)!=len(y):fail(dict(path=path,kind='SEQUENCE_SCHEMA'));return
            for i,(u,v) in enumerate(zip(x,y)):visit(u,v,path+'['+str(i)+']')
        elif type(x)!=type(y) or x!=y:
            # Scalar metadata/cursors, including numeric options, remain exact.
            fail(dict(path=path,kind='EXACT_SCALAR',reference=repr(x),actual=repr(y)))
    visit(a,b,'$')
    return dict(pass_=failure_count==0,atol=atol,rtol=rtol,exact=exact,tensors_compared=tensors,
        mismatch_count=failure_count,first_failure=failures[0] if failures else None,
        failures=failures,failures_truncated=failure_count>len(failures),
        max_abs_tensor_difference=maximum if math.isfinite(maximum) else 'NONFINITE')


def compare_runtime_semantics(a,b,*,exact=False):
    """Compare restored values and layout without equating offsets to identity.

    Fresh original-storage alias audits are mandatory. A nonoverlapping tensor
    can be restored in independent storage at offset zero. For overlapping
    views, offsets and alias topology remain strict. The unmodified physical
    snapshots must still be retained and compared separately by the caller.
    """
    audits=[value.get('storage_aliases') if isinstance(value,dict) else None for value in (a,b)]
    status='COMPLETE'
    if any(audit is None for audit in audits):status='NOT_CAPTURED'
    elif any(not isinstance(audit,dict) or audit.get('schema')!='ORIGINAL_TENSOR_STORAGE_ALIAS_V1'
        or audit.get('status')!='COMPLETE' or audit.get('unresolved_pairs')
        or audit.get('unresolved_internal_paths') for audit in audits):status='UNRESOLVED'
    if status!='COMPLETE':
        failure=dict(path='$.storage_aliases',kind='ALIAS_AUDIT_REQUIRED',status=status)
        return dict(pass_=False,atol=ATOL,rtol=RTOL,exact=exact,tensors_compared=0,
            mismatch_count=1,first_failure=failure,failures=[failure],failures_truncated=False,
            max_abs_tensor_difference=0.,alias_audit=status,storage_offset_observations=[])
    overlap_paths=set()
    for audit in audits:
        for pair in audit['overlapping_pairs']:overlap_paths.update(pair['paths'])
        overlap_paths.update(audit['internal_overlap_paths'])
    observations=[]
    locations=[value['tensor_locations'] for value in (a,b)]
    for path in locations[0].keys() & locations[1].keys():
        before=locations[0][path].get('storage_offset');after=locations[1][path].get('storage_offset')
        if before!=after:observations.append(dict(path=path,reference=before,actual=after,
            semantically_checked=path in overlap_paths))
    observations.sort(key=lambda row:row['path'])
    def semantic(value):
        result=dict(value)
        result['tensor_locations']={path:{key:item for key,item in metadata.items()
            if key!='storage_offset' or path in overlap_paths}
            for path,metadata in value['tensor_locations'].items()}
        # Shared but disjoint storage is physically different after cloning;
        # it has no cross-tensor writes. All overlap audit fields remain exact.
        result['storage_aliases']={key:item for key,item in value['storage_aliases'].items()
            if key!='observed_disjoint_storage_pairs'}
        return result
    result=compare_snapshots(semantic(a),semantic(b),exact=exact)
    result.update(alias_audit=status,storage_offset_observations=observations)
    return result


def _publish_tensor(path,value):
    path=Path(path);partial=path.with_suffix('.'+uuid.uuid4().hex+'.partial')
    if path.exists():raise FileExistsError(path)
    with partial.open('xb') as f:torch.save(independent_snapshot(value),f);f.flush();os.fsync(f.fileno())
    os.rename(partial,path)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def persist_comparison(root,label,a,b,*,atol=ATOL,rtol=RTOL,exact=False):
    root=Path(root).resolve()
    allowed=[(ROOT/('outputs/ranking_protocol/ranking_comparison/'+phase)).resolve() for phase in ('part2_1','part2_2')]
    if not any(root.is_relative_to(p) for p in allowed):raise PermissionError('Diagnostics only in RankingComparison engineering output roots')
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):raise ValueError('Unsafe diagnostic label')
    root.mkdir(parents=True,exist_ok=True)
    result=compare_snapshots(a,b,atol=atol,rtol=rtol,exact=exact)
    from .resource_guard import checkpoint_required_bytes
    from .runtime_budget import disk_guard
    disk_guard(root,checkpoint_required_bytes(a)+checkpoint_required_bytes(b))
    result['reference_payload']=_publish_tensor(root/(label+'_reference.pt'),a)
    result['actual_payload']=_publish_tensor(root/(label+'_actual.pt'),b)
    write_new(root/(label+'.json'),result)
    return result


def compare_trace_files(a,b,out):
    """Read two explicit stage traces, preserve them and report earliest split."""
    pa,pb=Path(a).resolve(),Path(b).resolve();out=Path(out).resolve()
    if not out.is_relative_to((ROOT/'outputs/ranking_protocol/ranking_comparison/part2_1').resolve()):raise PermissionError('New diagnostic output only')
    left=torch.load(pa,map_location='cpu',weights_only=True);right=torch.load(pb,map_location='cpu',weights_only=True)
    order=list(STAGES)+[x for x in left if x not in STAGES]
    rows=[]
    for stage in order:
        if stage in left or stage in right:
            if stage not in left or stage not in right:result=dict(pass_=False,first_failure=dict(path='$.'+stage,kind='MISSING_STAGE'))
            else:result=compare_snapshots(left[stage],right[stage],exact=stage in ('before_prepare','input','after_prepare'))
            rows.append(dict(stage=stage,**result))
    first=next((r for r in rows if not r['pass_']),None)
    result=dict(schema='RANKING_COMPARISON_STAGE_RECOVERY_COMPARISON_V1',pass_=first is None,
        reference=ref_file(pa),actual=ref_file(pb),stage_order=order,stages=rows,
        first_divergent_stage=first['stage'] if first else None,first_failure=first['first_failure'] if first else None,
        atol=ATOL,rtol=RTOL,source_sha256=sha(__file__),
        original_device_note='CPU-loaded trace device is storage device; original device must be retained by producer tensor_locations.')
    out.mkdir(parents=True,exist_ok=True);write_new(out/'comparison.json',result)
    return result


def ref_file(path):return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
