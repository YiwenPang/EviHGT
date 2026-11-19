"""Lossless, weights-only observation files; never a training checkpoint codec."""
from pathlib import Path
import os
import numpy as np
import torch
from .common import sha


def encode(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,np.generic):
        if x.dtype.hasobject:raise TypeError('Object scalar is not safe')
        return {'kind':'numpy_scalar','dtype':x.dtype.str,'hex':x.tobytes().hex()}
    if isinstance(x,dict):return {'kind':'dict','items':[(encode(k),encode(v)) for k,v in x.items()]}
    if isinstance(x,(tuple,list)):return {'kind':type(x).__name__,'items':[encode(v) for v in x]}
    if x is None or type(x) in (str,int,float,bool):return x
    raise TypeError('Unsupported observation type: '+type(x).__name__)


def decode(x):
    if not isinstance(x,dict):return x
    if x['kind']=='numpy_scalar':
        dtype=np.dtype(x['dtype'])
        if dtype.hasobject:raise ValueError('Object scalar is not safe')
        return np.frombuffer(bytes.fromhex(x['hex']),dtype=dtype,count=1)[0]
    if x['kind']=='dict':return {decode(k):decode(v) for k,v in x['items']}
    if x['kind'] in ('list','tuple'):
        values=[decode(v) for v in x['items']]
        return tuple(values) if x['kind']=='tuple' else values
    raise ValueError('Unknown observation encoding')


def save(path,value):
    from .runtime_budget24 import disk_guard24
    from .resource_guard import checkpoint_required_bytes
    path=Path(path);encoded=encode(value)
    disk_guard24(path,checkpoint_required_bytes(encoded))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        torch.save({'schema':'PART24_SAFE_OBSERVATION_V1','value':encoded},f)
        f.flush();os.fsync(f.fileno())
    return {'path':str(path),'sha256':sha(path)}


def load(ref):
    path=Path(ref['path'])
    if sha(path)!=ref['sha256']:raise ValueError('Observation digest mismatch: '+str(path))
    value=torch.load(path,weights_only=True,map_location='cpu')
    if isinstance(value,dict) and value.get('schema')=='PART24_SAFE_OBSERVATION_V1':return decode(value['value'])
    return value  # Existing native input payload uses safe tensors/plain keys.
