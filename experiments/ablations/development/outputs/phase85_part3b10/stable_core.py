"""Diagnostic-only, first-order derivative of the unchanged continuous sigmoid target."""
import math
import torch
from torch.overrides import TorchFunctionMode

class StableNegativeSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z):
        if z.dtype not in (torch.float32,torch.float64):raise TypeError('Expected existing float32 margin or double gradcheck input')
        ctx.save_for_backward(z)
        with torch.autocast(device_type=z.device.type,enabled=False):return torch.sigmoid(-z)
    @staticmethod
    def backward(ctx,grad_output):
        if torch.is_grad_enabled():raise RuntimeError('StableNegativeSigmoid supports first-order backward only')
        z,=ctx.saved_tensors
        with torch.autocast(device_type=z.device.type,enabled=False):
            q=torch.exp(-z.abs())
            return grad_output*(-q/(1+q).square())

stable_sigmoid_negative=StableNegativeSigmoid.apply

class LossCapture(TorchFunctionMode):
    def __init__(self,candidate):self.candidate=candidate;self.values={}
    def __torch_function__(self,func,types,args=(),kwargs=None):
        name=func.__name__
        if name=='sigmoid' and self.candidate:
            z=self.values['z']
            if not torch.equal(args[0],-z):raise ValueError('Unexpected sigmoid input: not the captured -z')
            value=stable_sigmoid_negative(z)
        else:value=func(*args,**(kwargs or {}))
        if name in ('sub','__sub__') and isinstance(value,torch.Tensor) and value.ndim==2:
            if 'm' in self.values:raise ValueError('Multiple margins in one call are outside fixed small-group scope')
            self.values['m']=value
        elif name in ('div','__truediv__') and isinstance(value,torch.Tensor) and value.ndim==2:self.values['z']=value
        elif name=='sigmoid':self.values['items']=value
        return value

class ObservedChannel(torch.nn.Module):
    def __init__(self,original,candidate=False):
        super().__init__();self.original=original;self.candidate=candidate;self.calls=[]
        self.temperature=original.temperature;self.pairwise_chunk_size=original.pairwise_chunk_size
    def forward(self,p,u):
        cap=LossCapture(self.candidate)
        with cap:r=self.original(p,u)
        if set(cap.values)!={'m','z','items'} or cap.values['m'].numel()!=r.pair_count:raise ValueError('Incomplete production loss capture')
        self.calls.append(dict(p=p,u=u,risk=r,**cap.values));return r

def reference(z,weight):
    a=abs(float(z));q=math.exp(-a);logabs=-a-2*math.log1p(q)+math.log(weight)
    return dict(coefficient=-weight*q/(1+q)**2,log_abs=logabs)

def derivative_check(actual,ref,logref,dtype=torch.float32):
    tiny=torch.finfo(dtype).tiny;minsub=math.ulp(0.) if dtype==torch.float64 else 2.**-149
    if logref<math.log(minsub):category='UNDERFLOW_LIMIT_BELOW_SUBNORMAL'
    elif logref<math.log(tiny):category='UNDERFLOW_LIMIT_SUBNORMAL'
    else:category='NORMAL'
    err=abs(actual-ref);rel=err/abs(ref) if category=='NORMAL' else None
    finite=math.isfinite(actual);sign=actual<=0
    return dict(range=category,absolute_error=err,relative_error=rel,pass_check=finite and sign and (rel<=1e-5 if rel is not None else True),
                **{'pass':finite and sign and (rel<=1e-5 if rel is not None else True)})

def vector_metrics(value,reference_vector):
    a=value.double().cpu();b=reference_vector.double().cpu();na=float(a.norm());nb=float(b.norm());err=float((a-b).norm())
    return dict(norm=na,reference_norm=nb,absolute_residual=err,relative_residual=err/nb if nb>1e-12 else None,
                cosine=max(-1.,min(1.,float(a.dot(b))/(na*nb))) if na>1e-12 and nb>1e-12 else None)
