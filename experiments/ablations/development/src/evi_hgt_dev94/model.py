"""Two narrowly scoped, instance-local fusion interventions on original M0.

D2 deliberately replays only QueryReadout after original path reasoning with
its original dropout RNG. No graph/path input or non-drug query component is
replaced. The ordinary readout invocation reserves precisely the old RNG use.
"""
from dataclasses import replace
import math

import torch
from torch import nn

from evi_hgt.phase9.identity import CellContract
from evi_hgt.phase9.model import build_model
from evi_hgt.phase9.state import tensor_digest
from evi_hgt.phase85_v2.features import DRUG_DIM

ARMS=('D0_RAW_LONG','D1_PRETRAINED_EARLY','D2_RAW_LATE_RESIDUAL')

class EarlyFusion(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.projections=nn.ModuleDict({k:nn.Linear(768,hidden,bias=False) for k in ('DRG','DIS')})
        for layer in self.projections.values():nn.init.zeros_(layer.weight)

class LateFusion(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.raw=nn.Linear(DRUG_DIM,hidden,bias=False)
        nn.init.zeros_(self.raw.weight)
        self.a=nn.Parameter(torch.tensor(math.log(.1/.9),dtype=torch.float32))

class DevelopmentModel:
    def __init__(self,arm,pretrained=None):
        if arm not in ARMS:raise ValueError('Exactly three predeclared DEV94 arms')
        if (arm==ARMS[1]) != (pretrained is not None):raise ValueError('Only D1 requires pretrained features')
        self.arm=arm;self.pretrained=pretrained
        self.base=build_model(CellContract.load('M0',0,0))
        self.model=self.base.model
        self.common_keys=tuple(self.model.state_dict())
        self.common_digest=tensor_digest(self.model.state_dict())
        self.hidden=self.model.encoder.config.hidden_dim
        self.drug_type=self.model.encoder.node_encoder.drug_type
        self.disease_type=self.model.encoder.node_encoder.disease_type
        self.handles=[];self.last={};self._active=False;self._replaying=False;self.fusion_enabled=True
        # No construction-time torch random draw escapes the local fork.
        with torch.random.fork_rng(devices=[]):
            if arm==ARMS[1]:self.model.add_module('dev94_fusion',EarlyFusion(self.hidden))
            elif arm==ARMS[2]:self.model.add_module('dev94_fusion',LateFusion(self.hidden))
        if arm==ARMS[1]:
            self.handles.append(self.model.encoder.node_encoder.register_forward_hook(self._early))
        elif arm==ARMS[2]:
            self.handles.append(self.model.query_readout.register_forward_pre_hook(self._capture_readout,with_kwargs=True))
            self.handles.append(self.model.path_reasoner.register_forward_hook(self._path_done))
            self.handles.append(self.model.path_fusion.register_forward_pre_hook(self._late))
        if tensor_digest({k:self.model.state_dict()[k] for k in self.common_keys})!=self.common_digest:
            raise ValueError('Fusion initialization altered original parameters')

    def metadata(self):
        self.base.metadata()
        new={n:p for n,p in self.model.named_parameters() if n.startswith('dev94_fusion.')}
        return dict(arm=self.arm,seed=1289779183,common_initial_tensor_sha256=self.common_digest,
                    new_parameters=sum(p.numel() for p in new.values()),
                    new_parameter_shapes={k:list(v.shape) for k,v in new.items()},
                    initialization='zero bias-free projections; D2 a=log(.1/.9)',
                    hidden_dim=self.hidden,pretrained=self.pretrained.metadata if self.pretrained else None,
                    original=self.base.metadata(),scope='DEVELOPMENT_ADAPTIVE_NOT_FORMAL_NOT_OUTER_NOT_FINAL')

    def _early(self,module,args,output):
        nodes,=args
        if nodes.node_id is None:raise ValueError('D1 requires certified global node IDs')
        x,mask=self.pretrained.gather(nodes.node_id,nodes.node_type_id,self.drug_type,self.disease_type,output.device)
        if x.shape!=(nodes.num_nodes,768) or mask.shape!=(nodes.num_nodes,) or mask.dtype!=torch.bool:
            raise ValueError('Pretrained shape/mask mismatch')
        if x.requires_grad or not torch.isfinite(x).all():raise ValueError('Pretrained vectors must be finite and frozen')
        result=output.clone();counts={}
        for name,type_id in (('DRG',self.drug_type),('DIS',self.disease_type)):
            typed=nodes.node_type_id==type_id
            active=typed&mask
            # Zero unavailable contribution, including after projections have learned.
            if active.any() and self.fusion_enabled:result[active]=output[active]+self.model.dev94_fusion.projections[name](x[active]).to(output.dtype)
            counts[name]={'available':int(active.sum()),'total':int(typed.sum())}
        self.last['early']=counts
        return result

    def _capture_readout(self,module,args,kwargs):
        if self._replaying:return
        if not self._active:raise RuntimeError('D2 must execute through its bounded wrapper')
        self._core,self._queries=args
        self._readout_kwargs=kwargs
        self._readout_rng=torch.get_rng_state().clone()
        self._readout_cuda=(torch.cuda.get_rng_state(self._batch.nodes.node_type_id.device).clone()
                            if self._batch.nodes.node_type_id.is_cuda else None)

    def _path_done(self,module,args,output):
        self._path_completed=True

    def _late(self,module,args):
        if not self._path_completed:raise ValueError('D2 requires completed original path reasoning')
        _,context,available=args
        core=self._core;q=self._queries;nodes=self._batch.nodes
        if not torch.all(nodes.node_type_id[q.drug_node_index]==self.drug_type):raise ValueError('Late target is not DRG')
        if q.drug_node_index.unique().numel()!=q.num_queries:raise ValueError('Repeated local drug readout endpoints')
        raw=nodes.features[q.drug_node_index,:DRUG_DIM]
        mask=raw[:,-2:].bool().any(dim=1)
        branch=self.model.dev94_fusion
        delta=(branch.a.sigmoid()*branch.raw(raw)*mask.unsqueeze(-1)).to(core.node_embeddings.dtype)
        if not self.fusion_enabled:delta=delta*0
        final=core.node_embeddings.clone()
        graph=final[q.drug_node_index].clone()
        final[q.drug_node_index]=graph+delta
        changed=replace(core,node_embeddings=final)
        devices=[nodes.node_type_id.device.index or 0] if nodes.node_type_id.is_cuda else []
        self._replaying=True
        try:
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(self._readout_rng)
                if self._readout_cuda is not None:torch.cuda.set_rng_state(self._readout_cuda,nodes.node_type_id.device)
                fused=self.model.query_readout(changed,q,**self._readout_kwargs)
        finally:self._replaying=False
        self.last['late']=dict(alpha=float(branch.a.sigmoid().detach()),W_raw_norm=float(branch.raw.weight.detach().float().norm()),
            graph_norm=float(graph.detach().float().norm()),residual_norm=float(delta.detach().float().norm()),
            raw_available=int(mask.sum()),raw_queries=mask.numel(),path_completed_before_fusion=True,
            disease_unchanged=bool(torch.equal(final[q.disease_node_index],core.node_embeddings[q.disease_node_index])),
            phase4_query_unchanged=changed.query_embedding is core.query_embedding)
        self._late_calls+=1
        return fused,context,available

    def forward(self,batch,*,path_batch):
        if path_batch is None:raise ValueError('The frozen path batch, including empty batches, is required')
        if self._active:raise RuntimeError('Reentrant DEV94 forward is unsupported')
        self._active=True;self._batch=batch;self._path_completed=False;self._late_calls=0;self.last={}
        try:
            result=self.base.forward(batch,path_batch=path_batch)
            if self.arm==ARMS[2] and self._late_calls!=1:raise ValueError('Both-head shared late readout was not consumed exactly once')
            return result
        finally:
            self._active=False
            for key in ('_batch','_core','_queries','_readout_kwargs','_readout_rng','_readout_cuda'):
                if hasattr(self,key):delattr(self,key)

def build_arm(arm,pretrained=None):return DevelopmentModel(arm,pretrained)

def make_optimizer(built):
    built.metadata()
    return torch.optim.AdamW(built.model.parameters(),lr=5e-5,weight_decay=1e-5,betas=(.9,.999),eps=1e-8,foreach=None,fused=None)
