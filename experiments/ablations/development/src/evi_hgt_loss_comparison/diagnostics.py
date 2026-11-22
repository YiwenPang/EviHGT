"""Fixed hash panels; analytic loss-shape comparison on identical saved raw scores."""
import ablation_artifacts as artifacts
from ablation_protocol_ids import FUSION_LAYER_NAME
from contextlib import contextmanager
from dataclasses import asdict
import copy
import time
import numpy as np
import torch
from torch import nn

from evi_hgt_feature_baseline.diagnostics import preserve_training_state, selected_indices, query_exposure
from evi_hgt_feature_baseline.training import assemble_group
from .losses import objective, forward_risk
from .protocol import G, TOTAL, digest

LOW_DERIVATIVE = 1e-6
LOW_NORM = 1e-12


@contextmanager
def isolated(built, optimizer, assembler):
    sentinel = object()
    attrs = {k:copy.deepcopy(getattr(built,k)) if hasattr(built,k) else sentinel
             for k in ("last","fusion_enabled","_path_completed","_late_calls")}
    counters = {k:copy.deepcopy(getattr(assembler,k)) for k in
                ("_counts","_domains","_adapter_seconds","_test_graph_digest_seconds","test_graph_digests") if hasattr(assembler,k)}
    try:
        with preserve_training_state(built.model, optimizer):
            yield
    finally:
        for k,v in attrs.items():
            if v is sentinel:
                if hasattr(built,k): delattr(built,k)
            else: setattr(built,k,v)
        for k,v in counters.items(): setattr(assembler,k,v)


def panel_indices(groups):
    key = lambda i: digest([artifacts.LOSS_PANEL_NAMESPACE, [asdict(q) for q in groups[i]]])
    return dict(legacy20=selected_indices(groups), global32=sorted(range(G),key=key)[:32],
                conditional32=sorted(range(G,TOTAL),key=key)[:32])


def derivative_summary(margins):
    m = np.asarray(margins,dtype=np.float64)
    if not len(m) or not np.isfinite(m).all(): raise ValueError("Finite diagnostic margins required")
    e = np.exp(-np.abs(m))
    sigmoid = e/(1+e)**2
    softplus = .5*np.where(m>=0,e/(1+e),1/(1+e))
    q = lambda a: dict(zip(("min","q01","q10","q25","median","q75","q90","q99","max"),
                          np.quantile(a,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist()))
    result = dict(pairs=len(m),margins=m.tolist(),margin_quantiles=q(m),
                  positive=int((m>0).sum()),negative=int((m<0).sum()),zero=int((m==0).sum()),
                  derivative_threshold=LOW_DERIVATIVE, reduction_dtype="float64")
    for name,d in (("sigmoid",sigmoid),("softplus",softplus)):
        result[name]=dict(abs_derivatives=d.tolist(),quantiles=q(d),
            low_derivative=int((d<LOW_DERIVATIVE).sum()),
            violated_low_derivative=int(((m<0)&(d<LOW_DERIVATIVE)).sum()))
    return result


def gradient_comparison(a,b):
    aa=bb=ab=0.
    for x,y in zip(a,b,strict=True):
        if x is not None: aa+=float(x.detach().double().square().sum())
        if y is not None: bb+=float(y.detach().double().square().sum())
        if x is not None and y is not None: ab+=float((x.detach().double()*y.detach().double()).sum())
    na,nb=aa**.5,bb**.5
    valid=na>LOW_NORM and nb>LOW_NORM
    return dict(support_norm=na,contra_norm=nb,cosine=max(-1.,min(1.,ab/(na*nb))) if valid else None,
                status="DEFINED" if valid else "UNDEFINED/LOW_NORM",low_norm_threshold=LOW_NORM,reduction_dtype="float64")


class CaptureCalls(nn.Module):
    def __init__(self, actual):
        super().__init__(); self.actual=actual; self.calls=[]
        self.temperature=actual.temperature; self.pairwise_chunk_size=actual.pairwise_chunk_size

    def forward(self,p,u):
        r=self.actual(p,u)
        self.calls.append(dict(positive_logits=p.detach().float().cpu().reshape(-1).tolist(),
                              unlabeled_logits=u.detach().float().cpu().reshape(-1).tolist(),
                              pairs=r.pair_count))
        return r


def diagnose(assembler,built,optimizer,groups,records,step,*,arm,panels=None,ui=None):
    panels = panel_indices(groups) if panels is None else panels
    selected = sorted(set(i for values in panels.values() for i in values))
    params=list(built.model.named_parameters())
    shared_indices=[i for i,(n,p) in enumerate(params) if not n.startswith(("support_head.","contraindication_unsuitability_head."))]
    rows=[]; cost=dict(preparation_seconds=0.,forward_seconds=0.,backward_seconds=0.)
    start=time.monotonic()
    with isolated(built,optimizer,assembler):
        for j,i in enumerate(selected):
            record=records[i-G] if i>=G else None
            t=time.monotonic(); graphs,extras=assemble_group(assembler,groups[i],record)
            cost["preparation_seconds"]+=time.monotonic()-t
            obj=objective(arm); trace=CaptureCalls(obj.channel_loss); obj.channel_loss=trace
            exposures=[]
            torch.cuda.synchronize(); t=time.monotonic()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                risk,predictions,other=forward_risk(built,groups[i],graphs,extras,record,arm=arm,obj=obj,
                    observer=lambda g,p:exposures.append(query_exposure(g,p,built)))
            torch.cuda.synchronize(); cost["forward_seconds"]+=time.monotonic()-t
            t=time.monotonic()
            gs=torch.autograd.grad(risk.support.total,[p for n,p in params],retain_graph=True,allow_unused=True)
            gc=torch.autograd.grad(risk.unsuitability.total,[p for n,p in params],allow_unused=True)
            if not all(g is None or bool(torch.isfinite(g).all()) for g in (*gs,*gc)):
                raise FloatingPointError("Nonfinite diagnostic gradient")
            torch.cuda.synchronize(); cost["backward_seconds"]+=time.monotonic()-t
            channel={}
            for name,calls in (("support",trace.calls[:-1]),("contra",trace.calls[-1:])):
                margins=[p-u for c in calls for p in c["positive_logits"] for u in c["unlabeled_logits"]]
                channel[name]=dict(calls=calls,**derivative_summary(margins))
            inputs={}
            for k,(n,p) in enumerate(params):
                if n.startswith(FUSION_LAYER_NAME + ".") or any(x in n for x in ("drug_encoder","disease_encoder","drug_gate","disease_gate")):
                    inputs[n]=dict(support_norm=None if gs[k] is None else float(gs[k].double().norm()),
                                   contra_norm=None if gc[k] is None else float(gc[k].double().norm()))
            rows.append(dict(group_index=i,member_sha256=digest([asdict(q) for q in groups[i]]),
                record_sha256=digest(record) if record else None,channels=channel,
                head_gradients=gradient_comparison(gs,gc),
                shared_gradients=gradient_comparison([gs[k] for k in shared_indices],[gc[k] for k in shared_indices]),
                input_gradients=inputs,availability=copy.deepcopy(built.last),
                graph_exposures=exposures))
            if ui is not None: ui.preparation_progress("Fixed training diagnostic groups",j+1,len(selected))
    return dict(arm=arm,step=step,panels=panels,records=rows,seconds=time.monotonic()-start,**cost,
                actual_query_forwards=sum(len(groups[i])+(len(records[i-G]["extra_pairs"]) if i>=G else 0) for i in selected),
                training_state_unchanged=True,scope="LOCAL_FIXED_TRAINING_DIAGNOSTIC_NOT_POPULATION_OR_CLINICAL")
