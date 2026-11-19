"""Unchanged original B11/B10 group objective and bounded real optimizer steps."""
import math
import time
import torch
from evi_hgt.ranking_protocol.model import stable_objective
from evi_hgt.inner_validation.pilot import risks_for_group
from evi_hgt.conditional_sampling.pilot import conditional_group_risk
from evi_hgt.training.supervision import make_unlabeled_query_supervision

def assemble_group(assembler,queries,record=None,device='cuda'):
    extra=() if record is None else tuple(make_unlabeled_query_supervision(*p) for p in record['extra_pairs'])
    return (tuple(assembler.assemble(q,domain='training',device=device) for q in queries),
            tuple(assembler.assemble(q,domain='training',device=device) for q in extra))

def forward_risk(built,queries,graphs,extras=(),record=None):
    predictions=tuple(built.forward(g.batch.model_batch,path_batch=g.batch.path_batch) for g in graphs)
    other=tuple(built.forward(g.batch.model_batch,path_batch=g.batch.path_batch) for g in extras)
    obj=stable_objective()
    risk=(risks_for_group(obj,tuple(g.batch for g in graphs),predictions) if record is None
          else conditional_group_risk(obj,queries,graphs,predictions,record,other))
    if not risk.finite:raise FloatingPointError('Nonfinite original dual PU loss')
    return risk,predictions,other

def execute_update(assembler,built,optimizer,queries,record=None):
    t=time.monotonic();graphs,extras=assemble_group(assembler,queries,record)
    prep=time.monotonic()-t
    optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):risk,_,_=forward_risk(built,queries,graphs,extras,record)
    torch.cuda.synchronize();forward=time.monotonic()-t;t=time.monotonic()
    risk.total.backward()
    parameters=list(built.model.parameters())
    if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in parameters):raise FloatingPointError('No active gradient')
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step();torch.cuda.synchronize();backward=time.monotonic()-t
    losses=dict(loss=float(risk.total.detach()),support_loss=float(risk.support.total.detach()),
                contra_loss=float(risk.unsuitability.total.detach()))
    if not all(math.isfinite(v) for v in losses.values()):raise FloatingPointError('Nonfinite update loss')
    return dict(**losses,finite=True,gradient_norm_before_clip=float(norm),clipped=bool(norm>1),
        queries=len(graphs)+len(extras),original_queries=len(graphs),conditional_extra_U_queries=len(extras),
        preparation_seconds=prep,forward_seconds=forward,backward_optimizer_seconds=backward,
        seconds=prep+forward+backward)
