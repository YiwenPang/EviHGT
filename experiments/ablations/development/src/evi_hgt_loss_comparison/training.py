"""Original finite-zero AdamW semantics, selected fixed atomic objective."""
import math
import time
import torch
from evi_hgt_feature_baseline.training import assemble_group
from evi_hgt_feature_baseline_recovery.training import inspect_gradients
from .losses import forward_risk


def execute_update(assembler, built, optimizer, queries, record=None, *, arm, bf16=True):
    t = time.monotonic()
    graphs, extras = assemble_group(assembler, queries, record)
    preparation = time.monotonic()-t
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        risk, _, _ = forward_risk(built, queries, graphs, extras, record, arm=arm)
    torch.cuda.synchronize(); forward = time.monotonic()-t; t = time.monotonic()
    losses = dict(loss=float(risk.total.detach()), support_loss=float(risk.support.total.detach()),
                  contra_loss=float(risk.unsuitability.total.detach()))
    if not all(math.isfinite(v) for v in losses.values()):
        raise FloatingPointError("Nonfinite loss")
    risk.total.backward()
    parameters = list(built.model.parameters())
    gradients = inspect_gradients(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(); backward = time.monotonic()-t
    return dict(**losses, **gradients, gradient_norm_before_clip=float(norm), clipped=bool(norm>1),
        queries=len(graphs)+len(extras), original_queries=len(graphs), conditional_extra_U_queries=len(extras),
        support_pairs=risk.support.pair_count, contra_pairs=risk.unsuitability.pair_count,
        preparation_seconds=preparation, forward_seconds=forward, backward_optimizer_seconds=backward,
        seconds=preparation+forward+backward)
