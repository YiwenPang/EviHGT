"""Frozen half-softplus empirical PU ranking with independent channels.

Unlabeled pairs are comparison samples, not confirmed biological negatives.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Risk:
    support: torch.Tensor
    contra: torch.Tensor
    total: torch.Tensor


def half_softplus(positive, unlabeled, *, chunk_size=4096):
    """Exact original FP32 all-pairs reduction: mean(0.5 softplus(-(P-U)))."""
    p, u = positive.float().reshape(-1), unlabeled.float().reshape(-1)
    if not p.numel() or not u.numel() or not torch.isfinite(p).all() or not torch.isfinite(u).all():
        raise ValueError("Finite nonempty raw-logit P/U streams are required")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    count = p.numel() * u.numel()
    losses = []
    for start in range(0, count, chunk_size):
        i = torch.arange(start, min(start + chunk_size, count), device=p.device)
        margin = p[torch.div(i, u.numel(), rounding_mode="floor")] - u[i % u.numel()]
        losses.append((0.5 * F.softplus(-margin)).sum(dtype=torch.float32))
    total = torch.stack(losses).sum(dtype=torch.float32) / count
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite half-softplus reduction")
    return total


def dual_risk(support_p, support_u, contra_p, contra_u, *, support_diseases=None, local_u=None):
    """Preserve positive-occurrence weights and equal normalized channel weights."""
    local_u = {} if local_u is None else local_u
    if local_u:
        positive = support_p.float().reshape(-1)
        if support_diseases is None or len(support_diseases) != positive.numel() or support_u.numel() != 1:
            raise ValueError("Conditional support requires one global U and each P disease")
        if set(local_u) - set(support_diseases) or any(v.numel() != 1 for v in local_u.values()):
            raise ValueError("Each local U must match a positive disease with m=1")
        parts = []
        for disease in dict.fromkeys(support_diseases):
            indices = [i for i, value in enumerate(support_diseases) if value == disease]
            loss = half_softplus(positive[indices], local_u.get(disease, support_u))
            parts.append(loss * (len(indices) / positive.numel()))
        support = torch.stack(parts).sum()
    else:
        support = half_softplus(support_p, support_u)
    contra = half_softplus(contra_p, contra_u)
    total = (support + contra) / 2
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite dual-channel risk")
    return Risk(support, contra, total)


def group_risk(queries, predictions, *, record=None, extra_predictions=()):
    """Read the supervision mask for each head and preserve unobserved roles."""
    if len(queries) != len(predictions):
        raise ValueError("Query and prediction counts differ")
    streams = {}
    for channel in ("support", "unsuitability"):
        for kind in ("p", "u"):
            values = [getattr(pred, channel + "_logit").reshape(-1)
                      for query, pred in zip(queries, predictions, strict=True)
                      if (getattr(query, channel).observed_mask if kind == "p"
                          else getattr(query, channel).pu_risk_eligible)]
            if not values:
                raise ValueError(f"Missing {channel} {kind} stream in a complete training group")
            streams[channel + "_" + kind] = torch.cat(values)
    local = {}
    diseases = [q.disease_id for q in queries if q.support.observed_mask]
    if record is not None:
        if len(extra_predictions) != len(record["extra_pairs"]):
            raise ValueError("Conditional extra prediction count changed")
        lookup = {(q.drug_id, q.disease_id): p.support_logit
                  for q, p in zip(queries, predictions, strict=True)}
        for pair, pred in zip(record["extra_pairs"], extra_predictions, strict=True):
            if tuple(pair) in lookup:
                raise ValueError("Redundant conditional extra query")
            lookup[tuple(pair)] = pred.support_logit
        for disease, pairs in record["disease_unlabeled_pairs"].items():
            if pairs:
                if len(pairs) != 1 or pairs[0][1] != disease:
                    raise ValueError("Conditional U disease or count changed")
                local[disease] = lookup[tuple(pairs[0])]
    elif extra_predictions:
        raise ValueError("Global training cannot consume conditional extra predictions")
    return dual_risk(streams["support_p"], streams["support_u"],
                     streams["unsuitability_p"], streams["unsuitability_u"],
                     support_diseases=diseases, local_u=local)
