"""The only experimental intervention is the scalar atomic ranking surrogate."""
import torch
from torch import nn
from torch.nn import functional as F

from evi_hgt.phase9.model import stable_objective
from evi_hgt.training.formal_objective import PairwiseRankingRisk, DualChannelPairwiseRankingObjective
from evi_hgt.phase85_part2b.pilot import risks_for_group
from evi_hgt.phase85_part3a_r2.pilot import conditional_group_risk
from .protocol import ARMS


def atomic_softplus(margin):
    """Normal autograd; preserve float64 for mathematical verification."""
    return .5 * F.softplus(-margin)


class SoftplusRanking(nn.Module):
    temperature = 1.
    pairwise_chunk_size = 4096

    def forward(self, positive_logits, unlabeled_logits):
        p, u = positive_logits.float().reshape(-1), unlabeled_logits.float().reshape(-1)
        if not p.numel() or not u.numel() or not torch.isfinite(p).all() or not torch.isfinite(u).all():
            raise ValueError("Finite nonempty raw-logit P/U streams required")
        count = p.numel() * u.numel()
        losses, margins = [], []
        for start in range(0, count, self.pairwise_chunk_size):
            i = torch.arange(start, min(start+self.pairwise_chunk_size, count), device=p.device)
            m = (p[torch.div(i, u.numel(), rounding_mode="floor")] - u[i % u.numel()]) / self.temperature
            losses.append(atomic_softplus(m).sum(dtype=torch.float32))
            margins.append(m.sum(dtype=torch.float32))
        total = torch.stack(losses).sum(dtype=torch.float32) / count
        mean = torch.stack(margins).sum(dtype=torch.float32) / count
        if not bool(torch.isfinite(total) and torch.isfinite(mean)):
            raise FloatingPointError("Nonfinite softplus reduction")
        return PairwiseRankingRisk(total, p.numel(), u.numel(), count, mean, True,
                                   self.temperature, self.pairwise_chunk_size)


def objective(arm):
    if arm == ARMS[0]:
        return stable_objective()
    if arm != ARMS[1]:
        raise PermissionError("Unregistered DEV95 loss arm")
    obj = DualChannelPairwiseRankingObjective(temperature=1., lambda_support=1.,
                                            lambda_unsuitability=1., pairwise_chunk_size=4096)
    obj.channel_loss = SoftplusRanking()
    return obj


def forward_risk(built, queries, graphs, extras=(), record=None, *, arm, obj=None, observer=None):
    def forward(g):
        result = built.forward(g.batch.model_batch, path_batch=g.batch.path_batch)
        if observer is not None: observer(g, result)
        return result
    predictions = tuple(forward(g) for g in graphs)
    other = tuple(forward(g) for g in extras)
    obj = objective(arm) if obj is None else obj
    risk = (risks_for_group(obj, tuple(g.batch for g in graphs), predictions) if record is None
            else conditional_group_risk(obj, queries, graphs, predictions, record, other))
    if not risk.finite:
        raise FloatingPointError("Nonfinite DEV95 objective")
    return risk, predictions, other
