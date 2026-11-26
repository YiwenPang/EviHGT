"""Stable sigmoid control used in the paired development loss study."""
import torch

from evi_hgt.losses import Risk


class _StableNegativeSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, margin):
        ctx.save_for_backward(margin)
        with torch.autocast(device_type=margin.device.type, enabled=False):
            return torch.sigmoid(-margin)

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("The sigmoid control supports first-order backward only")
        margin, = ctx.saved_tensors
        with torch.autocast(device_type=margin.device.type, enabled=False):
            q = torch.exp(-margin.abs())
            return grad_output * (-q / (1 + q).square())


def sigmoid_risk(positive, unlabeled):
    """Mean sigmoid(-margin) in FP32 with the original stable first derivative."""
    p, u = positive.float().reshape(-1), unlabeled.float().reshape(-1)
    if not p.numel() or not u.numel() or not torch.isfinite(p).all() or not torch.isfinite(u).all():
        raise ValueError("Finite nonempty raw-logit P/U streams are required")
    count = p.numel() * u.numel()
    chunk_size = 4096
    if count <= chunk_size:
        margins = p[:, None] - u[None, :]
        loss_sum = _StableNegativeSigmoid.apply(margins).sum(dtype=torch.float32)
    else:
        sums = []
        for start in range(0, count, chunk_size):
            flat = torch.arange(start, min(start + chunk_size, count), device=p.device)
            margins = p[torch.div(flat, u.numel(), rounding_mode="floor")] - u[flat % u.numel()]
            sums.append(_StableNegativeSigmoid.apply(margins).sum(dtype=torch.float32))
        loss_sum = torch.stack(sums).sum(dtype=torch.float32)
    total = loss_sum / count
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite sigmoid reduction")
    return total


def sigmoid_group_risk(queries, predictions, *, record, extra_predictions):
    """Use the same positive-occurrence and head weights as the softplus arm."""
    pairs = tuple(zip(queries, predictions, strict=True))

    def stream(channel, mask):
        logits = [getattr(prediction, channel + "_logit").reshape(-1)
                  for query, prediction in pairs if getattr(getattr(query, channel), mask)]
        if not logits:
            raise ValueError(f"Missing {channel} {mask} stream in training group")
        return torch.cat(logits)

    support_p = stream("support", "observed_mask")
    support_u = stream("support", "pu_risk_eligible")
    contra_p = stream("unsuitability", "observed_mask")
    contra_u = stream("unsuitability", "pu_risk_eligible")
    if record is None and extra_predictions:
        raise ValueError("Global training cannot consume conditional predictions")
    local_pairs = {} if record is None else record["disease_unlabeled_pairs"]
    if not any(local_pairs.values()):
        support = sigmoid_risk(support_p, support_u)
    else:
        lookup = {(query.drug_id, query.disease_id): prediction.support_logit
                  for query, prediction in pairs}
        lookup.update((tuple(pair), prediction.support_logit) for pair, prediction in
                      zip(record["extra_pairs"], extra_predictions, strict=True))
        diseases = [query.disease_id for query in queries if query.support.observed_mask]
        weighted = []
        for disease in dict.fromkeys(diseases):
            indices = [index for index, name in enumerate(diseases) if name == disease]
            local = local_pairs[disease]
            unlabeled = lookup[tuple(local[0])] if local else support_u
            weighted.append(sigmoid_risk(support_p[indices], unlabeled) * (len(indices) / len(diseases)))
        support = torch.stack(weighted).sum()
    contra = sigmoid_risk(contra_p, contra_u)
    return Risk(support, contra, (support + contra) / 2)
