"""Pure-PyTorch packed segment operations with deterministic empty groups."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate(values: Tensor, index: Tensor, num_segments: int) -> None:
    if num_segments < 0:
        raise ValueError("num_segments must be non-negative")
    if values.ndim < 1:
        raise ValueError("segment values must have at least one dimension")
    if index.ndim != 1 or index.numel() != values.shape[0]:
        raise ValueError("segment index must match the first value dimension")
    if index.dtype != torch.long:
        raise ValueError("segment index must use torch.long")
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= num_segments):
        raise ValueError("segment index is out of range")


def _expanded_index(index: Tensor, values: Tensor) -> Tensor:
    shape = (index.shape[0],) + (1,) * (values.ndim - 1)
    return index.reshape(shape).expand_as(values)


def segment_sum(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Sum rows into segments and return zeros for empty groups."""

    _validate(values, index, num_segments)
    output = values.new_zeros((num_segments,) + tuple(values.shape[1:]))
    if values.shape[0]:
        output.scatter_add_(0, _expanded_index(index, values), values)
    return output


def segment_mean(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Mean rows into segments and return zeros for empty groups."""

    summed = segment_sum(values, index, num_segments)
    counts = values.new_zeros(num_segments)
    if index.numel():
        counts.scatter_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    count_shape = (num_segments,) + (1,) * (values.ndim - 1)
    return summed / counts.clamp_min(1).reshape(count_shape)


def segment_softmax(logits: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Normalize logits independently within each segment.

    Max reduction, exponentiation, and denominator accumulation use FP32 even
    for BF16 inputs. The normalized weights are returned in the input dtype.
    """

    _validate(logits, index, num_segments)
    if not logits.is_floating_point():
        raise ValueError("segment softmax logits must be floating point")
    if not logits.shape[0]:
        return logits.clone()
    work = logits.float()
    expanded = _expanded_index(index, work)
    maxima = work.new_full((num_segments,) + tuple(work.shape[1:]), -torch.inf)
    maxima.scatter_reduce_(
        0,
        expanded,
        work,
        reduce="amax",
        include_self=True,
    )
    shifted = work - maxima[index]
    exponentials = shifted.exp()
    denominator = work.new_zeros((num_segments,) + tuple(work.shape[1:]))
    denominator.scatter_add_(0, expanded, exponentials)
    result = exponentials / denominator[index].clamp_min(torch.finfo(work.dtype).tiny)
    return result.to(dtype=logits.dtype)

