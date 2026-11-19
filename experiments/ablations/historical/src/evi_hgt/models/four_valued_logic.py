"""Differentiable latent four-valued evidence masses with validity mask."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class LogicState:
    masses: Tensor
    available: Tensor


class FourValuedLogic(nn.Module):
    """Map independent support/counter strengths to T/F/B/U masses."""

    def forward(
        self,
        support_score: Tensor,
        counter_score: Tensor,
        available: Tensor,
    ) -> LogicState:
        if support_score.shape != counter_score.shape or support_score.shape != available.shape:
            raise ValueError("support, counter, and availability must have equal shape")
        if available.dtype != torch.bool:
            raise ValueError("logic availability must be Boolean")
        if not torch.isfinite(support_score).all() or not torch.isfinite(
            counter_score
        ).all():
            raise ValueError("logic strengths must be finite")
        if torch.any((support_score < 0) | (support_score > 1)) or torch.any(
            (counter_score < 0) | (counter_score > 1)
        ):
            raise ValueError("logic strengths must be in [0, 1]")
        # Product masses are an exact probability partition in real arithmetic.
        # Compute them in FP32 so BF16 rounding does not break normalization.
        support = support_score.float()
        counter = counter_score.float()
        masses = torch.stack(
            (
                support * (1 - counter),
                (1 - support) * counter,
                support * counter,
                (1 - support) * (1 - counter),
            ),
            dim=-1,
        )
        masses = torch.where(
            available.unsqueeze(-1), masses, torch.zeros_like(masses)
        )
        if not torch.isfinite(masses).all():
            raise FloatingPointError("Four-valued masses are non-finite")
        if available.any():
            sums = masses[available].float().sum(dim=-1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
                raise FloatingPointError("Available four-valued masses do not sum to one")
        return LogicState(masses=masses, available=available)
