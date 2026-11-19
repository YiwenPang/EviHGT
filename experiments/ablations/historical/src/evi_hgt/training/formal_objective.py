"""Production class-prior-free formal ranking objective for learning protocol."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


PRIMARY_OBJECTIVE_ID = "SYMMETRIC_PU_PAIRWISE_RANKING_V1"


@dataclass(frozen=True, slots=True)
class PairwiseRankingRisk:
    """Auditable exact all-pairs empirical risk for one PU channel."""

    total: Tensor
    positive_count: int
    unlabeled_count: int
    pair_count: int
    mean_margin: Tensor
    finite: bool
    temperature: float
    pairwise_chunk_size: int | None

    @property
    def loss(self) -> Tensor:
        """Compatibility-friendly name for the scalar channel loss."""

        return self.total


@dataclass(frozen=True, slots=True)
class DualChannelRankingRisk:
    """Independently reduced support and unsuitability ranking risks."""

    support: PairwiseRankingRisk
    unsuitability: PairwiseRankingRisk
    lambda_support: float
    lambda_unsuitability: float
    explicit_regularization_weight: float
    regularization: Tensor
    total: Tensor
    finite: bool

    @property
    def support_weight(self) -> float:
        return self.lambda_support

    @property
    def unsuitability_weight(self) -> float:
        return self.lambda_unsuitability


class SymmetricPUPairwiseRankingLoss(nn.Module):
    r"""Mean ``sigmoid(-(g_p-g_u)/tau)`` over every within-batch P×U pair."""

    def __init__(
        self,
        temperature: float = 1.0,
        pairwise_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if not self.temperature > 0.0:
            raise ValueError("temperature must be positive")
        if pairwise_chunk_size is not None and int(pairwise_chunk_size) <= 0:
            raise ValueError("pairwise chunk size must be positive or None")
        self.pairwise_chunk_size = (
            None if pairwise_chunk_size is None else int(pairwise_chunk_size)
        )

    def forward(
        self,
        positive_logits: Tensor,
        unlabeled_logits: Tensor,
    ) -> PairwiseRankingRisk:
        if positive_logits.numel() == 0:
            raise ValueError("positive ranking set must not be empty")
        if unlabeled_logits.numel() == 0:
            raise ValueError("unlabeled ranking set must not be empty")
        positive = positive_logits.float().reshape(-1)
        unlabeled = unlabeled_logits.float().reshape(-1)
        if not torch.isfinite(positive).all() or not torch.isfinite(unlabeled).all():
            raise ValueError("ranking logits must be finite")
        positive_count = int(positive.numel())
        unlabeled_count = int(unlabeled.numel())
        pair_count = positive_count * unlabeled_count
        chunk_size = self.pairwise_chunk_size
        if chunk_size is None or chunk_size >= pair_count:
            margins = (positive[:, None] - unlabeled[None, :]) / self.temperature
            loss_sum = torch.sigmoid(-margins).sum(dtype=torch.float32)
            margin_sum = margins.sum(dtype=torch.float32)
        else:
            loss_parts: list[Tensor] = []
            margin_parts: list[Tensor] = []
            for start in range(0, pair_count, chunk_size):
                stop = min(start + chunk_size, pair_count)
                flat = torch.arange(start, stop, device=positive.device)
                positive_index = torch.div(
                    flat, unlabeled_count, rounding_mode="floor"
                )
                unlabeled_index = torch.remainder(flat, unlabeled_count)
                margins = (
                    positive[positive_index] - unlabeled[unlabeled_index]
                ) / self.temperature
                loss_parts.append(torch.sigmoid(-margins).sum(dtype=torch.float32))
                margin_parts.append(margins.sum(dtype=torch.float32))
            loss_sum = torch.stack(loss_parts).sum(dtype=torch.float32)
            margin_sum = torch.stack(margin_parts).sum(dtype=torch.float32)
        total = loss_sum / pair_count
        mean_margin = margin_sum / pair_count
        finite = bool(torch.isfinite(total) and torch.isfinite(mean_margin))
        if not finite:
            raise FloatingPointError("pairwise ranking reduction produced non-finite output")
        return PairwiseRankingRisk(
            total=total,
            positive_count=positive_count,
            unlabeled_count=unlabeled_count,
            pair_count=pair_count,
            mean_margin=mean_margin,
            finite=finite,
            temperature=self.temperature,
            pairwise_chunk_size=chunk_size,
        )


class DualChannelPairwiseRankingObjective(nn.Module):
    """Normalize configured channel weights after independent P×U reductions."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        lambda_support: float = 1.0,
        lambda_unsuitability: float = 1.0,
        explicit_regularization_weight: float = 0.0,
        pairwise_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.channel_loss = SymmetricPUPairwiseRankingLoss(
            temperature=temperature,
            pairwise_chunk_size=pairwise_chunk_size,
        )
        self.lambda_support = float(lambda_support)
        self.lambda_unsuitability = float(lambda_unsuitability)
        self.explicit_regularization_weight = float(explicit_regularization_weight)
        for name, value in (
            ("lambda_support", self.lambda_support),
            ("lambda_unsuitability", self.lambda_unsuitability),
            ("explicit_regularization_weight", self.explicit_regularization_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.lambda_support + self.lambda_unsuitability <= 0.0:
            raise ValueError("at least one channel weight must be positive")

    def forward(
        self,
        *,
        support_positive_logits: Tensor,
        support_unlabeled_logits: Tensor,
        unsuitability_positive_logits: Tensor,
        unsuitability_unlabeled_logits: Tensor,
        regularization: Tensor | None = None,
    ) -> DualChannelRankingRisk:
        support = self.channel_loss(
            support_positive_logits, support_unlabeled_logits
        )
        unsuitability = self.channel_loss(
            unsuitability_positive_logits, unsuitability_unlabeled_logits
        )
        weight_sum = self.lambda_support + self.lambda_unsuitability
        channel_total = (
            self.lambda_support * support.total
            + self.lambda_unsuitability * unsuitability.total
        ) / weight_sum
        if regularization is None:
            if self.explicit_regularization_weight != 0.0:
                raise ValueError(
                    "a regularization tensor is required when its weight is nonzero"
                )
            regularization_value = torch.zeros_like(channel_total)
        else:
            regularization_value = regularization.float()
            if regularization_value.numel() != 1:
                raise ValueError("regularization must be a scalar tensor")
        total = (
            channel_total
            + self.explicit_regularization_weight * regularization_value
        )
        finite = bool(torch.isfinite(total))
        if not finite:
            raise FloatingPointError("dual-channel ranking objective is non-finite")
        return DualChannelRankingRisk(
            support=support,
            unsuitability=unsuitability,
            lambda_support=self.lambda_support,
            lambda_unsuitability=self.lambda_unsuitability,
            explicit_regularization_weight=self.explicit_regularization_weight,
            regularization=regularization_value,
            total=total,
            finite=finite,
        )
