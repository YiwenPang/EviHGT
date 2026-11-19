"""Production DDA construction for the frozen model variant catalog."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.evi_hgt import EviHGT
from evi_hgt.models.input_contract import EviHGTBatch
from evi_hgt.models.path_reasoning import PathReasoningConfig
from evi_hgt.models.variants import (
    IMPLEMENTED_STATUS,
    ModelVariantId,
    ModelVariantSpec,
    apply_variant_config,
    get_variant_spec,
)
from evi_hgt.training.prediction import EviHGTForDDA, QueryPrediction


class VariantUnavailableError(RuntimeError):
    """Raised when a fail-closed catalog entry is requested as a model."""


@dataclass(frozen=True, slots=True)
class BuiltModelVariant:
    spec: ModelVariantSpec
    core_config: EviHGTConfig
    model: EviHGTForDDA

    def forward(self, batch: EviHGTBatch, **kwargs: Any) -> QueryPrediction:
        """Apply the catalog-bound path switch on the shared DDA wrapper."""

        return self.model(
            batch,
            use_path_reasoning=self.spec.use_path_reasoning,
            **kwargs,
        )


def build_model_variant(
    variant_id: ModelVariantId | str,
    *,
    base_config: EviHGTConfig,
    path_config: PathReasoningConfig | None,
    dropout: float | None = None,
    allow_full_without_path_for_compatibility_test: bool = False,
) -> BuiltModelVariant:
    """Instantiate one implemented variant without duplicating model code."""

    spec = get_variant_spec(variant_id)
    if spec.status != IMPLEMENTED_STATUS:
        raise VariantUnavailableError(f"{spec.variant_id.value}: {spec.status}")
    core_config = apply_variant_config(base_config, spec.variant_id)
    selected_path_config = path_config if spec.use_path_reasoning else None
    if spec.use_path_reasoning and selected_path_config is None:
        if not (
            spec.variant_id is ModelVariantId.FULL_EVIHGT
            and allow_full_without_path_for_compatibility_test
        ):
            raise ValueError(
                f"{spec.variant_id.value} requires the frozen path encoder path config"
            )
    if selected_path_config is not None:
        if selected_path_config.core_hidden_dim != core_config.hidden_dim:
            raise ValueError("path and core hidden dimensions must match")
        core_config = replace(core_config, return_edge_context=True)
    model = EviHGTForDDA(
        EviHGT(core_config),
        dropout=dropout,
        path_config=selected_path_config,
    )
    for layer in model.encoder.layers:
        layer.attention.use_logic_mass_conditioning = spec.use_logic_mass_conditioning
    return BuiltModelVariant(spec=spec, core_config=core_config, model=model)


__all__ = [
    "BuiltModelVariant",
    "VariantUnavailableError",
    "build_model_variant",
]
