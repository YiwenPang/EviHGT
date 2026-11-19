"""Frozen config-driven model variants for variant catalog."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

from evi_hgt.models.config import EviHGTConfig


IMPLEMENTED_STATUS = "IMPLEMENTED"
PAIR_MLP_UNAVAILABLE_STATUS = (
    "NOT_IMPLEMENTED_FEATURE_CONTRACT_NOT_DEFENSIBLE"
)


class ModelVariantId(str, Enum):
    FULL_EVIHGT = "FULL_EVIHGT"
    BASE_HGT = "BASE_HGT"
    PAIR_MLP = "PAIR_MLP"
    NO_EVIDENCE_CONTEXT = "NO_EVIDENCE_CONTEXT"
    NO_FOUR_VALUED_LOGIC = "NO_FOUR_VALUED_LOGIC"
    NO_QUERY_CONDITIONING = "NO_QUERY_CONDITIONING"
    NO_PATH_REASONING = "NO_PATH_REASONING"
    NO_PROVENANCE = "NO_PROVENANCE"
    NO_LOGIC_MASS_CONDITIONING = "NO_LOGIC_MASS_CONDITIONING"


@dataclass(frozen=True, slots=True)
class ModelVariantSpec:
    variant_id: ModelVariantId
    status: str
    requirement: str
    disabled_mechanisms: tuple[str, ...]
    preserved_mechanisms: tuple[str, ...]
    config_diff: tuple[tuple[str, Any], ...]
    use_path_reasoning: bool
    rationale: str

    @property
    def config_overrides(self) -> dict[str, Any]:
        return dict(self.config_diff)

    @property
    def use_logic_mass_conditioning(self) -> bool:
        # A variant-bound switch, not a new sweep/configuration axis. Keep all
        # historical config dictionaries and state-dict shapes unchanged.
        return self.variant_id is not ModelVariantId.NO_LOGIC_MASS_CONDITIONING

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["variant_id"] = self.variant_id.value
        payload["config_diff"] = self.config_overrides
        return payload


_COMMON_PRESERVED = (
    "PHASE3_GRAPH",
    "COLD_SAFE_NODE_INITIALIZATION",
    "HIDDEN_DIM_LAYERS_HEADS",
    "PRE_LN_RESIDUAL",
    "QUERY_ENDPOINT_READOUT",
    "DUAL_SUPPORT_UNSUITABILITY_HEADS",
    "SYMMETRIC_PU_PAIRWISE_RANKING_V1",
    "FROZEN_CANDIDATE_AND_EVALUATION_PROTOCOLS",
)


_VARIANTS = (
    ModelVariantSpec(
        ModelVariantId.FULL_EVIHGT,
        IMPLEMENTED_STATUS,
        "PRIMARY",
        (),
        _COMMON_PRESERVED
        + (
            "EVIDENCE_PROPERTY_PROVENANCE_CONTEXT",
            "FOUR_VALUED_LOGIC_MESSAGES",
            "QUERY_CONDITIONED_EVIDENCE_ATTENTION",
            "PATH_REASONING",
        ),
        (),
        True,
        "Current complete Phase 7.2/7.3.2 EviHGT model.",
    ),
    ModelVariantSpec(
        ModelVariantId.BASE_HGT,
        IMPLEMENTED_STATUS,
        "REQUIRED_BASELINE",
        (
            "EVIDENCE_PROPERTY_PROVENANCE_CONTEXT",
            "PROPOSITION_SCOPED_EVIDENCE_BRANCHES",
            "FOUR_VALUED_LOGIC_MESSAGES",
            "QUERY_CONDITIONED_EVIDENCE_ATTENTION",
            "PATH_REASONING",
        ),
        _COMMON_PRESERVED + ("NODE_TYPES", "RELATION_TYPES", "HETEROGENEOUS_MESSAGE_PASSING"),
        (
            ("use_evidence_context", False),
            ("use_property", False),
            ("use_evidence", False),
            ("use_provenance", False),
            ("use_polarity_branches", False),
            ("use_four_valued_logic", False),
            ("use_query_conditioning", False),
        ),
        False,
        "Typed relation-only heterogeneous Transformer baseline.",
    ),
    ModelVariantSpec(
        ModelVariantId.PAIR_MLP,
        PAIR_MLP_UNAVAILABLE_STATUS,
        "OPTIONAL_BASELINE",
        ("MESSAGE_PASSING", "EDGE_AND_PATH_CONTEXT"),
        (
            "FORMAL_LABEL_OBJECTIVE_AND_EVALUATION_PROTOCOLS_IF_FUTURE_FEATURES_EXIST",
        ),
        (),
        False,
        (
            "Formal node features are unavailable and node-ID embeddings are disabled; "
            "type-only endpoints collapse every drug and every disease to identical "
            "initial representations."
        ),
    ),
    ModelVariantSpec(
        ModelVariantId.NO_EVIDENCE_CONTEXT,
        IMPLEMENTED_STATUS,
        "REQUIRED_ABLATION",
        ("EVIDENCE_PROPERTY_PROVENANCE_EDGE_CONTEXT",),
        _COMMON_PRESERVED + ("RELATION_ONLY_CONTEXT", "QUERY_CONDITIONING", "PATH_REASONING"),
        (("use_evidence_context", False),),
        True,
        "Relation-only context replaces the enhanced evidence context.",
    ),
    ModelVariantSpec(
        ModelVariantId.NO_FOUR_VALUED_LOGIC,
        IMPLEMENTED_STATUS,
        "REQUIRED_ABLATION",
        ("FOUR_VALUED_LOGIC_MESSAGE_TRANSFORMS",),
        _COMMON_PRESERVED + ("GENERAL_EVIDENCE_PROPERTY_CONTEXT", "QUERY_CONDITIONING", "PATH_REASONING"),
        (("use_four_valued_logic", False),),
        True,
        "Ordinary typed source-value messages replace T/F/B/U-conditioned values.",
    ),
    ModelVariantSpec(
        ModelVariantId.NO_QUERY_CONDITIONING,
        IMPLEMENTED_STATUS,
        "REQUIRED_ABLATION",
        ("QUERY_CONDITIONED_EVIDENCE_ATTENTION_TERM",),
        _COMMON_PRESERVED + ("QUERY_ENCODER", "FINAL_QUERY_READOUT", "EVIDENCE_AND_LOGIC", "PATH_REASONING"),
        (("use_query_conditioning", False),),
        True,
        "The edge-attention query term is absent; query readout remains.",
    ),
    ModelVariantSpec(
        ModelVariantId.NO_PATH_REASONING,
        IMPLEMENTED_STATUS,
        "REQUIRED_ABLATION",
        ("PATH_REASONING",),
        _COMMON_PRESERVED + ("EXACT_PHASE5_CORE_FALLBACK", "EVIDENCE_LOGIC_QUERY_ATTENTION"),
        (),
        False,
        "Uses the existing exact path-disabled Phase 5 fallback.",
    ),
    ModelVariantSpec(
        ModelVariantId.NO_PROVENANCE,
        IMPLEMENTED_STATUS,
        "OPTIONAL",
        ("PROVENANCE_ENCODING",),
        _COMMON_PRESERVED + ("PROPERTY_CONTEXT", "EVIDENCE_CONTEXT", "POLARITY_BRANCHES", "LOGIC", "PATH_REASONING"),
        (("use_provenance", False),),
        True,
        "Provenance is independently separable from property/evidence context.",
    ),
    ModelVariantSpec(
        ModelVariantId.NO_LOGIC_MASS_CONDITIONING,
        IMPLEMENTED_STATUS,
        "POST_FREEZE_CONFIRMATORY_ABLATION",
        ("DIRECT_DYNAMIC_FOUR_VALUED_MASS_MESSAGE_MIXTURE",),
        _COMMON_PRESERVED + (
            "EVIDENCE_PROPERTY_PROVENANCE_CONTEXT", "SUPPORT_COUNTER_BRANCHES",
            "ORIGINAL_LOGIC_MASSES_IN_EDGE_CONTEXT", "FOUR_TRAINABLE_TRANSFORMS",
            "EXACT_NA_FALLBACK", "QUERY_CONDITIONING", "PATH_REASONING",
        ),
        (),
        True,
        "Phase 8.2.6A amendment: fixed 0.25 four-transform mixture only; "
        "original masses remain in edge context and all other FULL paths.",
    ),
)

_BY_ID = {item.variant_id: item for item in _VARIANTS}


def variant_catalog() -> tuple[ModelVariantSpec, ...]:
    return _VARIANTS


def get_variant_spec(variant_id: ModelVariantId | str) -> ModelVariantSpec:
    try:
        return _BY_ID[ModelVariantId(variant_id)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unknown model variant: {variant_id}") from exc


def apply_variant_config(
    base_config: EviHGTConfig, variant_id: ModelVariantId | str
) -> EviHGTConfig:
    spec = get_variant_spec(variant_id)
    if spec.status != IMPLEMENTED_STATUS:
        raise ValueError(f"model variant is unavailable: {spec.status}")
    return replace(base_config, **spec.config_overrides)


__all__ = [
    "IMPLEMENTED_STATUS",
    "PAIR_MLP_UNAVAILABLE_STATUS",
    "ModelVariantId",
    "ModelVariantSpec",
    "apply_variant_config",
    "get_variant_spec",
    "variant_catalog",
]
