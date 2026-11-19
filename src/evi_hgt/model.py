"""Build EviHGT and its ablations.

FULL mixes the four direct T/F/B/U transforms uniformly. Learned fact masses
enter the edge context, and unavailable facts have a separate message branch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Protocol

import torch
from torch import Tensor, nn

from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.evi_hgt import EviHGT
from evi_hgt.models.four_valued_logic import LogicState
from evi_hgt.models.path_reasoning import PathReasoningConfig
from evi_hgt.models.side_information import PACKED_DIM, SideNodeEncoder
from evi_hgt.training.prediction import EviHGTForDDA


VARIANTS = ("FULL", "NO_D1", "NO_SIDE", "NO_PATH", "CONTENT_ZERO", "FACT_QUALITY_CONSTANT")
VARIANT_DESCRIPTIONS = {
    "FULL": "Full model with uniform direct logic-message mixing, side information, and D1 early fusion.",
    "NO_D1": "Bypass pretrained drug/disease residual contributions.",
    "NO_SIDE": "Bypass gated raw side-information fusion; retain type embeddings and D1.",
    "NO_PATH": "Bypass path encoding and residual path fusion; retain graph readout.",
    "CONTENT_ZERO": "Zero evidence-text and provenance content blocks plus the shared provenance output; preserve properties and masks.",
    "FACT_QUALITY_CONSTANT": "Replace available fact logic masses by (0.25, 0.25, 0.25, 0.25); preserve availability masks.",
}


class PretrainedStore(Protocol):
    def gather(self, node_id: Tensor, node_type_id: Tensor, drug_type: int,
               disease_type: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return frozen [N, 768] vectors and Boolean [N] availability."""
        ...


class EarlyFusion(nn.Module):
    """Zero-initialized, bias-free projections of frozen D1 vectors."""
    def __init__(self, hidden: int):
        super().__init__()
        self.projections = nn.ModuleDict({key: nn.Linear(768, hidden, bias=False) for key in ("DRG", "DIS")})
        for layer in self.projections.values():
            nn.init.zeros_(layer.weight)


def _zero_provenance(module, args, output):
    return torch.zeros_like(output)


@dataclass(frozen=True)
class _ZeroContentInput:
    first: int
    last: int

    def __call__(self, module, args):
        joined, = args
        return (torch.cat((joined[:, :self.first], torch.zeros_like(joined[:, self.first:self.last]),
                           joined[:, self.last:]), dim=-1),)


def _constant_quality(module, args, output):
    if not isinstance(output, LogicState):
        raise TypeError("Expected a fact LogicState")
    return LogicState(torch.where(output.available.unsqueeze(-1),
                      torch.full_like(output.masses, 0.25), torch.zeros_like(output.masses)), output.available)


def _side_seed(seed: int) -> int:
    # This stable namespace preserves the original independent side-feature RNG.
    payload = json.dumps(["PHASE852B_SIDE_INITIALIZATION_V1", seed], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


class BuiltEviHGT:
    """Model, public forward path, and explicit ablation identity.

    Ablations share the parameter schema and initializer. Bypassed modules
    receive no gradient. Save the configuration, variant and seed alongside
    ``model.state_dict()`` and rebuild the wrapper to restore its hooks.
    """
    def __init__(self, model, *, pretrained, seed, variant, drug_type, disease_type, handles):
        self.model = model
        self.pretrained = pretrained
        self.seed = seed
        self.variant = variant
        self.drug_type = drug_type
        self.disease_type = disease_type
        self.handles = list(handles)
        self.last = {}
        if variant != "NO_D1":
            self.handles.append(model.encoder.node_encoder.register_forward_hook(self._early))

    def _early(self, module, args, output):
        nodes, = args
        if nodes.node_id is None:
            raise ValueError("D1 early fusion requires global node IDs")
        values, available = self.pretrained.gather(
            nodes.node_id, nodes.node_type_id, self.drug_type, self.disease_type, output.device,
        )
        if values.shape != (nodes.num_nodes, 768) or available.shape != (nodes.num_nodes,) or available.dtype != torch.bool:
            raise ValueError("Pretrained vector shape or availability mask is invalid")
        if values.requires_grad or not torch.isfinite(values).all():
            raise ValueError("Pretrained vectors must be finite and frozen")
        if values.device != output.device or available.device != output.device:
            raise ValueError("Pretrained vectors and availability must be on the model device")
        result = output.clone()
        counts = {}
        for name, type_id in (("DRG", self.drug_type), ("DIS", self.disease_type)):
            typed = nodes.node_type_id == type_id
            active = typed & available
            if active.any():
                result[active] = output[active] + self.model.pretrained_fusion.projections[name](values[active]).to(output.dtype)
            counts[name] = {"available": int(active.sum()), "total": int(typed.sum())}
        self.last["early"] = counts
        return result

    def forward(self, batch, *, path_batch):
        if path_batch is None:
            raise ValueError("An explicit path batch is required; use an empty batch when no legal path exists")
        self.last = {}
        return self.model(batch, path_batch=path_batch, use_path_reasoning=self.variant != "NO_PATH")

    def metadata(self):
        return {
            "variant": self.variant,
            "description": VARIANT_DESCRIPTIONS[self.variant],
            "seed": self.seed,
            "config": asdict(self.model.encoder.config),
            "path_config": asdict(self.model.path_reasoner.config),
            "drug_type": self.drug_type,
            "disease_type": self.disease_type,
            "nominal_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "direct_logic_message_mixing": "uniform",
            "dynamic_fact_logic_in_edge_context": self.variant != "FACT_QUALITY_CONSTANT",
            "d1_pretrained_vectors_frozen": True,
            "d1_projection_trainable": self.variant != "NO_D1",
            "nominal_capacity_note": "Bypassed branches are retained for matching initialization; effective capacity differs.",
        }


def build_model(config: EviHGTConfig, *, pretrained: PretrainedStore | None,
                seed: int, variant: str = "FULL", drug_type: int = 4,
                disease_type: int = 3, path_config: PathReasoningConfig | None = None) -> BuiltEviHGT:
    """Build fresh CPU parameters without reading files or changing global RNG."""
    if not isinstance(config, EviHGTConfig):
        raise TypeError("config must be EviHGTConfig")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    if variant != "NO_D1" and pretrained is None:
        raise ValueError("This variant requires a frozen pretrained feature store")
    if not 0 <= drug_type < config.num_node_types or not 0 <= disease_type < config.num_node_types or drug_type == disease_type:
        raise ValueError("Distinct valid drug and disease node type IDs are required")
    if config.use_node_features or config.use_node_id_embedding or not config.use_node_type_embedding:
        raise ValueError("The portable model requires type embeddings plus the declared side-information encoder")
    if not all(getattr(config, key) for key in (
        "use_property", "use_evidence", "use_provenance", "use_evidence_context",
        "use_polarity_branches", "use_four_valued_logic", "use_query_conditioning",
    )):
        raise ValueError("Use an explicit named ablation; broad core switches must remain enabled")
    path_config = path_config or PathReasoningConfig(core_hidden_dim=config.hidden_dim)
    if path_config.core_hidden_dim != config.hidden_dim:
        raise ValueError("Path and graph hidden dimensions must match")
    core = replace(config, node_feature_dim=0, return_edge_context=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = EviHGTForDDA(EviHGT(core), path_config=path_config)
        side = SideNodeEncoder(model.encoder.node_encoder, hidden=core.hidden_dim,
                               drug_type=drug_type, disease_type=disease_type)
        model.encoder.node_encoder = side
        model.encoder.config = replace(core, node_feature_dim=PACKED_DIM)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(_side_seed(seed))
            for branch in (side.drug_encoder, side.disease_encoder, side.drug_gate, side.disease_gate):
                for module in branch.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
        model.add_module("pretrained_fusion", EarlyFusion(core.hidden_dim))
    for layer in model.encoder.layers:
        layer.attention.use_logic_mass_conditioning = False
    side.enabled = variant != "NO_SIDE"
    handles = []
    if variant == "CONTENT_ZERO":
        start = 2 * core.num_numeric_features + 2 * core.num_boolean_features + core.evidence_dim
        handles.append(model.encoder.evidence_encoder.encoder.register_forward_pre_hook(
            _ZeroContentInput(start, start + core.evidence_dim + core.provenance_dim)))
        handles.append(model.encoder.evidence_encoder.provenance_encoder.register_forward_hook(_zero_provenance))
    elif variant == "FACT_QUALITY_CONSTANT":
        handles.append(model.encoder.logic.register_forward_hook(_constant_quality))
    return BuiltEviHGT(model, pretrained=pretrained, seed=seed, variant=variant,
                       drug_type=drug_type, disease_type=disease_type, handles=handles)


__all__ = ["BuiltEviHGT", "EarlyFusion", "PretrainedStore", "VARIANTS", "VARIANT_DESCRIPTIONS", "build_model"]
