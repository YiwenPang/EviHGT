"""Instance-local interventions; legacy modules/default calls are unchanged.

M2 cuts the target concat block and the common final provenance output. M3
replaces the FourValuedLogic module output before *all* of its consumers.
No hooks are global, and no trained weights are read by this factory.
"""
from dataclasses import dataclass, asdict
import importlib.util
from pathlib import Path
import sys

import torch

from evi_hgt.models.four_valued_logic import LogicState
from evi_hgt.query_model.model import build_cell_model
from .identity import ROOT, CellContract, MODES, digest


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
        raise TypeError("Expected the original fact LogicState")
    return LogicState(torch.where(output.available.unsqueeze(-1),
                      torch.full_like(output.masses, .25), torch.zeros_like(output.masses)), output.available)


@dataclass
class Phase9Model:
    contract: CellContract
    base: object
    hooks: tuple

    @property
    def model(self):
        return self.base.model

    def metadata(self):
        encoder = self.model.encoder
        flags = [x.attention.use_logic_mass_conditioning for x in encoder.layers]
        if flags != [False, False]:
            raise ValueError("ranking_protocol B direct message flags changed")
        cfg = encoder.config
        if not all(getattr(cfg,k) for k in ("use_property", "use_evidence", "use_provenance",
                   "use_evidence_context", "use_polarity_branches", "use_four_valued_logic", "use_query_conditioning")):
            raise ValueError("A broad ablation switch changed scientific semantics")
        expected = 2 if self.contract.method == "M2" else 1 if self.contract.method == "M3" else 0
        if len(self.hooks) != expected:
            raise ValueError("Ablation hook identity missing")
        for module, kind, handle, callback in self.hooks:
            if getattr(module, kind).get(handle.id) is not callback:
                raise ValueError("Instance ablation was removed or replaced")
        return {"contract": self.contract.payload(), "intervention": MODES[self.contract.method],
                "direct_mass_flags": flags, "model_config": asdict(cfg),
                "nominal_parameters": sum(p.numel() for p in self.model.parameters()),
                "schema": {k:[list(v.shape),str(v.dtype)] for k,v in self.model.state_dict().items()}}

    def forward(self, batch, **kwargs):
        self.metadata()
        return self.base.forward(batch, **kwargs)


def build_model(contract):
    if not isinstance(contract, CellContract):
        raise TypeError("An exact frozen CellContract is required")
    contract.__post_init__()
    base = build_cell_model(contract.seed)
    enc = base.model.encoder
    for layer in enc.layers:
        layer.attention.use_logic_mass_conditioning = False
    hooks = []
    if contract.method == "M2":
        cfg = enc.config
        start = 2*cfg.num_numeric_features + 2*cfg.num_boolean_features + cfg.evidence_dim
        callback = _ZeroContentInput(start, start+cfg.evidence_dim+cfg.provenance_dim)
        module = enc.evidence_encoder.encoder
        handle = module.register_forward_pre_hook(callback)
        hooks.append((module,"_forward_pre_hooks",handle,callback))
        module = enc.evidence_encoder.provenance_encoder
        handle = module.register_forward_hook(_zero_provenance)
        hooks.append((module,"_forward_hooks",handle,_zero_provenance))
    elif contract.method == "M3":
        handle = enc.logic.register_forward_hook(_constant_quality)
        hooks.append((enc.logic,"_forward_hooks",handle,_constant_quality))
    result = Phase9Model(contract,base,tuple(hooks))
    result.metadata()
    return result


def make_optimizer(built):
    built.metadata()
    return torch.optim.AdamW(built.model.parameters(), lr=5e-5,weight_decay=1e-5,
                            betas=(.9,.999),eps=1e-8,foreach=None,fused=None)


def stable_objective():
    """Reuse the exact bound backend; no model sigmoid or global mode changes."""
    name = "_phase9_bound_stable_training_loss"
    if name not in sys.modules:
        path = ROOT / "outputs/phase85_part3b11/stable_training_loss.py"
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    if module.VERSION != "B11_PU_RISK_WITH_B10_STABLE_BACKWARD_V1":
        raise ValueError("Stable backend identity differs")
    return module.make_objective(True)
