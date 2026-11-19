"""All-15-cell wrapper around the unchanged historical FULL B + SideInfo + D1.

Import is inert. Construction loads original frozen metadata and fresh parameter
initialization only. No training, loss, checkpoint, or scoring route is added.
The DEV94 early-fusion hook and forward implementation are inherited verbatim.
"""
import re
import torch

from evi_hgt.phase9.identity import CellContract
from evi_hgt.phase9.model import build_model
from evi_hgt.phase9.state import tensor_digest
from evi_hgt_dev94.model import DevelopmentModel, EarlyFusion


def cell_coordinates(cell):
    if not isinstance(cell,str) or re.fullmatch(r'R[0-2]F[0-4]',cell) is None:
        raise ValueError('Exactly one of the original fifteen cells is required')
    return int(cell[1]), int(cell[3])


class FullD1(DevelopmentModel):
    def __init__(self, cell, pretrained):
        repeat, fold = cell_coordinates(cell)
        if pretrained is None:
            raise ValueError('FULL requires its certified frozen D1 store')
        contract = CellContract.load('M0', repeat, fold)
        self.cell_id = cell; self.method = 'FULL'; self.training_seed = contract.seed
        self.arm = 'D1_PRETRAINED_EARLY'; self.pretrained = pretrained
        self.base = build_model(contract)
        self.model = self.base.model
        self.common_keys = tuple(self.model.state_dict())
        self.common_digest = tensor_digest(self.model.state_dict())
        self.hidden = self.model.encoder.config.hidden_dim
        encoder = self.model.encoder.node_encoder
        self.drug_type, self.disease_type = encoder.drug_type, encoder.disease_type
        self.handles = []; self.last = {}; self._active = False; self._replaying = False
        self.fusion_enabled = True
        with torch.random.fork_rng(devices=[]):
            self.model.add_module('dev94_fusion', EarlyFusion(self.hidden))
        self._early_callback = self._early
        self.handles.append(encoder.register_forward_hook(self._early_callback))
        if tensor_digest({k:self.model.state_dict()[k] for k in self.common_keys}) != self.common_digest:
            raise ValueError('D1 registration changed historical FULL initialization')
        self.metadata()

    def metadata(self):
        result = super().metadata()
        result.update(seed=self.training_seed, cell=self.cell_id, method='FULL',
            factory='evi_hgt_phase98.full.FullD1', loss_external=True,
            architecture='B_UNIFORM_CANONICAL_SIDEINFO_D1_ORIGINAL_READOUT',
            scope='PHASE98_FACTORY_ONLY_NO_EXECUTION_AUTHORIZATION')
        original = result['original']
        repeat, fold = cell_coordinates(self.cell_id)
        cell = original['contract']['cell']
        if (original['contract']['method'] != 'M0' or original['intervention'] != 'FULL_B'
            or original['direct_mass_flags'] != [False,False]
            or cell['training_seed'] != self.training_seed
            or (cell['repeat_index'],cell['fold_index']) != (repeat,fold)
            or self.hidden != 128 or (self.drug_type,self.disease_type) != (4,3)
            or sum(p.numel() for p in self.model.parameters()) != 3846610):
            raise ValueError('Original FULL cell/architecture identity differs')
        if (len(self.handles) != 1 or self.model.encoder.node_encoder._forward_hooks.get(
            self.handles[0].id) is not self._early_callback):
            raise ValueError('Original instance-local D1 early hook missing or replaced')
        return result


def build_full(cell, pretrained):
    return FullD1(cell, pretrained)
