"""Cell-local construction of the unchanged D1 architecture and hooks."""
from ablation_protocol_ids import FUSION_LAYER_NAME
import torch
from evi_hgt.ranking_protocol.identity import CellContract
from evi_hgt.ranking_protocol.model import build_model
from evi_hgt.ranking_protocol.state import tensor_digest
from evi_hgt_feature_baseline.model import DevelopmentModel, EarlyFusion, make_optimizer
from .protocol import cell_spec


class CellD1(DevelopmentModel):
    def __init__(self,cell,pretrained):
        s=cell_spec(cell);self.cell_id=cell;self.arm='D1_PRETRAINED_EARLY';self.pretrained=pretrained
        self.base=build_model(CellContract.load('M0',s['repeat'],s['fold']))
        self.model=self.base.model;self.common_keys=tuple(self.model.state_dict())
        self.common_digest=tensor_digest(self.model.state_dict());self.hidden=self.model.encoder.config.hidden_dim
        self.drug_type=self.model.encoder.node_encoder.drug_type;self.disease_type=self.model.encoder.node_encoder.disease_type
        self.handles=[];self.last={};self._active=False;self._replaying=False;self.fusion_enabled=True
        with torch.random.fork_rng(devices=[]):self.model.add_module(FUSION_LAYER_NAME,EarlyFusion(self.hidden))
        self.handles.append(self.model.encoder.node_encoder.register_forward_hook(self._early))
        if tensor_digest({k:self.model.state_dict()[k] for k in self.common_keys})!=self.common_digest:
            raise ValueError('D1 projection changed base initialization')

    def metadata(self):
        m=super().metadata();m['seed']=cell_spec(self.cell_id)['seed'];m['cell']=self.cell_id
        if m['original']['contract']['cell']['training_seed']!=m['seed']:raise ValueError('Cell model seed mismatch')
        return m


def build_arm(cell,pretrained):return CellD1(cell,pretrained)
