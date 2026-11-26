"""EviHGT's HGTDR architecture adapter, following Bioinformatics btae349.

The relation layer is the separately licensed PyG 2.3.1 compatibility module.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import Linear

from experiments.native_adapters.HGTDR.src.vendor.hgt_conv_231 import HGTConv


class HGT(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.lin_dict = nn.ModuleDict({name: Linear(-1, 64) for name in metadata[0]})
        self.convs = nn.ModuleList(HGTConv(64, 64, metadata, heads=8, group="mean") for _ in range(3))
        self.lin = Linear(3 * 64, 64)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, edges):
        hidden = {kind: self.dropout(self.lin_dict[kind](value).relu_()) for kind, value in features.items()}
        levels = []
        for layer in self.convs:
            hidden = layer(hidden, edges)
            if any(value is None for value in hidden.values()):
                raise ValueError("Every node type must receive a relation in the HGTDR batch")
            levels.append(hidden)
        return tuple(self.lin(torch.cat([level[kind] for level in levels], dim=-1)).relu()
                     for kind in ("drug", "disease"))


class MLPPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(128, 64)
        self.L2 = nn.Linear(64, 1)
        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)

    def forward(self, drug, disease):
        return self.L2(self.dropout(self.bn(self.L1(torch.cat((drug, disease), dim=-1))).relu()))


def compute_loss(scores, labels):
    observed = labels == 1
    positive_fraction = observed.sum() / labels.numel()
    weights = torch.where(observed, (~observed).sum() / labels.numel(), positive_fraction)
    return F.binary_cross_entropy_with_logits(scores, labels, pos_weight=weights)
