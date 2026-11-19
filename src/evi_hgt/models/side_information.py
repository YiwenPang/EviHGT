"""Typed binary side-information encoders and gated composition."""
import torch
from torch import nn

DRUG_DIM = 2048 + 512 + 2
DISEASE_DIM = 512 + 512 + 2
PACKED_DIM = DRUG_DIM + DISEASE_DIM

class SideNodeEncoder(nn.Module):
    def __init__(self, base, *, hidden, drug_type, disease_type):
        super().__init__()
        if drug_type == disease_type:
            raise ValueError('side node types must differ')
        self.base = base
        self.drug_type, self.disease_type = drug_type, disease_type
        self.drug_encoder = nn.Sequential(nn.Linear(DRUG_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.disease_encoder = nn.Sequential(nn.Linear(DISEASE_DIM, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.drug_gate = nn.Linear(2 * hidden, hidden)
        self.disease_gate = nn.Linear(2 * hidden, hidden)
        self.last_gates = ()
        self.enabled = True

    def forward(self, nodes):
        raw = nodes.features
        if raw is None or raw.shape != (nodes.num_nodes, PACKED_DIM):
            raise ValueError('SIDE requires typed feature values and availability masks')
        if not torch.isfinite(raw).all() or not ((raw == 0) | (raw == 1)).all():
            raise ValueError('SIDE raw features and masks must be finite binary values')
        drug = nodes.node_type_id == self.drug_type
        disease = nodes.node_type_id == self.disease_type
        if raw[~drug, :DRUG_DIM].any() or raw[~disease, DRUG_DIM:].any():
            raise ValueError('side feature node type mismatch')
        graph = self.base(nodes)
        if not self.enabled:
            self.last_gates = ()
            return graph
        fused = graph.clone()
        gates = []
        for values, typed, encoder, gate_layer, widths in (
            (raw[:, :DRUG_DIM], drug, self.drug_encoder, self.drug_gate, (2048, 512)),
            (raw[:, DRUG_DIM:], disease, self.disease_encoder, self.disease_gate, (512, 512)),
        ):
            a, b = widths
            if values[values[:, -2] == 0, :a].any() or values[values[:, -1] == 0, a:a+b].any():
                raise ValueError('unavailable side family carries values')
            active = typed & values[:, -2:].bool().any(dim=1)
            if active.any():
                side = encoder(values[active].to(graph.dtype))
                gate = torch.sigmoid(gate_layer(torch.cat([graph[active], side], dim=-1)))
                fused[active] = gate * graph[active] + (1 - gate) * side
                gates.append(gate.detach())
        self.last_gates = tuple(gates)
        return fused

