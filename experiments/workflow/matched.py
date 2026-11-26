"""Input-matched neural controls for a prepared EviHGT development fold."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import torch
from torch import nn
from torch_geometric.nn import HGTConv

from evi_hgt.model import EarlyFusion
from evi_hgt.models.input_contract import NodeBatch
from evi_hgt.data.pretrained import NODE_TYPES
from evi_hgt.data.raw_features import PACKED_DIM
from experiments.comparisons.src.evi_hgt_ranking_comparison.protocol_ids import (
    HGT_LAYER_SEED,
    PAIR_READOUT_SEED,
)


def validate_nodes(nodes):
    size = nodes.num_nodes
    if nodes.node_id is None or nodes.node_id.dtype != torch.long or nodes.node_type_id.dtype != torch.long:
        raise ValueError("Canonical int64 entity/type IDs required")
    if nodes.features is None or nodes.features.shape != (size, PACKED_DIM):
        raise ValueError("SideInfo shape mismatch")
    if size and (int(nodes.node_type_id.min()) < 0 or int(nodes.node_type_id.max()) >= len(NODE_TYPES)):
        raise ValueError("Unknown node type")
    if not torch.isfinite(nodes.features).all():
        raise ValueError("Nonfinite SideInfo")


def endpoint_nodes(batch):
    queries, nodes = batch.queries, batch.nodes
    if queries.num_queries != 1:
        raise ValueError("Expected one query")
    indices = torch.cat((queries.drug_node_index, queries.disease_node_index))
    chosen = NodeBatch(node_type_id=nodes.node_type_id[indices], graph_id=nodes.graph_id[indices],
                       features=nodes.features[indices], node_id=nodes.node_id[indices])
    validate_nodes(chosen)
    if chosen.node_type_id.tolist() != [4, 3] or chosen.graph_id.tolist() != [0, 0]:
        raise ValueError("Expected one canonical drug/disease query")
    return chosen


def validate_graph(batch, edge_catalog):
    validate_nodes(batch.nodes)
    endpoint_nodes(batch)
    queries, nodes, edges = batch.queries, batch.nodes, batch.edges
    if nodes.num_nodes > 1000 or edges.num_edges > 5000:
        raise ValueError("Frozen query graph budget exceeded")
    if edges.num_edges:
        if int(edges.edge_type_id.min()) < 0 or int(edges.edge_type_id.max()) >= len(edge_catalog):
            raise ValueError("Unknown relation index")
        drug, disease = queries.drug_node_index[0], queries.disease_node_index[0]
        direct = (edges.src == drug) & (edges.dst == disease)
        inverse = (edges.src == disease) & (edges.dst == drug)
        if (direct | inverse).any():
            raise ValueError("Query target or reverse edge reached matched HGT")


MLP = "PAIR_MLP_D1_MATCHED"
HGT = "HGT_D1_MATCHED"


def state_digest(state):
    checksum = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        checksum.update(name.encode())
        checksum.update(str(tensor.dtype).encode())
        checksum.update(str(tuple(tensor.shape)).encode())
        checksum.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def _seed(seed, namespace):
    raw = json.dumps([namespace, seed], ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31)


@dataclass(frozen=True)
class MatchedPrediction:
    support_logit: torch.Tensor
    unsuitability_logit: torch.Tensor


class PairReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(512, 128, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(.1)
        self.support_head = nn.Linear(128, 1, bias=True)
        self.unsuitability_head = nn.Linear(128, 1, bias=True)

    def forward(self, drug, disease):
        joined = torch.cat((drug, disease, drug * disease, torch.abs(drug - disease)), dim=-1)
        hidden = self.dropout(self.activation(self.hidden(joined)))
        return MatchedPrediction(self.support_head(hidden).squeeze(-1),
                                 self.unsuitability_head(hidden).squeeze(-1))


class MatchedNodes(nn.Module):
    def __init__(self, node_encoder, pretrained):
        super().__init__()
        self.node_encoder = node_encoder
        self.fusion = EarlyFusion(128)
        self.pretrained = pretrained

    def encode(self, nodes):
        base = self.node_encoder(nodes)
        vectors, available = self.pretrained.gather(nodes.node_id, nodes.node_type_id, 4, 3, base.device)
        if vectors.requires_grad or vectors.shape != (nodes.num_nodes, 768):
            raise ValueError("Frozen D1 vectors are invalid")
        if available.shape != (nodes.num_nodes,) or available.dtype != torch.bool:
            raise ValueError("Frozen D1 availability is invalid")
        if not torch.isfinite(vectors).all():
            raise ValueError("Nonfinite D1 vector")
        result = base.clone()
        for kind, type_id in (("DRG", 4), ("DIS", 3)):
            active = (nodes.node_type_id == type_id) & available
            if active.any():
                residual = self.fusion.projections[kind](vectors[active]).to(base.dtype)
                result[active] = base[active] + residual
        return result


class PairMLP(MatchedNodes):
    def __init__(self, node_encoder, pretrained, pair_readout):
        super().__init__(node_encoder, pretrained)
        self.pair_readout = pair_readout

    def forward(self, batch):
        encoded = self.encode(endpoint_nodes(batch))
        return self.pair_readout(encoded[:1], encoded[1:])


def relation_metadata(edge_catalog):
    keys = []
    for i, (source, _, target) in enumerate(edge_catalog):
        if source not in NODE_TYPES or target not in NODE_TYPES:
            raise ValueError("Unknown relation endpoint type")
        for key in ((source, f"r{i}", target), (target, f"r{i}", source)):
            if key not in keys:
                keys.append(key)
    if not keys:
        raise ValueError("Global legal relation catalog must be nonempty")
    return list(NODE_TYPES), keys


class MatchedHGT(MatchedNodes):
    def __init__(self, node_encoder, pretrained, pair_readout, edge_catalog):
        super().__init__(node_encoder, pretrained)
        self.pair_readout = pair_readout
        self.edge_catalog = tuple(tuple(row) for row in edge_catalog)
        self.metadata = relation_metadata(self.edge_catalog)
        self.layers = nn.ModuleList(HGTConv(128, 128, self.metadata, heads=4) for _ in range(2))
        self.dropout = nn.Dropout(.1)

    def _relations(self, batch, positions):
        edges, types = batch.edges, batch.nodes.node_type_id
        local = torch.empty(types.numel(), device=types.device, dtype=torch.long)
        for ids in positions.values():
            local[ids] = torch.arange(len(ids), device=types.device)
        result = {}
        allowed = set(self.metadata[1])
        for relation in edges.edge_type_id.unique().tolist():
            candidates = (edges.edge_type_id == relation).nonzero().flatten()
            endpoint_types = torch.stack((types[edges.src[candidates]], types[edges.dst[candidates]]), dim=1)
            for source, target in endpoint_types.unique(dim=0).tolist():
                selected = candidates[(endpoint_types[:, 0] == source) & (endpoint_types[:, 1] == target)]
                key = NODE_TYPES[source], f"r{relation}", NODE_TYPES[target]
                if key not in allowed:
                    raise ValueError("Actual typed relation differs from bound canonical catalog")
                result[key] = torch.stack((local[edges.src[selected]], local[edges.dst[selected]]))
        return result

    def forward(self, batch):
        validate_graph(batch, self.edge_catalog)
        encoded = self.encode(batch.nodes)
        if batch.edges.num_edges:
            positions = {name: (batch.nodes.node_type_id == i).nonzero().flatten()
                         for i, name in enumerate(NODE_TYPES)}
            positions = {name: ids for name, ids in positions.items() if ids.numel()}
            edges = self._relations(batch, positions)
            receiving = torch.zeros(encoded.shape[0], device=encoded.device, dtype=torch.bool)
            receiving[batch.edges.dst] = True
            for layer in self.layers:
                output = layer({name: encoded[ids] for name, ids in positions.items()}, edges)
                updated = encoded.clone()
                for name, value in output.items():
                    if value is not None:
                        ids = positions[name]
                        active = receiving[ids]
                        # Preserve the previous state for rows with no incoming message.
                        if active.any():
                            updated[ids[active]] = self.dropout(value[active]).to(encoded.dtype)
                encoded = updated
        query = batch.queries
        return self.pair_readout(encoded[query.drug_node_index], encoded[query.disease_node_index])


def common_state(model):
    return {name: value for name, value in model.state_dict().items() if not name.startswith("layers.")}


@dataclass
class BuiltMatched:
    method: str
    model: nn.Module
    training_seed: int
    initialization: dict

    def forward(self, batch, *, path_batch=None):
        return self.model(batch)

    def metadata(self):
        return dict(method=self.method, training_seed=self.training_seed,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            parameter_shapes={name: list(p.shape) for name, p in self.model.named_parameters()},
            initialization=self.initialization, query_conditioning=False, EP_quality_context=False,
            path_input=False, hidden=128, layers=2 if self.method == HGT else 0,
            heads=4 if self.method == HGT else 0, dropout=.1, D1_cache_frozen=True,
            D1_projection_trainable=True, selection="FIXED_3G")


def build_matched_pair(training_seed, pretrained, edge_catalog, *, common_node_encoder):
    if type(training_seed) is not int or not 0 <= training_seed < 2**31:
        raise ValueError("Certified cell seed required")
    if common_node_encoder._forward_hooks:
        raise ValueError("Shared node initializer cannot carry the FULL D1 forward hook")
    source_digest = state_digest(common_node_encoder.state_dict())
    readout_seed = _seed(training_seed, PAIR_READOUT_SEED)
    hgt_seed = _seed(training_seed, HGT_LAYER_SEED)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(readout_seed)
        readout = PairReadout()
        torch.random.default_generator.manual_seed(hgt_seed)
        mlp = PairMLP(deepcopy(common_node_encoder), pretrained, deepcopy(readout))
        hgt = MatchedHGT(deepcopy(common_node_encoder), pretrained, deepcopy(readout), edge_catalog)

    if any(state_digest(model.node_encoder.state_dict()) != source_digest for model in (mlp, hgt)):
        raise ValueError("FULL common node initialization tensor identity differs")
    mlp_shared, hgt_shared = common_state(mlp), common_state(hgt)
    if mlp_shared.keys() != hgt_shared.keys():
        raise ValueError("Matched shared tensor names differ")
    if any(not torch.equal(value, hgt_shared[name]) for name, value in mlp_shared.items()):
        raise ValueError("Matched shared tensors differ")
    initialization = dict(full_node_initial_sha256=source_digest, matched_common_sha256=state_digest(mlp_shared),
        common_tensor_count=len(mlp_shared), copied_from_fresh_full_factory=True, shared_tensor_equal=True,
        D1_projection_initialization="zero and trainable", readout_seed=readout_seed, HGT_layer_seed=hgt_seed)
    return {method: BuiltMatched(method, model, training_seed, initialization)
            for method, model in ((MLP, mlp), (HGT, hgt))}
