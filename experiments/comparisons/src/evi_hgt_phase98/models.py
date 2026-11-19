"""Frozen Phase98 matched controls; historical EviHGT modules are unchanged.

The HGT control uses the installed PyG HGTConv, not an Evi switch variant.
No forward or optimizer step is performed during construction or import.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch_geometric.nn import HGTConv
from evi_hgt_dev94.model import EarlyFusion
from .input_contract import NODE_TYPES,endpoint_nodes,validate_graph

MLP='PAIR_MLP_D1_MATCHED';HGT='HGT_D1_MATCHED'

def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def _seed(seed,namespace):
    raw=json.dumps([namespace,seed],ensure_ascii=True,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')%(2**31)

@dataclass(frozen=True)
class MatchedPrediction:
    support_logit:torch.Tensor
    unsuitability_logit:torch.Tensor

class PairReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(512,128,bias=True)
        self.activation=nn.GELU();self.dropout=nn.Dropout(.1)
        self.support_head=nn.Linear(128,1,bias=True)
        self.unsuitability_head=nn.Linear(128,1,bias=True)

    def forward(self,drug,disease):
        joined=torch.cat((drug,disease,drug*disease,torch.abs(drug-disease)),dim=-1)
        h=self.dropout(self.activation(self.hidden(joined)))
        return MatchedPrediction(self.support_head(h).squeeze(-1),self.unsuitability_head(h).squeeze(-1))

class MatchedNodes(nn.Module):
    def __init__(self,node_encoder,pretrained):
        super().__init__();self.node_encoder=node_encoder
        self.fusion=EarlyFusion(128);self.pretrained=pretrained

    def encode(self,nodes):
        base=self.node_encoder(nodes)
        x,available=self.pretrained.gather(nodes.node_id,nodes.node_type_id,4,3,base.device)
        if x.requires_grad or x.shape!=(nodes.num_nodes,768) or available.shape!=(nodes.num_nodes,) or available.dtype!=torch.bool:
            raise ValueError('Frozen D1 vectors/availability invalid')
        if not torch.isfinite(x).all():raise ValueError('Nonfinite D1 vector')
        result=base.clone()
        for kind,type_id in (('DRG',4),('DIS',3)):
            active=(nodes.node_type_id==type_id)&available
            if active.any():result[active]=base[active]+self.fusion.projections[kind](x[active]).to(base.dtype)
        return result

class PairMLP(MatchedNodes):
    def __init__(self,node_encoder,pretrained,pair_readout):
        super().__init__(node_encoder,pretrained);self.pair_readout=pair_readout

    def forward(self,batch,**_unused):
        nodes=endpoint_nodes(batch)
        h=self.encode(nodes)
        return self.pair_readout(h[:1],h[1:])

def relation_metadata(edge_catalog):
    keys=[]
    for i,(source,_relation,target) in enumerate(edge_catalog):
        if source not in NODE_TYPES or target not in NODE_TYPES:raise ValueError('Unknown relation endpoint type')
        for key in ((source,f'r{i}',target),(target,f'r{i}',source)):
            if key not in keys:keys.append(key)
    if not keys:raise ValueError('Global legal relation catalog must be nonempty')
    return (list(NODE_TYPES),keys)

class MatchedHGT(MatchedNodes):
    def __init__(self,node_encoder,pretrained,pair_readout,edge_catalog):
        super().__init__(node_encoder,pretrained);self.pair_readout=pair_readout
        self.edge_catalog=tuple(tuple(x) for x in edge_catalog)
        self.metadata=relation_metadata(self.edge_catalog)
        self.layers=nn.ModuleList([HGTConv(128,128,self.metadata,heads=4) for _ in range(2)])
        self.dropout=nn.Dropout(.1)

    def _relations(self,batch,positions):
        e=batch.edges;kind=batch.nodes.node_type_id
        local=torch.empty(kind.numel(),device=kind.device,dtype=torch.long)
        for ids in positions.values():local[ids]=torch.arange(len(ids),device=ids.device)
        edges={};allowed=set(self.metadata[1])
        for rel in e.edge_type_id.unique().tolist():
            candidates=(e.edge_type_id==rel).nonzero().flatten()
            pairs=torch.stack((kind[e.src[candidates]],kind[e.dst[candidates]]),dim=1)
            for st,dt in pairs.unique(dim=0).tolist():
                selected=candidates[(pairs[:,0]==st)&(pairs[:,1]==dt)]
                key=(NODE_TYPES[st],f'r{rel}',NODE_TYPES[dt])
                if key not in allowed:raise ValueError('Actual typed relation differs from bound canonical catalog')
                edges[key]=torch.stack((local[e.src[selected]],local[e.dst[selected]]))
        return edges

    def forward(self,batch,**_unused):
        validate_graph(batch,self.edge_catalog)
        h=self.encode(batch.nodes)
        if batch.edges.num_edges:
            positions={name:(batch.nodes.node_type_id==i).nonzero().flatten() for i,name in enumerate(NODE_TYPES)}
            positions={name:ids for name,ids in positions.items() if ids.numel()}
            edge_dict=self._relations(batch,positions)
            receiving=torch.zeros(h.shape[0],device=h.device,dtype=torch.bool);receiving[batch.edges.dst]=True
            for layer in self.layers:
                output=layer({name:h[ids] for name,ids in positions.items()},edge_dict)
                updated=h.clone()
                for name,value in output.items():
                    if value is not None:
                        ids=positions[name];active=receiving[ids]
                        # Native HGT's bias for an empty row must not invent a message.
                        if active.any():updated[ids[active]]=self.dropout(value[active]).to(h.dtype)
                h=updated
        q=batch.queries
        return self.pair_readout(h[q.drug_node_index],h[q.disease_node_index])

def common_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('layers.')}

@dataclass
class BuiltMatched:
    method:str
    model:nn.Module
    training_seed:int
    initialization:dict

    def forward(self,batch,*,path_batch=None):
        # Compatibility argument only: neither control consumes this object.
        return self.model(batch)

    def metadata(self):
        return dict(method=self.method,training_seed=self.training_seed,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            trainable_parameter_count=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            parameter_shapes={n:list(p.shape) for n,p in self.model.named_parameters()},
            initialization=self.initialization,query_conditioning=False,EP_quality_context=False,path_input=False,
            hidden=128,layers=2 if self.method==HGT else 0,heads=4 if self.method==HGT else 0,dropout=.1,
            D1_cache_frozen=True,D1_projection_trainable=True,selection='FIXED_3G',
            capacity_note='nominal and active capacity differ; no dummy parameters; active state measured per tested sample')

def build_matched_pair(training_seed,pretrained,edge_catalog,*,common_node_encoder=None):
    if type(training_seed) is not int or not 0<=training_seed<2**31:raise ValueError('Certified cell seed required')
    if common_node_encoder is None:
        from evi_hgt.phase85_part3b1.model import build_cell_model
        source=build_cell_model(training_seed)
        common_node_encoder=source.model.encoder.node_encoder
    source_digest=state_digest(common_node_encoder.state_dict())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(_seed(training_seed,'PHASE98_PAIR_READOUT_V1'))
        pair=PairReadout()
        torch.random.default_generator.manual_seed(_seed(training_seed,'PHASE98_HGT_LAYERS_V1'))
        mlp=PairMLP(deepcopy(common_node_encoder),pretrained,deepcopy(pair))
        hgt=MatchedHGT(deepcopy(common_node_encoder),pretrained,deepcopy(pair),edge_catalog)
    if state_digest(mlp.node_encoder.state_dict())!=source_digest or state_digest(hgt.node_encoder.state_dict())!=source_digest:
        raise ValueError('FULL common node initialization tensor identity differs')
    a,b=common_state(mlp),common_state(hgt)
    if set(a)!=set(b) or any(not torch.equal(v,b[k]) for k,v in a.items()):raise ValueError('Matched shared tensors differ')
    ident=dict(full_node_initial_sha256=source_digest,matched_common_sha256=state_digest(a),
        common_tensor_count=len(a),copied_from_fresh_full_factory=True,shared_tensor_equal=True,
        D1_projection_initialization='zero and trainable',readout_seed=_seed(training_seed,'PHASE98_PAIR_READOUT_V1'),
        HGT_layer_seed=_seed(training_seed,'PHASE98_HGT_LAYERS_V1'))
    return {MLP:BuiltMatched(MLP,mlp,training_seed,ident),HGT:BuiltMatched(HGT,hgt,training_seed,ident)}

def make_optimizer(built):
    return torch.optim.AdamW(built.model.parameters(),lr=5e-5,weight_decay=1e-5,betas=(.9,.999),eps=1e-8,foreach=None,fused=None)
