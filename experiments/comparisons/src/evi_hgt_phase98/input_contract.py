"""Phase98 node/graph contract. No label or protected-panel readers."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from evi_hgt.models.input_contract import NodeBatch
from evi_hgt.phase85_v2.features import DRUG_DIM, DISEASE_DIM, PACKED_DIM

ROOT=Path(__file__).resolve().parents[2]
NODE_TYPES=('ANA','BPO','CCO','DIS','DRG','EXP','GEN','MFN','PHE','PWY')
SIDE_DIGEST='3082740f214cc114f02f298ced27b3cb274359cefbccdb30bffdffd4d4629133'

def file_sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def certify_side_features(root=None):
    root=Path(root or ROOT/'data/processed/phase85_part2/features_v2')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('digest')!=SIDE_DIGEST or manifest.get('learned_transform') is not False:
        raise ValueError('Frozen SideInfo manifest identity changed')
    observed={}
    for name in ('drug_ids','drug_values','disease_ids','disease_values'):
        p=root/(name+'.npy')
        if file_sha(p)!=manifest['files'][name+'.npy']:raise ValueError('SideInfo array changed: '+name)
        arr=np.load(p,mmap_mode='r',allow_pickle=False)
        observed[name]=dict(shape=list(arr.shape),dtype=str(arr.dtype),sha256=manifest['files'][name+'.npy'])
    if observed['drug_values']['shape'][1]!=DRUG_DIM or observed['disease_values']['shape'][1]!=DISEASE_DIM:
        raise ValueError('Actual SideInfo dimensions differ from imported feature packer')
    if observed['drug_ids']['shape'][0]!=observed['drug_values']['shape'][0] or observed['disease_ids']['shape'][0]!=observed['disease_values']['shape'][0]:
        raise ValueError('Feature entity row count mismatch')
    return dict(schema='PHASE98_SIDE_INPUT_CONTRACT_V1',manifest_sha256=file_sha(root/'manifest.json'),
        feature_digest=SIDE_DIGEST,drug_width=DRUG_DIM,disease_width=DISEASE_DIM,packed_width=PACKED_DIM,arrays=observed,
        drug_fields=[['morgan_radius2',0,2048],['drug_gene_profile',2048,2560],['structure_available',2560,2561],['gene_available',2561,2562]],
        disease_fields=[['ontology_profile',0,512],['HPO_profile',512,1024],['ontology_available',1024,1025],['HPO_available',1025,1026]],
        raw_sources_followed=False,protected_labels_read=False)

def validate_nodes(nodes):
    n=nodes.num_nodes
    if nodes.node_id is None or nodes.node_type_id.dtype!=torch.long or nodes.node_id.dtype!=torch.long:
        raise ValueError('Canonical int64 entity/type IDs required')
    if nodes.node_id.shape!=(n,) or nodes.node_type_id.shape!=(n,) or nodes.graph_id.shape!=(n,):
        raise ValueError('Node vectors differ')
    if nodes.features is None or nodes.features.shape!=(n,PACKED_DIM):raise ValueError('SideInfo shape mismatch')
    if n and (int(nodes.node_type_id.min())<0 or int(nodes.node_type_id.max())>=len(NODE_TYPES)):
        raise ValueError('Unknown node type')
    if not torch.isfinite(nodes.features).all():raise ValueError('Nonfinite SideInfo')

def endpoint_nodes(batch):
    """Read only query endpoint rows; no edges/evidence/neighbour row consumer."""
    q=batch.queries;n=batch.nodes
    if q.num_queries!=1:raise ValueError('Frozen graph batch is one')
    indices=torch.cat((q.drug_node_index,q.disease_node_index))
    if indices.dtype!=torch.long or int(indices.min())<0 or int(indices.max())>=n.num_nodes:
        raise ValueError('Invalid endpoint index')
    chosen=NodeBatch(node_type_id=n.node_type_id[indices],graph_id=n.graph_id[indices],
        features=n.features[indices] if n.features is not None else None,
        node_id=n.node_id[indices] if n.node_id is not None else None)
    validate_nodes(chosen)
    if chosen.node_type_id.tolist()!=[4,3]:raise ValueError('Query endpoint types must be DRG/DIS')
    if not torch.equal(chosen.graph_id,torch.zeros_like(chosen.graph_id)) or q.graph_id.tolist()!=[0]:
        raise ValueError('Query endpoint graph identity differs')
    return chosen

def validate_graph(batch,edge_catalog):
    validate_nodes(batch.nodes);q=batch.queries;nodes=batch.nodes;e=batch.edges
    if q.num_queries!=1 or nodes.num_nodes>1000 or e.num_edges>5000:raise ValueError('Frozen graph batch/budget exceeded')
    endpoint_nodes(batch)
    for value in (e.src,e.dst,e.edge_type_id,e.graph_id):
        if value.dtype!=torch.long or value.shape!=(e.num_edges,):raise ValueError('Message index vector invalid')
    if e.num_edges:
        if min(int(e.src.min()),int(e.dst.min()))<0 or max(int(e.src.max()),int(e.dst.max()))>=nodes.num_nodes:
            raise ValueError('Message node index invalid')
        if int(e.edge_type_id.min())<0 or int(e.edge_type_id.max())>=len(edge_catalog):raise ValueError('Unknown relation index')
        if not torch.equal(nodes.graph_id[e.src],e.graph_id) or not torch.equal(nodes.graph_id[e.dst],e.graph_id):
            raise ValueError('Message graph identity differs')
        d=q.drug_node_index[0];s=q.disease_node_index[0]
        if (((e.src==d)&(e.dst==s))|((e.src==s)&(e.dst==d))).any():
            raise ValueError('Target pair message including reverse/equivalent relation is forbidden')
    # Global held-out fact masking is certified by the upstream graph assembler;
    # this local guard cannot infer unprovided forbidden fact membership.
    return True
