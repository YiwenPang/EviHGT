"""Exact bounded tensorization and device transfer."""
from dataclasses import fields, replace
import torch
from evi_hgt.models.input_contract import EviHGTBatch, NodeBatch, EdgeBatch, QueryBatch

def tensorize_indexed(index, gathered, *, node_ids, traversal_edges,
                      drug_global, disease_global):
    ordered_nodes = sorted(node_ids)
    local_node = {v: i for i, v in enumerate(ordered_nodes)}
    local_fact = {int(v): i for i, v in enumerate(gathered['fact_ids'])}
    edge_facts = [local_fact[f] for _, _, f in traversal_edges]
    long = lambda values: torch.tensor(values, dtype=torch.long)
    evidence = gathered['evidence']
    edge_fact_tensor = long(edge_facts)
    return EviHGTBatch(
        nodes=NodeBatch(node_type_id=long(index.node_type_ids(ordered_nodes).tolist()),
                        graph_id=torch.zeros(len(ordered_nodes), dtype=torch.long),
                        node_id=long(ordered_nodes)),
        edges=EdgeBatch(src=long([local_node[a] for a, _, _ in traversal_edges]),
                        dst=long([local_node[b] for _, b, _ in traversal_edges]),
                        edge_type_id=evidence.fact_edge_type_id[edge_fact_tensor],
                        fact_index=edge_fact_tensor,
                        graph_id=torch.zeros(len(edge_facts), dtype=torch.long)),
        evidence=evidence,
        queries=QueryBatch(drug_node_index=long([local_node[drug_global]]),
                          disease_node_index=long([local_node[disease_global]]), graph_id=long([0])))

def move_model_batch(batch, device):
    device = torch.device(device)
    if device.type == 'cpu':
        return batch
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if hasattr(value, '__dataclass_fields__'):
            return replace(value, **{f.name: move(getattr(value, f.name)) for f in fields(value)})
        return value
    return move(batch)
