"""Common-corpus TxGNN adapter; the official model implementation is unchanged.

Contrast targets labelled zero in BCE are unlabeled samples, never verified
negative knowledge. Training and evaluation panels are supplied by the common
experiment protocol, not by TxData.prepare_split or TxGNN.finetune.
"""
from __future__ import annotations

import re
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Mapping, Sequence

import dgl
import numpy as np
import polars as pl
import psutil
import torch
from torch import nn
from torch.nn import functional as F

from .model import HeteroRGCN
from .utils import obtain_disease_profile, sim_matrix, Minibatch_NegSampler, initialize_node_embedding


TYPE_ALIASES = {
    'DRG': 'drug', 'DIS': 'disease', 'GEN': 'gene/protein',
    'ANA': 'anatomy', 'BPO': 'biological_process', 'CCO': 'cellular_component',
    'EXP': 'exposure', 'MFN': 'molecular_function',
    'PHE': 'effect/phenotype', 'PWY': 'pathway',
}
DDA_NAMES = {'INDICATION': 'indication', 'CONTRAINDICATION': 'contraindication', 'OFF_LABEL_USE': 'off-label use'}
DDA_ETYPES = [
    ('drug', 'contraindication', 'disease'),
    ('drug', 'indication', 'disease'),
    ('drug', 'off-label use', 'disease'),
    ('disease', 'rev_contraindication', 'drug'),
    ('disease', 'rev_indication', 'drug'),
    ('disease', 'rev_off-label use', 'drug'),
]
TRAIN_ETYPES = [e for e in DDA_ETYPES if 'off-label' not in e[1]]


def native_etype(row: Mapping) -> tuple[str, str, str]:
    """Audited aliases of real canonical relation semantics, never new facts."""
    src, dst = row['source_node_type'], row['target_node_type']
    relation, direction = row['relation'], row['direction']
    if relation in DDA_NAMES and {src, dst} == {'DRG', 'DIS'}:
        name = ('' if src == 'DRG' else 'rev_') + DDA_NAMES[relation]
    elif (src, relation, dst, direction) == ('DIS', 'PARENT', 'DIS', 'forward'):
        name = 'disease_disease'
    elif (src, relation, dst, direction) == ('DIS', 'ASSOCIATED_WITH', 'GEN', 'forward'):
        name = 'rev_disease_protein'
    elif (src, relation, dst, direction) == ('GEN', 'ASSOCIATED_WITH', 'DIS', 'reverse'):
        name = 'disease_protein'
    else:
        # The native RGCN keys weights by relation NAME, so typed relations must
        # have unique names even when their raw labels (e.g. PARENT) coincide.
        name = 'r%d__%s' % (row['relation_id'], re.sub(r'[^A-Za-z0-9_]', '_', row['relation_token']))
    return TYPE_ALIASES[src], name, TYPE_ALIASES[dst]


def native_model(graph, dimension=100, *, proto=True):
    return HeteroRGCN(
        graph, in_size=dimension, hidden_size=dimension, out_size=dimension,
        attention=False, proto=proto, proto_num=3,
        sim_measure='all_nodes_profile', bert_measure='disease_name',
        agg_measure='rarity', num_walks=200, walk_mode='bit', path_length=2,
        split='common_competition', data_folder='', exp_lambda=0.7,
        device=torch.device('cpu'),
    )


def _rebuild_graph(graph, replacements):
    edges = {etype: replacements.get(etype, graph.edges(etype=etype)) for etype in graph.canonical_etypes}
    result = dgl.heterograph(edges, num_nodes_dict={n: graph.num_nodes(n) for n in graph.ntypes})
    for n in graph.ntypes:
        if 'inp' in graph.nodes[n].data:
            result.nodes[n].data['inp'] = graph.nodes[n].data['inp']
    return result


def masked_encoder_graph(graph, pairs: torch.Tensor):
    """Remove each target's indication, contra, off-label and reverse edges."""
    pairs = torch.as_tensor(pairs, dtype=torch.int64)
    if pairs.ndim != 2 or pairs.shape[0] != 2:
        raise ValueError('Pairs must have shape [2, count] using drug/disease local IDs')
    radix = graph.num_nodes('disease')
    keys = torch.unique(pairs[0] * radix + pairs[1])
    replacements = {}
    for etype in DDA_ETYPES:
        if etype not in graph.canonical_etypes:
            continue
        src, dst = graph.edges(etype=etype)
        drugs, diseases = (src, dst) if etype[0] == 'drug' else (dst, src)
        keep = ~torch.isin(drugs * radix + diseases, keys)
        replacements[etype] = src[keep], dst[keep]
    return _rebuild_graph(graph, replacements)


def training_reference_graph(graph, support: torch.Tensor, contra: torch.Tensor):
    """Biology plus exact training P; DDA keys/degrees never use held-out P."""
    replacements = {}
    empty = torch.empty(0, dtype=torch.int64)
    for etype in DDA_ETYPES:
        if 'off-label' in etype[1]:
            replacements[etype] = empty, empty
        else:
            pairs = torch.as_tensor(contra if 'contraindication' in etype[1] else support, dtype=torch.int64)
            replacements[etype] = (pairs[0], pairs[1]) if etype[0] == 'drug' else (pairs[1], pairs[0])
    return _rebuild_graph(graph, replacements)


def _initialize_available_prototypes(pred, graph, dimension):
    """Native initialization, with empty supervised relations skipped explicitly.

    Native DistMultPredictor assumes six nonempty DDA types and torch.stack([])
    fails otherwise. No fake off-label labels are introduced to satisfy it.
    All subsequent prototype scoring uses its original forward method.
    """
    pred.proto = True
    pred.W_gate = {}
    for node_type in pred.node_types_dd:
        temp = nn.Linear(dimension * 2, 1)
        nn.init.xavier_uniform_(temp.weight)
        pred.W_gate[node_type] = temp.to(pred.device)
    pred.k, pred.m = 3, nn.Sigmoid()
    pred.diseases_profile = {}
    pred.sim_all_etypes, pred.diseaseid2id_etypes, pred.diseases_profile_etypes = {}, {}, {}
    for etype in pred.etypes_dd:
        degree = graph.out_degrees(etype=etype) if etype[0] == 'disease' else graph.in_degrees(etype=etype)
        ids = torch.where(degree != 0)[0]
        if not len(ids):
            continue
        if len(ids) <= pred.k:
            raise ValueError(f'Native k={pred.k} prototypes need at least {pred.k + 1} training diseases: {etype}')
        profiles = {i.item(): obtain_disease_profile(graph, i, ['disease_disease', 'rev_disease_protein'], ['disease', 'gene/protein']) for i in ids}
        matrix = torch.stack([profiles[i.item()] for i in ids])
        pred.sim_all_etypes[etype] = sim_matrix(matrix, matrix)
        pred.diseaseid2id_etypes[etype] = dict(zip(ids.numpy(), range(len(ids))))
        pred.diseases_profile_etypes[etype] = profiles


class NativeCompetitionModel:
    """Separate the encoder adjacency from legal training prototype priors."""
    def __init__(self, encoder_graph, training_reference_graph, *, dimension=100):
        self.encoder_graph = encoder_graph
        self.training_reference_graph = training_reference_graph
        self.model = native_model(training_reference_graph, dimension, proto=False)
        _initialize_available_prototypes(self.model.pred, training_reference_graph, dimension)

    def encode(self):
        # return_h exits before either decoder graph is inspected.
        return self.model(self.encoder_graph, None, return_h=True)

    def score_graph(self, graph, *, etypes: Sequence = TRAIN_ETYPES, embeddings=None):
        h = self.encode() if embeddings is None else embeddings
        scores = {}
        for etype in etypes:
            if etype not in self.model.pred.sim_all_etypes:
                raise ValueError(f'No prototype keys for untrained relation {etype}')
            if not graph.num_edges(etype):
                scores[etype] = torch.empty(0, device=self.model.device)
                continue
            single = dgl.heterograph({etype: graph.edges(etype=etype)}, num_nodes_dict={n: graph.num_nodes(n) for n in graph.ntypes})
            values, _ = self.model.pred(single, self.training_reference_graph, h, False, mode='common')
            scores[etype] = values[etype]
        return scores

    def loss(self, positive_graph, contrast_graph, *, etypes: Sequence = TRAIN_ETYPES):
        h = self.encode()
        positive = self.score_graph(positive_graph, etypes=etypes, embeddings=h)
        contrast = self.score_graph(contrast_graph, etypes=etypes, embeddings=h)
        pos, unlabeled = torch.cat([positive[e] for e in etypes]), torch.cat([contrast[e] for e in etypes])
        targets = torch.cat([torch.ones_like(pos), torch.zeros_like(unlabeled)])
        loss = F.binary_cross_entropy(torch.sigmoid(torch.cat([pos, unlabeled])), targets)
        return loss, {'positive': pos, 'unlabeled_contrast': unlabeled}


def _pair_keys(graph, etype, eids=None):
    src, dst = graph.edges(etype=etype) if eids is None else graph.find_edges(eids, etype=etype)
    drug, disease = (src, dst) if etype[0] == 'drug' else (dst, src)
    return drug * graph.num_nodes('disease') + disease


def pretraining_edge_ids(graph, support, contra):
    """All non-DDA relations; DDA optimization only exact training labels."""
    result = {}
    for etype in graph.canonical_etypes:
        if etype in DDA_ETYPES:
            if 'off-label' in etype[1]:
                continue
            pairs = contra if 'contraindication' in etype[1] else support
            keys = pairs[0] * graph.num_nodes('disease') + pairs[1]
            ids = torch.where(torch.isin(_pair_keys(graph, etype), keys))[0]
            # Each common canonical pair is a distinct supervised fact.
            if len(ids) != pairs.shape[1]:
                raise ValueError(f'Common training pair missing or duplicated in graph: {etype}')
        else:
            ids = graph.edges(form='eid', etype=etype)
        if len(ids):
            result[etype] = ids
    return result


class TargetPairExclusion:
    """DGL exclusion callback covering the current DDA target and equivalents."""
    def __init__(self, graph):
        self.graph = graph
        self.keys = {etype: _pair_keys(graph, etype) for etype in DDA_ETYPES}

    def __call__(self, seed_edges):
        values = [_pair_keys(self.graph, etype, ids) for etype, ids in seed_edges.items() if etype in DDA_ETYPES and len(ids)]
        keys = torch.unique(torch.cat(values)) if values else torch.empty(0, dtype=torch.int64)
        return {etype: torch.where(torch.isin(all_keys, keys))[0] for etype, all_keys in self.keys.items()}


class GatedNegativeSampler:
    """Preserve native non-DDA corruption and delegate all DDA U to common gate."""
    def __init__(self, graph, sample_u):
        self.native = Minibatch_NegSampler(graph, 1, 'fix_dst')
        self.sample_u = sample_u

    def __call__(self, graph, eids_dict):
        result = {}
        for etype, ids in eids_dict.items():
            if etype in DDA_ETYPES:
                pairs = self.sample_u(len(ids))
                if pairs.shape != (2, len(ids)):
                    raise ValueError('Common U gateway returned wrong pair shape')
                result[etype] = (pairs[0], pairs[1]) if etype[0] == 'drug' else (pairs[1], pairs[0])
            else:
                result.update(self.native(graph, {etype: ids}))
        return result


def pretraining_loader(graph, support, contra, sample_u, *, batch_size=512, shuffle=True):
    edge_sampler = dgl.dataloading.as_edge_prediction_sampler(
        dgl.dataloading.MultiLayerFullNeighborSampler(2),
        exclude=TargetPairExclusion(graph),
        negative_sampler=GatedNegativeSampler(graph, sample_u),
    )
    return dgl.dataloading.DataLoader(
        graph, pretraining_edge_ids(graph, support, contra), edge_sampler,
        batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0,
    )


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def _save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def _state_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)


def _memory():
    info = psutil.Process().memory_info()
    return {'rss_bytes': info.rss, 'peak_rss_bytes': getattr(info, 'peak_wset', info.rss)}


def _check_gradients(model):
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError('Native model has absent or nonfinite gradients')
    nonzero = sum(int(torch.count_nonzero(g)) for g in gradients)
    if not nonzero:
        raise ValueError('Native model has no nonzero gradient')
    return {'gradient_tensors': len(gradients), 'nonzero_elements': nonzero, 'finite_nonzero': True}


def _pair_tensor(frame):
    return torch.from_numpy(frame.select('drug_type_local_id', 'disease_type_local_id').to_numpy().astype(np.int64).T.copy())


def dda_graph(graph, support, contra):
    edges = {}
    for etype in TRAIN_ETYPES:
        pairs = contra if 'contraindication' in etype[1] else support
        edges[etype] = (pairs[0], pairs[1]) if etype[0] == 'drug' else (pairs[1], pairs[0])
    return dgl.heterograph(edges, num_nodes_dict={n: graph.num_nodes(n) for n in graph.ntypes})


class CommonExport:
    """Only the frozen shared graph and named inner-development panels are read."""
    def __init__(self, dataset, *, seed=42, dimension=100):
        self.root = Path(dataset).resolve()
        self.manifest_path = self.root / 'manifest.json'
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        self.manifest_sha256 = _file_sha(self.manifest_path)
        nodes = pl.read_parquet(self.root / 'nodes.parquet')
        self.drug_local = dict(nodes.filter(pl.col('node_type') == 'DRG').select('node_id', 'type_local_id').iter_rows())
        self.disease_local = dict(nodes.filter(pl.col('node_type') == 'DIS').select('node_id', 'type_local_id').iter_rows())
        counts = {}
        for code, name in TYPE_ALIASES.items():
            local = nodes.filter(pl.col('node_type') == code)['type_local_id'].to_numpy()
            if len(local):
                if not np.array_equal(np.sort(local), np.arange(len(local))):
                    raise ValueError(f'Node IDs are not contiguous for {code}')
                counts[name] = len(local)
        relations = pl.read_parquet(self.root / 'relations.parquet').to_dicts()
        edges, self.relation_mapping = {}, []
        for row in relations:
            etype = native_etype(row)
            if {etype[0], etype[2]} == {'drug', 'disease'} and etype not in DDA_ETYPES:
                raise ValueError(f'Unverified additional clinical relation: {etype}')
            if etype in edges or any(etype[1] == k[1] for k in edges):
                raise ValueError(f'Native typed relation alias collision: {etype}')
            path = Path(row['edge_index_path'])
            if not path.is_absolute():
                path = self.root / path
            array = np.load(path, mmap_mode='c', allow_pickle=False)
            if array.dtype != np.int64 or array.shape != (2, row['edge_count']):
                raise ValueError(f'Invalid exported COO: {path}')
            edges[etype] = torch.from_numpy(array[0]), torch.from_numpy(array[1])
            self.relation_mapping.append({**row, 'native_etype': list(etype)})
        for etype in DDA_ETYPES:
            if etype not in edges:
                raise ValueError(f'Missing real DDA canonical relation: {etype}')
        _seed(seed)
        self.graph = initialize_node_embedding(dgl.heterograph(edges, num_nodes_dict=counts), dimension)
        self.support = _pair_tensor(self.panel('train_support'))
        self.contra = _pair_tensor(self.panel('train_contra'))
        self.reference_graph = training_reference_graph(self.graph, self.support, self.contra)
        self.encoder_graph = masked_encoder_graph(self.graph, torch.cat([self.support, self.contra], dim=1))
        self.positive_graph = dda_graph(self.graph, self.support, self.contra)
        self.allowed_pretrain_eids = pretraining_edge_ids(self.graph, self.support, self.contra)

    def panel(self, name):
        permitted = {'train_support', 'train_contra', 'validation', 'train_probe', 'auxiliary_validation', 'auxiliary_probe'}
        if name not in permitted:
            raise ValueError(f'Panel outside this development competition: {name}')
        return pl.read_parquet(self.root / 'pairs' / f'{name}.parquet')

    def metadata(self):
        return {
            'dataset': str(self.root), 'manifest_sha256': self.manifest_sha256,
            'node_counts': {n: self.graph.num_nodes(n) for n in self.graph.ntypes},
            'full_graph_edges': self.graph.num_edges(),
            'encoder_graph_edges': self.encoder_graph.num_edges(),
            'training_reference_edges': self.reference_graph.num_edges(),
            'typed_relations': len(self.graph.canonical_etypes),
            'pretraining_optimization_edges': sum(len(e) for e in self.allowed_pretrain_eids.values()),
            'train_support_pairs': self.support.shape[1], 'train_contra_pairs': self.contra.shape[1],
        }


class CommonUStream:
    def __init__(self, data, evi_root, seed):
        sys.path.insert(0, str(Path(evi_root).resolve() / 'src'))
        from evi_hgt.competition.data import UnlabeledPolicy
        self.policy = UnlabeledPolicy.from_export(data.root)
        self.drug_local, self.disease_local = data.drug_local, data.disease_local
        self.seed, self.calls, self.pairs = int(seed), 0, 0

    def __call__(self, count):
        if count == 0:
            return torch.empty((2, 0), dtype=torch.int64)
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.calls}'.encode()).digest()[:4], 'little')
        sampled = self.policy.sample(int(count), seed=seed)
        self.policy.validate(sampled)
        self.calls += 1
        self.pairs += count
        return torch.tensor([[self.drug_local[a] for a, _ in sampled], [self.disease_local[b] for _, b in sampled]], dtype=torch.int64)


def _contrast_graph(data, stream):
    edges = {}
    for etype in TRAIN_ETYPES:
        count = data.contra.shape[1] if 'contraindication' in etype[1] else data.support.shape[1]
        pairs = stream(count)
        edges[etype] = (pairs[0], pairs[1]) if etype[0] == 'drug' else (pairs[1], pairs[0])
    return dgl.heterograph(edges, num_nodes_dict={n: data.graph.num_nodes(n) for n in data.graph.ntypes})


def _pretrain_loss(adapter, batch, graph):
    _, positive, contrast, blocks = batch
    _, _, p, u = adapter.model.forward_minibatch(positive, contrast, blocks, graph, pretrain_mode=True)
    return F.binary_cross_entropy(torch.cat([p, u]).reshape(-1), torch.cat([torch.ones_like(p), torch.zeros_like(u)]).reshape(-1))


def resource_preflight(config, data, adapter, output, started):
    """Two zero-update backward probes; no development or held-out metrics."""
    stream = CommonUStream(data, config['evi_root'], config.get('seed', 42) + 101)
    before = _state_digest(adapter.model)
    report = {
        'schema': 'TXGNN_COMMON_RESOURCE_PREFLIGHT_V1', 'status': 'RUNNING',
        'optimizer_steps': 0, 'evaluation_called': False,
        'dimension': config.get('dimension', 100), 'threads': torch.get_num_threads(),
        'protocol_digest': config.get('protocol_digest'), **data.metadata(),
        'load_and_initialize_seconds': time.perf_counter() - started, **_memory(),
    }
    _save_json(output / 'resource_preflight.json', report)
    print(json.dumps({'stage': 'full_finetune_backward', 'status': 'STARTED', **_memory()}), flush=True)
    tick = time.perf_counter()
    loss, _ = adapter.loss(data.positive_graph, _contrast_graph(data, stream))
    loss.backward()
    report['finetune'] = {'seconds': time.perf_counter() - tick, 'loss': float(loss.detach()), **_check_gradients(adapter.model), **_memory()}
    adapter.model.zero_grad(set_to_none=True)
    del loss
    _save_json(output / 'resource_preflight.json', report)
    print(json.dumps({'stage': 'pretrain_batch', 'status': 'STARTED', **report['finetune']}), flush=True)
    loader = pretraining_loader(data.graph, data.support, data.contra, stream, batch_size=config.get('pretrain_batch_size', 1024))
    tick = time.perf_counter()
    prior_u_pairs = stream.pairs
    batch = next(iter(loader))
    sampling_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    loss = _pretrain_loss(adapter, batch, data.reference_graph)
    loss.backward()
    report['pretrain_batch'] = {
        'batch_size': config.get('pretrain_batch_size', 1024), 'sampling_seconds': sampling_seconds,
        'forward_backward_seconds': time.perf_counter() - tick,
        'loss': float(loss.detach()), 'blocks': [{'edges': b.num_edges(), 'source_nodes': sum(b.num_src_nodes(n) for n in b.srctypes), 'destination_nodes': sum(b.num_dst_nodes(n) for n in b.dsttypes)} for b in batch[3]],
        'positive_edges': batch[1].num_edges(), 'contrast_edges': batch[2].num_edges(),
        'DDA_positive_edges': {e[1]: batch[1].num_edges(e) for e in TRAIN_ETYPES},
        'common_gate_DDA_U_pairs': stream.pairs - prior_u_pairs,
        **_check_gradients(adapter.model), **_memory(),
    }
    after = _state_digest(adapter.model)
    if after != before:
        raise ValueError('Zero-update preflight changed model parameters')
    total_batch = report['pretrain_batch']['sampling_seconds'] + report['pretrain_batch']['forward_backward_seconds']
    report.update(
        status='PASS', weight_digest_before=before, weight_digest_after=after,
        total_seconds=time.perf_counter() - started,
        estimated_pretrain_epoch_seconds_from_one_batch=math.ceil(data.metadata()['pretraining_optimization_edges'] / config.get('pretrain_batch_size', 1024)) * total_batch,
        estimated_finetune_500_epoch_seconds_excluding_evaluation=500 * report['finetune']['seconds'],
        estimate_limitation='One observed batch is a rough cost probe, not a measured full-epoch runtime.',
    )
    _save_json(output / 'resource_preflight.json', report)
    return report


@torch.no_grad()
def predict_panels(data, adapter):
    """Role-free pair scoring with the original indication and contra logits."""
    adapter.model.eval()
    h = adapter.encode()
    panels = {}
    for name in ('validation', 'train_probe', 'auxiliary_validation', 'auxiliary_probe'):
        table = data.panel(name)
        pairs = _pair_tensor(table)
        query = dda_graph(data.graph, pairs, pairs)
        result = adapter.score_graph(query, etypes=[('drug', 'indication', 'disease'), ('drug', 'contraindication', 'disease')], embeddings=h)
        support = result[('drug', 'indication', 'disease')].detach().numpy()
        contra = result[('drug', 'contraindication', 'disease')].detach().numpy()
        if not np.isfinite(support).all() or not np.isfinite(contra).all():
            raise ValueError('Nonfinite prediction in development panel')
        panels[name] = table.with_columns(pl.Series('support_logit', support), pl.Series('unsuitability_logit', contra))
    return panels


def _evaluate_panels(panels):
    # NumPy 2 renamed the trapezoidal integrator. TxGNN's native torch/DGL
    # environment needs NumPy 1.x; retain the common metric's exact algorithm.
    if not hasattr(np, 'trapezoid'):
        np.trapezoid = np.trapz
    from evi_hgt.phase85_part2b.runtime import inner_metrics
    result = {name: inner_metrics(panels[name].to_dicts()) for name in ('validation', 'train_probe')}
    objective_logits, objective_labels = [], []
    for row in panels['validation'].iter_rows(named=True):
        objective_logits.append(row['support_logit'])
        objective_labels.append(float(row['role'] == 'P'))
    for name, source in [('auxiliary_validation', 'validation'), ('auxiliary_probe', 'train_probe')]:
        rows = panels[name].to_dicts() + panels[source].filter(pl.col('role') == 'U').to_dicts()
        for row in rows:
            row['support_logit'] = row['unsuitability_logit']
            if name == 'auxiliary_validation':
                objective_logits.append(row['support_logit'])
                objective_labels.append(float(row['role'] == 'P'))
        result[name] = inner_metrics(rows)
        result[name]['channel'] = 'unsuitability'
    z, y = np.asarray(objective_logits, dtype=np.float64), np.asarray(objective_labels)
    result['validation_objective'] = float(np.mean(np.logaddexp(0, z) - y * z))
    for value in result.values():
        if isinstance(value, dict):
            value.pop('disease_metrics', None)
    return result


def _selection_key(metrics, epoch):
    return float(metrics['validation']['macro_auc']), float(metrics['validation']['macro_ap']), -int(epoch)


def _candidate_selection_key(candidate):
    auc, ap = candidate['selection_key'][:2]
    lr = float(candidate['learning_rate'])
    return auc, ap, lr == 0.001, -lr, -int(candidate['epoch'])


def _save_panels(directory, panels):
    paths = {}
    for name, frame in panels.items():
        path = directory / f'{name}_scores.parquet'
        temporary = path.with_suffix('.parquet.tmp')
        frame.write_parquet(temporary)
        os.replace(temporary, path)
        paths[name] = str(path)
    return paths


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def _native_pretrain(config, data, adapter, output, binding):
    receipt_path, checkpoint_path = output / 'pretrain_COMPLETE.json', output / 'pretrain.pt'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['bindings'] != binding or receipt['checkpoint_sha256'] != _file_sha(checkpoint_path):
            raise ValueError('Pretraining recovery binding/checkpoint mismatch')
        return receipt
    epochs = int(config.get('pretrain_epochs', 1))
    if epochs < 1:
        raise ValueError('Competition must retain native all-relation pretraining')
    _seed(int(config.get('seed', 42)) + 11)
    stream = CommonUStream(data, config['evi_root'], config.get('seed', 42) + 102)
    loader = pretraining_loader(data.graph, data.support, data.contra, stream, batch_size=int(config.get('pretrain_batch_size', 65536)))
    optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=float(config.get('pretrain_lr', 1e-3)))
    steps, history = 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        coverage = {e: 0 for e in data.allowed_pretrain_eids}
        weighted_loss, total_edges = 0.0, 0
        adapter.model.train()
        for batch_index, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = _pretrain_loss(adapter, batch, data.reference_graph)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite native pretraining loss')
            loss.backward()
            _check_gradients(adapter.model)
            optimizer.step()
            steps += 1
            count = batch[1].num_edges()
            total_edges += count
            weighted_loss += float(loss.detach()) * count
            for etype in coverage:
                coverage[etype] += batch[1].num_edges(etype)
            progress = {'stage': 'pretrain', 'status': 'RUNNING', 'epoch': epoch, 'batch': batch_index, 'optimizer_steps': steps, 'loss': float(loss.detach()), 'elapsed_seconds': time.perf_counter() - started, 'common_gate_U_pairs': stream.pairs, **_memory()}
            _save_json(output / 'progress.json', progress)
            if batch_index == 1 or batch_index % 10 == 0:
                print(json.dumps(progress), flush=True)
        if any(coverage[e] != len(ids) for e, ids in data.allowed_pretrain_eids.items()):
            raise ValueError('Native pretraining did not cover every allowed relation edge exactly once')
        history.append({'epoch': epoch, 'mean_loss': weighted_loss / total_edges, 'optimization_edges': total_edges, 'relation_coverage': {'|'.join(e): n for e, n in coverage.items()}})
        _save_json(output / 'pretrain_history.json', history)
    _save_checkpoint(checkpoint_path, {'model': adapter.model.state_dict(), 'binding': binding, 'optimizer_steps': steps, 'seed': config.get('seed', 42)})
    receipt = {'status': 'COMPLETE', 'bindings': binding, 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': _file_sha(checkpoint_path), 'optimizer_steps': steps, 'epochs': epochs, 'history': history, 'elapsed_seconds': time.perf_counter() - started, 'common_gate_U_pairs': stream.pairs}
    _save_json(receipt_path, receipt)
    return receipt


def _train_candidate(config, data, adapter, output, binding, pretrain, learning_rate):
    name = 'lr_' + format(float(learning_rate), '.10g').replace('.', 'p')
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    completed_path = directory / 'COMPLETED.json'
    if completed_path.exists():
        result = json.loads(completed_path.read_text(encoding='utf-8'))
        if result['bindings'] != binding or result['checkpoint_sha256'] != _file_sha(result['checkpoint']):
            raise ValueError('Completed candidate recovery binding/checkpoint mismatch')
        return result
    adapter.model.load_state_dict(torch.load(pretrain['checkpoint'], map_location='cpu')['model'])
    _seed(int(config.get('decoder_seed', config.get('seed', 42) + 1001)))
    # Native finetune resets the relation decoder before AdamW optimization.
    nn.init.xavier_uniform_(adapter.model.w_rels)
    start_digest = _state_digest(adapter.model)
    optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=float(learning_rate))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8)
    stream = CommonUStream(data, config['evi_root'], config.get('seed', 42) + 103)
    epochs, every = int(config.get('epochs', 500)), int(config.get('evaluate_every', 25))
    history, best, start_epoch = [], None, 1
    latest_path = directory / 'latest.pt'
    if latest_path.exists():
        previous = torch.load(latest_path, map_location='cpu')
        if previous['binding'] != binding or previous['learning_rate'] != learning_rate:
            raise ValueError('Partial candidate recovery binding mismatch')
        adapter.model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        history, best, start_epoch = previous['history'], previous['best'], previous['epoch'] + 1
        stream.calls, stream.pairs = previous['u_calls'], previous['u_pairs']
        _restore_rng(previous['rng'])
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        adapter.model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = adapter.loss(data.positive_graph, _contrast_graph(data, stream))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite native finetuning loss')
        loss.backward()
        _check_gradients(adapter.model)
        optimizer.step()
        scheduler.step(float(loss.detach()))
        record = {'epoch': epoch, 'optimizer_steps': epoch, 'train_loss': float(loss.detach()), 'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_seconds': time.perf_counter() - started}
        if epoch % every == 0 or epoch == epochs:
            panels = predict_panels(data, adapter)
            metrics = _evaluate_panels(panels)
            record['metrics'] = metrics
            record.update(validation_AUC=metrics['validation']['macro_auc'], validation_AP=metrics['validation']['macro_ap'], validation_objective=metrics['validation_objective'], matched_G_AUC=metrics['train_probe']['macro_auc'], matched_G_AP=metrics['train_probe']['macro_ap'])
            key = _selection_key(metrics, epoch)
            if best is None or key > tuple(best['selection_key']):
                path = directory / 'best.pt'
                _save_checkpoint(path, {'model': adapter.model.state_dict(), 'binding': binding, 'epoch': epoch, 'learning_rate': learning_rate, 'pretrain_checkpoint_sha256': pretrain['checkpoint_sha256']})
                score_paths = _save_panels(directory, panels)
                best = {'epoch': epoch, 'selection_key': list(key), 'checkpoint': str(path), 'checkpoint_sha256': _file_sha(path), 'metrics': metrics, 'scores': score_paths, 'weight_digest': _state_digest(adapter.model)}
                _save_json(directory / 'best.json', best)
            print(json.dumps({'candidate': name, **record}), flush=True)
        history.append(record)
        _save_json(directory / 'history.json', history)
        progress = {'candidate': name, 'stage': 'finetune', 'status': 'RUNNING', 'epoch': epoch, 'optimizer_steps': epoch, 'driver_optimizer_steps_minimum': pretrain['optimizer_steps'] + epoch, 'train_loss': record['train_loss'], 'common_gate_U_pairs': stream.pairs, **_memory()}
        _save_json(directory / 'progress.json', progress)
        _save_json(output / 'progress.json', progress)
        if epoch % every == 0 or epoch == epochs:
            _save_checkpoint(latest_path, {'model': adapter.model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'binding': binding, 'epoch': epoch, 'learning_rate': learning_rate, 'history': history, 'best': best, 'u_calls': stream.calls, 'u_pairs': stream.pairs, 'rng': _rng_state()})
    if best is None:
        raise ValueError('Candidate completed without a valid inner-development checkpoint')
    result = {'status': 'COMPLETE', 'candidate': name, 'bindings': binding, 'learning_rate': learning_rate, 'epochs': epochs, 'optimizer_steps': epochs, 'training_started': True, 'initial_state_digest': start_digest, 'pretrain_checkpoint_sha256': pretrain['checkpoint_sha256'], **best}
    _save_json(directory / 'progress.json', {'stage': 'finetune', 'status': 'COMPLETED', 'epoch': epochs, 'optimizer_steps': epochs})
    result['artifact_hashes'] = {p.name: _file_sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'COMPLETED.json' and not p.name.endswith('.tmp')}
    _save_json(completed_path, result)
    return result


def train_native(config, data, adapter, output):
    if not config.get('protocol_digest'):
        raise ValueError('Training requires the frozen common protocol digest')
    binding = {'protocol_digest': config['protocol_digest'], 'manifest_sha256': data.manifest_sha256, 'config_digest': _json_digest(config), 'source_sha256': {name: _file_sha(Path(__file__).parent / name) for name in ('competition.py', 'model.py', 'utils.py')}}
    binding_path = output / 'bindings.json'
    if binding_path.exists() and json.loads(binding_path.read_text(encoding='utf-8')) != binding:
        raise ValueError('Output directory is bound to another protocol/config/source')
    _save_json(binding_path, binding)
    pretrain = _native_pretrain(config, data, adapter, output, binding)
    candidates = [_train_candidate(config, data, adapter, output, binding, pretrain, lr) for lr in config.get('learning_rates', [3e-4, 1e-3, 3e-3])]
    if len({c['initial_state_digest'] for c in candidates}) != 1:
        raise ValueError('Learning-rate candidates did not start from identical pretrained initialization')
    selected = max(candidates, key=_candidate_selection_key)
    result = {'schema': 'TXGNN_COMMON_COMPETITION_RESULT_V1', 'status': 'COMPLETE', 'protocol_digest': config['protocol_digest'], 'training_started': True, 'optimizer_steps': pretrain['optimizer_steps'] + sum(c['optimizer_steps'] for c in candidates), 'bindings': binding, 'data': data.metadata(), 'pretraining': pretrain, 'candidates': candidates, 'selected': selected, 'selected_learning_rate': selected['learning_rate'], 'scope': 'DEVELOPMENT_INNER_ONLY', 'outer_or_final_evaluated': False, 'negative_semantics': 'U_IS_UNKNOWN_BCE_ZERO_IS_CONTRAST_ONLY', 'native_deviations': ['Common canonical graph and exact training/development panels replace PrimeKG/native split.', 'DDA negatives use common eligible/observed/heldout gate; non-DDA negatives retain native fix_dst.', 'Current DDA target pairs and all equivalent/reverse DDA facts are excluded from encoder messages.', 'Exact train support/contra only are optimizer positives and prototype keys/degrees; off-label remains background context only.', 'Pretraining batch size is 65536, selected only from zero-update CPU resource probes.', 'Untrained empty off-label prototype initialization is skipped; original predictor handles trained relations.', 'Selection uses common inner disease-macro AUC, AP, then earliest epoch; no native automatic test evaluation.']}
    _save_json(output / 'COMPLETE.json', result)
    _save_json(output / 'progress.json', {'status': 'COMPLETE', 'stage': 'complete', 'optimizer_steps': result['optimizer_steps']})
    return result


def run_config(config):
    action = config.get('action', 'preflight')
    if action not in {'preflight', 'train'}:
        raise ValueError('Action must be preflight or train')
    output = Path(config['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if 'outputs' not in output.parts:
        raise ValueError('Runtime artifacts must be under outputs/')
    if int(config.get('dimension', 100)) != 100 and not config.get('synthetic_test_only', False):
        raise ValueError('Competition preserves native 100-dimensional model')
    config = dict(config)
    config.setdefault('evi_root', str(Path(__file__).resolve().parents[2] / 'EviHGT'))
    torch.set_num_threads(int(config.get('cpu_threads', 8)))
    _seed(int(config.get('seed', 42)))
    sys.path.insert(0, str(Path(config['evi_root']) / 'src'))
    from evi_hgt.competition.data import verify_export
    started = time.perf_counter()
    try:
        _save_json(output / 'progress.json', {'stage': 'data_verification', 'status': 'RUNNING', 'optimizer_steps': 0})
        verify_export(config['dataset'])
        data = CommonExport(config['dataset'], seed=config.get('seed', 42), dimension=config.get('dimension', 100))
        _save_json(output / 'relation_mapping.json', data.relation_mapping)
        _save_json(output / 'progress.json', {'stage': 'native_model_initialization', 'status': 'RUNNING', 'optimizer_steps': 0, **data.metadata()})
        adapter = NativeCompetitionModel(data.encoder_graph, data.reference_graph, dimension=config.get('dimension', 100))
        if action == 'preflight':
            result = resource_preflight(config, data, adapter, output, started)
            _save_json(output / 'progress.json', {'stage': 'preflight', 'status': 'COMPLETED', 'optimizer_steps': 0, **_memory()})
        else:
            result = train_native(config, data, adapter, output)
        return result
    except Exception as exc:
        _save_json(output / 'FAILED.json', {'action': action, 'error_type': type(exc).__name__, 'error': str(exc), **_memory()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    result = run_config(config)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
