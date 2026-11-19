"""HGTDR on the frozen common development corpus.

The default action is a finite resource preflight with zero optimizer steps.
Training needs an explicit, digest-bound config. The full graph stays on CPU;
native HGT sampling receives both query endpoints and masks every current DDA
query before sampling. Unknown comparators are never called clinical negatives.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import numpy as np
import polars as pl
import torch
from torch import nn
import torch_sparse  # Registers the native CPU hgt_sample operation.

from src.rebuild_features import file_sha256, write_json
from src.runtime import HGT, MLPPredictor, compute_loss

ROOT = Path(__file__).resolve().parents[1]
EVI = ROOT.parent / 'EviHGT'
sys.path.insert(0, str(EVI / 'src'))


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def seed_for(seed, *parts):
    return int(digest([int(seed), *parts])[:15], 16) % (2**31 - 1)


def source_hashes():
    files = ['src/competition.py', 'src/competition_features.py', 'src/runtime.py',
             'src/vendor/hgt_conv_231.py', 'src/rebuild_features.py', 'src/rebuild_smiles.py']
    return {name: file_sha256(ROOT / name) for name in files}


def to_csc(edge_index, target_count):
    require(edge_index.shape[0] == 2 and edge_index.dtype == np.int64, 'COO must be int64 [2,E]')
    order = np.lexsort((edge_index[0], edge_index[1]))
    source, target = edge_index[:, order]
    pointer = np.concatenate(([0], np.cumsum(np.bincount(target, minlength=target_count)))).astype(np.int64)
    return np.ascontiguousarray(source), np.ascontiguousarray(target), pointer, order


class QueryMaskedSampler:
    """Native hgt_sample with all-relation pair exclusion before neighborhood search."""
    def __init__(self, node_counts, relations, arrays):
        self.node_counts, self.relations = dict(node_counts), list(relations)
        self.edge_types, self.colptr, self.rows, self.dda = [], {}, {}, {}
        self.last_removed = 0
        for record, array in zip(self.relations, arrays):
            key = (record['source_type'], record['relation_token'], record['target_type'])
            token = '__'.join(key)
            require('__' not in key[0] and '__' not in key[2] and '__' not in key[1], 'Reserved relation delimiter')
            require(token not in self.rows, 'Duplicate typed traversal key')
            require(np.all(array >= 0) and np.all(array[0] < node_counts[key[0]]) and
                    np.all(array[1] < node_counts[key[2]]), 'Edge endpoint out of range')
            source, target, pointer, _ = to_csc(array, node_counts[key[2]])
            self.edge_types.append(key)
            self.rows[token], self.colptr[token] = torch.from_numpy(source), torch.from_numpy(pointer)
            if {key[0], key[2]} == {'drug', 'disease'}:
                drug, disease = (source, target) if key[0] == 'drug' else (target, source)
                self.dda[token] = {'source': source, 'target': target, 'key': key,
                    'pair_code': drug * node_counts['disease'] + disease}
        require(len(arrays) == len(self.relations), 'Relation array count mismatch')

    def sample(self, pairs, fanout, *, seed, max_nodes=20000, max_edges=1500000):
        pairs = np.asarray(pairs, dtype=np.int64)
        require(pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) > 0, 'Expected query pairs [N,2]')
        require(np.all(pairs >= 0) and np.all(pairs[:, 0] < self.node_counts['drug']) and
                np.all(pairs[:, 1] < self.node_counts['disease']), 'Query endpoint outside domain')
        require(len(fanout) == 3 and all(int(n) > 0 for n in fanout), 'HGTDR requires three bounded sampling budgets')
        query_codes = np.unique(pairs[:, 0] * self.node_counts['disease'] + pairs[:, 1])
        rows, colptr = dict(self.rows), dict(self.colptr)
        removed = 0
        for token, item in self.dda.items():
            keep = ~np.isin(item['pair_code'], query_codes)
            removed += int((~keep).sum())
            if keep.all():
                continue
            rows[token] = torch.from_numpy(np.ascontiguousarray(item['source'][keep]))
            counts = np.bincount(item['target'][keep], minlength=self.node_counts[item['key'][2]])
            colptr[token] = torch.from_numpy(np.concatenate(([0], np.cumsum(counts))).astype(np.int64))
        seeds = {'drug': torch.from_numpy(np.unique(pairs[:, 0])),
                 'disease': torch.from_numpy(np.unique(pairs[:, 1]))}
        cpu_rng = torch.random.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(int(seed))
            nodes, sources, targets, _ = torch.ops.torch_sparse.hgt_sample(
                colptr, rows, seeds, {kind: list(fanout) for kind in self.node_counts}, 3)
        finally:
            torch.random.set_rng_state(cpu_rng)
        nodes = {kind: nodes.get(kind, torch.empty(0, dtype=torch.long)) for kind in self.node_counts}
        edges = {}
        for key in self.edge_types:
            token = '__'.join(key)
            edges[key] = torch.stack((sources[token], targets[token])) if token in sources else torch.empty((2, 0), dtype=torch.long)
            if token in self.dda and edges[key].shape[1]:
                source, target = nodes[key[0]][edges[key][0]], nodes[key[2]][edges[key][1]]
                drug, disease = (source, target) if key[0] == 'drug' else (target, source)
                require(not np.isin((drug * self.node_counts['disease'] + disease).numpy(), query_codes).any(),
                        'Current query DDA edge survived masking')
        node_total, edge_total = sum(len(v) for v in nodes.values()), sum(v.shape[1] for v in edges.values())
        require(node_total <= max_nodes and edge_total <= max_edges, 'Sample exceeds the frozen node/edge cap')
        local = []
        for column, kind in enumerate(('drug', 'disease')):
            inverse = torch.full((self.node_counts[kind],), -1, dtype=torch.long)
            inverse[nodes[kind]] = torch.arange(len(nodes[kind]))
            mapped = inverse[torch.from_numpy(pairs[:, column].copy())]
            require(bool((mapped >= 0).all()), 'Native sampling dropped a query endpoint')
            local.append(mapped)
        self.last_removed = removed
        return {'nodes': nodes, 'edges': edges, 'pair_index': torch.stack(local),
                'node_count': node_total, 'edge_count': edge_total, 'masked_traversals': removed}


class CommonDataset:
    def __init__(self, dataset, features_dir):
        from evi_hgt.competition.data import UnlabeledPolicy, verify_export
        self.root, features_dir = Path(dataset).resolve(), Path(features_dir).resolve()
        self.export_verification = verify_export(self.root)
        self.features_root = features_dir if features_dir.name == 'features' else features_dir / 'features'
        self.features_parent = self.features_root.parent
        self.nodes = pl.read_parquet(self.root / 'nodes.parquet').sort('global_node_id')
        self.relations = pl.read_parquet(self.root / 'relations.parquet').sort('relation_id').to_dicts()
        self.manifest_path = self.root / 'manifest.json'
        require(self.manifest_path.is_file(), 'Common export manifest is required')
        self.manifest_sha256 = file_sha256(self.manifest_path)
        self.feature_manifest_path = self.features_root / 'manifest.json'
        self.feature_manifest = json.loads(self.feature_manifest_path.read_text(encoding='utf-8'))
        require(self.feature_manifest['status'] == 'COMPLETE', 'Frozen encoder features are incomplete')
        require(self.feature_manifest.get('optimizer_steps') == 0 and self.feature_manifest.get('training') is False,
                'Feature generation must be fixed inference')
        require(file_sha256(self.features_parent / 'input_manifest.json') == self.feature_manifest['input_manifest_sha256'],
                'Feature source provenance receipt changed')
        inputs_path = self.features_parent / 'feature_inputs.parquet'
        require(file_sha256(inputs_path) == self.feature_manifest['feature_inputs_sha256'], 'Feature inputs hash mismatch')
        inputs = pl.read_parquet(inputs_path).sort('global_node_id')
        for column in ('global_node_id', 'type_local_id', 'node_type', 'node_id'):
            require(inputs[column].to_list() == self.nodes[column].to_list(), f'Common feature/domain mismatch: {column}')
        self.counts, self.features, self.available, self.abbreviations = {}, {}, {}, {}
        records = {item['node_type']: item for item in self.feature_manifest['types']}
        for group in self.nodes.partition_by('node_type', maintain_order=True):
            group = group.sort('type_local_id')
            abbreviation, kind = group['node_type'][0], group['human_node_type'][0]
            require(group['human_node_type'].n_unique() == 1 and kind not in self.counts, 'Node type name mismatch')
            require(np.array_equal(group['type_local_id'].to_numpy(), np.arange(group.height)), 'Type-local row IDs not contiguous')
            record = records[abbreviation]
            array_path = self.features_root / f"type_{record['node_type_id']:02d}.npy"
            mask_path = self.features_root / f"type_{record['node_type_id']:02d}_available.npy"
            require(file_sha256(array_path) == record['sha256'] and file_sha256(mask_path) == record['mask_sha256'],
                    'Feature or availability hash mismatch')
            array, mask = np.load(array_path, allow_pickle=False), np.load(mask_path, allow_pickle=False)
            require(array.shape == (group.height, 768) and array.dtype == np.float32 and np.isfinite(array).all(),
                    'Invalid feature values or dimensions')
            require(mask.shape == (group.height,) and mask.dtype == np.bool_ and not np.any(array[~mask]),
                    'Missing encoder rows must have explicit masks and zero storage')
            expected = inputs.filter(pl.col('node_type') == abbreviation).sort('type_local_id')
            require(np.array_equal(mask, expected['available'].to_numpy()), 'Feature masks disagree with source inputs')
            input_digest = hashlib.sha256()
            for entity, text, present in expected.select('node_id', 'encoder_text', 'available').iter_rows():
                input_digest.update(json.dumps([entity, text if present else None, bool(present)],
                                               ensure_ascii=False).encode('utf-8'))
                input_digest.update(b'\n')
            require(input_digest.hexdigest() == record['input_sha256'], 'Encoder feature row/input binding mismatch')
            self.counts[kind], self.abbreviations[kind] = group.height, abbreviation
            self.features[kind], self.available[kind] = torch.from_numpy(array), torch.from_numpy(mask)
        arrays = [np.load(self.root / row['edge_index_path'], allow_pickle=False) for row in self.relations]
        self.sampler = QueryMaskedSampler(self.counts, self.relations, arrays)
        self.u_policy = UnlabeledPolicy.from_export(self.root)
        self.panels = {}
        for name in ('train_support', 'validation', 'train_probe'):
            panel = pl.read_parquet(self.root / 'pairs' / f'{name}.parquet')
            require(panel.select('drug_id', 'disease_id').n_unique() == panel.height, 'Duplicate common panel pair')
            require(set(panel['role']).issubset({'P'} if name == 'train_support' else {'P', 'U'}), 'Unexpected panel role')
            require(set(panel['channel']) == {'support'}, 'HGTDR supports the indication channel')
            self.panels[name] = panel
        self.lookup = {kind: dict(group.select('node_id', 'type_local_id').iter_rows()) for kind, group in (
            (kind, self.nodes.filter(pl.col('human_node_type') == kind)) for kind in ('drug', 'disease'))}
        self.metadata = (list(self.counts), self.sampler.edge_types)

    def native_pairs(self, frame):
        return frame.select('drug_type_local_id', 'disease_type_local_id').to_numpy().astype(np.int64)

    def unlabeled(self, count, seed):
        values = self.u_policy.sample(count, seed=seed)
        self.u_policy.validate(values)
        if isinstance(values, pl.DataFrame):
            values = values.to_dicts()
        result = []
        for row in values:
            if isinstance(row, dict):
                drug, disease = row['drug_id'], row['disease_id']
            else:
                drug, disease = row
            result.append([self.lookup['drug'][drug], self.lookup['disease'][disease]])
        return np.asarray(result, dtype=np.int64)

    def batch(self, pairs, config, seed, device):
        sample = self.sampler.sample(pairs, config['fanout'], seed=seed,
            max_nodes=config.get('max_nodes', 20000), max_edges=config.get('max_edges', 1500000))
        sample['features'] = {kind: self.features[kind][ids].to(device) for kind, ids in sample['nodes'].items()}
        sample['available'] = {kind: self.available[kind][ids].to(device) for kind, ids in sample['nodes'].items()}
        sample['edges'] = {key: value.to(device) for key, value in sample['edges'].items()}
        sample['pair_index'] = sample['pair_index'].to(device)
        return sample


class CompetitionHGTDR(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.missing = nn.ParameterDict({kind: nn.Parameter(torch.zeros(768)) for kind in metadata[0]})
        self.encoder = HGT(metadata)
        self.predictor = MLPPredictor()

    def forward(self, batch):
        features = {kind: torch.where(batch['available'][kind].unsqueeze(-1), value,
                                     self.missing[kind].unsqueeze(0))
                    for kind, value in batch['features'].items()}
        drug, disease = self.encoder(features, batch['edges'])
        pair = batch['pair_index']
        return self.predictor(drug[pair[0]], disease[pair[1]])


def initialized_model(data, batch, device, seed):
    torch.manual_seed(seed)
    model = CompetitionHGTDR(data.metadata).to(device)
    model.eval()
    with torch.no_grad():
        model(batch)
    return model


def compact_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != 'disease_metrics'}


def score_panel(model, data, name, config, device):
    from evi_hgt.phase85_part2b.runtime import inner_metrics
    require(name in {'validation', 'train_probe'}, 'Only the frozen development panels can be scored')
    frame = data.panels[name]
    pairs, scores = data.native_pairs(frame), []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(pairs), config['batch_size']):
            batch = data.batch(pairs[offset:offset + config['batch_size']], config,
                seed_for(config['seed'], 'evaluation', name, offset), device)
            result = model(batch).view(-1)
            require(torch.isfinite(result).all(), 'Nonfinite development score')
            scores.extend(result.cpu().tolist())
    result = frame.with_columns(pl.Series('support_logit', scores))
    metrics = inner_metrics(result.select('drug_id', 'disease_id', 'role', 'support_logit').to_dicts())
    return result, metrics


def checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.pt.partial')
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return file_sha256(path)


def resource_preflight(data, config, output):
    device = config.get('device', 'cuda')
    supports = data.native_pairs(data.panels['train_support'])
    validation_pairs = data.native_pairs(data.panels['validation'])
    positive_order = np.random.default_rng(seed_for(config['seed'], 'preflight-positive-order')).permutation(len(supports))
    validation_order = np.random.default_rng(seed_for(config['seed'], 'preflight-validation-order')).permutation(len(validation_pairs))
    trials = config.get('probe_trials', [
        {'batch_size': 64, 'fanout': [512, 512, 512]},
        {'batch_size': 128, 'fanout': [512, 512, 512]}])
    reports = []
    for trial_index, trial in enumerate(trials):
        active = dict(config, **trial)
        half = active['batch_size'] // 2
        pairs = np.concatenate((supports[positive_order[:half]], data.unlabeled(half, seed_for(config['seed'], 'preflight'))))
        labels = torch.cat((torch.ones(half, 1), torch.zeros(half, 1))).to(device)
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model, batch, scores, loss = None, None, None, None
        try:
            sample_started = time.perf_counter()
            batch = data.batch(pairs, active, seed_for(config['seed'], 'preflight-sample'), device)
            sample_seconds = time.perf_counter() - sample_started
            model = initialized_model(data, batch, device, config['seed'])
            model.train()
            times, losses = [], []
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                if device == 'cuda':
                    torch.cuda.synchronize()
                before = time.perf_counter()
                scores = model(batch)
                loss = compute_loss(scores, labels)
                require(torch.isfinite(loss), 'Nonfinite resource-probe loss')
                loss.backward()
                require(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),
                        'Nonfinite resource-probe gradient')
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - before)
                losses.append(float(loss.detach().cpu()))
            model.zero_grad(set_to_none=True)
            model.eval()
            evaluation_times = []
            for repeat in range(2):
                before = time.perf_counter()
                evaluation_batch = data.batch(validation_pairs[validation_order[:active['batch_size']]], active,
                    seed_for(config['seed'], 'preflight-validation-sampling', repeat), device)
                with torch.no_grad():
                    evaluation_scores = model(evaluation_batch)
                require(torch.isfinite(evaluation_scores).all(), 'Nonfinite evaluation resource-probe scores')
                if device == 'cuda':
                    torch.cuda.synchronize()
                evaluation_times.append(time.perf_counter() - before)
                del evaluation_batch, evaluation_scores
            training_steps = math.ceil(len(supports) / half)
            validation_steps = math.ceil(len(validation_pairs) / active['batch_size'])
            report = {'status': 'PASS', **trial, 'nodes': batch['node_count'], 'edges': batch['edge_count'],
                'masked_current_query_traversals': batch['masked_traversals'], 'sampling_seconds': sample_seconds,
                'forward_backward_seconds': times, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                'evaluation_batch_seconds_including_sampling': evaluation_times,
                'training_steps_per_epoch': training_steps, 'validation_batches_per_epoch': validation_steps,
                'estimated_epoch_seconds_including_validation': float(training_steps * (sample_seconds + np.mean(times[1:])) +
                    validation_steps * np.mean(evaluation_times)),
                'loss_is_finite': all(math.isfinite(v) for v in losses), 'all_gradients_finite': True,
                'optimizer_steps': 0, 'wall_seconds': time.perf_counter() - started}
            del model, batch, scores, loss
        except (RuntimeError, ValueError) as exc:
            report = {'status': 'FAILED', **trial, 'error': str(exc), 'optimizer_steps': 0,
                      'wall_seconds': time.perf_counter() - started}
            if device == 'cuda':
                torch.cuda.empty_cache()
        reports.append(report)
        model, batch, scores, loss = None, None, None, None
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        write_json(output / 'resource_preflight.json', {'status': 'RUNNING', 'trials': reports, 'optimizer_steps': 0})
        print(json.dumps(report), flush=True)
    successful = [r for r in reports if r['status'] == 'PASS']
    require(successful, 'No bounded native HGTDR resource probe passed')
    selected = min(successful, key=lambda r: r['estimated_epoch_seconds_including_validation'])
    steps = math.ceil(len(supports) / (selected['batch_size'] // 2))
    result = {'status': 'PASS', 'trials': reports, 'recommendation': {key: selected[key] for key in ('batch_size', 'fanout')},
        'train_support_pairs': len(supports), 'optimizer_steps_per_epoch': steps,
        'estimated_training_seconds_per_epoch_excluding_evaluation': float(steps * (np.mean(selected['forward_backward_seconds'][1:]) + selected['sampling_seconds'])),
        'estimated_epoch_seconds_including_validation': selected['estimated_epoch_seconds_including_validation'],
        'estimated_3_candidates_300_epochs_hours': selected['estimated_epoch_seconds_including_validation'] * 900 / 3600,
        'optimizer_steps': 0, 'selected_on_resource_measurements_only': True,
        'dataset_manifest_sha256': data.manifest_sha256,
        'feature_manifest_sha256': file_sha256(data.feature_manifest_path),
        'source_hashes': source_hashes(), 'sealed_final_labels_read': False}
    write_json(output / 'resource_preflight.json', result)
    return result


def candidate_name(lr):
    return 'lr_' + format(float(lr), '.8g').replace('.', 'p').replace('-', 'm')


def metric_key(metrics, epoch):
    return (float(metrics['macro_auc']), float(metrics['macro_ap']), -int(epoch))


def run_candidate(data, config, root, lr, bindings):
    directory = root / candidate_name(lr)
    directory.mkdir(parents=True, exist_ok=True)
    complete_path = directory / 'COMPLETED.json'
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding='utf-8'))
        require(complete['bindings'] == bindings, 'Completed candidate binding differs')
        for filename, expected in complete['artifact_hashes'].items():
            require(file_sha256(directory / filename) == expected, 'Completed candidate artifact changed')
        return complete
    device, seed, batch_size = config.get('device', 'cuda'), config['seed'], config['batch_size']
    require(batch_size >= 4 and batch_size % 2 == 0, 'Training pair batch must be even and >= 4')
    supports = data.native_pairs(data.panels['train_support'])
    first = data.batch(supports[:min(batch_size, len(supports))], config, seed_for(seed, 'initialize'), device)
    model = initialized_model(data, first, device, seed)
    del first
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)
    history, best, optimizer_steps, first_epoch, elapsed_before = [], None, 0, 1, 0.0
    last_path, last_receipt_path = directory / 'last.pt', directory / 'last.json'
    if last_receipt_path.exists():
        receipt = json.loads(last_receipt_path.read_text(encoding='utf-8'))
        require(receipt['bindings'] == bindings and receipt['learning_rate'] == lr, 'Resume binding mismatch')
        require(file_sha256(last_path) == receipt['checkpoint_sha256'], 'Resume checkpoint hash mismatch')
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.random.set_rng_state(state['cpu_rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        best, history = state['best'], state['history']
        optimizer_steps, first_epoch, elapsed_before = state['optimizer_steps'], state['epoch'] + 1, state['elapsed_seconds']
        require(file_sha256(directory / 'best.pt') == best['checkpoint_sha256'], 'Best checkpoint differs from last complete epoch')
    elif last_path.exists() or (directory / 'history.json').exists() or (directory / 'best.pt').exists():
        raise ValueError('Unreceipted candidate artifacts exist; cannot silently restart')
    start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch, config['epochs'] + 1):
        permutation = np.random.default_rng(seed_for(seed, 'positive-order', epoch)).permutation(len(supports))
        unlabeled = data.unlabeled(len(supports), seed_for(seed, 'unlabeled', epoch))
        model.train()
        epoch_loss, count, current_lr = 0.0, 0, float(optimizer.param_groups[0]['lr'])
        epoch_start = time.perf_counter()
        masked, maximum_nodes, maximum_edges = 0, 0, 0
        for batch_index, offset in enumerate(range(0, len(supports), batch_size // 2)):
            indices = permutation[offset:offset + batch_size // 2]
            p = supports[indices]
            u = unlabeled[offset:offset + len(indices)]
            pairs = np.concatenate((p, u))
            labels = torch.cat((torch.ones(len(p), 1), torch.zeros(len(u), 1))).to(device)
            batch = data.batch(pairs, config, seed_for(seed, 'training-sample', epoch, batch_index), device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch)
            loss = compute_loss(scores, labels)
            require(torch.isfinite(loss), 'Nonfinite training loss')
            loss.backward()
            require(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), 'Nonfinite training gradient')
            optimizer.step()
            optimizer_steps += 1
            epoch_loss += float(loss.detach().cpu()) * len(pairs)
            count += len(pairs)
            masked += batch['masked_traversals']
            maximum_nodes, maximum_edges = max(maximum_nodes, batch['node_count']), max(maximum_edges, batch['edge_count'])
            if batch_index % 10 == 0:
                write_json(directory / 'progress.json', {'status': 'RUNNING', 'stage': 'TRAINING', 'epoch': epoch,
                    'learning_rate': lr, 'optimizer_steps': optimizer_steps, 'positive_pairs_seen_this_epoch': min(offset + len(indices), len(supports)),
                    'total_positive_pairs': len(supports), 'elapsed_seconds': elapsed_before + time.perf_counter() - start,
                    'protocol_digest': config['protocol_digest']})
        require(count == 2 * len(supports), 'Training epoch did not cover every frozen positive')
        scheduler.step()
        write_json(directory / 'progress.json', {'status': 'RUNNING', 'stage': 'INNER_VALIDATION', 'epoch': epoch,
            'learning_rate': lr, 'optimizer_steps': optimizer_steps, 'elapsed_seconds': elapsed_before + time.perf_counter() - start})
        validation, metrics = score_panel(model, data, 'validation', config, device)
        elapsed = elapsed_before + time.perf_counter() - start
        item = {'epoch': epoch, 'optimizer_steps': optimizer_steps, 'lr': current_lr, 'train_loss': epoch_loss / count,
            'positive_pairs_seen': len(supports), 'unlabeled_pairs_seen': len(supports), 'all_training_positives_covered': True,
            'masked_current_query_traversals': masked, 'max_sample_nodes': maximum_nodes, 'max_sample_edges': maximum_edges,
            'epoch_seconds': time.perf_counter() - epoch_start, 'elapsed_seconds': elapsed,
            'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
            **compact_metrics(metrics)}
        if best is None or metric_key(metrics, epoch) > metric_key(best['metrics'], best['epoch']):
            sha = checkpoint(directory / 'best.pt', {'model': model.state_dict(), 'epoch': epoch,
                'optimizer_steps': optimizer_steps, 'bindings': bindings, 'learning_rate': float(lr)})
            best = {'epoch': epoch, 'optimizer_steps': optimizer_steps, 'metrics': metrics,
                    'checkpoint_sha256': sha, 'learning_rate': lr, 'bindings': bindings}
            write_json(directory / 'best.json', best)
            validation.write_parquet(directory / 'validation_scores.parquet', compression='zstd')
        history.append(item)
        write_json(directory / 'history.json', history)
        sha = checkpoint(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'epoch': epoch, 'optimizer_steps': optimizer_steps,
            'best': best, 'history': history, 'elapsed_seconds': elapsed, 'cpu_rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else []})
        write_json(last_receipt_path, {'bindings': bindings, 'learning_rate': lr, 'epoch': epoch,
            'optimizer_steps': optimizer_steps, 'checkpoint_sha256': sha})
        print(json.dumps({'candidate': candidate_name(lr), **{k: item[k] for k in (
            'epoch', 'optimizer_steps', 'train_loss', 'macro_auc', 'macro_ap', 'epoch_seconds', 'elapsed_seconds')}}), flush=True)
    saved = torch.load(directory / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(saved['model'])
    probe_scores, probe = score_panel(model, data, 'train_probe', config, device)
    probe_scores.write_parquet(directory / 'train_probe_scores.parquet', compression='zstd')
    write_json(directory / 'train_probe_metrics.json', probe)
    completed = {'status': 'COMPLETE', 'method': 'HGTDR_COMMON_DOMAIN_ADAPTED', 'learning_rate': lr,
        'epochs': config['epochs'], 'optimizer_steps': optimizer_steps, 'best_epoch': best['epoch'],
        'best_metrics': best['metrics'], 'train_probe_metrics': probe,
        'elapsed_seconds': elapsed_before + time.perf_counter() - start, 'bindings': bindings,
        'scope': 'DEVELOPMENT_INNER_VALIDATION', 'outer_test_accessed': False, 'sealed_final_labels_read': False,
        'artifact_hashes': {name: file_sha256(directory / name) for name in ('best.pt', 'best.json', 'history.json',
            'validation_scores.parquet', 'train_probe_scores.parquet', 'train_probe_metrics.json')}}
    write_json(complete_path, completed)
    write_json(directory / 'progress.json', {'status': 'COMPLETE', 'stage': 'COMPLETE', 'epoch': config['epochs'],
        'optimizer_steps': optimizer_steps, 'learning_rate': lr, 'best_epoch': best['epoch']})
    del model, optimizer, scheduler
    if device == 'cuda':
        torch.cuda.empty_cache()
    return completed


def run(config):
    action = config.get('action', 'preflight')
    require(action in {'preflight', 'train'}, 'Unknown competition action')
    require(config.get('cpu_threads', 4) in range(1, 5), 'CPU thread budget must be <=4')
    torch.set_num_threads(config.get('cpu_threads', 4))
    config = dict({'seed': 42, 'batch_size': 128, 'fanout': [512, 512, 512], 'epochs': 300,
                   'device': 'cuda', 'cpu_threads': 4}, **config)
    output = Path(config['output_dir']).resolve()
    require(ROOT / 'outputs' in output.parents, 'Runtime artifacts must be under HGTDR/outputs')
    output.mkdir(parents=True, exist_ok=True)
    if config['device'] == 'cuda':
        require(torch.cuda.is_available(), 'CUDA device unavailable')
        torch.cuda.set_per_process_memory_fraction(config.get('gpu_memory_fraction', .80))
    if action == 'train':
        require(config.get('protocol_digest') and isinstance(config['protocol_digest'], str), 'Frozen protocol digest required')
        require(config['epochs'] == 300 and config.get('learning_rates') == [.0003, .001, .003],
                'Training must match the frozen three-candidate 300-epoch development budget')
    data = CommonDataset(config['dataset'], config['features_dir'])
    if action == 'preflight':
        return resource_preflight(data, config, output)
    bindings = {'config_digest': digest(config), 'protocol_digest': config['protocol_digest'],
        'dataset_manifest_sha256': data.manifest_sha256,
        'feature_manifest_sha256': file_sha256(data.feature_manifest_path), 'source_hashes': source_hashes(),
        'supervision': 'ALL_TRAIN_SUPPORT_P_PLUS_EQUAL_GLOBAL_UNIFORM_U',
        'loss': 'UPSTREAM_PER_EXAMPLE_POS_WEIGHT_BCE_P_VS_U_NOT_CONFIRMED_NEGATIVE',
        'contraindications': 'SEPARATE_CONTEXT_RELATION_NOT_INDICATION_NEGATIVES',
        'missing_feature_policy': 'TYPE_SHARED_LEARNABLE_768_VECTOR_ZERO_INITIALIZED',
        'evaluation_sampling': 'DETERMINISTIC_FIXED_PER_PANEL_BATCH'}
    path = output / 'bindings.json'
    if path.exists():
        require(json.loads(path.read_text(encoding='utf-8')) == bindings, 'Run binding changed; no silent resume')
    else:
        write_json(path, bindings)
    write_json(output / 'config.json', config)
    completed = []
    try:
        for lr in config['learning_rates']:
            write_json(output / 'progress.json', {'status': 'RUNNING', 'stage': 'CANDIDATE', 'learning_rate': lr,
                'completed_candidates': len(completed), 'protocol_digest': config['protocol_digest']})
            completed.append(run_candidate(data, config, output, lr, bindings))
        # Across candidates exact inner-score ties prefer the native LR; within
        # one candidate the checkpoint rule already prefers the earlier epoch.
        chosen = max(completed, key=lambda r: (float(r['best_metrics']['macro_auc']),
                     float(r['best_metrics']['macro_ap']), r['learning_rate'] == .001,
                     -r['learning_rate'], -r['best_epoch']))
        result = {'status': 'COMPLETE', 'scope': 'DEVELOPMENT_INNER_VALIDATION', 'bindings': bindings,
            'selected_learning_rate': chosen['learning_rate'], 'selected_candidate': candidate_name(chosen['learning_rate']),
            'selection_rule': 'MAX_INNER_AUC_THEN_AP_THEN_NATIVE_LR_THEN_SMALLER_LR_THEN_EARLIER_EPOCH',
            'protocol_digest': config['protocol_digest'], 'training_started': True,
            'optimizer_steps': sum(row['optimizer_steps'] for row in completed),
            'candidates': completed, 'outer_test_accessed': False, 'sealed_final_labels_read': False}
        write_json(output / 'COMPLETE.json', result)
        write_json(output / 'progress.json', {'status': 'COMPLETE', 'stage': 'COMPLETE', 'completed_candidates': len(completed)})
        return result
    except Exception as exc:
        write_json(output / 'FAILED.json', {'status': 'FAILED', 'error': str(exc), 'traceback': traceback.format_exc(),
            'completed_candidates': len(completed), 'bindings': bindings, 'sealed_final_labels_read': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    result = run(config)
    print(json.dumps({key: result[key] for key in ('status', 'selected_learning_rate') if key in result}), flush=True)


if __name__ == '__main__':
    main()
