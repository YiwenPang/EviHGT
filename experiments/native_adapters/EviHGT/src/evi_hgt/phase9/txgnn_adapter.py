"""Bounded, offline, training-only qualification of the optional TxGNN baseline.

This module never calls the historical automatic train/evaluate driver. M4 is
not in frozen G1. These APIs grant neither a full epoch nor formal execution.
"""
from pathlib import Path
import hashlib
import importlib
import json
import random
import sys
import types

from .identity import ROOT, G1_SHA256, frozen_science, digest, file_sha

NATIVE_ROOT = ROOT.parent / 'TxGNN'
EXPORT_ROOT = ROOT / 'data/processed/competition_20261009/pilot_v1'
PARENT_HASHES = {
    'competition.py': 'c73960492068322159b0bf185708477b412be2d3eb07bd3db208d7382a9adc66',
    'model.py': '09aad184845bdfc8e65b6bc10f1012324a36923cce27d6c56317552db16fdfb2',
    'utils.py': 'b5e57e3b20c21186dd867937e5edd106f980646598efb6f5cd6b279556f02618',
}


def verify_native_sources():
    result = {}
    for name, expected in PARENT_HASHES.items():
        path = NATIVE_ROOT / 'txgnn' / name
        actual = file_sha(path)
        if actual != expected:
            raise ValueError('TxGNN parent source changed: ' + name)
        result[str(path)] = actual
    return result


def require_qualification(operation):
    if operation not in {'STATIC', 'TEST_ONLY_SINGLE_BATCH'}:
        raise PermissionError('Optional M4 is not frozen or authorized for formal execution')


def qualification_binding(repeat, fold):
    if (repeat, fold) not in {(0, 1), (1, 0)}:
        raise PermissionError('Only the two prespecified training cells may be qualified')
    cell = next(c for c in frozen_science()['cells']
                if (c['repeat_index'], c['fold_index']) == (repeat, fold))
    return {'schema': 'PHASE92B_TXGNN_TEST_ONLY_V1', 'method': 'TXGNN_TEST_ONLY',
            'repeat': repeat, 'fold': fold, 'seed': cell['training_seed'],
            'cell_digest': cell['cell_digest'], 'g1_sha256': G1_SHA256,
            'native_sources': verify_native_sources(), 'formal_authorized': False}


def verify_fold_binding(corpus, binding, support_pairs, contra_pairs):
    expected = qualification_binding(binding['repeat'], binding['fold'])
    if binding != expected:
        raise ValueError('Wrong training cell binding')
    cb = corpus['bindings']
    for name, value in [('fold_manifest_digest', binding['cell_digest']),
                        ('original_repeat_index', binding['repeat']),
                        ('fold_index', binding['fold']), ('training_seed', binding['seed'])]:
        if cb.get(name) != value:
            raise ValueError('Cross-cell corpus identity: ' + name)
    cell = next(c for c in frozen_science()['cells'] if c['cell_digest'] == binding['cell_digest'])
    if (len(support_pairs), len(contra_pairs)) != (cell['budget']['full_training_support_P'], cell['budget']['full_training_contra_P']):
        raise ValueError('Training supervision counts differ from frozen cell')


class TrainingOnlyU:
    """Native uniform global U policy with a committed deterministic call cursor."""
    def __init__(self, *, drugs, diseases, observed, forbidden, drug_local, disease_local, seed):
        self.drugs, self.diseases = tuple(sorted(drugs)), tuple(sorted(diseases))
        if not self.drugs or not self.diseases or len(set(self.drugs)) != len(self.drugs) or len(set(self.diseases)) != len(self.diseases):
            raise ValueError('Nonempty unique eligible drug/disease domains required')
        self.excluded = frozenset(tuple(p) for p in observed) | frozenset(tuple(p) for p in forbidden)
        self.drug_local, self.disease_local = dict(drug_local), dict(disease_local)
        if not set(self.drugs) <= self.drug_local.keys() or not set(self.diseases) <= self.disease_local.keys():
            raise ValueError('Eligible node identity missing')
        drug_set, disease_set = frozenset(self.drugs), frozenset(self.diseases)
        n = sum(d in drug_set and s in disease_set for d, s in self.excluded)
        if n >= len(self.drugs) * len(self.diseases):
            raise ValueError('No legal U remains')
        self.seed, self.calls, self.pairs = int(seed), 0, 0
        self.domain_digest = digest({'drugs': self.drugs, 'diseases': self.diseases,
                                     'excluded': sorted(self.excluded)})

    def sample_pairs(self, count):
        if type(count) is not int or count < 0:
            raise ValueError('U count must be nonnegative integer')
        if count == 0:
            return ()
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.calls}'.encode()).digest()[:4], 'little')
        rng, values = random.Random(seed), []
        for _ in range(max(1000, count * 100)):
            pair = rng.choice(self.drugs), rng.choice(self.diseases)
            if pair not in self.excluded:
                values.append(pair)
            if len(values) == count:
                self.calls += 1
                self.pairs += count
                return tuple(values)
        raise ValueError('Legal U sampling failed; no fallback allowed')

    def __call__(self, count):
        import torch
        pairs = self.sample_pairs(count)
        return torch.tensor([[self.drug_local[d] for d, _ in pairs],
                             [self.disease_local[s] for _, s in pairs]], dtype=torch.int64)

    def state(self):
        return dict(seed=self.seed, calls=self.calls, pairs=self.pairs, domain_digest=self.domain_digest)

    def restore(self, value):
        if set(value) != {'seed', 'calls', 'pairs', 'domain_digest'} or value['seed'] != self.seed or value['domain_digest'] != self.domain_digest:
            raise ValueError('TxGNN U stream identity mismatch')
        if any(type(value[k]) is not int or value[k] < 0 for k in ('calls', 'pairs')):
            raise ValueError('Invalid committed U cursor')
        self.calls, self.pairs = value['calls'], value['pairs']


def load_native():
    """Import only bound local native modules, bypassing package download APIs."""
    verify_native_sources()
    name = '_phase92b_native_txgnn'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(NATIVE_ROOT / 'txgnn')]
        sys.modules[name] = package
    return importlib.import_module(name + '.competition')


def prepare_fold(*, corpus, support_pairs, contra_pairs, forbidden_pairs, binding, export_root=EXPORT_ROOT, audit_only=False):
    """Build native graphs from current corpus IDs, never old pilot DDA/panels.

    The old export supplies only certified non-DDA COO and immutable node IDs.
    Every retained non-DDA fact must be in the new corpus, and coverage must be
    complete. All DDA adjacency is reconstructed from current legal fact rows.
    """
    require_qualification('TEST_ONLY_SINGLE_BATCH')
    expected = qualification_binding(binding['repeat'], binding['fold'])
    if binding != expected:
        raise ValueError('Wrong training cell binding')
    support_pairs, contra_pairs = set(map(tuple, support_pairs)), set(map(tuple, contra_pairs))
    verify_fold_binding(corpus, binding, support_pairs, contra_pairs)
    return _prepare_fold_core(corpus=corpus, support_pairs=support_pairs, contra_pairs=contra_pairs,
        forbidden_pairs=forbidden_pairs, binding=binding, export_root=export_root, audit_only=audit_only)


def _prepare_fold_core(*, corpus, support_pairs, contra_pairs, forbidden_pairs, binding,
                       export_root=EXPORT_ROOT, audit_only=False):
    """Shared construction only; the public caller must first verify its cell contract.

    The Phase92b public API keeps its original two-cell gate. Phase92c validates
    complete original membership hashes before reusing this identical data flow.
    """
    import numpy as np
    import polars as pl
    import torch
    import dgl
    native = load_native()
    support_pairs, contra_pairs = set(map(tuple, support_pairs)), set(map(tuple, contra_pairs))
    forbidden_pairs = set(map(tuple, forbidden_pairs))
    if (support_pairs | contra_pairs) & forbidden_pairs:
        raise ValueError('Protected pair in training supervision')
    export_root = Path(export_root).resolve()
    manifest = json.loads((export_root / 'manifest.json').read_text(encoding='utf-8'))
    if file_sha(export_root / 'manifest.json') != '06d1286b03d5fee126b3a0221820218035217f7d68f6572bdc90ab46380d1f72':
        raise ValueError('Historical non-DDA export binding changed')
    if corpus.get('digest') != digest({k: v for k, v in corpus.items() if k != 'digest'}):
        raise ValueError('Current corpus manifest digest changed')
    for name in ('nodes.parquet', 'relations.parquet'):
        if file_sha(export_root / name) != manifest['files'][name]['sha256']:
            raise ValueError('Native export identity metadata changed')
    nodes = pl.read_parquet(export_root / 'nodes.parquet')
    drug_local = dict(nodes.filter(pl.col('node_type') == 'DRG').select('node_id', 'type_local_id').iter_rows())
    disease_local = dict(nodes.filter(pl.col('node_type') == 'DIS').select('node_id', 'type_local_id').iter_rows())
    # Polars partition dictionary uses one-element tuple keys.
    counts = {native.TYPE_ALIASES[k[0] if isinstance(k, tuple) else k]: len(v) for k, v in nodes.partition_by('node_type', as_dict=True).items()}
    dda_rows, nondda_ids = [], []
    for entry in corpus['fact_files']:
        expected_sha = corpus['outputs'].get(entry['path'], corpus['inputs'].get(entry['path']))
        if expected_sha is None or file_sha(entry['path']) != expected_sha:
            raise ValueError('Current corpus fact identity changed')
        if entry.get('component', Path(entry['path']).stem) == 'drug_disease':
            frame = pl.read_parquet(entry['path'], columns=['canonical_fact_id', 'source_node_type', 'source_node_id', 'target_node_type', 'target_node_id', 'relation'])
            clinical = ((pl.col('source_node_type') == 'DRG') & (pl.col('target_node_type') == 'DIS')) | ((pl.col('source_node_type') == 'DIS') & (pl.col('target_node_type') == 'DRG'))
            if frame.filter(~clinical).height:
                raise ValueError('Clinical corpus component has inconsistent entity types')
            dda_rows.extend(frame.to_dicts())
        else:
            nondda_ids.append(pl.read_parquet(entry['path'], columns=['canonical_fact_id'])['canonical_fact_id'].to_numpy())
    admitted = np.concatenate(nondda_ids).astype(np.uint64, copy=False)
    admitted.sort()
    if len(np.unique(admitted)) != len(admitted):
        raise ValueError('Duplicate corpus non-DDA fact')
    edges, covered, edge_counts = {}, [], {}
    for row in pl.read_parquet(export_root / 'relations.parquet').iter_rows(named=True):
        etype = native.native_etype(row)
        if etype in native.DDA_ETYPES:
            continue
        if {etype[0], etype[2]} == {'drug', 'disease'}:
            raise ValueError('Unverified equivalent clinical relation')
        for key in ('edge_index_path', 'fact_ids_path'):
            file = export_root / row[key]
            if file_sha(file) != manifest['files'][row[key]]['sha256']:
                raise ValueError('Non-DDA export payload changed')
        ids = np.load(export_root / row['fact_ids_path'], mmap_mode='r', allow_pickle=False)
        array = np.load(export_root / row['edge_index_path'], mmap_mode='c', allow_pickle=False)
        keep = np.isin(ids, admitted, assume_unique=False)
        edge_counts[str(etype)] = int(keep.sum())
        if not audit_only:
            edges[etype] = torch.from_numpy(array[0][keep]), torch.from_numpy(array[1][keep])
        if row['direction'] == 'forward':
            covered.append(ids[keep])
    coverage = np.concatenate(covered) if covered else np.empty(0, dtype=np.uint64)
    coverage.sort()
    if not np.array_equal(coverage, admitted):
        raise ValueError('Native non-DDA export lacks current corpus fact coverage')
    clinical = {e: [[], []] for e in native.DDA_ETYPES}
    for row in dda_rows:
        if row['relation'] not in native.DDA_NAMES:
            raise ValueError('Unverified clinical fact semantic')
        if row['source_node_type'] == 'DRG':
            pair = row['source_node_id'], row['target_node_id']
        else:
            pair = row['target_node_id'], row['source_node_id']
        if pair in forbidden_pairs:
            raise ValueError('Current corpus contains a protected clinical fact')
        d, s = drug_local[pair[0]], disease_local[pair[1]]
        rel = native.DDA_NAMES[row['relation']]
        for e, a, b in [(('drug', rel, 'disease'), d, s), (('disease', 'rev_' + rel, 'drug'), s, d)]:
            clinical[e][0].append(a)
            clinical[e][1].append(b)
    edges.update({e: (torch.tensor(a, dtype=torch.int64), torch.tensor(b, dtype=torch.int64)) for e, (a, b) in clinical.items()})
    if audit_only:
        return dict(status='PASS', scope='TEST_ONLY_STATIC_FOLD_GRAPH_AUDIT', binding=binding,
                    corpus_digest=corpus['digest'], nondda_facts_verified=len(admitted),
                    nondda_message_edges=sum(edge_counts.values()), clinical_fact_count=len(dda_rows),
                    support_training_pairs=len(support_pairs), contra_training_pairs=len(contra_pairs),
                    native_dgl_graph_created=False, optimizer_updates=0, evaluation_calls=0,
                    protected_clinical_facts=0, old_pilot_dda_loaded=False)
    native._seed(binding['seed'])
    graph = native.initialize_node_embedding(dgl.heterograph(edges, num_nodes_dict=counts), 100)
    def tensor(pairs):
        pairs = sorted(pairs)
        return torch.tensor([[drug_local[d] for d, _ in pairs], [disease_local[s] for _, s in pairs]], dtype=torch.int64)
    support, contra = tensor(support_pairs), tensor(contra_pairs)
    reference = native.training_reference_graph(graph, support, contra)
    encoder = native.masked_encoder_graph(graph, torch.cat([support, contra], dim=1))
    result = types.SimpleNamespace(graph=graph, reference_graph=reference, encoder_graph=encoder,
        positive_graph=native.dda_graph(graph, support, contra), support=support, contra=contra,
        allowed_pretrain_eids=native.pretraining_edge_ids(graph, support, contra), drug_local=drug_local,
        disease_local=disease_local, binding=binding, corpus_digest=corpus['digest'],
        graph_edges=graph.num_edges(), nondda_facts_verified=len(admitted), clinical_fact_count=len(dda_rows))
    return result
