"""Shared pilot graph and legal P/U panels for the external-model competition.

The source is the certified EviHGT corpus, never the sealed holdout or legacy
supervision/split loaders. Computed reverse traversals retain the original
canonical fact ID and are not new biological observations. Large edge arrays
are written in bounded batches and remain CPU memory-mappable.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('POLARS_MAX_THREADS', '4')

import numpy as np
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / 'data/processed/competition_20261009/pilot_v1'
DEFAULT_REPORT = ROOT / 'outputs/competition_20261009/data_audit.json'
HUMAN_TYPES = {
    'ANA': 'anatomy', 'BPO': 'biological_process', 'CCO': 'cellular_component',
    'DIS': 'disease', 'DRG': 'drug', 'EXP': 'exposure', 'GEN': 'gene/protein',
    'MFN': 'molecular_function', 'PHE': 'effect/phenotype', 'PWY': 'pathway',
}
PAIR_COLUMNS = ['drug_id', 'disease_id', 'role', 'channel', 'drug_global_id',
                'disease_global_id', 'drug_type_local_id', 'disease_type_local_id']
PANELS = ('train_support', 'train_contra', 'validation', 'train_probe',
          'auxiliary_validation', 'auxiliary_probe')
FACT_COLUMNS = ['canonical_fact_id', 'source_node_type', 'source_node_id',
                'source_global_id', 'target_node_type', 'target_node_id',
                'target_global_id', 'relation']


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _bound(value):
    return dict(value, digest=_digest(value))


def _read_bound(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('digest') != _digest({k: v for k, v in value.items() if k != 'digest'}):
        raise ValueError('Shared dataset manifest digest mismatch')
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    with partial.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def _relative_file(directory, relative):
    directory = Path(directory).resolve()
    result = (directory / relative).resolve()
    if result == directory or directory not in result.parents:
        raise ValueError('Export artifact path escapes the dataset directory')
    return result


def normalize_nodes(frame):
    """Add readable types while preserving every canonical/global/local ID."""
    required = ['global_node_id', 'type_local_id', 'node_type', 'node_id']
    if not set(required) <= set(frame.columns) or frame.height == 0:
        raise ValueError('A nonempty canonical node identity table is required')
    if frame.select(pl.any_horizontal(pl.col(c).is_null() for c in required)).to_series().any():
        raise ValueError('Node identity must be nonnull')
    if frame['global_node_id'].n_unique() != frame.height or frame.select('node_type', 'node_id').n_unique() != frame.height:
        raise ValueError('Node identity must be unique; duplicate node detected')
    if set(frame['node_type'].unique()) - set(HUMAN_TYPES):
        raise ValueError('Unverified node type in canonical index')
    result = frame.sort('global_node_id')
    if not np.array_equal(result['global_node_id'].to_numpy(), np.arange(result.height)):
        raise ValueError('Canonical global node identity is not contiguous from zero')
    for kind in result['node_type'].unique():
        values = result.filter(pl.col('node_type') == kind)['type_local_id'].sort().to_numpy()
        if not np.array_equal(values, np.arange(len(values))):
            raise ValueError('Type-local node identity must be unique and contiguous')
    return result.with_columns(pl.col('node_type').replace_strict(HUMAN_TYPES).alias('human_node_type'))


def _pair(value):
    if len(value) != 2 or any(not isinstance(v, str) or not v for v in value):
        raise ValueError('Pairs require two nonempty canonical string IDs')
    return tuple(value)


@dataclass(frozen=True)
class UnlabeledPolicy:
    """Uniform global U sampling with replacement, never confirmed negatives."""
    eligible_drugs: tuple[str, ...]
    eligible_diseases: tuple[str, ...]
    observed_pairs: frozenset[tuple[str, str]]
    forbidden_pairs: frozenset[tuple[str, str]]
    _drugs: frozenset = field(init=False, repr=False)
    _diseases: frozenset = field(init=False, repr=False)
    _excluded: frozenset = field(init=False, repr=False)
    _legal_pair_count: int = field(init=False, repr=False)

    def __post_init__(self):
        for name in ('eligible_drugs', 'eligible_diseases'):
            raw = tuple(getattr(self, name))
            if not raw or len(set(raw)) != len(raw) or any(not isinstance(v, str) or not v for v in raw):
                raise ValueError('Eligible IDs must be nonempty, unique canonical strings')
            object.__setattr__(self, name, tuple(sorted(raw)))
        object.__setattr__(self, 'observed_pairs', frozenset(_pair(v) for v in self.observed_pairs))
        object.__setattr__(self, 'forbidden_pairs', frozenset(_pair(v) for v in self.forbidden_pairs))
        object.__setattr__(self, '_drugs', frozenset(self.eligible_drugs))
        object.__setattr__(self, '_diseases', frozenset(self.eligible_diseases))
        excluded = self.observed_pairs | self.forbidden_pairs
        object.__setattr__(self, '_excluded', excluded)
        blocked = sum(d in self._drugs and s in self._diseases for d, s in excluded)
        object.__setattr__(self, '_legal_pair_count', len(self._drugs) * len(self._diseases) - blocked)

    def validate(self, pairs):
        for value in pairs:
            pair = _pair(value)
            if pair[0] not in self._drugs or pair[1] not in self._diseases:
                raise ValueError('U endpoint outside the eligible domain')
            if pair in self.observed_pairs:
                raise ValueError('Observed clinical pair cannot be sampled as U')
            if pair in self.forbidden_pairs:
                raise ValueError('Heldout or forbidden pair cannot be sampled as U')

    def sample(self, count, *, seed):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('U count must be a nonnegative integer')
        if count == 0:
            return ()
        if self._legal_pair_count == 0:
            raise ValueError('No legal U pair remains; no fallback is allowed')
        rng = random.Random(int(seed))
        values = []
        for _ in range(max(1000, count * 100)):
            pair = (rng.choice(self.eligible_drugs), rng.choice(self.eligible_diseases))
            if pair in self._excluded:
                continue
            values.append(pair)
            if len(values) == count:
                return tuple(values)
        raise ValueError('Legal U rejection budget exhausted; no fallback is allowed')

    @classmethod
    def from_export(cls, dataset):
        dataset = Path(dataset).resolve()
        names = ('eligible_ids.json', 'observed_pairs.parquet', 'rolefree_forbidden.parquet', 'nodes.parquet')
        if not (dataset / 'manifest.json').is_file():
            raise ValueError('U sampling requires a complete READY export manifest')
        manifest = _read_bound(dataset / 'manifest.json')
        if manifest.get('schema') != 'EVIHGT_COMPETITION_PILOT_DATA_V1' or manifest.get('status') != 'READY':
            raise ValueError('U sampling requires a complete READY export manifest')
        if not set(names) <= set(manifest['files']):
            raise ValueError('U policy source is missing from the shared export manifest')
        for name in names:
            record = manifest['files'][name]
            path = _relative_file(dataset, name)
            if _sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
                raise ValueError('U policy source hash differs from the shared export')
        nodes = normalize_nodes(pl.read_parquet(dataset / 'nodes.parquet'))
        return _read_policy(dataset, nodes)


def _read_policy(dataset, nodes):
    """Load role-free exclusions and bind the eligible domain to canonical nodes."""
    eligible = json.loads((dataset / 'eligible_ids.json').read_text(encoding='utf-8'))
    observed = pl.read_parquet(dataset / 'observed_pairs.parquet', columns=['drug_id', 'disease_id'])
    forbidden = pl.read_parquet(dataset / 'rolefree_forbidden.parquet', columns=['drug_id', 'disease_id'])
    policy = UnlabeledPolicy(tuple(eligible['eligible_drugs']), tuple(eligible['eligible_diseases']),
                             frozenset(observed.iter_rows()), frozenset(forbidden.iter_rows()))
    for kind, domain in (('DRG', policy._drugs), ('DIS', policy._diseases)):
        canonical = set(nodes.filter(pl.col('node_type') == kind)['node_id'])
        if not domain <= canonical:
            raise ValueError('U eligible domain contains an unknown canonical node identity')
    return policy


def sample_unlabeled_pairs(*, eligible_drugs, eligible_diseases, observed_pairs,
                           forbidden_pairs, count, seed):
    return UnlabeledPolicy(tuple(eligible_drugs), tuple(eligible_diseases),
                           frozenset(observed_pairs), frozenset(forbidden_pairs)).sample(count, seed=seed)


def assert_no_forbidden_facts(frame, forbidden_pairs):
    """Mask by typed endpoint identity, covering every DDA relation and alias."""
    dda = frame.filter(((pl.col('source_node_type') == 'DRG') & (pl.col('target_node_type') == 'DIS')) |
                       ((pl.col('source_node_type') == 'DIS') & (pl.col('target_node_type') == 'DRG')))
    if dda.height == 0 or not forbidden_pairs:
        return
    pairs = dda.select(
        pl.when(pl.col('source_node_type') == 'DRG').then(pl.col('source_node_id'))
        .otherwise(pl.col('target_node_id')).alias('drug_id'),
        pl.when(pl.col('source_node_type') == 'DIS').then(pl.col('source_node_id'))
        .otherwise(pl.col('target_node_id')).alias('disease_id'))
    blocked = pl.DataFrame(sorted(forbidden_pairs), schema={'drug_id': pl.String, 'disease_id': pl.String}, orient='row')
    if pairs.join(blocked, on=['drug_id', 'disease_id'], how='semi').height:
        raise ValueError('A forbidden drug-disease endpoint pair survives in the graph')


def map_pairs(rows, nodes, *, channel):
    if channel not in ('support', 'unsuitability'):
        raise ValueError('Unverified supervision channel')
    rows = list(rows)
    basic = pl.DataFrame([dict(drug_id=r['drug_id'], disease_id=r['disease_id'], role=r.get('role', 'P'))
                          for r in rows], schema={'drug_id': pl.String, 'disease_id': pl.String, 'role': pl.String})
    if basic.select('drug_id', 'disease_id').n_unique() != basic.height or set(basic['role']) - {'P', 'U'}:
        raise ValueError('Pair identities must be unique with original P/U roles')
    drugs = nodes.filter(pl.col('node_type') == 'DRG').select(pl.col('node_id').alias('drug_id'),
        pl.col('global_node_id').alias('drug_global_id'), pl.col('type_local_id').alias('drug_type_local_id'))
    diseases = nodes.filter(pl.col('node_type') == 'DIS').select(pl.col('node_id').alias('disease_id'),
        pl.col('global_node_id').alias('disease_global_id'), pl.col('type_local_id').alias('disease_type_local_id'))
    result = (basic.join(drugs, on='drug_id', how='left', validate='m:1')
              .join(diseases, on='disease_id', how='left', validate='m:1')
              .with_columns(pl.lit(channel).alias('channel')).select(PAIR_COLUMNS))
    if result.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        raise ValueError('A panel endpoint is missing from the canonical graph')
    return result.sort('drug_id', 'disease_id')


def export_graph_arrays(nodes, fact_files, edge_catalog, destination, *, forbidden_pairs):
    """Export each certified fact once forward and once as a computed inverse."""
    nodes = normalize_nodes(nodes)
    destination = Path(destination)
    edge_dir = destination / 'edges'
    if edge_dir.exists() and any(edge_dir.iterdir()):
        raise FileExistsError('Refusing to replace existing graph edge arrays')
    catalog = sorted(edge_catalog, key=lambda r: r['edge_type_id'])
    if [r['edge_type_id'] for r in catalog] != list(range(len(catalog))):
        raise ValueError('Source edge catalog IDs must be unique and contiguous')
    lookup = {(r['source_node_type'], r['relation'], r['target_node_type']): r for r in catalog}
    if len(lookup) != len(catalog):
        raise ValueError('Source typed relation catalog must be unique')
    counts = Counter()
    sources = [Path(path) for path in fact_files]
    source_hashes = {str(path.resolve()): _sha(path) for path in sources}
    for path in sources:
        groups = pl.scan_parquet(path).group_by('source_node_type', 'relation', 'target_node_type').len().collect()
        for src, relation, dst, count in groups.iter_rows():
            key = (src, relation, dst)
            if key not in lookup:
                raise ValueError('Canonical fact relation is absent from the preserved source catalog')
            counts[lookup[key]['edge_type_id']] += count
    edge_dir.mkdir(parents=True, exist_ok=True)
    records, arrays, offsets = [], {}, Counter()
    for row in catalog:
        source_id, count = row['edge_type_id'], int(counts[row['edge_type_id']])
        for direction_index, direction in enumerate(('forward', 'reverse')):
            relation_id = source_id * 2 + direction_index
            src, dst = row['source_node_type'], row['target_node_type']
            if direction_index:
                src, dst = dst, src
            edge_path = f'edges/relation_{relation_id:04d}.npy'
            fact_path = f'edges/relation_{relation_id:04d}_fact_ids.npy'
            records.append(dict(relation_id=relation_id, source_catalog_edge_type_id=source_id,
                direction=direction, source_node_type=src, target_node_type=dst,
                source_type=HUMAN_TYPES[src], target_type=HUMAN_TYPES[dst], relation=row['relation'],
                relation_token=f'r{source_id:04d}_{direction}', edge_count=count,
                edge_index_path=edge_path, fact_ids_path=fact_path,
                is_computed_inverse=bool(direction_index)))
            if count:
                arrays[relation_id] = (
                    np.lib.format.open_memmap(destination / edge_path, mode='w+', dtype=np.int64, shape=(2, count)),
                    np.lib.format.open_memmap(destination / fact_path, mode='w+', dtype=np.uint64, shape=(count,)))
            else:
                np.save(destination / edge_path, np.empty((2, 0), dtype=np.int64), allow_pickle=False)
                np.save(destination / fact_path, np.empty(0, dtype=np.uint64), allow_pickle=False)
    local_ids = nodes['type_local_id'].to_numpy()
    canonical_ids = nodes['node_id'].to_numpy()
    node_types = nodes['node_type'].to_numpy()
    written = 0
    for path in sources:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=FACT_COLUMNS, use_threads=False):
            frame = pl.from_arrow(batch)
            if frame.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
                raise ValueError('Null canonical fact identity')
            assert_no_forbidden_facts(frame, forbidden_pairs)
            for side in ('source', 'target'):
                ids = frame[side + '_global_id'].to_numpy().astype(np.int64, copy=False)
                if len(ids) and (ids.min() < 0 or ids.max() >= nodes.height):
                    raise ValueError('Graph endpoint outside canonical identity domain')
                if not np.array_equal(canonical_ids[ids], frame[side + '_node_id'].to_numpy()) or not np.array_equal(
                        node_types[ids], frame[side + '_node_type'].to_numpy()):
                    raise ValueError('Canonical endpoint identity disagrees with its global index')
            for key, group in frame.partition_by(['source_node_type', 'relation', 'target_node_type'], as_dict=True).items():
                source_id = lookup[key]['edge_type_id']
                src = local_ids[group['source_global_id'].to_numpy()]
                dst = local_ids[group['target_global_id'].to_numpy()]
                fact_ids = group['canonical_fact_id'].to_numpy()
                for direction in (0, 1):
                    relation_id = source_id * 2 + direction
                    start, stop = offsets[relation_id], offsets[relation_id] + group.height
                    edge, facts = arrays[relation_id]
                    edge[:, start:stop] = np.stack((src, dst) if direction == 0 else (dst, src))
                    facts[start:stop] = fact_ids
                    offsets[relation_id] = stop
            written += frame.height
        print(f'Competition graph export: {path.name}; cumulative facts={written}', flush=True)
    for relation_id, (edge, facts) in arrays.items():
        expected = records[relation_id]['edge_count']
        if offsets[relation_id] != expected:
            raise ValueError('Typed traversal count differs from the source facts')
        edge.flush()
        facts.flush()
    arrays.clear()
    if written != sum(counts.values()) or any(_sha(path) != expected for path, expected in source_hashes.items()):
        raise ValueError('Canonical source facts changed during export')
    return pl.DataFrame(records), dict(canonical_facts=written, traversal_edges=2 * written,
        typed_traversal_relations=len(records), source_catalog_relations=len(catalog),
        inverse_edges_are_new_biological_facts=False, inverse_fact_identity_preserved=True,
        construction='Every canonical source fact has one forward and one computed inverse traversal, including directed source facts.',
        source_fact_file_sha256=source_hashes)


def _rolefree_frame(pairs):
    return pl.DataFrame(sorted(set(pairs)), schema={'drug_id': pl.String, 'disease_id': pl.String}, orient='row')


def _verify_payload(dataset, manifest):
    required = {'nodes.parquet', 'relations.parquet', 'eligible_ids.json', 'observed_pairs.parquet',
                'rolefree_forbidden.parquet'} | {f'pairs/{name}.parquet' for name in PANELS}
    if not required <= set(manifest['files']):
        raise ValueError('Required payload file is missing from the export manifest')
    for relative, record in manifest['files'].items():
        path = _relative_file(dataset, relative)
        if path.stat().st_size != record['bytes'] or _sha(path) != record['sha256']:
            raise ValueError('Export artifact SHA-256/size mismatch: ' + relative)
    nodes = normalize_nodes(pl.read_parquet(dataset / 'nodes.parquet'))
    if nodes.height != manifest['nodes']:
        raise ValueError('Export node count mismatch')
    node_counts = {row['node_type']: row['len'] for row in nodes.group_by('node_type').len().to_dicts()}
    if node_counts != manifest['node_type_counts']:
        raise ValueError('Export node type count metadata mismatch')
    by_type = {kind: nodes.filter(pl.col('node_type') == kind).sort('type_local_id')['node_id'].to_numpy()
               for kind in nodes['node_type'].unique()}
    policy = _read_policy(dataset, nodes)
    policy_counts = dict(eligible_drugs=len(policy.eligible_drugs), eligible_diseases=len(policy.eligible_diseases),
                         observed_pairs=len(policy.observed_pairs), rolefree_forbidden_pairs=len(policy.forbidden_pairs))
    if any(manifest[name] != count for name, count in policy_counts.items()):
        raise ValueError('Export U policy count metadata mismatch')
    forbidden = policy.forbidden_pairs
    relations = pl.read_parquet(dataset / 'relations.parquet').sort('relation_id')
    if relations['relation_token'].n_unique() != relations.height:
        raise ValueError('Duplicate typed relation token')
    if relations['relation_id'].to_list() != list(range(relations.height)) or relations.height % 2:
        raise ValueError('Typed relation IDs must be contiguous forward/inverse pairs')
    referenced = set(relations['edge_index_path']) | set(relations['fact_ids_path'])
    if not referenced <= set(manifest['files']):
        raise ValueError('Typed relation payload file is missing from the export manifest')
    if (relations.height != manifest['graph']['typed_traversal_relations'] or
            relations.height // 2 != manifest['graph']['source_catalog_relations']):
        raise ValueError('Typed relation count metadata mismatch')
    total = 0
    forward_arrays = {}
    forward_fact_ids = []
    for row in relations.iter_rows(named=True):
        source_id, direction_index = divmod(row['relation_id'], 2)
        expected_direction = 'reverse' if direction_index else 'forward'
        if (row['source_catalog_edge_type_id'] != source_id or row['direction'] != expected_direction or
                row['is_computed_inverse'] is not bool(direction_index) or
                row['relation_token'] != f'r{source_id:04d}_{expected_direction}' or
                row['source_type'] != HUMAN_TYPES.get(row['source_node_type']) or
                row['target_type'] != HUMAN_TYPES.get(row['target_node_type'])):
            raise ValueError('Typed relation metadata disagrees with its source catalog or inverse direction')
        edges = np.load(_relative_file(dataset, row['edge_index_path']), mmap_mode='r', allow_pickle=False)
        fact_ids = np.load(_relative_file(dataset, row['fact_ids_path']), mmap_mode='r', allow_pickle=False)
        count = row['edge_count']
        if edges.dtype != np.int64 or edges.shape != (2, count) or fact_ids.dtype != np.uint64 or fact_ids.shape != (count,):
            raise ValueError('Typed COO or fact identity array shape/dtype mismatch')
        src_ids, dst_ids = by_type[row['source_node_type']], by_type[row['target_node_type']]
        for offset in range(0, count, 65536):
            block = edges[:, offset:offset + 65536]
            if (block < 0).any() or (block[0] >= len(src_ids)).any() or (block[1] >= len(dst_ids)).any():
                raise ValueError('Typed COO endpoint outside its type-local domain')
            if {row['source_node_type'], row['target_node_type']} == {'DRG', 'DIS'}:
                frame = pl.DataFrame({'source_node_id': src_ids[block[0]], 'target_node_id': dst_ids[block[1]]})
                frame = frame.with_columns(pl.lit(row['source_node_type']).alias('source_node_type'),
                                           pl.lit(row['target_node_type']).alias('target_node_type'))
                assert_no_forbidden_facts(frame, forbidden)
        if row['direction'] == 'forward':
            forward_arrays[source_id] = (edges, fact_ids, row)
            forward_fact_ids.append(fact_ids)
        elif row['direction'] == 'reverse':
            if source_id not in forward_arrays:
                raise ValueError('Computed inverse lacks its source relation')
            forward, source_facts, source_row = forward_arrays.pop(source_id)
            if (row['relation'] != source_row['relation'] or
                    row['source_node_type'] != source_row['target_node_type'] or
                    row['target_node_type'] != source_row['source_node_type']):
                raise ValueError('Computed inverse relation metadata differs from its source relation')
            if edges.shape != forward.shape:
                raise ValueError('Computed inverse edge count mismatch')
            for offset in range(0, count, 65536):
                end = offset + 65536
                if not np.array_equal(edges[:, offset:end], forward[::-1, offset:end]) or not np.array_equal(
                        fact_ids[offset:end], source_facts[offset:end]):
                    raise ValueError('Computed inverse changed canonical fact identity or endpoints')
        else:
            raise ValueError('Unknown traversal direction')
        total += count
    if forward_arrays or total != manifest['graph']['traversal_edges']:
        raise ValueError('Incomplete forward/reverse graph export')
    fact_count = sum(len(values) for values in forward_fact_ids)
    if fact_count != manifest['graph']['canonical_facts'] or total != 2 * fact_count:
        raise ValueError('Canonical fact count metadata differs from forward/inverse arrays')
    # One uint64 vector is about 118 MB for the full pilot; no Python fact objects.
    all_fact_ids = np.concatenate(forward_fact_ids) if forward_fact_ids else np.empty(0, dtype=np.uint64)
    all_fact_ids.sort()
    if np.any(all_fact_ids[1:] == all_fact_ids[:-1]):
        raise ValueError('Duplicate canonical fact identity in forward graph traversals')
    del all_fact_ids, forward_fact_ids
    for name in PANELS:
        saved = pl.read_parquet(dataset / f'pairs/{name}.parquet')
        channel = 'unsuitability' if name in ('train_contra', 'auxiliary_validation', 'auxiliary_probe') else 'support'
        rebuilt = map_pairs(saved.to_dicts(), nodes, channel=channel)
        if not saved.equals(rebuilt) or saved.height != manifest['panels'][name]['rows']:
            raise ValueError('Frozen panel mapping, role, order or count mismatch')
        record = manifest['panels'][name]
        if (dict(Counter(saved['role'].to_list())) != record['roles'] or
                _digest(saved.select('drug_id', 'disease_id', 'role').rows()) != record['membership_digest']):
            raise ValueError('Frozen panel role or membership digest metadata mismatch')
        pairs = set(saved.select('drug_id', 'disease_id').iter_rows())
        positives = set(saved.filter(pl.col('role') == 'P').select('drug_id', 'disease_id').iter_rows())
        unlabeled = pairs - positives
        if not positives <= policy.observed_pairs:
            raise ValueError('Positive panel member is absent from observed clinical pairs')
        if unlabeled & policy.observed_pairs:
            raise ValueError('An observed clinical pair is labeled U in a frozen panel')
        eligible_pairs = pairs if name in ('train_support', 'train_contra') else unlabeled
        if any(drug not in policy._drugs or disease not in policy._diseases for drug, disease in eligible_pairs):
            raise ValueError('Training or U panel endpoint outside the eligible domain')
        if name in ('train_support', 'train_contra'):
            if set(saved['role']) - {'P'} or pairs & forbidden:
                raise ValueError('Forbidden or nonpositive training pair')
        elif not pairs <= forbidden:
            raise ValueError('Heldout panel member absent from role-free exclusions')
    return manifest


def verify_export(destination=DEFAULT_DATASET):
    """Check actual files, every edge endpoint and all computed inverse IDs."""
    dataset = Path(destination).resolve()
    manifest = _read_bound(dataset / 'manifest.json')
    if manifest.get('schema') != 'EVIHGT_COMPETITION_PILOT_DATA_V1' or manifest.get('status') != 'READY':
        raise ValueError('Shared competition export is incomplete')
    return _verify_payload(dataset, manifest)


def export_dataset(*, nodes, fact_files, edge_catalog, panels, eligible_drugs, eligible_diseases,
                   observed_pairs, forbidden_pairs, bindings, destination, expected_fact_count=None):
    """Write a complete immutable export from an already certified source contract."""
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing shared dataset')
    if set(panels) != set(PANELS):
        raise ValueError('All six frozen training/development panels are required')
    started = time.monotonic()
    policy = UnlabeledPolicy(tuple(eligible_drugs), tuple(eligible_diseases),
                             frozenset(observed_pairs), frozenset(forbidden_pairs))
    nodes = normalize_nodes(nodes)
    destination.mkdir(parents=True, exist_ok=True)
    nodes.write_parquet(destination / 'nodes.parquet')
    pairs_dir = destination / 'pairs'
    pairs_dir.mkdir()
    panel_records = {}
    for name in PANELS:
        channel = 'unsuitability' if name in ('train_contra', 'auxiliary_validation', 'auxiliary_probe') else 'support'
        frame = map_pairs(panels[name], nodes, channel=channel)
        frame.write_parquet(pairs_dir / f'{name}.parquet')
        panel_records[name] = dict(rows=frame.height, roles=dict(Counter(frame['role'].to_list())),
            membership_digest=_digest(frame.select('drug_id', 'disease_id', 'role').rows()))
    _rolefree_frame(policy.observed_pairs).write_parquet(destination / 'observed_pairs.parquet')
    _rolefree_frame(policy.forbidden_pairs).write_parquet(destination / 'rolefree_forbidden.parquet')
    _write_json(destination / 'eligible_ids.json', dict(eligible_drugs=list(policy.eligible_drugs),
        eligible_diseases=list(policy.eligible_diseases), sampling='GLOBAL_UNIFORM_WITH_REPLACEMENT',
        all_observed_clinical_pairs_excluded=True, heldout_U_excluded=True,
        U_is_confirmed_negative=False))
    relations, graph = export_graph_arrays(nodes, fact_files, edge_catalog, destination,
                                            forbidden_pairs=policy.forbidden_pairs)
    if expected_fact_count is not None and graph['canonical_facts'] != expected_fact_count:
        raise ValueError('Exported canonical fact count differs from the certified corpus')
    relations.write_parquet(destination / 'relations.parquet')
    files = {path.relative_to(destination).as_posix(): dict(bytes=path.stat().st_size, sha256=_sha(path))
             for path in sorted(destination.rglob('*')) if path.is_file()}
    manifest = _bound(dict(schema='EVIHGT_COMPETITION_PILOT_DATA_V1', status='READY',
        dataset=str(destination), bindings=dict(bindings), nodes=nodes.height,
        node_type_counts={row['node_type']: row['len'] for row in nodes.group_by('node_type').len().to_dicts()},
        human_type_mapping=HUMAN_TYPES, graph=graph, panels=panel_records, files=files,
        rolefree_forbidden_pairs=len(policy.forbidden_pairs), observed_pairs=len(policy.observed_pairs),
        eligible_drugs=len(policy.eligible_drugs), eligible_diseases=len(policy.eligible_diseases),
        U_is_confirmed_negative=False, final_roles_read=False, original_sources_modified=False,
        training_started=False, optimizer_steps=0,
        query_target_mask_required='Every scored/optimized DDA pair must be removed by canonical endpoint identity in both directions before graph encoding, including all semantic aliases.',
        information_source='CERTIFIED_PILOT_CANONICAL_CORPUS',
        computation_graph_note='Computed inverse traversals are model computation, not new facts; different from the source CSR traversal orientation policy.',
        exporter_code_sha256=_sha(__file__), elapsed_seconds=time.monotonic() - started))
    _verify_payload(destination, manifest)
    _write_json(destination / 'manifest.json', manifest)
    return manifest


def export_pilot_data(destination=DEFAULT_DATASET, report_path=DEFAULT_REPORT):
    """Validate frozen PilotDataset/corpus bindings and export without training."""
    from evi_hgt.phase85_v2.holdout import HoldoutReadGuard
    from evi_hgt.phase85_part2b.production import PilotDataset, build_corpus, PHASE3
    destination, report_path = Path(destination).resolve(), Path(report_path).resolve()
    processed = (ROOT / 'data/processed').resolve()
    outputs = (ROOT / 'outputs').resolve()
    if processed not in destination.parents or outputs not in report_path.parents:
        raise ValueError('Competition datasets belong under data/processed and reports under outputs')
    with ExitStack() as stack:
        for folder in (ROOT / 'data/processed/phase85_part2/sealed_final', PHASE3 / 'supervision', PHASE3 / 'splits'):
            stack.enter_context(HoldoutReadGuard(folder))
        data = PilotDataset()
        corpus = build_corpus(data)
        bindings = dict(corpus['bindings'], corpus_digest=corpus['digest'],
                        pilot_cell_digest=data.cell['digest'], fold=data.cell['fold'],
                        repeat_key=data.cell['repeat_key'], training_u_manifest_digest=data.u['digest'])
        if (destination / 'manifest.json').is_file():
            manifest = verify_export(destination)
            if manifest['bindings'] != bindings:
                raise ValueError('Existing competition export has different frozen source bindings')
        else:
            panels = dict(train_support=[r for r in data.cell['training'] if r['role'] == 'P'],
                train_contra=data.cell['auxiliary_training'], validation=data.cell['validation'],
                train_probe=data.cell['train_probe'], auxiliary_validation=data.cell['auxiliary_validation'],
                auxiliary_probe=data.cell['auxiliary_probe'])
            manifest = export_dataset(nodes=pl.read_parquet(PHASE3 / 'nodes/node_index.parquet'),
                fact_files=[Path(row['path']) for row in corpus['fact_files']],
                edge_catalog=json.loads((PHASE3 / 'graph/csr/edge_type_catalog.json').read_text()),
                panels=panels, eligible_drugs=data.u['eligible_drugs'], eligible_diseases=data.u['eligible_diseases'],
                observed_pairs=data.pairs.observed, forbidden_pairs=data.blocked,
                bindings=bindings, destination=destination, expected_fact_count=corpus['fact_count'])
        if manifest['graph']['canonical_facts'] != corpus['fact_count']:
            raise ValueError('Exported canonical fact count differs from the certified corpus')
        report = _bound(dict(status='PASS', dataset=str(destination), export_digest=manifest['digest'],
            source_bindings=bindings, nodes=manifest['nodes'], graph=manifest['graph'], panels=manifest['panels'],
            all_files_sha256_checked=True, all_typed_endpoints_checked=True,
            inverse_fact_identity_checked=True, forbidden_forward_reverse_equivalent_DDA_absent=True,
            heldout_U_excluded_from_sampler=True, observed_clinical_pairs_excluded_from_U=True,
            final_roles_read=False, old_supervision_or_splits_read=False,
            source_corpus_original_traversals=corpus['traversal_count'],
            source_corpus_shared_across_models=True, U_is_confirmed_negative=False,
            training_started=False, optimizer_steps=0))
        if report_path.exists():
            previous = _read_bound(report_path)
            if previous != report:
                raise FileExistsError('Existing data audit differs; not overwriting it')
        else:
            _write_json(report_path, report)
        return manifest
