"""Shared raw preparation and fold-specific portable corpora for the experiments."""
from pathlib import Path
import json

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from evi_hgt.data.common import digest, sha256
from evi_hgt.preprocessing.pipeline import (StageCache, PreparationLog, _configuration, assemble_corpus,
    preparation_lock, preparation_packages, reuse_corpus, write_json)
from evi_hgt.preprocessing.clean_graph_source import mask_endpoint_pairs
from evi_hgt.settings import ROOT
from .memberships import (load_recipe, reconstruct_pools, save_pools, load_pools,
                          partition_pools, build_cell_splits, frame)
from .sampling import build_cell_sampling


def validate_destinations(sources, destinations):
    sources = [Path(p).resolve() for p in sources.values()]
    destinations = [Path(p).resolve() for p in destinations]
    if any(source.is_relative_to(destination) for source in sources for destination in destinations):
        raise ValueError('Experiment output destinations may not contain an original input file')


def mask_cell_graph(shared_graph, splits_root, destination, *, progress=None):
    """Apply the extra cell mask to the already certified common graph."""
    destination, splits_root = Path(destination), Path(splits_root)
    if destination.exists():
        raise FileExistsError('Choose a new fold graph destination')
    blocked_frame = pl.read_parquet(splits_root / 'blocked_pairs.parquet')
    blocked = set(blocked_frame.select('drug_id', 'disease_id').iter_rows())
    targets = pl.read_parquet(shared_graph['target_map'])
    newly_forbidden = targets.join(blocked_frame, on=['drug_id', 'disease_id'], how='semi')['canonical_fact_id']
    inherited = pl.read_parquet(shared_graph['forbidden_facts'])['canonical_fact_id']
    forbidden = pl.concat([newly_forbidden.cast(pl.UInt64), inherited.cast(pl.UInt64)]).unique().sort()
    destination.mkdir(parents=True)
    def write(name, value):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        value.write_parquet(path)
        return str(path.resolve())
    forbidden_path = write('forbidden_fact_ids.parquet', pl.DataFrame({'canonical_fact_id': forbidden}))
    evidence_by_component = {r['component']: r for r in shared_graph['evidence_files']}
    facts, evidence, admitted_parts = [], [], []
    for i, entry in enumerate(shared_graph['fact_files']):
        component, fp = entry['component'], Path(entry['path'])
        ep = Path(evidence_by_component[component]['path'])
        identity = pl.read_parquet(fp, columns=['canonical_fact_id', 'source_node_id', 'target_node_id',
                                                'source_node_type', 'target_node_type'])
        safe = mask_endpoint_pairs(identity, blocked).filter(~pl.col('canonical_fact_id').is_in(forbidden.implode()))
        ids = safe.select('canonical_fact_id')
        if safe.height != identity.height:
            fp = Path(write('facts/' + component + '.parquet',
                            pl.scan_parquet(fp).join(ids.lazy(), on='canonical_fact_id', how='semi').collect()))
            ep = Path(write('evidence/' + component + '.parquet',
                            pl.scan_parquet(ep).join(ids.lazy(), on='canonical_fact_id', how='semi').collect()))
        facts.append(dict(component=component, rows=pq.ParquetFile(fp).metadata.num_rows, path=str(fp.resolve())))
        evidence.append(dict(component=component, rows=pq.ParquetFile(ep).metadata.num_rows, path=str(ep.resolve())))
        admitted_parts.append(ids.to_series().to_numpy())
        if progress:
            progress('按当前实验划分屏蔽事实', i+1, len(shared_graph['fact_files']), 0)
    admitted = np.concatenate(admitted_parts)
    if len(np.unique(admitted)) != len(admitted) or np.isin(admitted, forbidden.to_numpy()).any():
        raise ValueError('Duplicate or forbidden fact remains in fold graph')
    original = {k: np.load(v, mmap_mode='r', allow_pickle=False) for k, v in shared_graph['csr'].items()}
    keep = np.isin(original['fact_ids'], admitted)
    prefix = np.empty(len(keep)+1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(keep, dtype=np.int64, out=prefix[1:])
    arrays = {'indptr': prefix[original['indptr']]}
    arrays.update({k: v[keep] for k, v in original.items() if k != 'indptr'})
    csr = {}
    for key, values in arrays.items():
        path = destination / 'csr' / (key + '.npy')
        path.parent.mkdir(exist_ok=True)
        np.save(path, values, allow_pickle=False)
        csr[key] = str(path.resolve())
    target_path = write('target_map.parquet', targets.filter(pl.col('canonical_fact_id').is_in(pl.Series(admitted).implode())))
    result = dict(schema='EVIHGT_EXPERIMENT_MASKED_GRAPH_V1', fact_files=facts, evidence_files=evidence,
        csr=csr, target_map=target_path, forbidden_facts=forbidden_path, nodes=shared_graph['nodes'],
        edge_catalog=shared_graph['edge_catalog'], target_reverse_equivalent_endpoint_mask_applied=True)
    write_json(destination / 'manifest.json', result)
    return result


def _source_code():
    code = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / 'src/evi_hgt').rglob('*'))
            if p.suffix in ('.py', '.json')}
    for name in ('preparation.py', 'memberships.py', 'sampling.py', 'recipes.json'):
        path = Path(__file__).with_name(name)
        code[path.relative_to(ROOT).as_posix()] = sha256(path)
    for path in sorted((ROOT / 'configs').rglob('*.json')):
        code[path.relative_to(ROOT).as_posix()] = sha256(path)
    return code


def _all_bound_files(cache):
    result = {}
    for path in sorted(cache.root.glob('*.receipt.json')):
        receipt = json.loads(path.read_text('utf-8'))
        for name, expected in receipt['files'].items():
            result[str((cache.root / name).resolve())] = expected
    return result


def prepare_shared(raw_config=None, *, stage_root=None, audit_dir, offline=False):
    """Prepare common verified stages once, keeping fold masks separate."""
    config = _configuration(raw_config)
    stage_root = Path(stage_root or ROOT / 'data/interim/experiments').resolve()
    audit_dir = Path(audit_dir).resolve()
    sources = {key: config[key] for key in ('nodes', 'edges', 'drugbank_xml', 'drugbank_xsd')}
    validate_destinations(sources, [stage_root, audit_dir, config['open_targets_cache'], config['model_cache']])
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError('缺少原始文件：' + str(source) + '。请按 README 的四文件目录放置。')
    packages = preparation_packages()
    from evi_hgt.preprocessing.canonical import build_canonical
    from evi_hgt.preprocessing.clean_graph import build_clean_graph
    from evi_hgt.preprocessing.features import build_features
    from evi_hgt.preprocessing.resources import ensure_open_targets
    with PreparationLog(audit_dir) as log:
        log.progress('核验实验公共输入和代码')
        hashes = {k: sha256(v) for k, v in sources.items()}
        identity = dict(schema='EVIHGT_SHARED_EXPERIMENT_PREPARATION_V1', sources=hashes,
            code=_source_code(), packages=packages, feature_device=config['feature_device'],
            batch_size=config['feature_batch_size'])
        source_identity = dict(recipe='REGISTERED_EXPERIMENTS_V1', build_identity=digest(identity), raw_sha256=hashes)
        cache = StageCache(stage_root / digest(identity)[:20], identity, log.progress)
        with preparation_lock(cache.root):
            canonical = Path(cache.run('canonical', lambda p: build_canonical(config['nodes'], config['edges'],
                config['drugbank_xml'], config['drugbank_xsd'], p, audit_dir=audit_dir, progress=log.progress)))
            pool_descriptor = cache.run('pools', lambda p: save_pools(reconstruct_pools(canonical, progress=log.progress), p))
            pools = load_pools(pool_descriptor)
            def base_split_inputs(path):
                path.mkdir(parents=True)
                frame(pools['frozen']).write_parquet(path / 'blocked_pairs.parquet')
                frame(pools['legal_source_rows']).write_parquet(path / 'legal_source_rows.parquet')
                return str(path)
            base_splits = Path(cache.run('base_graph_inputs', base_split_inputs))
            resources, policy = ensure_open_targets(config['open_targets_cache'], offline=offline, progress=log.progress)
            graph = cache.run('common_graph', lambda p: build_clean_graph(canonical, base_splits, resources, policy,
                p, audit_dir=audit_dir, progress=log.progress))
            features = cache.run('features', lambda p: build_features(canonical, config['nodes'], config['drugbank_xml'],
                p, model_cache=config['model_cache'], device=config['feature_device'], batch_size=config['feature_batch_size'],
                offline=offline, progress=log.progress, drugbank_xsd=config['drugbank_xsd']), resumable=True)
            if hashes != {k: sha256(v) for k, v in sources.items()}:
                raise ValueError('An original source changed during experiment preparation')
        clean_dg = {name: next(r['path'] for r in graph[key] if r['component'] == 'disease_gene')
                    for name, key in (('facts', 'fact_files'), ('evidence', 'evidence_files'))}
        result = dict(canonical_root=str(canonical), nodes=str(config['nodes']), features=features,
            graph=graph, clean_dg=clean_dg, pools=pool_descriptor, source_identity=source_identity,
            cache_root=str(cache.root), model_cache=str(config['model_cache']),
            native_cache=str(ROOT / 'data/cache/experiments/native'), feature_device=config['feature_device'],
            feature_batch_size=config['feature_batch_size'], offline=offline,
            file_hashes={**_all_bound_files(cache), **{str(v): hashes[k] for k, v in sources.items()}})
        write_json(audit_dir / 'completion.json', dict(status='COMPLETE', source_identity=source_identity,
                                                     shared=str(cache.root), raw_files_modified=False))
        return result


def prepare_cell(shared, cell, *, destination_root=None, audit_dir, progress=None, protocol=None):
    """Build and authenticate one exact published fold from common raw stages."""
    expected = load_recipe(cell, protocol=protocol)
    destination_root = Path(destination_root or ROOT / 'data/processed/experiments').resolve()
    source_identity = dict(shared['source_identity'], cell=cell, recipe_sha256=digest(expected))
    cell_path = Path(cell)
    if protocol is not None:
        source_identity['protocol'] = protocol
        cell_path = Path(protocol) / cell
    destination = destination_root / shared['source_identity']['build_identity'][:20] / cell_path
    validate_destinations({'nodes': shared['nodes']}, [destination])
    manifest = destination / 'manifest.json'
    if manifest.is_file():
        reuse_corpus(manifest, source_identity, progress=progress)
        return manifest
    identity = dict(schema='EVIHGT_EXPERIMENT_CELL_PREPARATION_V1', source_identity=source_identity)
    cache = StageCache(Path(shared['cache_root']) / 'cells' / cell_path, identity, progress)
    with preparation_lock(cache.root):
        pools = load_pools(shared['pools'])
        split = cache.run('splits', lambda p: build_cell_splits(pools, p, cell=cell, audit_dir=audit_dir, progress=progress))
        split_root = Path(split['files']['blocked_pairs']).parent
        graph = cache.run('graph', lambda p: mask_cell_graph(shared['graph'], split_root, p, progress=progress))
        schedule = cache.run('sampling', lambda p: build_cell_sampling(split_root, p, cell=cell,
                                     audit_dir=audit_dir, progress=progress, protocol=protocol))
        assemble_corpus(graph, split, shared['features'], schedule, destination,
                        source_identity=source_identity, progress=progress)
    return manifest
