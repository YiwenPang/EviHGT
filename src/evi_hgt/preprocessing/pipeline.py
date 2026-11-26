"""Raw database preparation, content-bound stages and portable publication."""
from contextlib import contextmanager
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from uuid import uuid4

import yaml

from ..data.common import digest, sha256
from ..data.corpus import SCHEMA, load_corpus
from ..data.prepare import destination_path
from ..settings import ROOT


def write_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False), encoding='utf8')
    os.replace(temporary, path)


class StageCache:
    """Reuse completed stages only after every published file passes its hash."""
    def __init__(self, root, identity, progress=None):
        self.root, self.identity, self.progress = Path(root).resolve(), identity, progress
        self.root.mkdir(parents=True, exist_ok=True)
        binding = self.root / 'identity.json'
        if binding.exists():
            if json.loads(binding.read_text('utf8')) != identity:
                raise ValueError('Raw preparation identity changed; use a new stage_root')
        else:
            if any(self.root.iterdir()):
                raise ValueError('Unbound stage directory; use a new stage_root')
            write_json(binding, identity)

    def run(self, name, build, *, resumable=False):
        receipt = self.root / (name + '.receipt.json')
        if receipt.is_file():
            saved = json.loads(receipt.read_text('utf8'))
            for relative, expected in saved['files'].items():
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root) or not path.is_file() or sha256(path) != expected:
                    raise ValueError('Completed stage output changed: ' + str(path))
            if self.progress:
                self.progress('复用已核验阶段：' + name)
            return saved['result']
        pending = self.root / (name + '.pending.json')
        if resumable and pending.is_file():
            attempt = self.root / json.loads(pending.read_text('utf8'))['attempt']
            if attempt.parent != self.root or not attempt.name.startswith(name + '_'):
                raise ValueError('Invalid pending stage path')
        else:
            attempt = self.root / (name + '_' + uuid4().hex)
            write_json(pending, {'attempt': attempt.name})
        result = build(attempt)
        result = json.loads(json.dumps(result, default=str))
        inventory = {p.relative_to(self.root).as_posix(): sha256(p)
                     for p in sorted(attempt.rglob('*')) if p.is_file()}
        if not inventory:
            raise ValueError('Preparation stage produced no verified files: ' + name)
        write_json(receipt, {'result': result, 'files': inventory})
        pending.unlink(missing_ok=True)
        return result


def assemble_corpus(graph, splits, features, schedule, output, *, source_identity, progress=None):
    """Copy stage outputs into a relocatable corpus and validate before publishing."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Corpus output already exists: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + '.attempt_' + uuid4().hex)
    staging.mkdir()
    files = {}
    def copy(name, source):
        destination_path(name)
        source = Path(source).resolve()
        target = staging / name
        if source.is_relative_to(staging) or source.is_relative_to(output):
            raise ValueError('Corpus output may not contain its input')
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(source)
        shutil.copyfile(source, target)
        if sha256(target) != before:
            raise ValueError('Preparation output changed while copying: ' + str(source))
        files[name] = before
        if progress:
            progress('封装便携语料', len(files), 0, 0)
        return name
    m = dict(schema=SCHEMA, training_seed=splits['training_seed'], token_buckets=8192,
             source_cell=splits.get('cell', 'R0F1'), preparation=source_identity,
             semantics={key: True for key in ('canonical_entities_verified', 'canonical_facts_deduplicated',
                 'evidence_records_deduplicated', 'clinical_labels_verified', 'target_aliases_resolved',
                 'disease_gene_label_leakage_removed', 'split_mask_applied', 'features_label_independent')})
    m['semantics']['unknown_is_negative'] = False
    for key, name in (('nodes', 'nodes.parquet'), ('edge_catalog', 'edge_catalog.json'),
                      ('target_map', 'target_map.parquet'), ('forbidden_facts', 'forbidden_facts.parquet')):
        m[key] = copy(name, graph[key])
    for kind in ('fact', 'evidence'):
        m[kind + '_files'] = [dict(component=row['component'], rows=row['rows'],
            path=copy(f'{kind}/{i:02d}_{row["component"]}.parquet', row['path']))
            for i, row in enumerate(graph[kind + '_files'])]
    m['csr'] = {key: copy('csr/' + key + '.npy', graph['csr'][key])
                for key in ('indptr', 'indices', 'fact_ids')}
    for key in ('blocked_pairs', 'observed_pairs', 'training_pairs', 'eligibility'):
        m[key] = copy(key + ('.json' if key == 'eligibility' else '.parquet'), splits['files'][key])
    m['panels'] = {key: copy(key + '.parquet', splits['files'][key])
                   for key in ('validation', 'train_probe', 'auxiliary_validation') if key in splits['files']}
    m['schedule'] = dict(schedule)
    for key, name in (('groups', 'global_groups.parquet'), ('conditional_records', 'conditional_records.json')):
        m['schedule'][key] = copy(name, schedule[key])
    for kind in ('raw_features', 'pretrained'):
        m[kind] = {key: copy(kind + '/' + key + ('.json' if key == 'metadata' else '.npy'), path)
                   for key, path in features[kind].items()}
    m['files'] = files
    write_json(staging / 'manifest.json', m)
    corpus = load_corpus(staging / 'manifest.json', progress=progress)
    identity = corpus.identity
    del corpus
    gc.collect()
    os.rename(staging, output)
    return {'manifest': str(output / 'manifest.json'), 'identity': identity,
            'files': len(files), 'validation': 'PASSED', 'source_files_modified': False}


class PreparationLog:
    def __init__(self, directory):
        self.directory = directory
        self.last = 0.0
        self.label = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / 'events.jsonl').open('a', encoding='utf8')
        return self

    def __exit__(self, typ, exc, tb):
        if exc is not None:
            path = self.directory / 'errors.log'
            with path.open('a', encoding='utf8') as handle:
                traceback.print_exception(typ, exc, tb, file=handle)
            print(f'数据准备失败：{type(exc).__name__}: {exc}\n完整日志：{path}', file=sys.stderr, flush=True)
            try:
                exc.evihgt_reported = True
            except AttributeError:
                pass
        self.stream.close()

    def progress(self, label, current=0, total=0, byte_count=0):
        now = time.monotonic()
        if label != self.label or now - self.last >= 5 or (total and current == total):
            suffix = f' {current:,}/{total:,}' if total else ''
            print(f'[数据准备] {label}{suffix}', flush=True)
            self.stream.write(json.dumps(dict(label=label, current=current, total=total,
                bytes=byte_count, time=time.time()), ensure_ascii=True) + '\n')
            self.stream.flush()
            self.last, self.label = now, label


@contextmanager
def preparation_lock(root):
    path = root / '.prepare.lock'
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RuntimeError(f'Another preparation may be active. Check the PID in {path}; remove only this lock after confirming that process has stopped.') from None
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump({'pid': os.getpid()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def _configuration(path):
    path = ROOT / 'configs/raw.yaml' if path is None else Path(path).resolve()
    config = yaml.safe_load(path.read_text('utf8'))
    required = {'nodes', 'edges', 'drugbank_xml', 'drugbank_xsd', 'stage_root',
                'open_targets_cache', 'model_cache', 'cell', 'feature_device', 'feature_batch_size'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Raw configuration requires exactly: ' + ', '.join(sorted(required)))
    if config['cell'] != 'R0F1' or config['feature_device'] not in ('cpu', 'cuda'):
        raise ValueError('The published raw recipe requires R0F1 and feature_device cpu/cuda')
    if type(config['feature_batch_size']) is not int or config['feature_batch_size'] < 1:
        raise ValueError('feature_batch_size must be a positive integer')
    for key in required - {'cell', 'feature_device', 'feature_batch_size'}:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError('Expected a nonempty raw configuration path: ' + key)
        p = Path(config[key]).expanduser()
        config[key] = (p if p.is_absolute() else ROOT / p).resolve()
    return config


def reuse_corpus(manifest, expected, progress=None):
    recorded = json.loads(Path(manifest).read_text('utf8')).get('preparation')
    if recorded != expected:
        raise ValueError('Existing corpus has a different preparation identity. Use a new manifest directory and stage_root for the new inputs/code; use --mode validate to inspect an existing corpus without rebuilding.')
    corpus = load_corpus(manifest, progress=progress)
    result = {'manifest': str(manifest), 'identity': corpus.identity, 'validation': 'PASSED', 'reused': True}
    del corpus
    gc.collect()
    return result


def preparation_packages():
    names = ('numpy', 'polars', 'pyarrow', 'lxml', 'rdkit', 'torch', 'transformers', 'tokenizers')
    try:
        packages = {name: importlib.metadata.version(name) for name in names}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('Raw preparation dependencies are missing. Install requirements-preprocess.txt with this interpreter: '
                           + sys.executable) from exc
    for name, expected in (('rdkit', '2026.3.6'), ('transformers', '5.19.0')):
        if packages[name] != expected:
            raise RuntimeError(f'Raw feature preparation requires {name}=={expected}; install requirements-preprocess.txt with {sys.executable}')
    return packages


def prepare_raw(raw_config=None, *, manifest_path=None, output_dir=None, offline=False):
    """Prepare the original R0F1 corpus from immutable licensed local databases."""
    config = _configuration(raw_config)
    manifest = Path(manifest_path or ROOT / 'data/processed/corpus/manifest.json').resolve()
    if manifest.name != 'manifest.json':
        raise ValueError('Prepared corpus path must end with manifest.json')
    audit = Path(output_dir or ROOT / 'outputs').resolve() / 'preparation' / uuid4().hex
    with PreparationLog(audit) as log:
        sources = {key: config[key] for key in ('nodes', 'edges', 'drugbank_xml', 'drugbank_xsd')}
        for source in sources.values():
            if not source.is_file():
                raise FileNotFoundError(f'缺少原始文件：{source}。请按 docs/data.md 放置指定版本文件，或修改 --raw-config。')
        derived = config['stage_root']
        destinations = [derived, manifest.parent, audit, config['open_targets_cache'], config['model_cache']]
        if any(source.is_relative_to(dest) for source in sources.values() for dest in destinations):
            raise ValueError('Preparation destinations may not contain any original input file')
        if derived.is_relative_to(manifest.parent) or manifest.parent.is_relative_to(derived):
            raise ValueError('stage_root and final corpus directory must be separate')
        packages = preparation_packages()
        # Defer optional preprocessing imports so existing prepared workflows retain core dependencies.
        from .canonical import build_canonical
        from .clean_graph import build_clean_graph
        from .features import build_features
        from .resources import ensure_open_targets
        from .sampling import build_sampling
        from .splits import build_splits
        log.progress('核验原始输入及构建代码')
        source_hashes = {key: sha256(path) for key, path in sources.items()}
        code_root = Path(__file__).parent.parent
        code = {p.relative_to(code_root).as_posix(): sha256(p) for p in sorted(code_root.rglob('*'))
                if p.suffix in ('.py', '.json')}
        for p in sorted((ROOT / 'configs').rglob('*.json')):
            code['configs/' + p.relative_to(ROOT / 'configs').as_posix()] = sha256(p)
        identity = dict(schema='EVIHGT_RAW_PREPARATION_V1', sources=source_hashes, code=code, packages=packages,
                        recipe=config['cell'], feature_device=config['feature_device'], batch_size=config['feature_batch_size'])
        source_identity = {'recipe': config['cell'], 'build_identity': digest(identity), 'raw_sha256': source_hashes}
        if manifest.is_file():
            log.progress('核验已有语料的原始输入及构建身份')
            result = reuse_corpus(manifest, source_identity, progress=log.progress)
            write_json(audit / 'completion.json', result)
            return result
        cache = StageCache(derived, identity, log.progress)
        with preparation_lock(derived):
            canonical = Path(cache.run('canonical', lambda p: build_canonical(config['nodes'], config['edges'],
                config['drugbank_xml'], config['drugbank_xsd'], p, audit_dir=audit, progress=log.progress)))
            split_result = cache.run('splits', lambda p: build_splits(canonical, p, cell=config['cell'],
                audit_dir=audit, progress=log.progress))
            splits_root = Path(split_result['files']['blocked_pairs']).parent
            resources, policy = ensure_open_targets(config['open_targets_cache'], offline=offline, progress=log.progress)
            graph = cache.run('clean_graph', lambda p: build_clean_graph(canonical, splits_root, resources, policy,
                p, audit_dir=audit, progress=log.progress))
            features = cache.run('features', lambda p: build_features(canonical, config['nodes'], config['drugbank_xml'],
                p, model_cache=config['model_cache'], device=config['feature_device'], batch_size=config['feature_batch_size'],
                offline=offline, progress=log.progress, drugbank_xsd=config['drugbank_xsd']), resumable=True)
            schedule = cache.run('sampling', lambda p: build_sampling(splits_root, p, cell=config['cell'],
                audit_dir=audit, progress=log.progress))
            if any(sha256(path) != source_hashes[key] for key, path in sources.items()):
                raise ValueError('An original input changed during preparation; no corpus will be published')
            result = assemble_corpus(graph, split_result, features, schedule, manifest.parent,
                source_identity=source_identity,
                progress=log.progress)
        write_json(audit / 'completion.json', result)
        log.progress('数据准备和完整语料校验通过：' + str(manifest))
        return result
