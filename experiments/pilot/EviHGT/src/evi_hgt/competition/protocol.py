"""Immutable admission contract for the shared development competition."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/competition_20261009'
DATASET = ROOT / 'data/processed/competition_20261009/pilot_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def fingerprint_files(paths):
    return {str(Path(p).resolve()): sha256(p) for p in sorted(map(Path, paths))}


def verify_fingerprints(mapping):
    for name, expected in mapping.items():
        if not Path(name).is_file() or sha256(name) != expected:
            raise ValueError(f'Frozen file changed: {name}')


def validate_scope(contract):
    if (contract.get('scope') != 'COMMON_PILOT_DEVELOPMENT'
            or contract.get('training_authorized') is not True
            or any(contract.get(k) is not False for k in
                   ('outer_scoring', 'final_scoring', 'unlabeled_is_negative'))):
        raise ValueError('Competition scope or scientific semantics are not authorized')


def validate_score_panel(rows, expected):
    def keys(records):
        return [(r['drug_id'], r['disease_id'], r['role']) for r in records]
    actual, required = keys(rows), keys(expected)
    pairs = [(r['drug_id'], r['disease_id']) for r in rows]
    if (len(actual) != len(required) or len(set(pairs)) != len(pairs)
            or set(actual) != set(required)):
        raise ValueError('Scores do not preserve the exact frozen pairs and roles')
    if any(not math.isfinite(float(r['support_logit'])) for r in rows):
        raise ValueError('Nonfinite support score')


def select_candidate(candidates):
    if not candidates:
        raise ValueError('No candidate scores')
    for candidate in candidates:
        if any(not math.isfinite(float(candidate[k])) for k in ('macro_auc', 'macro_ap', 'lr')):
            raise ValueError('Nonfinite candidate metric')
    return max(candidates, key=lambda r: (r['macro_auc'], r['macro_ap'],
               r['lr'] == .001, -r['lr'], -r['epoch']))


def admit_job_config(manifest, job):
    validate_scope(manifest)
    path = Path(job['config'])
    if sha256(path) != job['config_sha256']:
        raise ValueError('Job config changed')
    config = read_json(path)
    if config.get('action') != 'train' or config.get('method') != job['method'] or job['method'] not in ('HGTDR', 'TxGNN'):
        raise ValueError('Job action or method is not authorized')
    if config.get('protocol_digest') != manifest['digest']:
        raise ValueError('Job protocol binding does not match')
    return config


def certify_cached_control(dataset=DATASET):
    """Recompute common metrics from the already selected, hash-bound R2 endpoint."""
    import polars as pl
    from evi_hgt.phase85_part2b.runtime import inner_metrics
    dataset = Path(dataset)
    data_manifest = read_json(dataset / 'manifest.json')
    review_path = ROOT / 'outputs/phase85_part3a_r4/REVIEW_COMPLETE.json'
    event_path = ROOT / 'outputs/phase85_part3a_r2/events/B.json'
    review, event = read_json(review_path), read_json(event_path)
    if review['status'] != 'COMPLETE_VERIFIED' or review['recommendation']['cell'] != 'conditional_5e5':
        raise ValueError('Previously frozen R4 control selection differs')
    for value in (review, event):
        if value['digest'] != canonical_digest({k:v for k,v in value.items() if k != 'digest'}):
            raise ValueError('Historical receipt digest changed')
    for key in ('corpus_digest', 'repeat_key', 'fold', 'fold_manifest_digest', 'holdout_exclusion_digest'):
        if event['bindings'][key] != data_manifest['bindings'][key]:
            raise ValueError(f'Cached control has different {key}')
    verify_fingerprints({event['checkpoint']: event['checkpoint_sha256'], **event['artifacts']})
    panels = {}
    for name in ('validation', 'train_probe'):
        source = ROOT / f'outputs/phase85_part3a_r2/scores/B/{name}.parquet'
        rows = pl.read_parquet(source).to_dicts()
        validate_score_panel(rows, pl.read_parquet(dataset / f'pairs/{name}.parquet').to_dicts())
        metrics = inner_metrics(rows)
        if name == 'validation':
            for key in ('macro_auc', 'macro_ap'):
                if abs(metrics[key] - event['metrics']['validation_metrics'][key]) > 1e-12:
                    raise ValueError('Cached control metric recomputation mismatch')
        panels[name] = dict(scores=str(source), sha256=sha256(source),
                            metrics={k:v for k,v in metrics.items() if k != 'disease_metrics'})
    return dict(status='VERIFIED_CACHED_CONTROL', method='EVIHGT_SIDE', cached=True,
                selection='R4 conditional_5e5 endpoint', lr=5e-5, optimizer_steps=10246,
                checkpoint=event['checkpoint'], checkpoint_sha256=event['checkpoint_sha256'],
                selection_receipt=str(review_path), selection_receipt_sha256=sha256(review_path),
                event_receipt=str(event_path), event_receipt_sha256=sha256(event_path),
                prior_development_budget_equal_to_external_baselines=False,
                independent_confirmatory_result=False, panels=panels)


def prepare(spec_path=None):
    from .data import verify_export
    if (OUT / 'protocol.json').exists():
        return verify()
    data = verify_export(DATASET)
    specs = read_json(spec_path or OUT / 'job_specs.json')
    if sorted(j['method'] for j in specs['jobs']) != ['HGTDR', 'TxGNN']:
        raise ValueError('Both external baseline jobs are required')
    control = certify_cached_control()
    source_paths = [ROOT / 'main.py', *Path(__file__).parent.glob('*.py'),
                    ROOT / 'src/evi_hgt/phase85_part2b/runtime.py',
                    ROOT / 'src/evi_hgt/evaluation/validation_diagnostics.py',
                    ROOT / 'src/evi_hgt/evaluation/ranking.py',
                    ROOT / 'src/evi_hgt/evaluation/protocol.py']
    source_paths += [Path(p) for j in specs['jobs'] for p in j['source_files']]
    source_paths += [Path(j['preflight']) for j in specs['jobs']]
    source_paths += [Path(p) for p in specs.get('additional_bindings', [])]
    for job in specs['jobs']:
        preflight = read_json(job['preflight'])
        if preflight.get('status') not in ('PASS', 'PREFLIGHT_PASS', 'PREFLIGHT_COMPLETE'):
            raise ValueError(f"{job['method']} resource preflight has not passed")
        settings = job['settings']
        if settings['action'] != 'train' or Path(settings['dataset']).resolve() != DATASET.resolve():
            raise ValueError('Job must train on the exact common dataset')
        if settings['method'] != job['method'] or settings['learning_rates'] != [.0003, .001, .003]:
            raise ValueError('Unexpected method or unregistered learning-rate search')
        if 'protocol_digest' in settings:
            raise ValueError('Protocol binding is added only after freezing')
    manifest = dict(schema='COMMON_PILOT_COMPETITION_V1', scope='COMMON_PILOT_DEVELOPMENT',
        training_authorized=True, authorization='User approved the pilot training runs',
        outer_scoring=False, final_scoring=False, unlabeled_is_negative=False,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), dataset=str(DATASET),
        data_digest=data['digest'], data_manifest_sha256=sha256(DATASET / 'manifest.json'),
        cached_control=control, source_fingerprints=fingerprint_files(set(source_paths)),
        jobs=specs['jobs'], execution='PARALLEL_EXTERNAL_MODELS_SEQUENTIAL_LR_WITHIN_MODEL',
        selection='INNER_MACRO_AUC_THEN_AP; EXACT_CROSS_LR_TIE_DEFAULT_1E3_THEN_SMALLER; EARLIER_EPOCH',
        unknown_semantics='BCE zero is an algorithmic contrast target, never confirmed clinical negative',
        method_objectives={'HGTDR':'Native indication P/U objective; contraindication context only',
                           'TxGNN':'Native multi-relation pretraining and indication/contraindication P/U fine-tuning',
                           'EVIHGT_SIDE':'Existing independent support and unsuitability channels'},
        final_holdout='SEALED', future_formal_cv='3 original repeat keys x 5 folds; not started by this runner')
    manifest['digest'] = canonical_digest(manifest)
    write_json(OUT / 'protocol.json', manifest)
    write_json(OUT / 'cached_control.json', control)
    jobs = []
    for job in specs['jobs']:
        config = dict(job['settings'], protocol_digest=manifest['digest'])
        config_path = OUT / 'configs' / (job['method'].lower() + '.json')
        write_json(config_path, config)
        jobs.append(dict(method=job['method'], python=job['python'], cwd=job['cwd'],
                         module=job['module'], config=str(config_path), config_sha256=sha256(config_path)))
    write_json(OUT / 'launch.json', dict(protocol_sha256=sha256(OUT / 'protocol.json'), jobs=jobs))
    return verify()


def verify():
    from .data import verify_export
    manifest, launch = read_json(OUT / 'protocol.json'), read_json(OUT / 'launch.json')
    validate_scope(manifest)
    if manifest['digest'] != canonical_digest({k:v for k,v in manifest.items() if k != 'digest'}):
        raise ValueError('Frozen protocol digest mismatch')
    if launch['protocol_sha256'] != sha256(OUT / 'protocol.json'):
        raise ValueError('Launch protocol changed')
    verify_fingerprints(manifest['source_fingerprints'])
    if sha256(DATASET / 'manifest.json') != manifest['data_manifest_sha256']:
        raise ValueError('Data manifest changed')
    if verify_export(DATASET)['digest'] != manifest['data_digest']:
        raise ValueError('Shared data changed')
    if certify_cached_control() != manifest['cached_control']:
        raise ValueError('Previously selected EviHGT control changed')
    expected = {j['method']:j for j in manifest['jobs']}
    if [j['method'] for j in launch['jobs']] != [j['method'] for j in manifest['jobs']]:
        raise ValueError('Execution order changed')
    for job in launch['jobs']:
        cfg = admit_job_config(manifest, job)
        if {k:v for k,v in cfg.items() if k != 'protocol_digest'} != expected[job['method']]['settings']:
            raise ValueError('Frozen settings changed')
        if any(job[k] != expected[job['method']][k] for k in ('python', 'cwd', 'module')):
            raise ValueError('Frozen launch command changed')
    return manifest
