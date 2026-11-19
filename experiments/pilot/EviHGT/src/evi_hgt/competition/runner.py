"""Run the predeclared external baselines and preserve every process outcome."""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from .protocol import OUT, DATASET, admit_job_config, read_json, verify, verify_fingerprints, write_json, sha256, validate_score_panel, select_candidate


def validate_completion(output_dir, protocol_digest=None):
    path = Path(output_dir) / 'COMPLETE.json'
    if not path.is_file():
        raise ValueError('Missing child completion receipt')
    receipt = read_json(path)
    if (receipt.get('status') != 'COMPLETE' or receipt.get('training_started') is not True
            or receipt.get('optimizer_steps', 0) <= 0):
        raise ValueError('Invalid child completion receipt')
    if protocol_digest is not None and receipt.get('protocol_digest') != protocol_digest:
        raise ValueError('Child completion belongs to another protocol')
    return receipt


def audit_results(output_dir, config):
    """Independently recompute metrics; neither missing pairs nor child-selected scores pass silently."""
    import polars as pl
    from evi_hgt.inner_validation.runtime import inner_metrics
    root = Path(output_dir)
    receipt = validate_completion(root, config['protocol_digest'])
    if receipt.get('bindings', {}).get('protocol_digest') != config['protocol_digest']:
        raise ValueError('Top-level baseline training bindings differ from protocol')
    recorded = {float(r['learning_rate']):r for r in receipt.get('candidates', [])}
    if set(recorded) != set(config['learning_rates']) or len(receipt['candidates']) != len(recorded):
        raise ValueError('Completion does not contain exactly the registered candidates')
    candidates = []
    for lr in config['learning_rates']:
        name = 'lr_' + format(float(lr), '.8g').replace('.', 'p').replace('-', 'm')
        directory = root / name
        candidate_receipt = read_json(directory / 'COMPLETED.json')
        if candidate_receipt != recorded[lr] or candidate_receipt.get('bindings') != receipt['bindings']:
            raise ValueError('Candidate training or score receipt changed after completion')
        if candidate_receipt.get('status') != 'COMPLETE' or candidate_receipt.get('optimizer_steps', 0) <= 0:
            raise ValueError('Candidate has no completed training receipt')
        artifact_hashes = candidate_receipt.get('artifact_hashes', {})
        required = {'best.pt', 'best.json', 'validation_scores.parquet', 'train_probe_scores.parquet'}
        if not required <= set(artifact_hashes):
            raise ValueError('Candidate receipt does not bind all required artifacts')
        for filename, expected_hash in artifact_hashes.items():
            artifact = (directory / filename).resolve()
            if directory.resolve() not in artifact.parents:
                raise ValueError('Candidate artifact escapes its output directory')
            if sha256(artifact) != expected_hash:
                raise ValueError('Completed candidate artifact changed')
        best = read_json(directory / 'best.json')
        if sha256(directory / 'best.pt') != best['checkpoint_sha256']:
            raise ValueError('Selected baseline checkpoint hash mismatch')
        panels = {}
        for panel in ('validation', 'train_probe'):
            path = directory / (panel + '_scores.parquet')
            rows = pl.read_parquet(path).to_dicts()
            validate_score_panel(rows, pl.read_parquet(DATASET / f'pairs/{panel}.parquet').to_dicts())
            metrics = inner_metrics(rows)
            saved_metrics = best.get('metrics', {})
            if panel in saved_metrics:
                saved_metrics = saved_metrics[panel]
            elif panel == 'train_probe':
                saved_metrics = candidate_receipt.get('train_probe_metrics', {})
            if not all(k in saved_metrics for k in ('macro_auc', 'macro_ap')):
                raise ValueError('Candidate receipt is missing its reported panel metrics')
            for key in ('macro_auc', 'macro_ap'):
                if abs(float(metrics[key]) - float(saved_metrics[key])) > 1e-10:
                    raise ValueError('Independent common-metric recomputation differs from baseline')
            panels[panel] = dict(path=str(path), sha256=sha256(path),
                metrics={k:v for k,v in metrics.items() if k != 'disease_metrics'})
        metrics = panels['validation']['metrics']
        candidates.append(dict(lr=lr, epoch=best['epoch'], macro_auc=metrics['macro_auc'],
            macro_ap=metrics['macro_ap'], panels=panels, checkpoint=str(directory / 'best.pt'),
            checkpoint_sha256=best['checkpoint_sha256'], candidate=name))
    chosen = select_candidate(candidates)
    if receipt['selected_learning_rate'] != chosen['lr']:
        raise ValueError('Child selection differs from the frozen common rule')
    return dict(method=config['method'], candidates=candidates, selected=chosen,
                protocol_digest=config['protocol_digest'], scope='COMMON_PILOT_DEVELOPMENT',
                optimizer_steps=receipt['optimizer_steps'], validation='EXACT_PANELS_AND_RECOMPUTED_METRICS')


def write_comparison(manifest, completed):
    control = manifest['cached_control']
    rows = [('EVIHGT_SIDE (cached R4 control)', control['panels']['validation']['metrics'],
             control['panels']['train_probe']['metrics'])]
    for result in completed:
        chosen = result['audit']['selected']
        rows.append((result['method'] + ' adapted', chosen['panels']['validation']['metrics'],
                     chosen['panels']['train_probe']['metrics']))
    lines = ['# Common pilot development comparison', '',
        'Development results only. These are P-vs-U metrics; U is not a confirmed negative. '
        'The EviHGT control had a larger prior development budget. No outer or sealed-final scores were opened.', '',
        '| Method | Inner macro AUC | Inner macro AP | Probe macro AUC | Probe macro AP |',
        '|---|---:|---:|---:|---:|']
    for name, validation, probe in rows:
        lines.append(f"| {name} | {validation['macro_auc']:.6f} | {validation['macro_ap']:.6f} | {probe['macro_auc']:.6f} | {probe['macro_ap']:.6f} |")
    lines.extend(['', 'All three learning-rate candidates are retained in the result audit JSON. '
                  'Candidate selection used inner validation only. Device costs differ: HGTDR used CUDA, TxGNN used CPU.', ''])
    (OUT / 'COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')


def run():
    manifest = verify()
    launch = read_json(OUT / 'launch.json')
    lock = OUT / 'RUNNING.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('A run lock already exists; inspect process state before restarting')
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    completed = []
    active = []
    failed = []
    try:
        write_json(OUT / 'RUNNING.json', dict(status='RUNNING', pid=os.getpid(), protocol_digest=manifest['digest']))
        for job in launch['jobs']:
            verify_fingerprints(manifest['source_fingerprints'])
            config = admit_job_config(manifest, job)
            target = Path(config['output_dir'])
            if (target / 'COMPLETE.json').exists():
                completed.append(dict(method=job['method'], audit=audit_results(target, config)))
                continue
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f'Refusing to overwrite incomplete experiment: {target}')
            target.mkdir(parents=True, exist_ok=True)
            log_path = OUT / (job['method'].lower() + '_training.log')
            command = [job['python'], '-u', '-m', job['module'], '--config', job['config']]
            log = log_path.open('a', encoding='utf-8')
            try:
                child = subprocess.Popen(command, cwd=job['cwd'], stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            finally:
                log.close()
            active.append(dict(child=child, method=job['method'], config=config,
                               output_dir=str(target), log=str(log_path)))
        while active:
            write_json(OUT / 'RUNNING.json', dict(status='RUNNING', pid=os.getpid(),
                protocol_digest=manifest['digest'], completed_methods=[x['method'] for x in completed],
                failed_methods=failed,
                jobs=[dict(method=j['method'], child_pid=j['child'].pid, output_dir=j['output_dir'], log=j['log']) for j in active]))
            for item in list(active):
                code = item['child'].poll()
                if code is None:
                    continue
                if code != 0:
                    failed.append(dict(method=item['method'], exit_code=code, log=item['log']))
                else:
                    try:
                        audit = audit_results(item['output_dir'], item['config'])
                        write_json(OUT / (item['method'].lower() + '_result_audit.json'), audit)
                        completed.append(dict(method=item['method'], audit=audit))
                    except Exception as exc:
                        failed.append(dict(method=item['method'], error=str(exc), log=item['log']))
                active.remove(item)
                if failed:
                    write_json(OUT / 'CHILD_FAILURES.json', failed)
            if active:
                time.sleep(5)
        if failed:
            raise RuntimeError(f'External experiments failed; successful jobs were preserved: {failed}')
        verify()
        write_comparison(manifest, completed)
        write_json(OUT / 'COMPLETE.json', dict(status='COMPLETE', protocol_digest=manifest['digest'],
            scope=manifest['scope'], external_jobs=completed, cached_control=manifest['cached_control'],
            outer_scoring=False, final_scoring=False))
        write_json(OUT / 'RUNNING.json', dict(status='COMPLETE', protocol_digest=manifest['digest']))
        return 0
    except BaseException as exc:
        for item in active:
            if item['child'].poll() is None:
                item['child'].terminate()
                item['child'].wait(timeout=30)
        write_json(OUT / 'FAILED.json', dict(status='FAILED', protocol_digest=manifest['digest'],
            error=str(exc), traceback=traceback.format_exc(), completed_methods=[x['method'] for x in completed]))
        write_json(OUT / 'RUNNING.json', dict(status='FAILED', error=str(exc), protocol_digest=manifest['digest']))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(action):
    if action == 'prepare':
        from .protocol import prepare
        manifest = prepare()
        print(json.dumps(dict(status='FROZEN', digest=manifest['digest'])))
    elif action == 'verify':
        manifest = verify()
        print(json.dumps(dict(status='VERIFIED', digest=manifest['digest'])))
    elif action == 'run':
        return run()
    elif action == 'status':
        path = OUT / 'RUNNING.json'
        print(json.dumps(read_json(path) if path.exists() else dict(status='NOT_STARTED'), ensure_ascii=False))
    return 0
