"""Disease-first aggregation of newly completed, candidate-matched experiments."""
from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
from statistics import fmean

from evi_hgt.io import write_json

KEYS = ('auc', 'ap', 'aupr_trapezoid') + tuple(f'{name}@{k}' for k in (5, 10) for name in ('P', 'R', 'Hits', 'NDCG'))


def aggregate(rows):
    candidates, pools, required = {}, defaultdict(list), set()
    for row in rows:
        pair = row['group'], row['cell'], row['head']
        metrics = row['metrics']
        identity = (metrics['candidate_identity_sha256'],
                    sorted((v['disease_id'], v['P'], v['U'], v['N']) for v in metrics['disease_metrics']))
        if pair in candidates and candidates[pair] != identity:
            raise ValueError('Methods use different candidate identities within one experiment cell')
        candidates[pair] = identity
        required.add((row['group'], row['method'], row['head']))
        for local in row['metrics']['disease_metrics']:
            key = row['group'], row['method'], row['head'], local['disease_id']
            pools[key].append((row['cell'], local))
    diseases, totals = [], defaultdict(list)
    for (group, method, head, disease), values in sorted(pools.items()):
        if len({cell for cell, _ in values}) != len(values):
            raise ValueError('Duplicate disease/cell metric in report')
        local = dict(group=group, method=method, head=head, disease_id=disease, folds=len(values),
                     **{key: fmean(r[key] for _, r in values) for key in KEYS})
        diseases.append(local)
        totals[group, method, head].append(local)
    summary = [dict(group=g, method=m, head=h, diseases=len(totals[g, m, h]),
                    **{key: fmean(r[key] for r in totals[g, m, h]) if totals[g, m, h] else None for key in KEYS})
               for g, m, h in sorted(required)]
    return summary, diseases


def metric_rows(job, root):
    def read(name):
        return json.loads((root / name).read_text('utf-8'))
    if job['kind'] == 'cold_hgtdr':
        complete = read('result.json')
        metrics = {head: read(f'test/{head}_metrics.json') for head in ('support', 'contra')}
    elif job['kind'] == 'cold':
        complete = read('result.json')
        metrics = read('test/metrics.json')['metrics']
    elif job['kind'] == 'comparison':
        complete = read('completion.json')
        metrics = read('metrics.json')
    else:
        complete = read('completion.json')
        metrics = read(f"evaluation/step_{complete['updates']}/metrics.json")['metrics']
    if complete.get('status') != 'COMPLETE':
        raise ValueError('Result has no successful completion record')
    if job['kind'] == 'comparison' and complete.get('method') != job['method']:
        raise ValueError('Completed comparison method differs from its job')
    if job['kind'] in ('full', 'ablation') and complete.get('variant') != job['method']:
        raise ValueError('Completed variant differs from its job')
    if job['kind'] == 'ablation' and complete.get('loss') != job['loss']:
        raise ValueError('Completed loss differs from its job')
    if job['kind'] in ('cold', 'cold_hgtdr') and complete.get('case') != job['cell']:
        raise ValueError('Completed cold case differs from its job')
    expected = {'support'} if job['method'] == 'HGTDR_ADAPTED' else {'contra'} if job['method'] == 'HGTDR_ADAPTED_CONTRA' else {'support', 'contra'}
    if set(metrics) != expected:
        raise ValueError('Result is missing required output heads')
    for panel in metrics.values():
        if (panel.get('status') not in ('COMPLETE', 'NO_EVALUABLE_DISEASES')
                or not panel.get('candidate_identity_sha256')
                or panel.get('evaluable_diseases') != len(panel.get('disease_metrics', []))):
            raise ValueError('Required metric panel is incomplete')
        if (panel['status'] == 'COMPLETE') != bool(panel['disease_metrics']):
            raise ValueError('Metric completion status and disease coverage disagree')
    method = job['method'] + ('/' + job['loss'] if job['kind'] == 'ablation' else '')
    return [dict(group=group if group != 'cold' else 'cold/' + job['cell'], method=method,
                 cell=job['cell'], head=head, metrics=panel, run=str(root))
            for group in job['groups'] for head, panel in metrics.items()]


def write_report(root, results, *, smoke):
    rows = [row for job, path in results for row in metric_rows(job, path)]
    summary, diseases = aggregate(rows)
    scope = 'SYNTHETIC_TEST_ONLY' if smoke else 'NEW_LOCAL_EXPERIMENT'
    write_json(root / 'summary.json', dict(scope=scope, aggregation='mean within disease across eligible folds, then mean across diseases',
        metrics=summary, runs=[dict(job=job, path=str(path)) for job, path in results]))
    for name, values in (('metrics.csv', summary), ('disease_metrics.csv', diseases)):
        with (root / name).open('w', encoding='utf-8-sig', newline='') as handle:
            if values:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    columns = ('group', 'method', 'head', 'diseases', *KEYS)
    table = '<tr>' + ''.join('<th>' + escape(k) + '</th>' for k in columns) + '</tr>'
    for row in summary:
        table += '<tr>' + ''.join('<td>' + escape(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k])) + '</td>' for k in columns) + '</tr>'
    links = ''.join(f'<li>{escape(job["id"])}: <a href="{escape(Path(path).relative_to(root).as_posix())}/">运行目录</a></li>' for job, path in results)
    (root / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EviHGT 实验结果</title>'
        '<style>body{font:16px system-ui;margin:40px;color:#172b42}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #ccd6df;text-align:right}main{overflow:auto}</style>'
        f'<h1>EviHGT 实验结果</h1><p>{scope}</p><p>每个疾病先在可评估折内取均值，再跨疾病取均值。P 与 U 的排序评估；U 表示未观测。</p>'
        f'<p><a href="metrics.csv">指标 CSV</a> · <a href="disease_metrics.csv">逐疾病 CSV</a> · <a href="summary.json">JSON</a></p><main><table>{table}</table></main><ul>{links}</ul></html>', encoding='utf-8')
