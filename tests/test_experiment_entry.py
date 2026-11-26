import json

import pytest

from evi_hgt.io import write_json


def test_paper_plan_has_separate_loss_protocol_and_deduplicates_shared_arms():
    from experiments.workflow.cli import parser, build_jobs
    jobs = build_jobs(parser().parse_args(['--experiment', 'all']))
    assert len({job['id'] for job in jobs}) == len(jobs)
    assert sum(job['kind'] == 'cold' for job in jobs) == 4
    assert sum(job['kind'] == 'cold_hgtdr' for job in jobs) == 1
    assert sum(job['kind'] == 'comparison' for job in jobs) == 30
    loss0 = [job for job in jobs if job['group'] == 'losses' and job['cell'] == 'R0F0']
    assert {job['loss'] for job in loss0} == {'sigmoid', 'softplus'}
    assert all(job['protocol'] == 'losses' for job in loss0)


def test_summary_averages_folds_within_disease_first(tmp_path):
    from experiments.workflow.reporting import aggregate
    def panel(values):
        return dict(status='COMPLETE', candidate_identity_sha256='same', disease_metrics=[
            dict(disease_id=d, P=1, U=1, N=2, auc=v, ap=v, aupr_trapezoid=v,
                 **{f'{k}@{n}': v for n in (5, 10) for k in ('P', 'R', 'Hits', 'NDCG')})
            for d, v in values])
    rows = [dict(group='comparisons', method='FULL', cell='R0F0', head='support',
                 metrics=panel([('A', 1), ('B', 0)])),
            dict(group='comparisons', method='FULL', cell='R0F1', head='support',
                 metrics=panel([('A', 1)]))]
    summary, diseases = aggregate(rows)
    assert summary[0]['ap'] == .5
    assert summary[0]['diseases'] == 2
    assert len(diseases) == 2
    rows.append(dict(rows[0], method='OTHER', metrics=dict(panel([('A', 1)]), candidate_identity_sha256='different')))
    with pytest.raises(ValueError, match='candidate'):
        aggregate(rows)


def test_completed_job_receipt_rejects_changed_output(tmp_path):
    from experiments.workflow.execution import seal_job, verify_job
    directory = tmp_path / 'job'
    run = directory / 'runs/run'
    write_json(run / 'completion.json', {'status': 'COMPLETE'})
    write_json(run / 'metrics.json', {'support': {}})
    seal_job(directory, run, {'method': 'A'})
    assert verify_job(directory, {'method': 'A'}) == run
    with pytest.raises(ValueError, match='binding'):
        verify_job(directory, {'method': 'B'})
    (run / 'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        verify_job(directory, {'method': 'A'})


def test_duplicate_cli_values_do_not_duplicate_report_rows():
    from experiments.workflow.cli import parser, build_jobs
    jobs = build_jobs(parser().parse_args(['--experiment', 'ablations', '--cells', 'R0F1', 'R0F1',
                                         '--variants', 'FULL', 'FULL']))
    assert len(jobs) == 1 and jobs[0]['groups'] == ['ablations']


def test_result_reader_rejects_missing_required_head(tmp_path):
    from experiments.workflow.reporting import metric_rows
    job = dict(kind='comparison', method='TRAIN_VISIBLE_POPULARITY', cell='R0F1', groups=['comparisons'])
    write_json(tmp_path / 'completion.json', dict(status='COMPLETE', method=job['method']))
    write_json(tmp_path / 'metrics.json', dict(support={}))
    with pytest.raises(ValueError, match='heads'):
        metric_rows(job, tmp_path)


def test_cli_actual_smoke_and_verified_rerun(tmp_path):
    from experiments.workflow.cli import main
    output = tmp_path / 'suite'
    args = ['--experiment', 'ablations', '--variants', 'FULL', '--smoke', '--run-dir', str(output)]
    assert main(args) == 0
    result = json.loads((output / 'completion.json').read_text())
    assert result['status'] == 'COMPLETE' and result['scope'] == 'SYNTHETIC_TEST_ONLY'
    assert result['jobs'] == 1
    assert (output / 'report.html').is_file()
    receipts = list((output / 'jobs').glob('*/receipt.json'))
    before = receipts[0].read_bytes()
    assert main(args) == 0
    assert receipts[0].read_bytes() == before


def test_preparation_failure_never_writes_suite_completion(tmp_path, monkeypatch):
    from experiments.workflow import execution
    from experiments.workflow.cli import main
    def fail(*args, **kwargs):
        raise RuntimeError('controlled preparation failure')
    monkeypatch.setattr(execution, 'prepare_inputs', fail)
    out = tmp_path / 'failed'
    assert main(['--experiment', 'losses', '--smoke', '--run-dir', str(out)]) == 1
    assert not (out / 'completion.json').exists()
    assert 'controlled preparation failure' in (out / 'logs/errors.log').read_text()


def test_suite_resumes_endpoint_without_new_updates_and_binds_parent_checkpoint(tmp_path, monkeypatch):
    from experiments.workflow.cli import main
    from experiments.workflow.execution import verify_job
    from evi_hgt import runner
    original = runner.create_report
    def fail(*args, **kwargs):
        raise RuntimeError('interrupt after final checkpoint')
    monkeypatch.setattr(runner, 'create_report', fail)
    output = tmp_path / 'resumed'
    args = ['--experiment', 'comparisons', '--methods', 'FULL', '--smoke', '--run-dir', str(output)]
    assert main(args) == 1
    assert not (output / 'completion.json').exists()
    monkeypatch.setattr(runner, 'create_report', original)
    assert main(args + ['--resume']) == 0
    receipt_path, = (output / 'jobs').glob('*/receipt.json')
    receipt = json.loads(receipt_path.read_text())
    directory = receipt_path.parent
    run = verify_job(directory, receipt['binding'])
    complete = json.loads((run / 'completion.json').read_text())
    assert complete['updates_this_invocation'] == 0
    assert receipt['dependencies']
    assert not (run / 'checkpoints/latest.pt').exists()
    dependency = directory / next(iter(receipt['dependencies']))
    dependency.write_bytes(dependency.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='checkpoint dependency changed'):
        verify_job(directory, receipt['binding'])


def test_completed_cold_suite_revalidates_sealed_input_before_reusing_results(tmp_path):
    from experiments.workflow.cli import main
    output = tmp_path / 'cold'
    args = ['--experiment', 'cold', '--skip-cold-hgtdr', '--smoke', '--run-dir', str(output)]
    assert main(args) == 0
    manifests = json.loads((output / 'inputs.json').read_text())
    from pathlib import Path
    manifest = next(Path(name) for name in manifests if Path(name).name == 'manifest.json')
    target = manifest.parent / 'test_support.parquet'
    target.write_bytes(target.read_bytes() + b'changed after successful run')
    assert main(args) == 1
    assert not (output / 'completion.json').exists()
