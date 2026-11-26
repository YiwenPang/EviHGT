"""Exercise stage recovery, portable assembly and raw CLI dispatch on tiny inputs."""
import json
from pathlib import Path
import sys

import pytest

from data_fixture import make_corpus
from evi_hgt.data.corpus import load_corpus
from evi_hgt.preprocessing.pipeline import StageCache, assemble_corpus, preparation_lock, reuse_corpus, preparation_packages


def test_completed_stage_reuses_verified_outputs_and_rejects_mutation(tmp_path):
    calls = []
    def build(path):
        calls.append(path)
        path.mkdir()
        (path / 'value').write_text('immutable')
        return {'value': str(path / 'value')}
    cache = StageCache(tmp_path, {'raw': 'one'})
    result = cache.run('canonical', build)
    assert cache.run('canonical', build) == result
    assert len(calls) == 1
    Path(result['value']).write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        cache.run('canonical', build)
    with pytest.raises(ValueError, match='identity'):
        StageCache(tmp_path, {'raw': 'two'})


def test_failed_stage_preserves_attempt_and_can_retry(tmp_path):
    def fail(path):
        path.mkdir()
        (path / 'partial').write_text('diagnostic')
        raise RuntimeError('interrupted')
    cache = StageCache(tmp_path, {'raw': 'one'})
    with pytest.raises(RuntimeError, match='interrupted'):
        cache.run('canonical', fail)
    failed = list(tmp_path.glob('canonical_*'))
    assert len(failed) == 1 and (failed[0] / 'partial').is_file()
    def recover(path):
        path.mkdir()
        (path / 'complete').write_text('ok')
        return str(path)
    assert Path(cache.run('canonical', recover)) != failed[0]
    assert (failed[0] / 'partial').read_text() == 'diagnostic'


def test_resumable_feature_stage_retains_same_attempt_and_excludes_other_writers(tmp_path):
    seen = []
    cache = StageCache(tmp_path, {'raw': 'one'})
    def build(path):
        seen.append(path)
        path.mkdir(exist_ok=True)
        if len(seen) == 1:
            (path / 'finished_type').write_text('verified encoder type')
            raise RuntimeError('interrupted between encoder types')
        assert (path / 'finished_type').read_text() == 'verified encoder type'
        (path / 'other_type').write_text('complete')
        return str(path)
    with preparation_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another preparation'):
            with preparation_lock(tmp_path):
                pytest.fail('Second writer acquired the same lock')
        with pytest.raises(RuntimeError, match='interrupted'):
            cache.run('features', build, resumable=True)
        cache.run('features', build, resumable=True)
    assert seen[0] == seen[1] and not (tmp_path / '.prepare.lock').exists()


def test_failed_corpus_validation_never_publishes_final_directory(tmp_path):
    _, graph, splits, features, schedule = components(tmp_path)
    splits['training_seed'] = -1
    with pytest.raises(ValueError, match='training_seed'):
        assemble_corpus(graph, splits, features, schedule, tmp_path / 'prepared', source_identity={})
    assert not (tmp_path / 'prepared').exists()
    assert len(list(tmp_path.glob('prepared.attempt_*'))) == 1


def test_existing_raw_corpus_rejects_a_changed_preparation_identity(tmp_path):
    path = make_corpus(tmp_path / 'corpus')
    m = json.loads(path.read_text())
    expected = {'recipe': 'R0F1', 'build_identity': 'original', 'raw_sha256': {'nodes': 'one'}}
    m['preparation'] = expected
    path.write_text(json.dumps(m))
    assert reuse_corpus(path, expected)['reused'] is True
    for changed in (dict(expected, recipe='R0F0'), dict(expected, build_identity='newcode'),
                    dict(expected, raw_sha256={'nodes': 'two'})):
        with pytest.raises(ValueError, match='different preparation identity'):
            reuse_corpus(path, changed)


def test_missing_or_unverified_preprocessing_dependency_is_actionable(monkeypatch):
    import importlib.metadata
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, 'version', missing)
    with pytest.raises(RuntimeError, match='requirements-preprocess.txt'):
        preparation_packages()
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.0')
    with pytest.raises(RuntimeError, match='rdkit==2026.3.6'):
        preparation_packages()


def components(tmp_path):
    path = make_corpus(tmp_path / 'source')
    m = json.loads(path.read_text())
    def absolute(value):
        return str(path.parent / value)
    graph = {k: absolute(m[k]) for k in ('nodes', 'edge_catalog', 'target_map', 'forbidden_facts')}
    graph['csr'] = {k: absolute(v) for k, v in m['csr'].items()}
    for kind in ('fact_files', 'evidence_files'):
        graph[kind] = [dict(row, path=absolute(row['path'])) for row in m[kind]]
    splits = {'training_seed': m['training_seed'], 'files': {
        k: absolute(m[k]) for k in ('blocked_pairs', 'observed_pairs', 'eligibility', 'training_pairs')}}
    splits['files'].update({k: absolute(v) for k, v in m['panels'].items()})
    features = {kind: {k: absolute(v) for k, v in m[kind].items()} for kind in ('raw_features', 'pretrained')}
    schedule = dict(m['schedule'])
    for key in ('groups', 'conditional_records'):
        schedule[key] = absolute(schedule[key])
    return path, graph, splits, features, schedule


def test_assembly_is_portable_and_executes_synthetic_training(tmp_path):
    import torch
    from evi_hgt.runner import run
    from evi_hgt.settings import Settings
    original, graph, splits, features, schedule = components(tmp_path)
    result = assemble_corpus(graph, splits, features, schedule, tmp_path / 'prepared',
                             source_identity={'scope': 'SYNTHETIC_TEST_ONLY'})
    manifest = Path(result['manifest'])
    corpus = load_corpus(manifest)
    assert corpus.plan.total_steps == load_corpus(original).plan.total_steps == 3
    assert all(not Path(p).is_absolute() for p in corpus.manifest['files'])
    del corpus
    torch.set_num_threads(1)
    settings = Settings(manifest=str(manifest), output_dir=str(tmp_path / 'outputs'),
                        device='cpu', precision='fp32', checkpoint_every=1)
    completion = json.loads((run(settings) / 'completion.json').read_text())
    assert completion['updates'] == 3
    with pytest.raises(FileExistsError):
        assemble_corpus(graph, splits, features, schedule, tmp_path / 'prepared', source_identity={})


@pytest.mark.parametrize('mode,checkpoint,expected_prepare,expected_runs', [
    ('train', None, 1, 1), ('prepare', None, 1, 0),
    ('validate', None, 0, 1), ('evaluate', 'saved.pt', 0, 1), ('train', 'saved.pt', 0, 1)])
def test_cli_prepares_only_new_training_or_explicit_prepare(tmp_path, monkeypatch, mode,
                                                         checkpoint, expected_prepare, expected_runs):
    from evi_hgt import cli
    from evi_hgt.settings import Settings
    prepared, runs = [], []
    monkeypatch.setattr(cli, 'load_settings', lambda _: Settings(manifest=str(tmp_path/'missing/manifest.json')))
    monkeypatch.setattr(cli, 'prepare_raw', lambda *a, **kw: prepared.append((a,kw)))
    monkeypatch.setattr(cli, 'run', lambda *a, **kw: runs.append(kw))
    args = ['main.py', '--mode', mode, '--raw-config', 'myraw.yaml', '--offline']
    if checkpoint:
        args += ['--checkpoint', checkpoint]
    monkeypatch.setattr(sys, 'argv', args)
    assert cli.main() == 0
    assert len(prepared) == expected_prepare and len(runs) == expected_runs


def test_failed_preparation_cannot_start_training(tmp_path, monkeypatch):
    from evi_hgt import cli
    from evi_hgt.settings import Settings
    monkeypatch.setattr(cli, 'load_settings', lambda _: Settings(manifest=str(tmp_path/'missing/manifest.json')))
    def fail(*a, **kw):
        raise ValueError('invalid raw identity')
    monkeypatch.setattr(cli, 'prepare_raw', fail)
    monkeypatch.setattr(cli, 'run', lambda *a, **kw: pytest.fail('Training must not run'))
    monkeypatch.setattr(cli, 'startup_failure', lambda exc: None)
    monkeypatch.setattr(sys, 'argv', ['main.py'])
    assert cli.main() == 1
