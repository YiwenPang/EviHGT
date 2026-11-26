"""Bound serial experiment execution with verifiable completion and recovery."""
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from filelock import FileLock
import torch

from evi_hgt.console import Console
from evi_hgt.data.common import sha256
from evi_hgt.io import digest, write_json
from evi_hgt.settings import ROOT


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def seal_job(directory, run, expected):
    directory, run = directory.resolve(), run.resolve()
    if not run.is_relative_to(directory):
        raise ValueError('Experiment run must be inside its job directory')
    files = {path.relative_to(directory).as_posix(): sha256(path) for path in sorted(run.rglob('*'))
             if path.is_file() and 'logs' not in path.relative_to(run).parts and path.suffix in
             ('.json', '.csv', '.html', '.parquet', '.pt', '.npy')}
    dependencies = {}
    if (run / 'summary.json').is_file():
        checkpoint = read(run / 'summary.json').get('checkpoint')
        if checkpoint:
            path = (run / checkpoint['path']).resolve(strict=True)
            if not path.is_relative_to(directory) or sha256(path) != checkpoint['sha256']:
                raise ValueError('Reported checkpoint differs or belongs to another job')
            dependencies[path.relative_to(directory).as_posix()] = checkpoint['sha256']
    write_json(directory / 'receipt.json', dict(status='COMPLETE', binding=expected,
        run=run.relative_to(directory).as_posix(), files=files, dependencies=dependencies))


def verify_job(directory, expected):
    directory = directory.resolve()
    receipt = read(directory / 'receipt.json')
    if receipt['status'] != 'COMPLETE' or not receipt['files'] or receipt.get('binding') != expected:
        raise ValueError('Incomplete or different experiment job binding')
    run = (directory / receipt['run']).resolve()
    if not run.is_relative_to(directory):
        raise ValueError('Experiment run is outside its bound job directory')
    for name, checksum in receipt['files'].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha256(path) != checksum:
            raise ValueError('Completed experiment output changed: ' + name)
    for name, checksum in receipt.get('dependencies', {}).items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != checksum:
            raise ValueError('Completed checkpoint dependency changed: ' + name)
    return run


def binding(args, jobs, settings):
    code = {p.relative_to(ROOT).as_posix(): sha256(p)
            for folder in ('src/evi_hgt', 'experiments', 'configs')
            for p in sorted((ROOT / folder).rglob('*')) if p.suffix in ('.py', '.json', '.yaml', '.patch')}
    for path in [ROOT / 'main.py', *sorted(ROOT.glob('requirements*.txt'))]:
        code[path.relative_to(ROOT).as_posix()] = sha256(path)
    packages = {}
    for name in ('torch', 'numpy', 'polars', 'pyarrow', 'torch-geometric', 'torch-sparse', 'transformers', 'filelock'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    config = Path(args.raw_config).resolve() if args.raw_config else ROOT / 'configs/raw.yaml'
    return dict(jobs=jobs, settings=settings.as_dict(), smoke=args.smoke,
                raw_config=str(config), raw_config_sha256=sha256(config), source=code, packages=packages,
                txgnn_python=args.txgnn_python, txgnn_source=args.txgnn_source)


def prepare_inputs(args, jobs, root, console):
    from .preparation import prepare_shared, prepare_cell
    from .memberships import load_pools
    from .cold import prepare_cold_case, archived_scripts, _verify_test_files
    from .cold_smoke import prepare_cold_smoke
    from .comparison_smoke import create_comparison_smoke_corpus
    from evi_hgt.synthetic import make_corpus
    from evi_hgt.data.corpus import load_corpus
    shared = None if args.smoke else prepare_shared(args.raw_config, audit_dir=root / 'preparation', offline=args.offline)
    context = dict(shared or {}, offline=args.offline or args.smoke)
    context.update({k: getattr(args, k) for k in ('txgnn_python', 'txgnn_source') if getattr(args, k)})
    manifests, cases = {}, {}
    data_root = ROOT / 'data/interim/experiments/smoke' / digest(str(root))[:24] if args.smoke else ROOT / 'data/processed/experiments'
    for job in jobs:
        cell, protocol = job['cell'], job['protocol']
        if job['kind'] in ('cold', 'cold_hgtdr'):
            if cell in cases:
                continue
            destination = data_root / 'cold' / cell if args.smoke else data_root / shared['source_identity']['build_identity'][:20] / cell
            if args.smoke:
                if (destination / 'manifest.json').is_file():
                    from .cold import _case_result
                    info = _case_result(cell, destination)
                else:
                    info = prepare_cold_smoke(cell, destination, audit_dir=root / 'preparation')
                load_corpus(info['manifest'])
                _verify_test_files(archived_scripts()[1], info['manifest'])
            else:
                info = prepare_cold_case(cell, shared, load_pools(shared['pools']), destination,
                                         audit_dir=root / 'preparation')
            cases[cell] = info
        else:
            native_fixture = args.smoke and job['kind'] in ('full', 'comparison')
            key = (cell, protocol, native_fixture)
            if key in manifests:
                continue
            if args.smoke:
                destination = data_root / (cell + '_' + protocol + ('_comparison' if native_fixture else '_ablation'))
                manifest = destination / 'manifest.json'
                if not manifest.is_file():
                    manifest = (create_comparison_smoke_corpus if native_fixture else make_corpus)(destination)
                load_corpus(manifest)
            else:
                kwargs = dict(protocol='losses') if protocol == 'losses' else {}
                manifest = prepare_cell(shared, cell, audit_dir=root / 'preparation', **kwargs)
            manifests[key] = Path(manifest)
    return context, manifests, cases


def _checkpoint(directory):
    paths = list(directory.glob('runs/*/checkpoints/latest.pt')) + list(directory.glob('runs/*/checkpoint.pt'))
    if any(not path.resolve().is_relative_to(directory.resolve()) for path in paths):
        raise ValueError('Resume checkpoint must belong to the same experiment job')
    return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None


def execute_job(job, settings, directory, context, manifests, cases, results, args):
    from .ablations import run_ablation
    from .comparisons import run_comparison
    from .cold import run_cold
    from evi_hgt.runner import run
    cell, kind = job['cell'], job['kind']
    checkpoint = _checkpoint(directory) if args.resume else None
    settings = replace(settings, output_dir=str(directory / 'runs'))
    if kind in ('cold', 'cold_hgtdr'):
        info = cases[cell]
        if kind == 'cold':
            return run_cold(settings, dict(info, run_dir=directory / 'cold_run'), smoke=args.smoke)
        from .cold_hgtdr import run_cold_hgtdr
        reference = next(path for prior, path in results if prior['kind'] == 'cold' and prior['cell'] == cell)
        return run_cold_hgtdr(settings, info, reference, context, output=directory / 'cold_hgtdr', smoke=args.smoke)
    manifest = manifests[cell, job['protocol'], args.smoke and kind in ('full', 'comparison')]
    settings = replace(settings, manifest=str(manifest))
    if kind == 'ablation':
        return run_ablation(settings, variant=job['method'], loss=job['loss'], checkpoint=checkpoint, smoke=args.smoke)
    if kind == 'full':
        return run(settings, checkpoint=checkpoint)
    return run_comparison(job['method'], settings, smoke=args.smoke, checkpoint=checkpoint, context=context)


def run_suite(args, jobs, settings, root):
    from .comparisons import check_native_requirements
    from .reporting import metric_rows, write_report
    root = root.resolve()
    for protected in (ROOT / 'src', ROOT / 'experiments', ROOT / 'configs', ROOT / 'data', ROOT / '.git'):
        if root == ROOT or root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError('Choose a distinct runtime output directory, for example outputs/experiments/run1')
    if not args.smoke:
        from evi_hgt.preprocessing.pipeline import _configuration
        from .preparation import validate_destinations
        config = _configuration(args.raw_config)
        validate_destinations({k: config[k] for k in ('nodes', 'edges', 'drugbank_xml', 'drugbank_xsd')}, [root])
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'suite.lock')), Console(root / 'logs', refresh_seconds=settings.refresh_seconds) as console:
        previous_threads = torch.get_num_threads()
        try:
            if args.smoke:
                torch.set_num_threads(1)
                console.say('[SYNTHETIC_TEST_ONLY] 实验入口软件测试，结果不属于生物医学实验。')
            expected = binding(args, jobs, settings)
            first = root / 'plan.json'
            if first.exists() and read(first) != expected:
                raise ValueError('Experiment selection, source or configuration changed. Use a new --run-dir.')
            write_json(first, expected)
            (root / 'completion.json').unlink(missing_ok=True)
            methods = [job['method'] for job in jobs if job['kind'] == 'comparison']
            if any(job['kind'] == 'cold_hgtdr' for job in jobs):
                methods += ['HGTDR_ADAPTED', 'HGTDR_ADAPTED_CONTRA']
            context = {k: getattr(args, k) for k in ('txgnn_python', 'txgnn_source') if getattr(args, k)}
            if args.mode == 'train':
                if not args.smoke and any(method.startswith('HGTDR_') for method in methods) and settings.device != 'cuda':
                    raise ValueError('Recorded HGTDR experiments require CUDA; use --smoke for the CPU software test')
                check_native_requirements(context, methods, offline=args.offline, smoke=args.smoke)
            console.say(f'[实验计划] {args.experiment}，共 {len(jobs)} 个串行任务。')
            with console.phase(1, '准备并核验当前实验输入'):
                context, manifests, cases = prepare_inputs(args, jobs, root, console)
            prepared = {str(path): sha256(path) for path in manifests.values()}
            prepared.update({str(info['manifest']): sha256(info['manifest']) for info in cases.values()})
            for info in cases.values():
                for path in info['test_panels'].values():
                    prepared[str(path)] = sha256(path)
            if (root / 'inputs.json').exists() and read(root / 'inputs.json') != prepared:
                raise ValueError('Prepared corpus identity changed; choose a new run directory')
            write_json(root / 'inputs.json', prepared)
            if args.mode == 'prepare':
                write_json(root / 'preparation_complete.json', dict(status='INPUTS_PREPARED', manifests=prepared))
                console.say(f'[输入准备完成] {root}')
                return root
            results = []
            for index, job in enumerate(jobs, 1):
                directory = root / 'jobs' / job['id']
                directory.mkdir(parents=True, exist_ok=True)
                console.say(f'[任务 {index}/{len(jobs)}] {job["id"]}')
                job_binding = dict(job=job, suite=digest(expected), inputs=prepared)
                if (directory / 'receipt.json').is_file():
                    path = verify_job(directory, job_binding)
                    console.say('[已完成] 本地结果哈希校验通过，复用结果。')
                else:
                    if (directory / 'started.json').exists() and not args.resume:
                        raise ValueError('An unfinished task exists. Reuse this command with --resume or choose a new --run-dir.')
                    write_json(directory / 'started.json', job)
                    path = execute_job(job, settings, directory, context, manifests, cases, results, args)
                    metric_rows(job, path)
                    seal_job(directory, path, job_binding)
                metric_rows(job, path)
                results.append((job, path))
                write_json(root / 'progress.json', dict(completed=len(results), total=len(jobs), current=job['id']))
            with console.phase(2, '汇总同一候选集上的逐疾病指标'):
                write_report(root, results, smoke=args.smoke)
            write_json(root / 'completion.json', dict(status='COMPLETE', jobs=len(jobs),
                scope='SYNTHETIC_TEST_ONLY' if args.smoke else 'NEW_LOCAL_EXPERIMENT', report='report.html',
                files={name: sha256(root / name) for name in ('summary.json', 'metrics.csv', 'disease_metrics.csv', 'report.html')}))
            console.say(f'[实验完成] {root / "report.html"}')
            return root
        except Exception as exc:
            console.failure(exc)
            exc.evihgt_reported = True
            raise
        finally:
            torch.set_num_threads(previous_threads)
