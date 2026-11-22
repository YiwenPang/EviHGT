"""Read-only five-point LR artifact report. Never instantiate or execute a model."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from evi_hgt.protocols.formal_experiments import (
    load_formal_experiment_manifest, execution_identity, _canonical_digest,
)
from evi_hgt.training.checkpoint import read_training_checkpoint
from evi_hgt.training.formal_checkpoint import FormalCheckpointSnapshot
from evi_hgt.training.model_selection import (
    LEARNING_RATE_CANDIDATES, model_selection_protocol_payload, select_learning_rate,
)

CSV_FIELDS = ('learning_rate', 'cycle1_validation_objective', 'cycle2_validation_objective',
              'best_objective', 'best_cycle', 'wall_clock_seconds', 'wall_clock', 'run_id', 'status')
SELECTION_PATH = Path('outputs/phase8_formal/tuning/selected_lr.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tuning_runs(root):
    manifest = load_formal_experiment_manifest(root)
    runs = sorted((r for r in manifest['runs'] if r['run_family'] == 'lr_tuning'
                   and r['learning_rate'] in LEARNING_RATE_CANDIDATES),
                  key=lambda r: r['learning_rate'])
    require(tuple(r['learning_rate'] for r in runs) == LEARNING_RATE_CANDIDATES,
            'Exactly five frozen LR candidates required')
    return runs


def check_fresh(root, run):
    require(run['execution_status'] == 'PLANNED', 'Fresh run requires PLANNED status')
    attempt, = run['execution_attempts']
    require(all(attempt.get(k) == v for k, v in execution_identity(run['run_id'], 1).items()),
            'Fresh execution identity mismatch')
    require(attempt['status'] == 'PLANNED' and attempt.get('fresh_start_only') is True
            and attempt.get('resumed') is False and attempt.get('formal_test_count') == 0
            and all(attempt.get(k) == 0 for k in (
                'observed_optimizer_steps', 'recoverable_optimizer_steps', 'completed_epochs'))
            and not attempt.get('best_exists') and not attempt.get('last_exists'),
            'Fresh execution requires sequence 1, zero counters and no checkpoints')
    expected = f'outputs/phase8_formal/tuning/{run["run_id"]}'
    require(attempt['output_directory'] == expected, 'Fresh run directory identity mismatch')
    require(not (Path(root)/expected).exists(), 'Fresh run directory already exists; STOP')


def _finite_state(value):
    import torch
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor; invalid run')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_state(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite checkpoint scalar; invalid run')


def _read_snapshot(path, run, attempt, sources):
    # Only trusted local EviHGT checkpoint artifacts; deserialize on CPU, no restore.
    digest = sha256(path)
    payload = read_training_checkpoint(path, map_location='cpu')
    snapshot = FormalCheckpointSnapshot.from_payload(payload['formal_checkpoint_state'])
    identity = snapshot.identity
    expected = {
        'formal_run_id': run['run_id'], 'execution_id': attempt['execution_id'],
        'model_variant': 'FULL_EVIHGT', 'policy': 'B', 'split': 'standard', 'seed': 42,
        'learning_rate': run['learning_rate'], 'data_digest': run['data_digest'],
        'architecture_config_digest': run['architecture_digest'],
        'learning_protocol_digest': run['learning_protocol_digest'],
        'candidate_protocol_digest': run['candidate_protocol_digest'],
        'evaluation_protocol_digest': run['evaluation_protocol_digest'],
        'objective_id': run['learning_protocol']['objective_id'],
        'training_protocol_digest': _canonical_digest(run['formal_training_protocol']['formal_training']),
    }
    require(all(getattr(identity, k) == v for k, v in expected.items()), 'Checkpoint run identity mismatch')
    require(payload['metadata']['artifact_binding']['formal_checkpoint_identity'] == identity.to_payload()
            and payload['engine_state'] == snapshot.engine_state, 'Checkpoint metadata identity mismatch')
    runtime = payload['metadata']['runtime_config']
    require(str(runtime['device']).startswith('cuda') and runtime['precision'] == 'bf16'
            and runtime['learning_rate'] == run['learning_rate'] and runtime['weight_decay'] == 1e-5
            and runtime['gradient_accumulation_steps'] == 1 and runtime['max_grad_norm'] == 1.0
            and runtime['scheduler'] == 'none', 'Checkpoint training protocol mismatch')
    require(payload['scheduler_state_dict'] is None and not payload['pending_gradients']
            and payload.get('amp_scaler_state_dict') is None, 'Unexpected scheduler/scaler/pending state')
    groups = payload['optimizer_state_dict']['param_groups']
    require(bool(groups) and all(g['lr'] == run['learning_rate'] and g['weight_decay'] == 1e-5 for g in groups),
            'Optimizer identity mismatch')
    require(bool(payload['optimizer_state_dict']['state']), 'Optimizer state missing')
    _finite_state(payload['model_state_dict']); _finite_state(payload['optimizer_state_dict'])
    require(snapshot.global_optimizer_step == 9380 * snapshot.last_completed_epoch,
            'Incomplete standard positive coverage cycle')
    sources[str(path)] = digest
    return snapshot


def read_candidate(root, run):
    root = Path(root)
    require(run['execution_status'] == 'COMPLETED', 'All five runs must be COMPLETE / VALID')
    require(len(run.get('execution_attempts', [])) == 1, 'Exactly one fresh execution required')
    attempt = run['execution_attempts'][0]
    require(all(attempt.get(k) == v for k, v in execution_identity(run['run_id'], 1).items()),
            'Execution identity mismatch')
    require(attempt['status'] == 'COMPLETED' and attempt['fresh_start_only'] is True
            and attempt['resumed'] is False and attempt['formal_test_count'] == 0
            and attempt['completed_epochs'] == 2 and attempt['observed_optimizer_steps'] == 18760
            and attempt['recoverable_optimizer_steps'] == 18760 and attempt['best_exists'] and attempt['last_exists'],
            'COMPLETE / VALID fresh two-cycle execution with formal test count 0 required')
    relative = f'outputs/phase8_formal/tuning/{run["run_id"]}'
    require(attempt['output_directory'] == relative, 'Run directory identity mismatch')
    directory = root/relative
    checkpoints = directory/'checkpoints'
    sources = {}
    last = _read_snapshot(checkpoints/'last.pt', run, attempt, sources)
    best = _read_snapshot(checkpoints/'best.pt', run, attempt, sources)
    require(last.checkpoint_role == 'LAST' and last.last_completed_epoch == 2
            and last.early_stopping_state.should_stop and best.checkpoint_role == 'BEST'
            and last.identity == best.identity, 'BEST/LAST completion or identity mismatch')
    binding = last.best_artifact
    require(binding and Path(binding['filename']).name == binding['filename'], 'Invalid committed BEST path')
    require(binding['filename'] == f'best.{best.logical_checkpoint_digest}.pt'
            and sha256(checkpoints/binding['filename']) == binding['sha256']
            and sources[str(checkpoints/'best.pt')] == binding['sha256'], 'Committed BEST binary mismatch')
    cycle1 = []
    for file in sorted(checkpoints.glob('best.*.pt')):
        snapshot = _read_snapshot(file, run, attempt, sources)
        require(snapshot.identity == last.identity and snapshot.checkpoint_role == 'BEST'
                and file.name == f'best.{snapshot.logical_checkpoint_digest}.pt', 'Immutable BEST identity mismatch')
        if snapshot.last_completed_epoch == 1:
            cycle1.append(snapshot)
    require(len(cycle1) == 1, 'Exactly one cycle 1 immutable BEST artifact required')
    values = (cycle1[0].validation_objective, last.validation_objective)
    best_cycle = min((1, 2), key=lambda epoch: (values[epoch-1], epoch))
    require(best.last_completed_epoch == best_cycle and best.validation_objective == min(values)
            and last.best_epoch == best_cycle and last.best_validation_objective == min(values),
            'BEST differs from exact minimum / earliest exact cycle tie')
    require(last.early_stopping_state.best_checkpoint.checkpoint_digest == best.logical_checkpoint_digest,
            'BEST logical selection mismatch')
    log_path = directory/'logs/run.log'
    log = log_path.read_text(encoding='utf-8')
    sources[str(log_path)] = sha256(log_path)
    require(not re.search(r'\| (ERROR|CRITICAL) \||Traceback|non.?finite|interrupted', log, re.I),
            'Fatal or interrupted run log; STOP')
    starts = [line for line in log.splitlines() if 'Formal run started ' in line]
    ends = [line for line in log.splitlines() if 'Formal run completed ' in line]
    require(len(starts) == len(ends) == 1 and all(f'run_id={run["run_id"]}' in line for line in starts+ends)
            and 'start_epoch=1 checkpoint_action=START_LEGITIMATE_INITIAL_RUN' in starts[0]
            and 'reason=MAX_TRAINING_HORIZON_REACHED epoch=2 optimizer_step=18760' in ends[0],
            'Complete fresh run log required')
    epochs = [line for line in log.splitlines() if 'Formal epoch=' in line]
    require(len(epochs) == 2, 'Exactly two cycle validation log records required')
    for epoch, line in enumerate(epochs, 1):
        require(re.search(rf'Formal epoch={epoch}\s', line) and
                f'validation_objective={values[epoch-1]:.9f}' in line and
                re.search(rf'microbatches=9380 optimizer_step={9380*epoch}(?:\s|$)', line),
                'Cycle log/checkpoint objective or coverage mismatch')
    elapsed = (datetime.fromisoformat(ends[0][:19])-datetime.fromisoformat(starts[0][:19])).total_seconds()
    require(elapsed > 0, 'Invalid wall-clock interval')
    return dict(learning_rate=run['learning_rate'], cycle1_validation_objective=values[0],
        cycle2_validation_objective=values[1], best_objective=min(values), best_cycle=best_cycle,
        wall_clock_seconds=int(elapsed), wall_clock=f'{int(elapsed)//3600:02d}:{int(elapsed)%3600//60:02d}:{int(elapsed)%60:02d}',
        run_id=run['run_id'], status='COMPLETE / VALID', source_artifacts=sources,
        validation_manifest_digest=last.identity.validation_manifest_digest)


def plot_results(csv_path, directory):
    """The CSV is the sole numeric source for both figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    with Path(csv_path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    lrs = [float(r['learning_rate']) for r in rows]
    require(tuple(lrs) == LEARNING_RATE_CANDIDATES, 'Figure requires all five CSV candidates')
    with plt.rc_context({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8, 'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':.8, 'legend.frameon':False}):
        specs = [('lr_tuning_best_objective', [('best_objective','BEST','#355C7D','o')]),
                 ('lr_tuning_cycle_objectives', [('cycle1_validation_objective','Cycle 1','#355C7D','o'),
                                                ('cycle2_validation_objective','Cycle 2','#777777','s')])]
        for name, series in specs:
            fig, ax = plt.subplots(figsize=(4.5,3.2), layout='constrained')
            for field,label,color,marker in series:
                ax.plot(lrs,[float(r[field]) for r in rows],label=label,color=color,marker=marker,
                        markersize=4,linewidth=1.1)
            ax.set_xscale('log')
            ax.set_xticks(lrs, [f'{lr:.0e}' for lr in lrs])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel('Learning rate')
            ax.set_ylabel('BEST deterministic validation objective' if len(series)==1 else
                          'Deterministic validation objective')
            ax.grid(axis='y',color='#dddddd',linewidth=.5)
            if len(series)>1:
                ax.legend()
            fig.savefig(Path(directory)/f'{name}.png',dpi=600)
            fig.savefig(Path(directory)/f'{name}.pdf')
            plt.close(fig)


def generate_report(root, output_directory):
    root, output = Path(root).resolve(), Path(output_directory).resolve()
    runs = tuning_runs(root)
    amendment = load_formal_experiment_manifest(root).get('lr_search_amendment')
    require(amendment is not None, 'Lower-LR refinement provenance required')
    # Check completeness before reading large checkpoints or creating any output.
    require(all(r['execution_status']=='COMPLETED' for r in runs), 'All five runs must be COMPLETE / VALID')
    rows = [read_candidate(root, r) for r in runs]
    require(len({r['validation_manifest_digest'] for r in rows}) == 1, 'Validation manifest binding differs')
    decision = select_learning_rate(validation_objectives={r['learning_rate']:r['best_objective'] for r in rows})
    selected = {**decision.to_payload(), 'candidate_learning_rates':list(LEARNING_RATE_CANDIDATES),
        'candidate_best_objectives':{str(r['learning_rate']):r['best_objective'] for r in rows},
        'candidate_results':[{k:r[k] for k in CSV_FIELDS} for r in rows],
        'search_plan_amendment':amendment,
        'selection_criterion':model_selection_protocol_payload(),
        'tie_rule':'ABSOLUTE_DIFFERENCE_FROM_GLOBAL_MINIMUM_LE_1E-8_THEN_SMALLER_LR',
        'source_run_ids':{str(r['learning_rate']):r['run_id'] for r in rows}, 'formal_test_count':0,
        'source_csv_sha256':None}
    selection_path = root/SELECTION_PATH
    require(not output.exists() and not selection_path.exists(), 'Report or selected LR artifact already exists; refusing overwrite')
    require(not output.is_relative_to(root/'data') and not output.is_relative_to(root/'outputs/phase8_formal'),
            'Paper output must be separate from raw data and formal run artifacts')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.lr-report-',dir=output.parent) as temporary:
        staging = Path(temporary)/'report'; staging.mkdir()
        csv_path = staging/'lr_tuning_results.csv'
        with csv_path.open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=CSV_FIELDS,extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)
        selected['source_csv_sha256'] = sha256(csv_path)
        (staging/'lr_tuning_results.json').write_text(json.dumps({'protocol':model_selection_protocol_payload(),
            'search_plan_amendment':amendment,
            'formal_test_count':0,'wall_clock_definition':'PRODUCTION_RUN_STARTED_TO_COMPLETED_LOG_SECONDS_EXCLUDES_STARTUP',
            'results':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        table = '\n'.join(f'| {r["learning_rate"]:.0e} | {r["cycle1_validation_objective"]:.9f} | '
            f'{r["cycle2_validation_objective"]:.9f} | {r["best_objective"]:.9f} | {r["best_cycle"]} | {r["wall_clock"]} |' for r in rows)
        text = f'''# Learning-rate sensitivity

## Learning-rate settings
Of the first three completed rates (1e-4, 3e-4 and 1e-3), 1e-4 had the lowest validation objective. Its second cycle improved on the first.
That rate was at the lower end of the initial search, so the study added 2e-5 and 5e-5.
The final comparison covers 2e-5, 5e-5, 1e-4, 3e-4 and 1e-3. The two lower rates were added after the initial search.
Each candidate used an independent initialization with seed 42. The run at 2e-4 was interrupted by a file replacement error, and its first cycle was excluded from selection.
The planned 5e-4 run was dropped. Both records are retained. The two-cycle search did not extend to 1e-5.

## Training protocol
The study used the frozen PRACTICAL_FORMAL_V1 protocol with FULL_EVIHGT, Policy B and the standard split.
Each query used up to 1,000 nodes, 5,000 traversals and two hops. The path search kept up to 16 paths of length 2–4 with beam width 32 and 64 neighbors.
AdamW used a weight decay of 1e-5, a gradient accumulation factor of 1 and gradient clipping at 1.0. The learning rate stayed constant.
The run used CUDA BF16 autocast and FP32 pairwise reduction with the gradient scaler disabled.
The objective was SYMMETRIC_PU_PAIRWISE_RANKING_V1 at temperature 1, with chunk size 4096 and hard-negative sampling disabled.
Each run covered all training positives twice, with one deterministic validation pass at the end of each cycle.

## Development results
| LR | Cycle 1 | Cycle 2 | BEST | BEST cycle | Wall-clock |
|---|---:|---:|---:|---:|---|
{table}

The table rounds values for display. The CSV and JSON retain the precision used for selection. Reported duration spans training start to completion, excluding startup.
Figure 1 plots the minimum validation objective, and Figure 2 shows both cycles. Both use the same CSV and a logarithmic learning-rate axis.

## Selection
Each rate is scored by its lower validation objective across the two cycles. A tie within a run goes to the earlier cycle.
If a candidate is within 1e-8 of the global minimum, the smallest qualifying rate is selected.
The prespecified rule selected {decision.selected_learning_rate:.0e} by the validation objective.
The test set stayed sealed throughout learning-rate selection.

## Interpretation
The scores describe learning-rate sensitivity at a fixed seed and fixed graph and path budgets over two training cycles.
The selected rate applies to this two-cycle validation protocol.
The study reports descriptive validation differences.
'''
        (staging/'LR_TUNING_REPORT.md').write_text(text,encoding='utf-8')
        plot_results(csv_path,staging)
        # Refuse publication if any source changed while reading/rendering.
        for row in rows:
            for path,digest in row['source_artifacts'].items():
                require(sha256(path)==digest, 'Source artifact changed during report generation')
        require(tuning_runs(root)==runs, 'Manifest changed during report generation')
        staging.rename(output)
    selection_path.parent.mkdir(parents=True,exist_ok=True)
    with selection_path.open('x',encoding='utf-8') as stream:
        json.dump(selected,stream,ensure_ascii=False,indent=2); stream.write('\n')
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs'/'lr_tuning_report')
    parser.add_argument('--check-run',help='Read-only artifact/freshness check of one exact tuning ID')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--require-fresh',action='store_true')
    group.add_argument('--require-complete',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.check_run:
            run=next((r for r in tuning_runs(ROOT) if r['run_id']==args.check_run),None)
            require(run is not None,'Unknown tuning run ID')
            if args.require_fresh:
                check_fresh(ROOT,run)
                result={'run_id':run['run_id'],'status':'PLANNED / FRESH / NOT STARTED'}
            else:
                result=read_candidate(ROOT,run)
            print(json.dumps(result,ensure_ascii=True,indent=2)); return 0
        require(not args.require_fresh and not args.require_complete,'--check-run required')
        print(json.dumps(generate_report(ROOT,args.output_dir),ensure_ascii=True,indent=2)); return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        print(f'LR report STOP: {exc}',file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
