"""Conditional Part4 launch guard; verify-launch is read-only and never scores.

Only execute calls the frozen overnight runtime. Exception messages, source
lines, locals and scientific values are excluded from all failure output.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from historical_run import run_original
    raise SystemExit(run_original(Path(__file__)))

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from evi_hgt.evaluation.formal_redesign import digest, read_json

OUT = ROOT/'outputs/phase842_evaluable'
ACTION = 'FINAL_OVERNIGHT_EVALUABLE_FORMAL_EXECUTION'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normal_user():
    import ctypes
    require(os.name == 'nt' and not ctypes.windll.shell32.IsUserAnAdmin(),
        'normal Windows user required; administrator forbidden')


def frozen_preflight():
    from scripts.freeze_panels import preflight
    return preflight()


def validate_grant(grant, frozen, plan):
    require(plan['execution_authorized'] is False and plan['graph_batch_size'] == 1 and
        plan['execution_order'] == ['A','B','C','D'] and plan['formal_test_count'] == 1,
        'frozen serial plan or exactness contract differs')
    expected = dict(authorized=True, action=ACTION, formal_run_id=plan['formal_run_id'],
        formal_test_count=1, amendment_id=frozen['amendment_id'], plan_sha256=frozen['plan_sha256'],
        plan_digest=frozen['execution_plan_digest'], execution_plan_digest=frozen['execution_plan_digest'],
        freeze_sha256=frozen['freeze_sha256'], manifest_bindings=plan['manifest_bindings'],
        checkpoint_binding_digest=plan['checkpoint_binding_digest'], source_binding_digest=digest(plan['data_binding']),
        units=[u['unit_id'] for u in plan['units']], execution_order=['A','B','C','D'],
        scientific_result_visibility=False, automatic_retry=False)
    for key, value in expected.items():
        require(key in grant and grant[key] == value, 'conditional execution grant differs: '+key)
    require(grant['authorized'] is True and grant['scientific_result_visibility'] is False and
        grant['automatic_retry'] is False, 'execution grant booleans differ')


def verify_launch(resume=False):
    normal_user()
    frozen, plan, layer1, manifests = frozen_preflight()
    grant = read_json(OUT/'execution_authorization.json')
    validate_grant(grant, frozen, plan)
    marker = ROOT/plan['output_root']/'final_execution_access.json'
    require(marker.exists() == resume, 'fresh/resume entry mismatch; no automatic retry')
    print('FORMAL TEST COUNT = '+str(plan['formal_test_count']), flush=True)
    print('FORMAL RUN ID = '+plan['formal_run_id'], flush=True)
    print('AMENDMENT = '+frozen['amendment_id'], flush=True)
    for stage, value in plan['manifest_bindings'].items():
        print('STAGE '+stage+' MANIFEST DIGEST = '+value, flush=True)
    print('EXECUTION PLAN DIGEST = '+frozen['execution_plan_digest'], flush=True)
    print('[PHASE][GATE] FINAL_PRE_LAUNCH = PASS; SERIAL_A_B_C_D; SCIENTIFIC_VISIBILITY=HIDDEN', flush=True)
    return frozen, plan, grant


def safe_error(error):
    return dict(error_type=type(error).__name__,
        frames=[dict(file=Path(f.filename).name, function=f.name, line=f.lineno)
            for f in traceback.extract_tb(error.__traceback__)])


def sanitized_cli_error(parser, message):
    # The immutable primary CLI catches RuntimeError and passes its value here.
    # Preserve progress streams, while refusing to print any exception value.
    raise RuntimeError('Primary evaluation CLI rejected execution; details withheld')


def execute(resume=False):
    frozen, plan, grant = verify_launch(resume)
    try:
        from evi_hgt.evaluation.overnight_runtime import dispatch_main
        arguments = ['--mode','evaluate','--formal-evaluation-plan',str(ROOT/frozen['plan_path']),
            '--evaluation-authorization',str(OUT/'execution_authorization.json')]
        if resume:
            arguments.append('--resume')
        with patch.object(argparse.ArgumentParser, 'error', sanitized_cli_error):
            result = dispatch_main(arguments, project_root=ROOT)
        require(result in (None, 0), 'overnight runtime did not complete successfully')
        return 0
    except BaseException as error:
        from evi_hgt.evaluation.formal_runtime import atomic_json
        atomic_json(OUT/'launcher_error_status.json', dict(state='INTERRUPTED',
            formal_run_id=plan['formal_run_id'], formal_test_count=1,
            amendment_id=frozen['amendment_id'], plan_sha256=frozen['plan_sha256'],
            scientific_results_printed=False, scientific_results_inspected=False,
            automatic_retry=False, next_stage_started=False,
            timestamp_utc=datetime.now(timezone.utc).isoformat(), **safe_error(error)))
        print('[PHASE][GATE] FORMAL_EXECUTION = INTERRUPTED; error_type='+type(error).__name__+
            '; sanitized engineering receipt saved; no automatic retry', flush=True)
        return 2


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify-launch','execute'])
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.action == 'verify-launch':
            verify_launch(args.resume)
            return 0
        return execute(args.resume)
    except BaseException as error:
        print('[PHASE][GATE] FINAL_PRE_LAUNCH = BLOCKED; error_type='+type(error).__name__+
            '; no automatic launch or retry', flush=True)
        return 2
