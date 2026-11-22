"""Independent native processes, dispatched by the single RankingComparison CLI.

The parent never constructs a receiver. The child verifies the exact authority
again before data/model I/O and owns the common engineering session/worker lock.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT, read, sha, write_new

METHODS=('HGTDR_ADAPTED','TXGNN_ADAPTED')

def verify_coordinator_parent(method,coordinator_pid,*,rows=None):
    """Accept only the coordinator or its exact native venv redirector."""
    if rows is None:
        from .independent_launcher import windows_ancestors
        rows=windows_ancestors()
    if rows and rows[0]['pid']==coordinator_pid:return
    executable=worker_command(method,'test-only','unused')[0]
    if (len(rows)!=2 or rows[0]['parent_pid']!=coordinator_pid
            or rows[1]['pid']!=coordinator_pid
            or Path(rows[0]['executable']).resolve()!=Path(executable).resolve()):
        raise PermissionError('Native worker requires its exact live coordinator/venv parent chain')


def worker_command(method, action, path, *, grant=None, approved_digest=None, task_index=None):
    if method not in METHODS:raise ValueError('Unknown native method')
    executable=(ROOT.parent/('TxGNN' if method=='TXGNN_ADAPTED' else 'HGTDR'))/'.venv/Scripts/python.exe'
    command=[str(executable),'-m','evi_hgt_ranking_comparison.native_worker',action]
    if action=='test-only':command+=['--registration',str(path)]
    elif action=='formal':
        if not grant or not approved_digest or type(task_index) is not int:raise PermissionError('Exact new G3 authorization required')
        command+=['--manifest',str(path),'--grant',str(grant),'--approved-grant-digest',approved_digest,'--task-index',str(task_index)]
    else:raise ValueError('Unknown native action')
    return command


def _launch(command, log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    # M4 native CPU policy inherits the launch environment without failed
    # Part2.2 raw-order thread overrides. The Tx registration binds its values.
    with log.open('xb') as handle:
        import time
        started=time.monotonic();child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        mode='TEST_ONLY' if 'test-only' in command else 'G3'
        while child.poll() is None:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f'[{mode}] Native worker running for {time.monotonic()-started:.0f} s. Log: {log.resolve()}',flush=True)
    if '--registration' in command:
        spec=read(command[command.index('--registration')+1])
        if spec.get('family') in ('native23','native24'):
            from .common import append
            if spec['family']=='native24':
                from .runtime_budget24 import OUT24 as OUT23
            else:
                from .runtime_budget23 import OUT23
            append(OUT23/'TEST_ONLY/BUDGET_EVENTS.jsonl',dict(event='PROCESS_END',run_id=spec['run_id'],seconds=time.monotonic()-started,exit_code=child.returncode))
    if child.returncode:raise RuntimeError(f'Native worker exited with code {child.returncode}. Log: {log.resolve()}')
    return child.returncode


def launch_test_only(registration):
    spec=read(registration)
    if spec.get('family')=='native24':
        from .native_qualification24 import validate_registration
    elif spec.get('family')=='native23':
        from .native_qualification23 import validate_registration
    elif spec.get('family')=='native22':
        from .native_qualification22 import validate_registration
    else:
        from .native_qualification import validate_registration
    validate_registration(spec)
    part={'native24':'part2_4','native23':'part2_3','native22':'part2_2'}.get(spec.get('family'),'part2_1')
    dest=ROOT/f'outputs/ranking_protocol/ranking_comparison/{part}/TEST_ONLY'/spec['run_id']
    _launch(worker_command(spec['methods'][0],'test-only',registration),dest.parent/(spec['run_id']+'.worker.log'))
    return read(dest/'receipt.json')


def validate_formal_request(manifest,grant,approved_digest,task_index):
    from .authorization import require_formal_execution
    capability=require_formal_execution(manifest,grant,'G3',approved_grant_digest=approved_digest)
    if type(task_index) is not int or not 0<=task_index<len(manifest['training_tasks']):raise ValueError('Native task index invalid')
    task=manifest['training_tasks'][task_index]
    if task['method'] not in METHODS:raise ValueError('Not a native task')
    for path,value in manifest['source_sha256'].items():
        if sha(path)!=value:raise ValueError('Native source binding changed')
    identity={k:task[k] for k in ('run_id','method','cell','source_digest','data_digest')}
    capability.permits(dict(identity,mode='RANKING_COMPARISON_FORMAL'),task['output_root'])
    return task,capability


def launch_formal(manifest_path,grant_path,approved_digest,task_index):
    manifest=read(manifest_path);grant=read(grant_path) if grant_path else None
    task,_=validate_formal_request(manifest,grant,approved_digest,task_index)
    command=worker_command(task['method'],'formal',manifest_path,grant=grant_path,
        approved_digest=approved_digest,task_index=task_index)
    log=Path(task['output_root'])/('worker_'+__import__('uuid').uuid4().hex+'.log')
    _launch(command,log)
    return read(Path(task['output_root'])/'COMPLETE.json')


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='action',required=True)
    test=commands.add_parser('test-only');test.add_argument('--registration',required=True)
    formal=commands.add_parser('formal');formal.add_argument('--manifest',required=True);formal.add_argument('--grant')
    formal.add_argument('--approved-grant-digest');formal.add_argument('--task-index',type=int,required=True)
    args=parser.parse_args()
    if args.action=='test-only':
        spec=read(args.registration)
        if spec.get('family')=='native24':
            from .native_qualification24 import execute,validate_registration
        elif spec.get('family')=='native23':
            from .native_qualification23 import execute,validate_registration
        elif spec.get('family')=='native22':
            from .native_qualification22 import execute,validate_registration
        else:
            from .native_qualification import execute,validate_registration
        validate_registration(spec)
        expected=worker_command(spec['methods'][0],'test-only',args.registration)[0]
        if Path(sys.executable).resolve()!=Path(expected).resolve():raise PermissionError('Wrong native executable')
        result=execute(spec)
        return 0 if result['status']=='PASS' else 1
    task,capability=validate_formal_request(read(args.manifest),read(args.grant) if args.grant else None,args.approved_grant_digest,args.task_index)
    coordinator=ROOT/'outputs/ranking_protocol/ranking_comparison/runtime_worker.lock'
    if not coordinator.is_file():
        raise PermissionError('Native formal worker requires its live serial coordinator parent')
    verify_coordinator_parent(task['method'],read(coordinator).get('pid'))
    expected=worker_command(task['method'],'test-only','unused')[0]
    if Path(sys.executable).resolve()!=Path(expected).resolve():raise PermissionError('Wrong native executable')
    if task['method']=='TXGNN_ADAPTED':
        from evi_hgt.ranking_protocol.m4_amendment import verify_native_runtime
        verify_native_runtime()
    from .native_runtime import run_formal_native
    from .resource_guard import configure
    configure(read(args.manifest)['resource_limits'])
    run_formal_native(task,capability,read(args.manifest));return 0


if __name__=='__main__':raise SystemExit(main())
