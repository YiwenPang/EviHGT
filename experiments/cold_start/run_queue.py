"""Run the four authorized cold cases serially on one GPU; never retry training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from locations import CACHE, EXPERIMENT, SOURCE, WORKSPACE, check_frozen
from prepare_case import CASES, file_sha, write_json


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def status(row):
    p = EXPERIMENT/'queue_status.json'
    part = p.with_suffix('.json.partial')
    write_json(part,row)
    for i in range(4):
        try:
            os.replace(part,p)
            return
        except PermissionError:
            if i == 3: raise
            time.sleep(.3)


def child(file, case, *args):
    log = EXPERIMENT/'queue_logs'/f'{case}.{Path(file).stem}.log'
    log.parent.mkdir(exist_ok=True)
    env = {**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'}
    with log.open('xb') as handle:
        return subprocess.run([sys.executable,'-B','-u',str(SOURCE/file),'--case',case,*args],
                              cwd=WORKSPACE,stdout=handle,stderr=subprocess.STDOUT,env=env,check=False).returncode


def main():
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    queue = dict(status='RUNNING',pid=os.getpid(),started_at=stamp(),order=list(CASES),
                 cases={c:dict(status='PENDING') for c in CASES})
    # Persistent once marker, separate from the per-case runner's once markers.
    with (EXPERIMENT/'queue_started.json').open('x',encoding='utf8') as handle:
        json.dump(queue,handle,ensure_ascii=False,indent=2)
    status(queue)
    try:
        queue['frozen_initial'] = check_frozen()
        for case in CASES:
            row = queue['cases'][case]
            row['started_at'] = stamp()
            try:
                row['status'] = 'PREPARING'
                status(queue)
                if not (CACHE/case/'manifest.json').is_file():
                    code = child('prepare_case.py',case)
                    if code:
                        row.update(status='PREPARATION_FAILED',exit_code=code)
                        continue
                check_frozen()
                row['status'] = 'RUNNING_TRAIN_AND_TEST'
                status(queue)
                code = child('run_case.py',case,'--mode','train')
                row['exit_code'] = code
                result = EXPERIMENT/'runs'/case/'result.json'
                if code or not result.is_file():
                    row['status'] = 'RUN_FAILED'
                else:
                    result_row = json.loads(result.read_text(encoding='utf8'))
                    if result_row.get('status') != 'COMPLETE':
                        raise ValueError('Run finished without valid completion')
                    row.update(status='COMPLETE',result=str(result.relative_to(EXPERIMENT)),result_sha256=file_sha(result),metrics=result_row['metrics'])
            finally:
                row['finished_at'] = stamp()
                status(queue)
        queue['frozen_final'] = check_frozen()
        queue['status'] = 'COMPLETE' if all(v['status']=='COMPLETE' for v in queue['cases'].values()) else 'INCOMPLETE'
    except BaseException:
        (EXPERIMENT/'queue_error.log').write_text(traceback.format_exc(),encoding='utf8')
        queue['status'] = 'ERROR'
        raise
    finally:
        queue['finished_at'] = stamp()
        status(queue)
    return int(queue['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
