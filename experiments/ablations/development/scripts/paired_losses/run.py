"""One autonomous local queue; no automatic training retries or next phase."""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from historical_run import run_original
    raise SystemExit(run_original(Path(__file__)))

from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import ablation_records as records
from evi_hgt_paired_losses.protocol import *
from evi_hgt_paired_losses.safety import no_conflicts,old_hold,exclusive_queue,awake
from evi_hgt_paired_losses.console import Console
EVENTS={'stage','preparation_progress','training_start','update','checkpoint','evaluation_start','evaluation_update','evaluation_result','failure'}


def queue(resume=False):
    m=validate_manifest();no_conflicts();old_hold()
    if shutil.disk_usage(OUT).free<m['required_free_bytes']:raise OSError('本轮四条轨迹所需磁盘空间不足')
    if (OUT/'queue_started.json').exists() and not resume:raise FileExistsError('PAIRED_LOSSES队列已启动')
    ui=Console(stream=sys.stderr,color=False);ui.banner(QUEUE,82482);rc=0
    try:
        with exclusive_queue(),awake():
            write(OUT/('queue_resume_'+str(time.time_ns())+'.json' if resume else 'queue_started.json'),
                dict(status='DEVELOPMENT_RUNNING',queue_pid=os.getpid(),supervisor_pid=os.getppid(),manifest_sha256=file_sha(MANIFEST),
                     queue=[list(q) for q in QUEUE],GPU_workers=1,resume=resume,old_G3='HELD',outer='UNSCORED',new_final='SEALED'))
            for i,(cell,arm) in enumerate(QUEUE):
                no_conflicts();validate_manifest();root=OUT/cell/arm;root.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',cell,arm]
                if resume and (root/'started.json').exists():command.append('--resume')
                ui.stage('固定开发队列',f'当前 {cell}/{arm}；完成轨迹数 {i}/4；预算{cell_spec(cell)["total"]}')
                with (root/'stdout.log').open('a',encoding='utf-8') as log:
                    worker=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
                    write(OUT/'active_worker.json',dict(cell=cell,arm=arm,pid=worker.pid,queue_pid=os.getpid(),command=command),immutable=False)
                    for line in worker.stdout:
                        log.write(line);log.flush()
                        try:event=json.loads(line)
                        except ValueError:continue
                        try:
                            if event.get('event') in EVENTS and isinstance(event.get('args'),list):
                                args=event['args']
                                if event['event'] in ('evaluation_start','evaluation_result'):args[0]=cell+'/'+args[0]
                                getattr(ui,event['event'])(*args)
                        except Exception:
                            with (OUT/'console_render_errors.log').open('a',encoding='utf-8') as f:f.write(traceback.format_exc())
                    code=worker.wait()
                if code:raise RuntimeError(f'{cell}/{arm}已停止（退出码{code}），后续未启动；详见{root / "stdout.log"}')
                from evi_hgt_paired_losses.runner import verify_receipt
                verify_receipt(read(root/'receipt.json'),identity(m,cell,arm))
                ui.stage('开发轨迹完成',f'{cell}/{arm}；完成轨迹数 {i+1}/4')
    except BaseException as exc:
        rc=1;(OUT/'queue_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        write(OUT/('QUEUE_FAILED_'+str(time.time_ns())+'.json'),dict(status='PARTIAL_FAILED',error=repr(exc)))
        ui.failure('开发队列停止：'+str(exc),str(OUT/'queue_traceback.log'))
    finally:
        try:
            from evi_hgt_paired_losses.reporting import summarize
            result=summarize(failed=bool(rc),ui=ui)
            if result['status']==records.PAIRED_COMPLETE:write(OUT/'QUEUE_COMPLETE.json',dict(status=result['status'],manifest_sha256=file_sha(MANIFEST)))
        except BaseException as exc:
            rc=1;(OUT/'reporting_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
            write(OUT/records.PAIRED_JSON,dict(status='BLOCKED',error=repr(exc),old_G3='HELD',outer='UNSCORED',new_final='SEALED'),immutable=False)
            (OUT/records.PAIRED_MARKDOWN).write_text('# 跨划分损失比较汇总中止\n\n汇总身份验证失败；原始记录保留。\n\n'+str(OUT/'reporting_traceback.log')+'\n',encoding='utf-8')
            ui.failure('最终汇总验证失败',str(OUT/'reporting_traceback.log'));ui._write('FINAL STATUS = BLOCKED')
            ui._write(str(OUT/records.PAIRED_MARKDOWN));ui._write(str(OUT/records.PAIRED_JSON))
        ui.close()
    return rc


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--queue',action='store_true');g.add_argument('--worker',nargs=2,metavar=('CELL','ARM'));g.add_argument('--verify',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.verify:
        validate_manifest();no_conflicts();old_hold();print('PAIRED_LOSSES清单及启动边界通过。',flush=True);return 0
    if a.worker:
        from evi_hgt_paired_losses.runner import run_arm
        run_arm(*a.worker,resume=a.resume);return 0
    return queue(a.resume)
