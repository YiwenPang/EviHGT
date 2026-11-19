"""Fixed four-arm single-worker queue, no retries or next-phase dispatch."""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from historical_run import run_original
    raise SystemExit(run_original(Path(__file__)))

from pathlib import Path
import argparse
import os
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import ablation_records as records
from evi_hgt_evidence_ablation.protocol import *
from evi_hgt_evidence_ablation.safety import no_conflicts,old_hold,exclusive_queue,awake
from evi_hgt_evidence_ablation.console import Console
from evi_hgt_evidence_ablation.relay import consume


def queue(resume=False):
    m=validate_manifest();no_conflicts();old_hold()
    if shutil.disk_usage(OUT).free<m['required_free_bytes']:raise OSError('EVIDENCE_ABLATION四条轨迹实际容量不足')
    if (OUT/'queue_started.json').exists() and not resume:raise FileExistsError('Queue already started; no duplicate launch')
    ui=Console(stream=sys.stderr,color=False);ui.banner(QUEUE,82482);rc=0
    try:
        with exclusive_queue(),awake():
            write(OUT/('queue_resume_'+str(time.time_ns())+'.json' if resume else 'queue_started.json'),dict(status='DEVELOPMENT_RUNNING',
                queue_pid=os.getpid(),supervisor_pid=os.getppid(),queue=[list(q) for q in QUEUE],GPU_workers=1,resume=resume,
                manifest_sha256=file_sha(MANIFEST),old_G3='HELD',outer='UNSCORED',new_final='SEALED'))
            for i,(cell,arm) in enumerate(QUEUE):
                no_conflicts();validate_manifest();ui.completed=i;root=OUT/cell/arm;root.mkdir(parents=True,exist_ok=True)
                ui.stage('固定开发队列',f'当前{cell}/{arm}；已完成{i}/4；预算{cell_spec(cell)["total"]:,}')
                command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',cell,arm]
                if resume and (root/'started.json').exists():command.append('--resume')
                def started(worker):write(OUT/'active_worker.json',dict(cell=cell,arm=arm,pid=worker.pid,queue_pid=os.getpid(),command=command),immutable=False)
                code=consume(command,ui,root/'stdout.log',cell=cell,on_start=started,render_log=OUT/'console_frames.jsonl')
                if code:raise RuntimeError(f'{cell}/{arm}停止，退出码{code}；后续未启动。日志：{root / "stdout.log"}')
                from evi_hgt_evidence_ablation.runner import verify_receipt
                verify_receipt(read(root/'receipt.json'),identity(m,cell,arm));ui.completed=i+1
                ui.stage('轨迹训练及全部许可评价完成',f'{cell}/{arm}；已完成{i+1}/4')
    except BaseException as exc:
        rc=1;(OUT/'queue_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
        write(OUT/('QUEUE_FAILED_'+str(time.time_ns())+'.json'),dict(status='PARTIAL_FAILED',error=repr(exc)))
        ui.failure('队列停止：'+str(exc),str(OUT/'queue_traceback.log'))
    finally:
        try:
            from evi_hgt_evidence_ablation.reporting import summarize
            result=summarize(failed=bool(rc),ui=ui)
            if result['status']==records.EVIDENCE_COMPLETE:write(OUT/'QUEUE_COMPLETE.json',dict(status=result['status'],manifest_sha256=file_sha(MANIFEST)))
        except BaseException as exc:
            rc=1;(OUT/'reporting_traceback.log').write_text(traceback.format_exc(),encoding='utf-8')
            write(OUT/records.EVIDENCE_JSON,dict(status='BLOCKED',error=repr(exc),old_G3='HELD',outer='UNSCORED',new_final='SEALED'),immutable=False)
            (OUT/records.EVIDENCE_MARKDOWN).write_text('# 证据消融汇总中止\n\n汇总核验失败，原记录保留。\n'+str(OUT/'reporting_traceback.log')+'\n',encoding='utf-8')
            ui.failure('最终汇总失败',str(OUT/'reporting_traceback.log'))
        ui.close()
    return rc


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--queue',action='store_true');g.add_argument('--worker',nargs=2);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.worker:
        from evi_hgt_evidence_ablation.runner import run_arm
        run_arm(*a.worker,resume=a.resume);return 0
    return queue(a.resume)
