"""Sequential supervisor for the paired development runs."""

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
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import ablation_records as records
from evi_hgt_loss_comparison.protocol import *
from evi_hgt_loss_comparison.safety import no_conflicts,old_hold,exclusive_queue,awake
from evi_hgt_loss_comparison.console import Console

EVENTS={"stage","preparation_progress","training_start","update","checkpoint",
        "evaluation_start","evaluation_update","evaluation_result","failure"}


def queue(resume=False):
    m=validate_manifest();no_conflicts();old_hold()
    if shutil.disk_usage(OUT).free<m["required_free_bytes"]:raise OSError("Insufficient free space")
    if (OUT/"queue_started.json").exists() and not resume:raise FileExistsError("The loss comparison queue has already started")
    ui=Console(stream=sys.stderr,color=False);rc=0;worker=None
    ui.banner(ARMS,TOTAL)
    try:
        with exclusive_queue(),awake():
            write(OUT/("queue_resume_"+str(time.time_ns())+".json" if resume else "queue_started.json"),
                dict(status="DEVELOPMENT_RUNNING",queue_pid=os.getpid(),supervisor_pid=os.getppid(),
                     manifest_sha256=file_sha(MANIFEST),arms=list(ARMS),GPU_workers=1,resume=resume,
                     old_G3="HELD",outer="UNSCORED",new_final="SEALED"))
            for arm in ARMS:
                no_conflicts();validate_manifest()
                dest=OUT/arm;dest.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,"-B","-u",str(Path(__file__).resolve()),"--worker",arm]
                if resume and (dest/"started.json").exists():command.append("--resume")
                with (dest/"stdout.log").open("a",encoding="utf-8") as log:
                    worker=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        text=True,encoding="utf-8",errors="replace",bufsize=1)
                    write(OUT/"active_worker.json",dict(arm=arm,pid=worker.pid,queue_pid=os.getpid(),command=command),immutable=False)
                    for line in worker.stdout:
                        log.write(line);log.flush()
                        try:event=json.loads(line)
                        except ValueError:continue
                        try:
                            if event.get("event") in EVENTS and isinstance(event.get("args"),list):
                                getattr(ui,event["event"])(*event["args"])
                        except Exception:
                            with (OUT/"console_render_errors.log").open("a",encoding="utf-8") as f:f.write(traceback.format_exc())
                    code=worker.wait();worker=None
                if code:raise RuntimeError(f"{arm} exited with code {code}. See its run log.")
                from evi_hgt_loss_comparison.runner import verify_receipt
                verify_receipt(read(dest/"receipt.json"),identity(m,arm))
    except BaseException as exc:
        rc=1
        # Do not kill any process. A worker owns its fail-closed exception path.
        (OUT/"queue_traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        write(OUT/("QUEUE_FAILED_"+str(time.time_ns())+".json"),dict(status="PARTIAL_FAILED",error=repr(exc)))
        ui.failure("Development queue stopped: "+str(exc),str(OUT/"queue_traceback.log"))
    finally:
        from evi_hgt_loss_comparison.reporting import summarize
        try:
            result=summarize(failed=bool(rc),ui=ui)
            if result["status"]==records.LOSS_COMPLETE:
                write(OUT/"QUEUE_COMPLETE.json",dict(status=result["status"],manifest_sha256=file_sha(MANIFEST)))
        except BaseException as exc:
            rc=1;(OUT/"reporting_traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
            # Never label a failed verification as COMPLETE.
            write(OUT/records.LOSS_JSON,dict(status="BLOCKED",error=repr(exc),old_G3="HELD",
                  outer="UNSCORED",new_final="SEALED"),immutable=False)
            (OUT/records.LOSS_MARKDOWN).write_text("# Loss comparison summary interrupted\n\nSummary verification failed. Run records are available.\n\n"+
                str(OUT/"reporting_traceback.log")+"\n",encoding="utf-8")
            ui.failure("Summary verification failed",str(OUT/"reporting_traceback.log"))
            ui._write("FINAL STATUS = BLOCKED")
            ui._write(str(OUT/records.LOSS_MARKDOWN));ui._write(str(OUT/records.LOSS_JSON))
        ui.close()
    return rc


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--queue",action="store_true");g.add_argument("--worker",choices=ARMS);g.add_argument("--verify",action="store_true")
    p.add_argument("--resume",action="store_true");a=p.parse_args()
    if a.verify:
        validate_manifest();no_conflicts();old_hold();print("Loss comparison manifest and launch conditions verified.",flush=True);return 0
    if a.worker:
        from evi_hgt_loss_comparison.runner import run_arm
        run_arm(a.worker,resume=a.resume);return 0
    return queue(a.resume)
