"""Chinese progress display; structured events and tracebacks stay in log files."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
import traceback

from tqdm import tqdm


def number(value):
    return "N/A" if value is None else f"{value:.6g}"


class Console:
    def __init__(self, log_dir, *, stream=None, refresh_seconds=5):
        self.stream = sys.stdout if stream is None else stream
        if stream is None and callable(getattr(self.stream, "reconfigure", None)):
            self.stream.reconfigure(encoding="utf-8", errors="replace")
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.refresh = refresh_seconds
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events = (self.log_dir / "events.jsonl").open("a", encoding="utf8")
        self.error_path = self.log_dir / "errors.log"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.events.close()

    def say(self, text):
        print(text, file=self.stream, flush=True)

    def event(self, kind, **fields):
        self.events.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=kind, **fields), ensure_ascii=True, allow_nan=False) + "\n")
        self.events.flush()

    def banner(self):
        self.say("=" * 72)
        self.say("EviHGT | Evidence-aware Heterogeneous Graph Transformer")
        self.say("药物–疾病排序 | support 与 contra 独立预测")
        self.say("=" * 72)

    @contextmanager
    def phase(self, index, title):
        self.say(f"\n--- [阶段 {index}：{title}] ---")
        self.event("phase_started", phase=index, title=title)
        started = time.monotonic()
        try:
            yield
        except BaseException:
            self.event("phase_failed", phase=index, title=title)
            raise
        else:
            seconds = time.monotonic() - started
            self.say(f"[完成] {title}，用时 {seconds:.2f}s。")
            self.event("phase_completed", phase=index, title=title, seconds=seconds)

    @contextmanager
    def status(self, title):
        """Unknown total: update one elapsed line in a terminal, keep completion."""
        start = time.monotonic()
        stop = threading.Event()
        bar = tqdm(total=None, desc=title, file=self.stream, disable=not self.tty,
                   ascii=True, bar_format="{desc} | elapsed {elapsed}", leave=True)
        if not self.tty:
            self.say(f"[开始] {title}")
        def tick():
            while not stop.wait(self.refresh):
                bar.refresh()
        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            stop.set()
            thread.join()
            bar.close()
            self.say(f"[{'完成' if succeeded else '失败'}] {title}：{time.monotonic() - start:.2f}s。")

    def progress(self, total, title, *, unit="item"):
        return Progress(self, total, title, unit)

    def stage_progress(self, prefix="加载"):
        return StageProgress(self, prefix)

    def metrics(self, metrics):
        self.say("\n开发评估：已记录正例与未标注候选")
        self.say(f"{'Head':<12} {'ROC-AUC':>11} {'AP':>11} {'AUPR':>11} {'Diseases':>10}")
        for head in ("support", "contra"):
            m = metrics[head]
            self.say(f"{head:<12} {number(m.get('macro_auc')):>11} {number(m.get('macro_ap')):>11} {number(m.get('macro_aupr_trapezoid')):>11} {m.get('evaluable_diseases', 0):>10}")
        self.say(f"{'Head':<12} {'K':>3} {'Precision':>11} {'Recall':>11} {'Hits':>11} {'NDCG':>11}")
        for head in ("support", "contra"):
            for k in (5, 10):
                r = metrics[head].get("ranking", {})
                values = " ".join(f"{number(r.get(f'{key}@{k}')):>11}" for key in ("P", "R", "Hits", "NDCG"))
                self.say(f"{head:<12} {k:>3} {values}")
        self.say("N/A 表示不可用。AP 使用阶梯积分，AUPR 使用梯形积分。")
        self.event("evaluation_completed", metrics=metrics)

    def failure(self, exc):
        with self.error_path.open("a", encoding="utf8") as handle:
            traceback.print_exception(exc, file=handle)
        self.event("failure", type=type(exc).__name__, message=str(exc))
        self.say(f"[Error] 运行失败（{type(exc).__name__}）：{exc}")
        self.say(f"完整错误日志：{self.error_path}")


class Progress:
    def __init__(self, console, total, title, unit):
        self.console, self.total, self.title = console, total, title
        self.n, self.fields = 0, {}
        self.started = self.last = time.monotonic()
        self.bar = tqdm(total=total, desc=title, unit=unit, file=console.stream,
                        disable=not console.tty, ascii=True, dynamic_ncols=True,
                        mininterval=console.refresh, maxinterval=console.refresh, leave=True)
        if not console.tty:
            console.say(f"[开始] {title}：{total} {unit}")
        self.stop = threading.Event()
        def refresh_elapsed():
            while not self.stop.wait(console.refresh):
                if console.tty:
                    self.bar.refresh()
        self.thread = threading.Thread(target=refresh_elapsed, daemon=True)
        self.thread.start()

    def __enter__(self):
        return self

    def update(self, count=1, **fields):
        self.n += count
        self.fields.update(fields)
        shown = {k: number(v) if isinstance(v, (int, float)) else v for k, v in self.fields.items()}
        self.bar.set_postfix(shown, refresh=False)
        self.bar.update(count)
        now = time.monotonic()
        if not self.console.tty and now - self.last >= max(60, self.console.refresh):
            self._summary(now)
        self.last = now if self.console.tty else self.last

    def _summary(self, now):
        elapsed = now - self.started
        speed = self.n / elapsed if elapsed else 0
        eta = (self.total - self.n) / speed if speed else None
        fields = " | ".join(f"{k}={number(v) if isinstance(v, (float, int)) else v}" for k, v in self.fields.items())
        self.console.say(f"{self.title}: {self.n}/{self.total} | {speed:.2f}/s | ETA {number(eta)}s | {fields}")
        self.last = now

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.bar.close()
        if not self.console.tty:
            self._summary(time.monotonic())


class StageProgress:
    """Adapt data callbacks; total=0 means elapsed status with no percentage."""
    def __init__(self, console, prefix):
        self.console, self.prefix = console, prefix
        self.current, self.stage, self.total = None, None, None

    def __enter__(self):
        return self

    def __call__(self, stage, completed, total, detail=0):
        if stage != self.stage:
            if self.current is not None:
                self.current.__exit__(None, None, None)
            title = self.prefix + " " + stage
            if total == 0:
                self.current = self.console.status(title)
                self.current.__enter__()
            else:
                self.current = self.console.progress(total, title)
            self.stage, self.total = stage, total
        if self.total:
            self.current.update(completed - self.current.n,
                                **({"memory_MiB": detail / 2**20} if detail else {}))

    def __exit__(self, *exc):
        if self.current is not None:
            self.current.__exit__(*exc)
