"""Standard-library startup reporting, available even when a binary dependency fails."""
from pathlib import Path
import sys
import traceback


def entry(*, ablation=False):
    # Windows redirected streams may otherwise use an encoding without Chinese.
    for stream in (sys.stdout, sys.stderr):
        if callable(getattr(stream, "reconfigure", None)):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        from .cli import main
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError):
            message = f"缺少依赖 {exc.name}，请用当前 Python 安装 requirements-preprocess.txt。"
        else:
            message = f"依赖初始化失败（{type(exc).__name__}），请检查 Python 和 PyTorch 安装。"
        print("[Error] " + message, file=sys.stderr)
        path = Path(__file__).resolve().parents[2] / "outputs/logs/startup_errors.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf8") as handle:
                traceback.print_exception(exc, file=handle)
            print(f"完整错误日志：{path}", file=sys.stderr)
        except OSError:
            print("错误日志写入失败，请检查输出目录权限。", file=sys.stderr)
        return 2
    return main(ablation=ablation)
