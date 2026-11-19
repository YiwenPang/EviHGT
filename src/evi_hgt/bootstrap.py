"""Standard-library startup reporting, available even when a binary dependency fails."""
from pathlib import Path
import sys
import traceback


def entry(*, ablation=False):
    try:
        from .cli import main
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError):
            message = f"Missing dependency: {exc.name}. Install requirements.txt with this Python interpreter."
        else:
            message = f"Dependency initialization failed ({type(exc).__name__}). Check the installed Python and PyTorch builds."
        print("[Error] " + message, file=sys.stderr)
        path = Path(__file__).resolve().parents[2] / "outputs/logs/startup_errors.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf8") as handle:
                traceback.print_exception(exc, file=handle)
            print(f"Full traceback: {path}", file=sys.stderr)
        except OSError:
            print("The traceback could not be saved; check output-directory permissions.", file=sys.stderr)
        return 2
    return main(ablation=ablation)
