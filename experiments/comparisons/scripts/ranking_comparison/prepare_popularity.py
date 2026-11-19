"""Derive the two sets of training-only drug frequency counts."""
from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from historical_run import run_original

    raise SystemExit(run_original(Path(__file__)))
