"""Run EviHGT ablations serially with common inputs and initialization."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


if __name__ == "__main__":
    from evi_hgt.bootstrap import entry
    raise SystemExit(entry(ablation=True))
