"""Run EviHGT directly with Python or the PyCharm Run button."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


if __name__ == "__main__":
    if any(arg == "--experiment" or arg.startswith("--experiment=") for arg in sys.argv[1:]):
        from experiments.workflow.cli import main
        raise SystemExit(main())
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Paper experiments: main.py --experiment {ablations,losses,comparisons,cold,all} [--help]")
    from evi_hgt.bootstrap import entry
    raise SystemExit(entry())
