"""Run the public experiment checks on CPU using synthetic data and the published ledger."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "experiments" / "ablations" / "historical"


def main():
    suites = [
        ("Portable model and HGTDR-contra scoring and checkpoints", ROOT, [
            "tests", "experiments/hgtdr_contra/tests/test_run.py",
        ]),
        ("Ranking metrics and Popularity", ROOT, [
            "-o", "pythonpath=experiments/comparisons/src",
            "experiments/comparisons/tests/ranking_comparison",
        ]),
        ("Ablation protocols", ROOT, ["experiments/ablations/tests"]),
        ("Historical holdout analysis", HISTORICAL, [
            "-o", "pythonpath=src", "tests/test_results_analysis.py",
        ]),
        ("Cold-start splits and evaluation", ROOT, ["experiments/cold_start"]),
    ]
    with TemporaryDirectory(prefix="evihgt-public-checks-") as temporary:
        env = os.environ.copy()
        for name in ("PYTHONPATH", "EVIHGT_FROZEN_ROOT", "EVIHGT_RESEARCH_ROOT", "EVIHGT_D1_FEATURES"):
            env.pop(name, None)
        env.update(TMPDIR=temporary, TEMP=temporary, TMP=temporary, CUDA_VISIBLE_DEVICES="-1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        checks = [
            ("Five-fold comparison ledger", ROOT, ["experiments/audit/verify_r0.py"]),
            ("Published ablation source hashes", ROOT, ["experiments/ablations/verify.py", "sources"]),
        ]
        # The historical and portable packages each define evi_hgt. Give each suite a fresh interpreter.
        for index, (name, directory, arguments) in enumerate(suites):
            scratch = str(Path(temporary) / f"pytest-{index}")
            checks.append((name, directory, ["-m", "pytest", "-q", "-p", "no:cacheprovider",
                                              "--basetemp", scratch, *arguments]))
        checks.append(("Cold-start CPU training and held-out scoring", ROOT,
                       ["experiments/cold_start/synthetic/test_frozen_test_score.py"]))
        for name, directory, arguments in checks:
            print(f"\n{name}", flush=True)
            result = subprocess.run([sys.executable, "-B", *arguments], cwd=directory, env=env)
            if result.returncode:
                print(f"Failed: {name}", file=sys.stderr)
                return result.returncode
    print("\nAll public experiment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
