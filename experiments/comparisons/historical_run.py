"""Run one registered comparison script with the original research sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


DIRECTORY = Path(__file__).resolve().parent


def run_original(public_script):
    parser = argparse.ArgumentParser(
        description="Run this comparison step in its source-verified research workspace.",
        allow_abbrev=False,
        add_help=False,
    )
    parser.add_argument("--research-root", required=True, type=Path, help="original research workspace")
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]) and not any(
        arg == "--research-root" or arg.startswith("--research-root=") for arg in sys.argv[1:]
    ):
        parser.print_help()
        return 0
    args, forwarded = parser.parse_known_args()
    workspace = args.research_root.resolve(strict=True)
    public = json.loads((DIRECTORY / "SOURCE_SHA256.json").read_text(encoding="utf-8"))
    name = public_script.resolve().relative_to(DIRECTORY).as_posix()
    expected = public["files"][name]["original_sha256"]

    sys.path.insert(0, str(DIRECTORY.parent))
    from replay import historical_sources

    with historical_sources() as archive:
        inventory = json.loads((archive / "comparisons" / "SOURCE_SHA256.json").read_text(encoding="utf-8"))
    original, = [path for path, digest in inventory["files"].items() if digest == expected]
    script = (workspace / original).resolve(strict=True)
    if not script.is_relative_to(workspace) or hashlib.sha256(script.read_bytes()).hexdigest() != expected:
        raise ValueError("Research script differs from the registered comparison source")
    return subprocess.run([sys.executable, str(script), *forwarded], cwd=workspace).returncode
