"""Dispatch a public entry point to its checksum-bound research script."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


DIRECTORY = Path(__file__).resolve().parent


def run_original(public_script):
    parser = argparse.ArgumentParser(
        description="Use the original script in the supplied EviHGT research workspace.",
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument("--research-root", required=True, type=Path)
    if "--research-root" not in sys.argv[1:] and not any(
        arg.startswith("--research-root=") for arg in sys.argv[1:]
    ) and any(arg in ("--help", "-h") for arg in sys.argv[1:]):
        parser.print_help()
        return 0
    args, forwarded = parser.parse_known_args()
    research_root = args.research_root.resolve(strict=True)
    public_manifest = json.loads((DIRECTORY / "SHA256.json").read_text(encoding="utf-8"))
    relative_path = public_script.resolve().relative_to(DIRECTORY).as_posix()
    item, = [row for row in public_manifest["files"] if row["file"] == relative_path]

    sys.path.insert(0, str(DIRECTORY.parent))
    from replay import historical_sources

    with historical_sources() as history:
        archived = json.loads((history / "ablations" / "SHA256.json").read_text(encoding="utf-8"))
        original, = [
            row for row in archived["files"]
            if row["lf_sha256"] == item["prior_release_lf_sha256"]
            and row["source_sha256"] == item["research_sha256"]
        ]
    target = (research_root / original["source"]).resolve(strict=True)
    if not target.is_relative_to(research_root) or hashlib.sha256(target.read_bytes()).hexdigest() != item["research_sha256"]:
        raise ValueError("Research script does not match the archived study source")
    return subprocess.run([sys.executable, str(target), *forwarded], cwd=research_root).returncode
