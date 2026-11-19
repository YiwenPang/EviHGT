"""Check published ablation source or verify the original study records."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


DIRECTORY = Path(__file__).resolve().parent


def verify_public_sources():
    manifest = json.loads((DIRECTORY / "SHA256.json").read_text(encoding="utf-8"))
    expected = [(item["file"], item["published_sha256"]) for item in manifest["files"]]
    expected.extend(manifest["public_helpers"].items())
    entries = [name for name, _ in expected]
    files = {
        path.relative_to(DIRECTORY).as_posix()
        for study in ("development", "historical")
        for path in (DIRECTORY / study).rglob("*.py")
    }
    if len(entries) != len(set(entries)) or set(entries) != files:
        raise ValueError("Published ablation source inventory differs from the manifest")
    for name, checksum in expected:
        data = (DIRECTORY / name).read_bytes().replace(b"\r\n", b"\n")
        if sha256(data).hexdigest() != checksum:
            raise ValueError(f"Public source differs: {name}")
    print(f"Verified {len(manifest['files'])} published study sources and "
          f"{len(manifest['public_helpers'])} supporting files.", flush=True)
    return manifest


def verify_archive_references(manifest, archive):
    published = Counter(
        (row["prior_release_lf_sha256"], row["research_sha256"],
         tuple(sorted(manifest["receipts"][name] for name in row["receipts"])))
        for row in manifest["files"]
    )
    previous = Counter(
        (row["lf_sha256"], row["source_sha256"],
         tuple(sorted(archive["receipts"][name]["sha256"] for name in row["receipts"])))
        for row in archive["files"]
    )
    if published != previous:
        raise ValueError("Historical source or receipt references differ from the archive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="check", required=True)
    commands.add_parser("sources").add_argument("--research-root", type=Path)
    commands.add_parser("pairing").add_argument("--research-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = verify_public_sources()
    if args.research_root:
        sys.path.insert(0, str(DIRECTORY.parent))
        from replay import historical_sources

        with historical_sources() as history:
            verifier = history / "ablations" / "verify.py"
            archive = json.loads(verifier.with_name("SHA256.json").read_text(encoding="utf-8"))
            verify_archive_references(manifest, archive)
            subprocess.run(
                [sys.executable, str(verifier), args.check,
                 "--research-root", str(args.research_root.resolve())],
                check=True,
            )


if __name__ == "__main__":
    main()
