"""Restore the public source layout used by the archived experiment checks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

SOURCE_REVISION = "5251bd6ad157a169c23e42d0cc8d59d20f621fbd"
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "experiments", "src", "tests", "configs", "main.py", "ablation.py",
    "export_data.py", "prepare_data.py", "check_environment.py", "pyproject.toml",
    "requirements.txt", "requirements-dev.txt", "requirements-observed.txt", "README.md",
)


def restore_sources(destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new directory for the replay sources: {destination}")
    try:
        result = subprocess.run(
            [
                "git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "-C", str(REPOSITORY),
                "archive", "--format=zip", SOURCE_REVISION, "--", *SOURCE_PATHS,
            ],
            check=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Source replay requires a Git clone containing revision {SOURCE_REVISION}. "
            "In a shallow clone, fetch that revision from origin."
        ) from exc
    with ZipFile(BytesIO(result.stdout)) as archive:
        for member in archive.namelist():
            if not (destination / member).resolve().is_relative_to(destination):
                raise ValueError(f"Archive path outside the destination: {member}")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)
    return destination


@contextmanager
def historical_sources():
    with TemporaryDirectory(prefix="evihgt-source-replay-") as temporary:
        yield restore_sources(Path(temporary) / "source") / "experiments"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for the source tree")
    args = parser.parse_args()
    print(restore_sources(args.output))


if __name__ == "__main__":
    main()
