"""Verified acquisition of explicitly pinned public auxiliary resources."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from urllib.request import urlopen
from uuid import uuid4


def ensure_cache_files(files, cache_dir, *, offline=False, opener=None, progress=None):
    """Return checked paths; never repair or overwrite an existing source file."""
    entries = list(files)
    names = set()
    for entry in entries:
        name = entry.get("name")
        if (not isinstance(name, str) or not name or name in (".", "..")
                or any(c in name for c in ("/", "\\", ":", "\x00")) or name in names):
            raise ValueError("Resource manifest requires distinct plain filenames")
        names.add(name)
        if (type(entry.get("bytes")) is not int or entry["bytes"] < 1
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
                or not isinstance(entry.get("url"), str)):
            raise ValueError("Resource manifest requires size, SHA-256 and URL")
    cache = Path(cache_dir).resolve()
    paths = [cache / item["name"] for item in entries]

    def matches(path, entry):
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            return False
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        return digest.hexdigest() == entry["sha256"]

    # Check every present input before starting any downloads.
    for path, entry in zip(paths, entries):
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not matches(path, entry):
                raise ValueError(f"Preserving existing resource with unexpected identity: {path}")
        elif offline:
            raise FileNotFoundError(f"Missing resource in offline mode: {path}; run once online or supply the pinned cache.")
    fetch = opener or (lambda url: urlopen(url, timeout=120))
    for index, (path, entry) in enumerate(zip(paths, entries), 1):
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            temporary = cache / (entry["name"] + "." + uuid4().hex + ".partial")
            try:
                count = 0
                digest = sha256()
                with fetch(entry["url"]) as source, temporary.open("xb") as target:
                    while block := source.read(1 << 20):
                        count += len(block)
                        if count > entry["bytes"]:
                            raise ValueError(f"Unexpected download size: {entry['name']}")
                        digest.update(block)
                        target.write(block)
                        if progress:
                            progress("Downloading " + entry["name"], count, entry["bytes"], count)
                    target.flush()
                    os.fsync(target.fileno())
                if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise ValueError(f"Unexpected download identity: {entry['name']}")
                # Hard-link publication is atomic and fails if another writer won.
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if not matches(path, entry):
                        raise ValueError(f"Preserving conflicting existing resource: {path}")
            finally:
                temporary.unlink(missing_ok=True)
        if progress:
            progress("Verified public resources", index, len(entries), entry["bytes"])
    return paths


def ensure_open_targets(cache_dir, *, offline=False, progress=None):
    """Acquire the exact source-level associations used for leakage cleaning."""
    recipe = Path(__file__).resolve().parents[3] / "configs/preprocessing/opentargets.json"
    manifest = json.loads(recipe.read_text(encoding="utf8"))
    base = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/25.06/output/association_by_datasource_indirect/"
    if (manifest.get("schema") != "EVIHGT_OPENTARGETS_SOURCE_RECIPE_V1"
            or manifest.get("release") != "25.06"
            or any(item["url"] != base + item["name"] for item in manifest["files"])):
        raise ValueError("Open Targets source recipe differs from the pinned release")
    paths = ensure_cache_files(manifest["files"], cache_dir, offline=offline, progress=progress)
    return paths, manifest["datasource_policy"]
