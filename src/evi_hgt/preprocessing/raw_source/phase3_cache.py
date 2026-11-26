"""Verified project-local caching for Phase 3 OptimusKG components."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from evi_hgt.preprocessing.raw_source.optimus_loader import (
    DEFAULT_SERVER,
    USER_AGENT,
    ReleaseFile,
    ReleaseMetadata,
    official_access_url,
    safe_cache_path,
)


class InputIntegrityError(RuntimeError):
    """Raised when a release binding cannot be used without guessing."""


@dataclass(frozen=True, slots=True)
class ComponentBinding:
    """Observed and publisher-declared identity of one component file."""

    release_version: str
    file_id: int
    relative_path: str
    cache_path: Path
    size_bytes: int
    manifest_md5: str | None
    observed_md5: str
    observed_sha256: str
    status: str
    usable: bool
    quarantine_path: Path | None = None
    manifest_size_bytes: int | None = None
    content_type: str | None = None
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cache_path"] = str(self.cache_path)
        value["quarantine_path"] = (
            str(self.quarantine_path) if self.quarantine_path else None
        )
        return value


def _digests(path: Path) -> tuple[int, str, str]:
    md5 = hashlib.md5()  # noqa: S324 - publisher release integrity field.
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def resolve_component_plan(
    release: ReleaseMetadata, *, expected_release: str = "2.0"
) -> tuple[ReleaseFile, ...]:
    """Select only stratified node/edge Parquets from an exact release."""

    if release.version != expected_release:
        raise InputIntegrityError(
            f"Phase 3 requires OptimusKG release 2.0; observed {release.version}"
        )
    selected = [
        item
        for path, item in release.files.items()
        if path.endswith(".parquet")
        and (path.startswith("nodes/") or path.startswith("edges/"))
        and path.count("/") == 1
    ]
    if not selected:
        raise InputIntegrityError("Release contains no stratified Parquet components")
    return tuple(sorted(selected, key=lambda item: item.relative_path))


def _matches_release(
    file: ReleaseFile, *, size: int, observed_md5: str
) -> bool:
    return (
        (file.size_bytes is None or size == file.size_bytes)
        and (file.md5 is None or observed_md5.lower() == file.md5.lower())
    )


def _quarantine_destination(
    root: Path, version: str, file: ReleaseFile, observed_sha256: str
) -> Path:
    relative = Path(file.relative_path)
    destination = (
        root.resolve()
        / version
        / relative.parent
        / f"{relative.name}.sha256_{observed_sha256}.quarantine"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _quarantine(
    source: Path,
    *,
    root: Path,
    version: str,
    file: ReleaseFile,
    observed_sha256: str,
) -> Path:
    destination = _quarantine_destination(root, version, file, observed_sha256)
    if destination.exists():
        source.unlink(missing_ok=True)
    else:
        source.replace(destination)
    return destination


def _existing_quarantine(
    root: Path, *, version: str, file: ReleaseFile
) -> Path | None:
    relative = Path(file.relative_path)
    directory = root.resolve() / version / relative.parent
    if not directory.is_dir():
        return None
    candidates = sorted(
        path
        for path in directory.glob(
            f"{relative.name}.sha256_*.quarantine"
        )
        if path.is_file()
    )
    if len(candidates) > 1:
        raise InputIntegrityError(
            f"Multiple quarantined objects exist for {file.relative_path}: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0] if candidates else None


def _binding(
    file: ReleaseFile,
    *,
    version: str,
    cache_path: Path,
    size: int,
    observed_md5: str,
    observed_sha256: str,
    status: str,
    usable: bool,
    quarantine_path: Path | None = None,
    server: str = DEFAULT_SERVER,
) -> ComponentBinding:
    return ComponentBinding(
        release_version=version,
        file_id=file.file_id,
        relative_path=file.relative_path,
        cache_path=cache_path,
        size_bytes=size,
        manifest_md5=file.md5,
        observed_md5=observed_md5,
        observed_sha256=observed_sha256,
        status=status,
        usable=usable,
        quarantine_path=quarantine_path,
        manifest_size_bytes=file.size_bytes,
        content_type=file.content_type,
        source_url=official_access_url(file, server=server),
    )


def cache_component(
    file: ReleaseFile,
    *,
    version: str,
    cache_root: str | Path,
    quarantine_root: str | Path,
    server: str = DEFAULT_SERVER,
    session: Any | None = None,
) -> ComponentBinding:
    """Download or verify one normal release component.

    A mismatched local or downloaded object is quarantined and returned as
    unusable.  It is never silently replaced and never used by the graph build.
    """

    cache_root_path = Path(cache_root).resolve()
    quarantine_root_path = Path(quarantine_root).resolve()
    destination = safe_cache_path(cache_root_path, version, file.relative_path)
    if destination.is_file():
        size, observed_md5, observed_sha256 = _digests(destination)
        if _matches_release(file, size=size, observed_md5=observed_md5):
            return _binding(
                file,
                version=version,
                cache_path=destination,
                size=size,
                observed_md5=observed_md5,
                observed_sha256=observed_sha256,
                status="VERIFIED_CACHED",
                usable=True,
                server=server,
            )
        quarantine = _quarantine(
            destination,
            root=quarantine_root_path,
            version=version,
            file=file,
            observed_sha256=observed_sha256,
        )
        return _binding(
            file,
            version=version,
            cache_path=destination,
            size=size,
            observed_md5=observed_md5,
            observed_sha256=observed_sha256,
            status="QUARANTINED_CHECKSUM_MISMATCH",
            usable=False,
            quarantine_path=quarantine,
            server=server,
        )

    existing_quarantine = _existing_quarantine(
        quarantine_root_path, version=version, file=file
    )
    if existing_quarantine is not None:
        size, observed_md5, observed_sha256 = _digests(existing_quarantine)
        if _matches_release(file, size=size, observed_md5=observed_md5):
            raise InputIntegrityError(
                f"Quarantined object now matches the release metadata for "
                f"{file.relative_path}; manual integrity review is required"
            )
        expected_suffix = f".sha256_{observed_sha256}.quarantine"
        if not existing_quarantine.name.endswith(expected_suffix):
            raise InputIntegrityError(
                f"Quarantine filename digest mismatch: {existing_quarantine}"
            )
        return _binding(
            file,
            version=version,
            cache_path=destination,
            size=size,
            observed_md5=observed_md5,
            observed_sha256=observed_sha256,
            status="QUARANTINED_CHECKSUM_MISMATCH_CACHED",
            usable=False,
            quarantine_path=existing_quarantine,
            server=server,
        )

    client = session or requests.Session()
    response = client.get(
        official_access_url(file, server=server),
        headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"},
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    md5 = hashlib.md5()  # noqa: S324 - publisher release integrity field.
    sha256 = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    md5.update(chunk)
                    sha256.update(chunk)
                    size += len(chunk)
        observed_md5 = md5.hexdigest()
        observed_sha256 = sha256.hexdigest()
        if not _matches_release(file, size=size, observed_md5=observed_md5):
            quarantine = _quarantine(
                temporary,
                root=quarantine_root_path,
                version=version,
                file=file,
                observed_sha256=observed_sha256,
            )
            return _binding(
                file,
                version=version,
                cache_path=destination,
                size=size,
                observed_md5=observed_md5,
                observed_sha256=observed_sha256,
                status="QUARANTINED_CHECKSUM_MISMATCH",
                usable=False,
                quarantine_path=quarantine,
                server=server,
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return _binding(
        file,
        version=version,
        cache_path=destination,
        size=size,
        observed_md5=observed_md5,
        observed_sha256=observed_sha256,
        status="VERIFIED_DOWNLOADED",
        usable=True,
        server=server,
    )


def bind_checksum_drift_component(
    file: ReleaseFile,
    *,
    isolated_path: str | Path,
    observed_md5: str,
    observed_sha256: str,
    observed_size_bytes: int,
) -> ComponentBinding:
    """Bind the immutable audited ANA-GEN checksum-drift object."""

    path = Path(isolated_path).resolve()
    if not path.is_file():
        raise InputIntegrityError(f"Audited ANA-GEN object is missing: {path}")
    size, actual_md5, actual_sha256 = _digests(path)
    expected = (
        observed_size_bytes,
        observed_md5.lower(),
        observed_sha256.lower(),
    )
    actual = (size, actual_md5.lower(), actual_sha256.lower())
    if actual != expected:
        raise InputIntegrityError(
            "Audited ANA-GEN observed object no longer matches its pinned digest"
        )
    return ComponentBinding(
        release_version="2.0",
        file_id=file.file_id,
        relative_path=file.relative_path,
        cache_path=path,
        size_bytes=size,
        manifest_md5=file.md5,
        observed_md5=actual_md5,
        observed_sha256=actual_sha256,
        status="UPSTREAM_CHECKSUM_DRIFT_UNRESOLVED",
        usable=True,
        manifest_size_bytes=file.size_bytes,
        content_type=file.content_type,
        source_url=official_access_url(file),
    )


def assert_all_components_usable(bindings: Iterable[ComponentBinding]) -> None:
    unusable = [item.relative_path for item in bindings if not item.usable]
    if unusable:
        raise InputIntegrityError(
            f"Unusable OptimusKG components: {sorted(unusable)}"
        )
