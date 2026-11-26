"""Columnar, lazy OptimusKG access helpers.

This module deliberately does not expose a default NetworkX loading path.  It
uses the official Harvard Dataverse release API for metadata/file access and
Polars lazy scans for Parquet data.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests


DEFAULT_SERVER = "https://dataverse.harvard.edu"
DEFAULT_DOI = "doi:10.7910/DVN/IYNGEV"
USER_AGENT = "EviHGT-Phase2-schema-audit/1.0"


@dataclass(frozen=True, slots=True)
class ReleaseFile:
    """One file in an official OptimusKG Dataverse release."""

    file_id: int
    relative_path: str
    size_bytes: int | None
    md5: str | None
    content_type: str | None


@dataclass(frozen=True, slots=True)
class ReleaseMetadata:
    """Version and file manifest from the official Dataverse API."""

    version: str
    files: dict[str, ReleaseFile]


class HTTPRangeReader(io.RawIOBase):
    """Seekable read-only HTTP file backed by explicit byte-range requests."""

    def __init__(
        self,
        url: str,
        size: int,
        *,
        session: Any | None = None,
    ) -> None:
        super().__init__()
        if size < 0:
            raise ValueError("HTTP object size must be non-negative")
        self.url = url
        self.size = size
        self.position = 0
        self.session = session or requests.Session()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise OSError("Cannot seek before the start of an HTTP object")
        self.position = min(position, self.size)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed HTTP range reader")
        if self.position >= self.size or size == 0:
            return b""
        if size is None or size < 0:
            end = self.size - 1
        else:
            end = min(self.position + size - 1, self.size - 1)
        start = self.position
        response = self.session.get(
            self.url,
            headers={
                "User-Agent": USER_AGENT,
                "Range": f"bytes={start}-{end}",
                "Accept": "application/octet-stream",
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.content
        expected = end - start + 1
        if len(data) == self.size and expected != self.size:
            data = data[start : end + 1]
        if len(data) != expected:
            raise OSError(
                f"HTTP range length mismatch for {start}-{end}: "
                f"expected {expected}, got {len(data)}"
            )
        self.position = end + 1
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


@dataclass(slots=True)
class RemoteParquetReader:
    """Read Parquet metadata/projected columns over authenticated HTTP ranges."""

    file: ReleaseFile
    server: str = DEFAULT_SERVER
    session: Any | None = None
    buffer_size: int = 1 << 20

    def _reader(self) -> io.BufferedReader:
        if self.file.size_bytes is None:
            raise ValueError(
                f"Remote file size is missing for {self.file.relative_path}"
            )
        raw = HTTPRangeReader(
            official_access_url(self.file, server=self.server),
            self.file.size_bytes,
            session=self.session,
        )
        return io.BufferedReader(raw, buffer_size=self.buffer_size)

    def row_count(self) -> int:
        reader = self._reader()
        try:
            parquet = pq.ParquetFile(reader)
            return int(parquet.metadata.num_rows)
        finally:
            reader.close()

    def schema(self) -> pl.Schema:
        reader = self._reader()
        try:
            parquet = pq.ParquetFile(reader)
            empty = pa.Table.from_batches([], schema=parquet.schema_arrow)
            frame = pl.from_arrow(empty)
            if not isinstance(frame, pl.DataFrame):
                raise TypeError("Expected a Polars DataFrame from Arrow schema")
            return frame.schema
        finally:
            reader.close()

    def read_columns(self, columns: Iterable[str]) -> pl.DataFrame:
        reader = self._reader()
        try:
            parquet = pq.ParquetFile(reader)
            table = parquet.read(columns=list(columns), use_threads=True)
            frame = pl.from_arrow(table)
            if not isinstance(frame, pl.DataFrame):
                raise TypeError("Expected a Polars DataFrame from Arrow table")
            return frame
        finally:
            reader.close()


def _request_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def fetch_release_metadata(
    *,
    server: str = DEFAULT_SERVER,
    doi: str = DEFAULT_DOI,
    version: str | None = None,
    session: Any | None = None,
) -> ReleaseMetadata:
    """Fetch an official file manifest without downloading graph data.

    When ``version`` is supplied, the version-specific Dataverse endpoint is
    used.  This prevents a later publisher release from being substituted for
    an explicitly pinned build.
    """

    client = session or requests.Session()
    if version is not None and not re.fullmatch(r"\d+(?:\.\d+)?", version):
        raise ValueError(f"Invalid Dataverse release version: {version!r}")
    endpoint = f"{server.rstrip('/')}/api/datasets/:persistentId/"
    if version is not None:
        endpoint = f"{endpoint}versions/{version}"
    response = client.get(
        endpoint,
        params={"persistentId": doi},
        headers=_request_headers(),
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()["data"]
    latest = data if version is not None else data["latestVersion"]
    major = latest.get("versionNumber")
    minor = latest.get("versionMinorNumber")
    version = (
        f"{major}.{minor}"
        if major is not None and minor is not None
        else str(latest.get("versionState", "DRAFT")).lower()
    )

    files: dict[str, ReleaseFile] = {}
    for raw_file in latest.get("files", []):
        data_file = raw_file["dataFile"]
        directory = (raw_file.get("directoryLabel") or "").strip("/")
        filename = data_file["filename"]
        relative_path = f"{directory}/{filename}" if directory else filename
        files[relative_path] = ReleaseFile(
            file_id=int(data_file["id"]),
            relative_path=relative_path,
            size_bytes=(
                int(data_file["filesize"])
                if data_file.get("filesize") is not None
                else None
            ),
            md5=data_file.get("md5"),
            content_type=data_file.get("contentType"),
        )
    return ReleaseMetadata(version=version, files=files)


def official_access_url(
    file: ReleaseFile, *, server: str = DEFAULT_SERVER
) -> str:
    """Return the official Dataverse data-file access URL."""

    return f"{server.rstrip('/')}/api/access/datafile/{file.file_id}"


def safe_cache_path(
    cache_dir: str | Path, version: str, relative_path: str
) -> Path:
    """Resolve a release path inside a cache root and reject traversal."""

    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("OptimusKG file path must be a safe relative path")
    if any(part in {"", "."} for part in relative.parts):
        raise ValueError("OptimusKG file path must be a safe relative path")

    root = Path(cache_dir).resolve()
    candidate = root / version / Path(*relative.parts)
    resolved = candidate.resolve()
    if root not in resolved.parents:
        raise ValueError("OptimusKG file path must remain inside the cache")
    return resolved


def download_release_file(
    file: ReleaseFile,
    *,
    version: str,
    cache_dir: str | Path,
    server: str = DEFAULT_SERVER,
    session: Any | None = None,
    force: bool = False,
) -> Path:
    """Download one explicitly selected release file and verify size/MD5."""

    destination = safe_cache_path(cache_dir, version, file.relative_path)
    if destination.exists() and not force:
        return destination

    client = session or requests.Session()
    response = client.get(
        official_access_url(file, server=server),
        headers={"User-Agent": USER_AGENT},
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.md5()  # noqa: S324 - verifies the publisher-supplied MD5.
    size = 0
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if file.size_bytes is not None and size != file.size_bytes:
            raise OSError(
                f"Downloaded size mismatch for {file.relative_path}: "
                f"expected {file.size_bytes}, got {size}"
            )
        if file.md5 is not None and digest.hexdigest().lower() != file.md5.lower():
            raise OSError(f"Downloaded MD5 mismatch for {file.relative_path}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def scan_parquet(
    path_or_url: str | Path, *, columns: Iterable[str] | None = None
) -> pl.LazyFrame:
    """Return a lazy Parquet scan, optionally projected to selected columns."""

    lazy = pl.scan_parquet(str(path_or_url))
    if columns is not None:
        lazy = lazy.select(list(columns))
    return lazy


def parquet_row_count(path_or_url: str | Path) -> int:
    """Read a Parquet row count through metadata-aware lazy execution."""

    return int(scan_parquet(path_or_url).select(pl.len()).collect().item())


def summarize_relations(lazy: pl.LazyFrame) -> pl.DataFrame:
    """Count relation/direction combinations present in an edge table."""

    names = set(lazy.collect_schema().names())
    missing = {"relation", "undirected"} - names
    if missing:
        raise ValueError(f"Edge table is missing required columns: {sorted(missing)}")
    return (
        lazy.group_by(["relation", "undirected"])
        .agg(pl.len().alias("count"))
        .sort(["relation", "undirected"])
        .collect()
    )


def flatten_schema(schema: pl.Schema) -> list[dict[str, str]]:
    """Flatten nested Struct fields into auditable dotted property paths."""

    rows: list[dict[str, str]] = []

    def visit(prefix: str, dtype: pl.DataType) -> None:
        if isinstance(dtype, pl.Struct):
            for field in dtype.fields:
                child = f"{prefix}.{field.name}" if prefix else field.name
                visit(child, field.dtype)
            return
        if isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct):
            for field in dtype.inner.fields:
                visit(f"{prefix}[].{field.name}", field.dtype)
            return
        rows.append({"field": prefix, "dtype": str(dtype)})

    for name, dtype in schema.items():
        visit(name, dtype)
    return rows


@dataclass(slots=True)
class OptimusKGLoader:
    """Official release manifest plus selective lazy/cache access."""

    cache_dir: Path
    server: str = DEFAULT_SERVER
    doi: str = DEFAULT_DOI
    _release: ReleaseMetadata | None = None

    def release(self, *, refresh: bool = False) -> ReleaseMetadata:
        if self._release is None or refresh:
            self._release = fetch_release_metadata(server=self.server, doi=self.doi)
        return self._release

    def remote_scan(
        self, relative_path: str, *, columns: Iterable[str] | None = None
    ) -> pl.LazyFrame:
        file = self.release().files[relative_path]
        return scan_parquet(
            official_access_url(file, server=self.server), columns=columns
        )

    def cached_path(self, relative_path: str) -> Path:
        release = self.release()
        return safe_cache_path(self.cache_dir, release.version, relative_path)

    def download(self, relative_path: str, *, force: bool = False) -> Path:
        release = self.release()
        return download_release_file(
            release.files[relative_path],
            version=release.version,
            cache_dir=self.cache_dir,
            server=self.server,
            force=force,
        )

    def cached_scan(
        self, relative_path: str, *, columns: Iterable[str] | None = None
    ) -> pl.LazyFrame:
        path = self.cached_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"OptimusKG cache file is not present: {path}. "
                "Call download() explicitly for the selected file."
            )
        return scan_parquet(path, columns=columns)
