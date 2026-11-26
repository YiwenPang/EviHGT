"""Source downloads must be verified and must preserve existing inputs."""
from hashlib import sha256
from io import BytesIO

import pytest

from evi_hgt.preprocessing.resources import ensure_cache_files


def descriptor(body=b"public auxiliary fixture"):
    return dict(name="part-00000.parquet", url="https://example.org/part-00000.parquet",
                bytes=len(body), sha256=sha256(body).hexdigest())


def test_verified_cache_is_reused_without_network(tmp_path):
    body = b"public auxiliary fixture"
    (tmp_path / "part-00000.parquet").write_bytes(body)
    def forbidden(url):
        raise AssertionError("Verified cached inputs must not access the network")
    paths = ensure_cache_files([descriptor(body)], tmp_path, offline=True, opener=forbidden)
    assert paths == [tmp_path / "part-00000.parquet"]
    assert paths[0].read_bytes() == body


def test_offline_missing_resource_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline"):
        ensure_cache_files([descriptor()], tmp_path / "cache", offline=True)
    assert not (tmp_path / "cache" / "part-00000.parquet").exists()


def test_existing_wrong_source_is_never_replaced(tmp_path):
    path = tmp_path / "part-00000.parquet"
    path.write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="existing"):
        ensure_cache_files([descriptor()], tmp_path, opener=lambda url: BytesIO(b"replacement"))
    assert path.read_bytes() == b"preserve me"


def test_download_is_published_only_after_size_and_hash_match(tmp_path):
    body = b"public auxiliary fixture"
    paths = ensure_cache_files([descriptor(body)], tmp_path, opener=lambda url: BytesIO(body))
    assert paths[0].read_bytes() == body
    assert not list(tmp_path.glob("*.partial"))


def test_wrong_download_is_not_published(tmp_path):
    with pytest.raises(ValueError, match="download"):
        ensure_cache_files([descriptor()], tmp_path, opener=lambda url: BytesIO(b"bad"))
    assert not (tmp_path / "part-00000.parquet").exists()


@pytest.mark.parametrize("name", ["../outside.parquet", "/outside.parquet", "C:/outside.parquet", "a\\b.parquet"])
def test_manifest_cannot_escape_cache(tmp_path, name):
    entry = descriptor()
    entry["name"] = name
    with pytest.raises(ValueError, match="filename"):
        ensure_cache_files([entry], tmp_path, offline=True)
