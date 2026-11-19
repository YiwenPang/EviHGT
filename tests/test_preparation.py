"""Check corpus preparation and input identity on synthetic files."""
import json
from pathlib import Path
import shutil
import pytest

from evi_hgt.data.common import sha256
from evi_hgt.data.corpus import load_corpus
from evi_hgt.data.prepare import prepare_from_spec
from data_fixture import make_corpus


def source_map(tmp_path):
    original = make_corpus(tmp_path/'original')
    manifest = json.loads(original.read_text())
    spec = dict(schema='EVIHGT_SOURCE_MAP_V1',manifest=manifest,sources={name:dict(
        path=str(original.parent/name),sha256=value) for name,value in manifest['files'].items()})
    path = tmp_path/'sources.json'
    path.write_text(json.dumps(spec))
    return original,path


def test_prepare_relocation_preserves_input_identity(tmp_path):
    original,spec = source_map(tmp_path)
    before = {p:sha256(p) for p in original.parent.iterdir() if p.is_file()}
    result = prepare_from_spec(spec,tmp_path/'prepared')
    assert result['source_files_modified'] is False
    assert result['identity'] == load_corpus(original).identity
    moved = tmp_path/'relocated'
    shutil.copytree(tmp_path/'prepared',moved)
    assert load_corpus(moved/'manifest.json').identity == result['identity']
    assert all(sha256(p) == value for p,value in before.items())


def test_prepare_rejects_changed_source_before_publication(tmp_path):
    original,spec = source_map(tmp_path)
    (original.parent/'eligibility.json').write_text('{}')
    with pytest.raises(ValueError,match='Source hash mismatch'):
        prepare_from_spec(spec,tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_prepare_never_overwrites_existing_directory(tmp_path):
    original,spec = source_map(tmp_path)
    with pytest.raises(FileExistsError,match='already exists'):
        prepare_from_spec(spec,original.parent)
