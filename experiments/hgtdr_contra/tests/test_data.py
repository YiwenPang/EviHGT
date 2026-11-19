import json
from types import SimpleNamespace

import pytest

from evi_hgt.phase85_v2.common import digest
from experiments.hgtdr_contra.data import load_contra_fold, project_contra_candidates


def fixture(tmp_path):
    root = tmp_path / 'repeat_0_fold_1'
    root.mkdir()
    validation = [
        {'drug_id': 'D1', 'disease_id': 'S2', 'role': 'U'},
        {'drug_id': 'D2', 'disease_id': 'S1', 'role': 'P'},
    ]
    contra = [{'drug_id': 'D0', 'disease_id': 'S1', 'raw_relation': 'CONTRAINDICATION'}]
    source = root / 'cell_manifest.json'
    source.write_text(json.dumps({
        'validation': validation,
        'auxiliary_validation': contra,
        'outer_test': [{'unrelated': 'this field must stay untouched'}],
    }), encoding='utf-8')
    data = SimpleNamespace(
        original_corpus_root=root / 'corpus_v1',
        cell_id='R0F1',
        cell={'fold': 1},
        repeat_index=0,
        frozen={('D7', 'S1')},
        blocked={('D0', 'S1'), ('D1', 'S2'), ('D2', 'S1'), ('D7', 'S1')},
        contract=SimpleNamespace(bindings={'membership_digests': {
            'validation': digest(validation), 'auxiliary_validation': digest(contra),
        }}),
    )
    return data, validation, contra


def test_common_contra_panel_uses_only_contra_p_and_support_u(tmp_path):
    data, validation, contra = fixture(tmp_path)
    assert project_contra_candidates(data) == (dict(contra[0], role='P'), validation[0])


def test_rejects_protected_member_even_when_in_validation(tmp_path):
    data, _, _ = fixture(tmp_path)
    data.frozen.add(('D0', 'S1'))
    with pytest.raises(ValueError, match='protected'):
        project_contra_candidates(data)


def test_rejects_unmasked_candidate(tmp_path):
    data, _, _ = fixture(tmp_path)
    data.blocked.remove(('D1', 'S2'))
    with pytest.raises(ValueError, match='mask'):
        project_contra_candidates(data)


def test_rejects_changed_frozen_development_membership(tmp_path):
    data, _, _ = fixture(tmp_path)
    data.contract.bindings['membership_digests']['auxiliary_validation'] = 'wrong'
    with pytest.raises(ValueError, match='membership'):
        project_contra_candidates(data)


def test_contra_run_only_accepts_five_r0_cells():
    with pytest.raises(ValueError, match='R0'):
        load_contra_fold('R1F0')
