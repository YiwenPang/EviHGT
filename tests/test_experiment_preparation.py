"""Portable experiment inputs retain the original folds and immutable sources."""
import importlib
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest


def module():
    name = 'experiments.workflow.preparation'
    assert importlib.util.find_spec(name) is not None, 'Portable experiment preparation is missing'
    return importlib.import_module(name)


def small_pools():
    panel = [dict(drug_id=f'D{i}', disease_id=f'S{j}', role='P' if i < 30 else 'U')
             for j in range(2) for i in range(60)]
    auxiliary = [(f'D{i}', f'S{j}') for j in range(2) for i in range(60, 90)]
    positives = {(r['drug_id'], r['disease_id']) for r in panel if r['role'] == 'P'}
    observed = positives | set(auxiliary) | {('RESERVED', 'S0')}
    return dict(panel=panel, auxiliary=auxiliary, observed=observed,
                frozen={('RESERVED', 'S0')}, historical=set(),
                eligible_drugs=[f'D{i}' for i in range(100)], eligible_diseases=['S0', 'S1'],
                legal_source_rows=observed - {('RESERVED', 'S0')})


def test_registered_recipes_keep_original_r0f1_and_no_private_memberships():
    m = module()
    from evi_hgt.preprocessing.splits import recipe
    expected = recipe()
    rebuilt = m.load_recipe('R0F1')
    for key in ('conditional_bindings', 'membership_digests', 'mask_digest', 'training_seed',
                'cycles', 'G', 'record_digest'):
        assert rebuilt[key] == expected[key]
    seeds = {'R0F0': 1289779183, 'R0F1': 342565966, 'R0F2': 1867477778,
             'R0F3': 214283396, 'R0F4': 1680030835, 'R1F0': 625224385}
    for cell, seed in seeds.items():
        record = m.load_recipe(cell)
        assert record['training_seed'] == seed
        assert len(record['cycles']) == 3
        assert all(len(value) == 64 for value in record['membership_digests'].values())
        serialized = json.dumps(record)
        assert 'C:\\' not in serialized and 'C:/' not in serialized
        assert not any(key in record for key in ('records', 'assignments', 'training_pairs'))
    with pytest.raises(ValueError, match='registered'):
        m.load_recipe('R9F0')


def test_losses_recipe_keeps_original_dev95_seed_and_two_cycle_prefix():
    m = module()
    ordinary = m.load_recipe('R0F0')
    losses = m.load_recipe('R0F0', protocol='losses')
    for key in ('conditional_bindings', 'membership_digests', 'mask_digest', 'training_seed',
                'cycles', 'G', 'record_digest'):
        assert losses[key] == ordinary[key]
    assert losses['protocol'] == 'DEV95_R0F0_LOSSES'
    assert losses['source_plan']['schema'] == 'DEV94_SHARED_CONDITIONAL_V1'
    assert losses['source_plan']['records_selected'] == 2 * losses['G'] == 13712
    assert losses['source_plan']['records_total'] == 34280
    assert losses['source_plan']['record_range'] == [0, 13712]
    assert losses['source_plan']['sha256'] == json.loads(
        (m.ROOT / 'experiments/workflow/recipes.json').read_text('utf-8'))[
            'source_metadata_sha256']['dev95_plan']
    with pytest.raises(ValueError, match='losses'):
        m.load_recipe('R0F1', protocol='losses')
    with pytest.raises(ValueError, match='protocol'):
        m.load_recipe('R0F0', protocol='unknown')


def test_losses_preparation_has_independent_cache_and_published_identity(tmp_path, monkeypatch):
    m = module()
    observed = []
    monkeypatch.setattr(m, 'reuse_corpus', lambda path, identity, **kw: observed.append((path, identity)))
    shared = dict(source_identity={'build_identity': 'a' * 64}, nodes=str(tmp_path / 'raw/nodes.parquet'))
    destination = tmp_path / 'corpora' / ('a' * 20)
    for suffix in ('R0F0', 'losses/R0F0'):
        path = destination / suffix / 'manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    normal = m.prepare_cell(shared, 'R0F0', destination_root=tmp_path / 'corpora', audit_dir=tmp_path / 'audit')
    losses = m.prepare_cell(shared, 'R0F0', protocol='losses', destination_root=tmp_path / 'corpora',
                            audit_dir=tmp_path / 'audit')
    assert normal != losses
    assert losses == destination / 'losses/R0F0/manifest.json'
    assert observed[0][1] != observed[1][1]
    assert observed[1][1]['protocol'] == 'losses'


def test_partition_masks_every_reserved_pair_across_both_heads():
    m = module()
    pools = small_pools()
    partitions = [m.partition_pools(pools, cell) for cell in ('R0F0', 'R0F1', 'R1F0')]
    assert len({p['training_seed'] for p in partitions}) == 3
    for result in partitions:
        parts = result['parts']
        training = {(r['drug_id'], r['disease_id']) for name in ('training', 'auxiliary_training')
                    for r in parts[name]}
        reserved = {(r['drug_id'], r['disease_id']) for name in ('validation', 'train_probe', 'outer_test',
                    'auxiliary_validation', 'auxiliary_probe') for r in parts[name]}
        assert training.isdisjoint(result['blocked'])
        assert reserved | pools['frozen'] <= result['blocked']
        assert result['training_seed'] == m.load_recipe(result['cell'])['training_seed']
        assert parts['unlabeled_is_negative'] is False


def test_real_recipe_mismatch_stops_before_publishing_a_synthetic_fold(tmp_path):
    m = module()
    target = tmp_path / 'fold'
    with pytest.raises(ValueError, match='membership'):
        m.build_cell_splits(small_pools(), target, cell='R0F1', audit_dir=tmp_path / 'audit')
    assert not target.exists()


def test_pooled_manifest_roundtrip_preserves_pair_sets_and_u_roles(tmp_path):
    m = module()
    values = small_pools()
    values['source'] = pl.DataFrame(dict(drug_id=['D1'], disease_id=['S1'],
                                       raw_relation=['INDICATION']))
    result = m.save_pools(values, tmp_path / 'pools')
    loaded = m.load_pools(result)
    for key in ('panel', 'auxiliary', 'frozen', 'historical', 'observed', 'legal_source_rows',
                'eligible_drugs', 'eligible_diseases'):
        assert loaded[key] == values[key]
    path = Path(result['files']['panel'])
    original = path.read_bytes()
    path.write_bytes(original + b'corruption')
    with pytest.raises(ValueError, match='hash'):
        m.load_pools(result)


def test_preparation_configuration_does_not_allow_raw_input_as_output(tmp_path):
    m = module()
    source = tmp_path / 'raw' / 'nodes.parquet'
    source.parent.mkdir()
    source.write_bytes(b'raw input unchanged')
    with pytest.raises(ValueError, match='original'):
        m.validate_destinations({'nodes': source}, [source.parent])
    assert source.read_bytes() == b'raw input unchanged'


def test_shared_build_identity_binds_nested_source_policy():
    m = module()
    from evi_hgt.data.common import sha256
    path = m.ROOT / 'configs/preprocessing/opentargets.json'
    assert m._source_code().get('configs/preprocessing/opentargets.json') == sha256(path)


def test_fold_graph_mask_drops_reverse_alias_evidence_and_csr(tmp_path):
    m = module()
    import numpy as np
    source = tmp_path / 'source'
    source.mkdir()
    facts = pl.DataFrame(dict(canonical_fact_id=[1, 2, 3], source_node_id=['D1', 'S1', 'G1'],
        target_node_id=['S1', 'D1', 'S1'], source_node_type=['DRG', 'DIS', 'GEN'],
        target_node_type=['DIS', 'DRG', 'DIS']))
    facts.write_parquet(source / 'facts.parquet')
    pl.DataFrame(dict(canonical_fact_id=[1, 2, 3], properties=['a', 'b', 'c'])).write_parquet(source / 'evidence.parquet')
    pl.DataFrame(dict(drug_id=['D1', 'D1'], disease_id=['S1', 'S1'], canonical_fact_id=[1, 2])).write_parquet(source / 'target.parquet')
    pl.DataFrame({'canonical_fact_id': pl.Series([], dtype=pl.Int64)}).write_parquet(source / 'forbidden.parquet')
    csr = {}
    for name, values in dict(indptr=[0, 2, 4, 6], indices=[1, 1, 0, 2, 1, 0], fact_ids=[1, 2, 1, 3, 3, 2]).items():
        np.save(source / (name + '.npy'), np.asarray(values, dtype=np.int64))
        csr[name] = str(source / (name + '.npy'))
    shared_graph = dict(nodes='not-read', edge_catalog='not-read', csr=csr,
        target_map=str(source / 'target.parquet'), forbidden_facts=str(source / 'forbidden.parquet'),
        fact_files=[dict(component='mixed', rows=3, path=str(source / 'facts.parquet'))],
        evidence_files=[dict(component='mixed', rows=3, path=str(source / 'evidence.parquet'))])
    splits = tmp_path / 'split'
    splits.mkdir()
    pl.DataFrame(dict(drug_id=['D1'], disease_id=['S1'])).write_parquet(splits / 'blocked_pairs.parquet')
    masked = m.mask_cell_graph(shared_graph, splits, tmp_path / 'masked')
    assert pl.read_parquet(masked['fact_files'][0]['path'])['canonical_fact_id'].to_list() == [3]
    assert pl.read_parquet(masked['evidence_files'][0]['path'])['canonical_fact_id'].to_list() == [3]
    assert np.load(masked['csr']['fact_ids']).tolist() == [3, 3]
    assert pl.read_parquet(source / 'facts.parquet').height == 3
    assert set(pl.read_parquet(masked['forbidden_facts'])['canonical_fact_id']) == {1, 2}
