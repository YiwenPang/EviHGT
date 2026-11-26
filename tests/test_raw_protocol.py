import importlib

import polars as pl
import pytest


def module(name):
    spec = importlib.util.find_spec('evi_hgt.preprocessing.' + name)
    assert spec is not None, 'Raw protocol implementation is missing: ' + name
    return importlib.import_module('evi_hgt.preprocessing.' + name)


def test_source_policy_rejects_unclassified_and_retains_independent_presence():
    m = module('clean_graph')
    frame = pl.DataFrame(dict(datasourceId=['genetics', 'drug'], datatypeId=['genetic', 'known_drug'],
        diseaseId=['D', 'D'], targetId=['G', 'G'], score=[.3, .9], evidenceCount=[1, 2]))
    policy = {'genetics': {'eligible': True, 'datatype_id': 'genetic'},
              'drug': {'eligible': False, 'datatype_id': 'known_drug'}}
    clean = m.clean_source_rows(frame, policy)
    assert clean['datasourceId'].to_list() == ['genetics']
    assert m.aggregate_clean_edges(clean)['clean_presence'].to_list() == [1.]
    with pytest.raises(ValueError, match='unclassified'):
        m.clean_source_rows(frame, {'genetics': policy['genetics']})
    with pytest.raises(ValueError, match='duplicate'):
        m.clean_source_rows(pl.concat([frame, frame.head(1)]), policy)
    with pytest.raises(ValueError, match='score'):
        m.clean_source_rows(frame.with_columns(pl.lit(float('nan')).alias('score')), policy)
    with pytest.raises(ValueError, match='datatype'):
        m.clean_source_rows(frame.with_columns(pl.lit('wrong').alias('datatypeId')), policy)


def test_endpoint_mask_removes_reverse_and_equivalent_relations():
    m = module('clean_graph')
    facts = pl.DataFrame(dict(source_node_id=['D', 'S', 'D', 'G'], target_node_id=['S', 'D', 'S', 'S'],
        source_node_type=['DRG', 'DIS', 'DRG', 'GEN'], target_node_type=['DIS', 'DRG', 'DIS', 'DIS'],
        relation=['INDICATION', 'inverse', 'alias', 'association']))
    assert m.mask_endpoint_pairs(facts, {('D', 'S')})['relation'].to_list() == ['association']


def test_cv_reservations_isolate_both_channels_and_do_not_label_u_negative():
    m = module('splits')
    panel = [dict(drug_id=f'D{i}', disease_id='S', role='P' if i < 60 else 'U') for i in range(100)]
    auxiliary = [dict(drug_id=f'D{i}', disease_id='S', raw_relation='CONTRAINDICATION') for i in range(60)]
    folds = m.assign_folds(panel, 5, 'fixed')
    cell = m.make_cell_manifest(panel, folds, 1, validation_fraction=.2, protocol_digest='test',
        auxiliary_training_positives=auxiliary)
    training = {(r['drug_id'], r['disease_id']) for n in ('training', 'auxiliary_training') for r in cell[n]}
    reserved = {(r['drug_id'], r['disease_id']) for n in ('outer_test', 'validation', 'train_probe',
        'auxiliary_validation', 'auxiliary_probe') for r in cell[n]}
    assert training.isdisjoint(reserved)
    assert cell['unlabeled_is_negative'] is False


def test_candidate_projection_uses_requested_split_not_attached_standard_assignment():
    m = module('splits')
    source = pl.DataFrame(dict(drug_id=['D1', 'D2'], disease_id=['S', 'S'],
        raw_relation=['INDICATION', 'INDICATION'], support_mask=[True, True], policy_b_member=[True, True],
        split_assignment=['test', 'train']))
    cold = pl.DataFrame(dict(drug_id=['D1', 'D2'], disease_id=['S', 'S'],
        split_assignment=['train', 'test'], drug_partition=['train', 'test'], disease_partition=[None, None]))
    result = list(m.candidate_sources(source, cold, 'cold_drug'))
    assert result == [('S', {'D2'}, {'D2'})]


def test_private_sampler_matches_rejection_draws_and_exact_positive_coverage():
    m = module('sampling')
    import random
    state = random.getstate()
    support = {('D1', 'S1'), ('D2', 'S1'), ('D3', 'S1')}
    contra = {('D1', 'S2'), ('D2', 'S2')}
    observed = support | contra
    groups = m.epoch_groups(support, contra, ['D1', 'D2', 'D3', 'D4'], ['S1', 'S2'],
        observed, set(), training_seed=42, repeat_key='fixed', cycle=1)
    assert len(groups) == 2
    assert [p for g in groups for role,p in g if role == 'support'] != []
    assert {p for g in groups for role,p in g if role == 'support'} == support
    assert {p for g in groups for role,p in g if role == 'unsuitability'} == contra
    assert all(p not in observed for g in groups for role,p in g if role == 'U')
    assert random.getstate() == state


def test_conditional_fallback_retains_every_positive_and_original_u():
    m = module('sampling')
    support = [('support', (f'P{i}', 'S1')) for i in range(20)] + [('support', ('PX', 'S2'))]
    group = tuple(support + [('unsuitability', ('C', 'S1')), ('U', ('U1', 'S1'))])
    # S2 has no legal candidate, while 20/21 support occurrences remain covered.
    forbidden = {('U1', 'S2'), ('U2', 'S2')} | {p for _, p in support} | {('C', 'S1')}
    result = m.conditional_records([group], ['U1', 'U2'], ['S1', 'S2'], forbidden,
        G=1, bindings={'fixture': 'no score or checkpoint'})[0]
    assert result['support_positive_pairs'] == [list(p) for _, p in support]
    assert result['original_global_u'] == ['U1', 'S1']
    assert result['disease_unlabeled_pairs']['S2'] == []
    assert result['disease_status']['S2'] == 'NO_LEGAL_U_ORIGINAL_PAIRING_FALLBACK'
    assert len(result['disease_unlabeled_pairs']['S1']) == 1


def test_exhausted_global_domain_fails_without_relabeling_observed_pairs():
    m = module('sampling')
    with pytest.raises(ValueError, match='rejection budget exhausted'):
        m.epoch_groups({('D1', 'S1')}, {('D1', 'S2')}, ['D1'], ['S1', 'S2'],
            {('D1', 'S1'), ('D1', 'S2')}, set(), training_seed=42, repeat_key='fixed', cycle=1)


def test_raw_recipe_rejects_other_cells_before_touching_destination(tmp_path):
    m = module('splits')
    with pytest.raises(ValueError, match='only.*R0F1'):
        m.build_splits(tmp_path / 'missing', tmp_path / 'destination', cell='R2F4', audit_dir=tmp_path)
    assert not (tmp_path / 'destination').exists()


def test_clean_builder_physically_filters_csr_and_replaces_contaminated_properties(tmp_path):
    import json
    import numpy as np
    m = module('clean_graph')
    canonical, split, dest = tmp_path / 'canonical', tmp_path / 'splits', tmp_path / 'clean'
    for name in ('canonical_facts', 'evidence_records', 'views', 'nodes', 'graph/csr'):
        (canonical / name).mkdir(parents=True)
    split.mkdir()
    def fact(fid, a, b, at, bt, relation):
        return dict(canonical_fact_id=fid, canonical_fact_key=f'key{fid}', source_node_id=a,
            target_node_id=b, source_node_type=at, target_node_type=bt, source_global_id=0,
            target_global_id=1, relation=relation, source_component_path='original', aggregated_observation_count=1)
    families = {
        'disease_gene': [fact(1, 'S1', 'G1', 'DIS', 'GEN', 'association'), fact(2, 'S2', 'G1', 'DIS', 'GEN', 'association')],
        'drug_disease': [fact(3, 'D1', 'S1', 'DRG', 'DIS', 'INDICATION'), fact(4, 'D2', 'S1', 'DRG', 'DIS', 'INDICATION')],
        'reverse_alias': [fact(5, 'S1', 'D1', 'DIS', 'DRG', 'equivalent')],
        'drug_gene': [fact(6, 'D1', 'G1', 'DRG', 'GEN', 'target')]}
    for name, values in families.items():
        pl.DataFrame(values).write_parquet(canonical / 'canonical_facts' / (name + '.parquet'))
        pl.DataFrame([dict(canonical_fact_id=r['canonical_fact_id'], properties='CONTAMINATED_OLD_DG')
            for r in values]).write_parquet(canonical / 'evidence_records' / (name + '.parquet'))
    target = pl.DataFrame(dict(drug_id=['D1', 'D2', 'D1'], disease_id=['S1'] * 3,
        canonical_fact_id=[3, 4, 5], canonical_pair_key=['D1\x1fS1', 'D2\x1fS1', 'D1\x1fS1'],
        raw_relation=['INDICATION', 'INDICATION', 'equivalent']))
    target.write_parquet(canonical / 'views/query_target_fact_map.parquet')
    pl.DataFrame(dict(drug_id=['D1'], disease_id=['S1'])).write_parquet(split / 'blocked_pairs.parquet')
    pl.DataFrame(dict(drug_id=['D2'], disease_id=['S1'])).write_parquet(split / 'legal_source_rows.parquet')
    for name, values in {'indptr': [0, 3, 6], 'indices': [1, 1, 1, 0, 0, 0],
                         'fact_ids': [1, 2, 3, 4, 5, 6], 'edge_type_ids': [0, 0, 1, 1, 2, 3]}.items():
        np.save(canonical / 'graph/csr' / (name + '.npy'), np.asarray(values, dtype=np.int64))
    raw = tmp_path / 'official.parquet'
    pl.DataFrame(dict(datasourceId=['genetics', 'drug'], datatypeId=['genetic', 'known_drug'],
        diseaseId=['S1', 'S2'], targetId=['G1', 'G1'], score=[.3, .9], evidenceCount=[1, 2])).write_parquet(raw)
    policy = {'genetics': {'eligible': True, 'datatype_id': 'genetic'},
              'drug': {'eligible': False, 'datatype_id': 'known_drug'}}
    result = m.build_clean_graph(canonical, split, [raw], policy, dest, audit_dir=tmp_path / 'audit')
    assert np.load(result['csr']['fact_ids']).tolist() == [1, 4, 6]
    assert np.load(result['csr']['indptr']).tolist() == [0, 1, 3]
    assert pl.read_parquet(result['target_map'])['canonical_fact_id'].to_list() == [4]
    assert pl.read_parquet(result['target_map']).columns == ['drug_id', 'disease_id', 'canonical_fact_id']
    evidence = pl.read_parquet(dest / 'evidence_records/disease_gene.parquet')
    assert evidence['edge_support_observed'].to_list() == [False]
    assert evidence['edge_counter_evidence_observed'].to_list() == [False]
    assert json.loads(evidence['properties'][0])['clean_datasources'] == ['genetics']
    assert pl.read_parquet(canonical / 'evidence_records/disease_gene.parquet')['properties'][0] == 'CONTAMINATED_OLD_DG'
