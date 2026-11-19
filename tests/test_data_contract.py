"""Check the data contract with a synthetic corpus."""
import json
import numpy as np
import polars as pl
import pytest

from evi_hgt.data.common import sha256
from evi_hgt.data.corpus import load_corpus
from evi_hgt.training.supervision import make_unlabeled_query_supervision
from data_fixture import make_corpus


def rebind(path, relative):
    manifest = json.loads(path.read_text())
    manifest['files'][relative] = sha256(path.parent/relative)
    path.write_text(json.dumps(manifest))


def test_full_portable_query_contract(tmp_path):
    path = make_corpus(tmp_path)
    corpus = load_corpus(path)
    assert corpus.plan.total_steps == 3
    for step in range(1,4):
        spec = corpus.plan.step(step)
        spec.validate()
        for query in spec.queries:
            graph = corpus.assembler.assemble(query)
            assert len(graph.batch.model_batch.nodes.features[0]) == 3588
            assert not set(f.canonical_fact_id for f in graph.batch.path_source.facts) & corpus.target_ids.get((query.drug_id,query.disease_id),set())
            assert graph.batch.path_batch.num_queries == 1
    query = make_unlabeled_query_supervision('D0','S2')
    assert corpus.assembler.assemble(query,domain='validation').biological_nodes > 0
    with pytest.raises(ValueError,match='Blocked'):
        corpus.assembler.assemble(query,domain='training')


def test_file_tampering_rejected(tmp_path):
    path = make_corpus(tmp_path)
    with (tmp_path/'indptr.npy').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_corpus(path)


def test_observed_cannot_be_global_u(tmp_path):
    path = make_corpus(tmp_path)
    frame = pl.read_parquet(tmp_path/'global_groups.parquet').with_columns(
        pl.when(pl.col('role')=='U').then(pl.lit('D0')).otherwise(pl.col('drug_id')).alias('drug_id'))
    frame.write_parquet(tmp_path/'global_groups.parquet')
    rebind(path,'global_groups.parquet')
    with pytest.raises(ValueError,match='Unlabeled pair is observed'):
        load_corpus(path)


def test_target_map_cannot_omit_any_direct_relation(tmp_path):
    path = make_corpus(tmp_path)
    frame = pl.read_parquet(tmp_path/'target_map.parquet').head(0)
    frame.write_parquet(tmp_path/'target_map.parquet')
    rebind(path,'target_map.parquet')
    with pytest.raises(ValueError,match='every direct/reverse'):
        load_corpus(path)


def test_unavailable_d1_must_be_zero(tmp_path):
    path = make_corpus(tmp_path)
    mask = np.load(tmp_path/'drug_available.npy')
    mask[0] = False
    np.save(tmp_path/'drug_available.npy',mask,allow_pickle=False)
    rebind(path,'drug_available.npy')
    with pytest.raises(ValueError,match='nonzero unavailable'):
        load_corpus(path)


def test_conditional_record_cannot_be_resampled(tmp_path):
    path = make_corpus(tmp_path)
    records = json.loads((tmp_path/'conditional_records.json').read_text())
    records[0]['extra_pairs'] = []
    (tmp_path/'conditional_records.json').write_text(json.dumps(records))
    rebind(path,'conditional_records.json')
    with pytest.raises(ValueError,match='extra-U order'):
        load_corpus(path)
