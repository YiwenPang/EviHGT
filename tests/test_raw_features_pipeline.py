"""Raw feature reconstruction uses synthetic source data, never licensed fixtures."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from evi_hgt.preprocessing.features import build_features
from evi_hgt.preprocessing.feature_encoders import masked_token_mean, verify_file_set
from evi_hgt.preprocessing.feature_sources import extract_candidate_smiles, resolved_smiles


class TinyTokenizer:
    def __call__(self, texts, *, add_special_tokens=True, truncation=False,
                 padding=False, max_length=None, return_tensors=None,
                 return_special_tokens_mask=False):
        rows = [[1000] + [ord(c) for c in text] + [2000] for text in texts]
        if truncation:
            rows = [row[:max_length] for row in rows]
        if not return_tensors:
            return {'input_ids': rows}
        width = max(map(len, rows))
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention = torch.zeros_like(ids)
        special = torch.ones_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
            attention[i, :len(row)] = 1
            special[i, 1:len(row)-1] = 0
        return {'input_ids': ids, 'attention_mask': attention,
                'special_tokens_mask': special}


class TinyEncoder(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1, -1, 768))


def tiny_loader(key, cache, device, offline):
    return TinyTokenizer(), TinyEncoder().to(device), {'synthetic_test_only': True}


@pytest.fixture
def sources(tmp_path):
    canonical = tmp_path / 'canonical'
    for name in ['nodes', 'canonical_facts', 'evidence_records', 'drugbank']:
        (canonical / name).mkdir(parents=True)
    nodes = [
        {'node_id': 'DIS:1', 'node_type': 'DIS', 'global_node_id': 0},
        {'node_id': 'DIS:2', 'node_type': 'DIS', 'global_node_id': 1},
        {'node_id': 'DRG:1', 'node_type': 'DRG', 'global_node_id': 2},
        {'node_id': 'DRG:2', 'node_type': 'DRG', 'global_node_id': 3},
        {'node_id': 'DRG:3', 'node_type': 'DRG', 'global_node_id': 4},
        {'node_id': 'GEN:1', 'node_type': 'GEN', 'global_node_id': 5},
    ]
    pl.DataFrame(nodes).write_parquet(canonical / 'nodes/node_index.parquet')
    raw = tmp_path / 'nodes.parquet'
    pl.DataFrame([
        {'id': 'DIS:1', 'label': 'DIS', 'properties': json.dumps({'name': 'alpha', 'parents': [], 'ancestors': None})},
        {'id': 'DIS:2', 'label': 'DIS', 'properties': json.dumps({'name': None, 'parents': None, 'ancestors': None})},
        {'id': 'DRG:1', 'label': 'DRG', 'properties': json.dumps({'name': 'drug one', 'canonical_smiles': 'CC'})},
        {'id': 'DRG:2', 'label': 'DRG', 'properties': json.dumps({'canonical_smiles': None})},
        {'id': 'DRG:3', 'label': 'DRG', 'properties': json.dumps({'canonical_smiles': None})},
    ]).write_parquet(raw)
    facts = pl.DataFrame({'source_node_type': ['DRG'], 'source_node_id': ['DRG:1'],
                          'target_node_type': ['GEN'], 'target_node_id': ['GEN:1'],
                          'relation': ['TARGET']})
    facts.write_parquet(canonical / 'canonical_facts/drug_gene.parquet')
    facts.head(0).write_parquet(canonical / 'canonical_facts/drugbank_drug_gene.parquet')
    pl.DataFrame({'canonical_fact_id': [10, 11, 12], 'source_node_type': ['DIS']*3,
                  'source_node_id': ['DIS:1']*3, 'target_node_type': ['PHE']*3,
                  'target_node_id': ['HP:1', 'HP:2', 'HP:3']}).write_parquet(
        canonical / 'canonical_facts/disease_phenotype.parquet')
    pl.DataFrame({'canonical_fact_id': [10, 11, 12],
                  'relation': ['PHENOTYPE_PRESENT']*3,
                  'raw_indirect_sources': [['HPO'], ['HPO'], ['OTHER']],
                  'properties': ['{"qualifier_not":false,"frequency":0}', '{"qualifier_not":true}', '{"qualifier_not":false}']}).write_parquet(
        canonical / 'evidence_records/disease_phenotype.parquet')
    pl.DataFrame({'drugbank_id': ['DB1', 'DB2', 'DB3'],
                  'canonical_drug_id': ['DRG:2', 'DRG:3', 'DRG:3']}).write_parquet(
        canonical / 'drugbank/canonical_drug_map.parquet')
    xml = tmp_path / 'drugbank.xml'
    xml.write_text('''<drugbank xmlns="http://www.drugbank.ca" version="TEST">
    <drug><drugbank-id primary="true">DB1</drugbank-id><calculated-properties><property><kind>SMILES</kind><value>CCN</value><source>synthetic</source></property></calculated-properties></drug>
    <drug><drugbank-id primary="true">DB2</drugbank-id><calculated-properties><property><kind>SMILES</kind><value>CCO</value></property></calculated-properties></drug>
    <drug><drugbank-id primary="true">DB3</drugbank-id><calculated-properties><property><kind>SMILES</kind><value>CCCC</value></property></calculated-properties></drug>
    </drugbank>''', encoding='utf8')
    (tmp_path / 'drugbank.xsd').write_text('''<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="http://www.drugbank.ca">
      <xs:complexType name="drug-type"><xs:sequence><xs:element name="drugbank-id"/><xs:element name="calculated-properties"/></xs:sequence></xs:complexType>
      <xs:complexType name="calculated-property-type"><xs:sequence><xs:element name="value" type="xs:string"/><xs:element name="source"/></xs:sequence></xs:complexType>
      <xs:complexType name="drugbank-drug-salt-id-type"><xs:attribute name="primary" type="xs:boolean"/></xs:complexType>
      <xs:simpleType name="calculated-property-kind-type"><xs:restriction base="xs:string"><xs:enumeration value="SMILES"/></xs:restriction></xs:simpleType>
    </xs:schema>''', encoding='utf8')
    return canonical, raw, xml


def test_raw_sources_produce_portable_sideinfo_and_d1(sources, tmp_path):
    canonical, raw, xml = sources
    result = build_features(canonical, raw, xml, tmp_path / 'features',
                            model_cache=tmp_path / 'cache', encoder_loader=tiny_loader)
    raw_paths, d1 = result['raw_features'], result['pretrained']
    drugs = np.load(raw_paths['drug_values'])
    diseases = np.load(raw_paths['disease_values'])
    assert drugs.shape == (3, 2562) and diseases.shape == (2, 1026)
    assert drugs[0, -2:].tolist() == [1, 1]
    assert drugs[1:, -2:].tolist() == [[0, 0], [0, 0]]  # No DrugBank fallback in SideInfo.
    assert diseases[0, -2:].tolist() == [1, 1]  # Empty ontology and frequency-zero HPO remain admitted.
    assert diseases[1, -2:].tolist() == [0, 0]
    assert np.load(d1['drug_available']).tolist() == [True, True, False]
    assert np.load(d1['disease_available']).tolist() == [True, False]
    vectors = np.load(d1['drug_values'])
    assert np.all(vectors[0] == ord('C'))
    assert np.allclose(vectors[1], (ord('C')*2+ord('N'))/3)
    assert not np.any(vectors[2])
    meta = json.loads(Path(d1['metadata']).read_text('utf8'))
    assert meta['clinical_text_encoded'] is False and meta['optimizer_steps'] == 0
    assert meta['upstream_pretraining_target_exposure'] == 'NOT_VERIFIED'


def test_completed_feature_stage_reuse_and_changed_source_rejection(sources, tmp_path):
    canonical, raw, xml = sources
    args = (canonical, raw, xml, tmp_path / 'features')
    result = build_features(*args, model_cache=tmp_path / 'cache', encoder_loader=tiny_loader)
    assert build_features(*args, model_cache=tmp_path / 'cache', encoder_loader=tiny_loader) == result
    xml.write_text(xml.read_text('utf8')+'\n', encoding='utf8')
    with pytest.raises(ValueError, match='source|identity|changed'):
        build_features(*args, model_cache=tmp_path / 'cache', encoder_loader=tiny_loader)


def test_masked_token_pooling_excludes_padding_and_special_tokens():
    hidden = torch.tensor([[[99.0], [2.0], [4.0], [88.0], [77.0]]])
    pooled = masked_token_mean(hidden, torch.tensor([[1, 1, 1, 1, 0]]), torch.tensor([[1, 0, 0, 1, 0]]))
    assert pooled.item() == 3.0
    with pytest.raises(ValueError, match='content token'):
        masked_token_mean(hidden, torch.zeros((1,5)), torch.zeros((1,5)))


def test_structured_smiles_conflicts_and_optimum_precedence(sources):
    records, _ = extract_candidate_smiles(sources[2], {'DB1', 'DB2', 'MISSING'})
    assert records['DB1']['smiles'] == 'CCN'
    assert records['MISSING']['structure_available'] is False
    assert resolved_smiles({'structure_available': True, 'smiles': 'CC'}, [records['DB1']])[0] == 'CC'
    assert resolved_smiles(None, [records['DB1'], records['DB2']])[0] is None


def test_model_cache_rejects_corruption_and_unpinned_files(tmp_path):
    import hashlib
    path=tmp_path/'config.json';path.write_bytes(b'{}')
    contract={'config.json':{'bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}}
    assert verify_file_set(tmp_path,contract)['config.json']['bytes']==2
    path.write_bytes(b'[]')
    with pytest.raises(ValueError,match='checksum'):
        verify_file_set(tmp_path,contract)
    path.write_bytes(b'{}');(tmp_path/'model.py').write_text('')
    with pytest.raises(ValueError,match='Unpinned'):
        verify_file_set(tmp_path,contract)


def test_completed_array_corruption_is_not_silently_reused(sources, tmp_path):
    canonical, raw, xml = sources
    args = (canonical, raw, xml, tmp_path / 'features')
    result = build_features(*args, model_cache=tmp_path / 'cache', encoder_loader=tiny_loader)
    path = Path(result['pretrained']['drug_values'])
    values = np.load(path)
    values[0, 0] += 1
    np.save(path, values)
    with pytest.raises(ValueError, match='output changed'):
        build_features(*args, model_cache=tmp_path / 'cache', encoder_loader=tiny_loader)


def test_interruption_resumes_completed_type_without_reencoding(sources, tmp_path):
    calls, interrupted = [], [False]

    def loader(key, cache, device, offline):
        calls.append(key)
        if key == 'biobert' and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError('Synthetic interruption')
        return tiny_loader(key, cache, device, offline)

    canonical, raw, xml = sources
    args = (canonical, raw, xml, tmp_path / 'features')
    with pytest.raises(RuntimeError, match='Synthetic interruption'):
        build_features(*args, model_cache=tmp_path / 'cache', encoder_loader=loader)
    result = build_features(*args, model_cache=tmp_path / 'cache', encoder_loader=loader)
    assert calls == ['chemberta', 'biobert', 'biobert']
    assert np.load(result['pretrained']['drug_available']).tolist() == [True, True, False]


def test_encoder_batching_freezes_model_and_preserves_missing_zeros(tmp_path):
    from evi_hgt.preprocessing.feature_encoders import encode_texts

    class TrainableEncoder(TinyEncoder):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids, attention_mask):
            assert not self.training and not self.bias.requires_grad
            assert not torch.is_grad_enabled()
            return super().forward(input_ids, attention_mask)

    texts, mask = ['CC', None, 'COC', 'N'], [True, False, True, True]
    arrays = []
    for batch_size in (1, 3):
        path = tmp_path / f'{batch_size}.npy'
        info = encode_texts(texts, mask, 'chemberta', path, TinyTokenizer(), TrainableEncoder(),
                            device='cpu', batch_size=batch_size)
        assert info['available_rows'] == 3 and info['missing_rows'] == 1
        arrays.append(np.load(path))
    np.testing.assert_array_equal(*arrays)
    assert np.all(arrays[0][1] == 0)


def test_offline_encoder_requires_exact_pinned_files(tmp_path):
    from evi_hgt.preprocessing.feature_encoders import prepare_encoder
    with pytest.raises(FileNotFoundError, match='offline'):
        prepare_encoder('chemberta', tmp_path, offline=True)


def test_raw_schema_mismatch_is_not_accepted(sources, tmp_path):
    canonical, raw, xml = sources
    xsd = xml.with_name('drugbank.xsd')
    xsd.write_text(xsd.read_text('utf8').replace('value="SMILES"', 'value="MASS"'), encoding='utf8')
    with pytest.raises(ValueError, match='schema contract mismatch'):
        build_features(canonical, raw, xml, tmp_path / 'features',
                       model_cache=tmp_path / 'cache', encoder_loader=tiny_loader)
