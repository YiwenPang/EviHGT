"""Reconstruct SideInfo and portable D1 directly from local verified raw sources."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ..data import raw_features as raw
from ..data.pretrained import ENCODER_SPECS
from . import feature_encoders as encoders
from . import feature_sources as source


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def _save_array(path, array):
    temporary = path.with_name(path.stem + '.partial.npy')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()


def _packages():
    result = {}
    for name in ('numpy', 'polars', 'pyarrow', 'rdkit', 'torch', 'transformers', 'tokenizers'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'NOT_INSTALLED'
    return result


def _raw_properties(path, requested):
    """Scan aggregate nodes in bounded batches; retain only the selected fields."""
    properties = {'DRG': {}, 'DIS': {}}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=['id', 'label', 'properties']):
        frame = pl.from_arrow(batch).filter(pl.col('label').is_in(['DRG', 'DIS']))
        for identifier, kind, serialized in frame.iter_rows():
            if identifier not in requested[kind]:
                continue
            if identifier in properties[kind]:
                raise ValueError('Duplicate canonical source node ID: ' + kind)
            value = json.loads(serialized) if isinstance(serialized, str) else serialized
            if not isinstance(value, dict):
                raise ValueError('Aggregate node properties must be a JSON object')
            fields = ('canonical_smiles',) if kind == 'DRG' else ('name', 'parents', 'ancestors')
            properties[kind][identifier] = {field: value.get(field) for field in fields}
    return properties


def _profiles(canonical):
    biology, phenotypes = defaultdict(set), defaultdict(set)
    for component in ('drug_gene', 'drugbank_drug_gene'):
        raw.require_feature_source(component)
        frame = pl.read_parquet(canonical / f'canonical_facts/{component}.parquet',
                                columns=['source_node_type', 'source_node_id', 'target_node_type', 'target_node_id', 'relation'])
        for st, si, tt, ti, relation in frame.iter_rows():
            if st != 'DRG' or tt != 'GEN':
                raise ValueError('Unexpected drug biology endpoint type')
            biology[si].add(relation + ':' + ti)
    admitted = set()
    path = canonical / 'evidence_records/disease_phenotype.parquet'
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=['canonical_fact_id', 'relation', 'raw_indirect_sources', 'properties']):
        for fid, relation, sources, serialized in pl.from_arrow(batch).iter_rows():
            prop = json.loads(serialized) if isinstance(serialized, str) else serialized
            if 'HPO' in (sources or []) and relation == 'PHENOTYPE_PRESENT' and prop.get('qualifier_not') is False:
                admitted.add(fid)
    facts = pl.read_parquet(canonical / 'canonical_facts/disease_phenotype.parquet',
                            columns=['canonical_fact_id', 'source_node_type', 'source_node_id', 'target_node_type', 'target_node_id'])
    for fid, st, si, tt, ti in facts.iter_rows():
        if fid not in admitted:
            continue
        if st != 'DIS' or tt != 'PHE':
            raise ValueError('Unexpected disease phenotype endpoint type')
        phenotypes[si].add(ti)
    return biology, phenotypes


def _check_completed(directory, record):
    for filename, digest in record['files'].items():
        path = directory / filename
        if not path.is_file() or source.sha256_file(path) != digest:
            raise ValueError('Completed feature output changed: ' + filename)


def _make_sideinfo(canonical, selected, props, directory, progress):
    biology, phenotypes = _profiles(canonical)
    statistics = {}
    last = time.monotonic()
    for kind, prefix, width in (('DRG', 'drug', raw.DRUG_DIM), ('DIS', 'disease', raw.DISEASE_DIM)):
        nodes = selected[kind]
        values = np.zeros((nodes.height, width), dtype=np.uint8)
        for index, identifier in enumerate(nodes['node_id']):
            prop = props[kind].get(identifier, {})
            if kind == 'DRG':
                fp, available = raw.fingerprint(prop.get('canonical_smiles'))
                values[index] = raw.pack_drug(fp, raw.profile(biology[identifier], 'drug-biology'), available, bool(biology[identifier]))
            else:
                terms, available = raw.ontology_input(parents=prop.get('parents'), ancestors=prop.get('ancestors'))
                values[index] = raw.pack_disease(raw.profile(terms, 'disease-ontology'), raw.profile(phenotypes[identifier], 'disease-HPO'), available, bool(phenotypes[identifier]))
            if progress and time.monotonic()-last >= 5:
                progress('确定性实体特征 ' + kind, index+1, nodes.height, 0)
                last = time.monotonic()
        _save_array(directory / (prefix + '_values.npy'), values)
        _save_array(directory / (prefix + '_ids.npy'), nodes['global_node_id'].to_numpy())
        statistics[kind] = {'rows': nodes.height, 'family_available': values[:, -2:].sum(axis=0).tolist()}
    return statistics


def _d1_inputs(canonical, selected, props, xml_path, xsd_path, progress=None):
    schema = source.verify_schema_contract(xsd_path)
    original = {identifier: {'smiles': row.get('canonical_smiles'),
                             'structure_available': isinstance(row.get('canonical_smiles'), str) and bool(row['canonical_smiles'].strip())}
                for identifier, row in props['DRG'].items()}
    original_validation = source.validate_smiles_records(original)
    mapping = pl.read_parquet(canonical / 'drugbank/canonical_drug_map.parquet', columns=['drugbank_id', 'canonical_drug_id'])
    mapping = mapping.filter(pl.col('canonical_drug_id').is_in(selected['DRG']['node_id'].implode()))
    if mapping['drugbank_id'].n_unique() != mapping.height:
        raise ValueError('A DrugBank identifier maps to multiple canonical drugs')
    if mapping.height:
        records, provenance = source.extract_candidate_smiles(xml_path, set(mapping['drugbank_id']), progress=progress)
        validation = source.validate_smiles_records(records)
    else:
        records, provenance, validation = {}, {'candidate_ids': 0}, {'syntax_valid': 0}
    fallback = defaultdict(list)
    for dbid, canonical_id in mapping.iter_rows():
        fallback[canonical_id].append(records[dbid])
    texts, masks, digests = {}, {}, {}
    for kind in ('DRG', 'DIS'):
        rows = []
        for identifier in selected[kind]['node_id']:
            if kind == 'DRG':
                text = source.resolved_smiles(original.get(identifier), fallback[identifier])[0]
            else:
                text = props[kind].get(identifier, {}).get('name')
                text = text if isinstance(text, str) and text.strip() else None
            rows.append(text)
        texts[kind] = rows
        masks[kind] = np.asarray([value is not None for value in rows], dtype=bool)
        digests[kind] = _identity_hash(list(zip(selected[kind]['global_node_id'].to_list(), rows)))
    return texts, masks, {'input_text_identity_sha256': digests, 'drugbank_schema': schema,
                          'drugbank_provenance': provenance, 'drugbank_smiles_validation': validation,
                          'optimus_smiles_validation': original_validation}


def build_features(canonical_root, nodes_path, drugbank_xml, destination, *, model_cache,
                   device='cpu', batch_size=16, offline=False, progress=None,
                   encoder_loader=None, drugbank_xsd=None):
    """Build source-bound mmap arrays; resume completed types and reject changed inputs.

    ``encoder_loader`` is a dependency-injection hook for deterministic integration
    tests. Production calls use the fixed official file contract automatically.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Feature batch size must be positive')
    canonical, nodes_path, xml_path, destination = map(lambda p: Path(p).resolve(),
                                                       (canonical_root, nodes_path, drugbank_xml, destination))
    xsd_path = Path(drugbank_xsd).resolve() if drugbank_xsd else xml_path.with_name('drugbank.xsd')
    source_files = {'aggregate_nodes': nodes_path, 'drugbank_xml': xml_path, 'drugbank_xsd': xsd_path}
    for name in ('nodes/node_index.parquet', 'canonical_facts/drug_gene.parquet',
                 'canonical_facts/drugbank_drug_gene.parquet', 'canonical_facts/disease_phenotype.parquet',
                 'evidence_records/disease_phenotype.parquet', 'drugbank/canonical_drug_map.parquet'):
        source_files[name] = canonical / name
    if progress:
        progress('核验特征源文件', 0, len(source_files), 0)
    hashes = {key: source.sha256_file(path) for key, path in source_files.items()}
    recipe = {path.name: source.sha256_file(path) for path in
              (Path(__file__), Path(source.__file__), Path(encoders.__file__), Path(raw.__file__), encoders.CONTRACT)}
    loader_identity = 'official_pinned_encoders'
    if encoder_loader is not None:
        try:
            loader_code = inspect.getsource(encoder_loader)
        except (OSError, TypeError):
            loader_code = repr(encoder_loader)
        loader_identity = _identity_hash({'module': encoder_loader.__module__, 'code': loader_code})
    identity = {'sources': hashes, 'recipe': recipe, 'packages': _packages(),
                'device': str(device), 'batch_size': batch_size, 'encoder_loader': loader_identity}
    receipt_path = destination / 'feature_stage.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text('utf8'))
        if receipt['identity'] != identity:
            raise ValueError('Feature source or build identity changed; use a new destination')
        for stage in receipt['completed'].values():
            _check_completed(destination, stage)
        if receipt['status'] == 'COMPLETE':
            return _result(destination)
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError('Unreceipted feature outputs exist; use an empty destination')
        destination.mkdir(parents=True, exist_ok=True)
        receipt = {'schema': 'EVIHGT_RAW_FEATURE_BUILD_V1', 'status': 'BUILDING', 'identity': identity, 'completed': {}}
        _write_json(receipt_path, receipt)
    for directory in ('raw', 'pretrained'):
        (destination / directory).mkdir(exist_ok=True)
    selected = {kind: pl.scan_parquet(canonical / 'nodes/node_index.parquet').filter(pl.col('node_type') == kind)
                .select('node_id', 'global_node_id').sort('global_node_id').collect() for kind in ('DRG', 'DIS')}
    for kind, frame in selected.items():
        if frame['node_id'].n_unique() != frame.height or frame['global_node_id'].n_unique() != frame.height:
            raise ValueError('Duplicate feature node identity: ' + kind)
    props = _raw_properties(nodes_path, {kind: set(frame['node_id']) for kind, frame in selected.items()})

    def complete(stage, paths, metadata):
        receipt['completed'][stage] = {'files': {str(path.relative_to(destination)).replace('\\', '/'): source.sha256_file(path) for path in paths},
                                       'metadata': metadata}
        _write_json(receipt_path, receipt)

    if 'raw' not in receipt['completed']:
        info = _make_sideinfo(canonical, selected, props, destination / 'raw', progress)
        complete('raw', list((destination / 'raw').glob('*.npy')), info)
    texts, masks, provenance = _d1_inputs(canonical, selected, props, xml_path, xsd_path, progress=progress)
    loader = encoder_loader or encoders.load_encoder
    for kind, prefix, key in (('DRG', 'drug', 'chemberta'), ('DIS', 'disease', 'biobert')):
        if kind in receipt['completed']:
            continue
        folder = destination / 'pretrained'
        values_path = folder / (prefix + '_values.npy')
        # An interrupted publication has no per-type receipt. Preserve it for
        # diagnosis rather than accepting or replacing unverified embeddings.
        if values_path.exists():
            raise ValueError('Unreceipted completed encoder array exists: ' + values_path.name)
        if encoder_loader is None:
            encoders.prepare_encoder(key, model_cache, offline=offline, progress=progress)
        tokenizer, model, load_info = loader(key, model_cache, device, offline)
        info = encoders.encode_texts(texts[kind], masks[kind], key, values_path, tokenizer, model,
                                     device=device, batch_size=batch_size, progress=progress)
        del tokenizer, model
        if str(device).startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
        _save_array(folder / (prefix + '_ids.npy'), selected[kind]['global_node_id'].to_numpy())
        _save_array(folder / (prefix + '_available.npy'), masks[kind])
        complete(kind, [folder / (prefix + suffix) for suffix in ('_ids.npy', '_values.npy', '_available.npy')],
                 {'encoding': info, 'loading': load_info})
    metadata = {'schema': 'EVIHGT_D1_FEATURES_V1', 'width': 768, 'fixed_encoder_specs': ENCODER_SPECS,
                'clinical_text_encoded': False, 'supervised_dda_features_present': False,
                'trained_node_embeddings_loaded': False, 'prediction_scores_loaded': False,
                'optimizer_steps': 0, 'upstream_pretraining_target_exposure': 'NOT_VERIFIED',
                'pooling': 'last_hidden_state mean over nonpadding, nonspecial tokens',
                'input_text_sent_to_network': False, 'device': str(device), 'batch_size': batch_size,
                'packages': identity['packages'], 'source_hashes': hashes,
                'synthetic_test_only': encoder_loader is not None,
                'encoding': {kind: receipt['completed'][kind]['metadata'] for kind in ('DRG', 'DIS')}, **provenance}
    metadata_path = destination / 'pretrained/metadata.json'
    _write_json(metadata_path, metadata)
    if hashes != {key: source.sha256_file(path) for key, path in source_files.items()}:
        raise ValueError('Feature source changed during reconstruction')
    complete('metadata', [metadata_path], {'source_files_unchanged': True})
    receipt['status'] = 'COMPLETE'
    _write_json(receipt_path, receipt)
    return _result(destination)


def _result(destination):
    return {'raw_features': {name: str(destination / 'raw' / (name + '.npy')) for name in
                              ('drug_ids', 'drug_values', 'disease_ids', 'disease_values')},
            'pretrained': {name: str(destination / 'pretrained' / (name + ('.json' if name == 'metadata' else '.npy')))
                           for name in ('drug_ids', 'drug_values', 'drug_available', 'disease_ids',
                                        'disease_values', 'disease_available', 'metadata')}}
