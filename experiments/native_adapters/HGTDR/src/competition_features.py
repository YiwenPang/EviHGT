"""Frozen, label-free common-domain HGTDR inputs and encoder features.

DrugBank data stay local. Only verified structured SMILES or canonical entity
names are encoded. Missing rows remain present and explicitly masked.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

import numpy as np
import polars as pl

from src.rebuild_features import (MODELS, encode_type, file_sha256, load_encoder,
                                  write_json)
from src.rebuild_smiles import (extract_candidate_smiles, validate_smiles_records,
                                verify_schema_contract)

ROOT = Path(__file__).resolve().parents[1]
EVI = ROOT.parent / 'EviHGT'
DEFAULT_FEATURES = ROOT / 'data/processed/competition_pilot_v1'
HUMAN_TYPES = {'ANA': 'anatomy', 'BPO': 'biological_process',
               'CCO': 'cellular_component', 'DIS': 'disease', 'DRG': 'drug',
               'EXP': 'exposure', 'GEN': 'gene/protein',
               'MFN': 'molecular_function', 'PHE': 'effect/phenotype',
               'PWY': 'pathway'}


def resolved_smiles(original, fallback_records):
    """Existing validated canonical structure wins; ambiguous fallback is absent."""
    if original and original.get('structure_available'):
        return original['smiles'], 'optimus_canonical_smiles', 'available'
    candidates = {r['smiles'] for r in fallback_records if r['structure_available']}
    if len(candidates) == 1:
        return next(iter(candidates)), 'drugbank_structured_smiles', 'available'
    if len(candidates) > 1:
        return None, None, 'conflicting_mapped_structures'
    return None, None, 'structured_smiles_unavailable'


def build_inputs(evi_root=EVI, output=DEFAULT_FEATURES):
    evi_root, output = Path(evi_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_path, receipt_path = output / 'feature_inputs.parquet', output / 'input_manifest.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if file_sha256(input_path) != receipt['feature_inputs_sha256']:
            raise ValueError('Existing feature inputs changed')
        for path, expected in receipt['source_hashes'].items():
            if file_sha256(path) != expected:
                raise ValueError('A bound feature input source changed')
        return receipt
    if input_path.exists():
        raise ValueError('Unreceipted feature inputs exist; refusing silent overwrite')
    index_path = evi_root / 'data/processed/phase3/v1/nodes/node_index.parquet'
    nodes = pl.read_parquet(index_path).sort('global_node_id')
    if not np.array_equal(nodes['global_node_id'].to_numpy(), np.arange(nodes.height)):
        raise ValueError('Global node IDs must be contiguous')
    if set(nodes['node_type']) != set(HUMAN_TYPES):
        raise ValueError('Unexpected node types in the common-domain source')
    sources = {str(index_path): file_sha256(index_path)}
    names, original_smiles = {}, {}
    for kind, component in nodes.select('node_type', 'source_component_path').unique().sort('node_type').iter_rows():
        if not component.startswith('nodes/'):
            continue
        raw = evi_root / 'data/cache/optimuskg/doi_10_7910_DVN_IYNGEV/2.0' / component
        path = raw if raw.exists() else evi_root / 'data/processed/phase3/v1/partitions' / component
        sources[str(path)] = file_sha256(path)
        schema = pl.read_parquet_schema(path)
        def field(key):
            return (pl.col('properties').struct.field(key) if isinstance(schema['properties'], pl.Struct)
                    else pl.col('properties').str.json_path_match(f'$.{key}'))
        expressions = [pl.col('id'), field('name').alias('name')]
        if kind == 'DRG':
            expressions.append(field('canonical_smiles').alias('smiles'))
        frame = pl.scan_parquet(path).select(expressions).collect()
        if frame['id'].n_unique() != frame.height:
            raise ValueError(f'Duplicate canonical source node IDs: {kind}')
        for row in frame.iter_rows(named=True):
            names[(kind, row['id'])] = row['name']
            if kind == 'DRG':
                value = row['smiles']
                original_smiles[row['id']] = {'smiles': value,
                    'structure_available': isinstance(value, str) and bool(value.strip())}
    original_validation = validate_smiles_records(original_smiles)
    supplements_path = evi_root / 'data/processed/phase3/v1/drugbank/supplemental_drug_nodes.parquet'
    db_names_path = evi_root / 'data/interim/drugbank/drugbank_drugs.parquet'
    map_path = evi_root / 'data/processed/phase3/v1/drugbank/canonical_drug_map.parquet'
    for path in (supplements_path, db_names_path, map_path):
        sources[str(path)] = file_sha256(path)
    supplements = pl.read_parquet(supplements_path).join(
        pl.read_parquet(db_names_path, columns=['drugbank_id', 'name']), on='drugbank_id', how='left')
    if supplements['node_id'].n_unique() != supplements.height:
        raise ValueError('Supplemental drug names have an ambiguous canonical mapping')
    for row in supplements.select('node_id', 'name').iter_rows(named=True):
        names[('DRG', row['node_id'])] = row['name']
    mapping = pl.read_parquet(map_path).join(
        nodes.filter(pl.col('node_type') == 'DRG').select(pl.col('node_id').alias('canonical_drug_id')),
        on='canonical_drug_id', how='inner')
    if mapping['drugbank_id'].n_unique() != mapping.height:
        raise ValueError('A DrugBank identifier maps to multiple canonical drugs')
    xml_path, xsd_path = evi_root / 'data/drugbank_all_full_database.xml', evi_root / 'data/drugbank.xsd'
    schema = verify_schema_contract(xsd_path)
    print('Reading local structured DrugBank SMILES; no source strings are logged', flush=True)
    db_records, provenance = extract_candidate_smiles(xml_path, set(mapping['drugbank_id']))
    db_validation = validate_smiles_records(db_records)
    sources[str(xml_path)], sources[str(xsd_path)] = provenance['xml_sha256'], schema['xsd_sha256']
    fallback = {}
    for drugbank_id, canonical in mapping.select('drugbank_id', 'canonical_drug_id').iter_rows():
        fallback.setdefault(canonical, []).append(db_records[drugbank_id])
    rows = []
    for node in nodes.iter_rows(named=True):
        kind, identifier = node['node_type'], node['node_id']
        if kind == 'DRG':
            value, source, status = resolved_smiles(original_smiles.get(identifier), fallback.get(identifier, []))
        else:
            value = names.get((kind, identifier))
            value = value if isinstance(value, str) and value.strip() else None
            source, status = ('canonical_name', 'available') if value is not None else (None, 'canonical_name_unavailable')
        rows.append({k: node[k] for k in ('global_node_id', 'type_local_id', 'node_type', 'node_id')} | {
            'human_node_type': HUMAN_TYPES[kind], 'encoder': 'chemberta' if kind == 'DRG' else 'biobert',
            'encoder_text': value, 'available': value is not None, 'source_field': source, 'input_status': status})
    inputs = pl.DataFrame(rows).sort('global_node_id')
    inputs.write_parquet(input_path, compression='zstd')
    receipt = {'schema': 'HGTDR_COMMON_FEATURE_INPUTS_V1', 'status': 'COMPLETE',
        'feature_inputs_sha256': file_sha256(input_path), 'source_hashes': sources,
        'node_index_sha256': sources[str(index_path)], 'rows': inputs.height,
        'coverage': inputs.group_by('node_type', 'available', 'source_field', 'input_status').len().sort('node_type', 'available').to_dicts(),
        'drugbank_schema': schema, 'drugbank_provenance': provenance,
        'drugbank_smiles_validation': db_validation, 'optimus_smiles_validation': original_validation,
        'clinical_text_encoded': False, 'sealed_final_labels_read': False,
        'missing_policy': 'ALL_NODES_RETAINED_MASKED_SHARED_LEARNABLE_TYPE_VECTOR_ZERO_INITIALIZATION',
        'optimizer_steps': 0}
    write_json(receipt_path, receipt)
    return receipt


def encode_features(evi_root=EVI, output=DEFAULT_FEATURES, *, device='cuda', batch_size=16):
    import torch
    torch.set_num_threads(4)
    start = time.perf_counter()
    output = Path(output).resolve()
    receipt = build_inputs(evi_root, output)
    inputs = pl.read_parquet(output / 'feature_inputs.parquet')
    directory = output / 'features'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {
        'schema': 'HGTDR_COMMON_FIXED_FEATURES_V1', 'status': 'ENCODING',
        'feature_inputs_sha256': receipt['feature_inputs_sha256'], 'types': [], 'encoder_loading': {},
        'fixed_encoder_specs': MODELS, 'training': False, 'optimizer_steps': 0, 'dtype': 'float32',
        'width': 768, 'batch_size': batch_size, 'device': device,
        'missing_policy': receipt['missing_policy'], 'sealed_final_labels_read': False}
    if manifest['feature_inputs_sha256'] != receipt['feature_inputs_sha256']:
        raise ValueError('Cannot mix features from changed canonical inputs')
    finished = {row['node_type'] for row in manifest['types']}
    for row in manifest['types']:
        for key, filename in [('sha256', f"type_{row['node_type_id']:02d}.npy"),
                              ('mask_sha256', f"type_{row['node_type_id']:02d}_available.npy")]:
            if file_sha256(directory / filename) != row[key]:
                raise ValueError('Existing completed feature array changed')
    for encoder in ('chemberta', 'biobert'):
        remaining = [kind for kind in sorted(HUMAN_TYPES) if kind not in finished and ('chemberta' if kind == 'DRG' else 'biobert') == encoder]
        if not remaining:
            continue
        tokenizer, model, loading = load_encoder(encoder, device)
        manifest['encoder_loading'][encoder] = loading
        for kind in remaining:
            selected = inputs.filter(pl.col('node_type') == kind).sort('type_local_id')
            type_id = sorted(HUMAN_TYPES).index(kind)
            synthetic_schema = selected.with_columns(pl.col('type_local_id').alias('type_node_id'),
                pl.lit(type_id).alias('node_type_id'), pl.col('node_id').alias('entity_key'))
            record = encode_type(synthetic_schema, selected['encoder_text'].to_list(),
                selected['available'].to_numpy(), encoder, directory, tokenizer, model, device, batch_size)
            record['human_node_type'] = HUMAN_TYPES[kind]
            manifest['types'].append(record)
            write_json(manifest_path, manifest)
            print(f"Completed {kind}: {record['available_rows']}/{record['rows']} encoded", flush=True)
        del model, tokenizer
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    manifest.update(status='COMPLETE', wall_seconds=time.perf_counter() - start,
                    input_manifest_sha256=file_sha256(output / 'input_manifest.json'))
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evi-root', type=Path, default=EVI)
    parser.add_argument('--output', type=Path, default=DEFAULT_FEATURES)
    parser.add_argument('--inputs-only', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    result = build_inputs(args.evi_root, args.output) if args.inputs_only else encode_features(
        args.evi_root, args.output, device=args.device, batch_size=args.batch_size)
    print(json.dumps({'status': result['status'], 'optimizer_steps': 0}, indent=2))


if __name__ == '__main__':
    main()
