"""Local frozen BioBERT/ChemBERTa feature generation, never model training.

Only model files are fetched over the network. Input names and DrugBank SMILES
are read locally; encoder loading is offline with remote code disabled.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

ROOT = Path(__file__).resolve().parents[1]
PINNED_SMALL_FILES_SHA256 = 'ddf914dbf8f7272641586afa7865bc2072f4f82ca3b3ce25fc287cf4bb75e931'
MODELS = {
    'biobert': {
        'repo': 'dmis-lab/biobert-v1.1',
        'revision': '551ca18efd7f052c8dfa0b01c94c2a8e68bc5488',
        'weight_bytes': 433286112,
        'weight_sha256': '8e87cb27d889186c442feff2bea53f8cce2c5fc24d4bc88bf08ccd3821b138dc',
        'files': ['config.json', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt', 'pytorch_model.bin'],
        'max_length': 128,
    },
    'chemberta': {
        'repo': 'seyonec/ChemBERTa-zinc-base-v1',
        'revision': '761d6a18cf99db371e0b43baf3e2d21b3e865a20',
        'weight_bytes': 178812144,
        'weight_sha256': 'bf8a23d93cca196c3712c94ec379494a74be64c83d86a63b08f03e26db547267',
        'files': ['config.json', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'README.md', 'pytorch_model.bin'],
        'max_length': 512,
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def verify_file_set(directory, expected):
    """Reject a changed tokenizer/config as well as changed checkpoint weights."""
    unexpected = {path.name for path in Path(directory).iterdir()} - set(expected)
    if unexpected:
        raise ValueError(f'Unpinned encoder files could change loading: {sorted(unexpected)}')
    actual = {}
    for name, record in expected.items():
        path = Path(directory) / name
        sha = file_sha256(path)
        size = path.stat().st_size
        if sha != record['sha256'] or size != record['bytes']:
            raise ValueError(f'Pinned encoder file checksum mismatch: {name}')
        actual[name] = {'sha256': sha, 'bytes': size}
    return actual


def verify_encoder_files(key):
    contract_path = ROOT / 'configs' / 'feature_model_files.json'
    if file_sha256(contract_path) != PINNED_SMALL_FILES_SHA256:
        raise ValueError('Pinned model file contract changed')
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    spec = MODELS[key]
    record = next(item for item in contract['models'] if item['model_id'] == spec['repo'])
    if record['revision'] != spec['revision']:
        raise ValueError('Pinned encoder revisions disagree')
    expected = {item['filename']: item for item in record['files']}
    expected['pytorch_model.bin'] = {'bytes': spec['weight_bytes'], 'sha256': spec['weight_sha256']}
    if set(expected) != set(spec['files']):
        raise ValueError('Pinned encoder file lists disagree')
    return verify_file_set(ROOT / 'data' / 'cache' / 'pretrained' / key, expected)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def download_models():
    records = {}
    for key, spec in MODELS.items():
        directory = ROOT / 'data' / 'cache' / 'pretrained' / key
        directory.mkdir(parents=True, exist_ok=True)
        files = {}
        for filename in spec['files']:
            path = directory / filename
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{filename}"
            if not path.exists():
                partial = path.with_suffix(path.suffix + '.part')
                print(f'Downloading fixed model: {key}/{filename}', flush=True)
                request = urllib.request.Request(url, headers={'User-Agent': 'HGTDR-local-data-reconstruction'})
                with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as output:
                    total, last = 0, time.monotonic()
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                        total += len(block)
                        if time.monotonic() - last > 15:
                            print(f'{key}/{filename}: {total} bytes', flush=True)
                            last = time.monotonic()
                if filename == 'pytorch_model.bin':
                    if partial.stat().st_size != spec['weight_bytes'] or file_sha256(partial) != spec['weight_sha256']:
                        raise ValueError(f'{key} checkpoint checksum mismatch; partial file retained')
                partial.rename(path)
            sha = file_sha256(path)
            if filename == 'pytorch_model.bin' and (sha != spec['weight_sha256'] or path.stat().st_size != spec['weight_bytes']):
                raise ValueError(f'{key} cached checkpoint differs; not overwriting it')
            files[filename] = {'url': url, 'bytes': path.stat().st_size, 'sha256': sha}
        verified = verify_encoder_files(key)
        records[key] = {'directory': str(directory), **spec, 'downloaded_files': files,
                        'verified_pinned_files': verified}
    write_json(ROOT / 'outputs' / 'reconstruction' / 'model_sources.json', records)
    return records


def masked_token_mean(hidden, attention_mask, special_tokens_mask):
    import torch
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError('Encoder states must be finite')
    mask = attention_mask.bool() & ~special_tokens_mask.bool()
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError('Encoder input has no content token')
    return (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


def validate_row_ids(row_ids):
    if list(row_ids) != list(range(len(row_ids))):
        raise ValueError('Type-local feature row IDs must be unique, sorted and contiguous')


def validate_encoder_inputs(texts):
    if any(not isinstance(value, str) or not value.strip() for value in texts):
        raise ValueError('Refusing missing or empty encoder inputs')


def load_encoder(key, device):
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = ROOT / 'data' / 'cache' / 'pretrained' / key
    verified = verify_encoder_files(key)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model, info = AutoModel.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        use_safetensors=False, weights_only=True, add_pooling_layer=False,
        output_loading_info=True, dtype=torch.float32)
    if info.get('missing_keys') or info.get('mismatched_keys'):
        raise ValueError(f'Encoder parameters missing or mismatched: {info}')
    if model.config.hidden_size != 768:
        raise ValueError('This reconstruction requires 768-dimensional hidden states')
    model.eval().requires_grad_(False).to(device)
    return tokenizer, model, {'parameters': json_safe(info), 'verified_pinned_files': verified,
                              'file_contract_sha256': PINNED_SMALL_FILES_SHA256}


def encode_type(nodes, texts, available, key, directory, tokenizer, model, device, batch_size):
    import numpy as np
    import torch
    node_type = nodes['node_type'][0]
    type_id = int(nodes['node_type_id'][0])
    row_ids = nodes['type_node_id'].to_list()
    validate_row_ids(row_ids)
    n = len(row_ids)
    if len(texts) != n or len(available) != n:
        raise ValueError('Text, availability and node rows are misaligned')
    valid_ids = np.flatnonzero(available)
    validate_encoder_inputs([texts[i] for i in valid_ids])
    output_path = directory / f'type_{type_id:02d}.npy'
    partial = directory / f'type_{type_id:02d}.part.npy'
    if output_path.exists():
        raise FileExistsError(f'Refusing to replace completed features: {output_path}')
    output = np.lib.format.open_memmap(partial, mode='w+', dtype='float32', shape=(n, 768))
    output[:] = 0.0  # Explicit missing-storage placeholder, never called a generated embedding.
    truncated, last_print = 0, time.monotonic()
    max_length = MODELS[key]['max_length']
    for offset in range(0, len(valid_ids), batch_size):
        indices = valid_ids[offset:offset + batch_size]
        batch_text = [texts[i] for i in indices]
        raw_lengths = tokenizer(batch_text, add_special_tokens=True, truncation=False)['input_ids']
        truncated += sum(len(ids) > max_length for ids in raw_lengths)
        encoded = tokenizer(batch_text, padding=True, truncation=True, max_length=max_length,
                            return_tensors='pt', return_special_tokens_mask=True)
        special = encoded.pop('special_tokens_mask').to(device)
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            pooled = masked_token_mean(hidden, encoded['attention_mask'], special)
        output[indices] = pooled.cpu().numpy()
        if time.monotonic() - last_print > 15:
            output.flush()
            print(f'Encoded {node_type}: {min(offset + batch_size, len(valid_ids))}/{len(valid_ids)}', flush=True)
            last_print = time.monotonic()
        if device == 'cuda':
            time.sleep(0.002)
    output.flush()
    del output
    partial.rename(output_path)
    mask_path = directory / f'type_{type_id:02d}_available.npy'
    np.save(mask_path, np.asarray(available, dtype=bool), allow_pickle=False)
    input_digest = hashlib.sha256()
    for entity_key, text, present in zip(nodes['entity_key'], texts, available):
        input_digest.update(json.dumps([entity_key, text if present else None, bool(present)], ensure_ascii=False).encode('utf-8'))
        input_digest.update(b'\n')
    return {
        'node_type': node_type, 'node_type_id': type_id, 'rows': n, 'width': 768,
        'available_rows': int(np.count_nonzero(available)), 'missing_rows': int(n - np.count_nonzero(available)),
        'encoder': key, 'truncated_rows': truncated, 'input_sha256': input_digest.hexdigest(),
        'array': str(output_path), 'sha256': file_sha256(output_path),
        'availability_mask': str(mask_path), 'mask_sha256': file_sha256(mask_path),
    }


def build_features(dataset, device='cpu', batch_size=16):
    import importlib.metadata
    import numpy as np
    import polars as pl
    import torch
    dataset = Path(dataset)
    nodes = pl.read_parquet(dataset / 'nodes.parquet').sort(['node_type_id', 'type_node_id'])
    smiles = pl.read_parquet(dataset / 'drug_smiles.parquet')
    # The extraction contract is explicit; no label or clinical-description columns are inputs.
    required = {'node_id', 'smiles', 'structure_available'}
    if not required.issubset(smiles.columns):
        raise ValueError(f'SMILES table needs {sorted(required)}, found {smiles.columns}')
    if smiles['node_id'].n_unique() != len(smiles):
        raise ValueError('Duplicate DrugBank feature rows')
    source_hashes = {name: file_sha256(dataset / name) for name in ['nodes.parquet', 'drug_smiles.parquet']}
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(0.12)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
    directory = dataset / 'features'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('status') == 'COMPLETE':
            raise FileExistsError('Completed feature dataset exists; use verification, not silent overwriting')
    started = time.monotonic()
    report = {
        'status': 'BUILDING', 'protocol': 'HGTDR_SEMANTIC_REBUILD_FEATURES_V1',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_hashes': source_hashes, 'models': MODELS,
        'pooling': 'last_hidden_state mean over nonpadding, nonspecial tokens',
        'dtype': 'float32', 'device': device, 'batch_size': batch_size,
        'missing_policy': 'zero storage placeholder with separate availability mask; no imputed molecule or random embedding',
        'original_author_checkpoints_and_pooling': 'UNRESOLVED; these are explicitly chosen reconstruction settings',
        'original_notebook_drug_width': 767, 'reconstructed_hidden_width': 768,
        'width_note': 'Full hidden state retained. No arbitrary dimension truncation and no MLM vocabulary logits.',
        'network_receives_input_text': False, 'training': False, 'optimizer_steps': 0,
        'packages': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'tokenizers', 'safetensors']},
        'types': [],
    }
    write_json(manifest_path, report)
    for encoder in ['biobert', 'chemberta']:
        tokenizer, model, info = load_encoder(encoder, device)
        report.setdefault('loading_info', {})[encoder] = info
        for node_type in nodes['node_type'].unique(maintain_order=True):
            if (node_type == 'drug') != (encoder == 'chemberta'):
                continue
            selected = nodes.filter(pl.col('node_type') == node_type).sort('type_node_id')
            if encoder == 'chemberta':
                joined = selected.join(smiles.select(['node_id', 'smiles', 'structure_available']), on='node_id', how='left').sort('type_node_id')
                texts = joined['smiles'].to_list()
                available = joined['structure_available'].fill_null(False).to_numpy()
            else:
                texts = selected['source_name'].to_list()
                available = np.asarray([isinstance(value, str) and bool(value.strip()) for value in texts])
            print(f'Encoding {node_type}: {int(available.sum())}/{len(selected)} real inputs', flush=True)
            record = encode_type(selected, texts, available, encoder, directory, tokenizer, model, device, batch_size)
            report['types'].append(record)
            write_json(manifest_path, report)
        del tokenizer, model
        if device == 'cuda':
            torch.cuda.empty_cache()
    if any(file_sha256(dataset / name) != sha for name, sha in source_hashes.items()):
        raise ValueError('Feature input tables changed during encoding')
    report.update(status='COMPLETE', source_tables_unchanged=True, elapsed_seconds=time.monotonic() - started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0)
    write_json(manifest_path, report)
    write_json(ROOT / 'outputs' / 'reconstruction' / 'features_report.json', report)
    print(json.dumps({'status': report['status'], 'rows': sum(r['rows'] for r in report['types']),
                      'missing_rows': sum(r['missing_rows'] for r in report['types']),
                      'elapsed_seconds': report['elapsed_seconds'], 'optimizer_steps': 0}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data' / 'processed' / 'reconstructed_v1')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 64:
        parser.error('batch-size must be 1..64')
    if args.download_only:
        download_models()
    else:
        build_features(args.dataset, args.device, args.batch_size)


if __name__ == '__main__':
    main()
