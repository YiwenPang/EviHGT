"""Pinned local frozen encoders; only public model files leave the network boundary."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from ..data.pretrained import ENCODER_SPECS
from .feature_sources import sha256_file
from .resources import ensure_cache_files

CONTRACT = Path(__file__).resolve().parents[3] / 'configs/feature_model_files.json'
CONTRACT_SHA256 = 'ddf914dbf8f7272641586afa7865bc2072f4f82ca3b3ce25fc287cf4bb75e931'
WEIGHTS = {
    'chemberta': (178812144, 'bf8a23d93cca196c3712c94ec379494a74be64c83d86a63b08f03e26db547267'),
    'biobert': (433286112, '8e87cb27d889186c442feff2bea53f8cce2c5fc24d4bc88bf08ccd3821b138dc'),
}


def verify_file_set(directory, expected):
    """Reject changed weights, tokenizer settings, and any unpinned loading files."""
    directory = Path(directory)
    unexpected = {p.name for p in directory.iterdir()} - set(expected)
    if unexpected:
        raise ValueError(f'Unpinned encoder files could change loading: {sorted(unexpected)}')
    actual = {}
    for name, record in expected.items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Pinned encoder file missing or unsafe: {name}')
        digest, size = sha256_file(path), path.stat().st_size
        if digest != record['sha256'] or size != record['bytes']:
            raise ValueError(f'Pinned encoder file checksum mismatch: {name}')
        actual[name] = {'sha256': digest, 'bytes': size}
    return actual


def encoder_contract(key):
    if sha256_file(CONTRACT) != CONTRACT_SHA256:
        raise ValueError('Pinned model file contract changed')
    spec = ENCODER_SPECS[key]
    records = json.loads(CONTRACT.read_text(encoding='utf8'))['models']
    record = next(item for item in records if item['model_id'] == spec['repo'])
    if record['revision'] != spec['revision']:
        raise ValueError('Pinned encoder revisions disagree')
    expected = {item['filename']: dict(item) for item in record['files']}
    size, digest = WEIGHTS[key]
    expected['pytorch_model.bin'] = {'bytes': size, 'sha256': digest}
    for filename, item in expected.items():
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{filename}"
        if item.get('url', url) != url:
            raise ValueError('Unpinned encoder download URL')
        item.update(name=filename, url=url)
    return expected


def prepare_encoder(key, cache, *, offline=False, progress=None):
    expected = encoder_contract(key)
    directory = Path(cache) / key
    if directory.exists():
        unexpected = {p.name for p in directory.iterdir()} - set(expected)
        if unexpected:
            raise ValueError(f'Unpinned encoder files could change loading: {sorted(unexpected)}')
    ensure_cache_files(expected.values(), directory, offline=offline, progress=progress)
    return directory, verify_file_set(directory, expected)


def load_encoder(key, cache, device, offline=False):
    """Load only verified files, with remote code and online lookup disabled."""
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as transformer_logging
    # The pipeline owns Chinese progress output; library bars otherwise emit
    # carriage returns even when redirected to a plain execution log.
    transformer_logging.disable_progress_bar()
    path, verified = prepare_encoder(key, cache, offline=offline)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model, info = AutoModel.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False,
        use_safetensors=False, weights_only=True, add_pooling_layer=False,
        output_loading_info=True, dtype=torch.float32)
    if info.get('missing_keys') or info.get('mismatched_keys'):
        raise ValueError('Encoder parameters missing or mismatched')
    if model.config.hidden_size != 768:
        raise ValueError('D1 requires 768-dimensional hidden states')
    model.eval().requires_grad_(False).to(device)
    return tokenizer, model, {'verified_pinned_files': verified,
                              'file_contract_sha256': CONTRACT_SHA256}


def masked_token_mean(hidden, attention_mask, special_tokens_mask):
    import torch
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2] or special_tokens_mask.shape != attention_mask.shape:
        raise ValueError('Encoder token-mask shape mismatch')
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError('Encoder states must be finite')
    mask = attention_mask.bool() & ~special_tokens_mask.bool()
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError('Encoder input has no content token')
    return (hidden.float() * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)


def encode_texts(texts, available, key, output_path, tokenizer, model, *, device, batch_size, progress=None):
    """Write bounded batches to a temporary mmap, then publish a complete array."""
    import torch
    if batch_size < 1:
        raise ValueError('Encoder batch size must be positive')
    available = np.asarray(available, dtype=bool)
    if available.shape != (len(texts),):
        raise ValueError('Encoder text and availability rows differ')
    indices = np.flatnonzero(available)
    if any(not isinstance(texts[i], str) or not texts[i].strip() for i in indices):
        raise ValueError('Refusing missing or empty encoder inputs')
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f'Refusing to replace completed features: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + '.partial.npy')
    output = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(texts), 768))
    output[:] = 0
    truncated, last = 0, time.monotonic()
    model.eval().requires_grad_(False)
    try:
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset:offset + batch_size]
            batch = [texts[i] for i in selected]
            maximum = ENCODER_SPECS[key]['max_length']
            raw_ids = tokenizer(batch, add_special_tokens=True, truncation=False)['input_ids']
            truncated += sum(len(row) > maximum for row in raw_ids)
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=maximum,
                                return_tensors='pt', return_special_tokens_mask=True)
            special = encoded.pop('special_tokens_mask').to(device)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                pooled = masked_token_mean(hidden, encoded['attention_mask'], special)
            if pooled.shape != (len(selected), 768) or not bool(torch.isfinite(pooled).all()):
                raise ValueError('D1 encoder emitted invalid shape or nonfinite values')
            output[selected] = pooled.cpu().numpy()
            if progress and time.monotonic() - last >= 5:
                output.flush()
                progress('冻结特征编码 ' + key, min(offset + batch_size, len(indices)), len(indices), 0)
                last = time.monotonic()
        output.flush()
    finally:
        del output  # Windows must close the mmap before renaming.
    partial.replace(path)
    if progress:
        progress('冻结特征编码 ' + key, len(indices), len(indices), 0)
    return {'rows': len(texts), 'available_rows': len(indices), 'missing_rows': len(texts)-len(indices),
            'truncated_rows': truncated, 'max_length': ENCODER_SPECS[key]['max_length']}
