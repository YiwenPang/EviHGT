"""Reconstruct every registered three-cycle plan with its original seed bindings."""
from dataclasses import asdict
from pathlib import Path
import json
import random

import polars as pl

from evi_hgt.preprocessing.sampling import epoch_groups, conditional_records, query
from evi_hgt.preprocessing.splits_source import digest
from .memberships import load_recipe


def build_cell_sampling(splits_root, destination, *, cell, audit_dir, progress=None, protocol=None):
    expected = load_recipe(cell, protocol=protocol)
    splits_root, destination, audit_dir = map(Path, (splits_root, destination, audit_dir))
    if destination.exists():
        raise FileExistsError('Choose a new experiment sampling directory')
    source = json.loads((splits_root / 'manifest.json').read_text('utf-8'))
    if source['cell'] != cell or source['mask_digest'] != expected['mask_digest']:
        raise ValueError('Split recipe/mask mismatch')
    positive = pl.read_parquet(splits_root / 'training_pairs.parquet')
    support = set(positive.filter(pl.col('channel') == 'support').select('drug_id', 'disease_id').iter_rows())
    contra = set(positive.filter(pl.col('channel') == 'unsuitability').select('drug_id', 'disease_id').iter_rows())
    blocked = set(pl.read_parquet(splits_root / 'blocked_pairs.parquet').iter_rows())
    observed = set(pl.read_parquet(splits_root / 'observed_pairs.parquet').iter_rows())
    eligible = json.loads((splits_root / 'eligibility.json').read_text('utf-8'))
    drugs, diseases = eligible['eligible_drugs'], eligible['eligible_diseases']
    G = min(len(support), len(contra))
    if (G != expected['G'] or digest(sorted(blocked)) != expected['mask_digest']
            or source['training_seed'] != expected['training_seed']):
        raise ValueError('Registered sampling cardinality/mask/seed changed')
    state = random.getstate()
    groups, rows, cycles = [], [], []
    for cycle in (1, 2, 3):
        part = epoch_groups(support, contra, drugs, diseases, observed, blocked,
            training_seed=source['training_seed'], repeat_key=source['repeat_key'], cycle=cycle)
        serialized = [[asdict(query(role, pair)) for role, pair in group] for group in part]
        cycle_digest = digest(serialized)
        if cycle_digest != expected['cycles'][cycle-1]['group_digest']:
            raise ValueError(f'Registered {cell} cycle membership/order changed: {cycle}')
        cycles.append(cycle_digest)
        for group, members in zip(part, serialized, strict=True):
            step = len(groups) + 1
            for position, (role, pair) in enumerate(group):
                rows.append(dict(step=step, cycle=cycle, position=position, role=role,
                                 drug_id=pair[0], disease_id=pair[1], member_sha256=digest(members)))
            groups.append(group)
        if progress:
            progress('重建 ' + cell + ' 三个完整采样周期', cycle, 3, 0)
    records = conditional_records(groups[G:], drugs, diseases, observed | blocked | support | contra,
                                  G=G, bindings=expected['conditional_bindings'])
    if digest(records) != expected['record_digest'] or random.getstate() != state:
        raise ValueError('Registered conditional plan/RNG preservation differs: ' + cell)
    destination.mkdir(parents=True)
    gp, cp = destination / 'global_groups.parquet', destination / 'conditional_records.json'
    pl.DataFrame(rows).write_parquet(gp)
    cp.write_text(json.dumps(records, indent=2), encoding='utf-8')
    result = dict(G=G, groups=str(gp.resolve()), conditional_records=str(cp.resolve()), sampling_provenance=dict(
        reconstructed_from_raw=True, cell=cell, cycle_digests=cycles, conditional_record_digest=digest(records),
        seed_recipe=expected['conditional_bindings'], model_checkpoint_loaded=False, optimizer_steps=0))
    if protocol is not None:
        result['sampling_provenance'].update(protocol=expected['protocol'], source_plan=expected['source_plan'])
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_name = cell + ('_' + protocol if protocol else '') + '_sampling_reconstruction.json'
    (audit_dir / audit_name).write_text(json.dumps(dict(status='PASS', G=G,
        cell=cell, cycle_digests=cycles, conditional_record_digest=digest(records), model_checkpoint_loaded=False,
        historical_receipts_recreated=False, optimizer_steps=0,
        **({'protocol': expected['protocol'], 'source_plan': expected['source_plan']} if protocol else {})),
        indent=2), encoding='utf-8')
    return result
