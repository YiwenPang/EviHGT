"""Original pre-fold pools and registered development folds, without old receipts."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

import polars as pl

from evi_hgt.data.common import sha256
from evi_hgt.preprocessing.initial_splits_source import build_split
from evi_hgt.preprocessing.splits import historical_exclusions, frame, pairs
from evi_hgt.preprocessing.splits_source import (assign_folds, make_cell_manifest, make_repeat_keys,
                                               digest, order_key, select_holdout_positives)

PAIR = ['drug_id', 'disease_id']
SET_FIELDS = ('frozen', 'historical', 'observed', 'legal_source_rows')


def load_recipe(cell, *, protocol=None):
    recipes = json.loads(Path(__file__).with_name('recipes.json').read_text('utf-8'))
    if protocol is not None:
        if protocol != 'losses':
            raise ValueError('Experiment sampling protocol is not registered: ' + str(protocol))
        if cell != 'R0F0':
            raise ValueError('Original losses protocol is registered only for R0F0')
        return recipes['protocols']['losses'][cell]
    if cell not in recipes['cells']:
        raise ValueError('Experiment cell is not registered: ' + str(cell))
    return recipes['cells'][cell]


def reconstruct_pools(canonical_root, *, progress=None):
    """Use exactly the raw R0F1 pre-fold algorithm for every experiment."""
    source = pl.read_parquet(Path(canonical_root) / 'supervision/dda_supervision.parquet')
    if set(source['raw_relation']) - {'INDICATION', 'CONTRAINDICATION', 'OFF_LABEL_USE'}:
        raise ValueError('Unverified clinical relation')
    if source.select(PAIR).is_duplicated().any():
        raise ValueError('Canonical DDA source must contain one record per pair')
    initial = {kind: build_split(source.select(*PAIR, 'canonical_pair_key'), kind=kind, seed=42,
                                validation_ratio=.1, test_ratio=.1)
               for kind in ('standard', 'cold_drug', 'cold_disease')}
    historical = historical_exclusions(source, initial, progress)
    observed = pairs(source)
    legal_source = observed - historical
    support = pairs(source.filter(pl.col('support_mask') & pl.col('policy_b_member'))) - historical
    drugs = sorted(set(source['drug_id']))
    diseases = sorted({d for _, d in support})
    panel = [dict(drug_id=a, disease_id=b, role='P') for a, b in sorted(support)]
    excluded_by_disease = defaultdict(set)
    for a, b in observed | historical:
        excluded_by_disease[b].add(a)
    for i, disease in enumerate(diseases):
        def panel_key(drug):
            return hashlib.sha256(json.dumps(['PHASE85_DEVELOPMENT_U32_PANEL_V1', disease, drug],
                                              separators=(',', ':')).encode()).digest(), drug
        chosen = sorted((d for d in drugs if d not in excluded_by_disease[disease]), key=panel_key)[:32]
        panel.extend(dict(drug_id=d, disease_id=disease, role='U') for d in chosen)
        if progress and i % 100 == 0:
            progress('重建共同开发候选', i, len(diseases), 0)
    cold_drugs = set(initial['cold_drug'].filter(pl.col('drug_partition') == 'test')['drug_id'])
    cold_diseases = set(initial['cold_disease'].filter(pl.col('disease_partition') == 'test')['disease_id'])
    held_p = select_holdout_positives([dict(drug_id=a, disease_id=b) for a, b in sorted(support)])
    held = {(r['drug_id'], r['disease_id']) for r in held_p}
    eligible_drugs = [d for d in drugs if d not in cold_drugs]
    eligible_diseases = sorted(set(source['disease_id']) - cold_diseases)
    for disease in sorted({d for _, d in held}):
        chosen = sorted((d for d in eligible_drugs if d not in excluded_by_disease[disease]),
                        key=lambda d: order_key('U', d, disease))[:32]
        held.update((d, disease) for d in chosen)
    def legal(pair):
        return pair not in held and pair[0] not in cold_drugs and pair[1] not in cold_diseases
    panel = sorted((r for r in panel if legal((r['drug_id'], r['disease_id']))),
                   key=lambda r: (r['drug_id'], r['disease_id']))
    auxiliary = sorted(p for p in pairs(source.filter(pl.col('raw_relation') == 'CONTRAINDICATION'))
                       & legal_source if legal(p))
    return dict(source=source, panel=panel, auxiliary=auxiliary, historical=historical,
                frozen=historical | held, observed=observed,
                legal_source_rows={p for p in legal_source if legal(p)},
                eligible_drugs=eligible_drugs, eligible_diseases=eligible_diseases)


def save_pools(pools, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    def save(key, table):
        path = destination / (key + '.parquet')
        table.write_parquet(path)
        files[key] = str(path)
    save('source', pools['source'])
    save('panel', pl.DataFrame(pools['panel'], schema={**{k: pl.String for k in PAIR}, 'role': pl.String}))
    save('auxiliary', pl.DataFrame(pools['auxiliary'], schema={k: pl.String for k in PAIR}, orient='row'))
    for key in SET_FIELDS:
        save(key, frame(pools[key]))
    axes = destination / 'eligibility.json'
    axes.write_text(json.dumps({k: pools[k] for k in ('eligible_drugs', 'eligible_diseases')}), encoding='utf-8')
    files['eligibility'] = str(axes)
    result = dict(schema='EVIHGT_EXPERIMENT_POOLS_V1', root=str(destination), files=files,
                  hashes={key: sha256(path) for key, path in files.items()})
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def load_pools(descriptor):
    root = Path(descriptor['root']).resolve(strict=True)
    files = {k: Path(v).resolve(strict=True) for k, v in descriptor['files'].items()}
    for key, path in files.items():
        if not path.is_relative_to(root) or sha256(path) != descriptor['hashes'][key]:
            raise ValueError('Pooled experiment input hash mismatch: ' + key)
    values = {k: pairs(pl.read_parquet(files[k])) for k in SET_FIELDS}
    values.update(source=pl.read_parquet(files['source']), panel=pl.read_parquet(files['panel']).to_dicts(),
                  auxiliary=list(pl.read_parquet(files['auxiliary']).select(PAIR).iter_rows()),
                  **json.loads(files['eligibility'].read_text('utf-8')))
    return values


def partition_pools(pools, cell):
    expected = load_recipe(cell)
    repeat_index, fold = int(cell[1]), int(cell[3])
    repeat_key = make_repeat_keys(5, namespace='EVIHGT_PHASE85_PROTOCOL_V2_FINAL_5X5')[repeat_index]
    if repeat_key != expected['repeat_key']:
        raise ValueError('Registered repeat identity differs')
    folds = assign_folds(pools['panel'], 5, repeat_key)
    parts = make_cell_manifest(pools['panel'], folds, fold, validation_fraction=.2, probe_fraction=.05,
        protocol_digest='RAW_RECONSTRUCTION_RECIPE_ONLY', auxiliary_training_positives=[
            dict(drug_id=a, disease_id=b, raw_relation='CONTRAINDICATION') for a, b in pools['auxiliary']])
    training = {(r['drug_id'], r['disease_id']) for r in parts['training']}
    contra = {(r['drug_id'], r['disease_id']) for r in parts['auxiliary_training']}
    blocked = pools['frozen'] | ({(r['drug_id'], r['disease_id']) for r in pools['panel']} - training)
    blocked |= set(pools['auxiliary']) - contra
    seed = int(digest([repeat_key, fold, 'shared-training-seed']), 16) % 2**31
    return dict(cell=cell, repeat_key=repeat_key, training_seed=seed, parts=parts, blocked=blocked)


def build_cell_splits(pools, destination, *, cell, audit_dir, progress=None):
    expected = load_recipe(cell)
    destination, audit_dir = Path(destination).resolve(), Path(audit_dir).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new experiment split directory')
    result = partition_pools(pools, cell)
    parts, blocked = result['parts'], result['blocked']
    for name, value in expected['membership_digests'].items():
        if digest(parts[name]) != value:
            raise ValueError(f'Raw data do not reproduce registered {cell} membership: {name}')
    if digest(sorted(blocked)) != expected['mask_digest'] or result['training_seed'] != expected['training_seed']:
        raise ValueError('Registered cell exclusion mask or seed differs: ' + cell)
    destination.mkdir(parents=True)
    outputs = {}
    def parquet(name, value):
        path = destination / (name + '.parquet')
        value.write_parquet(path)
        outputs[name] = str(path)
    for name, values in [('blocked_pairs', blocked), ('observed_pairs', pools['observed']),
                         ('legal_source_rows', pools['legal_source_rows'])]:
        parquet(name, frame(values))
    parquet('training_pairs', pl.DataFrame([dict(drug_id=r['drug_id'], disease_id=r['disease_id'],
        channel=channel, raw_relation=relation)
        for name, channel, relation in [('training', 'support', 'INDICATION'),
                                        ('auxiliary_training', 'unsuitability', 'CONTRAINDICATION')]
        for r in parts[name] if r.get('role', 'P') == 'P']))
    for name in ('validation', 'auxiliary_validation'):
        parquet(name, pl.DataFrame([dict(r, role=r.get('role', 'P'), raw_relation=(
            'INDICATION' if r['role'] == 'P' else None) if name == 'validation' else 'CONTRAINDICATION')
            for r in parts[name]]))
    eligibility = dict(eligible_drugs=pools['eligible_drugs'], eligible_diseases=pools['eligible_diseases'])
    path = destination / 'eligibility.json'
    path.write_text(json.dumps(eligibility), encoding='utf-8')
    outputs['eligibility'] = str(path)
    result = dict(schema='EVIHGT_REBUILT_SPLITS_V1', cell=cell, training_seed=result['training_seed'],
        repeat_key=result['repeat_key'], mask_digest=expected['mask_digest'],
        membership_digests=expected['membership_digests'], files=outputs)
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / (cell + '_split_reconstruction.json')).write_text(json.dumps(dict(status='PASS',
        cell=cell, membership_digests=expected['membership_digests'], mask_digest=expected['mask_digest'],
        historical_receipts_recreated=False, optimizer_steps=0), indent=2), encoding='utf-8')
    if progress:
        progress('划分成员与原配方一致：' + cell)
    return result
