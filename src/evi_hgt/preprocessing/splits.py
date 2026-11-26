"""Reconstruct the original R0F1 membership from canonical source semantics.

Historical candidate recipes are exclusions only. This does not execute an old
evaluation, recreate its completion receipts, or expose final labels to training.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import hmac
import json

import polars as pl

from .initial_splits_source import build_split
from .splits_source import (assign_folds, make_cell_manifest, make_repeat_keys,
                           digest, order_key, select_holdout_positives)

PAIR = ['drug_id', 'disease_id']
PROFILE_RUN = 'EVIHGT-FORMAL-ad110867a0e6f6db-a6fdc76814484f4bb228673646a03e82'


def frame(pairs):
    return pl.DataFrame(sorted(pairs), schema={k: pl.String for k in PAIR}, orient='row')


def pairs(table):
    return set(table.select(PAIR).iter_rows())


def recipe():
    return json.loads(Path(__file__).with_name('r0f1_recipe.json').read_text(encoding='utf-8'))


def hmac_key(version, *ids):
    message = json.dumps(list(ids), ensure_ascii=False, separators=(',', ':')).encode()
    return hmac.digest(version.encode(), message, 'sha256'), ids[-1]


def candidate_sources(source, split, kind):
    """The original candidate builder's identity-only projection."""
    drugs = sorted(set(source['drug_id']))
    if kind == 'cold_drug':
        held = set(split.filter(pl.col('drug_partition') == 'test')['drug_id'])
        drugs = [d for d in drugs if d in held]
    disease_column = 'disease_partition' if kind == 'cold_disease' else 'split_assignment'
    diseases = sorted(set(split.filter(pl.col(disease_column) == 'test')['disease_id']))
    # Canonical supervision also carries its original Standard assignment.
    # Project semantic fields first, as the original candidate builder does,
    # so Cold membership comes only from the requested split table.
    joined = source.select(*PAIR, 'raw_relation', 'support_mask', 'policy_b_member').join(
        split.select(*PAIR, 'split_assignment'), on=PAIR, validate='1:1')
    excluded, positive = defaultdict(set), defaultdict(set)
    for a, b in joined.filter(pl.col('split_assignment') == 'train').select(PAIR).iter_rows():
        excluded[b].add(a)
    for a, b in joined.filter((pl.col('split_assignment') == 'test') & pl.col('support_mask')
                             & pl.col('policy_b_member')).select(PAIR).iter_rows():
        positive[b].add(a)
    for disease in diseases:
        eligible = set(drugs) - excluded[disease]
        if not positive[disease] <= eligible:
            raise ValueError('Historical relevance outside original candidate scope')
        yield disease, eligible, positive[disease]


def full_diseases(counts, *, evaluable):
    version = ('STANDARD_FULL_RETRIEVAL_EVALUABLE_STRATIFIED_6_V2' if evaluable
               else 'STANDARD_FULL_RETRIEVAL_STRATIFIED_6_V1')
    ordered = sorted([(d, n) for d, n in counts if not evaluable or n >= 1], key=lambda x: (x[1], x[0]))
    if len(ordered) < 6:
        raise ValueError('Original six-disease recipe requires at least six eligible diseases')
    q, r = divmod(len(ordered), 3)
    start, selected = 0, set()
    for i in range(3):
        stop = start + q + int(i < r)
        selected.update(sorted((d for d, _ in ordered[start:stop]), key=lambda d: hmac_key(version, d))[:2])
        start = stop
    return selected


def historical_exclusions(source, splits, progress=None):
    """Union of all frozen sampled/full/profiling memberships and source test."""
    forbidden = pairs(splits['standard'].filter(pl.col('split_assignment') == 'test'))
    counts = [(d, len(p)) for d, _, p in candidate_sources(source, splits['standard'], 'standard')]
    full = full_diseases(counts, evaluable=False) | full_diseases(counts, evaluable=True)
    for kind, split in splits.items():
        sources = candidate_sources(source, split, kind)
        disease_column = 'disease_partition' if kind == 'cold_disease' else 'split_assignment'
        diseases = sorted(set(split.filter(pl.col(disease_column) == 'test')['disease_id']))
        def profile_select(values, scope, count):
            return sorted(set(values), key=lambda v: (hashlib.sha256(
                (PROFILE_RUN + '\x1f' + scope + '\x1f' + v).encode()).digest(), v))[:count]
        profile = profile_select(diseases, kind + ':query', 8)
        for i, (d, eligible, positives) in enumerate(sources):
            selected = set(positives)
            versions = [('STANDARD_SAMPLED_PU_COMPARISON_HMAC_SHA256_U1024_V1', 1024),
                        ('STANDARD_SAMPLED_PU_COMPARISON_HMAC_SHA256_U32_V2', 32)] if kind == 'standard' else [
                            ('COLD_DRUG_SAMPLED_PU_U64_V1' if kind == 'cold_drug' else 'COLD_DISEASE_SAMPLED_PU_U64_V1', 64)]
            for version, quota in versions:
                selected.update(sorted(eligible - positives, key=lambda drug: hmac_key(version, d, drug))[:quota])
            if kind == 'standard' and d in full:
                selected.update(eligible)
            if d in profile:
                quota = 256 if profile.index(d) == 0 else 64
                selected.update(profile_select(eligible, kind + ':candidate:' + d, quota))
            forbidden.update((drug, d) for drug in selected)
            if progress and i % 100 == 0:
                progress('历史候选排除重建 ' + kind, i, len(diseases), 0)
    return forbidden


def build_splits(canonical_root, destination, *, cell='R0F1', audit_dir, progress=None):
    if cell != 'R0F1':
        raise ValueError('Raw reconstruction currently implements only the original R0F1 recipe')
    canonical_root, destination, audit_dir = map(Path, (canonical_root, destination, audit_dir))
    if destination.exists():
        raise FileExistsError('Choose a new split output directory')
    source = pl.read_parquet(canonical_root / 'supervision/dda_supervision.parquet')
    if set(source['raw_relation']) - {'INDICATION', 'CONTRAINDICATION', 'OFF_LABEL_USE'}:
        raise ValueError('Unverified clinical relation')
    if source.select(PAIR).is_duplicated().any():
        raise ValueError('Canonical DDA source must contain one record per pair')
    initial = {kind: build_split(source.select(*PAIR, 'canonical_pair_key'), kind=kind, seed=42,
        validation_ratio=.1, test_ratio=.1) for kind in ('standard', 'cold_drug', 'cold_disease')}
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
            progress('开发候选重建', i, len(diseases), 0)
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
    panel = sorted((r for r in panel if legal((r['drug_id'], r['disease_id']))), key=lambda r: (r['drug_id'], r['disease_id']))
    auxiliary = sorted(p for p in pairs(source.filter(pl.col('raw_relation') == 'CONTRAINDICATION')) & legal_source if legal(p))
    repeat_key = make_repeat_keys(5, namespace='EVIHGT_PHASE85_PROTOCOL_V2_FINAL_5X5')[0]
    folds = assign_folds(panel, 5, repeat_key)
    parts = make_cell_manifest(panel, folds, 1, validation_fraction=.2, probe_fraction=.05,
        protocol_digest='RAW_RECONSTRUCTION_RECIPE_ONLY', auxiliary_training_positives=[
            dict(drug_id=a, disease_id=b, raw_relation='CONTRAINDICATION') for a, b in auxiliary])
    frozen = historical | held
    training = {(r['drug_id'], r['disease_id']) for r in parts['training']}
    contra = {(r['drug_id'], r['disease_id']) for r in parts['auxiliary_training']}
    blocked = frozen | ({(r['drug_id'], r['disease_id']) for r in panel} - training) | (set(auxiliary) - contra)
    expected = recipe()
    # Recipe hashes authenticate membership without distributing actual pairs.
    for name, expected_digest in expected['membership_digests'].items():
        if name in parts and digest(parts[name]) != expected_digest:
            raise ValueError('Raw data do not reproduce original R0F1 membership: ' + name)
    if digest(sorted(blocked)) != expected['mask_digest']:
        raise ValueError('Raw data do not reproduce original R0F1 exclusion mask')
    seed = int(digest([repeat_key, 1, 'shared-training-seed']), 16) % 2**31
    if seed != expected['training_seed']:
        raise ValueError('Original cell seed differs')
    destination.mkdir(parents=True)
    outputs = {}
    def parquet(name, value):
        path = destination / (name + '.parquet'); value.write_parquet(path); outputs[name] = str(path.resolve())
    for name, values in [('blocked_pairs', blocked), ('observed_pairs', observed),
        ('legal_source_rows', {p for p in legal_source if legal(p)})]:
        parquet(name, frame(values))
    parquet('training_pairs', pl.DataFrame([dict(drug_id=r['drug_id'], disease_id=r['disease_id'],
        channel=channel, raw_relation=relation) for name, channel, relation in [('training', 'support', 'INDICATION'),
        ('auxiliary_training', 'unsuitability', 'CONTRAINDICATION')] for r in parts[name] if r.get('role', 'P') == 'P']))
    for name in ('validation', 'auxiliary_validation'):
        parquet(name, pl.DataFrame([dict(r, role=r.get('role', 'P'), raw_relation=(
            'INDICATION' if r['role'] == 'P' else None) if name == 'validation' else 'CONTRAINDICATION') for r in parts[name]]))
    eligibility = dict(eligible_drugs=eligible_drugs, eligible_diseases=eligible_diseases)
    (destination / 'eligibility.json').write_text(json.dumps(eligibility), encoding='utf-8')
    outputs['eligibility'] = str((destination / 'eligibility.json').resolve())
    result = dict(schema='EVIHGT_REBUILT_SPLITS_V1', cell=cell, training_seed=seed, repeat_key=repeat_key,
        mask_digest=expected['mask_digest'], membership_digests=expected['membership_digests'], files=outputs)
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / 'split_reconstruction.json').write_text(json.dumps(dict(cell=cell, status='PASS',
        historical_receipts_recreated=False, optimizer_steps=0, final_evaluation_executed=False,
        membership_digests=result['membership_digests'], mask_digest=result['mask_digest']), indent=2), encoding='utf-8')
    return result
