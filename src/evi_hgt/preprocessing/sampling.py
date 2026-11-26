"""Rebuild three original positive cycles and score-blind conditional U draws."""
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import random

import polars as pl

from evi_hgt.training.supervision import derive_observed_query_supervision, make_unlabeled_query_supervision
from .splits import digest, recipe


def epoch_groups(support, contra, drugs, diseases, observed, blocked, *, training_seed, repeat_key, cycle):
    if type(cycle) is not int or cycle < 1:
        raise ValueError('Production cycle numbering starts at 1')
    G = min(len(support), len(contra))
    if not G or not drugs or not diseases:
        raise ValueError('Both positive channels and a nonempty candidate domain are required')
    key = int(digest([training_seed, repeat_key, cycle, 'PU'])[:8], 16)
    rng, unlabeled = random.Random(key), []
    forbidden = observed | blocked | support | contra
    for _ in range(max(1000, G * 100)):
        pair = rng.choice(drugs), rng.choice(diseases)
        if pair in forbidden:
            continue
        unlabeled.append(pair)
        if len(unlabeled) == G:
            break
    if len(unlabeled) != G:
        raise ValueError('Original PU rejection budget exhausted; no fallback allowed')
    records = [('support', p) for p in sorted(support)] + [('unsuitability', p) for p in sorted(contra)]
    records += [('U', p) for p in unlabeled]
    random.Random(key).shuffle(records)
    channels = {role: [x for x in records if x[0] == role] for role in ('support', 'unsuitability', 'U')}
    groups = []
    for index in range(G):
        group = []
        for role in ('support', 'unsuitability'):
            values = channels[role]
            group.extend(values[(index * len(values)) // G: ((index + 1) * len(values)) // G])
        group.append(channels['U'][index])
        # Canonical source pairs have one verified relation; preserve historical
        # first-occurrence deduplication of the original role grouping function.
        ordered = {}
        for role, pair in group:
            ordered.setdefault(pair, (role, pair))
        groups.append(tuple(ordered.values()))
    return groups


def query(role, pair):
    if role == 'U':
        return make_unlabeled_query_supervision(*pair)
    return derive_observed_query_supervision(dict(drug_id=pair[0], disease_id=pair[1],
        raw_relation='INDICATION' if role == 'support' else 'CONTRAINDICATION', policy_b_member=True,
        support_target=1.0 if role == 'support' else None, support_mask=role == 'support',
        unsuitability_target=1.0 if role == 'unsuitability' else None, unsuitability_mask=role == 'unsuitability'))


def conditional_records(groups, drugs, diseases, forbidden, *, G, bindings):
    excluded = defaultdict(set)
    wanted = {p[1] for group in groups for role, p in group if role == 'support'}
    drug_set, disease_set = set(drugs), set(diseases)
    for drug, disease in forbidden:
        if drug in drug_set and disease in wanted:
            excluded[disease].add(drug)
    legal = {d: tuple(x for x in drugs if x not in excluded[d]) if d in disease_set else () for d in wanted}
    records = []
    occurrences = covered = 0
    for index, group in enumerate(groups):
        support = [p for role, p in group if role == 'support']
        contra = [p for role, p in group if role == 'unsuitability']
        original = next(p for role, p in group if role == 'U')
        local, seeds, statuses, extras = {}, {}, {}, []
        original_pairs = {p for _, p in group}
        for disease, count in Counter(p[1] for p in support).items():
            occurrences += count
            seed = int(digest(['PART3A_R2_CONDITIONAL_UNIFORM_V1', bindings, index, disease])[:16], 16)
            seeds[disease] = seed
            if not legal[disease]:
                local[disease] = []; statuses[disease] = 'NO_LEGAL_U_ORIGINAL_PAIRING_FALLBACK'
                continue
            pair = (legal[disease][random.Random(seed).randrange(len(legal[disease]))], disease)
            if pair in forbidden:
                raise ValueError('Forbidden conditional U')
            local[disease] = [list(pair)]; statuses[disease] = 'DISEASE_CONDITIONAL_UNIFORM'; covered += count
            if pair not in original_pairs and pair not in extras:
                extras.append(pair)
        records.append(dict(group_index=index, logical_step=G+index+1, original_global_u=list(original),
            support_positive_pairs=[list(p) for p in support], contra_positive_pairs=[list(p) for p in contra],
            original_support_u_count=1, disease_unlabeled_pairs=local, draw_seed_by_disease=seeds,
            disease_status=statuses, extra_pairs=[list(p) for p in extras]))
    if not occurrences or covered / occurrences < .95:
        raise ValueError('Original 95% conditional positive coverage gate failed')
    return records


def build_sampling(splits_root, destination, *, cell='R0F1', audit_dir, progress=None):
    if cell != 'R0F1':
        raise ValueError('Only the original R0F1 seed recipe is supported')
    splits_root, destination, audit_dir = map(Path, (splits_root, destination, audit_dir))
    if destination.exists():
        raise FileExistsError('Choose a new sampling output directory')
    source = json.loads((splits_root / 'manifest.json').read_text(encoding='utf-8'))
    expected = recipe()
    if source['cell'] != cell or source['mask_digest'] != expected['mask_digest']:
        raise ValueError('Split recipe/mask mismatch')
    positive = pl.read_parquet(splits_root / 'training_pairs.parquet')
    support = set(positive.filter(pl.col('channel') == 'support').select('drug_id', 'disease_id').iter_rows())
    contra = set(positive.filter(pl.col('channel') == 'unsuitability').select('drug_id', 'disease_id').iter_rows())
    blocked = set(pl.read_parquet(splits_root / 'blocked_pairs.parquet').iter_rows())
    observed = set(pl.read_parquet(splits_root / 'observed_pairs.parquet').iter_rows())
    eligible = json.loads((splits_root / 'eligibility.json').read_text(encoding='utf-8'))
    drugs, diseases = eligible['eligible_drugs'], eligible['eligible_diseases']
    G = min(len(support), len(contra))
    if G != expected['G'] or digest(sorted(blocked)) != expected['mask_digest']:
        raise ValueError('Original sampling cardinality/mask changed')
    state = random.getstate()
    groups, rows, cycles = [], [], []
    for cycle in (1, 2, 3):
        part = epoch_groups(support, contra, drugs, diseases, observed, blocked,
            training_seed=source['training_seed'], repeat_key=source['repeat_key'], cycle=cycle)
        serialized = [[asdict(query(role, pair)) for role, pair in group] for group in part]
        cycle_digest = digest(serialized)
        if cycle_digest != expected['cycles'][cycle-1]['group_digest']:
            raise ValueError('Original cycle membership/order changed: ' + str(cycle))
        cycles.append(cycle_digest)
        for group, members in zip(part, serialized, strict=True):
            step = len(groups) + 1
            for position, (role, pair) in enumerate(group):
                rows.append(dict(step=step, cycle=cycle, position=position, role=role,
                    drug_id=pair[0], disease_id=pair[1], member_sha256=digest(members)))
            groups.append(group)
        if progress:
            progress('重建三个完整采样周期', cycle, 3, 0)
    records = conditional_records(groups[G:], drugs, diseases, observed | blocked | support | contra,
        G=G, bindings=expected['conditional_bindings'])
    if digest(records) != expected['record_digest'] or random.getstate() != state:
        raise ValueError('Original conditional plan/RNG preservation differs')
    destination.mkdir(parents=True)
    gp, cp = destination / 'global_groups.parquet', destination / 'conditional_records.json'
    pl.DataFrame(rows).write_parquet(gp)
    cp.write_text(json.dumps(records, indent=2), encoding='utf-8')
    result = dict(G=G, groups=str(gp.resolve()), conditional_records=str(cp.resolve()), sampling_provenance=dict(
        reconstructed_from_raw=True, cell=cell, cycle_digests=cycles, conditional_record_digest=digest(records),
        seed_recipe=expected['conditional_bindings'], model_checkpoint_loaded=False, optimizer_steps=0))
    (destination / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / 'sampling_reconstruction.json').write_text(json.dumps(dict(status='PASS', G=G,
        cycle_digests=cycles, conditional_record_digest=digest(records), model_checkpoint_loaded=False,
        historical_receipts_recreated=False, optimizer_steps=0), indent=2), encoding='utf-8')
    return result
