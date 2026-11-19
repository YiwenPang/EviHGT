"""One fixed disease-conditional support suffix; contra retains global U.

The sampling distribution changes. No negative labels, inverse-frequency P
weights, score mining, teacher, or new optimizer behavior are introduced.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import math
import random

import numpy as np
import torch

from evi_hgt.phase85_v2.common import bound, digest, verify
from evi_hgt.training.formal_objective import (
    DualChannelPairwiseRankingObjective, DualChannelRankingRisk, PairwiseRankingRisk,
)
from evi_hgt.training.supervision import SampleRole


VARIANT = 'SIDE_DCOND_PU_SUFFIX_V1'
_REQUIRED = ('fold_manifest_digest', 'repeat_key', 'fold', 'training_seed',
             'training_u_manifest_digest', 'exclusion_digest', 'warmup_checkpoint_sha256')
_HASH_FIELDS = ('fold_manifest_digest', 'training_u_manifest_digest', 'exclusion_digest',
                'warmup_checkpoint_sha256')


def group_digest(groups):
    return digest([[asdict(q) for q in group] for group in groups])


def group_parts(group):
    support = [(q.drug_id, q.disease_id) for q in group if q.support.role is SampleRole.POSITIVE]
    contra = [(q.drug_id, q.disease_id) for q in group if q.unsuitability.role is SampleRole.POSITIVE]
    unlabeled = [(q.drug_id, q.disease_id) for q in group if q.observed_relation is None
                 and q.support.pu_risk_eligible and q.unsuitability.pu_risk_eligible]
    if not support or not contra or len(unlabeled) != 1:
        raise ValueError('R2 requires the original positive streams and exactly one shared global U')
    return support, contra, unlabeled[0]


def _check_bindings(bindings):
    if any(k not in bindings for k in _REQUIRED):
        raise ValueError('Missing conditional fold/U-source/exclusion/warm-up binding')
    if any(not isinstance(bindings[k], str) or len(bindings[k]) != 64 for k in _HASH_FIELDS):
        raise ValueError('Invalid conditional digest binding')


def original_group_census(groups):
    support_n = contra_n = support_same = contra_same = 0
    support_counts, contra_counts = Counter(), Counter()
    for group in groups:
        support, contra, original = group_parts(group)
        support_n += len(support); contra_n += len(contra)
        support_same += sum(p[1] == original[1] for p in support)
        contra_same += sum(p[1] == original[1] for p in contra)
        support_counts[len(support)] += 1; contra_counts[len(contra)] += 1
    return dict(groups=len(groups), support_positive_occurrences=support_n,
        contra_positive_occurrences=contra_n, support_same_disease_comparisons=support_same,
        support_cross_disease_comparisons=support_n-support_same,
        contra_same_disease_comparisons=contra_same, contra_cross_disease_comparisons=contra_n-contra_same,
        support_same_disease_fraction=support_same/support_n if support_n else None,
        contra_same_disease_fraction=contra_same/contra_n if contra_n else None,
        support_group_cardinality={str(k):v for k,v in sorted(support_counts.items())},
        contra_group_cardinality={str(k):v for k,v in sorted(contra_counts.items())}, shared_global_u_per_group=1,
        support_primarily_cross_disease=bool(support_n and support_same/support_n < .5))


def build_pairing_plan(data, suffix_groups, *, warmup_steps, bindings):
    """Uniform legal drugs conditional on each disease, with replacement.

    Every legal drug occupies exactly one rank in the original eligible-drug
    sequence. random.Random(seed).randrange(Nlegal) is uniform on those ranks;
    no rejection timeout, score, or ID-based difficulty ordering is involved.
    Only exclusion IDs and actual suffix queries are enumerated, never a
    drug-by-disease score matrix. RNG is private and does not touch torch RNG.
    """
    _check_bindings(bindings)
    if warmup_steps <= 0 or not suffix_groups:
        raise ValueError('Complete warm-up and nonempty fixed suffix required')
    universe, gate = data.pairs.universe, data.pairs.gate
    drugs = tuple(universe.eligible_drugs)
    if len(drugs) != len(set(drugs)) or not drugs:
        raise ValueError('Eligible drug universe must be nonempty and unique')
    drug_set = set(drugs)
    diseases = set(universe.eligible_diseases)
    heldout_drugs = set(universe.heldout_drugs)
    if heldout_drugs.intersection(drug_set) or set(universe.heldout_diseases).intersection(diseases):
        raise ValueError('Training U universe contains heldout endpoints')
    forbidden = set(universe.observed_pairs) | set(gate.formal) | set(gate.blocked) | set(gate.positive)
    wanted = {p[1] for g in suffix_groups for p in group_parts(g)[0]}
    excluded = defaultdict(set)
    for drug, disease in forbidden:
        if disease in wanted and drug in drug_set:
            excluded[disease].add(drug)
    legal_counts = {d: len(drugs)-len(excluded[d]) if d in diseases else 0 for d in wanted}
    rows, all_u, distinct_by_disease = [], set(), defaultdict(set)
    occurrences = covered = extra_forwards = 0
    per_disease = defaultdict(lambda: dict(positive_occurrences=0,local_draws=0,fallback_positive_occurrences=0))
    for index, group in enumerate(suffix_groups):
        support, contra, original = group_parts(group)
        for pair in support:
            if pair not in data.support: raise ValueError('Conditional P absent from training support')
            data.pairs.assert_optimizer_pair(pair, 'P')
        for pair in contra:
            if pair not in data.unsuitability: raise ValueError('Contra P absent from original training channel')
            data.pairs.assert_optimizer_pair(pair, 'P')
        data.pairs.assert_optimizer_pair(original, 'U')
        occurrence_counts = Counter(p[1] for p in support)
        local, seeds, statuses, extras = {}, {}, {}, []
        original_pairs = {(q.drug_id,q.disease_id) for q in group}
        for disease, count in occurrence_counts.items():
            occurrences += count
            per_disease[disease]['positive_occurrences'] += count
            seed = int(digest(['PART3A_R2_CONDITIONAL_UNIFORM_V1',bindings,index,disease])[:16],16)
            seeds[disease] = seed
            if legal_counts[disease] == 0:
                local[disease] = []
                statuses[disease] = 'NO_LEGAL_U_ORIGINAL_PAIRING_FALLBACK'
                per_disease[disease]['fallback_positive_occurrences'] += count
                continue
            rank = random.Random(seed).randrange(legal_counts[disease])
            # One uniform rank selection, preserving with-replacement semantics
            # across groups. Multiple positives of this disease share the draw.
            for drug in drugs:
                if drug in excluded[disease]: continue
                if rank == 0:
                    pair = (drug,disease)
                    break
                rank -= 1
            else: raise RuntimeError('Legal U rank could not be resolved')
            data.pairs.assert_optimizer_pair(pair,'U')
            local[disease] = [list(pair)]
            statuses[disease] = 'DISEASE_CONDITIONAL_UNIFORM'
            covered += count; per_disease[disease]['local_draws'] += 1
            all_u.add(pair); distinct_by_disease[disease].add(pair)
            if pair not in original_pairs and pair not in extras: extras.append(pair)
        extra_forwards += len(extras)
        rows.append(dict(group_index=index,logical_step=warmup_steps+index+1,
            original_global_u=list(original),support_positive_pairs=[list(p) for p in support],
            contra_positive_pairs=[list(p) for p in contra],original_support_u_count=1,
            disease_unlabeled_pairs=local,draw_seed_by_disease=seeds,disease_status=statuses,
            extra_pairs=[list(p) for p in extras]))
    fraction = covered/occurrences
    return bound(dict(schema='PART3A_R2_DISEASE_CONDITIONAL_PAIRING_V1',variant=VARIANT,
        status='PAIRING_READY' if fraction>=.95 else 'CONDITIONAL_COVERAGE_BLOCKED',bindings=dict(bindings),
        warmup_steps=warmup_steps,suffix_groups=len(suffix_groups),source_group_digest=group_digest(suffix_groups),
        original_support_u_count=1,records=rows,support_positive_occurrences=occurrences,
        conditional_positive_occurrences=covered,fallback_positive_occurrences=occurrences-covered,
        support_occurrence_coverage=fraction,coverage_gate_threshold=.95,coverage_gate_pass=fraction>=.95,
        distinct_conditional_u=len(all_u),extra_forward_count=extra_forwards,
        original_suffix_forward_count=sum(len(g) for g in suffix_groups),
        new_suffix_forward_count=sum(len(g) for g in suffix_groups)+extra_forwards,
        per_disease={d:dict(per_disease[d],legal_drug_count=legal_counts[d],
            distinct_u=len(distinct_by_disease[d])) for d in sorted(wanted)},
        original_census=original_group_census(suffix_groups),
        uniformity_proof='One uniform integer rank in the exact legal drug complement per disease; with replacement across groups; one shared draw per group/disease.',
        rng_namespace='PART3A_R2_CONDITIONAL_UNIFORM_V1',training_rng_untouched=True,
        contra_global_u_unchanged=True,support_p_order_and_frequency_unchanged=True,
        support_risk='mean_P mean_same_disease_U sigmoid(-(support_P-support_U)/tau); fallback P uses original U',
        theory='Conditional PU interpretation needs disease-specific mixture/selection assumptions. Symmetry alone is not a project-wide consistency proof. Positive frequency unchanged; not exact disease-macro risk.'))


def validate_pairing_plan(plan, data, suffix_groups, *, bindings):
    verify(plan); _check_bindings(bindings)
    if plan['bindings'] != bindings or plan['source_group_digest'] != group_digest(suffix_groups):
        raise ValueError('Conditional pairing fold/source group binding mismatch')
    expected = build_pairing_plan(data,suffix_groups,warmup_steps=plan['warmup_steps'],bindings=bindings)
    if plan != expected:
        raise ValueError('Conditional pairing records/exclusions/uniform stream do not reproduce')


def conditional_risks(objective, *, support_positive_logits, support_diseases,
                      support_u_by_disease, global_support_u_logits,
                      contra_positive_logits, contra_global_u_logits):
    """Keep each P's weight 1/|P_g| and the original normalized channel weights.

    sum_d |P_g,d|/|P_g| * mean_{P_g,d x U_g,d}(ell) equals
    mean_{p in P_g} mean_{u in U_g,d(p)}(ell). Disease counts are *not* equalized.
    Original m=1 was verified in planning; missing d is the declared fallback.
    """
    if not isinstance(objective,DualChannelPairwiseRankingObjective):
        raise ValueError('R2 requires the original symmetric PU pairwise loss')
    if objective.explicit_regularization_weight != 0:
        raise ValueError('R2 frozen recipe has no new explicit regularizer')
    positive = support_positive_logits.float().reshape(-1)
    global_u = global_support_u_logits.float().reshape(-1)
    if len(support_diseases) != positive.numel() or positive.numel()==0 or global_u.numel()!=1:
        raise ValueError('Conditional logits do not match the original P and m=1 group')
    if set(support_u_by_disease)-set(support_diseases):
        raise ValueError('Local U supplied for a disease absent from group positives')
    if any(v.numel()!=1 for v in support_u_by_disease.values()):
        raise ValueError('Local m must equal original group support U count one')
    if not support_u_by_disease:
        return objective(support_positive_logits=positive,support_unlabeled_logits=global_u,
            unsuitability_positive_logits=contra_positive_logits,unsuitability_unlabeled_logits=contra_global_u_logits)
    parts, margins = [], []
    for disease in dict.fromkeys(support_diseases):
        indices = [i for i,d in enumerate(support_diseases) if d==disease]
        risk = objective.channel_loss(positive[indices],support_u_by_disease.get(disease,global_u))
        weight = len(indices)/positive.numel()
        parts.append(risk.total*weight); margins.append(risk.mean_margin*weight)
    support_loss = torch.stack(parts).sum()
    support = PairwiseRankingRisk(total=support_loss,positive_count=positive.numel(),
        unlabeled_count=len(set(support_diseases)),pair_count=positive.numel(),mean_margin=torch.stack(margins).sum(),
        finite=bool(torch.isfinite(support_loss)),temperature=objective.channel_loss.temperature,
        pairwise_chunk_size=objective.channel_loss.pairwise_chunk_size)
    contra = objective.channel_loss(contra_positive_logits,contra_global_u_logits)
    total = (objective.lambda_support*support.total + objective.lambda_unsuitability*contra.total)/(
        objective.lambda_support+objective.lambda_unsuitability)
    if not bool(torch.isfinite(total)): raise FloatingPointError('Nonfinite conditional dual PU risk')
    return DualChannelRankingRisk(support=support,unsuitability=contra,
        lambda_support=objective.lambda_support,lambda_unsuitability=objective.lambda_unsuitability,
        explicit_regularization_weight=0.,regularization=torch.zeros_like(total),total=total,finite=True)


def margin_diagnostics(positive, unlabeled, *, temperature=1.):
    if not math.isfinite(temperature) or temperature<=0: raise ValueError('Positive finite tau required')
    p,u=np.asarray(positive,dtype=np.float64).reshape(-1),np.asarray(unlabeled,dtype=np.float64).reshape(-1)
    if not p.size or not u.size or not np.isfinite(p).all() or not np.isfinite(u).all():
        raise ValueError('Finite nonempty P/U diagnostics required')
    z=((p[:,None]-u[None,:])/temperature).reshape(-1)
    # Stable float64 analytic quantities; this is not a training precision change.
    ez=np.exp(-np.abs(z));small=ez/(1+ez)
    loss=np.where(z>=0,small,1-small);derivative=small*(1-small)
    bins=[-np.inf,-8.,-4.,0.,4.,8.,np.inf]
    return dict(pairs=int(z.size),temperature=temperature,
        normalized_margin_min=float(z.min()),normalized_margin_max=float(z.max()),
        normalized_margin_mean=float(z.mean()),bin_edges=['-inf',-8,-4,0,4,8,'inf'],
        bin_counts=np.histogram(z,bins=bins)[0].tolist(),bin_semantics='left closed/right open except last',
        correct_fraction=float((z>0).mean()),reversed_fraction=float((z<0).mean()),tie_fraction=float((z==0).mean()),
        loss_mean=float(loss.mean()),abs_derivative_wrt_z_mean=float(derivative.mean()),
        abs_derivative_wrt_margin_mean=float(derivative.mean()/temperature),
        abs_derivative_wrt_margin_max=float(derivative.max()/temperature),
        derivative_below_1e_6_fraction=float((derivative/temperature<1e-6).mean()),
        derivative_formula='abs(d ell / d(gP-gU)) = sigmoid(z)*sigmoid(-z)/tau',
        analytic_precision='float64; source model scores retain their observed precision')


def candidate_census(rows, *, default_role=None):
    counts=defaultdict(Counter);seen=set()
    for r in rows:
        pair=(r['drug_id'],r['disease_id'])
        role=r.get('role',default_role)
        if pair in seen or role not in ('P','U'): raise ValueError('Invalid fixed candidate panel')
        seen.add(pair);counts[pair[1]][role]+=1
    local=[dict(disease_id=d,P=c['P'],U=c['U'],candidates=c['P']+c['U'],evaluable=bool(c['P'] and c['U']))
           for d,c in sorted(counts.items())]
    valid=[r for r in local if r['evaluable']]
    def share(k): return sum(r['candidates']<=k for r in valid)/len(valid) if valid else None
    def quantiles(key): return dict(zip(['min','q25','median','q75','max'],np.quantile([r[key] for r in valid],[0,.25,.5,.75,1]).tolist())) if valid else None
    return dict(pairs=len(seen),all_diseases=len(local),evaluable_diseases=len(valid),
        skipped_diseases=len(local)-len(valid),evaluable_fraction_candidates_le5=share(5),
        evaluable_fraction_candidates_le10=share(10),P_quantiles=quantiles('P'),U_quantiles=quantiles('U'),
        candidate_quantiles=quantiles('candidates'),per_disease=local,
        semantics='Fixed P-vs-U sampled-candidate panel; U is not confirmed negative; no panel reselection')


def select_diagnostic_groups(groups, *, bindings, max_groups=128):
    if not isinstance(max_groups,int) or not 1<=max_groups<=128:
        raise ValueError('Diagnostics require 1..128 legal training groups')
    return sorted(range(len(groups)),key=lambda i:digest(['PART3A_R2_MARGIN_PANEL_V1',bindings,i]))[:max_groups]


def select_gradient_groups(rows, *, max_groups=3):
    """Bounded descriptive probes, not training/selection: retain error modes."""
    if not isinstance(max_groups,int) or not 0<=max_groups<=12:
        raise ValueError('At most 12 actual gradient group probes')
    chosen=[]
    # Input rows are already in the preregistered hash order. Choose the first
    # available example of each declared analytic diagnostic condition.
    for predicate in (lambda r:r['support']['reversed_fraction']>0,
                      lambda r:r['support']['abs_derivative_wrt_margin_max']>=1e-3,
                      lambda r:True):
        for row in rows:
            if len(chosen)>=max_groups: return chosen
            if row['group_index'] not in chosen and predicate(row):
                chosen.append(row['group_index'])
                break
    for row in rows:
        if len(chosen)>=max_groups:break
        if row['group_index'] not in chosen:chosen.append(row['group_index'])
    return chosen
