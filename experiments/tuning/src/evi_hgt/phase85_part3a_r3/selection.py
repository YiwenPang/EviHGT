"""Exact, prospective inner-development selection; no access to test scores.

Within each LR: maximum macro support AUC, then AP, then earlier checkpoint.
Across LRs: selected AUC, then AP, then original 5e-5, then smaller LR.
The legacy minimum global dual-PU objective remains a separate diagnostic.
"""
from __future__ import annotations

from copy import deepcopy
import math

from evi_hgt.phase85_v2.common import bound, verify


LR_CANDIDATES = (2.5e-5, 5e-5, 1e-4)
CONTROL_LR = 5e-5
METRIC_SELECTOR = 'MAX_INNER_MACRO_SUPPORT_AUC_THEN_AP_THEN_EARLIEST_STEP_EXACT'
LEGACY_SELECTOR = 'MIN_GLOBAL_DUAL_PU_OBJECTIVE_THEN_EARLIEST_STEP_EXACT'
LR_SELECTOR = 'MAX_SELECTED_AUC_THEN_AP_THEN_ORIGINAL_5E_5_THEN_SMALLER_LR_EXACT'


def _events(events, groups):
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 2:
        raise ValueError('Invalid complete coverage group budget')
    values = list(events)
    if len(values) != 2 or sorted(e.get('step', -1) for e in values) != [groups, 6*groups//4]:
        raise ValueError('Only the two prescribed G/B evaluation events are allowed')
    normalized = []
    for raw in values:
        item = deepcopy(raw)
        if 'objective' not in item:
            item['objective'] = item.get('validation_objective')
        if 'validation_objective' in item and item['objective'] != item['validation_objective']:
            raise ValueError('Global objective aliases disagree')
        for key in ('validation_AUC', 'validation_AP', 'objective'):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Selection metric must be finite: ' + key)
        if not all(0 <= item[k] <= 1 for k in ('validation_AUC', 'validation_AP')):
            raise ValueError('AUC/AP must be in [0,1]')
        normalized.append(item)
    return sorted(normalized, key=lambda e: e['step'])


def select_metric_event(events, groups):
    """Pure full-precision selection. No threshold, tolerance or composite score."""
    return min(_events(events, groups), key=lambda e: (-e['validation_AUC'], -e['validation_AP'], e['step']))


def select_legacy_event(events, groups):
    return min(_events(events, groups), key=lambda e: (e['objective'], e['step']))


def summarize_candidate(lr, events, groups, *, cached_control=False):
    """Keep both complete metric/source records; never rewrite historical BEST.

    ``events`` are flattened metrics with an attached ``source_event`` mapping
    supplied by the scoring/closure layer. That mapping is preserved verbatim.
    This pure function does not open checkpoints or evaluate the model.
    """
    if lr not in LR_CANDIDATES:
        raise ValueError('LR is outside the three fixed candidates')
    if cached_control and lr != CONTROL_LR:
        raise ValueError('Only the original 5e-5 can be cached control')
    values = _events(events, groups)
    for event in values:
        for key in ('lr', 'learning_rate'):
            if key in event and event[key] != lr:
                raise ValueError('Event learning rate / LR binding mismatch')
        source = event.get('source_event')
        if not isinstance(source, dict) or not source:
            raise ValueError('Complete source_event checkpoint/score bindings required')
        if 'step' in source and source['step'] != event['step']:
            raise ValueError('Source event step binding mismatch')
    metric = select_metric_event(values, groups)
    legacy = select_legacy_event(values, groups)
    return bound(dict(schema='PHASE85_R3_CANDIDATE_SELECTION_V1', status='CANDIDATE_COMPLETE',
        lr=lr, model='EVIHGT_SIDE', groups_per_cycle=groups, budget=6*groups//4,
        cached_control=bool(cached_control), logical_optimizer_updates=6*groups//4,
        new_optimizer_updates=0 if cached_control else 6*groups//4,
        events=values, endpoint=values[-1], metric_selected=metric,
        legacy_objective_selected=legacy,
        selectors_disagree=metric['step'] != legacy['step'],
        metric_selector=METRIC_SELECTOR, legacy_selector=LEGACY_SELECTOR,
        training_objective='ORIGINAL_GLOBAL_EQUAL_WEIGHT_DUAL_SYMMETRIC_PU',
        evaluation_scope='REUSED_INNER_DEVELOPMENT_NOT_INDEPENDENT_TEST',
        probe_used_for_selection=False, outer_scoring=False, final_scoring=False,
        default_lr_changed=False, no_automatic_extension=True))


def rank_candidates(candidates):
    """Rank all three completed candidates; step tie-break is within-LR only."""
    values = list(candidates)
    if len(values) != 3 or {c.get('lr') for c in values} != set(LR_CANDIDATES):
        raise ValueError('Ranking requires exactly the three fixed LR candidates')
    if len({c['groups_per_cycle'] for c in values}) != 1:
        raise ValueError('Candidate group budgets differ')
    for c in values:
        if 'digest' in c:
            verify(c)
        expected = summarize_candidate(c['lr'], c['events'], c['groups_per_cycle'], cached_control=c['cached_control'])
        for key in ('metric_selected', 'legacy_objective_selected', 'endpoint', 'new_optimizer_updates'):
            if c[key] != expected[key]:
                raise ValueError('Candidate summary selection mismatch: ' + key)
    ranked = sorted(values, key=lambda c: (-c['metric_selected']['validation_AUC'],
        -c['metric_selected']['validation_AP'], c['lr'] != CONTROL_LR, c['lr']))
    # Preserve the candidate digest as a source identity instead of keeping a
    # now-invalid digest after appending rank.
    return [bound(dict({k: deepcopy(v) for k, v in c.items() if k != 'digest'},
        rank=i, source_candidate_digest=c.get('digest'), lr_selector=LR_SELECTOR))
        for i, c in enumerate(ranked, 1)]
