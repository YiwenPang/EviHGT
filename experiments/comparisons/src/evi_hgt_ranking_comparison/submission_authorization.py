"""Exact R0 CV5 submission scope derived from the immutable V6 science plan.

This validates a versioned reduction of the execution roster, never a grant.
Historical planning/reuse rows remain unchanged; completed FULL results are
accounted for by a separate completion audit, with no new scoring permission.
"""
from copy import deepcopy
from pathlib import Path

from .authorization import APPROVED_V6, APPROVED_V6_SHA
from .common import canonical, digest, read, sha


SUBMISSION_SCHEMA = 'RANKING_COMPARISON_SUBMISSION_CV5_MANIFEST_V1'
SUBMISSION_CELLS = tuple(f'R0F{i}' for i in range(5))
SUBMISSION_SCOPE = {
    'scope_id': 'SUBMISSION_CV5_R0_SIX_METHODS_V1',
    'original_repeat_index': 0,
    'cell_count': 5,
    'method_count': 6,
    'new_training_count': 19,
    'development_count': 5,
    'reused_training_count': 6,
    'planning_rows_semantics': 'IMMUTABLE_PARENT_V6_HISTORY',
    'reuse_evidence_semantics': 'SEPARATE_COMPLETION_AUDIT_NO_NEW_FORWARD',
}


def validate_submission_cv5_scope(manifest, role):
    """Reject any non-authorized roster, input, seed, budget, or scope change."""
    if manifest.get('schema') != SUBMISSION_SCHEMA or role != 'G3':
        raise PermissionError('Only the submission CV5 G3 execution format is supported')
    if sha(APPROVED_V6) != APPROVED_V6_SHA:
        raise ValueError('Original approved V6 bytes changed')
    original = read(APPROVED_V6)
    sources = manifest.get('source_sha256', {})
    required_sources = set(original['source_sha256']) | {str(Path(__file__).resolve())}
    if not isinstance(sources, dict) or not required_sources.issubset(sources):
        raise ValueError('CV5 source roster must retain V6 dependencies and bind its validator')
    if manifest.get('approval_base') != {'path': str(APPROVED_V6), 'sha256': APPROVED_V6_SHA}:
        raise PermissionError('Exact immutable V6 approval reference required')
    for key in ('outer_eval_authorized', 'final_eval_authorized',
                'research_diagnostic_forward_authorized', 'G4_authorized',
                'G5_authorized', 'diagnostics_authorized'):
        if manifest.get(key) is not False:
            raise PermissionError('Scope remains closed: ' + key)
    if manifest.get('max_model_workers') != 1 or manifest.get('G3_authorized') is not True:
        raise PermissionError('One worker and explicit G3 scope required')

    expected = deepcopy(original)
    expected.update(
        schema=SUBMISSION_SCHEMA,
        cell_order=list(SUBMISSION_CELLS),
        scientific_roster_status='SUBMISSION_SIX_METHODS_R0_CV5',
        nominal_training=25, accepted_training_credits=6, new_training_count=19,
        submission_scope=deepcopy(SUBMISSION_SCOPE),
        approval_base={'path': str(APPROVED_V6), 'sha256': APPROVED_V6_SHA},
    )
    # These are launch/version explanation fields, not scientific contracts.
    metadata = ('created_utc', 'enabled', 'grant_id', 'formal_training_authorized',
                'G3_authorized', 'source_sha256', 'source_closure', 'digest',
                'launch_status', 'qualification_stop_current', 'note', 'revision_reason')
    for key in metadata:
        if key not in manifest:
            raise ValueError('Missing CV5 field: ' + key)
        expected[key] = manifest[key]

    expected['training_tasks'] = [r for r in expected['training_tasks']
                                  if r['method'] != 'FULL' and r['cell'] in SUBMISSION_CELLS]
    expected['development_tasks'] = [r for r in expected['development_tasks']
                                     if r['cell'] in SUBMISSION_CELLS]
    if len(expected['training_tasks']) != 19 or len(expected['development_tasks']) != 5:
        raise ValueError('Immutable V6 does not yield the required CV5 roster')
    for key in ('training_tasks', 'development_tasks'):
        actual = manifest.get(key)
        if not isinstance(actual, list) or len(actual) != len(expected[key]):
            raise ValueError('Changed CV5 roster: ' + key)
        for before, after in zip(expected[key], actual):
            if not isinstance(after, dict):
                raise ValueError('Invalid CV5 task')
            for field in ('source_digest', 'source_closure', 'launch_status'):
                if field not in after:
                    raise ValueError('Missing CV5 task field: ' + field)
                before[field] = after[field]
            before['training_authorized'] = key == 'training_tasks'
            before['score_authorized'] = True

    executable = {(r['method'], r['cell'])
                  for r in expected['training_tasks'] + expected['development_tasks']}
    expected['g3_evaluation_tasks'] = [r for r in expected['g3_evaluation_tasks']
                                      if (r['method'], r['cell']) in executable]
    for row in expected['g3_evaluation_tasks']:
        row['authorized'] = True
    expected['g3_new_panel_scoring_jobs'] = sum(
        r['new_panel_jobs'] for r in expected['g3_evaluation_tasks'])
    # Canonical JSON equality also rejects bool/int substitutions. Every field
    # not explicitly changed above, including nested denied scopes, is frozen.
    if canonical(expected) != canonical(manifest):
        raise ValueError('CV5 scientific content or G3-only scope differs from approved subset')
    if manifest['digest'] != digest({k: v for k, v in manifest.items() if k != 'digest'}):
        raise ValueError('CV5 execution content digest differs')
