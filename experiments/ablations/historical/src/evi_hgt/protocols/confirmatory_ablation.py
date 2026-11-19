"""Single explicitly post-freeze, pre-test confirmatory ablation supplemental cell."""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

from evi_hgt.protocols.formal_experiments import (
    FORMAL_MANIFEST_RELATIVE_PATH, FORMAL_RUN_STATUSES, PRACTICAL_PROTOCOL_SHA256,
    _canonical_digest, _file_sha256, _replace_manifest_with_retry,
    compute_formal_run_id, execution_identity, validate_execution_lineage,
)
from evi_hgt.protocols.scoped_execution import (
    MAIN_ID, SCOPED_PATH, SELECTED_PATH, validate_selection,
)

VARIANT = 'NO_LOGIC_MASS_CONDITIONING'
SCOPE = 'PHASE8_2_6A_CONFIRMATORY_V1'
AMENDMENT_PATH = 'outputs/phase8_2_6a/confirmatory_amendment.json'
MANIFEST_PATH = 'outputs/phase8_2_6a/supplemental_execution_manifest.json'
CATALOG_PATH = 'outputs/phase7_3_3_1_protocol/variant_catalog.json'
NO_FOUR_ID = 'e5d577b3c5bfb203520f6541d3c78357420cd7a084076e57fe3fc2d909312e48'
MODEL_SOURCES = ('src/evi_hgt/models/variants.py', 'src/evi_hgt/models/evi_hgt_layer.py',
                 'src/evi_hgt/training/model_variants.py')


def _read(root, relative):
    return json.loads((Path(root)/relative).read_text(encoding='utf-8'))


def _test_separation(payload):
    # Historical 52-cell schema has per-attempt counts, no top-level count.
    # Scoped generations require both levels. Missing attempt counts fail closed.
    if payload.get('formal_test_count', 0 if 'execution_scope' not in payload else None) != 0 or any(
        a.get('formal_test_count') != 0
        for r in payload['runs'] for a in r.get('execution_attempts', [])
    ):
        raise ValueError('confirmatory amendment requires formal test count zero')
    if any(r['execution_status'] != 'PLANNED' and not r.get('execution_attempts') for r in payload['runs']):
        raise ValueError('executed source lacks auditable test separation lineage')


def build_amendment(root, amendment_time):
    """Derive an honest amendment from existing validation-only observations."""
    from evi_hgt.models.variants import get_variant_spec
    root = Path(root)
    if datetime.fromisoformat(amendment_time).tzinfo is None:
        raise ValueError('amendment time requires an explicit timezone')
    history, scoped = _read(root, FORMAL_MANIFEST_RELATIVE_PATH), _read(root, SCOPED_PATH)
    _test_separation(history); _test_separation(scoped)
    selected = _read(root, SELECTED_PATH)
    validate_selection(root, selected, history)
    old = next(r for r in scoped['runs'] if r['run_id'] == NO_FOUR_ID)
    if old['execution_status'] != 'COMPLETED' or old['execution_attempts'][-1]['completed_epochs'] != 2:
        raise ValueError('existing logic ablation lacks completed validation trajectory')
    log_path = old['execution_attempts'][-1]['output_directory']+'/logs/run.log'
    log = (root/log_path).read_text(encoding='utf-8')
    if ('validation_objective=0.123508669' not in log or
        'reason=MAX_TRAINING_HORIZON_REACHED epoch=2 optimizer_step=18760' not in log):
        raise ValueError('observed validation-only result does not match source log')
    full = next(r for r in _read(root, CATALOG_PATH)['variants'] if r['variant_id']=='FULL_EVIHGT')
    payload = {
        'amendment_time': amendment_time,
        'classification': 'POST_FREEZE_PRE_FORMAL_TEST_CONFIRMATORY_ABLATION_AMENDMENT',
        'originally_preregistered': False,
        'reason': 'Existing NO_FOUR_VALUED_LOGIC changes multiple graph-internal logic paths; '
                  'it does not purely isolate dynamic logic-mass message conditioning.',
        'observed_validation_objectives': {'FULL_EVIHGT': 0.13095593452453613,
                                          'NO_FOUR_VALUED_LOGIC': 0.123508669},
        'observation_scope': 'DETERMINISTIC_VALIDATION_ONLY_NOT_FINAL_RETRIEVAL',
        'observation_source': {'path': log_path, 'sha256': _file_sha256(root/log_path)},
        'formal_test_count': 0,
        'no_formal_test_information_used': True,
        'added_variants': [VARIANT],
        'other_experiments_added': 0,
        'hyperparameters_changed': False,
        'no_further_logic_ablations_regardless_of_outcome': True,
        'question': 'Incremental information in direct edge-specific T/F/B/U mass-conditioned routing.',
        'routing_weights_when_available': [0.25, 0.25, 0.25, 0.25],
        'unavailable_rule': 'EXACT_FULL_W_NA_NOT_A_FIFTH_LOGIC_STATE',
        'retained_mass_uses': 'Original logic computation and masses in edge context/path inputs remain FULL.',
        'claim_limit': 'Not four-valued theory versus binary logic; not removal of all logic information.',
        'formal_training_protocol_digest': PRACTICAL_PROTOCOL_SHA256,
        'selected_lr_source': {'path': SELECTED_PATH, 'sha256': _file_sha256(root/SELECTED_PATH),
                              'run_id': MAIN_ID, 'test_used_for_selection': False},
        'variant_record': {**get_variant_spec(VARIANT).to_payload(),
                           'total_parameters': full['total_parameters'],
                           'trainable_parameters': full['trainable_parameters']},
        'implementation_sha256': {p: _file_sha256(root/p) for p in MODEL_SOURCES},
    }
    # Normalize tuples exactly as persisted JSON before computing identity.
    payload = json.loads(json.dumps(payload))
    return {**payload, 'digest': _canonical_digest(payload)}


def validate_amendment(root, payload):
    expected = build_amendment(root, payload['amendment_time'])
    if payload != expected:
        raise ValueError('confirmatory amendment provenance/semantics/digest mismatch')
    return payload


def load_amendment(root):
    return validate_amendment(root, _read(root, AMENDMENT_PATH))


def build_confirmatory_manifest(root, amendment):
    from evi_hgt.formal_execution import formal_output_directory
    validate_amendment(root, amendment)
    history = _read(root, FORMAL_MANIFEST_RELATIVE_PATH)
    run = deepcopy(next(r for r in history['runs'] if r['run_id']==MAIN_ID))
    run.update(model_variant=VARIANT, run_family='ablations', requirement='CONFIRMATORY',
               execution_scope=SCOPE, source_cell_id=MAIN_ID,
               source_cell_role='FULL_SCIENTIFIC_CONFIGURATION_REFERENCE_ONLY_NOT_TRAINING_LINEAGE',
               selected_lr_source=deepcopy(amendment['selected_lr_source']),
               amendment_digest=amendment['digest'], amendment_path=AMENDMENT_PATH,
               learning_rate=5e-5, learning_rate_binding=5e-5, execution_status='PLANNED')
    run['architecture'].update(model_variant=VARIANT, use_logic_mass_conditioning=False)
    run['architecture_digest'] = _canonical_digest(run['architecture'])
    run['run_id'] = compute_formal_run_id(run)
    run['execution_attempts'] = [{**execution_identity(run['run_id'], 1), 'status':'PLANNED',
        'registration_scope':'POST_FREEZE_CONFIRMATORY_REQUIRES_EXACT_RUN_AUTHORIZATION',
        'fresh_start_only':True, 'resumed':False, 'completed_epochs':0,
        'observed_optimizer_steps':0, 'recoverable_optimizer_steps':0,
        'best_exists':False, 'last_exists':False, 'validation_reached':False,
        'formal_test_count':0, 'output_directory':formal_output_directory('ablations',run['run_id'])}]
    return {'execution_scope':SCOPE, 'amendment_digest':amendment['digest'],
            'amendment_path':AMENDMENT_PATH, 'formal_test_count':0,
            'execution_authorized':False, 'runs':[run]}


def validate_confirmatory_manifest(root, payload, *, amendment=None):
    amendment = load_amendment(root) if amendment is None else amendment
    expected = build_confirmatory_manifest(root, amendment)
    if len(payload.get('runs',[])) != 1:
        raise ValueError('confirmatory scope permits exactly one cell')
    _test_separation(payload)
    actual, template = payload['runs'][0], expected['runs'][0]
    if actual.get('execution_status') not in FORMAL_RUN_STATUSES:
        raise ValueError('invalid supplemental execution status')
    validate_execution_lineage(actual)
    if len(actual['execution_attempts']) != 1:
        raise ValueError('supplemental amendment registers one initial attempt only')
    for key, value in template.items():
        if key not in {'execution_status','execution_attempts'} and actual.get(key)!=value:
            raise ValueError(f'confirmatory immutable binding drift: {key}')
    if set(actual)!=set(template):
        raise ValueError('unexpected supplemental run fields')
    for key in ('output_directory','registration_scope','fresh_start_only'):
        if actual['execution_attempts'][0].get(key)!=template['execution_attempts'][0][key]:
            raise ValueError(f'confirmatory attempt binding drift: {key}')
    for key in expected.keys()-{'runs'}:
        if payload.get(key)!=expected[key]:
            raise ValueError(f'confirmatory manifest binding drift: {key}')
    return dict(payload)


def supplemental_variant_record(root):
    amendment = load_amendment(root)
    record = amendment['variant_record']
    return {key:record[key] for key in ('variant_id','status','total_parameters','trainable_parameters')} | {
        'catalog_artifact':AMENDMENT_PATH, 'catalog_artifact_sha256':_file_sha256(Path(root)/AMENDMENT_PATH)}


def write_once_json(path, payload):
    """Immutable registration using the existing finite-retry atomic replacement."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name('.'+path.name+'.registration.lock')
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) == payload:
                return
            raise ValueError('refusing to overwrite an existing amendment/registration')
        with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix='.'+path.name+'.',
                                          suffix='.tmp',delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(payload,indent=2,sort_keys=True)+'\n').encode('utf-8'))
            stream.flush(); os.fsync(stream.fileno())
        try:
            _replace_manifest_with_retry(temporary,path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        os.close(fd); lock.unlink()
