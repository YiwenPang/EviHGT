"""Original-fold M4 data identity and training-only bridge; no evaluation route.

Metadata for all 15 cells is readable. Materialization remains restricted to the
two approved TEST_ONLY cells until a separate, exact M4 G3 permit exists.
"""
import json
from pathlib import Path
import sys

from .identity import ROOT, G1_SHA256, canonical, digest, file_sha, frozen_science
from . import txgnn_adapter as legacy

PAYLOAD_SCHEMA = 'PHASE92C_TXGNN_TRAINING_ONLY_PAYLOAD_V1'
TEST_ROOT = ROOT / 'outputs/phase9/phase92c/test_only'
U_MANIFEST = ROOT / 'data/processed/phase85_part2/training_u_manifest.json'
U_MANIFEST_SHA = '55cb64bea05d4647dda115b2e338659f6ad49a14360bda249d4075faf9771038'
OBSERVED = ROOT / 'data/processed/phase85_part2/observed_pair_exclusions.parquet'
OBSERVED_SHA = 'b0645292873711cf1df5c26391d5cb40cad62e14ca5b46341981c72ffad29083'


def _cell(repeat, fold):
    if type(repeat) is not int or type(fold) is not int:
        raise ValueError('Original integer repeat/fold required')
    rows = [c for c in frozen_science()['cells']
            if c['repeat_index'] == repeat and c['fold_index'] == fold]
    if len(rows) != 1:
        raise ValueError('Unknown original M4 cell')
    return rows[0]


def cell_binding(repeat, fold):
    cell = _cell(repeat, fold)
    return {'schema': 'PHASE92C_M4_CELL_BINDING_V1', 'method': 'M4',
            'repeat': repeat, 'fold': fold, 'seed': cell['training_seed'],
            'cell_digest': cell['cell_digest'], 'cell': cell,
            'g1_sha256': G1_SHA256, 'native_sources': legacy.verify_native_sources()}


def require_materialization(mode, repeat, fold, permit=None):
    _cell(repeat, fold)
    if mode == 'TEST_ONLY':
        if permit is not None or (repeat, fold) not in {(0, 1), (1, 0)}:
            raise PermissionError('Only the two predeclared TEST_ONLY cells may be materialized')
        return None
    if mode == 'FORMAL':
        from .execution import verify_permit
        return verify_permit(permit, 'M4', repeat, fold)
    raise PermissionError('Only separately gated TEST_ONLY or FORMAL training data is exposed')


def _trusted_u():
    """Read only the frozen U domain and role-free observed-pair exclusions."""
    if file_sha(U_MANIFEST) != U_MANIFEST_SHA or file_sha(OBSERVED) != OBSERVED_SHA:
        raise ValueError('Frozen U/observed source identity changed')
    universe = json.loads(U_MANIFEST.read_text(encoding='utf-8'))
    if universe['digest'] != digest({k: v for k, v in universe.items() if k != 'digest'}):
        raise ValueError('Training U manifest content identity changed')
    if universe['observed_pair_sha256'] != OBSERVED_SHA or universe['U_is_confirmed_negative'] is not False:
        raise ValueError('Training U semantic/source binding changed')
    import polars as pl
    observed = set(pl.read_parquet(OBSERVED, columns=['drug_id', 'disease_id']).iter_rows())
    return universe, observed


def _pairs(values, name):
    if not isinstance(values, (list, tuple)):
        raise ValueError(name + ' must contain explicit pair identities')
    result = []
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != 2 or any(not isinstance(x, str) for x in row):
            raise ValueError('Malformed ' + name)
        result.append(tuple(row))
    if len(set(result)) != len(result):
        raise ValueError('Duplicate ' + name)
    return set(result)


def validate_training_payload(payload, corpus):
    """Verify true training memberships, rather than trusting a self-hashed pair list."""
    if payload.get('schema') != PAYLOAD_SCHEMA or payload.get('g1_sha256') != G1_SHA256:
        raise ValueError('M4 payload schema/G1 differs')
    if payload.get('digest') != digest({k: v for k, v in payload.items() if k != 'digest'}):
        raise ValueError('M4 payload integrity differs')
    cell = _cell(payload['repeat'], payload['fold'])
    if canonical(payload['cell']) != canonical(cell):
        raise ValueError('M4 original cell/seed/member identity differs')
    for name in ('training', 'auxiliary_training'):
        if digest(payload[name]) != cell['membership_digests'][name]:
            raise ValueError('Original M4 training membership differs: ' + name)
    if any(set(row) != {'drug_id', 'disease_id', 'role'} or row['role'] not in {'P', 'U'}
           for row in payload['training']):
        raise ValueError('Unverified support training role')
    if any(set(row) != {'drug_id', 'disease_id', 'raw_relation'} or row['raw_relation'] != 'CONTRAINDICATION'
           for row in payload['auxiliary_training']):
        raise ValueError('Unverified contraindication supervision')
    support = _pairs(payload['support_pairs'], 'support pairs')
    contra = _pairs(payload['contra_pairs'], 'contra pairs')
    expected_support = {(r['drug_id'], r['disease_id']) for r in payload['training'] if r['role'] == 'P'}
    expected_contra = {(r['drug_id'], r['disease_id']) for r in payload['auxiliary_training']}
    if support != expected_support or contra != expected_contra:
        raise ValueError('Support/contra pairs differ from original training members')
    if (len(support), len(contra)) != (cell['budget']['full_training_support_P'], cell['budget']['full_training_contra_P']):
        raise ValueError('Original training supervision count differs')
    forbidden = _pairs(payload['forbidden_pairs'], 'role-free forbidden pairs')
    if digest(sorted(forbidden)) != cell['mask_digest'] or (support | contra) & forbidden:
        raise ValueError('Frozen forbidden mask or supervised exclusion differs')
    universe, observed = _trusted_u()
    if _pairs(payload['observed_pairs'], 'role-free observed pairs') != observed:
        raise ValueError('Observed-pair exclusion domain differs')
    for field in ('eligible_drugs', 'eligible_diseases'):
        if payload[field] != universe[field]:
            raise ValueError('Frozen eligible U domain differs: ' + field)
    drugs, diseases = set(payload['eligible_drugs']), set(payload['eligible_diseases'])
    all_excluded = forbidden | observed
    excluded = sum(d in drugs and s in diseases for d, s in all_excluded)
    domain = dict(eligible_drugs=len(drugs), eligible_diseases=len(diseases),
                  excluded_unique_pairs=excluded, legal_U_count=len(drugs)*len(diseases)-excluded,
                  rectangular_pairs=len(drugs)*len(diseases), U_is_confirmed_negative=False,
                  with_replacement=True, role_free_final_exclusions_only=True)
    if domain != cell['legal_training_u_domain']:
        raise ValueError('Original legal U universe differs')
    for row in payload['training']:
        if row['role'] == 'U' and ((row['drug_id'], row['disease_id']) in all_excluded
                                 or row['drug_id'] not in drugs or row['disease_id'] not in diseases):
            raise ValueError('Original training U is illegal')
    if corpus.get('digest') != digest({k: v for k, v in corpus.items() if k != 'digest'}) or payload['corpus_digest'] != corpus['digest']:
        raise ValueError('Current corpus integrity differs')
    bound = payload['bindings']
    required = dict(fold_manifest_digest=cell['cell_digest'], original_repeat_index=payload['repeat'],
                    fold_index=payload['fold'], training_seed=cell['training_seed'],
                    fold_mask_digest=cell['mask_digest'], training_u_manifest_digest=universe['digest'])
    if any(bound.get(k) != v for k, v in required.items()):
        raise ValueError('Current corpus/data binding differs')
    if corpus['bindings'] != dict(bound, clean_graph_digest=corpus['bindings']['clean_graph_digest']):
        raise ValueError('Current corpus bindings differ from validated training data')
    native_sources = legacy.verify_native_sources()
    if payload['native_sources'] != native_sources:
        raise ValueError('Native M4 source identity changed')
    binding = cell_binding(payload['repeat'], payload['fold'])
    binding.update(mode=payload['mode'], corpus_digest=corpus['digest'],
        training_membership_digest=cell['membership_digests']['training'],
        auxiliary_training_membership_digest=cell['membership_digests']['auxiliary_training'],
        mask_digest=cell['mask_digest'], legal_u_domain=domain,
        legal_u_domain_digest=digest({'drugs': payload['eligible_drugs'], 'diseases': payload['eligible_diseases'],
                                     'excluded': sorted(all_excluded)}),
        supervision_digest=digest({'support': sorted(support), 'contra': sorted(contra)}),
        payload_digest=payload['digest'], training_u_manifest_sha256=U_MANIFEST_SHA,
        observed_pairs_sha256=OBSERVED_SHA)
    return binding


def training_payload(data, corpus, *, mode='TEST_ONLY', permit=None):
    repeat, fold = data.repeat_index, data.cell['fold']
    require_materialization(mode, repeat, fold, permit)
    payload = dict(schema=PAYLOAD_SCHEMA, mode=mode, repeat=repeat, fold=fold,
        cell=_cell(repeat, fold), bindings=dict(data.bindings), corpus_digest=corpus['digest'],
        corpus_path=str(Path(corpus['corpus_root']) / 'manifest.json'),
        training=data.cell['training'], auxiliary_training=data.cell['auxiliary_training'],
        support_pairs=sorted(data.support), contra_pairs=sorted(data.unsuitability),
        forbidden_pairs=sorted(data.blocked), observed_pairs=sorted(data.shared.observed),
        eligible_drugs=data.u['eligible_drugs'], eligible_diseases=data.u['eligible_diseases'],
        g1_sha256=G1_SHA256, native_sources=legacy.verify_native_sources())
    payload['digest'] = digest(payload)
    validate_training_payload(payload, corpus)
    return payload


def prepare_payload(payload, corpus, *, mode='TEST_ONLY', permit=None, audit_only=False):
    require_materialization(mode, payload['repeat'], payload['fold'], permit)
    if payload.get('mode') != mode:
        raise PermissionError('TEST_ONLY data payload cannot be relabelled FORMAL')
    binding = validate_training_payload(payload, corpus)
    return legacy._prepare_fold_core(corpus=corpus, support_pairs=payload['support_pairs'],
        contra_pairs=payload['contra_pairs'], forbidden_pairs=payload['forbidden_pairs'],
        binding=binding, audit_only=audit_only)


def prepare_cell(data, corpus, *, mode='TEST_ONLY', permit=None, audit_only=False):
    payload = training_payload(data, corpus, mode=mode, permit=permit)
    return prepare_payload(payload, corpus, mode=mode, permit=permit, audit_only=audit_only)


def stream_factory_for_cell(native_data, data):
    """Return independent native stream seeds as cell_seed + stage offset."""
    def factory(offset):
        if type(offset) is not int:
            raise ValueError('Integer native stage seed offset required')
        return legacy.TrainingOnlyU(drugs=data.u['eligible_drugs'], diseases=data.u['eligible_diseases'],
            observed=data.shared.observed, forbidden=data.blocked,
            drug_local=native_data.drug_local, disease_local=native_data.disease_local,
            seed=data.training_seed + offset)
    return factory


def runtime_identity(native_data):
    """Actual native environment and adapter identity, without claiming G3 authority."""
    import dgl
    import numpy as np
    import torch
    source = {**legacy.verify_native_sources(), str(Path(__file__).resolve()): file_sha(__file__),
              str(Path(legacy.__file__).resolve()): file_sha(legacy.__file__)}
    environment = {'executable': str(Path(sys.executable).resolve()), 'python': str(sys.version),
        'python_executable_sha256': file_sha(sys.executable), 'dgl': str(dgl.__version__),
        'torch': str(torch.__version__), 'numpy': str(np.__version__), 'device': 'cpu',
        'cpu_threads': torch.get_num_threads(),
        'dgl_loader_sha256': file_sha(Path(dgl.__file__).parent / 'dataloading/dataloader.py'),
        'torch_loader_sha256': file_sha(Path(torch.__file__).parent / 'utils/data/dataloader.py')}
    return dict(native_data.binding, source_digest=digest(source), environment_digest=digest(environment),
                source_sha256=source, environment=environment)


def export_training_payload(data, corpus, path, *, mode='TEST_ONLY', permit=None):
    """Create an immutable bridge; no model, scoring, or role-bearing panel is loaded."""
    path = Path(path).resolve()
    if mode == 'TEST_ONLY':
        if not path.is_relative_to(TEST_ROOT.resolve()) or path == TEST_ROOT.resolve():
            raise PermissionError('M4 TEST_ONLY payload must stay inside phase92c/test_only')
    else:
        permit = require_materialization(mode, data.repeat_index, data.cell['fold'], permit)
        if not path.is_relative_to(Path(permit['output_root']).resolve()):
            raise PermissionError('M4 formal payload differs from approved namespace')
    payload = training_payload(data, corpus, mode=mode, permit=permit)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, separators=(',', ':'), allow_nan=False)
    return {'path': str(path), 'sha256': file_sha(path), 'payload_digest': payload['digest'],
            'corpus_digest': corpus['digest'], 'mode': mode}


def load_training_payload(path, *, mode='TEST_ONLY', permit=None):
    """Native-environment bridge with a fresh legal-U stream factory, never weights."""
    if mode not in {'TEST_ONLY', 'FORMAL'}:
        raise PermissionError('Only separately authorized training modes may load M4 data')
    if mode == 'FORMAL':
        from .execution import verify_permit
        # The exact manifest must cover every fixed cell. Verify authority before
        # reading the caller-supplied file, then verify its actual cell below.
        permit = verify_permit(permit, 'M4', 0, 0)
    path = Path(path).resolve()
    if mode == 'TEST_ONLY' and not path.is_relative_to(TEST_ROOT.resolve()):
        raise PermissionError('Only newly bound Phase92c TEST_ONLY payloads may be loaded')
    if mode == 'FORMAL' and not path.is_relative_to(Path(permit['output_root']).resolve()):
        raise PermissionError('Formal M4 payload differs from the approved namespace')
    payload = json.loads(path.read_text(encoding='utf-8'))
    require_materialization(mode, payload['repeat'], payload['fold'], permit)
    corpus = json.loads(Path(payload['corpus_path']).read_text(encoding='utf-8'))
    data = prepare_payload(payload, corpus, mode=mode, permit=permit)
    def stream_factory(offset):
        if type(offset) is not int:
            raise ValueError('Integer native stage seed offset required')
        return legacy.TrainingOnlyU(drugs=payload['eligible_drugs'], diseases=payload['eligible_diseases'],
            observed=payload['observed_pairs'], forbidden=payload['forbidden_pairs'],
            drug_local=data.drug_local, disease_local=data.disease_local,
            seed=data.binding['seed'] + offset)
    identity = dict(runtime_identity(data), payload_path=str(path), payload_sha256=file_sha(path))
    return data, stream_factory, identity
