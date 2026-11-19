"""Strict, CPU selected-weight loading; no scorer, optimizer, or execution grant.

FULL reads its exact fixed-3G model. HGTDR reads the retained epoch-222 best
model from the complete epoch-300 checkpoint. The endpoint optimizer is never
attached to either scoring receiver, and the returned object cannot resume.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

import torch

from evi_hgt.phase85_v2.common import digest, verify
from evi_hgt.phase9.production_state import training_state_digest
from .common import ROOT, OUT, read, sha
from .reuse import select_weights, tensor_schema


def extract_bound_selected(state, receipt):
    """Extract only tensor weights, with exact source, selection, and value checks."""
    try:
        selected, selection = select_weights(state, receipt['method'])
        schema = tensor_schema(selected)
        recovery = tensor_schema(state['model'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Selected tensor field is missing or malformed; no fallback') from exc
    selected_digest = training_state_digest(selected)
    recovery_digest = training_state_digest(state['model'])
    if (selection != receipt['selection'] or schema != recovery
        or selected_digest != receipt['selected_tensor_sha256']
        or recovery_digest != receipt['recovery_tensor_sha256']
        or (selected_digest == recovery_digest) != receipt['selected_equals_recovery']
        or digest(schema) != receipt['tensor_schema_digest']
        or len(schema) != receipt['tensor_entries']
        or sum(v.numel() for v in selected.values()) != receipt['tensor_values']):
        raise ValueError('Selected source/selection/tensor identity differs from bound receipt')
    # Detach ownership from the complete checkpoint, including native_control.
    return {name:value.detach().clone() for name,value in selected.items()}


def validate_receiver_schema(current, selected):
    """All receiver keys, shapes, dtypes and CPU devices are checked before load."""
    actual, expected = tensor_schema(current), tensor_schema(selected)
    if actual != expected:
        raise ValueError('Scoring receiver tensor schema differs from selected checkpoint')
    return digest(actual)


@dataclass(frozen=True)
class SelectedWeights:
    weights: dict
    identity: dict

    def validate(self):
        if (training_state_digest(self.weights) != self.identity['selected_tensor_sha256']
            or digest(tensor_schema(self.weights)) != self.identity['tensor_schema_digest']):
            raise ValueError('Selected weights changed after receipt validation')
        if self.identity.get('can_resume_training') is not False:
            raise ValueError('Scoring-only selected state cannot authorize training recovery')
        return True


def read_selected_weights(receipt_path, *, expected_receipt_sha256, method, cell,
                          expected_data_bindings, expected_input_identity):
    """CPU-read a registered reuse receipt, COMMIT, and full-state checkpoint.

    Expected receipt hash and current input/data identities must come from the
    caller's separately registered run specification. No role/panel data is read.
    """
    if method not in ('FULL','HGTDR_ADAPTED') or re.fullmatch(r'R[0-2]F[0-4]',cell or '') is None:
        raise ValueError('Registered historical method/cell required')
    if not isinstance(expected_receipt_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_receipt_sha256) is None:
        raise ValueError('An externally bound receipt SHA256 is required')
    path = Path(receipt_path).resolve()
    if path != (OUT/'reuse'/f'{method}_{cell}.json').resolve() or sha(path) != expected_receipt_sha256:
        raise ValueError('Selected reuse receipt path/hash differs')
    receipt = verify(read(path))
    if (receipt['schema'] != 'PHASE98_SELECTED_WEIGHT_REUSE_BINDING_V1'
        or receipt['status'] != 'CPU_WEIGHT_AND_SOURCE_VERIFIED_NO_FORWARD'
        or receipt['method'] != method or receipt['cell'] != cell
        or receipt['data_bindings'] != expected_data_bindings
        or receipt['input_identity'] != expected_input_identity
        or receipt['map_location'] != 'cpu' or receipt['weights_only'] is not True):
        raise ValueError('Selected reuse method/cell/data/input binding differs')
    summary = verify(read(OUT/'reuse/REUSE_VERIFIED.json'))
    matches = [r for r in summary['rows'] if r['method'] == method and r['cell'] == cell]
    if summary['status'] != 'VERIFIED_NO_FORWARD' or matches != [receipt]:
        raise ValueError('Receipt is not one of the audited historical reuse bindings')
    if sha(ROOT/'src/evi_hgt_phase98/reuse.py') != receipt['new_wrapper_sha256']:
        raise ValueError('Sealed reuse auditor implementation changed')
    for row in receipt['source_verification']+receipt['implementation_verification']:
        if (row['match'] is not True or row['actual_sha256'] != row['expected_sha256']
            or sha(row['path']) != row['expected_sha256']):
            raise ValueError('Selected historical source closure changed: '+row['path'])
    checkpoint = Path(receipt['checkpoint_path'])
    commit_path = Path(receipt['commit_path'])
    if sha(commit_path) != receipt['commit_sha256']:
        raise ValueError('Historical COMMIT bytes changed')
    commit = read(commit_path)
    checkpoint_hash = sha(checkpoint)
    if (checkpoint_hash != receipt['checkpoint_sha256']
        or checkpoint_hash != commit['checkpoint_sha256']
        or checkpoint.stat().st_size != commit['checkpoint_size']
        or commit['status'] != 'COMMITTED'
        or digest({k:v for k,v in commit.items() if k != 'commit_digest'}) != commit['commit_digest']):
        raise ValueError('Historical checkpoint/COMMIT integrity differs')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    actual_state_digest = training_state_digest({k:v for k,v in state.items() if k != 'state_digest'})
    if (actual_state_digest != state['state_digest'] or actual_state_digest != commit['state_digest']
        or actual_state_digest != receipt['complete_state_digest'] or state['identity'] != commit['identity']):
        raise ValueError('Historical complete-state identity changed')
    selected = extract_bound_selected(state,receipt)
    if sha(checkpoint) != checkpoint_hash or sha(path) != expected_receipt_sha256:
        raise ValueError('Historical read-only source changed during CPU load')
    del state
    result = SelectedWeights(selected, dict(schema='PHASE98_SCORING_SELECTED_WEIGHTS_V1',
        method=method,cell=cell,receipt_path=str(path),receipt_sha256=expected_receipt_sha256,
        checkpoint_sha256=checkpoint_hash,selected_tensor_sha256=receipt['selected_tensor_sha256'],
        tensor_schema_digest=receipt['tensor_schema_digest'],selection=deepcopy(receipt['selection']),
        data_bindings=deepcopy(expected_data_bindings),input_identity=deepcopy(expected_input_identity),
        can_resume_training=False,optimizer_created=False,optimizer_restored=False,
        model_forward_calls=0,execution_authorized=False))
    result.validate()
    return result


def load_selected_for_scoring(model, receipt_path, *, expected_receipt_sha256, method, cell,
                              expected_data_bindings, expected_input_identity):
    """Strictly load an existing CPU model; never construct or restore an optimizer."""
    if not isinstance(model,torch.nn.Module):
        raise TypeError('Caller must supply a fresh, architecture-bound CPU module')
    selected = read_selected_weights(receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,method=method,cell=cell,
        expected_data_bindings=expected_data_bindings,expected_input_identity=expected_input_identity)
    validate_receiver_schema(model.state_dict(),selected.weights)
    selected.validate()
    loaded = model.load_state_dict(selected.weights,strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('Strict selected model load left unmatched tensors')
    if training_state_digest(model.state_dict()) != selected.identity['selected_tensor_sha256']:
        raise ValueError('Loaded scoring model values differ from selected weights')
    model.eval()
    return dict(deepcopy(selected.identity),strict_schema_loaded=True,model_mode='eval')
