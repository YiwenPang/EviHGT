"""Inner-development metrics and atomic, digest-bound pilot recovery."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import statistics

import numpy as np
import polars as pl
import torch

from evi_hgt.graph_inputs.common import bound, digest, read, save, sha, verify
from evi_hgt.evaluation.validation_diagnostics import _threshold_curve
from evi_hgt.evaluation.protocol import EvaluationProtocolBinding
from evi_hgt.evaluation.ranking import evaluate_disease_ranking


def weight_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def inner_metrics(rows):
    seen, groups = set(), defaultdict(list)
    for r in rows:
        pair = (r['drug_id'], r['disease_id'])
        if pair in seen or r['role'] not in ('P', 'U') or not math.isfinite(float(r['support_logit'])):
            raise ValueError('Invalid inner metric input')
        seen.add(pair)
        groups[pair[1]].append(r)
    local = []
    binding = EvaluationProtocolBinding('standard', 'validation', 'support')
    for disease, candidates in sorted(groups.items()):
        candidates = sorted(candidates, key=lambda r: r['drug_id'])
        p = np.array([r['support_logit'] for r in candidates if r['role'] == 'P'])
        u = np.array([r['support_logit'] for r in candidates if r['role'] == 'U'])
        if not len(p) or not len(u):
            continue
        _, _, auc, ap = _threshold_curve(p, u)
        ranking = evaluate_disease_ranking(binding=binding, disease_id=disease,
            candidate_drug_ids=[r['drug_id'] for r in candidates], scores=[r['support_logit'] for r in candidates],
            relevant_drug_ids=[r['drug_id'] for r in candidates if r['role'] == 'P'])
        item = dict(disease_id=disease, auc=float(auc), ap=float(ap), P=len(p), U=len(u))
        for k in ranking.metrics:
            if k.requested_k in (5, 10):
                item.update({f'P@{k.requested_k}': k.known_positive_precision,
                    f'R@{k.requested_k}': k.known_positive_recall, f'Hits@{k.requested_k}': k.hits_at_k,
                    f'NDCG@{k.requested_k}': k.ndcg})
        local.append(item)
    if not local:
        raise ValueError('No evaluable inner disease; no automatic stopping or selection')
    keys = ['P@5', 'R@5', 'Hits@5', 'NDCG@5', 'P@10', 'R@10', 'Hits@10', 'NDCG@10']
    return dict(scope='DEVELOPMENT_INNER_VALIDATION', macro_auc=statistics.fmean(r['auc'] for r in local),
        macro_ap=statistics.fmean(r['ap'] for r in local), evaluable_diseases=len(local),
        skipped_diseases=len(groups) - len(local), pairs=len(seen),
        ranking={k: statistics.fmean(r[k] for r in local) for k in keys}, disease_metrics=local,
        semantics='P_VS_U_NOT_CONFIRMED_NEGATIVE')


def duration_plateau(history):
    """Two consecutive inner-only cycle intervals, frozen before real scores."""
    if len(history) < 3:
        return False
    limits = dict(validation_objective=.0001, validation_AUC=.005, validation_AP=.005,
                  matched_G_AUC=.005, matched_G_AP=.005, R10=.02, NDCG10=.02)
    for a, b in zip(history[-3:-1], history[-2:]):
        for key, limit in limits.items():
            if a.get(key) is None or b.get(key) is None:
                return False
            if not all(math.isfinite(float(r[key])) for r in (a, b)):
                raise ValueError('Nonfinite duration criterion')
            if abs(float(b[key]) - float(a[key])) >= limit:
                return False
    return True


class CycleStore:
    """Checkpoint first, immutable completion receipt last; partial != complete."""
    def __init__(self, root, bindings):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bindings = bindings
        p = self.root / 'bindings.json'
        identity = bound(dict(schema='PILOT_CELL_BINDINGS_V1', bindings=bindings))
        if p.exists() and verify(read(p)) != identity:
            raise ValueError('Recovery binding mismatch; FAIL CLOSED')
        save(p, identity)

    def completed_cycles(self):
        values = []
        for path in sorted(self.root.glob('cycle_*.json')):
            r = verify(read(path))
            if r['bindings'] != self.bindings or r['status'] != 'CYCLE_COMPLETE':
                raise ValueError('Recovery receipt binding mismatch')
            checkpoint = self.root / r['checkpoint']
            if not checkpoint.exists() or sha(checkpoint) != r['checkpoint_sha256']:
                raise ValueError('Recovery checkpoint mismatch')
            for name, expected in r.get('artifacts', {}).items():
                if sha(self.root / name) != expected:
                    raise ValueError('Recovery metric artifact mismatch')
            values.append(r['cycle'])
        if values != list(range(1, len(values) + 1)):
            raise ValueError('Complete cycles must be consecutive')
        return values

    def load_cycle(self, cycle):
        if cycle not in self.completed_cycles():
            raise ValueError('Cannot resume an incomplete cycle')
        r = read(self.root / f'cycle_{cycle:02}.json')
        return torch.load(self.root / r['checkpoint'], map_location='cpu', weights_only=True)

    def commit_cycle(self, cycle, state, metrics, artifacts=None):
        complete = self.completed_cycles()
        if cycle in complete:
            raise ValueError('Cycle already complete; never retrain/overwrite')
        if cycle != len(complete) + 1:
            raise ValueError('Cycle commit must be consecutive')
        checkpoint = self.root / f'cycle_{cycle:02}.pt'
        if checkpoint.exists():
            raise ValueError('Unreceipted checkpoint exists; partial recovery requires review')
        partial = checkpoint.with_suffix('.pt.partial')
        with partial.open('wb') as h:
            torch.save(state, h)
            h.flush()
            os.fsync(h.fileno())
        os.replace(partial, checkpoint)
        receipt = bound(dict(status='CYCLE_COMPLETE', cycle=cycle, bindings=self.bindings,
            checkpoint=checkpoint.name, checkpoint_sha256=sha(checkpoint), metrics=metrics,
            artifacts=artifacts or {}, outer_test_accessed=False, final_test_accessed=False))
        save(self.root / f'cycle_{cycle:02}.json', receipt)
        return receipt

    def freeze_model(self, required_cycles):
        if self.completed_cycles() != list(range(1, required_cycles + 1)):
            raise ValueError('Partial model cell cannot become COMPLETE')
        receipts = [verify(read(self.root / f'cycle_{c:02}.json')) for c in range(1, required_cycles + 1)]
        selected = min(receipts, key=lambda r: (r['metrics']['validation_objective'], r['cycle']))
        result = bound(dict(status='MODEL_PILOT_COMPLETE', bindings=self.bindings,
            cycles=required_cycles, best_cycle=selected['cycle'],
            selected_checkpoint=selected['checkpoint'], selected_checkpoint_sha256=selected['checkpoint_sha256'],
            selected_cycle_receipt_digest=selected['digest'], metrics=selected['metrics'],
            total_elapsed_seconds=receipts[-1]['metrics'].get('elapsed_seconds'),
            peak_gpu_memory_bytes=max(r['metrics'].get('peak_gpu_memory_bytes', 0) for r in receipts),
            scope='DEVELOPMENT_INNER_VALIDATION', outer_test_accessed=False, final_test_accessed=False))
        save(self.root / 'MODEL_COMPLETE.json', result)
        return result


def verify_oof_score_receipt(receipt, *, bindings, expected_rows, checkpoint_sha256):
    """Receipt gate for a future authorized outer producer; no score calculation."""
    r = verify(receipt)
    if r.get('status') != 'OOF_SCORE_COMPLETE' or r.get('bindings') != bindings:
        raise ValueError('OOF receipt binding mismatch')
    if r.get('checkpoint_sha256') != checkpoint_sha256 or not r.get('training_complete') or not r.get('checkpoint_selection_complete'):
        raise ValueError('OOF score before frozen checkpoint is forbidden')
    expected = sorted((v['drug_id'], v['disease_id'], v['role'], v['fold']) for v in expected_rows)
    if r.get('expected_membership_digest') != digest(expected) or r.get('pair_count') != len(expected):
        raise ValueError('OOF expected panel mismatch')
    return r
