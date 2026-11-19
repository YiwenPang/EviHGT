"""Native TxGNN CPU two-stage cursor recovery, isolated from frozen old adapters.

The DGL 1.1.2 CPU iterator is synchronous. Its exact permutation and cursor are
saved. Restoring constructs an iterator then restores the committed RNG state,
so PyTorch's iterator base-seed consumption is not repeated scientifically.
No DGL reseeding, replacement negative sampler, or model changes are introduced.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import random
import re
import time
import uuid

import numpy as np
import torch

from .identity import ROOT, G1_SHA256, file_sha, digest
from . import production_state as state_api
from .txgnn_adapter import load_native, verify_native_sources

training_state_digest = state_api.training_state_digest
TEST_ROOT = ROOT / 'outputs/phase9/phase92c/test_only'
SCHEMA = 'PHASE92C_TXGNN_COMPLETE_STATE_V1'
TEST_UPDATE_AUDIT = []
RECIPE = dict(dimension=100, pretrain_epochs=1, pretrain_lr=.001,
    pretrain_batch_size=65536, finetune_lr=.003, finetune_epochs=500,
    evaluate_every=25, optimizer='AdamW', scheduler='ReduceLROnPlateau_min_factor_0.8',
    pretrain_commit_every=32, finetune_commit_every=25,
    native_cpu_threads=8, negative_sampler='native_fix_dst_plus_common_DDA_U')


def state_digest(state):
    return training_state_digest({k: v for k, v in state.items() if k != 'state_digest'})


def graph_digest(graph, *, features=False):
    entries = {'nodes': {n: graph.num_nodes(n) for n in graph.ntypes}, 'edges': {}}
    for etype in graph.canonical_etypes:
        entries['edges']['|'.join(etype)] = training_state_digest(graph.edges(etype=etype))
    if features:
        entries['features'] = {n: training_state_digest(graph.nodes[n].data['inp']) for n in graph.ntypes}
    return digest(entries)


def batch_digest(batch):
    import dgl
    _, positive, negative, blocks = batch
    result = {'positive': graph_digest(positive), 'negative': graph_digest(negative), 'blocks': []}
    for block in blocks:
        result['blocks'].append({
            'edges': {str(e): training_state_digest(block.edges[e].data[dgl.EID]) for e in block.canonical_etypes},
            'src': {n: training_state_digest(block.srcnodes[n].data[dgl.NID]) for n in block.srctypes},
            'dst': {n: training_state_digest(block.dstnodes[n].data[dgl.NID]) for n in block.dsttypes}})
    return digest(result)


def _restore_rng(value):
    numpy = state_api._validate_rng(value)
    random.setstate(value['python'])
    np.random.set_state(numpy)
    torch.set_rng_state(value['cpu'])
    if value['cuda']:
        torch.cuda.set_rng_state_all(value['cuda'])


def _scheduler_pack(value):
    # ReduceLROnPlateau deliberately stores +inf for mode_worse (and initial
    # best). Encode those native sentinels; all actual model/loss values remain finite.
    if isinstance(value, dict):
        return {k: ('NATIVE_POSITIVE_INFINITY' if k in {'mode_worse', 'best'} and
            isinstance(v, float) and v == math.inf else _scheduler_pack(v)) for k, v in value.items()}
    return value


def _scheduler_unpack(value):
    if isinstance(value, dict):
        return {k: (math.inf if k in {'mode_worse', 'best'} and v == 'NATIVE_POSITIVE_INFINITY'
            else _scheduler_unpack(v)) for k, v in value.items()}
    return value


def _basic(state):
    required = {'schema', 'identity', 'recipe', 'source', 'graph_binding', 'eid_binding',
        'scope', 'cursor', 'permutation', 'model', 'auxiliary_gates', 'optimizer',
        'optimizer_recipe', 'scheduler', 'scheduler_recipe', 'rng', 'module_modes',
        'stream', 'transition', 'selection', 'external_control', 'state_digest'}
    if not isinstance(state, dict) or set(state) != required or state['schema'] != SCHEMA:
        raise ValueError('Incomplete TxGNN two-stage state')
    if state['state_digest'] != state_digest(state):
        raise ValueError('TxGNN full state integrity differs')
    if state['identity'].get('method') != 'M4' or state['identity'].get('g1_sha256') != G1_SHA256:
        raise ValueError('TxGNN method or original G1 identity differs')
    state_api._finite(state)
    state_api._validate_rng(state['rng'])


class TxTwoStageTrainer:
    def __init__(self, data, stream_factory, identity, *, mode='TEST_ONLY', test_eids=None,
                 batch_size=65536, test_finetune_epochs=3, execution_permit=None):
        self.permit = None
        if mode == 'FORMAL':
            from .execution import verify_permit
            self.permit = verify_permit(execution_permit, 'M4', identity.get('repeat'), identity.get('fold'))
            from .m4_amendment import verify_m4_amendment
            amendment = verify_m4_amendment(self.permit)
            if test_eids is not None or batch_size != 65536 or test_finetune_epochs != 3 or torch.get_num_threads() != 8:
                raise ValueError('FORMAL M4 must use full allowed EIDs, native batch 65536 and eight CPU threads')
            if amendment['sha256'] != self.permit.get('amendment_sha256') or identity.get('amendment_sha256') != amendment['sha256'] or identity.get('execution_manifest_sha256') != self.permit['_manifest_sha256']:
                raise PermissionError('Frozen M4 amendment and exact execution manifest required')
        elif mode != 'TEST_ONLY':
            raise PermissionError('Unknown M4 execution mode')
        if identity.get('method') != 'M4' or identity.get('g1_sha256') != G1_SHA256:
            raise ValueError('Explicit M4 and frozen parent G1 required')
        needed = {'repeat', 'fold', 'seed', 'cell_digest', 'corpus_digest', 'source_digest', 'environment_digest'}
        if not needed <= identity.keys() or not 0 <= identity['repeat'] < 3 or not 0 <= identity['fold'] < 5:
            raise ValueError('Original cell/source/environment bindings required')
        if type(batch_size) is not int or not 1 <= batch_size <= 65536 or not 1 <= test_finetune_epochs <= 3:
            raise ValueError('Bounded TEST_ONLY budgets required')
        import dgl
        if dgl.__version__ != '1.1.2' or torch.cuda.is_initialized():
            raise ValueError('Only certified native DGL 1.1.2 CPU execution is qualified')
        self.data, self.stream_factory = data, stream_factory
        self.identity = dict(identity, mode=mode)
        self.native = load_native()
        self.adapter = self.native.NativeCompetitionModel(data.encoder_graph, data.reference_graph, dimension=100)
        self.source = {'native': verify_native_sources(), 'recovery': file_sha(__file__),
            'state_helpers': file_sha(state_api.__file__),
            'dgl_loader': file_sha(Path(dgl.__file__).parent/'dataloading/dataloader.py'),
            'torch_loader': file_sha(Path(torch.__file__).parent/'utils/data/dataloader.py')}
        self.graph_binding = {name: graph_digest(getattr(data, name), features=name == 'graph')
            for name in ('graph', 'reference_graph', 'encoder_graph', 'positive_graph')}
        self.graph_binding['prototype'] = training_state_digest(self.adapter.model.pred.sim_all_etypes)
        allowed = self.native.pretraining_edge_ids(data.graph, data.support, data.contra)
        selected = allowed if test_eids is None else test_eids
        if not selected or any(e not in allowed or v.ndim != 1 or v.dtype != torch.int64 or
                len(torch.unique(v)) != len(v) or not bool(torch.isin(v, allowed[e]).all()) for e, v in selected.items()):
            raise ValueError('Invalid or unauthorized pretrain EID domain')
        selected = {e: v.clone() for e, v in selected.items() if len(v)}
        self.eid_binding = {'allowed': training_state_digest(allowed), 'selected': training_state_digest(selected)}
        self.scope = {'mode': mode, 'batch_size': batch_size, 'test_eid_subset': test_eids is not None,
            'planned_pretrain_batches': math.ceil(sum(map(len, selected.values()))/batch_size),
            'test_finetune_epochs': test_finetune_epochs if mode == 'TEST_ONLY' else None,
            'formal_budget': RECIPE}
        # Even the full allowed domain can be diagnosed, but never completed in TEST_ONLY.
        self.max_pretrain_steps = min(3, self.scope['planned_pretrain_batches']) if mode == 'TEST_ONLY' else self.scope['planned_pretrain_batches']
        self.max_finetune_epochs = test_finetune_epochs if mode == 'TEST_ONLY' else 500
        self.native._seed(identity['seed']+11)
        self.stream = stream_factory(102)
        sampler = dgl.dataloading.as_edge_prediction_sampler(
            dgl.dataloading.MultiLayerFullNeighborSampler(2),
            exclude=self.native.TargetPairExclusion(data.graph),
            negative_sampler=self.native.GatedNegativeSampler(data.graph, self.stream))
        self.loader = dgl.dataloading.DataLoader(data.graph, selected, sampler,
            batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
        if self.loader.use_prefetch_thread or self.loader.num_workers:
            raise ValueError('Native CPU synchronous loader contract violated')
        self.optimizer = torch.optim.AdamW(self.adapter.model.parameters(), lr=.001)
        self.scheduler = None
        self.iterator = iter(self.loader)
        self.permutation = self.loader.dataset._indices.clone()
        self.native_id_digest = training_state_digest(self.loader.dataset._id_tensor)
        self.eid_binding['native_id_tensor'] = self.native_id_digest
        self.cursor = {'stage': 'PRETRAIN', 'pretrain_epoch': 1, 'pretrain_batch': 0,
            'pretrain_updates': 0, 'finetune_epoch': 0, 'finetune_updates': 0,
            'actual_optimizer_updates': 0, 'consumed_edges': 0, 'negative_batches': 0}
        self.transition_state = None
        self.selection = {'completed_epochs': [], 'best': None}
        self.external_control = {}
        self.last_step = None

    @property
    def stage(self):
        return self.cursor['stage']

    @property
    def pretrain_batch(self):
        return self.cursor['pretrain_batch']

    @property
    def planned_pretrain_batches(self):
        return self.scope['planned_pretrain_batches']

    @property
    def finetune_epoch(self):
        return self.cursor['finetune_epoch']

    @property
    def decoder_resets(self):
        return 0 if self.transition_state is None else self.transition_state['decoder_reset_count']

    @property
    def model(self):
        return self.adapter.model

    def _position_iterator(self, permutation, batch_cursor):
        self.loader.shuffle = False
        self.loader.dataset._indices = permutation.clone()
        self.iterator = iter(self.loader)
        internal = self.iterator.dataloader_it._dataset_fetcher.dataset_iter
        if type(internal).__name__ != '_TensorizedDatasetIter' or internal.index != 0:
            raise ValueError('Unrecognized native DGL iterator internals')
        internal.index = batch_cursor * self.scope['batch_size']
        self.permutation = permutation.clone()

    def next_batch_diagnostic(self):
        """Nontraining test instrumentation; caller must restore before optimizing."""
        batch = next(self.iterator)
        return {'batch_digest': batch_digest(batch), 'rng': training_state_digest(state_api._rng()),
            'stream': self.stream.state()}

    def _step(self, loss, stage):
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite native loss')
        loss.backward()
        gradients = self.native._check_gradients(self.adapter.model)
        self.optimizer.step()
        self.cursor['actual_optimizer_updates'] += 1
        TEST_UPDATE_AUDIT.append({'identity': self.identity, 'stage': stage,
            'optimizer_update': self.cursor['actual_optimizer_updates']})
        return float(loss.detach()), gradients

    def pretrain_step(self):
        if self.stage != 'PRETRAIN' or self.cursor['pretrain_batch'] >= self.max_pretrain_steps:
            raise ValueError('Pretrain stage/bounded TEST_ONLY budget exhausted')
        self.adapter.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = next(self.iterator)
        signature = batch_digest(batch)
        tick = time.perf_counter()
        loss = self.native._pretrain_loss(self.adapter, batch, self.data.reference_graph)
        value, gradients = self._step(loss, 'PRETRAIN')
        self.cursor['pretrain_batch'] += 1
        self.cursor['pretrain_updates'] += 1
        self.cursor['negative_batches'] += 1
        self.cursor['consumed_edges'] += batch[1].num_edges()
        self.last_step = {'stage': 'PRETRAIN', 'loss': value, 'batch_digest': signature,
            'positive_edges': batch[1].num_edges(), 'negative_edges': batch[2].num_edges(),
            'seconds': time.perf_counter()-tick, 'gradients': gradients, 'cursor': dict(self.cursor)}
        return self.last_step

    def transition(self, pretrain_commit):
        if self.stage != 'PRETRAIN' or (self.identity['mode'] == 'TEST_ONLY' and not self.scope['test_eid_subset']) or self.cursor['pretrain_batch'] != self.scope['planned_pretrain_batches']:
            raise ValueError('Only an actually completed declared TEST_ONLY pretrain subset may transition')
        receipt = verify_tx_commit(pretrain_commit.get('checkpoint_commit_path', ''))
        if receipt['identity'] != self.identity or receipt['state_digest'] != self.snapshot()['state_digest']:
            raise ValueError('Transition must originate from this exact completed pretrain commit')
        prior = training_state_digest(self.adapter.model.state_dict())
        self.native._seed(self.identity['seed']+1001)
        torch.nn.init.xavier_uniform_(self.adapter.model.w_rels)
        self.optimizer = torch.optim.AdamW(self.adapter.model.parameters(), lr=.003)
        for group in self.optimizer.param_groups:
            group['initial_lr'] = .003
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=.8)
        self.stream = self.stream_factory(103)
        self.cursor['stage'] = 'FINETUNE'
        self.transition_state = {'pretrain_commit_path': receipt['checkpoint_commit_path'],
            'pretrain_commit_digest': receipt['commit_digest'],
            'pretrain_state_digest': receipt['state_digest'], 'decoder_reset_count': 1,
            'decoder_seed': self.identity['seed']+1001, 'pre_reset_model_digest': prior,
            'post_reset_model_digest': training_state_digest(self.adapter.model.state_dict())}

    def finetune_step(self):
        if self.stage != 'FINETUNE' or self.cursor['finetune_epoch'] >= self.max_finetune_epochs:
            raise ValueError('Finetune stage/bounded TEST_ONLY budget exhausted')
        self.adapter.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        contrast = self.native._contrast_graph(self.data, self.stream)
        signature = graph_digest(contrast)
        tick = time.perf_counter()
        loss, _ = self.adapter.loss(self.data.positive_graph, contrast)
        value, gradients = self._step(loss, 'FINETUNE')
        self.scheduler.step(value)
        self.cursor['finetune_epoch'] += 1
        self.cursor['finetune_updates'] += 1
        self.cursor['negative_batches'] += 1
        self.last_step = {'stage': 'FINETUNE', 'loss': value, 'contrast_digest': signature,
            'seconds': time.perf_counter()-tick, 'gradients': gradients,
            'cursor': dict(self.cursor), 'evaluation_calls': 0}
        return self.last_step

    def snapshot(self):
        gates = {key: value.state_dict() for key, value in self.adapter.model.pred.W_gate.items()}
        state = state_api._snapshot({'schema': SCHEMA, 'identity': self.identity, 'recipe': RECIPE,
            'source': self.source, 'graph_binding': self.graph_binding, 'eid_binding': self.eid_binding,
            'scope': self.scope, 'cursor': self.cursor, 'permutation': self.permutation,
            'model': self.adapter.model.state_dict(), 'auxiliary_gates': gates,
            'optimizer': self.optimizer.state_dict(),
            'optimizer_recipe': state_api._optimizer_recipe(self.adapter.model, self.optimizer, self.scheduler),
            'scheduler': _scheduler_pack(self.scheduler.state_dict()) if self.scheduler else None,
            'scheduler_recipe': _scheduler_pack(state_api._scheduler_recipe(self.scheduler)), 'rng': state_api._rng(),
            'module_modes': {name: module.training for name, module in self.adapter.model.named_modules()},
            'stream': self.stream.state(), 'transition': self.transition_state, 'selection': self.selection,
            'external_control': self.external_control})
        state_api._finite(state)
        state['state_digest'] = state_digest(state)
        return state

    def restore(self, state):
        _basic(state)
        for key in ('identity', 'recipe', 'source', 'graph_binding', 'eid_binding', 'scope'):
            expected = RECIPE if key == 'recipe' else getattr(self, key)
            if state[key] != expected:
                raise ValueError('Cross-arm/fold/source/data/config restore forbidden: '+key)
        cursor = state['cursor']
        if set(cursor) != set(self.cursor) or cursor['stage'] not in {'PRETRAIN', 'FINETUNE'}:
            raise ValueError('Unknown stage/cursor schema')
        if any(type(v) is not int or v < 0 for k, v in cursor.items() if k != 'stage'):
            raise ValueError('Invalid stage counters')
        if cursor['actual_optimizer_updates'] != cursor['pretrain_updates']+cursor['finetune_updates'] or cursor['pretrain_updates'] != cursor['pretrain_batch'] or cursor['finetune_updates'] != cursor['finetune_epoch']:
            raise ValueError('Optimizer/stage cursor mismatch')
        if cursor['pretrain_batch'] > self.max_pretrain_steps or cursor['finetune_epoch'] > self.max_finetune_epochs:
            raise ValueError('Checkpoint exceeds authorized TEST_ONLY updates')
        if cursor['pretrain_epoch'] != 1 or cursor['negative_batches'] != cursor['actual_optimizer_updates'] or cursor['consumed_edges'] != min(cursor['pretrain_batch']*self.scope['batch_size'], len(self.permutation)):
            raise ValueError('Pretrain coverage/negative cursor mismatch')
        permutation = state['permutation']
        if permutation.dtype != torch.int64 or permutation.shape != self.permutation.shape or not torch.equal(torch.sort(permutation).values, torch.arange(len(permutation))):
            raise ValueError('Missing/duplicate/out-of-domain pretrain permutation')
        if cursor['stage'] == 'PRETRAIN' and (state['transition'] is not None or cursor['finetune_epoch']):
            raise ValueError('Illegal pretrain stage transition')
        if cursor['stage'] == 'FINETUNE' and (cursor['pretrain_batch'] != self.scope['planned_pretrain_batches'] or not state['transition'] or state['transition']['decoder_reset_count'] != 1):
            raise ValueError('Finetune lacks completed pretrain/reset provenance')
        current = self.adapter.model.state_dict()
        if current.keys() != state['model'].keys() or any(current[k].shape != v.shape or current[k].dtype != v.dtype for k, v in state['model'].items()):
            raise ValueError('Native model schema differs')
        optimizer = torch.optim.AdamW(self.adapter.model.parameters(), lr=.001 if cursor['stage'] == 'PRETRAIN' else .003)
        scheduler = None
        if cursor['stage'] == 'FINETUNE':
            for group in optimizer.param_groups:
                group['initial_lr'] = .003
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.8)
        checked = dict(state, scheduler=_scheduler_unpack(state['scheduler']),
            cursor={'actual_optimizer_updates': cursor['pretrain_updates'] if cursor['stage'] == 'PRETRAIN' else cursor['finetune_updates']})
        state_api._validate_optimizer(checked, self.adapter.model, optimizer, scheduler)
        if _scheduler_pack(state_api._scheduler_recipe(scheduler)) != state['scheduler_recipe'] or (scheduler is None) != (state['scheduler'] is None):
            raise ValueError('Scheduler stage/config differs')
        if scheduler is not None:
            saved_scheduler = _scheduler_unpack(state['scheduler'])
            if saved_scheduler.get('mode_worse') != math.inf or saved_scheduler.get('last_epoch') != cursor['finetune_epoch'] or (saved_scheduler.get('best') == math.inf) != (cursor['finetune_epoch'] == 0):
                raise ValueError('Native scheduler sentinel/epoch mismatch')
        if state['module_modes'].keys() != dict(self.adapter.model.named_modules()).keys():
            raise ValueError('Native module modes differ')
        if any(type(v) is not bool for v in state['module_modes'].values()) or set(state['auxiliary_gates']) != set(self.adapter.model.pred.W_gate):
            raise ValueError('Native gate/mode schema differs')
        for key, module in self.adapter.model.pred.W_gate.items():
            now = module.state_dict()
            saved = state['auxiliary_gates'][key]
            if now.keys() != saved.keys() or any(now[k].shape != v.shape or now[k].dtype != v.dtype for k, v in saved.items()):
                raise ValueError('Unregistered native gate tensor schema differs')
        stream = self.stream_factory(102 if cursor['stage'] == 'PRETRAIN' else 103)
        stream.restore(state['stream'])
        if cursor['stage'] == 'FINETUNE':
            receipt = verify_tx_commit(state['transition']['pretrain_commit_path'])
            if receipt['commit_digest'] != state['transition']['pretrain_commit_digest'] or receipt['state_digest'] != state['transition']['pretrain_state_digest'] or receipt['identity'] != self.identity:
                raise ValueError('Pretrain source commit identity differs')
        # All identities/state schemas checked before mutating receiving parameters.
        self.adapter.model.load_state_dict(state['model'])
        for key, value in state['auxiliary_gates'].items():
            self.adapter.model.pred.W_gate[key].load_state_dict(value)
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        if scheduler:
            scheduler.load_state_dict(_scheduler_unpack(copy.deepcopy(state['scheduler'])))
        self.optimizer, self.scheduler, self.stream = optimizer, scheduler, stream
        self.loader.graph_sampler.negative_sampler.sample_u = stream
        self._position_iterator(permutation, cursor['pretrain_batch'])
        self.cursor = copy.deepcopy(cursor)
        self.transition_state, self.selection = copy.deepcopy(state['transition']), copy.deepcopy(state['selection'])
        self.external_control = copy.deepcopy(state['external_control'])
        for name, module in self.adapter.model.named_modules():
            module.training = state['module_modes'][name]
        _restore_rng(state['rng'])
        if self.snapshot()['state_digest'] != state['state_digest']:
            raise ValueError('Exact complete-state restoration did not reproduce the committed state')

    def commit(self, store, name):
        return store.commit(name, self.snapshot())


def assert_replay_equal(left, right, *, atol=1e-7, rtol=1e-6):
    """Strict CPU replay acceptance, declared before real tests."""
    if left['cursor'] != right['cursor'] or left['stream'] != right['stream'] or training_state_digest(left['rng']) != training_state_digest(right['rng']):
        raise AssertionError('CPU replay cursor/U/RNG differs')
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                compare(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b
    for key in ('model', 'optimizer', 'scheduler', 'module_modes', 'auxiliary_gates'):
        compare(left[key], right[key])


def _allowed_path(path, mode='TEST_ONLY', permit=None):
    path = Path(path).resolve()
    allowed = TEST_ROOT.resolve()
    if mode == 'FORMAL':
        if not permit or not permit.get('amendment_sha256'):
            raise PermissionError('Independent M4 formal permit required')
        allowed = Path(permit['output_root']).resolve()
        if not allowed.is_relative_to(ROOT/'outputs/phase9/formal'):
            raise PermissionError('Invalid M4 formal output root')
    elif mode != 'TEST_ONLY':
        raise PermissionError('Unknown M4 checkpoint mode')
    if not path.is_relative_to(allowed) or path == allowed:
        raise PermissionError('M4 checkpoint outside authorized namespace')
    return path


def verify_tx_commit(path):
    marker = Path(path).resolve()
    formal = marker.is_relative_to(ROOT/'outputs/phase9/formal')
    if formal:
        from .execution import APPROVED_EXECUTIONS
        if not APPROVED_EXECUTIONS:
            raise PermissionError('G3_NOT_AUTHORIZED')
    else:
        _allowed_path(marker)
    if not marker.name.endswith('.COMMIT.json'):
        raise ValueError('Complete commit marker required')
    receipt = json.loads(marker.read_text(encoding='utf-8'))
    expected_mode = 'FORMAL' if formal else 'TEST_ONLY'
    if receipt.get('status') != 'COMMITTED' or receipt.get('schema') != 'PHASE92C_M4_ATOMIC_COMMIT_V1' or receipt.get('mode') != expected_mode:
        raise ValueError('Not a complete M4 test commit')
    if receipt.get('commit_digest') != digest({k: v for k, v in receipt.items() if k != 'commit_digest'}):
        raise ValueError('Commit marker integrity differs')
    permit = None
    if formal:
        from .execution import validate_execution_manifest
        identity = receipt['identity']
        permit = validate_execution_manifest(receipt['execution_manifest_path'], 'M4', identity['repeat'], identity['fold'])
        from .m4_amendment import verify_m4_amendment
        amendment = verify_m4_amendment(permit)
        if permit['_manifest_sha256'] != receipt['execution_manifest_sha256'] or amendment['sha256'] != identity.get('amendment_sha256'):
            raise PermissionError('M4 checkpoint execution/amendment differs')
    _allowed_path(marker, expected_mode, permit)
    statepath = _allowed_path(receipt['checkpoint_path'], expected_mode, permit)
    if marker.parent != statepath.parent or str(marker) != receipt['checkpoint_commit_path']:
        raise ValueError('Commit path identity differs')
    if statepath.stat().st_size != receipt['checkpoint_size'] or file_sha(statepath) != receipt['checkpoint_sha256']:
        raise ValueError('Checkpoint content corruption')
    state = torch.load(statepath, map_location='cpu', weights_only=True)
    _basic(state)
    if state['state_digest'] != receipt['state_digest'] or state['identity'] != receipt['identity'] or state['cursor'] != receipt['cursor']:
        raise ValueError('Commit and full state differ')
    return receipt


class TxAtomicStore:
    def __init__(self, root, mode='TEST_ONLY', execution_permit=None):
        self.mode, self.permit = mode, None
        if mode == 'FORMAL':
            from .execution import verify_permit
            # The manifest authorizes the entire fixed queue; actual cell ownership
            # is checked against the state at commit time, never aliased to M0.
            self.permit = verify_permit(execution_permit, 'M4', 0, 0)
            from .m4_amendment import verify_m4_amendment
            verify_m4_amendment(self.permit)
        self.root = _allowed_path(root, mode, self.permit)
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, name, state):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', name):
            raise ValueError('Safe checkpoint basename required')
        _basic(state)
        if state['identity']['mode'] != self.mode:
            raise PermissionError('M4 TEST_ONLY and FORMAL states cannot mix')
        if self.mode == 'FORMAL':
            from .execution import verify_permit
            self.permit = verify_permit(self.permit, 'M4', state['identity']['repeat'], state['identity']['fold'])
        marker, final = self.root/(name+'.COMMIT.json'), self.root/(name+'.state.pt')
        if marker.exists():
            prior = verify_tx_commit(marker)
            if prior['state_digest'] == state['state_digest']:
                return prior
            raise FileExistsError('Committed checkpoint cannot be overwritten')
        if final.exists():
            raise FileExistsError('Uncommitted checkpoint payload exists')
        token = uuid.uuid4().hex
        lock = self.root/(name+'.publish.lock')
        with lock.open('x', encoding='utf-8') as handle:
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        partial = self.root/(name+'.'+token+'.state.pt.partial')
        temporary_marker = self.root/(name+'.'+token+'.COMMIT.json.partial')
        try:
            with partial.open('xb') as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            checksum, size = file_sha(partial), partial.stat().st_size
            os.replace(partial, final)
            receipt = {'schema': 'PHASE92C_M4_ATOMIC_COMMIT_V1', 'status': 'COMMITTED',
                'mode': self.mode, 'checkpoint_path': str(final), 'checkpoint_commit_path': str(marker),
                'checkpoint_sha256': checksum, 'checkpoint_size': size, 'state_digest': state['state_digest'],
                'identity': state['identity'], 'cursor': state['cursor']}
            if self.permit:
                receipt.update(execution_manifest_path=self.permit['_manifest_path'],
                    execution_manifest_sha256=self.permit['_manifest_sha256'])
            receipt['commit_digest'] = digest(receipt)
            with temporary_marker.open('x', encoding='utf-8') as handle:
                json.dump(receipt, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_marker, marker)
            return verify_tx_commit(marker)
        finally:
            if lock.exists() and lock.read_text() == token:
                lock.unlink()

    def load(self, path):
        receipt = verify_tx_commit(path)
        if Path(receipt['checkpoint_path']).parent != self.root:
            raise ValueError('Wrong checkpoint store')
        return torch.load(receipt['checkpoint_path'], map_location='cpu', weights_only=True)
