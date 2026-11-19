"""Authorized RankingComparison native receivers; no old Pilot grants or global patches.

Numerical HGTDR and Tx kernels stay in their hash-verified historical modules.
These constructors and data adapters own the new RankingComparison gate. Only the root
runtime may construct or execute receivers while its shared ledger is active.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np
import polars as pl
import torch
from evi_hgt.ranking_protocol import hgtdr_adapter as hg
from evi_hgt.ranking_protocol import txgnn_adapter as txdata
from evi_hgt.ranking_protocol import txgnn_data as txbinding
from evi_hgt.ranking_protocol import production_state as state_api
from evi_hgt.ranking_protocol.identity import G1_SHA256, digest, file_sha
from .common import ROOT,sha
from .native_checkpoint import NativeCheckpointStore,source_bindings,load_private_native_module

class FormalNativeCapability:
    def __init__(self,capability,identity,output_root):
        from .authorization import ExecutionCapability
        if not isinstance(capability,ExecutionCapability):raise PermissionError('RankingComparison G3 capability required')
        capability.permits(identity,output_root)
        self.capability,self.identity,self.output_root=capability,copy.deepcopy(identity),Path(output_root)
        self.mode='RANKING_COMPARISON_FORMAL'
    def check_native(self,method,cell):
        if (method,cell)!=(self.identity['method'],self.identity['cell']):raise PermissionError('Native cell/method differs')
        self.capability.permits(self.identity,self.output_root)

def _authorize(authorization,method,cell):
    from .runtime_budget import EngineeringCapability
    if not isinstance(authorization,(EngineeringCapability,FormalNativeCapability)):
        raise PermissionError('Issued RankingComparison capability required before native data/model construction')
    authorization.check_native(method,cell)

def _cache_path(path):
    p=Path(path).resolve();base=(ROOT/'data/cache/ranking_comparison/native').resolve()
    if p==base or not p.is_relative_to(base):raise PermissionError('New derived native cache namespace required')
    return p


def assert_native_global_fact_mask(corpus):
    """The native export consumes the masked fact tables, never a parent view."""
    forbidden=pl.read_parquet(Path(corpus['corpus_root'])/'forbidden_fact_ids.parquet')['canonical_fact_id']
    for row in corpus['fact_files']:
        hit=pl.scan_parquet(row['path']).select('canonical_fact_id').filter(pl.col('canonical_fact_id').is_in(forbidden.implode())).limit(1).collect()
        if hit.height:raise ValueError('Globally forbidden fact reached native export: '+row['path'])

validate_cell_corpus=hg.validate_cell_corpus
verify_native_identity=hg.verify_native_identity
_read_features=hg._read_features
fixed_config=hg.fixed_config
native_module=hg.native_module

class RankingComparisonHGTDRFoldData(hg.HGTDRFoldData):
    def batch(self,pairs,config,seed,device):
        batch=super().batch(pairs,config,seed,device)
        # Observation only: preserve native order, seed and tensors unchanged.
        self.last_batch=dict(pair_order_sha256=state_api.training_state_digest(torch.as_tensor(pairs.copy())),
            input_sha256=state_api.training_state_digest(batch),
            entries=len(pairs),nodes=batch['node_count'],relation_edges=batch['edge_count'])
        return batch

    @classmethod
    def from_cell(cls, data, corpus, output, *, authorization):
        from evi_hgt.competition.data import export_graph_arrays, normalize_nodes, UnlabeledPolicy
        _authorize(authorization,'HGTDR_ADAPTED',data.cell_id)
        validate_cell_corpus(data, corpus)
        assert_native_global_fact_mask(corpus)
        verify_native_identity()
        output=_cache_path(output)
        from .runtime_budget import disk_guard
        disk_guard(output,sum(int(r['rows']) for r in corpus['fact_files'])*64+10_000_000)
        output.mkdir(parents=True, exist_ok=True)
        obj = cls()
        obj.training_seed, obj.bindings = data.training_seed, copy.deepcopy(data.bindings)
        obj.cell_source = data
        nodes = normalize_nodes(pl.read_parquet(ROOT / "data/processed/canonical_graph/v1/nodes/node_index.parquet"))
        obj.nodes = nodes
        obj.features, obj.available, obj.counts, obj.feature_binding = _read_features(nodes)
        obj.lookup = {kind: dict(nodes.filter(pl.col("human_node_type") == kind).select("node_id", "type_local_id").iter_rows())
                      for kind in ("drug", "disease")}
        obj.support_ids = tuple(sorted(data.support))
        if set(obj.support_ids) & set(data.blocked):
            raise ValueError("Protected support pair reached native training")
        for pair in obj.support_ids:
            data.pairs.assert_optimizer_pair(pair, "P")
        obj.supports = np.asarray([[obj.lookup["drug"][p[0]], obj.lookup["disease"][p[1]]] for p in obj.support_ids], dtype=np.int64)
        universe, gate = data.pairs.universe, data.pairs.gate
        obj.u_policy = UnlabeledPolicy(tuple(universe.eligible_drugs), tuple(universe.eligible_diseases),
            frozenset(universe.observed_pairs), frozenset(set(gate.formal) | set(gate.blocked) | set(gate.positive)))
        catalog_path = ROOT / "data/processed/canonical_graph/v1/graph/csr/edge_type_catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        fact_paths = [Path(r["path"]) for r in corpus["fact_files"]]
        certified = dict(corpus["inputs"], **corpus["outputs"])
        current_sources = {str(p): file_sha(p) for p in fact_paths}
        if any(certified.get(p) != sha for p, sha in current_sources.items()):
            raise ValueError("Native fact payload differs from the certified current-fold corpus")
        if certified.get(str(catalog_path)) != file_sha(catalog_path):
            raise ValueError("Native relation catalog differs from current-fold binding")
        bindings = dict(g1_sha256=G1_SHA256, method="M1", cell=copy.deepcopy(data.bindings),
            corpus_digest=corpus["digest"], feature_binding=obj.feature_binding,
            source_hashes=current_sources, native_sources=verify_native_identity(),
            graph_policy="NATIVE_ALL_CANONICAL_FORWARD_PLUS_COMPUTED_INVERSE_CURRENT_FOLD_LEGAL_FACTS",
            fixed_config=fixed_config(data.training_seed), support_digest=digest(sorted(data.support)),
            blocked_digest=digest(sorted(data.blocked)), U_is_confirmed_negative=False,
            only_training_roles_loaded=True, reference_or_prototype="NOT_PRESENT_IN_HGTDR_NATIVE_ARCHITECTURE")
        receipt_path = output / "RANKING_COMPARISON_native_graph.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["bindings"] != bindings:
                raise ValueError("Native graph cache binding differs")
            for relative, expected in receipt["artifacts"].items():
                if file_sha(output / relative) != expected:
                    raise ValueError("Native graph cache artifact differs")
            relations = pl.read_parquet(output / "relations.parquet")
        else:
            if (output / "edges").exists():
                raise ValueError("Uncommitted native graph export exists")
            relations, construction = export_graph_arrays(nodes, fact_paths, catalog, output, forbidden_pairs=set(data.blocked))
            relations.write_parquet(output / "relations.parquet")
            artifacts = {"relations.parquet": file_sha(output / "relations.parquet")}
            for r in relations.iter_rows(named=True):
                for key in ("edge_index_path", "fact_ids_path"):
                    artifacts[r[key]] = file_sha(output / r[key])
            receipt = dict(status="RANKING_COMPARISON_NATIVE_DATA_READY",
                           bindings=bindings, construction=construction, artifacts=artifacts)
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        obj.graph_receipt = receipt
        obj.relations = relations.to_dicts()
        arrays = [np.load(output / r["edge_index_path"], mmap_mode="r", allow_pickle=False) for r in obj.relations]
        obj.sampler = native_module().QueryMaskedSampler(obj.counts, obj.relations, arrays)
        obj.metadata = (list(obj.counts), obj.sampler.edge_types)
        return obj


def _tx_init(self,data,stream_factory,identity,*,authorization,cell,test_eids=None,batch_size=65536,test_finetune_epochs=4):
    _authorize(authorization,'TXGNN_ADAPTED',cell)
    if batch_size!=65536:raise ValueError('Native Tx batch size remains65536')
    if authorization.mode=='RANKING_COMPARISON_FORMAL' and test_eids is not None:raise PermissionError('Formal pretrain cannot use a subset')
    if not 1<=test_finetune_epochs<=4:raise ValueError('Bounded engineering limit')
    self.permit=None
    mode=authorization.mode
    import dgl
    if dgl.__version__ != '1.1.2' or torch.cuda.is_initialized():
        raise ValueError('Only certified native DGL 1.1.2 CPU execution is qualified')
    self.data, self.stream_factory = data, stream_factory
    self.identity = dict(identity, mode=mode)
    self.native = txdata.load_native()
    self.adapter = self.native.NativeCompetitionModel(data.encoder_graph, data.reference_graph, dimension=100)
    self.source = {'native': txdata.verify_native_sources(), 'recovery': file_sha(__file__),
        'state_helpers': file_sha(state_api.__file__),
        'dgl_loader': file_sha(Path(dgl.__file__).parent/'dataloading/dataloader.py'),
        'torch_loader': file_sha(Path(torch.__file__).parent/'utils/data/dataloader.py')}
    self.graph_binding = {name: self._module.graph_digest(getattr(data, name), features=name == 'graph')
        for name in ('graph', 'reference_graph', 'encoder_graph', 'positive_graph')}
    self.graph_binding['prototype'] = state_api.training_state_digest(self.adapter.model.pred.sim_all_etypes)
    allowed = self.native.pretraining_edge_ids(data.graph, data.support, data.contra)
    selected = allowed if test_eids is None else test_eids
    if not selected or any(e not in allowed or v.ndim != 1 or v.dtype != torch.int64 or
            len(torch.unique(v)) != len(v) or not bool(torch.isin(v, allowed[e]).all()) for e, v in selected.items()):
        raise ValueError('Invalid or unauthorized pretrain EID domain')
    selected = {e: v.clone() for e, v in selected.items() if len(v)}
    self.eid_binding = {'allowed': state_api.training_state_digest(allowed), 'selected': state_api.training_state_digest(selected)}
    self.scope = {'mode': mode, 'batch_size': batch_size, 'test_eid_subset': test_eids is not None,
        'planned_pretrain_batches': math.ceil(sum(map(len, selected.values()))/batch_size),
        'test_finetune_epochs': test_finetune_epochs if mode == 'TEST_ONLY' else None,
        'formal_budget': self._module.RECIPE}
    from .tx_numerical_policy import metadata
    self.scope['input_construction_policy']=metadata()
    # Even the full allowed domain can be diagnosed, but never completed in TEST_ONLY.
    self.max_pretrain_steps = min(4, self.scope['planned_pretrain_batches']) if mode == 'TEST_ONLY' else self.scope['planned_pretrain_batches']
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
    self.native_id_digest = state_api.training_state_digest(self.loader.dataset._id_tensor)
    self.eid_binding['native_id_tensor'] = self.native_id_digest
    self.cursor = {'stage': 'PRETRAIN', 'pretrain_epoch': 1, 'pretrain_batch': 0,
        'pretrain_updates': 0, 'finetune_epoch': 0, 'finetune_updates': 0,
        'actual_optimizer_updates': 0, 'consumed_edges': 0, 'negative_batches': 0}
    self.transition_state = None
    self.selection = {'completed_epochs': [], 'best': None}
    self.external_control = {}
    self.last_step = None


class RankingComparisonHGTDRReceiver(hg.SingleLRTrainer):
    def __init__(self,data,authorization,budget,cell,device='cuda'):
        _authorize(authorization,'HGTDR_ADAPTED',cell)
        from .native_numerical_policy import hgtdr_numerical_policy
        with hgtdr_numerical_policy():
            budget.native_call('HGTDR',lambda:hg.SingleLRTrainer.__init__(self,data,device=device,permit=None),
                optimizer_updates=0,forward_calls=1,positive_cap=min(4096,len(data.supports)),
                negative_cap=0,relation_edge_cap=1500000,
                **(dict(native_batches=0,init_calls=1) if hasattr(budget,'part22') else {}))
        _record_actual(budget,'HGTDR_INITIALIZE',dict(native_positive_entries=data.last_batch['entries'],
            native_negative_entries=0,native_relation_edges=data.last_batch['relation_edges'],native_batches=1,
            optimizer_updates=0,model_forward_calls=1,**data.last_batch))
        self.execution_scope=dict(mode=authorization.mode,ranking_comparison_authority=True,
            formal_authorized=authorization.mode=='RANKING_COMPARISON_FORMAL',
            runtime_sha256=sha(__file__))
    def metadata(self):
        from .native_numerical_policy import hgtdr_numerical_policy_metadata
        return dict(super().metadata(),ranking_comparison_receiver_sha256=sha(__file__),numerical_policy=hgtdr_numerical_policy_metadata())


def _record_actual(budget,kind,row):
    from .common import append
    path=(budget.events if hasattr(budget,'events') else budget.root/'native_execution.jsonl')
    append(path,dict(event='NATIVE_ACTUAL',kind=kind,session_id=getattr(budget,'sid',None),**row))


class _ObservedNativeIterator:
    """Pass through the exact DGL batch while counting its actual edge domains."""
    def __init__(self,iterator,cap,trace=None):self.iterator=iterator;self.cap=cap;self.counts=None;self.trace=trace
    def __iter__(self):return self
    def __next__(self):
        policy={};batch=next_tx_batch(self.iterator,policy)
        self.counts=dict(native_positive_entries=int(batch[1].num_edges()),native_negative_entries=int(batch[2].num_edges()),
            native_relation_edges=sum(int(b.num_edges()) for b in batch[3]),native_batches=1,
            native_input_construction_policy=policy)
        if self.counts['native_positive_entries']>65536 or self.counts['native_negative_entries']>65536 or self.counts['native_relation_edges']>self.cap:
            raise ValueError('Observed native pretrain batch exceeded registered sampler cap')
        if self.trace is not None:self.counts['native_input_observation']=self.trace(batch)
        return batch


def next_tx_batch(iterator,policy_record=None):
    """Apply registered controls; ordered equality requires separate qualification."""
    from .tx_numerical_policy import native_input_policy
    with native_input_policy() as metadata:
        batch=next(iterator)
    if policy_record is not None:policy_record.update(metadata)
    return batch


def _apply_native_selection(trainer,method,epoch,metrics):
    if method=='HGTDR_ADAPTED':
        trainer.selection.observe(epoch,metrics)
        if trainer.selection.best['epoch']==epoch:trainer.best_model=state_api._snapshot(trainer.model.state_dict())
    else:
        from evi_hgt.ranking_protocol.txgnn_protocol import InnerSelection
        control=trainer.external_control
        selection=InnerSelection.from_state(control['selection']) if 'selection' in control else InnerSelection()
        selection.observe(epoch,metrics);control['selection']=selection.state()
        if selection.best['epoch']==epoch:control['best_model']=state_api._snapshot(trainer.model.state_dict())


def restore_tx_selection_control(trainer,control,epoch):
    """Same selected-state receiver for formal disk restore and isolated fixtures."""
    from .native_checkpoint import validate_tx_selected_control
    validate_tx_selected_control(control,epoch,trainer.model.state_dict())
    trainer.external_control=state_api._snapshot(control)


def _native_run_identity(method,data,root,authorization):
    if authorization.mode=='RANKING_COMPARISON_FORMAL':
        return copy.deepcopy(authorization.identity)
    spec=authorization.budget.spec
    if spec.get('resume'):
        resume=spec['resume']
        if Path(root).resolve()!=Path(resume['output_root']).resolve():raise PermissionError('Native disk recovery root differs')
        return copy.deepcopy(resume['run_identity'])
    return dict(mode='TEST_ONLY',run_id=spec['run_id'],method=method,cell=data.cell_id,
        source_digest=digest(spec['source_sha256']),data_digest=digest(data.bindings))


class NativeRuntime:
    """Single production/test control path; budget/authority remain external."""
    def __init__(self,method,data,trainer,store,module,authorization,budget):
        self.method,self.data,self.trainer,self.store,self.module=method,data,trainer,store,module
        self.authorization,self.budget=authorization,budget
        self.cell=data.cell_id
        from .runtime_console import NativeConsole
        self.console=NativeConsole(method,authorization.mode)
    def next_action(self):
        if self.method=='HGTDR_ADAPTED':
            t=self.trainer
            if t.selection.opportunities<t.cursor['epoch']-1:return 'INNER_SELECTION'
            return 'ENDPOINT' if t.cursor['epoch']>300 else 'HGTDR_UPDATE'
        t=self.trainer
        if t.stage=='PRETRAIN':return 'TRANSITION' if t.pretrain_batch==t.planned_pretrain_batches else 'TX_PRETRAIN'
        control=t.external_control.get('selection',{})
        epochs={r['epoch'] for r in control.get('history',[])}
        if t.finetune_epoch and t.finetune_epoch%25==0 and t.finetune_epoch not in epochs:return 'INNER_SELECTION'
        return 'ENDPOINT' if t.finetune_epoch==500 else 'TX_FINETUNE'
    def snapshot(self):
        if self.method=='HGTDR_ADAPTED':
            return hg.capture_native_state(self.trainer,repeat=self.data.repeat_index,fold=self.data.cell['fold'])
        return self.trainer.snapshot()
    def commit(self,name,*,tags=()):
        return self.store.commit(name,self.snapshot(),tags=tags)
    def restore(self,commit):
        _authorize(self.authorization,self.method,self.cell)
        value=self.store.load(commit)
        if self.method=='HGTDR_ADAPTED':
            hg.restore_native_state(value,self.trainer,repeat=self.data.repeat_index,fold=self.data.cell['fold'])
        else:
            self.trainer.restore(value)
            from .tx_auxiliary_state import KEY
            restore_tx_selection_control(self.trainer,{k:v for k,v in value['external_control'].items() if k!=KEY},value['cursor']['finetune_epoch'])
        result=self.snapshot()
        if result['state_digest']!=value['state_digest']:raise ValueError('Native receiver did not exactly restore complete state')
        return result
    def step(self,*,replay=False):
        _authorize(self.authorization,self.method,self.cell)
        action=self.next_action();t=self.trainer
        if action=='HGTDR_UPDATE':
            from .native_numerical_policy import hgtdr_numerical_policy
            with hgtdr_numerical_policy():
                result=self.budget.native_call('HGTDR',t.step,optimizer_updates=1,forward_calls=1,
                    positive_cap=2048,negative_cap=2048,relation_edge_cap=1500000,replay=replay)
            result.update(native_positive_entries=result['positive_count'],native_negative_entries=result['U_count'],
                native_relation_edges=result['messages'],native_batches=1,
                actual_input_sha256=t.data.last_batch['input_sha256'],pair_order_sha256=t.data.last_batch['pair_order_sha256'])
        elif action in ('TX_PRETRAIN','TX_FINETUNE'):
            n=65536 if action=='TX_PRETRAIN' else int(t.data.positive_graph.num_edges())
            call=t.pretrain_step if action=='TX_PRETRAIN' else t.finetune_step
            observed=None;original=None
            if action=='TX_PRETRAIN':
                trace=None
                if self.authorization.mode=='TEST_ONLY' and self.budget.spec.get('capture_native_inputs'):
                    def trace(batch):
                        import dgl,uuid
                        from .native_observation import persist_native_batch
                        target=self.budget.root/self.budget.spec['run_id']/'input_payloads'
                        return persist_native_batch(batch,target,nid_key=dgl.NID,eid_key=dgl.EID,
                            label='pretrain_'+str(t.cursor['pretrain_batch'])+'_'+uuid.uuid4().hex,
                            context=dict(cursor=copy.deepcopy(t.cursor),permutation=t.permutation,
                                stream=t.stream.state(),rng=state_api._rng()))
                original=t.iterator;observed=_ObservedNativeIterator(original,2*t.data.graph.num_edges(),trace);t.iterator=observed
            try:
                result=self.budget.native_call(action,call,optimizer_updates=1,forward_calls=1,
                    positive_cap=n,negative_cap=n,relation_edge_cap=2*t.data.graph.num_edges(),replay=replay)
            finally:
                if original is not None:t.iterator=original
            result.update(native_positive_entries=result.get('positive_edges',n),
                native_negative_entries=result.get('negative_edges',n),native_batches=1,
                native_relation_edges=2*int(t.data.encoder_graph.num_edges()))
            if observed is not None:result.update(observed.counts)
        else:raise ValueError('Next action requires control handler, not optimizer: '+action)
        _record_actual(self.budget,action,dict(result,optimizer_updates=1,model_forward_calls=1,replay=replay))
        self.console.update(result,gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,replay=replay)
        return result
    def transition(self,commit):
        _authorize(self.authorization,self.method,self.cell)
        if self.method!='TXGNN_ADAPTED' or self.next_action()!='TRANSITION':raise ValueError('Not a Tx transition boundary')
        self.trainer.transition(commit)
        return self.snapshot()
    def observe_inner(self,metrics):
        """Caller must already have authorized exact G3 development scores."""
        _authorize(self.authorization,self.method,self.cell)
        if self.next_action()!='INNER_SELECTION':raise ValueError('Not a registered inner opportunity')
        if self.authorization.mode!='RANKING_COMPARISON_FORMAL':raise PermissionError('No real development metrics in engineering session')
        t=self.trainer;epoch=t.cursor['epoch']-1 if self.method=='HGTDR_ADAPTED' else t.finetune_epoch
        _apply_native_selection(t,self.method,epoch,metrics)
        return self.snapshot()


def qualify_selection_fixture(runtime,dest):
    """Zero-forward isolated virtual-boundary fixture, never a training state.

    The actual receiver's model/optimizer/cursor are not modified. Only native
    selection/control serialization is exercised with explicit synthetic metric
    values. No invented optimizer moments or scientifically completed epochs.
    """
    _authorize(runtime.authorization,runtime.method,runtime.cell)
    if runtime.authorization.mode!='TEST_ONLY':raise PermissionError('Selection fixture is engineering-only')
    from .common import write_new
    from .recovery_diagnostics import persist_comparison
    t=runtime.trainer;before=runtime.snapshot()['state_digest'];probe=copy.copy(t)
    fixture=dict(schema='RANKING_COMPARISON_NATIVE_ISOLATED_SELECTION_FIXTURE_V1',mode='TEST_ONLY',
        synthetic_epoch_boundary=True,full_training_state=False,optimizer_state_included=False,
        actual_optimizer_updates=0,model_forward_calls=0,candidate_scores=0,
        metrics_are_synthetic_fixture_values=True,metrics=dict(macro_auc=.5,macro_ap=.5))
    if runtime.method=='HGTDR_ADAPTED':
        epoch=1;steps=math.ceil(len(t.data.supports)/(t.config['batch_size']//2))
        probe.cursor=dict(epoch=2,batch_index=0,optimizer_steps=steps)
        probe.selection=hg.InnerCheckpointSelection();probe.best_model=None
        probe.training_history=[dict(epoch=1,positive_pairs_seen=len(t.data.supports),unlabeled_pairs_seen=len(t.data.supports))]
        probe.epoch_accumulator=dict(weighted_loss=0.,examples=0,P=0,U=0,masked=0)
        probe.forwards=max(steps+1,t.forwards);probe.query_forward_count=max(probe.forwards,t.query_forward_count)
        _apply_native_selection(probe,runtime.method,epoch,fixture['metrics'])
        payload=probe.control_state()
    else:
        epoch=25;probe.cursor=dict(t.cursor,finetune_epoch=epoch)
        probe.external_control={}
        _apply_native_selection(probe,runtime.method,epoch,fixture['metrics'])
        payload=state_api._snapshot(probe.external_control)
    path=Path(dest)/'isolated_selection_control.pt'
    with path.open('xb') as handle:torch.save(payload,handle)
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    receiver=copy.copy(t)
    if runtime.method=='HGTDR_ADAPTED':
        receiver.selection=hg.InnerCheckpointSelection();receiver.load_control_state(loaded)
        actual=receiver.control_state()
    else:
        receiver.cursor=dict(probe.cursor);receiver.external_control={}
        restore_tx_selection_control(receiver,loaded,epoch);actual=receiver.external_control
    compared=persist_comparison(dest,'isolated_selection_receiver',payload,actual,exact=True)
    runtime_probe=copy.copy(runtime);runtime_probe.trainer=receiver
    if runtime.snapshot()['state_digest']!=before:raise ValueError('Isolated fixture changed actual native receiver')
    fixture.update(pass_=compared['pass_'],virtual_epoch=epoch,comparison=compared,
        next_action=runtime_probe.next_action(),path=str(path),sha256=sha(path),
        actual_training_state_unchanged=True,
        scope='Native selected tensor/control reader only; not proof of training through this epoch')
    write_new(Path(dest)/'isolated_selection_control.json',fixture)
    return fixture


def build_hgtdr(data,corpus,output_root,authorization,budget,*,device='cuda',thin=False,retention_evidence=None):
    _authorize(authorization,'HGTDR_ADAPTED',data.cell_id)
    cache=ROOT/f'data/cache/ranking_comparison/native/HGTDR/{data.cell_id}'
    native_data=RankingComparisonHGTDRFoldData.from_cell(data,corpus,cache,authorization=authorization)
    trainer=RankingComparisonHGTDRReceiver(native_data,authorization,budget,data.cell_id,device)
    identity=_native_run_identity('HGTDR_ADAPTED',data,output_root,authorization)
    native_identity=hg.native_state_identity(trainer,data.repeat_index,data.cell['fold'])
    def validate(state):
        if state['model_metadata']!=trainer.metadata():raise ValueError('Native live data/model binding differs')
    store=NativeCheckpointStore(Path(output_root)/'checkpoints',identity,kind='HGTDR',native_identity=native_identity,
        bindings=source_bindings('HGTDR'),validate_state=validate,
        capability=getattr(authorization,'capability',None),retain_all=not thin,retention_evidence=retention_evidence)
    return NativeRuntime('HGTDR_ADAPTED',data,trainer,store,hg,authorization,budget)


def build_tx(data,corpus,output_root,authorization,budget,*,test_eid_limit=131072):
    _authorize(authorization,'TXGNN_ADAPTED',data.cell_id)
    if torch.cuda.is_initialized():raise ValueError('Tx requires independent certified CPU process')
    torch.set_num_threads(8)
    assert_native_global_fact_mask(corpus)
    payload=dict(schema=txbinding.PAYLOAD_SCHEMA,mode=authorization.mode,repeat=data.repeat_index,fold=data.cell['fold'],
        cell=txbinding._cell(data.repeat_index,data.cell['fold']),bindings=dict(data.bindings),corpus_digest=corpus['digest'],
        corpus_path=str(Path(corpus['corpus_root'])/'manifest.json'),training=data.cell['training'],auxiliary_training=data.cell['auxiliary_training'],
        support_pairs=sorted(data.support),contra_pairs=sorted(data.unsuitability),forbidden_pairs=sorted(data.blocked),
        observed_pairs=sorted(data.shared.observed),eligible_drugs=data.u['eligible_drugs'],eligible_diseases=data.u['eligible_diseases'],
        g1_sha256=G1_SHA256,native_sources=txdata.verify_native_sources())
    payload['digest']=digest(payload)
    binding=txbinding.validate_training_payload(payload,corpus)
    native_data=txdata._prepare_fold_core(corpus=corpus,support_pairs=payload['support_pairs'],contra_pairs=payload['contra_pairs'],
        forbidden_pairs=payload['forbidden_pairs'],binding=binding)
    info=txbinding.runtime_identity(native_data)
    native_identity={k:info[k] for k in ('method','repeat','fold','seed','cell_digest','g1_sha256','corpus_digest','source_digest','environment_digest')}
    native_identity.update(mode=authorization.mode,ranking_comparison_runtime_sha256=sha(__file__))
    identity=_native_run_identity('TXGNN_ADAPTED',data,output_root,authorization)
    holder={}
    def validate(state):
        trainer=holder.get('trainer')
        if trainer is not None:
            from .tx_auxiliary_state import KEY,validate as validate_auxiliary
            validate_auxiliary(state['external_control'].get(KEY))
            for k in ('source','graph_binding','eid_binding','scope'):
                if state[k]!=getattr(trainer,k):raise ValueError('Tx live source/data identity differs: '+k)
    store=NativeCheckpointStore(Path(output_root)/'checkpoints',identity,kind='TXGNN',native_identity=native_identity,
        bindings=source_bindings('TXGNN'),validate_state=validate,capability=getattr(authorization,'capability',None))
    module=load_private_native_module(store)
    from .tx_auxiliary_state import snapshot_receiver,restore_receiver
    receiver_class=type('RankingComparisonTxReceiver',(module.TxTwoStageTrainer,),{'__init__':_tx_init,'_module':module,
        'snapshot':snapshot_receiver,'restore':restore_receiver})
    subset=None
    if authorization.mode=='TEST_ONLY':
        if type(test_eid_limit) is not int or not 65536<=test_eid_limit<=196608:raise ValueError('One to three full native TEST_ONLY batches only')
        subset={};remaining=test_eid_limit
        for etype,ids in native_data.allowed_pretrain_eids.items():
            count=min(remaining,len(ids))
            if count:subset[etype]=ids[:count].clone();remaining-=count
            if not remaining:break
        if remaining:raise ValueError('Registered native EID subset unavailable')
    trainer=receiver_class(native_data,txbinding.stream_factory_for_cell(native_data,data),native_identity,
        authorization=authorization,cell=data.cell_id,test_eids=subset,batch_size=65536,test_finetune_epochs=4)
    holder['trainer']=trainer
    return NativeRuntime('TXGNN_ADAPTED',data,trainer,store,module,authorization,budget)


def evaluate_native(runtime,task,domain,token,*,selected=False):
    """G3 only; exact native sampling/decoder plus versioned common metrics."""
    from .runtime_evaluation import project_development
    from .metrics import evaluate_panel
    from .common import read,write_new
    auth=runtime.authorization
    if not isinstance(auth,FormalNativeCapability):raise PermissionError('No native candidate evaluation in TEST_ONLY')
    view=project_development(runtime.data,auth.capability,auth.identity,auth.output_root)
    if domain not in ('validation','train_probe','auxiliary_validation'):raise PermissionError('Not a G3 development domain')
    t=runtime.trainer;contra=domain=='auxiliary_validation'
    if runtime.method=='HGTDR_ADAPTED' and contra:raise ValueError('HGTDR has no contra head')
    rows=([dict(x,role='P') for x in view.parts[domain]]+[dict(x) for x in view.parts['validation'] if x['role']=='U']) if contra else [dict(x) for x in view.parts[domain]]
    rows.sort(key=lambda r:(r['drug_id'],r['disease_id']))
    for row in rows:view.assert_score_domain((row['drug_id'],row['disease_id']),'validation' if contra and row['role']=='U' else domain)
    if len({(r['drug_id'],r['disease_id']) for r in rows})!=len(rows):raise ValueError('Duplicate G3 candidate')
    weights=(t.best_model if runtime.method=='HGTDR_ADAPTED' else t.external_control.get('best_model')) if selected else t.model.state_dict()
    if weights is None:raise ValueError('Selected weights missing')
    binding=dict(run_identity=auth.identity,weights_sha256=state_api.training_state_digest(weights),head='contra' if contra else 'support',
        candidate_order_sha256=digest(rows),mask=runtime.data.bindings['fold_mask_digest'],corpus=task['corpus'],source=sha(__file__))
    dest=Path(task['output_root'])/'evaluation';dest.mkdir(exist_ok=True)
    path=dest/(token+'_'+domain+'.json')
    def counters():
        return {k:getattr(t,k) for k in ('forwards','query_forward_count') if hasattr(t,k)}
    def display(metrics,context):
        from .console import Console
        console=Console(context=context)
        step=t.cursor['optimizer_steps'] if runtime.method=='HGTDR_ADAPTED' else t.cursor['actual_optimizer_updates']
        console.evaluation_result(runtime.method,step,{'contra' if contra else 'support':metrics});console.close()
    if path.exists():
        cached=read(path)
        if cached['binding']!=binding:raise ValueError('Cached native evaluation binding differs')
        current=counters()
        if current==cached['counters_before']:
            for key,value in cached['counters_after'].items():setattr(t,key,value)
        elif current!=cached['counters_after']:raise ValueError('Native cached evaluation counters differ')
        display(cached['metrics'],'回放')
        return cached['metrics']
    counters_before=counters()
    rng=state_api._rng();original=state_api._snapshot(t.model.state_dict()) if selected else None
    modes={k:m.training for k,m in t.model.named_modules()}
    try:
        if selected:t.model.load_state_dict(weights)
        t.model.eval()
        with torch.inference_mode():
            if runtime.method=='HGTDR_ADAPTED':
                pairs=np.asarray([[t.data.lookup['drug'][r['drug_id']],t.data.lookup['disease'][r['disease_id']]] for r in rows],dtype=np.int64)
                values=[]
                for offset in range(0,len(pairs),t.config['batch_size']):
                    p=pairs[offset:offset+t.config['batch_size']]
                    batch=t.data.batch(p,t.config,t.native.seed_for(t.config['seed'],'evaluation',domain,offset),t.device)
                    values.extend(t.model(batch).view(-1).cpu().tolist())
                    t.forwards+=1;t.query_forward_count+=len(p)
            else:
                pairs=torch.tensor([[t.data.drug_local[r['drug_id']] for r in rows],[t.data.disease_local[r['disease_id']] for r in rows]],dtype=torch.int64)
                graph=t.native.dda_graph(t.data.graph,pairs,pairs)
                etype=('drug','contraindication' if contra else 'indication','disease')
                values=t.adapter.score_graph(graph,etypes=[etype])[etype].cpu().tolist()
        if len(values)!=len(rows) or not all(math.isfinite(v) for v in values):raise ValueError('Native scores invalid')
        for row,value in zip(rows,values):row['unsuitability_logit' if contra else 'support_logit']=float(value)
        metrics=evaluate_panel(rows,head='contra' if contra else 'support')
        write_new(path,dict(binding=binding,rows=rows,metrics=metrics,counters_before=counters_before,counters_after=counters()))
        display(metrics,'AUTHORIZED_RUNTIME')
        return metrics
    finally:
        if original is not None:t.model.load_state_dict(original)
        for k,m in t.model.named_modules():m.training=modes[k]
        import random
        random.setstate(rng['python']);np.random.set_state(state_api._validate_rng(rng));torch.set_rng_state(rng['cpu'])
        if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])


class _FormalNativeLedger:
    def __init__(self,root):self.root=Path(root)
    def native_call(self,kind,call,**counts):
        from .common import append
        from .runtime_budget import disk_guard
        disk_guard(self.root,1_000_000_000)
        append(self.root/'native_execution.jsonl',dict(event='BEGIN',kind=kind,**counts))
        value=call()
        append(self.root/'native_execution.jsonl',dict(event='COMPLETE',kind=kind,**counts))
        return value


def run_formal_native(task,capability,manifest):
    """Real serial 300epoch / one pretrain+500epoch driver; gated before I/O."""
    from .data import TrainingInputs
    from .corpus import verify_manifest
    from .common import write_new,read
    identity={k:task[k] for k in ('run_id','method','cell','source_digest','data_digest')}
    identity['mode']='RANKING_COMPARISON_FORMAL';root=Path(task['output_root'])
    auth=FormalNativeCapability(capability,identity,root)
    auth.check_native(task['method'],task['cell'])
    if (root/'COMPLETE.json').exists():
        receipt=read(root/'COMPLETE.json')
        if receipt['identity']!=identity or sha(receipt['selected_weights_path'])!=receipt['selected_weights_sha256']:raise ValueError('Completed native trajectory binding differs')
        return receipt
    root.mkdir(parents=True,exist_ok=True)
    data=TrainingInputs().cell(task['cell']);corpus=verify_manifest(data)
    from .startup import verify_task_inputs
    verify_task_inputs(task,data,corpus)
    ledger=_FormalNativeLedger(root)
    if task['method']=='HGTDR_ADAPTED':
        runtime=build_hgtdr(data,corpus,root,auth,ledger,thin=manifest.get('native_retention',{}).get('HGTDR')=='LATEST_PREVIOUS_PINNED',
            retention_evidence=manifest.get('native_retention_evidence'))
    elif task['method']=='TXGNN_ADAPTED':runtime=build_tx(data,corpus,root,auth,ledger,test_eid_limit=None)
    else:raise ValueError('Not a native method')
    if list(runtime.store.root.glob('native_*.COMMIT.json')):runtime.restore(runtime.store.latest()['checkpoint_commit_path'])
    else:runtime.commit('fresh_start')
    while runtime.next_action()!='ENDPOINT':
        action=runtime.next_action();t=runtime.trainer
        if action=='INNER_SELECTION':
            epoch=t.cursor['epoch']-1 if task['method']=='HGTDR_ADAPTED' else t.finetune_epoch
            prefix=f'epoch_{epoch:04d}'
            runtime.commit(prefix+'_pre_inner')
            metrics=evaluate_native(runtime,task,'validation',prefix)
            runtime.observe_inner(metrics);runtime.commit(prefix+'_inner_complete');runtime.store.collect()
        elif action=='TRANSITION':
            anchor=runtime.commit('pretrain_complete',tags=('PRETRAIN_TRANSITION_SOURCE',))
            runtime.transition(anchor);runtime.commit('finetune_start',tags=('DECODER_RESET',))
        else:
            runtime.step()
            if action=='TX_PRETRAIN' and t.pretrain_batch%32==0:runtime.commit(f'pretrain_{t.pretrain_batch:05d}')
    endpoint=runtime.commit('endpoint_complete',tags=('ENDPOINT',))
    selected=runtime.trainer.best_model if task['method']=='HGTDR_ADAPTED' else runtime.trainer.external_control['best_model']
    if selected is None:raise ValueError('Native endpoint lacks selected model')
    path=root/'selected.weights.pt'
    if not path.exists():
        from .resource_guard import checkpoint_required_bytes
        from .runtime_budget import disk_guard
        disk_guard(path,checkpoint_required_bytes(selected))
        import os,uuid
        partial=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
        with partial.open('xb') as f:torch.save(state_api._snapshot(selected),f);f.flush();os.fsync(f.fileno())
        os.rename(partial,path)
    if state_api.training_state_digest(torch.load(path,map_location='cpu',weights_only=True))!=state_api.training_state_digest(selected):raise ValueError('Selected native weight identity differs')
    evaluate_native(runtime,task,'train_probe','selected',selected=True)
    if task['method']=='TXGNN_ADAPTED':evaluate_native(runtime,task,'auxiliary_validation','selected',selected=True)
    receipt=dict(status='COMPLETE',run_id=identity['run_id'],identity=identity,endpoint=endpoint,selected_weights_path=str(path),selected_weights_sha256=sha(path),
        selected_tensor_sha256=state_api.training_state_digest(selected),selected_can_resume_training=False,
        evaluations=[dict(path=str(p),sha256=sha(p)) for p in sorted((root/'evaluation').glob('*.json'))])
    write_new(root/'COMPLETE.json',receipt);return receipt
