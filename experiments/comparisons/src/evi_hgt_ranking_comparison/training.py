"""Shared, inert preparation/reduction adapters for the three RankingComparison methods.

No model call, backward, optimizer update, execution grant, or launcher lives
here. A separately authorized caller supplies predictions and owns its durable
budget. One prepared step is the same sealed P/U record for all three methods.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from evi_hgt.graph_inputs.common import digest, verify
from evi_hgt.conditional_sampling.conditional import group_parts
from evi_hgt.training.supervision import make_unlabeled_query_supervision
from .common import ROOT, OUT, read, sha


METHODS = ('FULL', 'PAIR_MLP_D1_MATCHED', 'HGT_D1_MATCHED')
LOSS_ARM = 'E1_D1_SOFTPLUS_RANKING'
FIXED_CONTRACT = dict(
    schema='RANKING_COMPARISON_SHARED_TRAINING_V1', methods=list(METHODS),
    loss='EMPIRICAL_PU_HALF_SOFTPLUS_V1', temperature=1.,
    lambda_support=1., lambda_unsuitability=1., channel_weight_normalization='sum=2',
    pairwise_chunk_size=4096, reduction_dtype='float32',
    support_reduction='mean over original P occurrences; local U only in cycles 2/3',
    contra_reduction='original mean P x original shared global U for every cycle',
    U_semantics='UNKNOWN_NOT_CONFIRMED_NEGATIVE',
    optimizer='AdamW', learning_rate=5e-5, weight_decay=1e-5,
    betas=[.9, .999], eps=1e-8, foreach=None, fused=None,
    scheduler=None, max_grad_norm=1., gradient_accumulation=1,
    finite_zero_gradient='original LOSS_COMPARISON finite-zero optimizer step semantics',
    precision='BF16_FORWARD_FP32_LOSS', graph_batch_size=1,
    update_unit='one complete original P-group', cycles=3, selection='FIXED_3G',
    execution_authorized=False,
)


def validate_step_record(queries, record, *, step, G):
    """Validate ordered role records without reading graphs or executing models."""
    if type(G) is not int or G <= 0 or type(step) is not int or not 1 <= step <= 3*G:
        raise ValueError('Step outside fixed three complete P cycles')
    queries = tuple(queries)
    support, contra, original = group_parts(queries)
    pairs = [(q.drug_id, q.disease_id) for q in queries]
    extras = []; fallback = 0
    if step <= G:
        if record is not None:
            raise ValueError('Global cycle cannot consume a conditional record')
    else:
        if not isinstance(record, dict):
            raise ValueError('Conditional cycle requires its sealed record')
        if record['logical_step'] != step or record['group_index'] != step-G-1:
            raise ValueError('Conditional record cursor changed')
        if (record['support_positive_pairs'] != [list(p) for p in support]
            or record['contra_positive_pairs'] != [list(p) for p in contra]
            or record['original_global_u'] != list(original)
            or record['original_support_u_count'] != 1):
            raise ValueError('Original P/global-U order changed')
        diseases = tuple(dict.fromkeys(p[1] for p in support))
        for name in ('disease_unlabeled_pairs', 'disease_status', 'draw_seed_by_disease'):
            if set(record[name]) != set(diseases):
                raise ValueError('Conditional disease/seed domain changed')
        for disease in diseases:
            local = record['disease_unlabeled_pairs'][disease]
            status = record['disease_status'][disease]
            seed = record['draw_seed_by_disease'][disease]
            if type(seed) is not int or seed < 0:
                raise ValueError('Conditional private seed changed')
            if not local:
                if status != 'NO_LEGAL_U_ORIGINAL_PAIRING_FALLBACK':
                    raise ValueError('Missing local U has an undeclared fallback')
                fallback += sum(p[1] == disease for p in support)
                continue
            if (status != 'DISEASE_CONDITIONAL_UNIFORM' or len(local) != 1
                or len(local[0]) != 2 or local[0][1] != disease
                or not all(isinstance(v, str) and v for v in local[0])):
                raise ValueError('Local U must be one canonical pair of the positive disease')
            pair = tuple(local[0])
            if pair in set(support) | set(contra):
                raise ValueError('Known positive cannot become a local U')
            if pair not in pairs and pair not in extras:
                extras.append(pair)
        if record['extra_pairs'] != [list(p) for p in extras]:
            raise ValueError('Conditional extra U order/membership changed')
    return dict(stage='GLOBAL' if step <= G else 'CONDITIONAL',
        cycle=(step-1)//G+1, original_queries=len(queries),
        conditional_extra_U_queries=len(extras), query_scores=len(queries)+len(extras),
        query_pairs=[list(p) for p in pairs+extras], support_pairs=len(support),
        contra_pairs=len(contra), fallback_support_occurrences=fallback)


@dataclass(frozen=True)
class StepSpec:
    cell: str
    step: int
    G: int
    queries: tuple
    record: dict | None
    identity: dict
    content_digest: str

    def validate(self):
        content = dict(cell=self.cell, step=self.step, G=self.G,
            queries=[asdict(q) for q in self.queries], record=self.record,
            identity=self.identity)
        if digest(content) != self.content_digest:
            raise ValueError('Prepared sampling record changed after binding')
        result = validate_step_record(self.queries, self.record, step=self.step, G=self.G)
        if digest([asdict(q) for q in self.queries]) != self.identity['training_member_sha256']:
            raise ValueError('Training member order/hash changed')
        if (digest(self.record) if self.record is not None else None) != self.identity['record_sha256']:
            raise ValueError('Conditional record hash changed')
        return result

    def metadata(self, method):
        if method not in METHODS:
            raise ValueError('Only the three preregistered shared-plan methods')
        return dict(self.validate(), method=method, cell=self.cell, step=self.step,
                    total_steps=3*self.G, **deepcopy(self.identity))


class SealedTrainingPlan:
    """Columnar schedule reader bound to the completed 15-cell seal.

    The global schedule is retained as a compact Polars frame. Only a single
    group's Python supervision records are reconstructed per request. Stored
    historical plans are read-only; no sampling or alternative order is drawn.
    """
    def __init__(self, data, cell_seal, *, all_seals=None):
        all_seals = verify(read(OUT/'sampling/ALL_PLANS_SEALED.json')) if all_seals is None else verify(all_seals)
        seal = deepcopy(verify(cell_seal))
        if (all_seals.get('schema') != 'RANKING_COMPARISON_ALL_CELL_PLANS_SEALED_V1'
            or all_seals.get('status') != 'SEALED_15_CELLS'
            or {r['cell'] for r in all_seals['cells']} != {f'R{r}F{f}' for r in range(3) for f in range(5)}
            or len(all_seals['cells']) != 15
            or all_seals['shared_methods'] != list(METHODS)):
            raise ValueError('All fifteen sampling plans must be sealed before use')
        if next((r for r in all_seals['cells'] if r['cell'] == data.cell_id), None) != seal:
            raise ValueError('Cell seal is not in the completed shared plan')
        if (seal['schema'] != 'RANKING_COMPARISON_CELL_PLAN_SEAL_V1' or seal['status'] != 'SEALED'
            or seal['cell'] != data.cell_id or seal['seed'] != data.training_seed
            or seal['G'] != data.budget()['G_rf']
            or seal['mask_digest'] != data.bindings['fold_mask_digest']
            or seal['training_membership_digests'] != {n:digest(data.cell[n]) for n in ('training','auxiliary_training')}):
            raise ValueError('Current training data differs from cell seal')
        paths = {}
        for kind, name in (('schedule','schedule.json'), ('plan','conditional_plan.json')):
            path = Path(seal[kind+'_path']).resolve()
            if (not path.is_relative_to((ROOT/'data/cache/ranking_comparison'/data.cell_id).resolve())
                or path.name != name or sha(path) != seal[kind+'_sha256']):
                raise ValueError('Sealed '+kind+' path/hash changed')
            paths[kind] = path
        schedule, plan = (verify(read(paths[k])) for k in ('schedule','plan'))
        G = seal['G']
        if (schedule['schema'] != 'RANKING_COMPARISON_THREE_GLOBAL_CYCLES_V1'
            or schedule['digest'] != seal['schedule_digest']
            or schedule['identity']['cell'] != data.cell_id
            or schedule['identity']['data_bindings'] != data.bindings
            or schedule['G'] != G or schedule['total'] != 3*G
            or len(schedule['member_stream']) != 3*G
            or plan['schema'] != 'RANKING_COMPARISON_SHARED_CONDITIONAL_PLAN_V1'
            or plan['digest'] != seal['plan_digest'] or plan['cell'] != data.cell_id
            or plan['schedule_digest'] != schedule['digest'] or plan['warmup_steps'] != G
            or plan['suffix_groups'] != 2*G or len(plan['records']) != 2*G
            or plan['record_digest'] != digest(plan['records']) or plan['shared_by'] != list(METHODS)
            or plan['origin'] != schedule['identity']['origin']
            or plan['origin']['origin_sha256'] != seal['origin_sha256']):
            raise ValueError('Shared schedule/conditional plan identity changed')
        gp = Path(schedule['groups_path']).resolve()
        if gp != paths['schedule'].parent/'global_groups.parquet' or sha(gp) != schedule['groups_sha256']:
            raise ValueError('Global group bytes changed')
        self._frame = pl.read_parquet(gp)
        steps = self._frame['step'].to_numpy()
        if not len(steps) or np.any(steps[1:] < steps[:-1]):
            raise ValueError('Schedule rows are not in original step order')
        offsets = np.r_[0, np.flatnonzero(steps[1:] != steps[:-1])+1, len(steps)]
        if not np.array_equal(steps[offsets[:-1]], np.arange(1, 3*G+1)):
            raise ValueError('Schedule is missing/duplicating logical step blocks')
        self._offsets = offsets
        self.data, self.seal, self.schedule, self.plan, self.G = data, seal, schedule, plan, G
        self._record_digests = tuple(digest(r) for r in plan['records'])
        self._member_digests = tuple(digest(r) for r in schedule['member_stream'])
        self._metadata_digest = self._metadata_hash()

    def _metadata_hash(self):
        return digest(dict(seal=self.seal,
            schedule={k:v for k,v in self.schedule.items() if k != 'member_stream'},
            plan={k:v for k,v in self.plan.items() if k != 'records'}))

    def step(self, step):
        if type(step) is not int or not 1 <= step <= 3*self.G:
            raise ValueError('Step outside sealed 3G schedule')
        if self._metadata_hash() != self._metadata_digest:
            raise ValueError('In-memory sealed plan metadata changed')
        # In-memory metadata are copied once; callers do not need disk rescans
        # every update. Current record/group checks below still fail on drift.
        start, stop = map(int, self._offsets[step-1:step+1])
        rows = self._frame.slice(start, stop-start).to_dicts()
        queries = []; member = self.schedule['member_stream'][step-1]
        if member['step'] != step or digest(member) != self._member_digests[step-1]:
            raise ValueError('Member stream cursor changed')
        for position, row in enumerate(rows):
            if (row['step'] != step or row['cycle'] != (step-1)//self.G+1
                or row['position'] != position or row['member_sha256'] != member['member_sha256']):
                raise ValueError('Original intra-group row order changed')
            pair = row['drug_id'], row['disease_id']; role = row['role']
            if role not in ('support','unsuitability','U'):
                raise ValueError('Unknown sealed training role')
            self.data.pairs.assert_optimizer_pair(pair, 'U' if role == 'U' else 'P')
            if ((role == 'support' and pair not in self.data.support)
                or (role == 'unsuitability' and pair not in self.data.unsuitability)):
                raise ValueError('Positive channel membership changed')
            queries.append(make_unlabeled_query_supervision(*pair) if role == 'U' else self.data.positive(pair, role))
        queries = tuple(queries)
        record = None if step <= self.G else deepcopy(self.plan['records'][step-self.G-1])
        if record is not None and digest(record) != self._record_digests[step-self.G-1]:
            raise ValueError('In-memory sealed conditional record changed')
        counts = validate_step_record(queries, record, step=step, G=self.G)
        if record is not None:
            for values in record['disease_unlabeled_pairs'].values():
                for pair in values:
                    self.data.pairs.assert_optimizer_pair(tuple(pair), 'U')
        identity = dict(all_method_shared=True, training_member_sha256=member['member_sha256'],
            record_sha256=digest(record) if record is not None else None,
            sampling_sha256=self.schedule['digest'] if record is None else self.plan['digest'],
            schedule_sha256=self.seal['schedule_sha256'], schedule_digest=self.schedule['digest'],
            conditional_plan_sha256=self.seal['plan_sha256'], conditional_plan_digest=self.plan['digest'],
            mask_digest=self.seal['mask_digest'], training_seed=self.seal['seed'],
            ordered_query_pair_digest=digest(counts['query_pairs']), loss_id=FIXED_CONTRACT['loss'])
        content = dict(cell=self.data.cell_id, step=step, G=self.G,
            queries=[asdict(q) for q in queries], record=record, identity=identity)
        result = StepSpec(self.data.cell_id, step, self.G, queries, record, identity, digest(content))
        result.validate()
        return result


@dataclass(frozen=True)
class PreparedStep:
    spec: StepSpec
    graphs: tuple
    extras: tuple


def _validate_prepared(prepared):
    spec = prepared.spec; spec.validate()
    extra = () if spec.record is None else tuple(make_unlabeled_query_supervision(*p) for p in spec.record['extra_pairs'])
    if len(prepared.graphs) != len(spec.queries) or len(prepared.extras) != len(extra):
        raise ValueError('Prepared original/extra graph counts changed')
    for graph, query in zip(prepared.graphs+prepared.extras, spec.queries+extra, strict=True):
        if tuple(graph.batch.supervision) != (query,):
            raise ValueError('Prepared graph supervision/order differs from sealed query')


def prepare_step(assembler, spec, *, device='cpu'):
    """Preserve original graph order followed by declared extra U graph order."""
    spec.validate()
    extra = () if spec.record is None else tuple(make_unlabeled_query_supervision(*p) for p in spec.record['extra_pairs'])
    graphs = tuple(assembler.assemble(q, domain='training', device=device) for q in spec.queries)
    extras = tuple(assembler.assemble(q, domain='training', device=device) for q in extra)
    result = PreparedStep(spec, graphs, extras)
    _validate_prepared(result)
    return result


def reduce_step(prepared, predictions, extra_predictions=()):
    """Exact LOSS_COMPARISON FP32 risk on predictions supplied by a budgeted caller."""
    from evi_hgt_loss_comparison.losses import objective
    from evi_hgt.inner_validation.pilot import risks_for_group
    from evi_hgt.conditional_sampling.pilot import conditional_group_risk
    _validate_prepared(prepared)
    predictions, extra_predictions = tuple(predictions), tuple(extra_predictions)
    if len(predictions) != len(prepared.graphs) or len(extra_predictions) != len(prepared.extras):
        raise ValueError('Prediction original/extra ordering/count changed')
    obj = objective(LOSS_ARM)
    spec = prepared.spec
    risk = (risks_for_group(obj, tuple(g.batch for g in prepared.graphs), predictions)
            if spec.record is None else conditional_group_risk(
                obj, spec.queries, prepared.graphs, predictions, spec.record, extra_predictions))
    if not risk.finite:
        raise FloatingPointError('Nonfinite shared LOSS_COMPARISON loss')
    return risk
