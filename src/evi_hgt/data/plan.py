"""Read the supplied immutable three-cycle schedule without resampling."""
from dataclasses import asdict, dataclass
from copy import deepcopy
import numpy as np
import polars as pl

from .common import digest, read_json
from evi_hgt.training.supervision import make_unlabeled_query_supervision


def group_parts(group):
    support = [(q.drug_id, q.disease_id) for q in group if q.support.observed_mask]
    contra = [(q.drug_id, q.disease_id) for q in group if q.unsuitability.observed_mask]
    unlabeled = [(q.drug_id, q.disease_id) for q in group if q.observed_relation is None
                 and q.support.pu_risk_eligible and q.unsuitability.pu_risk_eligible]
    if not support or not contra or len(unlabeled) != 1:
        raise ValueError('Each update requires both positive channels and one shared global U')
    return support, contra, unlabeled[0]


def validate_step_record(queries, record, *, step, G):
    if type(G) is not int or G <= 0 or type(step) is not int or not 1 <= step <= 3*G:
        raise ValueError('Step outside the fixed three complete positive cycles')
    support, contra, original = group_parts(queries)
    pairs = [(q.drug_id, q.disease_id) for q in queries]
    extras = []
    if step <= G:
        if record is not None:
            raise ValueError('Global cycle cannot consume a conditional record')
    else:
        if not isinstance(record, dict) or record.get('logical_step') != step or record.get('group_index') != step-G-1:
            raise ValueError('Conditional record cursor changed')
        if (record.get('support_positive_pairs') != [list(p) for p in support]
            or record.get('contra_positive_pairs') != [list(p) for p in contra]
            or record.get('original_global_u') != list(original)
            or record.get('original_support_u_count') != 1):
            raise ValueError('Original positive/global-U order changed')
        diseases = tuple(dict.fromkeys(p[1] for p in support))
        for name in ('disease_unlabeled_pairs', 'disease_status', 'draw_seed_by_disease'):
            if set(record.get(name, {})) != set(diseases):
                raise ValueError('Conditional disease/seed domain changed')
        for disease in diseases:
            local = record['disease_unlabeled_pairs'][disease]
            status = record['disease_status'][disease]
            seed = record['draw_seed_by_disease'][disease]
            if type(seed) is not int or seed < 0:
                raise ValueError('Conditional private seed must be a nonnegative integer')
            if not local:
                if status != 'NO_LEGAL_U_ORIGINAL_PAIRING_FALLBACK':
                    raise ValueError('Missing local U requires its declared fallback')
                continue
            if (status != 'DISEASE_CONDITIONAL_UNIFORM' or len(local) != 1 or len(local[0]) != 2
                or local[0][1] != disease or not all(isinstance(v, str) and v for v in local[0])):
                raise ValueError('Local U must be one canonical pair of the positive disease')
            pair = tuple(local[0])
            if pair in set(support) | set(contra):
                raise ValueError('A known positive cannot become a local U')
            if pair not in pairs and pair not in extras:
                extras.append(pair)
        if record.get('extra_pairs') != [list(p) for p in extras]:
            raise ValueError('Conditional extra-U order/membership changed')
    return {'cycle': (step-1)//G+1, 'query_pairs': [list(p) for p in pairs+extras],
            'query_scores': len(pairs)+len(extras)}


@dataclass(frozen=True)
class StepSpec:
    step: int
    G: int
    queries: tuple
    record: dict | None
    identity: dict
    content_digest: str

    def validate(self):
        content = dict(step=self.step, G=self.G, queries=[asdict(q) for q in self.queries],
                       record=self.record, identity=self.identity)
        if digest(content) != self.content_digest:
            raise ValueError('Prepared schedule record changed after binding')
        return validate_step_record(self.queries, self.record, step=self.step, G=self.G)


class TrainingPlan:
    def __init__(self, corpus, spec, resolve):
        self.corpus = corpus
        self.G = spec['G']
        if type(self.G) is not int or self.G <= 0:
            raise ValueError('Schedule G must be a positive integer')
        self.total_steps = 3*self.G
        self._frame = pl.read_parquet(resolve(spec['groups']))
        required = {'step', 'cycle', 'position', 'drug_id', 'disease_id', 'role'}
        if not required <= set(self._frame.columns) or self._frame.select(sorted(required)).null_count().sum_horizontal().item():
            raise ValueError('Training schedule requires complete ordered role columns')
        steps = self._frame['step'].to_numpy()
        if not len(steps) or steps.dtype.kind not in 'iu' or np.any(steps[1:] < steps[:-1]):
            raise ValueError('Schedule rows must retain original step order')
        self._offsets = np.r_[0, np.flatnonzero(steps[1:] != steps[:-1])+1, len(steps)]
        if not np.array_equal(steps[self._offsets[:-1]], np.arange(1, self.total_steps+1)):
            raise ValueError('Missing or duplicate schedule step blocks')
        self._records = read_json(resolve(spec['conditional_records']))
        if not isinstance(self._records, list) or len(self._records) != 2*self.G:
            raise ValueError('Exactly two complete conditional cycles are required')
        self._record_digests = tuple(digest(v) for v in self._records)
        self.identity = digest(spec)
        # Check the entire immutable role schedule before any model is created.
        for cycle in range(1, 4):
            coverage = {'support': set(), 'unsuitability': set()}
            counts = {'support': 0, 'unsuitability': 0}
            for i in range((cycle-1)*self.G+1, cycle*self.G+1):
                step = self.step(i)
                for q in step.queries:
                    for channel in coverage:
                        if getattr(q, channel).observed_mask:
                            coverage[channel].add((q.drug_id, q.disease_id))
                            counts[channel] += 1
            for channel in coverage:
                expected = getattr(corpus, channel)
                if coverage[channel] != expected or counts[channel] != len(expected):
                    raise ValueError('Each complete cycle must cover every positive exactly once')
        if self.G != min(len(corpus.support), len(corpus.unsuitability)):
            raise ValueError('Schedule G must equal the smaller positive channel count')

    def step(self, step):
        if type(step) is not int or not 1 <= step <= self.total_steps:
            raise ValueError('Step outside the sealed 3G schedule')
        start, stop = map(int, self._offsets[step-1:step+1])
        rows = self._frame.slice(start, stop-start).to_dicts()
        queries = []
        for position, row in enumerate(rows):
            if row['step'] != step or row['cycle'] != (step-1)//self.G+1 or row['position'] != position:
                raise ValueError('Original intra-group row order changed')
            pair, role = (row['drug_id'], row['disease_id']), row['role']
            if role not in ('support', 'unsuitability', 'U'):
                raise ValueError('Unknown training role: ' + str(role))
            self.corpus.assert_training_pair(pair, role)
            queries.append(make_unlabeled_query_supervision(*pair) if role == 'U' else self.corpus.positive(pair, role))
        queries = tuple(queries)
        record = None if step <= self.G else deepcopy(self._records[step-self.G-1])
        if record is not None:
            if digest(record) != self._record_digests[step-self.G-1]:
                raise ValueError('Conditional plan changed in memory')
            for pairs in record['disease_unlabeled_pairs'].values():
                for pair in pairs:
                    self.corpus.assert_training_pair(tuple(pair), 'U')
        validate_step_record(queries, record, step=step, G=self.G)
        identity = dict(corpus=self.corpus.identity, sampling=self.identity,
                        training_seed=self.corpus.training_seed, member_digest=digest([asdict(q) for q in queries]))
        content = dict(step=step, G=self.G, queries=[asdict(q) for q in queries], record=record, identity=identity)
        return StepSpec(step, self.G, queries, record, identity, digest(content))
