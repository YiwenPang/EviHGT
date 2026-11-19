"""In-memory G4 cache assembly, no protected I/O or model calls.

Production callers must pass a separately authorized G4 capability. This module
checks before examining any supplied scores. Tests use only TEST_ONLY identities.
"""
from dataclasses import dataclass,asdict
import math
from .common import digest

IDENTITY_KEYS=('mode','method','cell','weight_sha256','checkpoint_identity','head','graph_sha256','mask_sha256','feature_sha256','numeric_identity','scoring_source_sha256')

@dataclass
class ScoringCounts:
    result_panel_count:int=0
    new_source_scoring_passes:int=0
    cache_reuse_count:int=0
    actual_model_forward_calls:int=0
    actual_query_scores:int=0

    def add_source_call(self,*,forward_calls,query_scores):
        if type(forward_calls) is not int or type(query_scores) is not int or min(forward_calls,query_scores)<0:raise ValueError('Explicit nonnegative actual counts required')
        self.new_source_scoring_passes+=1;self.actual_model_forward_calls+=forward_calls;self.actual_query_scores+=query_scores

def _identity(identity):
    if set(identity)!=set(IDENTITY_KEYS):raise ValueError('Complete versioned score identity required')
    if any(v is None or v=='' or v=={} for v in identity.values()):raise ValueError('Empty identity component')
    if identity['head'] not in ('support','contra'):raise ValueError('Unknown head')
    digest(identity)

def members(rows):return [{k:r[k] for k in ('drug_id','disease_id','role')} for r in rows]

def make_cache(rows,identity,source_domain):
    _identity(identity);rows=[dict(r) for r in rows];seen=set()
    for r in rows:
        if set(r)!= {'drug_id','disease_id','role','score'} or r['role'] not in ('P','U'):raise ValueError('Exact score row schema required')
        pair=(r['drug_id'],r['disease_id'])
        if pair in seen or any(not isinstance(v,str) or not v for v in pair):raise ValueError('Duplicate or invalid candidate')
        if isinstance(r['score'],bool) or not isinstance(r['score'],(float,int)) or not math.isfinite(r['score']):raise ValueError('Missing/nonfinite score')
        seen.add(pair)
    return dict(schema='RANKING_COMPARISON_BOUND_SCORE_CACHE_V1',status='COMPLETE',identity=dict(identity),source_domain=source_domain,
        rows=rows,rows_sha256=digest(rows),ordered_members_sha256=digest(members(rows)),count=len(rows))

def _verify(cache,expected_identity,domain,expected_members,role):
    if cache.get('schema')!='RANKING_COMPARISON_BOUND_SCORE_CACHE_V1' or cache.get('status')!='COMPLETE':raise ValueError('Incomplete/foreign cache')
    if cache['identity']!=expected_identity or cache['source_domain']!=domain:raise ValueError('Weight/cell/head/mask/feature/numeric/source/domain identity mismatch')
    fresh=make_cache(cache['rows'],cache['identity'],cache['source_domain'])
    if fresh!=cache or members(cache['rows'])!=list(expected_members):raise ValueError('Cache bytes, candidate set, order or score count changed')
    if any(r['role']!=role for r in cache['rows']):raise ValueError('Source role differs')

def assemble_contra(positive_cache,unlabeled_cache,*,expected_identity,expected_p_members,expected_u_members,test_only=False,g4_capability=None,source_executions=None):
    # No existence/read of protected files is performed here, even on rejection.
    if test_only:
        if expected_identity.get('mode')!='TEST_ONLY':raise PermissionError('Synthetic mode cannot authorize real G4 data')
    else:
        from .authorization import ExecutionCapability
        if not isinstance(g4_capability,ExecutionCapability):raise PermissionError('Independent exact-identity G4 permission required')
        g4_capability.permits_score(expected_identity)
    _identity(expected_identity)
    if expected_identity['head']!='contra':raise ValueError('Contra requires unsuitability scores, never support complement')
    _verify(positive_cache,expected_identity,'auxiliary_probe',expected_p_members,'P')
    _verify(unlabeled_cache,expected_identity,'validation_U',expected_u_members,'U')
    rows=positive_cache['rows']+unlabeled_cache['rows']
    if len({(r['drug_id'],r['disease_id']) for r in rows})!=len(rows):raise ValueError('P/U overlap or duplicate candidate')
    # A newly scored source is not charged again as historical cache reuse.
    # Actual counts must come from the scoring executor; assembly never invents
    # a forward count by multiplying heads or the size of a loaded cache.
    source_executions={} if source_executions is None else source_executions
    if set(source_executions)-{'positive','unlabeled'}:raise ValueError('Unknown source execution')
    counters=ScoringCounts(result_panel_count=1,cache_reuse_count=2-len(source_executions))
    for role,execution in source_executions.items():
        if set(execution)!={'new_source_scoring_passes','actual_model_forward_calls','actual_query_scores'}:raise ValueError('Explicit actual source counts required')
        if execution['new_source_scoring_passes']!=1:raise ValueError('One completed source pass expected')
        source=positive_cache if role=='positive' else unlabeled_cache
        if execution['actual_query_scores']<len(source['rows']):raise ValueError('Actual scored query count cannot omit source rows')
        counters.add_source_call(forward_calls=execution['actual_model_forward_calls'],query_scores=execution['actual_query_scores'])
    return dict(schema='RANKING_COMPARISON_AUX_PROBE_P_VALIDATION_U_V1',identity=expected_identity,rows=rows,rows_sha256=digest(rows),
        candidate_order_sha256=digest(members(rows)),counts=asdict(counters),role='AUXILIARY_PROBE_NOT_INDEPENDENT_CONTRA_TEST')
