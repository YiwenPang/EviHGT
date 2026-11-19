"""Pure grant validation before data access; Part2 creates no effective grant.

An approved digest is a trusted launch input supplied after independent human
approval. A digest proves equality, never consent or authenticity by itself.
"""
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from .common import digest,ROOT,read,sha

_ISSUER=object()

V6_SCHEMA='PHASE98_PENDING_EXECUTION_MANIFEST_V6'
APPROVED_V6=ROOT/'outputs/phase9/phase98/part2_4/PHASE98_PENDING_EXECUTION_MANIFEST_V6.json'
APPROVED_V6_SHA='f744d85dd601f2d590ccc74d26a1e3a88b1738a83799cd18da9244c2af71e489'

def validate_v6_scope(manifest,role):
    """Part3 permits metadata rebinding only; compare every scientific field.

    This is format/scope validation, not a grant. The caller must still verify
    the explicit grant and its launch-authority digest before issuing capability.
    """
    if manifest.get('schema')!=V6_SCHEMA or role!='G3':
        raise PermissionError('Only the approved V6 G3 execution format is supported')
    if sha(APPROVED_V6)!=APPROVED_V6_SHA:
        raise ValueError('Original approved V6 bytes changed')
    original=read(APPROVED_V6)
    sources=manifest.get('source_sha256',{})
    if not isinstance(sources,dict) or not set(original['source_sha256']).issubset(sources):
        raise ValueError('V6 source roster cannot lose registered dependencies')
    if manifest.get('approval_base')!={'path':str(APPROVED_V6),'sha256':APPROVED_V6_SHA}:
        raise PermissionError('Exact immutable V6 approval reference required')
    for key in ('outer_eval_authorized','final_eval_authorized','research_diagnostic_forward_authorized',
                'G4_authorized','G5_authorized','diagnostics_authorized'):
        if manifest.get(key) is not False:raise PermissionError('Scope remains closed: '+key)
    if manifest.get('max_model_workers')!=1 or manifest.get('G3_authorized') is not True:
        raise PermissionError('One worker and explicit G3 scope required')
    expected=deepcopy(original)
    # Only these launch metadata fields may differ. Data/mask/feature/sampling
    # digests, historical reuse identities and all budgets are compared intact.
    metadata=('created_utc','enabled','grant_id','formal_training_authorized','G3_authorized',
              'source_sha256','source_closure','digest','launch_status','qualification_stop_current')
    for key in metadata:
        if key not in manifest:raise ValueError('Missing V6 field: '+key)
        expected[key]=manifest[key]
    expected['approval_base']=manifest['approval_base']
    for key in ('training_tasks','development_tasks'):
        actual=manifest.get(key,[])
        if len(actual)!=len(expected[key]):raise ValueError('Changed V6 roster: '+key)
        for before,after in zip(expected[key],actual):
            for field in ('source_digest','source_closure','launch_status'):
                if field not in after:raise ValueError('Missing task field: '+field)
                before[field]=after[field]
            before['training_authorized']=key=='training_tasks'
            before['score_authorized']=True
    # Planning and historical reuse rows are retained verbatim as the original
    # plan. Execution permission lives in the exact training/development lists.
    for row in expected['g3_evaluation_tasks']:row['authorized']=True
    if expected!=manifest:raise ValueError('V6 scientific content or G3-only scope differs from approved plan')
    if manifest['digest']!=digest({k:v for k,v in manifest.items() if k!='digest'}):
        raise ValueError('V6 execution content digest differs')

@dataclass(frozen=True)
class ExecutionCapability:
    role:str
    grant_id:str
    manifest_digest:str
    runs:tuple
    score_identities:tuple
    issuer:object
    development_runs:tuple=()

    def permits(self,identity,root):
        if self.issuer is not _ISSUER or self.role!='G3':raise PermissionError('G3 capability required')
        actual={k:identity.get(k) for k in ('run_id','method','cell','source_digest','data_digest')}
        matches=[r for r in self.runs if all(r.get(k)==v for k,v in actual.items())]
        if len(matches)!=1 or Path(matches[0]['output_root']).resolve()!=Path(root).resolve():raise PermissionError('Grant run/source/data/output identity differs')

    def permits_score(self,identity):
        if self.issuer is not _ISSUER or self.role!='G4' or digest(identity) not in self.score_identities:raise PermissionError('Exact G4 scoring identity not authorized')

    def permits_development(self,identity,root):
        if self.issuer is not _ISSUER or self.role!='G3':raise PermissionError('G3 development capability required')
        actual={k:identity.get(k) for k in ('run_id','method','cell','source_digest','data_digest')}
        rows=[r for r in self.runs+self.development_runs if all(r.get(k)==v for k,v in actual.items())]
        if len(rows)!=1 or Path(rows[0]['output_root']).resolve()!=Path(root).resolve():raise PermissionError('Exact development run/data/source/output not authorized')

def require_formal_execution(manifest,grant,role,*,approved_grant_digest=None):
    if role not in ('G3','G4','G5'):
        raise PermissionError('Research diagnostics need an independent future authorization')
    if not manifest.get('enabled') or not manifest.get('grant_id'):
        raise PermissionError('Phase98 formal execution disabled; no dataset may be opened')
    if manifest['grant_id']=='G3_GRANT_000001':
        raise PermissionError('Historical Phase93 grant cannot authorize Phase98')
    if manifest.get('schema')=='PHASE98_SUBMISSION_CV5_MANIFEST_V1':
        from .submission_authorization import validate_submission_cv5_scope
        validate_submission_cv5_scope(manifest,role)
    else:
        validate_v6_scope(manifest,role)
    if not approved_grant_digest or not grant or digest(grant)!=approved_grant_digest:
        raise PermissionError('Independent human approval digest must be supplied by the launch authority')
    if grant.get('schema')!='PHASE98_EXPLICIT_EXECUTION_GRANT_V1' or grant.get('id')!=manifest['grant_id'] or grant.get('role')!=role:
        raise PermissionError('Wrong version/role/grant identity')
    if grant.get('manifest_digest')!=digest(manifest) or not grant.get('human_approval_reference'):
        raise PermissionError('Grant lacks exact manifest binding or approval evidence')
    flag={'G3':'formal_training_authorized','G4':'outer_eval_authorized','G5':'final_eval_authorized'}[role]
    if manifest.get(flag) is not True or grant.get('authorized') is not True:raise PermissionError('Requested role not authorized')
    runs=manifest.get('training_tasks',[])
    development=manifest.get('development_tasks',[])
    if role=='G3':
        if not runs or grant.get('run_ids')!=[r['run_id'] for r in runs]:raise PermissionError('Exact ordered run roster required')
        if len({r['run_id'] for r in runs})!=len(runs):raise PermissionError('Duplicate formal run')
        base=(ROOT/'outputs/phase9/phase98/runs').resolve()
        if development and grant.get('development_run_ids')!=[r['run_id'] for r in development]:raise PermissionError('Exact G3 development roster required')
        if len({r['run_id'] for r in runs+development})!=len(runs)+len(development):raise PermissionError('Duplicate training/development run')
        for row in runs+development:
            if any(not row.get(k) for k in ('run_id','method','cell','source_digest','data_digest','output_root')):raise PermissionError('Incomplete run binding')
            path=Path(row['output_root']).resolve()
            if path==base or not path.is_relative_to(base):raise PermissionError('Formal output must be a new Phase98 run')
    scores=manifest.get('score_identity_digests',[])
    if role=='G4' and (not scores or grant.get('score_identity_digests')!=scores):raise PermissionError('Exact G4 scoring identity roster required')
    return ExecutionCapability(role,grant['id'],digest(manifest),tuple(deepcopy(runs)),tuple(scores),_ISSUER,tuple(deepcopy(development)))

def validate_pending(manifest):
    if manifest.get('enabled') is not False or manifest.get('grant_id') is not None:
        raise ValueError('Pending manifest must be disabled and grant-free')
    for key in ('formal_training_authorized','outer_eval_authorized','final_eval_authorized','research_diagnostic_forward_authorized'):
        if manifest.get(key) is not False:raise ValueError('Missing explicit denied scope: '+key)
    return digest(manifest)
