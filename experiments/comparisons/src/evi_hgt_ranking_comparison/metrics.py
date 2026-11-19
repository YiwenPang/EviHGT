"""Versioned CPU metrics. No model, dataset loader, RNG, or prediction dependency.

AP integrates a right-continuous step curve; AUPR integrates adjacent group
endpoints including (recall=0,precision=1). With all scores tied, AP=prevalence
and AUPR=(1+prevalence)/2. These are different definitions, not a numerical bug.
"""
from .protocol_ids import METRICS_SCHEMA
from collections import defaultdict
import math
import statistics
from .common import digest

METRIC_SCHEMA_VERSION=METRICS_SCHEMA
SCORE_FIELDS={'support':'support_logit','contra':'unsuitability_logit'}
RANKING_KEYS=tuple(f'{name}@{k}' for k in (5,10) for name in ('P','R','Hits','NDCG'))

def _members(rows):
    return [{k:v for k,v in r.items() if k not in ('support_logit','unsuitability_logit')} for r in rows]

def threshold_metrics(positive,unlabeled):
    """Exact-score group integration; a missing class is a natural skip."""
    p,u=list(positive),list(unlabeled)
    if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in p+u):
        raise ValueError('Invalid/nonfinite score is a run failure, never a skip')
    if not p or not u:return dict(status='SKIP',reason='NO_P' if not p else 'NO_U')
    groups=defaultdict(lambda:[0,0])
    for x in p:groups[x][0]+=1
    for x in u:groups[x][1]+=1
    tp=fp=0;last_r=last_f=0.;last_p=1.;auc=ap=aupr=0.
    for score in sorted(groups,reverse=True):
        gp,gu=groups[score];tp+=gp;fp+=gu
        r=tp/len(p);f=fp/len(u);precision=tp/(tp+fp)
        auc+=(f-last_f)*(r+last_r)/2
        ap+=(r-last_r)*precision
        aupr+=(r-last_r)*(precision+last_p)/2
        last_r,last_f,last_p=r,f,precision
    return dict(status='EVALUABLE',auc=auc,ap=ap,aupr_trapezoid=aupr)

def _ranking(rows,score_field,P):
    ranked=sorted(rows,key=lambda r:(-r[score_field],r['drug_id']))
    out={};random_reference={};N=len(rows)
    for k in (5,10):
        eff=min(k,N);top=ranked[:eff];hits=sum(r['role']=='P' for r in top)
        dcg=sum(1/math.log2(i+1.) for i,r in enumerate(top,start=1) if r['role']=='P')
        idcg=sum(1/math.log2(i+1.) for i in range(1,min(P,eff)+1))
        out.update({f'P@{k}':hits/eff,f'R@{k}':hits/P,f'Hits@{k}':float(hits>0),f'NDCG@{k}':dcg/idcg,f'effective_k_{k}':eff})
        # Uniform random permutation of this fixed candidate set, not clinical negatives.
        random_reference.update({f'expected_precision_at_{k}':P/N,f'expected_recall_at_{k}':eff/N,
            f'expected_hits_at_{k}':1-(math.comb(N-P,eff)/math.comb(N,eff) if eff<=N-P else 0),
            f'top_{k}_hits_guaranteed':eff>N-P})
    out['random_reference']=random_reference
    return out

def evaluate_panel(rows,*,head='support',expected_members=None,scope='AUTHORIZED_PANEL_CPU_STATISTICS'):
    """Rows must be complete and have a canonical pair, P/U role and raw score.

    expected_members pins the ordered, fully specified input manifest; missing
    predictions therefore fail before any disease skip logic. Callers binding
    external caches must also validate their checkpoint/source identities.
    """
    if head not in SCORE_FIELDS:raise ValueError('Unknown independent head')
    rows=list(rows);score=SCORE_FIELDS[head]
    if not rows:raise ValueError('Empty panel is not a successful evaluation')
    if expected_members is not None and _members(rows)!=list(expected_members):
        raise ValueError('Missing/reordered/foreign predictions or member fields')
    groups=defaultdict(list);seen=set()
    for r in rows:
        if any(not isinstance(r.get(k),str) or not r[k].strip() for k in ('drug_id','disease_id')):raise ValueError('Canonical IDs required')
        if r.get('role') not in ('P','U'):raise ValueError('Verified P/U role required')
        value=r.get(score)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Missing/nonfinite raw score')
        pair=(r['drug_id'],r['disease_id'])
        if pair in seen:raise ValueError('Duplicate candidate pair')
        seen.add(pair);groups[r['disease_id']].append(r)
    local=[];skipped=[]
    for disease,candidates in sorted(groups.items()):
        p=[r[score] for r in candidates if r['role']=='P'];u=[r[score] for r in candidates if r['role']=='U']
        base=dict(disease_id=disease,P=len(p),U=len(u),N=len(candidates))
        metric=threshold_metrics(p,u)
        if metric['status']=='SKIP':skipped.append(dict(base,reason=metric['reason']));continue
        local.append(dict(base,**{k:metric[k] for k in ('auc','ap','aupr_trapezoid')},**_ranking(candidates,score,len(p))))
    mean=lambda k:statistics.fmean(r[k] for r in local) if local else None
    return dict(metric_schema_version=METRIC_SCHEMA_VERSION,status='COMPLETE' if local else 'NO_EVALUABLE_DISEASES',
        scope=scope,head=head,macro_auc=mean('auc'),macro_ap=mean('ap'),macro_aupr_trapezoid=mean('aupr_trapezoid'),
        evaluable_diseases=len(local),skipped_diseases=len(skipped),pairs=len(rows),P=sum(r['role']=='P' for r in rows),U=sum(r['role']=='U' for r in rows),
        ranking={k:mean(k) for k in RANKING_KEYS},disease_metrics=local,skipped_disease_details=skipped,
        candidate_identity_sha256=digest(sorted((r['drug_id'],r['disease_id'],r['role']) for r in rows)),
        AP_definition='threshold-group sum(delta recall * group-end precision)',
        AUPR_definition='trapezoid between all exact-score threshold-group endpoints; initial (R,P)=(0,1); no envelope',
        semantics='P_VS_U_NOT_CONFIRMED_NEGATIVE',Hits_interpretation='Fixed candidate pool; N<=K or P>N-effective_k can make Hits guaranteed, not a model success validation')

def metric_pair(support_rows,auxiliary_rows):
    support_rows=list(support_rows);auxiliary_rows=list(auxiliary_rows)
    contra=[dict(r,role='P') for r in auxiliary_rows]+[r for r in support_rows if r['role']=='U']
    return dict(support=evaluate_panel(support_rows),contra=evaluate_panel(contra,head='contra'))
