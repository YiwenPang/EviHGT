import math
import pytest
from evi_hgt_phase98.metrics import evaluate_panel

def rows(p=1.,u=0.):
    return [dict(drug_id='A',disease_id='D',role='P',support_logit=p),dict(drug_id='B',disease_id='D',role='U',support_logit=u)]

@pytest.mark.parametrize('p,u,expected',[(1.,0.,(1.,1.,1.)),(0.,1.,(0.,.5,.25)),(1.,1.,(.5,.5,.75))])
def test_hand_cases(p,u,expected):
    r=evaluate_panel(rows(p,u))
    assert tuple(r[k] for k in ('macro_auc','macro_ap','macro_aupr_trapezoid'))==expected

def test_ties_reorder_and_unrounded_scores():
    data=rows(1.,1.)+[dict(drug_id='C',disease_id='D',role='U',support_logit=1.)]
    a=evaluate_panel(data);b=evaluate_panel(list(reversed(data)))
    assert a==b
    assert a['macro_aupr_trapezoid']==pytest.approx((1+1/3)/2)
    assert evaluate_panel(rows(1.00000000001,1.))['macro_auc']==1

def test_skip_is_not_invalid_prediction():
    p=rows()[:1];u=rows()[1:]
    assert evaluate_panel(p)['skipped_disease_details'][0]['reason']=='NO_U'
    assert evaluate_panel(u)['skipped_disease_details'][0]['reason']=='NO_P'
    for bad in (float('nan'),float('inf'),float('-inf')):
        with pytest.raises(ValueError):evaluate_panel(rows(bad,0.))
    with pytest.raises(ValueError):evaluate_panel(p,expected_members=[{k:v for k,v in r.items() if k!='support_logit'} for r in rows()])
    with pytest.raises(ValueError):evaluate_panel(rows()+rows()[:1])

def test_independent_head_and_effective_k():
    data=[dict(r,unsuitability_logit=-r['support_logit']) for r in rows()]
    a=evaluate_panel(data,head='contra')
    assert a['macro_auc']==0 and a['ranking']['P@5']==.5
    assert a['disease_metrics'][0]['effective_k_5']==2
    assert a['disease_metrics'][0]['random_reference']['expected_hits_at_5']==1

