import math
import pytest
from evi_hgt_phase98.popularity import Popularity, member_digest

def test_popularity_unique_training_pairs_and_heads():
    s=[('d1','s1'),('d1','s1'),('d1','s2')];c=[('d2','s1')]
    model=Popularity.build(cell='R0F0',support=s,contra=c,
        expected_support_digest=member_digest(s),expected_contra_digest=member_digest(c),
        forbidden_pairs={('d3','s1')})
    assert model.score('d1','support')==math.log1p(2)
    assert model.score('d1','contra')==0
    assert model.score('d2','contra')==math.log1p(1)
    assert model.score('unknown','support')==0

def test_popularity_rejects_membership_and_role_errors():
    with pytest.raises(ValueError):
        Popularity.build(cell='R0F0',support=[('d','s')],contra=[],
            expected_support_digest='bad',expected_contra_digest=member_digest([]),forbidden_pairs=set())
    with pytest.raises(ValueError):
        Popularity.build(cell='R0F0',support=[('d','s')],contra=[],
            expected_support_digest=member_digest([('d','s')]),expected_contra_digest=member_digest([]),
            forbidden_pairs={('d','s')})
