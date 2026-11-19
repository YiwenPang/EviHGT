"""Synthetic, model-free tests for the post-completion analysis package."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_results as a


def test_unblinding_missing_is_closed(tmp_path):
    with pytest.raises((AssertionError, FileNotFoundError)):
        a.require_unblinding(tmp_path)


def gate_files(root, *, status="PASS", authorized=True):
    folder=root/"outputs/phase843_results";folder.mkdir(parents=True)
    closure={"status":status,"formal_run_id":a.RUN_ID,"formal_test_count":1,
             "execution_plan_digest":a.PLAN_DIGEST,"manifest_bindings":a.MANIFESTS}
    (folder/"closure_integrity.json").write_text(json.dumps(closure))
    marker={"authorized":authorized,"closure_integrity_status":status,
            "closure_integrity_sha256":a.sha(folder/"closure_integrity.json"),
            "formal_run_id":a.RUN_ID,"execution_plan_digest":a.PLAN_DIGEST,
            "manifest_bindings":a.MANIFESTS,"timestamp_utc":"2026-10-03T00:00:00+00:00"}
    (folder/"FIRST_RESULTS_UNBLINDING.json").write_text(json.dumps(marker))
    return folder


def test_unblinding_gate_pass(tmp_path):
    gate_files(tmp_path)
    assert a.require_unblinding(tmp_path)[0]["authorized"] is True


@pytest.mark.parametrize("change",["hash","authorization","state","plan","manifest"])
def test_unblinding_mismatch_rejected(tmp_path,change):
    folder=gate_files(tmp_path)
    marker=json.loads((folder/"FIRST_RESULTS_UNBLINDING.json").read_text())
    if change=="hash":marker["closure_integrity_sha256"]="0"*64
    if change=="authorization":marker["authorized"]=False
    if change=="state":marker["closure_integrity_status"]="FAILED"
    if change=="plan":marker["execution_plan_digest"]="x"
    if change=="manifest":marker["manifest_bindings"]={}
    (folder/"FIRST_RESULTS_UNBLINDING.json").write_text(json.dumps(marker))
    with pytest.raises(AssertionError):a.require_unblinding(tmp_path)


def test_tied_threshold_auc_ap_and_canonical_rank():
    out,roc,pr=a.recompute("D",["z","a","u"],[[1,99],[1,2],[0,0]],{"z"},[5,10],"standard")
    assert out["roc_auc"]==0.75
    assert out["average_precision"]==0.5
    assert out["P@5"]==1/3
    assert out["R@5"]==1
    assert out["Hits@5"]==1
    assert out["NDCG@5"]==1/np.log2(3)
    assert len(roc)==len(pr)==101
    assert out["ordered_ids"]==["a","z","u"]


def test_zero_positive_skip_never_imputed_to_zero():
    out,roc,pr=a.recompute("D",["a","b"],[[3,4],[5,6]],set(),[5,10],"standard")
    assert out["evaluable"] is False
    assert out["skip_reason"]=="NO_P_OR_NO_PROTOCOL_U"
    assert out["roc_auc"] is out["average_precision"] is out["P@5"] is None
    assert roc is pr is None


def test_metrics_match_is_exact_and_fails_on_mutation():
    out,_,_=a.recompute("D",["a","u"],[[1,9],[0,8]],{"a"},[5,10],"standard")
    saved={"disease_id":"D","state":"COMPLETE","pairs":2,"evaluable":True,
           "ranking":out["ranking_payload"],"secondary":{"roc_auc":1.,"average_precision":1.,"reason":""}}
    a.verify_query_metrics(saved,out,[5,10])
    saved["secondary"]["average_precision"]-=1e-12
    with pytest.raises(AssertionError):a.verify_query_metrics(saved,out,[5,10])


def test_paired_bootstrap_deterministic_and_query_level():
    d=np.array([[1.,0.],[0.,-1.],[-1.,1.],[0.5,0.5]])
    x=a.paired_bootstrap(d,seed=20261003,resamples=1000,chunk=17)
    y=a.paired_bootstrap(d,seed=20261003,resamples=1000,chunk=64)
    np.testing.assert_array_equal(x,y)
    assert x.shape==(2,2)
    assert np.all(x[0]<=x[1])


def test_common_membership_rejects_missing_query():
    rows=[dict(model="FULL_EVIHGT",query_id="a",evaluable=True),
          dict(model="BASE_HGT",query_id="b",evaluable=True)]
    with pytest.raises(AssertionError):a.assert_common_membership(rows,["FULL_EVIHGT","BASE_HGT"])


def test_common_membership_deterministic_order_and_skip():
    rows=[dict(model=m,query_id=d,evaluable=e) for m in ["M1","M2"] for d,e in [("b",True),("a",True),("z",False)]]
    assert a.assert_common_membership(rows,["M1","M2"])==["a","b"]


def test_topk_tie_break_and_sealed_match():
    ids=["z","a","b"]
    assert a.verify_topk(ids,[1,1,0],["a","z","b"])==["a","z","b"]
    with pytest.raises(AssertionError):a.verify_topk(ids,[1,1,0],["z","a","b"])


def test_no_negative_relabeling_or_model_invocation():
    source=Path(a.__file__).read_text(encoding="utf-8")
    assert "make_unlabeled_query_supervision" not in source
    assert "load_group" not in source
    assert "torch" not in source
    assert "NOT_SCIENTIFICALLY_DEFINED" in source
    assert "NOT_CONFIRMED_NEGATIVE" in source


def test_exclusive_artifact_write(tmp_path):
    p=tmp_path/"x.json"
    a.write_json(p,{"x":1})
    with pytest.raises(FileExistsError):a.write_json(p,{"x":2})
    assert json.loads(p.read_text())=={"x":1}
