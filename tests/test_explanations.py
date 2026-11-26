"""Explanation exports must identify the paths and evidence of a real forward."""
from dataclasses import replace
import importlib
import json

import pytest
import torch

from data_fixture import make_corpus
from evi_hgt.data.corpus import load_corpus
from evi_hgt.model import build_model
from evi_hgt.models.path_reasoning import PathReasoningConfig
from evi_hgt.sampling.path_schema import pack_path_candidates
from evi_hgt.training.supervision import make_unlabeled_query_supervision


def exporter():
    try:
        module = importlib.import_module("evi_hgt.explanations")
    except ModuleNotFoundError:
        pytest.fail("The public model is missing its exact-forward explanation exporter")
    assert callable(getattr(module, "build_query_explanation", None))
    return module.build_query_explanation


@pytest.fixture
def real_forward(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    corpus = load_corpus(make_corpus(tmp_path / "corpus"))
    config = replace(corpus.model_config, hidden_dim=16, evidence_dim=8,
                     relation_dim=4, provenance_dim=4, dropout=0.0, return_logic_state=True)
    built = build_model(config, pretrained=corpus.pretrained, seed=42,
                        path_config=PathReasoningConfig(core_hidden_dim=16, path_hidden_dim=16, dropout=0.0))
    built.model.eval()
    graph = corpus.assembler.assemble(make_unlabeled_query_supervision("D0", "S2"), domain="validation")
    with torch.no_grad():
        prediction = graph.forward(built)
    assert graph.batch.path_batch.num_paths > 0
    yield corpus, graph, built, prediction
    torch.set_num_threads(previous)


def export(corpus, graph, prediction, **kwargs):
    return exporter()(corpus, graph, prediction, drug_id="D0", disease_id="S2", domain="validation", **kwargs)


def test_export_uses_exact_forward_scores_paths_and_authentic_provenance(real_forward):
    corpus, graph, built, prediction = real_forward
    state = torch.random.get_rng_state().clone()
    result = export(corpus, graph, prediction)
    assert result["schema"] == "EVIHGT_QUERY_EXPLANATION_V1"
    assert (result["drug_id"], result["disease_id"], result["domain"]) == ("D0", "S2", "validation")
    assert result["support_logit"] == float(prediction.support_logit[0])
    assert result["unsuitability_logit"] == float(prediction.unsuitability_logit[0])
    assert result["support_score"] == float(prediction.support_score[0])
    assert result["unsuitability_score"] == float(prediction.unsuitability_score[0])
    assert result["scores_are_calibrated_probabilities"] is False
    assert result["model_explanation_is_causal_proof"] is False
    assert result["query_logic_masses"] == {key: float(getattr(prediction, "m_" + key)[0]) for key in "TFBU"}
    assert result["path_available"] is True
    assert result["path_gate"] == float(prediction.path_gate[0])
    assert result["total_forward_paths"] == graph.batch.path_batch.num_paths
    assert result["paths"]
    packed = graph.batch.path_batch
    source = graph.batch.path_source
    for path in result["paths"]:
        index = packed.path_ids.index(path["path_id"])
        length = int(packed.path_length[index])
        assert path["path_score"] == float(prediction.path_reasoning.path_score[index])
        assert path["beta"] == float(prediction.path_reasoning.beta[index])
        assert list(path["step_weights"]) == prediction.path_reasoning.step_weights[index, :length].tolist()
        facts = [source.facts[int(f)].canonical_fact_id for f in packed.step_fact_index[index, :length]]
        assert list(path["canonical_fact_ids"]) == facts
        for edge, fact_id in zip(path["edges"], facts, strict=True):
            records = corpus.index.evidence_for_facts([fact_id]).to_dicts()
            assert edge["evidence_records"] == records
            assert list(edge["provenance"]) == ["SYNTHETIC_TEST_ONLY"]
            fact_index = next(f.fact_index for f in source.facts if f.canonical_fact_id == fact_id)
            available = bool(prediction.edge_logic_state.available[fact_index])
            assert edge["logic_state_available"] is available
            expected = prediction.edge_logic_state.masses[fact_index].tolist() if available else None
            assert edge["logic_masses"] is None if expected is None else list(edge["logic_masses"]) == expected
    assert [p["beta"] for p in result["paths"]] == sorted((p["beta"] for p in result["paths"]), reverse=True)
    json.dumps(result, allow_nan=False)
    assert torch.equal(state, torch.random.get_rng_state())
    assert built.model.training is False


def test_empty_forward_paths_remain_empty_with_zero_gate(real_forward):
    corpus, graph, built, _ = real_forward
    paths = pack_path_candidates((), num_queries=1, max_length=4)
    graph = replace(graph, batch=replace(graph.batch, path_batch=paths))
    with torch.no_grad():
        prediction = graph.forward(built)
    result = export(corpus, graph, prediction)
    assert result["paths"] == []
    assert result["total_forward_paths"] == 0
    assert result["path_available"] is False
    assert result["path_gate"] == 0.0
    assert result["path_status"] == "NO_LEGAL_FORWARD_PATHS"


def test_unreturned_logic_stays_unavailable_without_inventing_unknown_mass(real_forward):
    corpus, graph, built, _ = real_forward
    built.model.encoder.config = replace(built.model.encoder.config, return_logic_state=False)
    with torch.no_grad():
        prediction = graph.forward(built)
    assert prediction.edge_logic_state is None
    result = export(corpus, graph, prediction)
    for path in result["paths"]:
        for edge in path["edges"]:
            assert edge["logic_state_available"] is False
            assert edge["logic_masses"] is None


def test_disabled_path_branch_has_no_model_path_scores(real_forward):
    corpus, graph, original, _ = real_forward
    built = build_model(original.model.encoder.config, pretrained=corpus.pretrained,
                        seed=42, variant="NO_PATH", path_config=original.model.path_reasoner.config)
    built.model.eval()
    with torch.no_grad():
        prediction = graph.forward(built)
    assert graph.batch.path_batch.num_paths > 0
    assert prediction.path_reasoning is None
    result = export(corpus, graph, prediction)
    assert result["paths"] == []
    assert result["path_status"] == "PATH_BRANCH_DISABLED"
    assert result["path_available"] is False
    assert result["path_gate"] is None
    assert result["total_forward_paths"] == 0
    assert result["total_candidate_paths"] == graph.batch.path_batch.num_paths
    assert result["support_score"] == float(prediction.support_score[0])


@pytest.mark.parametrize("damage", ["identity", "direction", "query", "forward_count", "corpus", "relation"])
def test_mismatched_forward_alignment_is_refused(real_forward, damage):
    corpus, graph, _, prediction = real_forward
    paths = graph.batch.path_batch
    if damage == "identity":
        paths = replace(paths, path_ids=("wrong-id",) + paths.path_ids[1:])
    elif damage == "direction":
        directions = paths.step_direction.clone()
        directions[0, 0] = 1 - directions[0, 0]
        paths = replace(paths, step_direction=directions)
    elif damage == "query":
        source = replace(graph.batch.path_source, query_pair_id="wrong-query")
        graph = replace(graph, batch=replace(graph.batch, path_source=source))
    elif damage == "corpus":
        source = replace(graph.batch.path_source, phase3_digest="wrong-corpus")
        graph = replace(graph, batch=replace(graph.batch, path_source=source))
    elif damage == "relation":
        source = graph.batch.path_source
        index = int(paths.step_fact_index[0, 0])
        facts = list(source.facts)
        facts[index] = replace(facts[index], relation="INVENTED_RELATION")
        graph = replace(graph, batch=replace(graph.batch, path_source=replace(source, facts=tuple(facts))))
    else:
        prediction = replace(prediction, path_reasoning=replace(prediction.path_reasoning,
                             beta=prediction.path_reasoning.beta[:0]))
    graph = replace(graph, batch=replace(graph.batch, path_batch=paths))
    with pytest.raises(ValueError):
        export(corpus, graph, prediction)


def test_top_k_limits_existing_paths_and_rejects_zero(real_forward):
    corpus, graph, _, prediction = real_forward
    result = export(corpus, graph, prediction, top_k=1)
    assert len(result["paths"]) == 1
    assert result["paths"][0]["beta"] == float(prediction.path_reasoning.beta.max())
    with pytest.raises(ValueError, match="top_k"):
        export(corpus, graph, prediction, top_k=0)
