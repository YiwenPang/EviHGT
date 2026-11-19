"""Check model outputs, gradients and ablations on synthetic graphs."""
from dataclasses import replace

import pytest
import torch

from evi_hgt.model import VARIANTS, build_model
from evi_hgt.models.config import EviHGTConfig
from evi_hgt.models.input_contract import (
    EdgeBatch, EviHGTBatch, NodeBatch, PackedEvidenceBatch, QueryBatch,
)
from evi_hgt.sampling.path_schema import PathCandidate, pack_path_candidates


def config():
    return EviHGTConfig(
        num_node_types=10, num_edge_types=4, num_nodes=8, hidden_dim=16,
        evidence_dim=8, relation_dim=4, provenance_dim=4,
        property_token_buckets=256, evidence_token_buckets=256,
        provenance_token_buckets=256, num_numeric_features=6,
        num_boolean_features=1, dropout=0.0,
    )


class FrozenFeatures:
    def gather(self, node_id, node_type_id, drug_type, disease_type, device):
        values = torch.arange(len(node_id) * 768, device=device).reshape(-1, 768).float() / 1000
        available = (node_type_id == drug_type) | (node_type_id == disease_type)
        return values, available


def fixture(*, paths=True):
    long = lambda values: torch.tensor(values, dtype=torch.long)
    boolean = lambda values: torch.tensor(values, dtype=torch.bool)
    features = torch.zeros(8, 3588)
    for index in (0, 4):
        features[index, 0] = features[index, 2560] = 1
    for index in (3, 7):
        features[index, 2562] = features[index, 3586] = 1
    nodes = NodeBatch(long([4, 6, 6, 3] * 2), long([0] * 4 + [1] * 4), features, long(range(8)))
    edges = EdgeBatch(long([0, 1, 0, 2, 4, 5, 4, 6]), long([1, 3, 2, 3, 5, 7, 6, 7]),
                      long([0, 1, 2, 3] * 2), long(range(8)), long([0] * 4 + [1] * 4))
    evidence = PackedEvidenceBatch(
        8, long([0, 1, 2, 3] * 2), long([0] * 4 + [1] * 4), long(range(8)), long([0, 1, 2, 3] * 2),
        torch.arange(48).float().reshape(8, 6) / 50, torch.ones(8, 6, dtype=torch.bool),
        torch.zeros(8, 1, dtype=torch.bool), torch.ones(8, 1, dtype=torch.bool),
        long(range(1, 9)), long(range(8)), long(range(11, 19)), long(range(8)),
        long(range(21, 29)), long([0] * 8), long([1] * 8), long(range(8)), long([1] * 8),
        boolean([True, False, False, False] * 2), boolean([False, False, False, True] * 2),
        boolean([True, False, True, True] * 2), boolean([True, False, True, True] * 2),
    )
    batch = EviHGTBatch(nodes, edges, evidence, QueryBatch(long([0, 4]), long([3, 7]), long([0, 1])))
    candidates = []
    if paths:
        for query, offset in enumerate((0, 4)):
            candidates.append(PathCandidate.create(
                query_index=query, node_indices=(offset, offset + 1, offset + 3),
                node_ids=(offset, offset + 1, offset + 3), node_types=("DRG", "GEN", "DIS"),
                fact_indices=(offset, offset + 1), canonical_fact_ids=(100 + offset, 101 + offset),
                relations=("SYNTHETIC_DRUG_GENE", "SYNTHETIC_GENE_DISEASE"),
                directions=(0, 0), structural_score=0.0,
            ))
    return batch, pack_path_candidates(candidates, num_queries=2, max_length=4)


def test_factory_is_deterministic_and_does_not_change_rng():
    before = torch.get_rng_state().clone()
    first = build_model(config(), pretrained=FrozenFeatures(), seed=42)
    assert torch.equal(before, torch.get_rng_state())
    second = build_model(config(), pretrained=FrozenFeatures(), seed=42)
    assert first.model.state_dict().keys() == second.model.state_dict().keys()
    assert all(torch.equal(value, second.model.state_dict()[name]) for name, value in first.model.state_dict().items())
    assert [layer.attention.use_logic_mass_conditioning for layer in first.model.encoder.layers] == [False, False]


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_has_finite_independent_heads_and_gradients(variant):
    built = build_model(config(), pretrained=FrozenFeatures(), seed=42, variant=variant)
    batch, paths = fixture()
    result = built.forward(batch, path_batch=paths)
    assert result.support_logit.shape == result.unsuitability_logit.shape == (2,)
    assert not torch.equal(result.support_logit, result.unsuitability_logit)
    torch.testing.assert_close(result.m_T + result.m_F + result.m_B + result.m_U, torch.ones(2))
    loss = result.support_logit.square().sum() + result.unsuitability_logit.square().sum()
    loss.backward()
    gradients = [parameter.grad for parameter in built.model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert built.metadata()["variant"] == variant
    assert bool(result.path_available.any()) == (variant != "NO_PATH")


def test_content_ablation_blocks_evidence_and_provenance_tokens_only():
    built = build_model(config(), pretrained=FrozenFeatures(), seed=42, variant="CONTENT_ZERO")
    built.model.eval()
    batch, paths = fixture()
    changed = replace(batch, evidence=replace(
        batch.evidence, evidence_token_ids=batch.evidence.evidence_token_ids + 100,
        provenance_token_ids=batch.evidence.provenance_token_ids + 100,
    ))
    first = built.forward(batch, path_batch=paths)
    second = built.forward(changed, path_batch=paths)
    assert torch.equal(first.support_logit, second.support_logit)
    assert torch.equal(first.unsuitability_logit, second.unsuitability_logit)


def test_zero_initialized_d1_is_identical_to_no_d1():
    full = build_model(config(), pretrained=FrozenFeatures(), seed=42)
    absent = build_model(config(), pretrained=None, seed=42, variant="NO_D1")
    full.model.eval(); absent.model.eval()
    batch, paths = fixture()
    first = full.forward(batch, path_batch=paths)
    second = absent.forward(batch, path_batch=paths)
    assert torch.equal(first.support_logit, second.support_logit)
    assert torch.equal(first.unsuitability_logit, second.unsuitability_logit)


def test_side_ablation_is_invariant_to_valid_side_features():
    built = build_model(config(), pretrained=FrozenFeatures(), seed=42, variant="NO_SIDE")
    built.model.eval()
    batch, paths = fixture()
    changed = replace(batch, nodes=replace(batch.nodes, features=torch.zeros_like(batch.nodes.features)))
    first = built.forward(batch, path_batch=paths)
    second = built.forward(changed, path_batch=paths)
    assert torch.equal(first.support_logit, second.support_logit)
    assert torch.equal(first.unsuitability_logit, second.unsuitability_logit)


def test_missing_paths_and_pretrained_store_fail_explicitly():
    with pytest.raises(ValueError, match="pretrained"):
        build_model(config(), pretrained=None, seed=42)
    built = build_model(config(), pretrained=FrozenFeatures(), seed=42)
    batch, _ = fixture()
    with pytest.raises(ValueError, match="path"):
        built.forward(batch, path_batch=None)


def test_constant_quality_preserves_unavailable_mask():
    built = build_model(replace(config(), return_logic_state=True), pretrained=FrozenFeatures(),
                        seed=42, variant="FACT_QUALITY_CONSTANT")
    batch, paths = fixture()
    logic = built.forward(batch, path_batch=paths).edge_logic_state
    assert torch.equal(logic.available, batch.evidence.fact_support_available & batch.evidence.fact_counter_available)
    assert torch.equal(logic.masses[logic.available], torch.full_like(logic.masses[logic.available], 0.25))
    assert torch.equal(logic.masses[~logic.available], torch.zeros_like(logic.masses[~logic.available]))


def test_unavailable_d1_stays_zero_after_projection_weights_change():
    class MissingFeatures(FrozenFeatures):
        def gather(self, node_id, node_type_id, drug_type, disease_type, device):
            values, available = super().gather(node_id, node_type_id, drug_type, disease_type, device)
            return values, available & (node_id >= 4)

    full = build_model(config(), pretrained=MissingFeatures(), seed=42)
    absent = build_model(config(), pretrained=None, seed=42, variant="NO_D1")
    with torch.no_grad():
        for layer in full.model.pretrained_fusion.projections.values():
            layer.weight.fill_(0.01)
    batch, _ = fixture()
    output = full.model.encoder.node_encoder(batch.nodes)
    base = absent.model.encoder.node_encoder(batch.nodes)
    assert torch.equal(output[:4], base[:4])
    assert torch.equal(output[[1, 2, 5, 6]], base[[1, 2, 5, 6]])
    assert not torch.equal(output[[4, 7]], base[[4, 7]])
