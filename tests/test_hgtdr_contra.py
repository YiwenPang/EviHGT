import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.hgtdr_contra.evaluation import (
    RANKING_KEYS,
    aggregate_fold_results,
    candidate_digest,
    checkpoint_key,
    score_panel,
)


class CandidateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_submodule = torch.nn.Identity()
        self.frozen_submodule.eval()

    def forward(self, batch):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        random.random()
        np.random.random()
        torch.rand(1)
        return torch.tensor(batch, dtype=torch.float32).unsqueeze(1)


def fixture(sampler_scores=(1.0, -1.0)):
    rows = (
        dict(drug_id="A", disease_id="D", role="P", raw_relation="CONTRAINDICATION"),
        dict(drug_id="B", disease_id="D", role="U"),
    )

    def batch(pairs, config, seed, device):
        assert config["batch_size"] == 1 and device == "cpu"
        assert seed in (0, 1)
        return [sampler_scores[a] for a, _ in pairs]

    native = SimpleNamespace(lookup={"drug": {"A": 0, "B": 1}, "disease": {"D": 0}}, batch=batch)
    trainer = SimpleNamespace(
        model=CandidateModel(),
        config={"batch_size": 1, "seed": 11},
        native=SimpleNamespace(seed_for=lambda *args: args[-1]),
        data=native,
        device="cpu",
        forwards=2,
        query_forward_count=3,
        actual_evaluation_seconds=0.0,
    )
    fold = SimpleNamespace(
        data=SimpleNamespace(cell_id="R0F0"), native=native, evaluation_rows=rows,
        binding={"candidate_order_sha256": candidate_digest(rows)},
    )
    return trainer, fold


def test_contra_panel_exports_matched_rows_and_restores_training_rng():
    trainer, fold = fixture()
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone()
    result = score_panel(trainer, fold)
    assert result["rows"] == [
        dict(fold.evaluation_rows[0], unsuitability_logit=1.0),
        dict(fold.evaluation_rows[1], unsuitability_logit=-1.0),
    ]
    assert result["metrics"]["head"] == "contra"
    assert result["metrics"]["macro_auc"] == result["metrics"]["macro_ap"] == 1.0
    assert result["metrics"]["candidate_identity_sha256"] == candidate_digest([("A", "D", "P"), ("B", "D", "U")])
    assert trainer.model.training and not trainer.model.frozen_submodule.training
    assert random.getstate() == expected_python
    actual_numpy = np.random.get_state()
    assert (actual_numpy[0], *actual_numpy[2:]) == (expected_numpy[0], *expected_numpy[2:])
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert trainer.forwards == 4 and trainer.query_forward_count == 5
    assert trainer.actual_evaluation_seconds > 0


def test_contra_score_rejects_changed_candidates_before_forward():
    trainer, fold = fixture()
    fold.evaluation_rows = fold.evaluation_rows[:-1]
    with pytest.raises(ValueError, match="candidate"):
        score_panel(trainer, fold)
    assert trainer.forwards == 2


def test_contra_score_rejects_nonfinite_result():
    trainer, fold = fixture((float("nan"), 1.0))
    with pytest.raises(ValueError, match="nonfinite"):
        score_panel(trainer, fold)
    assert trainer.model.training and not trainer.model.frozen_submodule.training
    assert trainer.forwards == 2


def test_checkpoint_key_orders_auc_then_ap_then_earlier_epoch():
    def key(epoch, auc, ap):
        return checkpoint_key(epoch, {"macro_auc": auc, "macro_ap": ap})

    assert key(300, .81, .2) > key(1, .8, .99)
    assert key(300, .8, .7) > key(1, .8, .699)
    assert key(20, .8, .7) > key(21, .8, .7)
    with pytest.raises(ValueError):
        key(3, float("inf"), .7)


def test_r0_aggregation_averages_per_disease_then_across_diseases():
    def disease(index, value):
        return dict(
            disease_id=f"D{index:03d}", P=1, U=1, auc=value, ap=value,
            aupr_trapezoid=value, **dict.fromkeys(RANKING_KEYS, value),
        )

    folds = {}
    for fold in range(5):
        diseases = [disease(0, 0.0)] if fold == 0 else [disease(i, 1.0) for i in range(314)]
        folds[f"R0F{fold}"] = dict(
            head="contra", status="COMPLETE", evaluable_diseases=len(diseases),
            disease_metrics=diseases, skipped_disease_details=[],
        )
    result = aggregate_fold_results(folds)
    assert result["evaluable_diseases"] == 314
    assert result["macro_auc"] == pytest.approx(313.8 / 314)
    assert result["macro_ap"] == pytest.approx(313.8 / 314)
    assert result["ranking"]["NDCG@10"] == pytest.approx(313.8 / 314)
    assert result["disease_metrics"][0]["legal_folds"] == 5
    folds.pop("R0F4")
    with pytest.raises(ValueError, match="five"):
        aggregate_fold_results(folds)
