from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from evi_hgt.phase9 import hgtdr_adapter as hg
from experiments.hgtdr_contra.trainer import build_trainer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 1)

    def forward(self, batch):
        return self.head(batch["features"])


class ToyFold:
    def __init__(self):
        self.data = SimpleNamespace(
            repeat_index=0,
            unsuitability={("d1", "s1"), ("d1", "s2"), ("d2", "s2")},
        )
        self.lookup = {"drug": {"d1": 0, "d2": 1}, "disease": {"s1": 0, "s2": 1}}
        self.cell_source = self.data
        self.supports = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
        self.training_seed = 93
        self.binding = {"cell": "R0F0", "candidate_sha256": "toy"}
        self.bindings = self.binding.copy()
        self.native = self

    def batch(self, pairs, config, seed, device):
        assert torch.are_deterministic_algorithms_enabled()
        return {
            "features": torch.tensor(pairs, dtype=torch.float32, device=device),
            "node_count": 4,
            "edge_count": 2,
            "masked_traversals": 1,
        }

    def unlabeled(self, count, seed):
        return np.repeat([[1, 0]], count, axis=0)


@pytest.fixture
def toy_backend(monkeypatch):
    fixed_config = hg.fixed_config
    monkeypatch.setattr(hg, "fixed_config", lambda seed: dict(fixed_config(seed), batch_size=4))
    monkeypatch.setattr(hg, "verify_native_identity", lambda: [])
    monkeypatch.setattr(
        hg, "native_module",
        lambda: SimpleNamespace(
            seed_for=lambda seed, *parts: seed,
            initialized_model=lambda data, batch, device, seed: ToyModel().to(device),
            compute_loss=torch.nn.functional.binary_cross_entropy_with_logits,
        ),
    )


def test_contra_uses_native_batch_recipe_and_roundtrips_control_state(toy_backend):
    fold = ToyFold()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    trainer = build_trainer(fold, device="cpu")

    first, last = trainer.step(), trainer.step()
    assert [first["positive_count"], last["positive_count"]] == [2, 1]
    assert [first["U_count"], last["U_count"]] == [2, 1]
    assert not first["epoch_finished"] and last["epoch_finished"]
    assert first["status"] == "CONTRA_TRAINING_UPDATE"
    assert trainer.cursor == {"epoch": 2, "batch_index": 0, "optimizer_steps": 2}
    assert trainer.scheduler.last_epoch == 1
    assert trainer.metadata()["prediction_target"] == "contra"
    assert torch.are_deterministic_algorithms_enabled() is previous_determinism

    resumed = build_trainer(fold, device="cpu")
    resumed.load_control_state(trainer.control_state())
    assert resumed.control_state() == trainer.control_state()


@pytest.mark.parametrize("mutation", ["wrong_target", "wrong_repeat", "wrong_cell"])
def test_contra_refuses_unbound_positives(mutation):
    fold = ToyFold()
    if mutation == "wrong_target":
        fold.supports = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64)
    elif mutation == "wrong_repeat":
        fold.data.repeat_index = 1
    else:
        fold.cell_source = SimpleNamespace()

    with pytest.raises(ValueError):
        build_trainer(fold, device="cpu")
