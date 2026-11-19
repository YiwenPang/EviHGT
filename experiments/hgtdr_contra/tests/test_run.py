import pytest
import torch

from experiments.hgtdr_contra.run import load_checkpoint, save_checkpoint


def test_epoch_checkpoint_replaces_latest_and_preserves_the_selected_weights(tmp_path):
    latest = tmp_path / "latest.pt"
    binding = {"cell": "R0F1", "head": "contra"}
    assert load_checkpoint(latest, binding) is None

    for completed_epoch in (1, 2):
        save_checkpoint(
            latest,
            {
                "binding": binding,
                "completed_epoch": completed_epoch,
                "best_weights": {"bias": torch.tensor([1.0])},
            },
        )

    saved = load_checkpoint(latest, binding)
    assert saved["completed_epoch"] == 2
    assert torch.equal(saved["best_weights"]["bias"], torch.tensor([1.0]))
    assert list(tmp_path.iterdir()) == [latest]


def test_resume_rejects_checkpoint_from_another_fold(tmp_path):
    latest = tmp_path / "latest.pt"
    save_checkpoint(latest, {"binding": {"cell": "R0F0"}, "completed_epoch": 1})

    with pytest.raises(ValueError, match="binding"):
        load_checkpoint(latest, {"cell": "R0F1"})


def test_archive_requires_the_research_workspace(tmp_path):
    from experiments.hgtdr_contra.run import add_research_source

    with pytest.raises(FileNotFoundError, match="research checkout"):
        add_research_source(tmp_path)
