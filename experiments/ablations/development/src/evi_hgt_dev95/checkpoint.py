"""A private instance of the certified complete-state store, with DEV95 boundaries.

No shared imported module globals or historical files are changed. The new
manifest binds this adapter AND the exact historical implementation it reuses.
"""
import importlib.util
from pathlib import Path

from evi_hgt_dev94 import checkpoint as historical
from .protocol import OUT, ARMS, LOSSES, TOTAL

spec = importlib.util.spec_from_file_location("_dev95_private_checkpoint_backend", historical.__file__)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
backend.DEV_ROOT = OUT.resolve()
backend.TEST_ROOT = (OUT / "TEST_ONLY").resolve()
backend.ARMS = set(ARMS)
backend.MAX_UPDATES = TOTAL
backend.SCHEMA = "DEV95_COMPLETE_TRAINING_STATE_V1"
backend.COMMIT_SCHEMA = "DEV95_ATOMIC_CHECKPOINT_COMMIT_V1"
original_identity = backend._identity
original_sources = backend._implementation_sources


def _identity(value):
    original_identity(value)
    if value.get("loss_id") != LOSSES[value["arm"]]:
        raise ValueError("DEV95 arm/loss identity mismatch")


def _sources():
    return {**original_sources(),
            str(Path(__file__).resolve().relative_to(backend.ROOT)).replace("\\", "/"): backend.file_sha(__file__)}


backend._identity = _identity
backend._implementation_sources = _sources
Store, capture, restore, verify_commit = backend.Store, backend.capture, backend.restore, backend.verify_commit
