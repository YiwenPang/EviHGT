"""Fixed synthetic examples recorded with the original sampling and initialization code."""

import hashlib
import json
from pathlib import Path
import random

import pytest


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "development" / "src"))
    import ablation_protocol_ids

    return ablation_protocol_ids


def digest(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_three_cycle_schedule_and_conditional_draws(protocol):
    # One support, one contra and one shared unlabeled query per group in each cycle.
    group_hash = "56f912faeb4928ea9e39f684e3bdab4d40199c8571de36eeca8a26d08a16ff16"
    member_hash = "8a7666f2e515e2b54fb0cdc326357549b53c79fb2a8421f5c0f195da77fd49c1"
    schedule = dict(
        schema=protocol.PAIRED_SCHEDULE_SCHEMA, cell="R0F1", spec={"G": 1},
        data_bindings={"fold_manifest_digest": "a" * 64, "fold_mask_digest": "b" * 64},
        corpus_digest="f" * 64, mask_digest="b" * 64,
        cycles=[dict(cycle=i, groups=1, support_P=1, contra_P=1, group_digest=group_hash,
                     positive_coverage="EXACTLY_ONCE_EACH_CHANNEL") for i in (1, 2, 3)],
        group_digest="14b855f8d55cf43022a72d8060649df3b905b6df92e111ca02cf39c2824cf09c",
        member_stream=[dict(step=i, member_sha256=member_hash) for i in (1, 2, 3)],
        order="original epoch_groups(1/2/3); original intragroup order",
    )
    schedule["digest"] = digest(schedule)
    assert schedule["digest"] == "a1c333666988c4538892081927af33a23d732ab54b48ad6c5e6ba4c769f3c37d"
    base = dict(fold_manifest_digest="a" * 64, repeat_key="c" * 64, fold=1, training_seed=342565966,
                training_u_manifest_digest="d" * 64, exclusion_digest="b" * 64, warmup_checkpoint_sha256="e" * 64)
    bindings = protocol.paired_sampling_bindings(base, "R0F1", schedule)
    assert digest(bindings) == "81f2c6be71b664ad4a4ca87aecd62c9b28e6bcd312ed15d1df6d4c7888abf04a"
    seeds = [protocol.conditional_draw_seed(bindings, index, "condition-example", digest) for index in (0, 1)]
    assert seeds == [8611264168981315217, 5473549329913094716]
    legal = [f"compound-{i}" for i in range(12) if i not in (1, 2)]
    assert [legal[random.Random(seed).randrange(len(legal))] for seed in seeds] == ["compound-10", "compound-7"]
    assert digest({"schema": protocol.PAIRED_PLAN_SCHEMA}) == "50be984a233556785efa0b5490c1a0908fb3b96c057ab9b42f5a2d1e100b698e"


def test_model_initialization_seeds(protocol):
    folds = [1289779183, 342565966, 625224385]
    assert [protocol.side_initialization_seed(seed, digest) for seed in folds] == [806919546, 4218594298, 1821244296]


def test_feature_fusion_checkpoint_keys(protocol):
    fields = [f"{protocol.FUSION_LAYER_NAME}.projections.{kind}.weight" for kind in ("DRG", "DIS")]
    assert digest([*fields, protocol.FEATURE_SCHEMA]) == "a252eadea8fe24d361908a62bceef8a254709ef28a6b2db66549fb931165c698"
