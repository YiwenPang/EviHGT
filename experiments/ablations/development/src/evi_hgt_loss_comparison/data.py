"""Reuse the certified data boundary and exact external FEATURE_BASELINE prefix plan."""
from ablation_protocol_ids import FIRST_FOLD_ORIGIN_USE
from contextlib import contextmanager
from dataclasses import asdict
import copy

from evi_hgt.ranking_protocol.production_state import _rng, training_state_digest
from evi_hgt.conditional_sampling.conditional import build_pairing_plan, group_digest
from evi_hgt.graph_inputs.common import verify
from evi_hgt_feature_baseline.data import DevelopmentData, load_data as original_load, load_index, load_canonical_assembler
from evi_hgt_feature_baseline.sampling import load_schedule
from .protocol import G, TOTAL, MILESTONES, SCHEDULE, PAIRING, read, digest, file_sha


class Data(DevelopmentData):
    @contextmanager
    def evaluation_scope(self, step):
        if step not in MILESTONES or self._evaluation_step is not None:
            raise PermissionError("LOSS_COMPARISON two fixed evaluation milestones only")
        self._evaluation_step = step
        try:
            yield
        finally:
            self._evaluation_step = None


def load_data():
    old, corpus = original_load()
    data = Data(old.original)
    data.corpus_root = old.corpus_root
    return data, corpus


def load_paired(data, *, reproduce=False):
    before = training_state_digest(_rng())
    full, schedule = load_schedule(data)
    plan = verify(read(PAIRING))
    legacy = verify(plan["legacy"])
    if plan["data_bindings"] != data.bindings or plan["groups_digest"] != group_digest(full[G:]):
        raise ValueError("Historical pairing cell/group identity differs")
    groups = full[:TOTAL]
    records = legacy["records"][:TOTAL-G]
    if reproduce:
        rebuilt = build_pairing_plan(data, groups[G:], warmup_steps=G, bindings=legacy["bindings"])
        if rebuilt["records"] != records:
            raise ValueError("Frozen FEATURE_BASELINE conditional prefix cannot be reproduced")
    stream = [dict(step=i+1, member_sha256=digest([asdict(q) for q in qs]),
                   record_sha256=digest(records[i-G]) if i>=G else None) for i,qs in enumerate(groups)]
    if before != training_state_digest(_rng()):
        raise RuntimeError("Plan validation advanced training RNG")
    binding = dict(schedule_sha256=file_sha(SCHEDULE), pairing_sha256=file_sha(PAIRING),
        paired_stream_sha256=digest(stream), group_digest=group_digest(groups),
        conditional_prefix_sha256=digest(records), cycle_coverage=schedule["cycles"][:3],
        sampling_origin_sha256=plan["origin_sha256"],
        origin_use=FIRST_FOLD_ORIGIN_USE,
        private_rng_unchanged=True, prefix_records_reproduced=reproduce)
    return groups, records, stream, binding, schedule
