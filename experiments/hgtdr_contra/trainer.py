"""Train HGTDR to rank contraindications on one R0 fold."""

import copy

import numpy as np

from evi_hgt.phase9 import hgtdr_adapter as hg
from evi_hgt.phase9.identity import file_sha
from evi_hgt_phase98.native_numerical_policy import (
    hgtdr_numerical_policy,
    hgtdr_numerical_policy_metadata,
)


METHOD = "HGTDR_ADAPTED_CONTRA"


class ContraTrainer(hg.SingleLRTrainer):
    """Bind HGTDR training to the contraindication positives in its fold."""

    def __init__(self, fold_data, *, device):
        data = fold_data.data
        native_data = fold_data.native
        if data.repeat_index != 0 or native_data.cell_source is not data:
            raise ValueError("Native data must be bound to the original R0 training cell")

        # HGTDR reads the positive pairs from its supports array.
        drug = native_data.lookup["drug"]
        disease = native_data.lookup["disease"]
        expected = np.asarray(
            [(drug[d], disease[s]) for d, s in sorted(data.unsuitability)],
            dtype=np.int64,
        )
        if not len(expected) or not np.array_equal(native_data.supports, expected):
            raise ValueError("Native positives differ from the bound R0 contraindications")

        self.fold_binding = copy.deepcopy(fold_data.binding)
        with hgtdr_numerical_policy():
            super().__init__(native_data, device=device, permit=None)
        self.execution_scope = dict(mode="R0_CONTRA_INNER_DEVELOPMENT", prediction_target="contra")

    def metadata(self):
        metadata = super().metadata()
        metadata.pop("has_refutation_head")
        return dict(
            metadata,
            method=METHOD,
            prediction_target="contra",
            output_logits=1,
            fold_binding=copy.deepcopy(self.fold_binding),
            contra_adapter_sha256=file_sha(__file__),
            numerical_policy=dict(hgtdr_numerical_policy_metadata(), method=METHOD),
        )

    def step(self):
        with hgtdr_numerical_policy():
            result = super().step()
        return dict(result, status="CONTRA_TRAINING_UPDATE")


def build_trainer(fold_data, device="cuda"):
    return ContraTrainer(fold_data, device=device)
