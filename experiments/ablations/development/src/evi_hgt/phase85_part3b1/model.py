"""The existing SIDE architecture with explicit, model-independent cell seed.

The seed-42 path is tensor-identical to the certified historical factory. No
trained checkpoint, learned transform, new module, or numeric flag is used.
"""
from __future__ import annotations

import random
import numpy as np
import torch

from evi_hgt.phase85_v2.common import bound,digest
from evi_hgt.phase85_v2.model import build_side
from evi_hgt.models.path_reasoning import PathReasoningConfig
from evi_hgt.phase85_part2b.pilot import model_configuration
from evi_hgt.phase85_part2b.runtime import weight_digest


def side_seed(training_seed):
    return int(digest(['PHASE852B_SIDE_INITIALIZATION_V1',training_seed])[:8],16)


def build_cell_model(training_seed):
    if type(training_seed) is not int or not 0<=training_seed<2**31:
        raise ValueError('A certified 31-bit integer cell seed is required')
    identity,config=model_configuration()
    # CPU-only construction; isolate its stream from consumers of this factory.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(training_seed)
        built=build_side(base_config=config,path_config=PathReasoningConfig(core_hidden_dim=config.hidden_dim),
            drug_type=identity['node_type_order'].index('DRG'),
            disease_type=identity['node_type_order'].index('DIS'))
        side=built.model.encoder.node_encoder
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(side_seed(training_seed))
            for branch in (side.drug_encoder,side.disease_encoder,side.drug_gate,side.disease_gate):
                for module in branch.modules():
                    if hasattr(module,'reset_parameters'):module.reset_parameters()
    return built


def seed_training_streams(seed):
    random.seed(seed);np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def certify_fresh_initialization(seed):
    built=build_cell_model(seed);other=build_cell_model(seed)
    state,repeat=built.model.state_dict(),other.model.state_dict()
    if state.keys()!=repeat.keys() or any(not torch.equal(v,repeat[k]) for k,v in state.items()):
        raise ValueError('Fresh factory tensors are not deterministic on CPU')
    optimizer=torch.optim.AdamW(built.model.parameters(),lr=5e-5,weight_decay=1e-5)
    if optimizer.state:raise ValueError('New AdamW must have no moments or prior steps')
    identity,_=model_configuration()
    return bound(dict(schema='PART3B1_FRESH_CELL_INITIALIZATION_V1',training_seed=seed,
        side_initialization_seed=side_seed(seed),model='EVIHGT_SIDE',model_config_digest=identity['digest'],
        fresh=True,loaded_training_checkpoint=False,factory_repeat_bitwise_equal=True,
        weight_digest=weight_digest(built.model),parameter_tensors=len(state),
        parameter_count=sum(p.numel() for p in built.model.parameters()),
        tensor_structure_digest=digest({k:[list(v.shape),str(v.dtype)] for k,v in state.items()}),
        optimizer_state_entries=0,lr=5e-5,weight_decay=1e-5,
        training_rng_rule='Python/NumPy/torch CPU/CUDA seeded with certified cell seed before first optimizer use',
        full_cuda_trajectory_bitwise_reproducibility_claim=False))
