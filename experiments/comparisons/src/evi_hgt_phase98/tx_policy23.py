"""M4 inherited native DGL execution; no failed raw-order thread overrides."""
from contextlib import contextmanager


def metadata():
    return dict(policy_id='PHASE98_TX_M4_NATIVE_CPU_V4',torch_model_threads=8,
        native_pool_environment={},changes_dgl_threads=False,expected_dgl_threads=24,fused=True,
        scope='TX_ONLY',sampler_algorithm_changed=False,seed_or_rng_reset=False)


@contextmanager
def native_input_policy():
    from dgl import utils
    before=utils.get_num_threads()
    if before!=24:raise RuntimeError('Registered native DGL thread count drift')
    yield dict(metadata(),observed_dgl_threads=before)
    if utils.get_num_threads()!=before:raise RuntimeError('Unexpected DGL thread mutation')
