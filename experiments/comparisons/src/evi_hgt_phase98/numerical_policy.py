"""Scoped HGT/FULL deterministic kernels; imports and other methods are inert.

The runtime is single-worker. Callers must keep this context around the entire
HGT forward/backward/clip/update operation, and must not run an unrelated method
concurrently while PyTorch's process-global flag is temporarily enabled.
Precision, dropout, seeds, data, optimizer options and tolerances are unchanged.
"""
from contextlib import contextmanager

import torch


HGT='HGT_D1_MATCHED'
VERSION='PHASE98_HGT_DETERMINISTIC_ALGORITHMS_V1'
FULL_VERSION='PHASE98_FULL_DETERMINISTIC_ALGORITHMS_V1'


def numerical_policy_metadata(method):
    if method not in (HGT,'FULL'):
        return dict(method=method,policy='UNCHANGED')
    return dict(method=method,policy=VERSION if method==HGT else FULL_VERSION,scope='COMPLETE_UPDATE',
        enabled=True,warn_only=False,restore_previous_flags=True)


@contextmanager
def method_numerical_policy(method):
    """Qualified method scopes use deterministic algorithms and restore flags."""
    if method not in (HGT,'FULL'):
        yield
        return
    enabled=torch.are_deterministic_algorithms_enabled()
    warning=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True,warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warning)
