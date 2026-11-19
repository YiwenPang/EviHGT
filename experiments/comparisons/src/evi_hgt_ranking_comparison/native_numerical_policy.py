"""Scoped HGTDR kernel policy; no model or authorization is created here.

The RankingComparison receiver must keep this scope around initialization and the complete
forward/backward/optimizer update. PyTorch's flag is process-global, so the
single-worker contract applies. The disabled variant is only for a separately
registered, zero-update baseline diagnosis; it must not be a formal option.
Precision, RNG, model mathematics, sampling and optimizer settings are untouched.
"""
from contextlib import contextmanager

import torch


VERSION = 'RANKING_COMPARISON_HGTDR_DETERMINISTIC_ALGORITHMS_V1'
BASELINE_VERSION = 'RANKING_COMPARISON_HGTDR_NONDETERMINISTIC_BASELINE_V1'


def hgtdr_numerical_policy_metadata(enabled=True):
    """Describe intended execution without mutating the process configuration."""
    if type(enabled) is not bool:
        raise TypeError('HGTDR numerical policy enabled must be bool')
    return dict(method='HGTDR_ADAPTED', policy=VERSION if enabled else BASELINE_VERSION,
                scope='INITIALIZATION_AND_COMPLETE_UPDATE', enabled=enabled,
                warn_only=False, restore_previous_flags=True,
                diagnostic_baseline_only=not enabled)


@contextmanager
def hgtdr_numerical_policy(enabled=True):
    """Apply strict kernels, restoring both previous flags even after failure."""
    hgtdr_numerical_policy_metadata(enabled)
    previous = torch.are_deterministic_algorithms_enabled()
    warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warning)
