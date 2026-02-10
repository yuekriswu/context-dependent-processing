import random
import contextlib

import numpy as np
import torch
from niarb import spatial


def log_normal(
    mean: torch.Tensor, std: torch.Tensor, validate_args: bool | None = None
) -> torch.Tensor:
    """Sample from a log-normal distribution with given mean and standard deviation.

    If validate_args is False, or it is None and code is run with -O flag,
    mean and std are allowed to be negative, but they will be treated as if they
    are positive.

    Args:
        mean: Mean of the log-normal distribution.
        std: Standard deviation of the log-normal distribution.
        validate_args (optional): Whether to validate the arguments.

    Returns:
        Sample from the log-normal distribution.

    Raises:
        ValueError: If validate_args is True or validate_args is None and code is not
          run wit -O flag, and either condition is met:
            - mean or std is negative.
            - std is non-zero when mean is zero.

    """
    # By default, we validate the arguments if code is not run with -O flag,
    # same as the default behavior of torch.distributions.LogNormal.
    if validate_args is None:
        validate_args = __debug__

    mean, std = torch.broadcast_tensors(mean, std)

    if validate_args:
        if (mean < 0).any() or (std < 0).any():
            raise ValueError("mean and std must be non-negative.")
        if (std[mean == 0] != 0).any():
            raise ValueError("std must be zero when mean is zero.")
        out = mean.clone()
    else:
        out = mean.abs()

    # handle zeros in std
    mask = std != 0
    mean, std = mean[mask], std[mask]

    loc = torch.log(mean**2 / torch.sqrt(mean**2 + std**2))
    scale = torch.sqrt(torch.log(1 + std**2 / mean**2))

    m = torch.distributions.LogNormal(loc, scale, validate_args=validate_args)
    out[mask] = m.rsample()

    return out


def resample_with_min_dist(x, m, min_dist):
    if min_dist <= 0:
        return x

    indices = spatial.get_pairs_within_dist(x, min_dist, unique=False, sort=True)

    while len(indices) > 0:
        # note: the problem of finding the minimal number of neurons to resample is equivalent
        # to finding the minimal vertex cover of a graph, where each problematic neuron is a vertex,
        # and an edge exists between two neurons if their pairwise distance is smaller than min_dist.
        # This is an NP-complete problem, so we settle for the following naive solution.
        indices = indices[:, 0].unique()
        x[indices] = m.sample(indices.shape)

        if len(indices) > 5:
            indices = spatial.get_pairs_within_dist(
                x,
                min_dist,
                indices=indices,
                unique=False,
                sort=True,
                assume_unique=True,
            )
        else:
            # slightly faster when there are very few neurons resampled
            indices = spatial.get_pairs_within_dist_naive(
                x, min_dist, indices=indices, sort=True
            )

    return x


@contextlib.contextmanager
def set_seed(seed):
    if seed is None:
        yield
        return

    python_state = random.getstate()
    random.seed(seed)

    np_state = np.random.get_state()
    np.random.seed(seed)

    torch_state = torch.random.get_rng_state()
    torch.random.manual_seed(seed)

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
