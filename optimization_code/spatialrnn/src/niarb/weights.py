import logging
from collections.abc import Iterable, Callable

import torch
from torch.nn import functional as F

from niarb.tensors import circulant
from niarb import utils, random
from niarb.nn.modules import frame

logger = logging.getLogger(__name__)


def discretize(
    kernel: Callable[[frame.ParameterFrame, frame.ParameterFrame], torch.Tensor],
    x: frame.ParameterFrame,
    ndim: int | None = None,
    dim: int | Iterable[int] = (),
    mul_dV: bool = True,
    diff_Gaussian_mask: torch.Tensor | None = None, # Kris: difference of Gaussians
    **kwargs,
) -> torch.Tensor:
    """Discretize a kernel function.

    Args:
        kernel: Callable with output.shape == broadcast(arg0.shape, arg1.shape)
        x: ParameterFrame with column ['dV'], with shape (*, *shape)
        ndim (optional): Number of non-batch dimensions (assumed to be trailing).
        dim (optional): Circulant dimensions, i.e. x is a regular,
            periodic grid along those dimensions. This is used to optimize the computation of weights.
        mul_dV (optional): If True, multiply kernel by dV.
        kwargs: keyword arguments passed to kernel

    Returns:
        Tensor with shape (*, *shape, *shape)

    """
    dim = utils.normalize_dim(dim, x.ndim, neg=True)

    if ndim is None:
        ndim = x.ndim

    if len(dim) > 0 and abs(min(dim)) > ndim:
        raise ValueError(
            f"circulant dimensions must be within the last {ndim=} dimensions, but {dim=}."
        )

    idx = tuple([0 if d in dim else slice(None) for d in range(-x.ndim, 0)])
    x_post, x_pre = x.datailoc[idx], x
    if len(dim) < ndim:
        dims = [(-(ndim - len(dim)), None), (-ndim, None)]
        x_post, x_pre = frame.meshgrid(x_post, x_pre, dims=dims, sparse=True)

    W = kernel(x_post, x_pre, **kwargs)  # (*, *shape[~dim], *shape), already include the weight

    if diff_Gaussian_mask is not None: # Kris: difference of Gaussians
        diff_Gaussian_mask = diff_Gaussian_mask.to(W.device)
        W = W * diff_Gaussian_mask # W has dimension [8, 8, 30, 30], post, pre, grid
        # print(W[0, 5, :, :])

    if mul_dV:
        W = W * x_pre.data["dV"]  # (*, *shape[~dim], *shape), x_pre.data["dV"] is 36.
        # W = W * x_post.data["dV"]  # (*, *shape[~dim], *shape)

    if len(dim) > 0:
        W = circulant.as_tensor(W, cdim=dim, ndim=ndim)

    return W


def sparsify(
    W: torch.Tensor,
    mean_p: float | None = None,
    min_weight: float | None = None,
    tol: float = 5.0e-3,
    approx: str | None = None,
    **kwargs,
) -> torch.Tensor:
    """Sparsify connectivity.

    Absolute value of weights are clipped below by min_weight. Connection probability is given by
    (W.abs() / min_weight).clip(max=1.0). If mean_p is not None, min_weight is chosen so that the
    mean connection probability is mean_p with absolute tolerance tol. Only one of mean_p
    and min_weight can be specified.

    Args:
        W: Connectivity tensor.
        mean_p (optional): Target mean connection probability, must be in (0, 1].
        min_weight (optional): Minimum absolute value of weights, must be non-negative.
        tol (optional): Tolerance for target mean connection probability.
        approx (optional): {None, "gumbel", "log_normal"}. If not None, use the specified
          approximation to sample binary connectivity. The approximate methods are differentiable.

    Returns:
        Tensor with same shape as W. Sparsified connectivity tensor.

    """
    if mean_p is not None and min_weight is not None:
        raise TypeError("Only one of mean_p and min_weight can be not None.")

    if tol < 0:
        raise ValueError(f"tolerance must be non-negative, but got {tol}.")

    if isinstance(W, circulant.CirculantTensor):
        W = W.dense()
    logger.debug(f"{W.shape=}")

    aW = W.abs()

    if mean_p is not None:
        if mean_p <= 0 or mean_p > 1:
            raise ValueError(f"mean_p must be in [0, 1], but got {mean_p}.")

        if mean_p == 1.0:
            min_weight = 0.0
        else:
            # find approximate min_weight by a binary search
            min_W, max_W = aW.min().item(), aW.mean().item() / mean_p
            min_weight = (min_W + max_W) / 2
            diff = (aW / min_weight).clip(max=1.0).mean().item() - mean_p
            while abs(diff) > tol:
                if diff > 0:
                    min_W = min_weight
                    min_weight = (min_weight + max_W) / 2
                else:
                    max_W = min_weight
                    min_weight = (min_W + min_weight) / 2
                diff = (aW / min_weight).clip(max=1.0).mean().item() - mean_p

    if min_weight is None or min_weight == 0:
        return W

    if min_weight < 0:
        raise ValueError(f"min_weight must be non-negative, but got {min_weight}.")

    p = (aW / min_weight).clip(max=1.0)

    if approx is None:
        binary = torch.bernoulli(p)
    elif approx == "gumbel":
        logits = torch.stack([p, 1 - p], dim=-1).log()
        binary = F.gumbel_softmax(logits, dim=-1, **kwargs)[..., 0]
    elif approx == "log_normal":
        binary = random.log_normal(p, torch.sqrt(p * (1 - p)), validate_args=False)
    else:
        raise ValueError(
            f"approx must be None, 'gumbel', or 'log_normal', but got {approx}."
        )

    return aW.clip(min=min_weight) * W.sign() * binary


def sample_log_normal(W: torch.Tensor, std: float | torch.Tensor) -> torch.Tensor:
    """Sample disordered weights from a log-normal distribution.

    Args:
        W: Mean weight matrix.
        std: Standard deviation of the log-normal distribution, as a fraction of the mean.

    Returns:
        Disordered weight matrix.

    """
    if isinstance(W, circulant.CirculantTensor):
        W = W.dense()
    logger.debug(f"{W.shape=}")

    return W.sign() * random.log_normal(W, W * std, validate_args=False)
