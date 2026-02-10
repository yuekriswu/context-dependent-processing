from collections.abc import Iterable, Sequence
import warnings

import numpy as np
import torch
import hyclib as lib

from .nn import functional as F
from niarb.tensors import periodic
from niarb.nn.modules import frame
from niarb import special


def min_distance(
    target_loc: torch.Tensor,
    cell_loc: torch.Tensor,
    scale: float | Sequence[float] = 1.0,
) -> torch.Tensor:
    """Computes minimum distance to target_loc for each cell_loc

    Args:
        target_loc: Tensor with shape (*, D)
        cell_loc: Tensor with shape (**, D)
        scale (optional): Scaling factor for each dimension, such that the distance
          in each dimension is divided by the corresponding scale. If a scalar, all
          dimensions are scaled by the same factor.

    Returns:
        Tensor with shape (**)

    """
    D = (target_loc.shape[-1], cell_loc.shape[-1])
    if D[0] != D[1]:
        raise ValueError(
            f"target_loc and cell_loc must have the same last dimension, but got {D}."
        )

    idx0, idx1 = lib.np.meshndim(target_loc.ndim - 1, cell_loc.ndim - 1)
    out = F.diff(target_loc[idx0], cell_loc[idx1])  # (*, **, D)
    out = out / torch.as_tensor(scale, device=out.device)
    out = out.norm(dim=-1).amin(dim=tuple(range(target_loc.ndim - 1)))

    return out


def inter_target_distances(target_loc: torch.Tensor) -> torch.Tensor:
    """Computes pairwise distances between targets

    These pairwise distances exclude duplicates that arise from symmetry as well zero distances.

    Args:
        target_loc: Tensor with shape (*, D)

    Returns:
        Tensor with shape (N(N-1)/2,), where N = prod(*)

    """
    target_loc = target_loc.reshape(-1, target_loc.shape[-1])  # (N, D)
    distances = F.diff(*lib.pt.meshgrid(target_loc, target_loc, dims=-1)).norm(
        dim=-1
    )  # (N, N)
    distances = distances[
        tuple(torch.triu_indices(*distances.shape, offset=1, device=distances.device))
    ]

    return distances


def min_distance_to_ensemble(
    x: frame.ParameterFrame,
    variable: str = "space",
    perturbation_var: str = "dh",
    **kwargs,
) -> torch.Tensor:
    """Compute each neuron's minimum distance to a perturbed neuron.

    Args:
        x: ParameterFrame with columns [variable, perturbation_var]
        variable: The variable for which distance is computed.
        perturbation_var: Name of the variable representing perturbation
        **kwargs: Additional arguments to min_distance

    Returns:
        Tensor with same shape as x

    """
    return min_distance(x[variable][x[perturbation_var] != 0], x[variable], **kwargs)


def inter_target_distance_statistics(
    x: frame.ParameterFrame,
    statistics: Iterable[str],
    variable: str = "space",
    perturbation_var: str = "dh",
    keepdim: bool = False,
) -> frame.ParameterFrame:
    """Compute statistics of pairwise distances between perturbed neurons.

    Args:
        x: ParameterFrame with columns [variable, perturbation_var]
        statistics: {'min', 'max', 'mean'}.
        variable: The variable for which distance is computed.
        perturbation_var: Name of the variable representing perturbation.
        keepdim: If True, output has same number of dimensions as x

    Returns:
        ParameterFrame with shape (1, ..., 1) if keepdim else ()

    """
    valid_statistics = {"min", "max", "mean"}
    if not all(s in valid_statistics for s in statistics):
        raise ValueError(
            f"statistics must be one of {valid_statistics}, but got {list(statistics)}."
        )

    distances = inter_target_distances(x[variable][x[perturbation_var] != 0])

    out = {}
    for s in statistics:
        out[s] = (
            torch.tensor(np.nan, device=x.device)
            if len(distances) == 0
            else getattr(distances, s)()
        )
        if keepdim:
            out[s] = out[s].reshape((1,) * x.ndim)

    return frame.ParameterFrame(out, ndim=x.ndim)


def inter_target_distance_mean(
    x: frame.ParameterFrame,
    variable: str = "space",
    perturbation_var: str = "dh",
    keepdim: bool = False,
) -> torch.Tensor:
    return inter_target_distance_statistics(
        x, ["mean"], variable, perturbation_var, keepdim
    )["mean"]


def ensemble_ori(
    x: frame.ParameterFrame, perturbation_var: str = "dh", keepdim: bool = False
) -> periodic.PeriodicTensor:
    """Compute ensemble orientation tuning

    Ensemble orientation is defined as the circular mean of orientation of perturbed neurons.
    If 'osi' is in x, then ensemble orientation tuning is weighted by x['osi'].

    Args:
        x: ParameterFrame with columns ['ori', perturbation_var], and optional column 'osi'.
           x['ori'] must be a PeriodicTensor
        perturbation_var: Name of the variable representing perturbation.
        keepdim: If True, output has same number of dimensions as x['ori']

    Returns:
        PeriodicTensor with shape (1, ..., 1, 1) if keepdim else (1,)

    """
    perturbed = x[perturbation_var] != 0

    weight = None
    if "osi" in x:
        weight = x["osi"][perturbed]

    out = x["ori"][perturbed].cmean(weight=weight)

    if keepdim:
        out = out.reshape((1,) * x["ori"].ndim)

    return out


def ensemble_osi(
    x: frame.ParameterFrame, perturbation_var: str = "dh", keepdim: bool = False
) -> torch.Tensor:
    """Compute ensemble OSI

    Ensemble OSI is defined by 1 - cvar, where cvar is the circular variance.
    If 'osi' is in x, then the circular variance is weighted by x['osi'].

    Args:
        x: ParameterFrame with columns ['ori', perturbation_var], and optional column 'osi'.
           x['ori'] must be a PeriodicTensor
        perturbation_var: Name of the variable representing perturbation.
        keepdim: If True, output has same number of dimensions as x['ori']

    Returns:
        Tensor with shape (1, ..., 1, 1) if keepdim else (1,)

    """
    perturbed = x[perturbation_var] != 0

    weight = None
    if "osi" in x:
        weight = x["osi"][perturbed]

    out = 1 - x["ori"][perturbed].cvar(weight=weight).squeeze(-1)

    if "osi" in x:
        out = out * weight.mean()

    if keepdim:
        out = out.reshape((1,) * x.ndim)

    return out


def abs_relative_ori(
    x: frame.ParameterFrame, perturbation_var: str = "dh", keepdim: bool = False
) -> periodic.PeriodicTensor:
    return F.diff(x["ori"], ensemble_ori(x, perturbation_var, keepdim)).norm(dim=-1)


def categorical_sample(
    prob: torch.Tensor, N: int, value: float | torch.Tensor = 1.0
) -> torch.Tensor:
    """Sample a vector of perturbations from given probabilities and number of targets.

    Args:
        prob: Tensor with shape (*), representing probability of each neuron being selected as a target
        N: Number of targets
        value: Perturbation strength

    Returns:
        Tensor with shape (*)

    """
    count = prob.count_nonzero()
    if N > count:
        raise ValueError(
            f"Number of targets must not exceed number of non-zero probabilities, but got {N=}, {prob.count_nonzero()=}."
        )  # Note: PyTorch says multinomial will handle this error, but there is a bug such that it doesn't
    elif N > count / 2:
        warnings.warn(
            "Number of targets more than half the number of non-zero probabilities, this may be undesirable."
        )

    shape = prob.shape
    prob = prob.reshape(-1)

    h = torch.zeros(prob.shape)
    h[torch.multinomial(prob, N)] = value
    h = h.reshape(shape)

    return h


def perturbation_prob(
    x: frame.ParameterFrame,
    *,
    cell_probs: dict[str, float] | None = None,
    space: Sequence = ("uniform", torch.inf),
    ori: Sequence = ("von_mises", 0.0),
    osi: Sequence = ("ubeta", 1.0, 1.0),
    space_loc: float | torch.Tensor = 0.0,
    ori_loc: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Probability that a neuron is perturbed given the perturbation parameters.

    Args:
        x: ParameterFrame with possible columns ['cell_type', 'space', 'ori', 'osi'].
        cell_probs: Dictionary of relative probabilities of each cell type.
          If a cell type is missing from the dictionary, it is assumed to have 0 probability.
          If None, assume uniform distribution.
        space: Distribution of spatial location, specified by (func_name, *args).
        ori: Distribution of orientation, specified by (func_name, *args).
        osi: Distribution of OSI, specified by (func_name, *args).
        space_loc: Mean of spatial location.
        ori_loc: Mean of orientation.

    Returns:
        Tensor with shape x.shape

    """
    prob = torch.ones(x.shape)

    for k, v in x.items():
        if k == "cell_type" and cell_probs is not None:
            cell_probs = [cell_probs.get(ct, 0.0) for ct in v.categories]
            prob *= torch.tensor(cell_probs)[v]
        elif k == "space":
            v = F.diff(v, space_loc)
            prob *= getattr(special, space[0])(*space[1:], v).prod(dim=-1)
        elif k == "ori":
            v = F.diff(v, ori_loc)
            prob *= getattr(special, ori[0])(*ori[1:], v).prod(dim=-1)
        elif k == "osi":
            prob *= getattr(special, osi[0])(*osi[1:], v)

    return prob


def sample(
    x: frame.ParameterFrame,
    N: int,
    affine_space: bool = False,
    affine_ori: bool = False,
    value: float | torch.Tensor = 1.0,
    **kwargs,
) -> torch.Tensor:
    """Sample a perturbation vector consisting of N perturbed neurons.

    Args:
        x: ParameterFrame with possible columns ['cell_type', 'space', 'ori', 'osi'].
        N: Number of perturbation targets.
        affine_space: If True, sample a random mean spatial location for the perturbed ensemble.
        affine_ori: If True, sample a random mean orientation for the perturbed ensemble.
        value: Perturbation strength.
        **kwargs: Additional arguments to perturbation_prob

    Returns:
        Tensor with shape x.shape

    """
    space_loc, ori_loc = 0.0, 0.0

    if affine_space:
        if "space" not in x:
            raise ValueError("If affine_space is True, x must contain 'space'.")

        space = x["space"]

        if not isinstance(space, periodic.PeriodicTensor):
            raise TypeError(
                "If affine_space is True, x['space'] must be a PeriodicTensor."
            )

        if list(sorted(space.w_dims)) != list(range(space.D)):
            raise ValueError(
                "If affine_space is True, x['space'] must be periodic in all dimensions."
            )

        space_loc = torch.distributions.Uniform(space.low, space.high).sample()

    if affine_ori:
        if "ori" not in x:
            raise ValueError("If affine_ori is True, x must contain 'ori'.")

        ori_loc = torch.distributions.Uniform(x["ori"].low, x["ori"].high).sample()

    prob = perturbation_prob(x, space_loc=space_loc, ori_loc=ori_loc, **kwargs)
    return categorical_sample(prob, N, value=value)


def convolve(
    x: frame.ParameterFrame,
    sigma: float,
    same_cell_type: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Convolve perturbations with a Gaussian to mimic physiological PSF.

    Args:
        x: ParameterFrame with columns ['dh', 'space', 'cell_type'].
        sigma: Standard deviation of the Gaussian.
        same_cell_type: If True, convolution only affects neurons with the same cell
          cell type as the perturbed neuron, mimicking opsin expression in a specific
          cell type.
        normalize: If True, normalize the convolution kernel to sum to 1.

    Returns:
        Tensor with shape x.shape

    """
    if "dh" not in x or "space" not in x:
        raise ValueError("x must contain 'dh' and 'space'.")

    if same_cell_type and "cell_type" not in x:
        raise ValueError("If same_cell_type is True, x must contain 'cell_type'")

    if x["dh"].ndim != x.ndim or (same_cell_type and x["cell_type"].ndim != x.ndim):
        raise ValueError(
            "x['dh'] and x['cell_type'] must have the same number of dimensions as x."
        )

    if x["space"].ndim != x.ndim + 1:
        raise ValueError("x['space'] must have one more dimension than x.")

    indices = x["dh"].nonzero()
    conv_perturbations = torch.zeros_like(x["dh"])
    for idx in indices:
        idx = tuple(idx)
        dist = F.diff(x["space"][idx], x["space"]).norm(dim=-1)
        conv = torch.exp(-(dist**2) / (2 * sigma**2))
        if same_cell_type:
            conv[x["cell_type"] != x["cell_type"][idx]] = 0.0
        if normalize:
            conv = conv / conv.sum()
        conv_perturbations += conv * x["dh"][idx]
    return conv_perturbations
