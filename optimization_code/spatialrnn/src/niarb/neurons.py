import math
from collections.abc import Sequence, Iterable

import torch
import numpy as np
from scipy import stats
import hyclib as lib

from niarb.nn.modules import frame
from niarb.tensors import periodic, categorical
from niarb import random
from niarb.cell_type import CellType


def as_grid(
    n: int = 0,
    N_space: Sequence[int] = (),
    N_ori: int = 0,
    N_osi: int = 0,
    *,
    cell_types: Sequence[CellType | str] = tuple(CellType),
    space_extent: Sequence[float] = (1000.0, 1000.0, 300.0),
    ori_extent: Sequence[float] = (-90.0, 90.0),
    osi_prob: torch.distributions.Distribution | Sequence = ("Uniform", 0.0, 1.0),
) -> frame.ParameterFrame:
    """Generate a grid of neurons.

    Args:
        n (optional): Number of cell types.
        N_space (optional): Number of neurons to along each spatial dimension.
        N_ori (optional): Number of orientation samples.
        N_osi (optional): Number of OSI samples.
        cell_types (optional): Cell types to include.
        space_extent (optional):
            Lengths of the spatial dimensions. Must have at least as many elements as the number
            of spatial dimensions.
        ori_extent (optional): (lb, ub) tuple specifying the lower and upper bound of orientation.
        osi_prob (optional):
            Distribution of OSI. If a tuple, the first argument is the name of the distribution,
            and the rest are the distribution parameters. batch_shape must be either () or (n,).

    Returns:
        ParameterFrame of the grid of neurons, with shape ([n], [*N_space], [N_ori], [N_osi]).

    """

    if isinstance(osi_prob, Sequence):
        osi_prob = getattr(torch.distributions, osi_prob[0])(
            *[torch.as_tensor(v) for v in osi_prob[1:]]
        )

    if osi_prob.event_shape != () or osi_prob.batch_shape not in {(), (n,)}:
        raise ValueError(
            f"osi_prob must have event_shape () and batch_shape () or {(n,)=}, "
            f"but {osi_prob.event_shape=}, {osi_prob.batch_shape=}."
        )

    d = len(N_space)
    if d > len(space_extent):
        raise ValueError(
            f"space_extent must have at least {d} elements, but {len(space_extent)=}."
        )

    cell_types = tuple(CellType[ct] if isinstance(ct, str) else ct for ct in cell_types)

    dV, x, ndims = 1.0, {}, []
    if n > 0:
        x["cell_type"] = categorical.as_tensor(
            torch.arange(n), categories=[ct.name for ct in cell_types]
        )
        ndims.append(1)

    if d > 0:
        x["space"] = [
            periodic.linspace(-extent / 2, extent / 2, Ni)
            for extent, Ni in zip(space_extent, N_space)
        ]
        x["space"] = torch.cat(lib.pt.meshgrid(*x["space"], dims=-1), dim=-1)
        dV *= torch.prod(x["space"].period).item() / math.prod(N_space)
        space_dV = dV
        ndims.append(d)

    if N_ori > 0:
        x["ori"] = periodic.linspace(*ori_extent, N_ori)
        dV *= torch.prod(x["ori"].period).item() / N_ori
        ndims.append(1)

    dims = [(None, ndim) for ndim in ndims]
    x = dict(zip(x.keys(), lib.pt.meshgrid(*x.values(), dims=dims, sparse=True)))
    x = frame.ParameterFrame(x, ndim=sum(ndims))

    if N_osi > 0:
        m = max(1, n)
        osi = torch.linspace(0.0, 1.0, steps=N_osi)

        osi_prob = osi_prob.expand((m,))
        if isinstance(osi_prob, torch.distributions.Beta):
            # PyTorch currently does not support icdf for the beta distribution, so we use scipy.
            alpha, beta = osi_prob.concentration1, osi_prob.concentration0
            osi = [stats.beta.ppf(osi.numpy(), a, b) for a, b in zip(alpha, beta)]
            osi = torch.from_numpy(np.stack(osi)).float()  # (m, N_osi)
        else:
            osi = osi_prob.icdf(osi[:, None]).t()  # (m, N_osi)

        x = x.datailoc[..., None]
        x["osi"] = osi[(slice(None), *((None,) * (d + (N_ori > 0))), ...)].squeeze(0)
        dV *= 1 / N_osi

    x["dV"] = torch.tensor(dV)[(None,) * x.ndim]
    if d > 0:
        x["space_dV"] = torch.tensor(space_dV)[(None,) * x.ndim]

    return x


def sample(
    N: int,
    variables: Sequence[str],
    *,
    cell_types: Sequence[CellType | str] = tuple(CellType),
    cell_probs: torch.Tensor | Sequence[float] | None = None,
    space_extent: Sequence[float] = (1000.0, 1000.0, 300.0),
    ori_extent: Sequence[float] = (-90.0, 90.0),
    osi_prob: torch.distributions.Distribution | Sequence = ("Uniform", 0.0, 1.0),
    min_dist: float = 0.0,
    w_dims: int | Iterable[int] | None = None,
) -> frame.ParameterFrame:
    """Generate samples of neurons.

    Args:
        N: Number of neurons to generate.
        variables: {"cell_type", "space", "ori", "osi"}. Variables to sample.
        cell_types (optional): Cell types to sample from.
        cell_probs (optional):
            Relative probabilities for each cell type. Defaults to the probabilities of each
            CellType Enum in cell_types, normalized such that the default E-I ratio is preserved.
        space_extent (optional): Lengths of the spatial dimensions.
        ori_extent (optional): (lb, ub) tuple of lower and upper bounds of the orientation extent.
        osi_prob (optional):
            Distribution of OSI. If a tuple, the first argument is the name of the distribution,
            and the rest are the distribution parameters. batch_shape must be either () or (n,).
        min_dist (optional): Minimum pairwise distance between neurons.
        w_dims (optional): spatial dimensions with periodic boundary conditions.
            If None, all spatial dimensions are periodic.

    Returns:
        ParameterFrame of sampled neurons with shape (N,).

    """
    if len(variables) == 0:
        raise ValueError("At least one variable must be specified.")

    cell_types = tuple(CellType[ct] if isinstance(ct, str) else ct for ct in cell_types)

    if cell_probs is None:
        total_E_prob = sum(ct.prob for ct in CellType if ct.sign == 1)
        total_I_prob = sum(ct.prob for ct in CellType if ct.sign == -1)
        subset_E_prob = sum(ct.prob for ct in cell_types if ct.sign == 1)
        subset_I_prob = sum(ct.prob for ct in cell_types if ct.sign == -1)
        E_ratio = total_E_prob / subset_E_prob if subset_E_prob > 0 else float("nan")
        I_ratio = total_I_prob / subset_I_prob if subset_I_prob > 0 else float("nan")
        cell_probs = [
            ct.prob * E_ratio if ct.sign == 1 else ct.prob * I_ratio
            for ct in cell_types
        ]

    cell_probs = torch.as_tensor(cell_probs)
    cell_probs = cell_probs / cell_probs.sum()
    space_extent = [(-extent / 2, extent / 2) for extent in space_extent]

    if isinstance(osi_prob, Sequence):
        osi_prob = getattr(torch.distributions, osi_prob[0])(
            *[torch.as_tensor(v) for v in osi_prob[1:]]
        )

    n = len(cell_probs)
    if osi_prob.event_shape != () or osi_prob.batch_shape not in {(), (n,)}:
        raise ValueError(
            f"osi_prob must have event_shape () and batch_shape () or {(n,)=}, "
            f"but {osi_prob.event_shape=}, {osi_prob.batch_shape=}."
        )

    if osi_prob.batch_shape == (n,) and "cell_type" not in variables:
        raise ValueError(
            "If osi_prob has batch_shape (n,), 'cell_type' must be included in variables."
        )

    x = frame.ParameterFrame(ndim=1)
    dV = torch.tensor([1.0 / N])

    if "space" in variables:
        m = torch.distributions.Uniform(*torch.tensor(list(zip(*space_extent))))
        extents = space_extent if w_dims is None else [space_extent[d] for d in w_dims]
        x["space"] = periodic.as_tensor(m.sample((N,)), extents=extents, w_dims=w_dims)
        x["space"] = random.resample_with_min_dist(x["space"], m, min_dist)
        dV = dV * math.prod((ub - lb for lb, ub in space_extent))

    if "ori" in variables:
        x["ori"] = periodic.as_tensor(
            torch.distributions.Uniform(*ori_extent).sample((N, 1)),
            extents=[ori_extent],
        )
        dV = dV * (ori_extent[1] - ori_extent[0])

    if "cell_type" in variables:
        x["cell_type"] = torch.distributions.Categorical(cell_probs).sample((N,))
        x["cell_type"] = categorical.as_tensor(
            x["cell_type"], categories=tuple(ct.name for ct in cell_types)
        )
        dV = dV / cell_probs  # (n,)
        dV = dV[x["cell_type"]]  # (N,)

    if "osi" in variables:
        osi = osi_prob.sample((N,))  # (N, n) or (N,) where n is number of cell types
        if osi.ndim == 1:
            x["osi"] = osi
        else:
            x["osi"] = osi.take_along_dim(x["cell_type"][:, None], dim=1).squeeze(1)

    x["dV"] = dV

    return x
