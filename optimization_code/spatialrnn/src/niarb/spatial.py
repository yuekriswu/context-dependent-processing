from scipy import spatial
from pandas.core.reshape import merge
import torch
import numpy as np
import hyclib as lib

from niarb.nn import functional
from niarb.tensors import periodic


def get_join_indexers(lkeys, rkeys, backend=None):
    """
    Args:
        lkeys, rkeys: one-dimensional torch.Tensors
        backend: {'torch', 'pandas', None} (optional).
    Returns:
        lidx, ridx: one-dimensional torch.Tensors
    """
    if backend not in {"torch", "pandas", None}:
        raise ValueError(f"backend must be 'torch', 'pandas', or None, but {backend=}.")

    if backend == "pandas" or (backend is None and lkeys.device.type == "cpu"):
        device = lkeys.device
        lkeys, rkeys = lkeys.detach().cpu().numpy(), rkeys.detach().cpu().numpy()
        lidx, ridx = merge.get_join_indexers([lkeys], [rkeys])
        if lidx is None:
            lidx = np.arange(len(lkeys))
        if ridx is None:
            ridx = np.arange(len(rkeys))

        lidx, ridx = torch.from_numpy(lidx).to(device), torch.from_numpy(ridx).to(
            device
        )

    else:
        # slight optimization of ordering
        if len(lkeys) > len(rkeys):
            lmask = torch.isin(lkeys, rkeys)
            lkeys = lkeys[lmask]
            rmask = torch.isin(rkeys, lkeys)
            rkeys = rkeys[rmask]
        else:
            rmask = torch.isin(rkeys, lkeys)
            rkeys = rkeys[rmask]
            lmask = torch.isin(lkeys, rkeys)
            lkeys = lkeys[lmask]

        # circumvent MPS bug with calling sort() on empty tensors
        if len(lkeys) == 0:
            return (torch.empty((0,), dtype=torch.long, device=lkeys.device),) * 2

        lkeys, lidx = lkeys.sort()
        rkeys, ridx = rkeys.sort()
        lidx = lmask.nonzero().squeeze(-1)[lidx]
        ridx = rmask.nonzero().squeeze(-1)[ridx]
        _, lcounts = lkeys.unique_consecutive(return_counts=True)
        _, rcounts = rkeys.unique_consecutive(return_counts=True)
        lidx = torch.repeat_interleave(lidx, torch.repeat_interleave(rcounts, lcounts))
        ridx = lib.pt.repeat_interleave(ridx, lcounts, chunks=rcounts, validate=False)

    return lidx, ridx


def get_neighbors_naive(nbins, binnumbers):
    diff = (binnumbers[:, :, None] - binnumbers[:, None, :]).abs()
    diff = torch.minimum(diff, nbins[:, None, None] - diff)
    return (diff <= 1).all(axis=0).triu(1).nonzero(as_tuple=True)


def get_neighbors(
    nbins, lbin, rbin=None, lid=None, rid=None, unique=True, validate=True, backend=None
):
    """
    Efficiently computes indices of pairs of elements whose bin numbers
    are the equal or adjacent (note that we also consider 0 and -1 to be adjacent).
    If unique is True, output contains only unique pairs of neurons. Otherwise,
    output may contain duplicate pairs (like (0, 1), (1, 0)) or self-pairs (like (0, 0)),
    but it will be slightly faster.
    The larger number of bins, the faster this should be (until there are roughly
    as many bins as number of elements).
    backend may be 'torch', 'pandas', or None. If None, defaults to 'torch' if device is a GPU.

    Args:
        nbins: int or torch.Tensor with shape (D,) and integer dtype, representing number of bins
        lbin, rbin: torch.Tensor with shapes (D, Nl), (D, Nr) and integer dtype
        lid, rid: torch.Tensor with shapes (Nl,), (Nr,) and integer dtype
        unique: bool (optional). If True, ensures output pairs are unique and does not contain self-pairs.
        validate: bool (optional). Performs checks on inputs arguments.
        backend: {'torch', 'pandas', None} (optional). Backend to use for the merge operation.
    Returns:
        2-tuple of torch.Tensor with shape (P,), where P is number of pairs
    """
    is_disjoint = True
    if rbin is None:
        rbin = lbin
        is_disjoint = False

    if lid is None:
        lid = torch.arange(lbin.shape[1], device=lbin.device)

    if rid is None:
        rid = lid

    nbins = torch.as_tensor(nbins, device=lbin.device).broadcast_to((lbin.shape[0],))

    if (nbins <= 2).all():
        # all points are neighbors
        lid, rid = torch.meshgrid(lid, rid, indexing="ij")

        if not unique or is_disjoint:
            return lid.reshape(-1), rid.reshape(-1)

        triu_indices = tuple(
            torch.triu_indices(*lid.shape, offset=1, device=lid.device)
        )
        return lid[triu_indices], rid[triu_indices]

    if (nbins <= 2).any():
        # if there are only 1 or 2 bins along a dimension, then all points are automatically
        # neighbors along that dimension, so we can ignore those dimensions.
        mask = nbins > 2
        return get_neighbors(
            nbins[mask],
            lbin[mask],
            rbin=(rbin[mask] if is_disjoint else None),
            lid=lid,
            rid=rid,
            unique=unique,
            validate=validate,
            backend=backend,
        )

    if validate:
        if nbins.is_floating_point():
            raise TypeError(f"nbins must be an integer tensor, but {nbins.dtype=}.")

        if lbin.is_floating_point() or rbin.is_floating_point():
            raise TypeError(
                f"lbin and rbin must be integer tensor, but {lbin.dtype=}, {rbin.dtype=}."
            )

        if lid.is_floating_point() or rid.is_floating_point():
            raise TypeError(
                f"lid and rid must be integer tensor, but {lid.dtype=}, {rid.dtype=}."
            )

        if lbin.ndim != 2 or rbin.ndim != 2:
            raise ValueError(
                f"lbin and rbin must be a 2-dimensional tensor, but {lbin.ndim=}, {rbin.ndim=}."
            )

        if lid.ndim != 1 or rid.ndim != 1:
            raise ValueError(
                f"lid and rid must be a 1-dimensional tensor, but {lid.ndim=}, {rid.ndim=}."
            )

        if (lbin.numel() > 0 and lbin.min() < 0) or (
            rbin.numel() > 0 and rbin.min() < 0
        ):
            raise ValueError(
                f"lbin and rbin must be non-negative, but {lbin.min()=}, {rbin.min()}."
            )

        if (lbin.numel() > 0 and (lbin.max(dim=1).values >= nbins).any()) or (
            rbin.numel() > 0 and (rbin.max(dim=1).values >= nbins).any()
        ):
            raise ValueError(
                f"lbin and rbin must be less than nbins, but {lbin.max(dim=1).values=}, {rbin.max(dim=1).values}, {nbins=}."
            )

        if len(nbins) != lbin.shape[0] or lbin.shape[0] != rbin.shape[0]:
            raise ValueError(
                "nbins, lbin and rbin must have compatible first dimension, "
                f"but {len(nbins)=}, {lbin.shape[0]=}, {rbin.shape[0]=}."
            )

        if len(lid) != lbin.shape[1] or len(rid) != rbin.shape[1]:
            raise ValueError(
                "lid and rid must have compatible shapes with lbin and rbin."
            )

        if not (
            lbin.device.type
            == rbin.device.type
            == lid.device.type
            == rid.device.type
            == nbins.device.type
        ):
            raise ValueError(
                "lbin, rbin, lid, rid, nbins must all be on the same device."
            )

    device = lbin.device

    if lbin.numel() == 0 or rbin.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=device), torch.empty(
            (0,), dtype=torch.long, device=device
        )

    swapped = False
    if rbin.shape[1] > lbin.shape[1]:
        lbin, rbin = (
            rbin,
            lbin,
        )  # swap for performance since we perform more operations on rbin
        lid, rid = rid, lid
        swapped = True

    D, Nl, Nr = (*lbin.shape, rbin.shape[-1])  # noqa

    lbin = lib.pt.ravel_multi_index(lbin, nbins.tolist(), validate=validate)  # (Nl,)

    shifts = [torch.tensor([0, 1, -1], device=device)] * D
    shift = torch.stack(torch.meshgrid(*shifts, indexing="ij"))  # (D, 3, ..., 3)

    if is_disjoint:
        shift = shift.reshape(D, -1)
    else:
        # exploit symmetry to reduce the number of shifts we need to perform by half
        mask = torch.ones(shift.shape[1:], dtype=torch.bool, device=device)
        for i in range(D):
            mask[(0,) * i + (-1,) + (...,)] = False

        if unique:
            mask[(0,) * D] = False
            lindexer, rindexer = get_join_indexers(lbin, lbin, backend=backend)
            lid0, rid0 = lid[lindexer], rid[rindexer]
            mask0 = lid0 < rid0
            lid0, rid0 = lid0[mask0], rid0[mask0]

        shift = shift[:, mask]  # (D, (3^D - 1) / 2 + 1)

    rbin = (rbin[:, :, None] + shift[:, None, :]) % nbins[
        :, None, None
    ]  # (D, Nr, 3^D or (3^D - 1) / 2 + 1)
    rbin = lib.pt.ravel_multi_index(
        rbin, nbins.tolist(), validate=validate
    )  # (Nr, 3^D or (3^D - 1) / 2 + 1)
    rid = rid[:, None].broadcast_to(rbin.shape)  # (Nr, 3^D or (3^D - 1) / 2 + 1)

    lindexer, rindexer = get_join_indexers(lbin, rbin.reshape(-1), backend=backend)
    lid, rid = lid[lindexer], rid.reshape(-1)[rindexer]

    if unique and not is_disjoint:
        lid, rid = torch.cat([lid, lid0]), torch.cat([rid, rid0])

    if swapped:
        lid, rid = rid, lid

    return lid, rid


def get_pairs_within_dist_Linf(
    x,
    dist,
    indices=None,
    unique=True,
    as_tuple=False,
    assume_unique=False,
    backend=None,
):
    if not assume_unique and indices is not None:
        indices = indices.unique()

    # calculate the maximum number of bins along each dimension such that
    # the side lengths of each bin are all greater than dist
    min, max = x.min(dim=0).values, x.max(dim=0).values  # (D,), (D,)

    # address machine precision issue with torch.bucketize
    min = torch.nextafter(min, -torch.inf * torch.ones_like(min))
    max = torch.nextafter(max, torch.inf * torch.ones_like(max))

    nbins = ((max - min) // dist).long()  # (D,)

    # compute binnumbers (could use lib.pt.stats.bin_dd, but it's slower since it is more general and accurate)
    N, D = x.shape
    edges = [
        torch.linspace(
            min[i].item(), max[i].item(), nbins[i].item() + 1, device=x.device
        )
        for i in range(D)
    ]
    binnumbers = torch.stack(
        [torch.bucketize(x[:, i], edges[i]) for i in range(D)]
    )  # (D, N)

    # subtract 1 so that the zeroth bin represents points lying outside the boundaries
    binnumbers = binnumbers - 1

    kwargs = {"validate": False, "unique": unique, "backend": backend}

    if indices is not None:
        lbin = binnumbers[:, indices]
        mask = torch.ones(binnumbers.shape[-1], dtype=torch.bool, device=x.device)
        mask[indices] = False
        rbin = binnumbers[:, mask]
        lidx = indices
        ridx = mask.nonzero().squeeze()

        lidx0, ridx0 = get_neighbors(nbins, lbin, lid=lidx, **kwargs)
        lidx1, ridx1 = get_neighbors(
            nbins, lbin, rbin=rbin, lid=lidx, rid=ridx, **kwargs
        )
        lidx, ridx = torch.cat([lidx0, lidx1]), torch.cat([ridx0, ridx1])
    else:
        lidx, ridx = get_neighbors(nbins, binnumbers, **kwargs)

    if as_tuple:
        return lidx, ridx

    return torch.stack([lidx, ridx], dim=1)


def get_pairs_within_dist(
    x,
    dist,
    indices=None,
    unique=True,
    sort=False,
    return_distances=False,
    assume_unique=False,
    backend=None,
):
    """
    Given an (N, D) tensor of neuron locations, efficiently computes
    the indices of all neuron pairs that are less than dist apart.
    If indices is provided, return only pairs where the index of at least
    one of them is in indices.
    If unique, output is gauaranteed to only contain unique pairs. Otherwise,
    output may contain duplicate pairs (like (0, 1), (1, 0)).
    If sort, ensures the first element of each pairs is smaller than the
    second element.
    If return_distances, returns pairwise distances corresponding to the returned indices.
    If assume_unique, assumes indices are unique. Ignored if indices is None.
    backend may be 'torch', 'pandas', or None. If None, defaults to 'torch' if device is a GPU.
    Scales with number of neurons much better (in both time and space
    complexity) than the naive approach.

    Args:
        x: torch.Tensor or periodic.PeriodicTensor with shapes (N, D)
        dist: float or scalar torch.Tensor.
        indices: torch.Tensor with shape (M,) (optional).
        unique: bool (optional).
        sort: bool (optional).
        return_indices: bool (optional).
        assume_unique: bool (optional).
        backend: {'torch', 'pandas', None} (optional). Backend to use for the merge operation.
    Returns:
        idx: torch.Tensor with shape (P, 2), where P is number of pairs
        distances (optional): torch.Tensor with shape (P,)
    """
    lidx, ridx = get_pairs_within_dist_Linf(
        x,
        dist,
        indices=indices,
        unique=unique,
        as_tuple=True,
        assume_unique=assume_unique,
        backend=backend,
    )

    distances = functional.diff(x[lidx], x[ridx]).norm(dim=-1)
    mask = (0 < distances) & (distances < dist)

    idx = torch.stack([lidx, ridx], dim=-1)[mask]

    if sort:
        mask = idx[:, 0] > idx[:, 1]
        idx[mask, 0], idx[mask, 1] = idx[mask, 1], idx[mask, 0]

    if return_distances:
        return idx, distances[mask]

    return idx


def get_pairs_within_dist_sp(x, dist, indices=None, assume_unique=False, sort=False):
    """
    Implementation based on scipy.spatial.KDTree. This seems to be faster for small
    problems (N < 10000) but slower for large problems (N > 10000) than my own algorithm.
    If dist is large then it is also faster than my own algorithm.
    Not that unlike my own algorithm, output does not contain duplicate pairs.
    """
    if isinstance(x, periodic.PeriodicTensor) and len(x.w_dims) > 0:
        if len(x.w_dims) != x.D:
            raise NotImplementedError(
                "Currently only handles inputs that are either non-periodic or periodic in all dimensions."
            )

        boxsize = x.period.detach().cpu().numpy()
        x = x - x.low
    else:
        boxsize = None

    if indices is not None:
        if not assume_unique:
            indices = indices.unique()

        mask = torch.ones(len(x), dtype=torch.bool, device=x.device)
        mask[indices] = False

        x0 = x[indices]
        x1 = x[mask]

        kdtree0 = spatial.KDTree(x0.detach().cpu().numpy(), boxsize=boxsize)
        kdtree1 = spatial.KDTree(x1.detach().cpu().numpy(), boxsize=boxsize)

        pairs0 = kdtree0.query_pairs(dist, output_type="ndarray")
        pairs0 = torch.from_numpy(pairs0).to(x.device)

        pairs1 = kdtree1.query_ball_tree(kdtree0, dist)
        pairs1 = torch.tensor(
            [(i, j) for i, js in enumerate(pairs1) for j in js],
            dtype=torch.long,
            device=x.device,
        ).reshape(-1, 2)

        pairs0 = indices[pairs0]
        pairs1[:, 1] = indices[pairs1[:, 1]]
        pairs1[:, 0] = mask.nonzero().squeeze()[pairs1[:, 0]]

        pairs = torch.cat([pairs0, pairs1])

        if sort:
            mask = pairs[:, 0] > pairs[:, 1]
            pairs[mask, 0], pairs[mask, 1] = pairs[mask, 1], pairs[mask, 0]

    else:
        kdtree = spatial.KDTree(x.detach().cpu().numpy(), boxsize=boxsize)

        pairs = kdtree.query_pairs(dist, output_type="ndarray")
        pairs = torch.from_numpy(pairs).to(x.device)

    return pairs


def get_pairs_within_dist_naive(x, dist, indices=None, sort=False):
    if indices is None:
        distances = functional.diff(x[:, None, :], x[None, :, :]).norm(dim=-1)
    else:
        distances = functional.diff(x[indices, None, :], x[None, :, :]).norm(dim=-1)

    mask = (distances > 0) & (distances < dist)
    idx = mask.nonzero()

    if indices is not None:
        idx[:, 0] = indices[idx[:, 0]]

    if sort:
        mask = idx[:, 0] > idx[:, 1]
        idx[mask, 0], idx[mask, 1] = idx[mask, 1], idx[mask, 0]

    return idx
