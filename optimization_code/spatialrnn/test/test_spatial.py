import torch
import numpy as np
import pytest
import hyclib as lib

from niarb.tensors import periodic
from niarb import spatial


def sort_pairs(pairs):
    mask = pairs[:, 0] > pairs[:, 1]
    pairs[mask, 0], pairs[mask, 1] = pairs[mask, 1], pairs[mask, 0]
    indices = np.lexsort(tuple(pairs.t().cpu().numpy()))
    return pairs[indices]


@pytest.mark.parametrize(
    "N, D, bins",
    [
        (10, 1, 1),
        (10, 1, 2),
        (10, 1, 100),
        (100, 1, 3),
        (100, 2, [1, 2]),
        (100, 2, [3, 2]),
        (1000, 2, [3, 3]),
        (1000, 2, [30, 30]),
        (10000, 3, 100),
    ],
)
@pytest.mark.parametrize("unique", [True, False])
@pytest.mark.parametrize("backend", ["torch", "pandas"])
@pytest.mark.parametrize("device", pytest.devices)
def test_get_neighbors(N, D, bins, unique, backend, device):
    with lib.random.set_seed(0):
        t = torch.rand((N, D), device=device)
    binnumbers, _, _ = lib.pt.stats.bin_dd(t, bins=bins)
    binnumbers = binnumbers - 1

    lidx, ridx = spatial.get_neighbors(bins, binnumbers, unique=unique, backend=backend)
    out = torch.stack([lidx, ridx], dim=1)

    if not unique and len(out) > 0:
        # output may contain duplicate and self-pairs, so remove those
        out = out[out[:, 0] != out[:, 1]]  # first remove self-pairs
        mask = out[:, 0] > out[:, 1]
        out[mask, 0], out[mask, 1] = out[mask, 1], out[mask, 0]
        out = out.unique(dim=0)  # remove duplicates

    expected = torch.stack(
        spatial.get_neighbors_naive(
            torch.atleast_1d(torch.tensor(bins, device=device)), binnumbers
        ),
        dim=1,
    )

    out, expected = sort_pairs(out), sort_pairs(expected)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("unique", [True, False])
@pytest.mark.parametrize("backend", ["torch", "pandas"])
@pytest.mark.parametrize("device", pytest.devices)
@pytest.mark.parametrize(
    "N, M",
    [
        (10, -1),
        (10, 0),
        (10, 1),
        (10, 10),
        (10000, -1),
        (10000, 0),
        (10000, 1000),
    ],
)
def test_get_pairs_within_dist(N, M, unique, backend, device):
    D = 3
    dist = 15
    extents = (-500, 500)
    indices = None

    with lib.random.set_seed(0):
        t = torch.rand((N, D), device=device) * (extents[1] - extents[0]) + extents[0]
        if M != -1:
            indices = np.random.choice(N, size=(M,), replace=False)
            indices = torch.from_numpy(indices).to(device)
    t = periodic.as_tensor(t, w_dims=range(D), extents=[extents] * D)

    out = spatial.get_pairs_within_dist(
        t,
        dist,
        indices=indices,
        sort=True,
        assume_unique=True,
        unique=unique,
        backend=backend,
    )
    expected = spatial.get_pairs_within_dist_naive(t, dist, indices=indices, sort=True)

    # output and expected may contain duplicates, so remove those first
    if not unique and out.numel() > 0:
        out = lib.pt.unique(out, dim=0)
    if expected.numel() > 0:
        expected = lib.pt.unique(expected, dim=0)

    # sort pairs for comparison
    out, expected = sort_pairs(out), sort_pairs(expected)

    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("device", pytest.devices)
@pytest.mark.parametrize(
    "N, M",
    [
        (10, -1),
        (10, 0),
        (10, 1),
        (10, 10),
        (10000, -1),
        (10000, 0),
        (10000, 1000),
    ],
)
def test_get_pairs_within_dist_sp(N, M, device):
    D = 3
    dist = 15
    extents = (-500, 500)

    with lib.random.set_seed(0):
        t = torch.rand((N, D), device=device) * (extents[1] - extents[0]) + extents[0]
        indices = (
            None
            if M == -1
            else torch.from_numpy(np.random.choice(N, size=(M,), replace=False)).to(
                device
            )
        )
    t = periodic.as_tensor(t, w_dims=range(D), extents=[extents] * D)

    out = spatial.get_pairs_within_dist_sp(
        t, dist, indices=indices, sort=True, assume_unique=True
    )
    expected = spatial.get_pairs_within_dist(
        t, dist, indices=indices, sort=True, assume_unique=True
    )

    # # expected may contain duplicates, so remove those first
    # if expected.numel() > 0:
    #     expected = lib.pt.unique(expected, dim=0)

    # sort pairs for comparison
    out, expected = sort_pairs(out), sort_pairs(expected)

    torch.testing.assert_close(out, expected)
