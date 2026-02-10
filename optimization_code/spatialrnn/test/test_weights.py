import math

import pytest
from scipy import stats
import torch
import pandas as pd
import hyclib as lib

from niarb import weights, random


@pytest.mark.parametrize(
    "W",
    [
        [4.0, -2.0, 2.0, 0.0, -0.0],
        [4.0, -2.0, 2.0],
    ],
)
@pytest.mark.parametrize("min_weight", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
@pytest.mark.parametrize("approx", [None, "gumbel", "log_normal"])
def test_sparsify(W, min_weight, approx):
    W = torch.tensor(W)
    with random.set_seed(0):
        out = weights.sparsify(
            W.broadcast_to(100000, len(W)),
            min_weight=min_weight,
            approx=approx,
            tau=0.1,
        )

    aout = out.abs()
    if approx is None:
        # output (excluding zeros) is lower bounded by min_weight
        assert (aout[aout > 0] >= min_weight).all()

    atol = torch.zeros_like(W)
    if min_weight > 0:
        p = (W.abs() / min_weight).clip(max=1.0)
        var = p * (1 - p)
        atol[var > 0] = 0.02

    absdiff = (out.mean(dim=0) - W).abs()
    print(absdiff)
    assert (absdiff <= atol).all()  # mean of output should equal W


def test_sparsify_log_normal():
    with random.set_seed(0):
        W = torch.randn(10000000)
    out1 = weights.sparsify(W, min_weight=2.0)
    out2 = weights.sparsify(W, min_weight=2.0, approx="log_normal")
    var1 = out1.var()
    var2 = out2.var()
    assert var1 > 1.0 and var2 > 1.0
    torch.testing.assert_close(var1, var2, rtol=0.01, atol=0.0)


@pytest.mark.parametrize("approx", ["gumbel", "log_normal"])
def test_sparsify_grad(approx):
    with random.set_seed(0):
        W = torch.randn(5, dtype=torch.double, requires_grad=True)

    def func(W):
        with random.set_seed(0):
            return weights.sparsify(W, min_weight=2.0, approx=approx, tau=0.1)

    torch.autograd.gradcheck(func, W)


def test_sample_log_normal():
    mean = torch.tensor([1.5, 0.0, -2.0])
    W = mean.broadcast_to(1000, 3)
    std = 0.1

    with lib.random.set_seed(0):
        out = weights.sample_log_normal(W, std)

    # Output tensor should preserve sign of input tensor
    assert (out.sign() == W.sign()).all()

    # Use stats.kstest to check if output has correct log-normal distribution
    for i in [0, 2]:
        mean_i, std_i = mean[i].abs().item(), mean[i].abs().item() * std

        loc = math.log(mean_i**2 / math.sqrt(mean_i**2 + std_i**2))
        scale = math.sqrt(math.log(1 + std_i**2 / mean_i**2))

        m = stats.lognorm(s=scale, scale=math.exp(loc))
        res = stats.kstest(out[:, i].abs(), m.cdf)

        print(out[:, i], m.mean(), m.std(), res.pvalue)
        assert math.isclose(m.mean(), mean_i, rel_tol=1.3e-6)
        assert math.isclose(m.std(), std_i, rel_tol=1.3e-6)
        assert res.pvalue > 0.05


def test_sample_log_normal_grad():
    W = torch.randn(3, 3, dtype=torch.double, requires_grad=True)

    def func(W, seed=0, std=0.5):
        with lib.random.set_seed(seed):
            return weights.sample_log_normal(W, std)

    torch.autograd.gradcheck(func, W)
