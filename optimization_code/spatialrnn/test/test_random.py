import math
from contextlib import nullcontext

import pytest
from scipy import stats
import torch

from niarb import random


@pytest.mark.parametrize("validate_args", [True, False])
def test_log_normal(validate_args):
    mean = torch.tensor([-1.5, 0.0, -2.0, 1.0])
    mean = mean.broadcast_to(1000, 4)
    std = torch.tensor([0.0, 0.0, 1.0, -0.5])

    if validate_args:
        mean, std = mean.abs(), std.abs()

    with random.set_seed(0):
        out = random.log_normal(mean, std, validate_args=validate_args)
    mean, std = mean.abs(), std.abs()  # negative mean and std are treated as positive

    # Check if output has correct log-normal distribution when std > 0
    for i in [2, 3]:
        mean_i, std_i = mean[0, i].item(), std[i].item()

        loc = math.log(mean_i**2 / math.sqrt(mean_i**2 + std_i**2))
        scale = math.sqrt(math.log(1 + std_i**2 / mean_i**2))

        m = stats.lognorm(s=scale, scale=math.exp(loc))
        assert math.isclose(m.mean(), mean_i, rel_tol=1.3e-6)
        assert math.isclose(m.std(), std_i, rel_tol=1.3e-6)

        res = stats.kstest(out[:, i], m.cdf)
        assert res.pvalue > 0.05

    # Check if output is equal to mean when std = 0
    assert (out[:, :2] == mean[:, :2]).all()


@pytest.mark.parametrize(
    "mean, std, valid",
    [
        (-1.0, -1.0, False),
        (0.0, -1.0, False),
        (1.0, -1.0, False),
        (-1.0, 0.0, False),
        (0.0, 0.0, True),
        (1.0, 0.0, True),
        (-1.0, 1.0, False),
        (0.0, 1.0, False),
        (1.0, 1.0, True),
    ],
)
def test_log_normal_validate_args(mean, std, valid):
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    with pytest.raises(ValueError) if not valid else nullcontext():
        random.log_normal(mean, std, validate_args=True)


def test_log_normal_grad():
    mean = torch.tensor(
        [-1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 1.0], dtype=torch.double, requires_grad=True
    )
    std = torch.tensor(
        [-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.double, requires_grad=True
    )

    def func(mean, std):
        with random.set_seed(0):
            return random.log_normal(mean, std, validate_args=False)

    torch.autograd.gradcheck(func, (mean, std))
