import pytest
import torch

from niarb import nn
from niarb.tensors import categorical


@pytest.fixture
def x():
    return torch.linspace(-5.0, 5.0, steps=50)


@pytest.mark.parametrize(
    "f, xlim",
    [
        ("nn.Identity()", (-5.0, 5.0)),
        ("nn.Pow(2.5,)", (0.0, 5.0)),
        ("nn.Threshold(-1.0, -2.0)", (-1.0 + 1.0e-5, 5.0)),
        ("nn.Rectified(-1.0,)", (-1.0 + 1.0e-5, 5.0)),
        ("nn.Rectified() ** 2", (1.0e-5, 5.0)),
    ],
)
def test_inv(f, xlim):
    f = eval(f)
    x = torch.linspace(*xlim, steps=50)
    torch.testing.assert_close(f.inv()(f(x)), x)
    torch.testing.assert_close(f(f.inv()(x)), x)


def test_identity(x):
    torch.testing.assert_close(nn.Identity()(x), x)


@pytest.mark.parametrize("pow", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_pow(pow, x):
    torch.testing.assert_close(nn.Pow(pow)(x), x**pow, equal_nan=True)


@pytest.mark.parametrize("f", [nn.Identity(), nn.Rectified(threshold=1.0)])
@pytest.mark.parametrize("pow", [-2.0, -0.5, 0.0, 0.5, 2.0])
def test_functional_pow(f, pow, x):
    torch.testing.assert_close((f**pow)(x), f(x) ** pow, equal_nan=True)


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_threshold(threshold, value, x):
    torch.testing.assert_close(
        nn.Threshold(threshold, value)(x), torch.where(x > threshold, x, value)
    )


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0])
def test_rectified(threshold, x):
    torch.testing.assert_close(
        nn.Rectified(threshold=threshold)(x), torch.where(x > threshold, x, threshold)
    )


def test_match():
    f = nn.Match({"PV": nn.SSN(3)}, nn.SSN(2))
    x = torch.randn(10)
    key = categorical.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2], categories=["PYR", "PV", "SST"]
    )
    out = f(x, key)
    expected = nn.SSN(2)(x)
    expected[3:6] = nn.SSN(3)(x[3:6])
    torch.testing.assert_close(out, expected)
