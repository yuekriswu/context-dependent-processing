from contextlib import nullcontext

import pytest
import torch

from niarb.optimize import elementwise


def func(x, a, b, c):
    return a * x**2 + b * x + c


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_bisect(dtype):
    func = torch.cos
    a = torch.tensor(0.0, dtype=dtype)
    b = torch.tensor([torch.pi / 4, 3 * torch.pi / 4], dtype=dtype)
    out = elementwise.bisect(func, a, b)
    expected = torch.tensor([torch.nan, torch.pi / 2], dtype=dtype)
    torch.testing.assert_close(out, expected, equal_nan=True)


@pytest.mark.parametrize("a", [1, -1])
@pytest.mark.parametrize("device", pytest.devices)
def test_minimize_newton(a, device):
    shape = (10,)
    a = torch.tensor(a, device=device)
    b = torch.randn(shape, device=device)
    c = torch.randn(shape, device=device)
    x0 = torch.zeros(shape, device=device)

    with pytest.raises(ValueError) if a < 0 else nullcontext():
        out = elementwise.minimize_newton(func, x0, args=(a, b, c))

    if a > 0:
        torch.testing.assert_close(out, -b / 2)
