import numpy as np
from scipy.special import kv
from scipy.integrate import quad
import torch
import pytest

from niarb import special


def get_x(is_complex, is_double):
    out = np.linspace(-5, 5, num=5)
    if not is_double:
        out = out.astype(np.float32)
    if is_complex:
        real, imag = np.meshgrid(out, out, indexing="ij")
        out = real + 1.0j * imag
    return out, torch.from_numpy(out)


@pytest.mark.parametrize("a", [-1.0, 0.0, 1.0, 1.0 + 1.0j, 1.0j])
@pytest.mark.parametrize("r", [-1.0, 0.0, 1.0, 1.0 + 1.0j, 1.0j, [1.0, 0.0]])
@pytest.mark.parametrize("s", [0.0, [0.0, 1.0]])
@pytest.mark.parametrize("device", pytest.devices)
def test_yukawa(a, r, s, device):
    a = torch.tensor(a, device=device)
    r = torch.tensor(r, device=device)
    s = torch.tensor(s, device=device)
    a, r, s = torch.broadcast_tensors(a, r, s)
    out = special.yukawa(a, r, s)
    expected = torch.where(r == 0, s, torch.exp(-a * r) / r)

    torch.testing.assert_close(expected, out, equal_nan=True)


@pytest.mark.parametrize("a", [-1.0, 0.0, 1.0, 1.0 + 1.0j, 1.0j])
@pytest.mark.parametrize("r", [-1.0, 0.0, 1.0, 1.0 + 1.0j, 1.0j, [1.0, 0.0]])
@pytest.mark.parametrize("s", [0.0, [0.0, 1.0]])
@pytest.mark.parametrize("a_requires_grad", [True, False])
@pytest.mark.parametrize("r_requires_grad", [True, False])
@pytest.mark.parametrize("s_requires_grad", [True, False])
def test_yukawa_grad(a, r, s, a_requires_grad, r_requires_grad, s_requires_grad):
    if not any([a_requires_grad, r_requires_grad, s_requires_grad]):
        pytest.skip("No input requires grad")

    if r_requires_grad and (r == 0 or (isinstance(r, list) and 0 in r)):
        pytest.skip("Gradient with respect to r is not defined when r contains 0.")

    dtype_a = torch.complex128 if isinstance(a, complex) else torch.double
    dtype_r = torch.complex128 if isinstance(r, complex) else torch.double
    a = torch.tensor(a, dtype=dtype_a)
    r = torch.tensor(r, dtype=dtype_r)
    s = torch.tensor(s, dtype=torch.double)
    a, r, s = torch.broadcast_tensors(a, r, s)
    a = a.contiguous().requires_grad_(a_requires_grad)
    r = r.contiguous().requires_grad_(r_requires_grad)
    s = s.contiguous().requires_grad_(s_requires_grad)

    torch.autograd.gradcheck(special.yukawa, (a, r, s))


@pytest.mark.parametrize("d", list(range(-1, 6)))
@pytest.mark.parametrize("is_complex", [False, True])
@pytest.mark.parametrize("is_double", [False, True])
@pytest.mark.parametrize("device", pytest.devices)
def test_kd(d, is_complex, is_double, device):
    if is_double and device == "mps":
        pytest.skip("Double precision not supported on MPS")

    x_np, x_torch = get_x(is_complex, is_double)
    x_torch = x_torch.to(device)

    y_np = kv(d / 2 - 1, x_np)
    y_torch = special.kd(d, x_torch)

    if d == 2:
        # my custom modified_bessel_k0 does not currently support inputs on
        # left half complex plane.
        y_np[x_np.real < 0] = np.nan
        y_torch[x_np.real < 0] = torch.nan

    torch.testing.assert_close(torch.from_numpy(y_np), y_torch.cpu(), equal_nan=True)


@pytest.mark.parametrize("d", list(range(-1, 6)))
@pytest.mark.parametrize("is_complex", [False, True])
@pytest.mark.parametrize("device", pytest.non_mps_devices)
def test_kd_grad(d, is_complex, device):
    _, x = get_x(is_complex, True)
    x = x.to(device)
    x.requires_grad = True

    # ignore points along the branch cut (-\infty, 0]
    if is_complex:
        x = x[(x.real > 0) | (x.imag != 0)]
    else:
        x = x[x.real > 0]

    torch.autograd.gradcheck(special.kd, (d, x))


@pytest.mark.parametrize("d", list(range(1, 6)))
@pytest.mark.parametrize("a", [2.0, 2.0 + 1.0j])
def test_irkd(d, a):
    y = special.irkd(d, torch.tensor(a))
    if isinstance(a, complex):
        func = lambda r: (a * r) ** (d / 2) * kv(d / 2 - 1, a * r)
        expected = a * quad(func, 0, 1, complex_func=True)[0]
    else:
        expected = quad(lambda r: r ** (d / 2) * kv(d / 2 - 1, r), 0, a)[0]
    expected = torch.tensor(expected, dtype=y.dtype)
    torch.testing.assert_close(y, expected)


@pytest.mark.parametrize("d", [0, 1, 2, 3])
def test_ball_volume(d):
    r = torch.tensor([1.0, 2.0])
    out = special.ball_volume(d, r)
    if d == 0:
        expected = torch.ones_like(r)
    elif d == 1:
        expected = 2 * r
    elif d == 2:
        expected = torch.pi * r**2
    elif d == 3:
        expected = 4 / 3 * torch.pi * r**3
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_ball_radius(d):
    r = torch.tensor([1.0, 2.0])
    out = special.ball_radius(d, special.ball_volume(d, r))
    torch.testing.assert_close(out, r)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_solid_angle(d):
    r = torch.tensor(2.0)
    out = special.solid_angle(d)
    expected = special.ball_volume(d, r) * d / r**d
    torch.testing.assert_close(torch.tensor(out, dtype=torch.float), expected)
