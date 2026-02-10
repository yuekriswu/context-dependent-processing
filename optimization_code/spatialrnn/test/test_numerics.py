import contextlib
from functools import partial

import pytest
import torch
import numpy as np

from niarb import exceptions, numerics, nn
from niarb.tensors import categorical

rng = np.random.default_rng()


def linear_response(W, r0, t, h, tau):
    assert (
        W.ndim == 2
        and W.shape[0] == W.shape[1]
        and r0.ndim == 1
        and h.ndim == 1
        and t.ndim == 1
        and tau.ndim == 1
        and tau.shape[0] == W.shape[1]
    )
    I = torch.eye(W.shape[0], dtype=W.dtype)
    Tinv = torch.diag(tau.reciprocal())
    rss = torch.linalg.solve(I - W, h)
    r = rss - torch.linalg.matrix_exp(-Tinv @ (I - W) * t[:, None, None]) @ (rss - r0)
    return r


def linear_steady_state(W, h):
    assert W.ndim == 2 and W.shape[0] == W.shape[1] and h.ndim == 1
    I = torch.eye(W.shape[0], dtype=W.dtype)
    rss = torch.linalg.solve(I - W, h)
    return rss


@pytest.fixture
def t():
    return torch.linspace(0.0, 10.0, steps=50)


@pytest.mark.parametrize(
    "W",
    [
        [[0.9004, -0.6447], [0.1701, -1.0695]],  # stable, real eigvals
        [[0.2047, -0.8562], [1.4895, -1.5838]],  # stable, complex eigvals
        [[1.3734, -0.6369], [0.6790, -1.2972]],  # unstable, real eigvals
        [[2.6492, -1.4848], [3.7162, -0.0712]],  # unstable, complex eigvals
    ],
)
@pytest.mark.parametrize("h", [[1.0, 0.5], [-0.5, -1.0]])
@pytest.mark.parametrize("x0", [[0.0, 0.0], [-1.0, 0.5]])
@pytest.mark.parametrize("t", [None, (0.0, 10.0, 50)])
@pytest.mark.parametrize("tau", [1.0, [1.0, 0.5]])
def test_simulate(W, h, x0, t, tau):
    W, h, x0, tau = map(torch.tensor, (W, h, x0, tau))
    if t is not None:
        t = torch.linspace(*t)

    rtol, atol = 1.3e-6, 1.0e-5
    unstable = torch.linalg.eigvals(W).real.max() >= 1
    if unstable:
        rtol, atol = 10 * rtol, 10 * atol  # greater tolerance for unstable networks

    if t is None:
        expected = linear_steady_state(W, h).float()
    else:
        tau_ = torch.tensor(tau).broadcast_to((2,))
        expected = linear_response(
            W.double(), x0.double(), t.double(), h.double(), tau_.double()
        ).float()  # need higher precision

    raises_error = t is None and unstable
    with (
        pytest.raises(exceptions.SimulationError)
        if raises_error
        else contextlib.nullcontext()
    ):
        out = numerics.simulate(
            W, nn.Identity(), h, t=t, tau=tau, x0=x0, options=dict(max_num_steps=200)
        )
        if t is None:
            # to check that steady state condition is satisfied,
            # we need to recompute with a slightly later time endpoint,
            # see integrate.odeint_ss for further explanation
            assert out.t.ndim == 0
            out = numerics.simulate(
                W,
                nn.Identity(),
                h,
                t=out.t + 1e-9,
                tau=tau,
                x0=x0,
                options=dict(max_num_steps=200),
            )

    if not raises_error:
        torch.testing.assert_close(out.x, expected, rtol=rtol, atol=atol)
        if t is None:
            # test steady state condition is satisfied
            assert (out.dxdt.abs() <= out.x.abs() * 1.3e-7 + 1.0e-6).all()


@pytest.mark.parametrize(
    "W",
    [
        [[0.9004, -0.6447], [0.1701, -1.0695]],  # stable, real eigvals
        [[0.2047, -0.8562], [1.4895, -1.5838]],  # stable, complex eigvals
    ],
)
@pytest.mark.parametrize("vf", [0.5, [0.5, 0.25]])
@pytest.mark.parametrize("dh", [[1.0, 0.5], [-0.5, -1.0]])
@pytest.mark.parametrize("dx0", [1.0, [1.0], [-1.0, 0.5]])
def test_perturbed_steady_state(W, vf, dh, dx0):
    W, vf, dh, dx0 = map(torch.tensor, (W, vf, dh, dx0))
    out = numerics.perturbed_steady_state(
        vf,
        W,
        nn.Identity(),
        dh,
        dx0=dx0,
        dx_rtol=1.0e-8,
        options=dict(max_num_steps=200),
    )

    expected = linear_steady_state(W, dh)
    torch.testing.assert_close(out.x, expected)


@pytest.mark.parametrize(
    "W",
    [
        [[0.9004, -0.6447], [0.1701, -1.0695]],  # stable, real eigvals
        [[0.2047, -0.8562], [1.4895, -1.5838]],  # stable, complex eigvals
    ],
)
@pytest.mark.parametrize("vf", [[0.5, 0.25]])
@pytest.mark.parametrize(
    "f",
    [
        "nn.Identity()",
        "nn.Identity() ** 2",
        "nn.Rectified() ** 2",
        "nn.Ricciardi(scale=0.025)",
    ],
)
def test_fixed_point(W, vf, f):
    W, vf = map(torch.tensor, (W, vf))
    f = eval(f)
    rf, hf = numerics.fixed_point(vf, W, f)
    torch.testing.assert_close(W @ rf + hf, vf.broadcast_to((2,)))


@pytest.mark.parametrize(
    "f, vf, expected",
    [
        ("nn.Rectified() ** 2", -0.5, 0.0),
        ("nn.Rectified() ** 2", [0.75], [1.5]),
        ("nn.Rectified() ** 2", [-0.5, 0.75], [0.0, 1.5]),
    ],
)
def test_compute_gain(f, vf, expected):
    vf, expected = map(torch.tensor, (vf, expected))
    f = eval(f)
    out = numerics.compute_gain(f, vf)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "f, vf",
    [
        ("nn.Rectified() ** 2", -0.5),
        ("nn.Rectified() ** 2", [0.75]),
        ("nn.Rectified() ** 2", [-0.5, 0.75]),
    ],
)
def test_compute_gain_gradient(f, vf):
    vf = torch.tensor(vf, dtype=torch.double, requires_grad=True)
    f = eval(f)
    torch.autograd.gradcheck(partial(numerics.compute_gain, f), vf)


def test_compute_gain_inference_mode():
    vf = torch.tensor(0.75, requires_grad=True)
    f = nn.Rectified() ** 2
    with torch.inference_mode():
        numerics.compute_gain(f, vf + 1.0)


@pytest.mark.parametrize(
    "f, vf, n, expected",
    [
        ("nn.Rectified() ** 2", -0.5, 1, 0.0),
        ("nn.Rectified() ** 2", [0.75], 1, [1.5]),
        ("nn.Rectified() ** 2", [[-0.5, 0.75]], 1, [[0.0, 1.5]]),
        ("nn.Rectified() ** 2", -0.5, 2, 0.0),
        ("nn.Rectified() ** 2", [0.75], 2, [2.0]),
        ("nn.Rectified() ** 2", [[-0.5, 0.75]], 2, [[0.0, 2.0]]),
        ("nn.Rectified() ** 3", -0.5, 2, 0.0),
        ("nn.Rectified() ** 3", [0.75], 2, [4.5]),
        ("nn.Rectified() ** 3", [[-0.5, 0.75]], 2, [[0.0, 4.5]]),
    ],
)
@pytest.mark.parametrize("device", pytest.devices)
def test_nth_deriv(f, vf, n, expected, device):
    vf = torch.tensor(vf, device=device)
    expected = torch.tensor(expected, device=device)
    f = eval(f)
    out = numerics.compute_nth_deriv(f, vf, n=n)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    "f, vf, n",
    [
        ("nn.Rectified() ** 2", -0.5, 1),
        ("nn.Rectified() ** 2", [0.75], 1),
        ("nn.Rectified() ** 2", [[-0.5, 0.75]], 1),
        ("nn.Rectified() ** 2", -0.5, 2),
        ("nn.Rectified() ** 2", [0.75], 2),
        ("nn.Rectified() ** 2", [[-0.5, 0.75]], 2),
        ("nn.Rectified() ** 3", -0.5, 2),
        ("nn.Rectified() ** 3", [0.75], 2),
        ("nn.Rectified() ** 3", [[-0.5, 0.75]], 2),
    ],
)
@pytest.mark.parametrize(
    "device", pytest.non_mps_devices
)  # MPS does not support float64
def test_nth_deriv_gradient(f, vf, n, device):
    vf = torch.tensor(vf, dtype=torch.double, requires_grad=True, device=device)
    f = eval(f)
    torch.autograd.gradcheck(partial(numerics.compute_nth_deriv, f, n=n), vf)


@pytest.mark.parametrize("n, expected", [(1, [1, 4, 1]), (2, [0, 2, 0])])
@pytest.mark.parametrize("device", pytest.devices)
def test_nth_deriv_match(n, expected, device):
    f = nn.Match({"PV": nn.SSN()}, nn.Rectified())
    vf = torch.tensor(2.0, device=device)
    args = (categorical.tensor([0, 1, 0], categories=("PYR", "PV"), device=device),)
    expected = torch.tensor(expected, device=device).float()
    out = numerics.compute_nth_deriv(f, vf, args=args, n=n)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("device", pytest.non_mps_devices)
def test_nth_deriv_match_gradient(n, device):
    f = nn.Match({"PV": nn.SSN()}, nn.Rectified())
    vf = torch.tensor(2.0, dtype=torch.double, requires_grad=True, device=device)
    args = (categorical.tensor([0, 1, 0], categories=("PYR", "PV"), device=device),)
    out = numerics.compute_nth_deriv(f, vf, args=args, n=n)
    torch.autograd.gradcheck(partial(numerics.compute_nth_deriv, f, args=args, n=n), vf)


@pytest.mark.parametrize("n", [1, 2])
def test_nth_deriv_inference_mode(n):
    vf = torch.tensor(0.75, requires_grad=True)
    f = nn.Rectified() ** 2
    with torch.inference_mode():
        numerics.compute_nth_deriv(f, vf + 1.0, n=n)


@pytest.mark.parametrize(
    "W, dx_atol",
    [
        ([[0.9004, -0.6447], [0.1701, -1.0695]], 1.0e-6),  # stable, real eigvals
        ([[0.2047, -0.8562], [1.4895, -1.5838]], 1.0e-5),  # stable, complex eigvals
    ],
)
@pytest.mark.parametrize("vf", [0.01, 0.5])
@pytest.mark.parametrize(
    "f",
    [
        "nn.Identity()",
        "nn.Identity() ** 2",
        "nn.Rectified() ** 2",
        "nn.Ricciardi(scale=0.025)",
    ],
)
@pytest.mark.parametrize("dh", [[0.5, 0.2]])
def test_perturbed_steady_state_approx(W, vf, f, dh, dx_atol):
    W, vf, dh = map(torch.tensor, (W, vf, dh))
    f = eval(f)
    W, dh = W.double(), dh.double()
    G = torch.autograd.functional.jacobian(
        f, torch.tensor([vf, vf], dtype=W.dtype, device=W.device)
    )
    J = torch.linalg.inv(torch.linalg.inv(G) - W)

    try:
        out = numerics.perturbed_steady_state_approx(vf, J, f, dh, max_num_steps=200)
    except exceptions.SimulationError:
        out_converged = False
    else:
        out_converged = True

    try:
        expected = numerics.perturbed_steady_state(
            torch.tensor(vf, dtype=W.dtype, device=W.device),
            W,
            f,
            dh,
            dx_atol=dx_atol,
            options=dict(max_num_steps=200),
        ).x
    except exceptions.SimulationError:
        pass
        # assert not out_converged
    else:
        assert out_converged
        torch.testing.assert_close(out.float(), expected.float())
