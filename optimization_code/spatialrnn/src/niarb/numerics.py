import logging
import functools
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from niarb import integrate, linalg, exceptions
from niarb.tensors.base_circulant import BaseCirculantTensor
import numpy as np

logger = logging.getLogger(__name__)


def simulate(
    W: Tensor,
    f: Callable[[Tensor], Tensor],
    h: Tensor,
    t: Tensor | None = None,
    tau: Tensor | float = 1.0,
    x0: float | Tensor = 0.0,
    kind: str = "rate",
    **kwargs,
) -> integrate.OdeResult:
    x0 = torch.as_tensor(x0, dtype=h.dtype, device=h.device)
    x0, h = torch.broadcast_tensors(x0, h)  # (*, N)

    # as of torch 2.3.1, there is a severe performance issue with matmul
    # that causes matmul to be 10x slower than einsum. See issue #110858.
    if isinstance(W, BaseCirculantTensor):
        matmul = torch.matmul
    else:
        matmul = functools.partial(torch.einsum, "...ij,...jk->...ik")

    if kind == "rate":

        def func(_, x, W=W, f=f, h=h, tau=tau):
            return (f(matmul(W, x.unsqueeze(-1)).squeeze(-1) + h) - x) / tau

    elif kind == "voltage":

        def func(_, x, W=W, f=f, h=h, tau=tau):
            return (matmul(W, f(x).unsqueeze(-1)).squeeze(-1) + h - x) / tau

    else:
        raise ValueError(f"kind must be 'rate' or 'voltage', but got {kind}.")

    logger.debug(f"{x0.shape=}, {W.shape=}, {h.shape=}")

    if t is None:
        out = integrate.odeint_ss(func, x0, **kwargs)
    else:
        out = integrate.odeint(func, x0, t, **kwargs)

    return out


def fixed_point(
    vf: Tensor, W: Tensor, f: Callable[[Tensor], Tensor]
) -> tuple[Tensor, Tensor]:
    """Compute fixed point of the dynamical system.

    Args:
        vf: Baseline voltage tensor with shape (*, N)
        W: Connection weight tensor with shape (*, N, N).
        f: Nonlinearity.

    Returns:
        A tuple of tensors (rf, hf), where rf is the fixed point firing rate and
        hf is the fixed point input. Both have shape (*, N).

    """

    rf = f(vf)  # (*, N)
    rf, vf = rf.unsqueeze(-1), vf.unsqueeze(-1)  # (*, N, 1), (*, N, 1)

    # as of torch 2.3.1, there is a severe performance issue with matmul
    # that causes matmul to be 10x slower than einsum. See issue #110858.
    if isinstance(W, BaseCirculantTensor):
        matmul = torch.matmul
    else:
        matmul = functools.partial(torch.einsum, "...ij,...jk->...ik")

    hf = vf - matmul(W, rf) # (*, N, 1)

    # print("here")
    # print(hf[0, 0, 0])

    return rf.squeeze(-1), hf.squeeze(-1)  # (*, N), (*, N)


def perturbed_steady_state(
    vf: Tensor,
    W: Tensor,
    f: Callable[[Tensor], Tensor],
    dh: Tensor,
    dx0: float | Tensor = 0.0,
    kind: str = "rate",
    **kwargs,
) -> integrate.OdeResult:
    """Compute perturbed steady state by simulating the dynamical system.

    Args:
        vf: Baseline voltage tensor with shape (), (*, 1), or (*, N).
        W: Connection weight tensor with shape (*, N, N).
        f: Nonlinearity.
        dh: Perturbation tensor with shape (*, N).
        dx0 (optional): Initial condition of the perturbation response. If a tensor,
          must have shape (), (*, 1), or (*, N).
        kind (optional): 'rate' or 'voltage'.
        **kwargs: Optional arguments passed to integrate.odeint_ss.

    Returns:
        integrate.OdeResult

    Raises:
        ValueError: If kind is not 'rate' or 'voltage'.

    """
    if kind not in {"rate", "voltage"}:
        raise ValueError(f"kind must be 'rate' or 'voltage', but got {kind}.")

    vf, dh = torch.broadcast_tensors(vf, dh)  # (*, N), (*, N)
    rf, hf = fixed_point(vf, W, f)  # (*, N), (*, N), hf is the bias
    # print("")
    # print(hf[:4, 0, 0])
    # print(hf.shape)

    xf = rf if kind == "rate" else vf

    out = simulate(W, f, hf + dh, x0=xf + dx0, kind=kind, **kwargs)

    return integrate.OdeResult(x=out.x - xf, t=out.t, dxdt=out.dxdt)


# As of torch 2.3.1, torch.autograd.functional.jacobian outputs all zeros in
# inference_mode. Not sure if this is a bug or intended behavior, but we need
# to set inference_mode(False) to get the correct Jacobian. See github issue #128264.
@torch.inference_mode(False)
def compute_gain(f, vf, create_graph=True, **kwargs):
    if vf.ndim > 1:
        raise ValueError(f"vf must be 0 or 1-dimensional, but {vf=}.")

    vf = vf.clone()  # needed if vf is the output of code that is run in inference mode

    jac = torch.autograd.functional.jacobian(
        f, vf, create_graph=create_graph, **kwargs
    )  # defaults to create_graph=True because we need to backprop through it

    if jac.ndim not in {0, 2} or (jac.ndim == 2 and jac.shape[0] != jac.shape[1]):
        raise ValueError(
            f"output of f must have same shape as its input, but {jac.shape=}"
        )

    if jac.ndim == 2 and not linalg.is_diagonal(jac):
        raise ValueError(
            "f must be an element-wise function, but its Jacobian is not diagonal."
        )

    return jac.diagonal() if jac.ndim == 2 else jac  # (N,) or ()


# Compared to compute_gain, compute_nth_deriv is much faster for large inputs.
# However compute_nth_deriv requires a bit of warmup time.
@torch.inference_mode(False)  # see comment in compute_gain
def compute_nth_deriv(
    f: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    vf: Tensor,
    args: tuple[Tensor, ...] = (),
    kwargs: dict[str, Any] | None = None,
    n: int = 1,
) -> Tensor:
    """Compute the nth derivative of a scalar function.

    Args:
        f: A scalar function.
        vf: Input tensors at which the derivative is evaluated.
        args (optional): Additional tensors to pass to `f`. The leading dimensions of
          each tensor must be broadcastable with `vf`. These tensors are vmapped over.
        kwargs (optional): Optional arguments passed to `f`. Note that these arguments
          are NOT vmapped over, unlike `args`.
        n (optional): Order of the derivative.

    Returns:
        Derivative tensor with shape broadcast(vf.shape, *[arg.shape[:vf.ndim] for arg in args]).

    """
    if n < 1:
        raise ValueError(f"n must be a positive integer, but {n=}.")

    if kwargs is None:
        kwargs = {}

    shape = torch.broadcast_shapes(vf.shape, *[arg.shape[vf.ndim :] for arg in args])
    vf = vf.broadcast_to(shape)
    args = [arg.broadcast_to((*shape, *arg.shape[vf.ndim :])) for arg in args]

    if hasattr(f, "nth_deriv"):
        return f.nth_deriv(n, vf, *args, **kwargs)

    for _ in range(n):
        f = torch.func.grad(f)

    for _ in range(vf.ndim):
        f = torch.func.vmap(f)

    return f(vf, *args, **kwargs)


def perturbed_steady_state_approx(
    vf: float | Tensor,
    J: Tensor,
    f: Callable[[Tensor], Tensor],
    dh: Tensor,
    gain: Tensor | None = None,
    kind: str = "rate",
    max_num_steps: int = 100,
    assert_convergence: bool = True,
    gain_kwargs: dict | None = None,
    max_dv_norm: float = 1e8,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tensor:
    r"""Compute approximate perturbed steady state with a recursive algorithm.

    Args:
        vf: Baseline voltage.
        J: Jacobian matrix $\frac{dv_i}{dh_j}$ if kind == 'voltage' else $\frac{dr_i}{dh_j}$.
          Note that in terms of the weights W and gain G matrices, we have J = (I - WG)^{-1} if
          kind == 'voltage' else J = (I - GW)^{-1}G = (G^{-1} - W)^{-1} = G(I - WG)^{-1}.
        f: Nonlinearity.
        dh: Perturbation tensor with shape (*, N).
        gain (optional): Tensor with shape ().
        kind (optional): 'rate' or 'voltage'.
        max_num_steps (optional): Maximum number of steps for the recursive algorithm.
        assert_convergence (optional): Whether to raise an error if the algorithm does not converge.
        gain_kwargs (optional): Optional arguments passed to compute_gain if gain is None.
        max_dv_norm (optional): Maximum norm of the perturbation voltage vector before termination.
        rtol, atol (optional): tolerances for convergence criterion.

    Returns:
        Perturbation response tensor with shape (*, N).

    Raises:
        NotImplementedError: If vf and gain are not scalar tensors.
        ValueError: If kind is not 'rate' or 'voltage', or max_num_steps is not a positive integer.
        SimulationError: If assert_convergence is True and the algorithm fails to converge due
          to one of the following reasons: the perturbation response vector has NaN values,
          the norm of the perturbation voltage vector exceeds max_dv_norm, or the convergence
          criterion is not satisfied within max_num_steps.

    """
    if isinstance(vf, float):
        vf = torch.tensor(vf, dtype=dh.dtype, device=dh.device)  # ()

    if vf.ndim != 0:
        raise NotImplementedError(
            f"Currently only accepts 0-dimensional vf, but got {vf.ndim=}."
        )

    if kind not in {"rate", "voltage"}:
        raise ValueError(f"kind must be 'rate' or 'voltage', but got {kind}.")

    if not isinstance(max_num_steps, int) or max_num_steps <= 0:
        raise ValueError(
            f"max_num_steps must be a positive integer, but {max_num_steps=}."
        )

    if gain is None:
        gain = compute_gain(f, vf, **(gain_kwargs if gain_kwargs else {}))  # ()

    if gain.ndim != 0:
        raise NotImplementedError(
            f"Currently only accepts 0-dimensional gain, but got {gain.ndim=}."
        )

    if kind == "rate":
        J = J / gain  # (*, N, N)

    I = linalg.eye_like(J)  # (N, N)
    T = (J - I) / gain  # (*, N, N)

    rf = f(vf)  # ()
    dh = dh.unsqueeze(-1)  # (*, N, 1)
    dv = J @ dh  # (*, N, 1)
    dr = f(vf + dv) - rf  # (*, N, 1)
    dv1 = dv.clone()  # (*, N, 1)

    # as of torch 2.3.1, there is a severe performance issue with matmul
    # that causes matmul to be 10x slower than einsum. See issue #110858.
    if isinstance(T, BaseCirculantTensor):
        matmul = torch.matmul
    else:
        matmul = functools.partial(torch.einsum, "...ij,...jk->...ik")

    logger.debug(f"{dv.shape=}, {dr.shape=}, {gain.shape=}, {T.shape=}")

    converged, n = False, 1
    while (
        n < max_num_steps
        and not converged
        and dr.isfinite().all()
        and dv.norm() <= max_dv_norm
    ):
        prev_dv, prev_dr = dv, dr  # (*, N, 1), (*, N, 1)
        dv = dv1 + matmul(T, dr - gain * dv)  # (*, N, 1)
        dr = f(vf + dv) - rf  # (*, N, 1)

        if logger.isEnabledFor(logging.DEBUG):
            # prevent unnecessary GPU-CPU sync which may impact performance
            # by only executing the debug statement if DEBUG level is enabled
            logger.debug(f"{(dr - prev_dr).norm().item()}, {dr.norm().item()}")

        if kind == "voltage":
            converged = torch.allclose(dv, prev_dv, rtol=rtol, atol=atol)
        else:
            converged = torch.allclose(dr, prev_dr, rtol=rtol, atol=atol)

        n += 1

    if assert_convergence and not converged:
        raise exceptions.SimulationError(
            f"Failed to converge with {rtol=}, {atol=} within {max_num_steps=} steps."
        )

    if kind == "voltage":
        return dv.squeeze(-1)
    return dr.squeeze(-1)
