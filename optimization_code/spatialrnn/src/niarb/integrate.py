import collections
import logging

import torch
import torchdiffeq
import hyclib as lib

from niarb import exceptions

logger = logging.getLogger(__name__)

ADAPTIVE_SOLVERS = ["dopri8", "dopri5", "bosh3", "adaptive_heun"]

OdeResult = collections.namedtuple("OdeResult", ["x", "t", "dxdt"])


def odeint(
    func, x0, t, *args, event_fn=None, odeint_interface=torchdiffeq.odeint, **kwargs
):
    """
    Solves an ODE with input *args and initial state x0 at time stamps t.
    Args:
        func: function that computes the derivative of x at time t
        x0: a tensor with arbitrary shape (*).
        t: If event_fn=None, t is a 1-dimensional tensor that specifies the time stamps
           at which the ODE is solved, and whose first element represents the initial time,
           or a 0-dimensional tensor that specifies the end time, with the initial time interpreted as 0.
           If event_fn is not None, t must be a 0-dimensional tensor that specifies start time.
        *args: Additional inputs passed to func.
        event_fn: Function that maps (t, x, *args) to a Tensor. The solve terminates when
                  any element of event_fn(t, x, *args) evaluates to zero.
        odenint_interface: either torchdiffeq.odeint or torchdiffeq.odeint_adjoint
        **kwargs: additional keyword arguments passed to odeint_interface

    Returns:
        out: OdeResult namedtuple
            out.x: If event_fn is None:
                       If t is a 1-dimensional tensor with length L, then
                       x is a tensor with shape (L, *) with x[i] being the state at time t[i].
                       If t is a 0-dimensional tensor, then x is tensor with shape (*)
                       that represents the state at time t.
                   else:
                       x is tensor with shape (*) that represents the state at the event time
            out.t: If event_fn is None:
                       t is the same as the input t
                   else:
                       t is the event time.
            out.dxdt: dxdt at every time point in out.t. Has same shape as out.x.
    """
    if "options" not in kwargs or (
        "max_num_steps" not in kwargs["options"] and "method" in ADAPTIVE_SOLVERS
    ):
        # set max_num_steps to 100 if it is not provided and method is not a fixed solver.
        kwargs["options"] = {"max_num_steps": 1000} # default 100, Kris updated to 1000

    if event_fn is None:
        if t.ndim == 0:
            _t = torch.tensor([0, t.item()], dtype=t.dtype, device=t.device)
        elif t.ndim == 1:
            _t = t
        else:
            raise ValueError(
                f"t must be a one or zero dimensional tensor, but {t.shape=}."
            )

        t0 = _t[0]

    else:
        if t.ndim != 0:
            raise ValueError(
                f"t must be a zero dimensional tensor when event_fn is None, but {t.shape=}."
            )

        t0 = t

    func = lib.functools.rpartial(func, *args)

    try:
        if event_fn is None:
            x = odeint_interface(func, x0, _t, **kwargs)  # (L, *)

            if t.ndim == 0:
                x = x[-1]

        else:
            event_fn = lib.functools.rpartial(event_fn, *args)
            t, x = torchdiffeq.odeint_event(
                func,
                x0,
                t0,
                event_fn=event_fn,
                odeint_interface=odeint_interface,
                **kwargs,
            )  # x - (L, *)
            x = x[-1]

        return OdeResult(x, t, func(t, x))

    except AssertionError as err:
        # Catches vague error messages in torchdiffeq.odeint and re-raises
        # the error with a more appropriate and informative error message.
        if str(err).startswith("underflow in dt"):
            raise exceptions.SimulationError(
                "Simulated dynamical system is too stiff."
            ) from err
        if str(err).startswith("max_num_steps exceeded"):
            raise exceptions.SimulationError(
                'Simulation steps exceeded max_num_steps. Try increasing options["max_num_steps"].'
            ) from err
        raise


def odeint_ss(
    func,
    x0,
    *args,
    max_t=200.0, # previous default is 100
    max_dxdt=torch.inf,
    assert_convergence=True,
    dx_rtol=1.3e-6, #1.3e-7,
    dx_atol=1.0e-4,#1.0e-6,
    **kwargs,
):
    """
    Integrates ODE until steady state.
    Args:
        func, x0, args: same as odeint
        max_t: maximum ODE system time allowed before termination
        max_dxdt: maximum absolute value of dxdt allowed before termination
        assert_convergence: If True, raise error if max_t is exceeded.
        dx_rtol, dx_atol: ODE is considered at steady state when (dxdt.abs() <= x.abs() * dx_rtol + dx_atol).all()
        Note that due to the limitation of torchdiffeq, the state at the time right BEFORE the steady state condition
        is satisfied is returned, not AFTER, so the result will NOT strictly satisfy the steady state condition.
        **kwargs: additional keyword arguments passed to odeint_interface
    Returns:
        out: OdeResult namedtuple
            out.x: tensor with shape x0.shape representing the steady state.
            out.t: scalar tensor representing termination time.
            out.dxdt: tensor with shape x0.shape representing dxdt at steady state.
    """
    if max_t <= 0:
        raise ValueError("max_t must be greater than 0.")

    has_exceeded_max_t, has_exceeded_max_dxdt = False, False

    def event_fn(
        t, x, *args, dx_rtol=dx_rtol, dx_atol=dx_atol, max_t=max_t, max_dxdt=max_dxdt
    ):
        nonlocal has_exceeded_max_t, has_exceeded_max_dxdt

        dxdt = func(t, x, *args)
        allclose = (dxdt.abs() <= x.abs() * dx_rtol + dx_atol).all()
        exceeded_max_t = t > max_t
        exceeded_max_dxdt = (dxdt.abs() > max_dxdt).any()
        result = (
            1.0 - (allclose | exceeded_max_t | exceeded_max_dxdt).float()
        ).unsqueeze(0)

        if logger.isEnabledFor(logging.DEBUG):
            # prevent unnecessary GPU-CPU sync which may impact performance
            # by only executing the debug statement if DEBUG level is enabled
            logger.debug(f"{dxdt.mean().item()}, {x.mean().item()}, {t.item()}")

        has_exceeded_max_t |= exceeded_max_t
        has_exceeded_max_dxdt |= exceeded_max_dxdt

        return result

    # torchdiffeq works with double precision time, so set t0 dtype to double,
    # otherwise torchdiffeq will convert output time to float, losing the required precision
    # unless we are using MPS, in which case since MPS doesn't support double we just use float
    t0 = torch.tensor(
        0.0,
        dtype=torch.double if x0.device.type != "mps" else torch.float,
        device=x0.device,
    )

    out = odeint(func, x0, t0, *args, event_fn=event_fn, **kwargs)

    if assert_convergence and has_exceeded_max_t:
        raise exceptions.SimulationError(
            f"Failed to converge to steady state with {dx_rtol=}, {dx_atol=} within maximum time {max_t}."
        )

    if has_exceeded_max_dxdt:
        raise exceptions.SimulationError(
            f"dxdt exceeded maximum allowed value {max_dxdt}."
        )

    if out.t == t0:
        raise exceptions.SimulationError(
            f"Initial state already satisfies {dx_rtol=} and {dx_atol=}."
        )

    return out
