from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from niarb import numerics


def bisect(
    func: Callable[[*tuple[Tensor, ...]], Tensor],
    a: Tensor,
    b: Tensor,
    args: tuple[Tensor, ...] = (),
    tol: float | None = None,
) -> Tensor:
    """Find a root of a function elementwise using the bisection method.

    Args:
        func: Function to find the root of
        a: Lower bound of root
        b: Upper bound of root
        args (optional): Additional arguments to pass to `func`
        tol (optional): Tolerance for root

    Returns:
        Root of the function between a and b. If a and b have the same sign, the output
        is nan.

    """
    if (b <= a).any():
        raise ValueError("b must be greater than a")

    if tol is None:
        tol = 1e-8 if a.dtype == torch.double and b.dtype == torch.double else 1e-6

    a, b, *args = torch.broadcast_tensors(a, b, *args)
    out = torch.full_like(a, torch.nan)
    fa = func(a, *args)
    fb = func(b, *args)
    valid = fa * fb < 0
    a, b, fa, fb = a[valid], b[valid], fa[valid], fb[valid]
    args = [arg[valid] for arg in args]
    while (b - a > tol).any():
        c = (a + b) / 2
        fc = func(c, *args)
        left = fa * fc < 0
        right = ~left
        b[left], fb[left] = c[left], fc[left]
        a[right], fa[right] = c[right], fc[right]
    out[valid] = (a + b) / 2
    return out


def minimize_bisect(
    func: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    a: Tensor,
    b: Tensor,
    args: tuple[Tensor, ...] = (),
    tol: float | None = None,
) -> Tensor:
    """Use bisection method to find the minimum of a scalar function.

    Args:
        func: The scalar function whose first argument is minimized. Dependence
          on other Tensor arguments MUST be passed through args.
        a: Lower bound of the minimum.
        b: Upper bound of the minimum.
        args (optional): Additional arguments to pass to `func`.
        tol (optional): Tolerance for minimum.

    Returns:
        The minimum of the function.

    """

    def grad(x, *args):
        return numerics.compute_nth_deriv(func, x, args=args)

    return bisect(grad, a, b, args=args, tol=tol)


def minimize_newton(
    func: Callable[[Tensor, *tuple[Tensor, ...]], Tensor],
    x0: Tensor,
    args: tuple[Tensor, ...] = (),
    kwargs: dict[str, Any] | None = None,
    bounds: tuple[float | None, float | None] = (None, None),
    maxiter: int = 100,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tensor:
    """Use Newton's method to find the minimum of a convex scalar function.

    Args:
        func: The convex scalar function whose first argument is minimized. Dependence
          on other Tensor arguments MUST be passed through args.
        x0: The initial guess.
        args (optional): Additional tensors to pass to `func`. The leading dimensions of
          each tensor must be broadcastable with `x0`. These tensors are vmapped over.
        kwargs (optional): Optional arguments passed to `ffunc`. Note that these
          arguments are NOT vmapped over, unlike `args`.
        bounds (optional): The bounds of the optimization.
        maxiter (optional): The maximum number of iterations.
        rtol (optional): Relative tolerance for convergence criterion based on step size.
        atol (optional): Absolute tolerance for convergence criterion based on step size.

    Returns:
        The minimum of the function.

    """
    cur = x0
    for _ in range(maxiter):
        prev = cur
        hessian = numerics.compute_nth_deriv(func, prev, args=args, kwargs=kwargs, n=2)
        if (hessian <= 0).any():
            raise ValueError("The optimized function is not convex.")

        gradient = numerics.compute_nth_deriv(func, prev, args=args, kwargs=kwargs, n=1)
        cur = prev - gradient / hessian

        if bounds != (None, None):
            cur = cur.clip(*bounds)

        if torch.allclose(cur, prev, rtol=rtol, atol=atol):
            break

    return cur
