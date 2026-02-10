import functools

import torch
import numpy as np

from niarb.tensors.periodic import PeriodicTensor

sqrtpi = torch.pi**0.5


def diff(x, y):
    """
    Returns y - x. If x or y is a PeriodicTensor, group addition is used.
    """
    is_periodic = isinstance(x, PeriodicTensor) or isinstance(y, PeriodicTensor)
    inv = is_periodic and not isinstance(x, PeriodicTensor)

    if is_periodic:
        if inv:
            x, y = y, x
        out = x.ginv().gprod(y)
        if inv:
            out = out.ginv()
    else:
        out = y - x

    return out


def wrapped(f, order=3, mode="parallel"):
    """
    f is a function from R^D to R.
    Returns a wrapped version of f, f_wrapped, which takes as argument a torch.Tensor or PeriodicTensor.
    If input is a PeriodicTensor, then along circular dimensions, we have
    f_wrapped(x) = sum_{n=-(order-1)/2}^{(order-1)/2} f(x+period*n)
    Otherwise, f_wrapped(x) = f(x)
    """
    if order % 2 != 1:
        raise ValueError(f"order should be an odd integer, but got {order=}.")

    n = torch.arange(order) - (order - 1) // 2  # e.g. if order = 3, we have [-1,0,1]

    # @profile
    @functools.wraps(f)
    def f_wrapped(x, n=n):
        if not isinstance(x, PeriodicTensor) or order == 1:
            return f(x)

        device = x.device
        w_dims = x.w_dims
        n = n.to(device)
        period = x.period.to(device)

        if mode == "parallel":
            # faster but larger GPU memory usage
            for i in range(len(w_dims)):
                basis = torch.zeros_like(x)
                basis[..., w_dims[i]] = 1.0
                x = x + torch.einsum("i, ...j -> i...j", period[i] * n, basis)
            result = f(x).sum(dim=tuple(range(len(w_dims))))

        elif mode == "sequential":
            # slower but lower GPU memory usage
            w_dims = torch.as_tensor(w_dims, device=device)
            result = 0
            for k in np.ndindex((order,) * len(w_dims)):
                dx = torch.zeros(x.shape[-1], dtype=x.dtype, device=device)
                dx[w_dims] = period * (
                    torch.as_tensor(k, device=device) - (order - 1) // 2
                )
                result += f(x + dx)

        return result

    return f_wrapped
