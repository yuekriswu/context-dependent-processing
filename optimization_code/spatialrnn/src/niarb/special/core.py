from collections.abc import Sequence, Callable

import torch
from torch import Tensor
import torch_bessel
import numpy as np
from scipy import special as scipy_special
from scipy.special import factorial

from niarb.tensors.periodic import PeriodicTensor

sqrtpi = torch.pi**0.5

__all__ = [
    "uniform",
    "uniform_",
    "normal",
    "von_mises",
    "ubeta",
    "wrapped",
    # "factorial",
    "yukawa",
    "k0",
    "k1",
    "kd",
    "scaled_kd",
    "irkd",
    "solid_angle",
    "ball_volume",
    "ball_radius",
]

BASIC_DTYPES = {torch.float, torch.cfloat, torch.double, torch.cdouble}


def uniform(extent: float | Sequence[float] | Tensor, x: Tensor) -> Tensor:
    """A function which outputs 1 if x is in the 'box' defined by extent and 0 otheriwse.

    Args:
        extent: Width(s) of the box
        x: Input tensor

    Returns:
        Tensor with same shape as x.

    """
    extent = torch.as_tensor(extent)
    return uniform_(-extent / 2, extent / 2, x)


def uniform_(
    low: float | Sequence[float] | Tensor,
    high: float | Sequence[float] | Tensor,
    x: Tensor,
) -> Tensor:
    """A function which outputs 1 if x is in the 'box' defined by (low, high) and 0 otheriwse.

    Args:
        low: Lower bound(s) of the box
        high: Upper bound(s) of the box
        x: Input tensor

    Returns:
        Tensor with same shape as x.

    """
    low = torch.as_tensor(low, dtype=x.dtype, device=x.device)
    high = torch.as_tensor(high, dtype=x.dtype, device=x.device)
    return ((low < x) & (x < high)).to(x.dtype)


def normal(scale: float | Sequence[float] | Tensor, x: Tensor, **kwargs) -> Tensor:
    """Probability density function of the normal distribution with standard deviation scale.

    Args:
        scale: standard deviation of the normal distribution.
        x: Input tensor
        **kwargs: Additional arguments to torch.distributions.Normal

    Returns:
        Tensor with same shape as x.

    """
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    return torch.distributions.Normal(0.0, scale, **kwargs).log_prob(x).exp()


def von_mises(kappa: float | Sequence[float] | Tensor, x: Tensor, **kwargs) -> Tensor:
    """Probability density function of the von Mises distribution with concentration kappa.

    Args:
        kappa: Concentration parameter of the von Mises distribution.
        x: Input tensor
        **kwargs: Additional arguments to torch.distributions.VonMises

    Returns:
        Tensor with same shape as x.

    """
    kappa = torch.as_tensor(kappa, dtype=x.dtype, device=x.device)
    if isinstance(x, PeriodicTensor):
        x = x.to_period(2 * torch.pi)
    # Note: pass validate_args=False to allow for kappa = 0
    return (
        torch.distributions.VonMises(0.0, kappa, validate_args=False, **kwargs)
        .log_prob(x)
        .exp()
    )


def ubeta(
    alpha: float | Sequence[float] | Tensor,
    beta: float | Sequence[float] | Tensor,
    x: Tensor,
) -> Tensor:
    """Unnormalized probability density function of the Beta distribution.

    Args:
        alpha: Alpha parameter of the Beta distribution.
        beta: Beta parameter of the Beta distribution.
        x: Input tensor

    Returns:
        Tensor with same shape as x.

    """
    alpha = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
    beta = torch.as_tensor(beta, dtype=x.dtype, device=x.device)
    return x ** (alpha - 1) * (1 - x) ** (beta - 1)


def wrapped(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    order: int = 3,
    mode: str = "parallel",
) -> Tensor:
    """Computes the wrapped version of f.

    Assumes f maps R^d to R, i.e. if x.shape = (*shape, d), then f(x).shape = shape.
    If input is a PeriodicTensor, then along circular dimensions, we have
    wrapped(f, x) = sum_{n=-(order-1)/2}^{(order-1)/2} f(x+period*n)
    Otherwise, wrapped(f, x) = f(x).

    Args:
        f: Function to wrap
        x: Input tensor
        order: Order of the wrapping
        mode: {'parallel', 'sequential'}. Method of wrapping.

    Returns:
        Tensor with shape f(x).shape

    """
    if order % 2 != 1:
        raise ValueError(f"order should be an odd integer, but got {order=}.")

    if not isinstance(x, PeriodicTensor):
        return f(x)

    device = x.device
    w_dims = x.metadata["w_dims"]
    period = x.period.to(device)
    n = (
        torch.arange(order, device=device) - (order - 1) // 2
    )  # e.g. if order = 3, we have [-1,0,1]

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
            dx[w_dims] = period * (torch.as_tensor(k, device=device) - (order - 1) // 2)
            result += f(x + dx)

    return result


# def factorial(x: Tensor) -> Tensor:
#     return (x + 1).lgamma().exp()


class Yukawa(torch.autograd.Function):
    @staticmethod
    def forward(a: Tensor, r: Tensor, s: Tensor) -> tuple[Tensor, Tensor]:
        mask = r != 0
        grad = (a * r).neg_().exp_()
        return (grad / r).where(mask, s), mask, grad

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        tensors = []
        if any(ctx.needs_input_grad):
            tensors.append(outputs[1])
        if any(ctx.needs_input_grad[:-1]):
            tensors.append(outputs[2])
        if ctx.needs_input_grad[1]:
            tensors += inputs[:-1]
        ctx.save_for_backward(*tensors)
        ctx.set_materialize_grads(False)
        ctx.dtype_a = inputs[0].dtype
        ctx.dtype_r = inputs[1].dtype
        ctx.dtype_s = inputs[2].dtype

    @staticmethod
    def backward(ctx, grad_output, _, __):
        # Note: We need to take the conjugate of the derivative. To understand why,
        # see https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc
        # where in equation 4 we note that since this is a holomorphic function,
        # the derivative w.r.t conjugate of z is zero, so the first term is zero.
        if grad_output is None:
            return None, None, None

        grad_a, grad_r, grad_s = None, None, None

        if any(ctx.needs_input_grad):
            mask = ctx.saved_tensors[0]

        if any(ctx.needs_input_grad[:-1]):
            grad = ctx.saved_tensors[1]
            grad_a = (-grad).conj().mul_(grad_output)

        if ctx.needs_input_grad[1]:
            a, r = ctx.saved_tensors[2:]
            grad_r = r.reciprocal()
            grad_r = (a + grad_r).mul_(grad_r).conj().mul_(grad_a)

        if grad_a is not None:
            grad_a = grad_a.where(mask, 0.0).to(ctx.dtype_a)

        if grad_r is not None:
            grad_r = grad_r.where(mask, 0.0).to(ctx.dtype_r)

        if ctx.needs_input_grad[2]:
            grad_s = grad_output.where(~mask, 0.0).to(ctx.dtype_s)

        return grad_a, grad_r, grad_s


def yukawa(a: Tensor, r: Tensor, singularity: float | Tensor = 0.0) -> Tensor:
    r"""Efficient computation of the yukawa potential $\frac{e^{-ar}}{r}$.

    Elements of the output tensor where r = 0 is set to singularity to avoid Infs in output and
    NaNs during backprop.

    This function is equivalent to `torch.where(r == 0, singularity, (-a * r).exp() / r)`
    but is much more memory-efficient (and also time-efficient) when only gradient with
    respect to `a` is required.

    Args:
        a, r, singularity: Tensors with arbitrary shapes that are broadcastable with each other.

    Returns:
        Tensor with shape broadcast_tensors(a, r).shape.

    """
    return Yukawa.apply(a, r, torch.as_tensor(singularity))[0]


class ModifiedBesselK1(torch.autograd.Function):
    @staticmethod
    def forward(z):
        if z.is_complex():
            out = scipy_special.kv(1, z.detach().cpu().numpy())
            out = torch.as_tensor(out).to(dtype=z.dtype, device=z.device)
        else:
            out = torch.special.modified_bessel_k1(z)
        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(*inputs, output)
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None or not ctx.needs_input_grad[0]:
            return None

        z, out = ctx.saved_tensors
        if z.is_complex():
            # Note: We need to take the conjugate of the derivative. To understand why,
            # see https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc
            # where in equation 4 we note that since this is a holomorphic function,
            # the derivative w.r.t conjugate of z is zero, so the first term is zero.
            grad = -scipy_special.kv(0, z.detach().cpu().numpy())
            grad = torch.as_tensor(grad).to(dtype=z.dtype, device=z.device)
            grad = (grad - out / z).conj()
        else:
            grad = -torch.special.modified_bessel_k0(z) - out / z

        return grad.mul_(grad_output)

    @staticmethod
    def vmap(info, in_dims, z):
        in_dim = in_dims[0]
        if in_dim is not None:
            z = z.movedim(in_dim, 0)

        out = ModifiedBesselK1.apply(z)
        return (out, 0 if in_dim is not None else None)


def k0(z: Tensor, **kwargs) -> Tensor:
    dtype = None
    if z.dtype not in BASIC_DTYPES:
        dtype = z.dtype
        z = z.to(torch.cfloat) if z.is_complex() else z.to(torch.float)
    out = torch_bessel.ops.modified_bessel_k0(z, **kwargs)
    if dtype:
        out = out.to(dtype)
    return out


def k1(z: Tensor) -> Tensor:
    # manually implement autograd since torch.special.modified_bessel_k1
    # is not differentiable
    dtype = None
    if z.dtype not in BASIC_DTYPES:
        dtype = z.dtype
        z = z.to(torch.cfloat) if z.is_complex() else z.to(torch.float)
    out = ModifiedBesselK1.apply(z)
    if dtype:
        out = out.to(dtype)
    return out


def kd(d: int, z: Tensor) -> Tensor:
    r"""Modified Bessel function of the second kind of order d/2-1.

    Computes K_{d/2-1}(z), where K_\nu is the modified Bessel function of the second kind.

    Args:
        d: 'Dimension' parameter, such that the order of the Bessel function is d/2-1.
        z: Input tensor

    Returns:
        Tensor with same shape as z.

    """
    if d < 2:
        # K_\nu(z) = K_{-\nu}(z)
        d = 4 - d

    if d % 2 == 1:
        out = z**-0.5 * scaled_kd(d, z)
    elif d == 2:
        out = k0(z)
    elif d in {0, 4}:
        out = k1(z)
    else:
        # use recurrence relationship of Bessel functions
        # see https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/introductions/Bessels/ShowAll.html
        out = kd(d - 4, z) + 2 * (d / 2 - 2) / z * kd(d - 2, z)

    return out


def scaled_kd(d: int, z: Tensor) -> Tensor:
    r"""Scaled modified Bessel function of the second kind of order d/2-1 for odd integers d.

    Computes z**0.5 * K_{d/2-1}(z), where K_\nu is the modified Bessel function of the second kind,
    for odd integers d. Slightly faster than kd. Moreover, since the output is real for all real z,
    negative real z with non-complex dtype does not output nan, unlike kd.

    Args:
        d: 'Dimension' parameter, such that the order of the Bessel function is d/2-1.
        z: Input tensor

    Returns:
        Tensor with same shape as z.

    """
    if not isinstance(d, int) or d % 2 == 0:
        raise ValueError(f"d must be an odd integer, but {d=}.")

    if d < 2:
        # K_\nu(z) = K_{-\nu}(z)
        d = 4 - d

    # See https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/introductions/Bessels/ShowAll.html
    out = (torch.pi / 2) ** 0.5 * torch.exp(-z)

    if d not in {1, 3}:
        nu = int(abs(d / 2 - 1) - 1 / 2)
        c = 1
        for j in range(1, nu + 1):
            c += factorial(j + nu) / (factorial(j) * factorial(nu - j)) / (2 * z) ** j
        out *= c

    return out


def irkd(d: int, a: Tensor) -> Tensor:
    r"""Integral of $r^{d/2}K_{d/2-1}(r)$ from 0 to a.

    By using the recurrence relation $(1/z d/dz)^k (z^n Z_n(z)) = z^{n-k} Z_{n-k}(z)$
    where $Z_n(z) = e^(n\pi i)K_n(z)$ (Abramowitz 9.6.28), and computing the limit as
    z -> 0 with asymptotic expressions for K_n(z), the integral can be computed as
    $2^{d/2-1} \Gamma(d/2) - a^{d/2} K_{d/2}(a)$.

    Args:
        d: 'Dimension' parameter, such that the order of the Bessel function is d/2-1.
        a: Upper limits of the integral, may be complex.

    Returns:
        Tensor with same shape as a.

    """
    return 2 ** (d / 2 - 1) * scipy_special.gamma(d / 2) - a ** (d / 2) * kd(d + 2, a)


def solid_angle(d: int) -> float:
    r"""Compute the solid angle subtended by the (d-1)-sphere.

    The solid angle subtended by the (d-1)-sphere is given by the formula
    $\Omega_d = 2 \pi^{d/2} / \Gamma(d/2)$.

    Args:
        d: Dimension of the Euclidean space.

    Returns:
        Solid angle

    """
    return 2 * torch.pi ** (d / 2) / scipy_special.gamma(d / 2)


def ball_volume(d: int, r: Tensor | float) -> Tensor | float:
    """Compute the volume of a d-ball with given radius.

    Args:
        d: Dimension of the ball, must be a non-negative integer.
        r: Radius of the d-ball.

    Returns:
        Volume of the d-ball

    """
    return r**d * (torch.pi ** (d / 2) / scipy_special.gamma(d / 2 + 1))


def ball_radius(d: int, vol: Tensor | float) -> Tensor | float:
    """Compute the radius of a d-ball with given volume.

    Args:
        d: Dimension of the ball, must be a positive integer.
        vol: Volume of the d-ball.

    Returns:
        Radius of the d-ball

    """
    return (vol * (scipy_special.gamma(d / 2 + 1) / torch.pi ** (d / 2))) ** (1 / d)
