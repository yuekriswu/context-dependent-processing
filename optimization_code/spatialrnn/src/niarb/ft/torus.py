import collections

import torch


def irftn(a, x, dim=None, norm="backward", period=2 * torch.pi):
    r"""
    Inverse real Fourier transform on the d-torus. Since Fourier transform
    of a real signal has Hermitian symmetry, i.e. f(k) = \overline{f(-k)},
    where k is a vector, half of the Fourier coefficients are redundant,
    so assume the last dimension of the Fourier transformed dims to be Hermitian.
    Like torch.fft.irfftn and np.fft.irfftn, imaginary part of coefficients
    which must be real are ignored (for example, the zero frequency component
    must always be real due to Hermitian symmetry).
    Args:
        a: torch.Tensor with shape (*). Fourier coefficients of a real function on the d-torus.
        x: torch.Tensor with shape (**, d).  Locations at which to evaluate the output function.
        dim: Iterable[int], optional. Dimensions along which to perform the inverse real Fourier transform.
             If None, all dimensions are used. Assumes that only the positive frequencies are provided
             along the last dim.
        norm: {'forward', 'ortho', 'backward'}, optional.
            Let period = (period,) * d if period is not a sequence of floats
            If norm == 'backward', scale by 1 / torch.prod(period).
            If norm == 'ortho', scale by 1 / torch.prod(period) ** 0.5.
            If norm == 'forward', no scaling.
        period: float | Iterable[float], optional. Period along each dimension. If an Iterable,
                then the number of elements must match number of dimensions.
    Returns:
        out: torch.Tensor with shape broadcast((s for s in a.shape if s not in dim), **)
    """
    if dim is None:
        dim = range(a.ndim)
    dim = tuple(dim)

    if not isinstance(period, collections.abc.Iterable):
        period = (period,) * len(dim)
    else:
        period = tuple(period)

    for i in range(len(dim) - 1):
        a = ift(a, x[..., i], dim=dim[i], norm=norm, period=period[i])
    a = irft(a, x[..., -1], dim=dim[-1], norm=norm, period=period[-1])

    return a


def irft(a, x, dim=-1, norm="backward", period=2 * torch.pi):
    r"""
    Inverse real Fourier transform on the 1-torus. Since Fourier transform
    of a real signal has Hermitian symmetry, input Fourier coefficients are
    taken to be the coefficients corresponding non-negative frequencies.
    Like torch.fft.irfft and np.fft.irfft, imaginary part of the first coefficient
    is ignored, since the first coefficient must be real in order to satisfy symmetry.
    Args:
        a: torch.Tensor with shape (*). Fourier coefficients of a real function on the 1-torus.
        x: torch.Tensor with shape (**).  Locations at which to evaluate the output function.
        dim: int, optional. Dimension along which to perform the inverse real Fourier transform.
        norm: {'forward', 'ortho', 'backward'}, optional.
            If norm == 'backward', scale by 1 / period.
            If norm == 'ortho', scale by 1 / period ** 0.5.
            If norm == 'forward', no scaling.
        period: float, optional. Period of the 1-torus.
    Returns:
        out: torch.Tensor with shape broadcast((s for s in a.shape if s != dim), **)
    """
    if norm not in {"forward", "ortho", "backward"}:
        raise ValueError(
            f"norm must be one of {'forward', 'ortho', 'backward'}, but got {norm=}."
        )

    # move dim to the end
    if dim != -1:
        a = a.movedim(dim, -1)

    # account for Hermitian symmetry in the Fourier coefficients
    a = a.clone()
    a[..., 1:] = 2 * a[..., 1:]

    k = torch.arange(a.shape[-1], dtype=x.dtype, device=x.device)  # (a.shape[-1],)
    theta = torch.tensordot(2 * torch.pi * x / period, k, dims=0)  # (**, a.shape[-1])

    out = a.real * torch.cos(theta)
    if a.is_complex():
        out = out - a.imag * torch.sin(theta)
    out = out.sum(dim=-1)

    if norm == "backward":
        out = out / period
    elif norm == "ortho":
        out = out / period**0.5

    return out


def ift(a, x, dim=-1, norm="backward", period=2 * torch.pi):
    r"""
    Inverse Fourier transform on the 1-torus. Similar to torch.fft and np.fft, the coefficients
    are interpreted as such (let n = a.shape[dim], and we index along dim):
        a[0] - zero frequency term
        a[1:n//2 + 1] - positive frequency terms
        a[n//2 + 1:] - negative frequency terms, in increasing order from most negative frequency

    Args:
        a: torch.Tensor with shape (*). Fourier coefficients of a complex function on the 1-torus.
        x: torch.Tensor with shape (**).  Locations at which to evaluate the output function.
        dim: int, optional. Dimension along which to perform the inverse Fourier transform.
        norm: {'forward', 'ortho', 'backward'}, optional.
            If norm == 'backward', scale by 1 / period.
            If norm == 'ortho', scale by 1 / period ** 0.5.
            If norm == 'forward', no scaling.
        period: float, optional. Period of the 1-torus.
    Returns:
        out: torch.Tensor with shape broadcast((s for s in a.shape if s != dim), **)
    """
    if norm not in {"forward", "ortho", "backward"}:
        raise ValueError(
            f"norm must be one of {'forward', 'ortho', 'backward'}, but got {norm=}."
        )

    # move dim to the end
    if dim != -1:
        a = a.movedim(dim, -1)

    n = a.shape[-1]
    k_pos = torch.arange(n // 2 + 1, device=x.device).float()
    k_neg = torch.arange(n // 2 + 1 - n, 0, device=x.device).float()
    k = torch.cat([k_pos, k_neg])
    theta = torch.tensordot(2 * torch.pi * x / period, k, dims=0)  # (**, a.shape[-1])
    out = (a * torch.exp(1.0j * theta)).sum(dim=-1)

    if norm == "backward":
        out = out / period
    elif norm == "ortho":
        out = out / period**0.5

    return out
