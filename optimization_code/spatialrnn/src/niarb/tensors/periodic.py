import itertools
import functools
from collections.abc import Iterable
from typing import Self

import torch
from torch import Tensor

from .base import tensor_class_factory, _rebuild, BaseTensor
from niarb import utils


def tensor(
    data,
    dtype=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
    w_dims=None,
    extents=None,
):
    data = torch.tensor(data, dtype=dtype, device=device, pin_memory=pin_memory)
    return _instantiate(
        data, requires_grad=requires_grad, w_dims=w_dims, extents=extents
    )


def as_tensor(data, dtype=None, device=None, w_dims=None, extents=None):
    data = torch.as_tensor(data, dtype=dtype, device=device)
    return _instantiate(data, w_dims=w_dims, extents=extents)


def linspace(start, stop, /, num=50, **kwargs):
    return as_tensor(
        torch.linspace(start, stop, num + 1, **kwargs)[:-1, None],
        extents=[(start, stop)],
    )


def _instantiate(data, requires_grad=False, w_dims=None, extents=None):
    if data.ndim == 0:
        raise ValueError(f"data must be at least 1 dimensional, but {data.ndim=}.")

    if w_dims is None:
        w_dims = tuple(range(data.shape[-1]))
    w_dims = utils.normalize_dim(w_dims, data.shape[-1])

    if extents is None:
        extents = [(-torch.pi, torch.pi)] * len(w_dims)

    # cast to tuples to ensure immutability
    w_dims, extents = tuple(w_dims), tuple(extents)
    cls = tensor_class_factory(PeriodicTensor, w_dims=w_dims, extents=extents)

    return cls(data, requires_grad=requires_grad)


class PeriodicTensor(BaseTensor):
    def __init__(self, *args, **kwargs):
        if self.ndim == 0:
            raise ValueError(
                f"PeriodicTensor must be at least 1 dimensional, but {self.ndim=}."
            )

        D = self.D
        w_dims, extents = self.w_dims, self.extents

        if not all(isinstance(v, int) and 0 <= v < D for v in w_dims):
            raise ValueError(
                f"w_dims must be a sequence of non-negative integers less than {D=}, but {w_dims=}."
            )

        if len(extents) != len(w_dims):
            raise ValueError(
                f"extents must be a sequence with length {len(w_dims)=}, but {len(extents)=}."
            )

        eps = 1.0e-5
        if (
            not (
                (self.low - eps <= self[..., w_dims]) | self[..., w_dims].isnan()
            ).all()
            or not (
                (self[..., w_dims] <= self.high + eps) | self[..., w_dims].isnan()
            ).all()
        ):
            domain = " X ".join(f"[{low}, {high}]" for (low, high) in extents)
            low = self.reshape(-1, D).min(dim=0).values
            high = self.reshape(-1, D).max(dim=0).values
            raise ValueError(f"data must be in {domain}, but got {low=}, {high=}.")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in HANDLED_FUNCTIONS:
            result = HANDLED_FUNCTIONS[func](cls, types, args=args, kwargs=kwargs)

        else:
            result = super().__torch_function__(func, types, args, kwargs)

        return result

    def __reduce_ex__(self, proto):
        # override __reduce_ex__ to support pickling
        func, args = super().__reduce_ex__(proto)
        args = list(args)
        args[1] = {"w_dims": self.w_dims, "extents": self.extents}
        return _rebuild, (func, tuple(args), PeriodicTensor)

    @property
    def D(self):
        return self.shape[-1]

    @property
    def _w_dims(self):
        """Return a slice representation of w_dims if possible for memory efficient indexing"""
        w_dims = self.w_dims
        if len(w_dims) > 0 and w_dims == tuple(range(w_dims[0], w_dims[-1] + 1)):
            return slice(w_dims[0], w_dims[-1] + 1)
        return list(w_dims)

    @property
    def period(self):
        return self.high - self.low

    @property
    def low(self):
        return torch.tensor([extent[0] for extent in self.extents], device=self.device)

    @property
    def high(self):
        return torch.tensor([extent[1] for extent in self.extents], device=self.device)

    def ginv(self):
        return -self

    def gprod(self, other: Tensor) -> Self:
        out = self + other
        w_dims = out._w_dims
        if isinstance(w_dims, slice):
            # memory efficient path by performing in-place operations
            out[..., w_dims].sub_(out.low).remainder_(out.period).add_(out.low)
        else:
            out[..., w_dims] = ((out[..., w_dims] - out.low) % out.period) + out.low
        return out

    def to_period(self, period: float | torch.Tensor) -> Self:
        """
        Scales periodic elements to the provided period
        Args:
            period: float or torch.Tensor.
        Returns:
            periodic.PeriodicTensor
        """
        out = self.clone()
        w_dims = out._w_dims
        if isinstance(w_dims, slice):
            # memory efficient path by performing in-place operations
            out[..., w_dims].mul_(period / out.period)
        else:
            out[..., w_dims] = out[..., w_dims] * (period / out.period)

        if isinstance(period, torch.Tensor):
            period = period.tolist()
        elif isinstance(period, float):
            period = [period] * len(out.w_dims)

        out_period = (end - start for start, end in out.extents)
        new_extents = [
            (start * new_p / old_p, end * new_p / old_p)
            for (start, end), new_p, old_p in zip(out.extents, period, out_period)
        ]

        return as_tensor(out, w_dims=out.w_dims, extents=new_extents)

    def cmean(
        self,
        weight: Tensor | None = None,
        dim: int | Iterable[int] | None = None,
        **kwargs,
    ) -> Self:
        """Computes circular mean along periodic dimensions, regular mean otherwise.

        Args:
            weight (optional): Tensor with shape broadcastable with self.shape[:-1].
              If not None, it is used to compute weighted mean.
            dim (optional). Dimensions to reduce over. Cannot include last dimension.
              If None, all dimensions except the last one are reduced.
            **kwargs: Additional keyword arguments to pass to torch.mean

        Returns:
            Tensor of means

        """
        if dim is None:
            dim = range(self.ndim - 1)
        dim = tuple(dim) if isinstance(dim, Iterable) else (dim,)

        if (self.ndim - 1) in dim or -1 in dim:
            raise ValueError(
                f"Averaging across the last dimension is not allowed, but {dim=}."
            )

        w_dims = self._w_dims
        if weight is not None:
            weight = weight[..., None]
            weight = weight / weight.sum(dim=dim, keepdim=True)
            out = (self * weight).sum(dim=dim, **kwargs)
            z = weight * torch.exp(1.0j * self[..., w_dims].to_period(2 * torch.pi))
            out[..., w_dims] = (
                z.sum(dim=dim, **kwargs).angle().to_period(out.period).tensor
            )  # .tensor call is needed since there may be mismatch in period due to numerical errors
        else:
            out = self.mean(dim=dim, **kwargs)
            z = torch.exp(1.0j * self[..., w_dims].to_period(2 * torch.pi))
            out[..., w_dims] = (
                z.mean(dim=dim, **kwargs).angle().to_period(out.period).tensor
            )  # .tensor call is needed since there may be mismatch in period due to numerical errors

        return out

    def cvar(
        self,
        weight: Tensor | None = None,
        dim: int | Iterable[int] | None = None,
        correction: int = 1,
        **kwargs,
    ) -> Self:
        """Computes circular variance along periodic dimensions, regular variance otherwise.

        Args:
            weight (optional): Tensor with shape broadcastable with self.shape[:-1].
              If not None, it is used to compute weighted mean.
            dim (optional). Dimensions to reduce over. Cannot include last dimension.
              If None, all dimensions except the last one are reduced.
            correction (optional): Bessel's correction to apply to the variance along
              non-periodic dimensions.
            **kwargs: Additional keyword arguments to pass to torch.var

        Returns:
            Tensor of variances

        """
        if dim is None:
            dim = range(self.ndim - 1)
        dim = tuple(dim) if isinstance(dim, Iterable) else (dim,)

        if (self.ndim - 1) in dim or -1 in dim:
            raise ValueError(
                f"Computing variance across the last dimension is not allowed, but {dim=}."
            )

        w_dims = self._w_dims
        if weight is not None:
            weight = weight[..., None]
            N = weight.sum(dim=dim, **kwargs)
            weight = weight / weight.sum(dim=dim, keepdim=True)
            m = (self * weight).sum(dim=dim, keepdim=True)
            out = ((self - m) ** 2 * weight).sum(dim=dim, **kwargs)
            out = out * N / (N - correction).clip(min=0)  # Bessel correction
            z = weight * torch.exp(1.0j * self[..., w_dims].to_period(2 * torch.pi))
            out[..., w_dims] = 1 - z.tensor.sum(dim=dim, **kwargs).abs()
        else:
            out = self.var(dim=dim, **kwargs)
            z = torch.exp(1.0j * self[..., w_dims].to_period(2 * torch.pi))
            out[..., w_dims] = 1 - z.tensor.mean(dim=dim, **kwargs).abs()

        return out


HANDLED_FUNCTIONS = {}


def implements(torch_functions):
    """Register a list of torch functions to override"""

    def decorator(func):
        for torch_function in torch_functions:
            HANDLED_FUNCTIONS[torch_function] = functools.partial(func, torch_function)

    return decorator


@implements(
    [torch.Tensor.norm, torch.norm, torch.linalg.norm, torch.linalg.vector_norm]
)
def norm(func, cls, types, args=(), kwargs=None):
    if not kwargs or "dim" not in kwargs or kwargs["dim"] != -1:
        raise ValueError("Only dim=-1 is supported.")

    result = super(BaseTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )

    return result.tensor


@implements([torch.Tensor.to])
def to(func, cls, types, args=(), kwargs=None):
    result = super(BaseTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )

    def is_integer(x):
        return not (torch.is_floating_point(x) or torch.is_complex(x))

    if (torch.is_floating_point(args[0]) and is_integer(result)) or (
        is_integer(args[0]) and torch.is_floating_point(result)
    ):
        result = result.tensor

    return result


@implements(
    [torch.Tensor.bool, torch.Tensor.long, torch.Tensor.int, torch.Tensor.short]
)
def floating_to_integer(func, cls, types, args=(), kwargs=None):
    result = super(BaseTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )

    if torch.is_floating_point(args[0]):
        result = result.tensor

    return result


@implements(
    [
        torch.floor_divide,
        torch.Tensor.floor_divide,
        torch.Tensor.floor_divide_,
        torch.cos,
        torch.Tensor.cos,
        torch.Tensor.cos_,
        torch.sin,
        torch.Tensor.sin,
        torch.Tensor.sin_,
        torch.tan,
        torch.Tensor.tan,
        torch.Tensor.tan_,
        torch.bucketize,
    ]
)
def to_pure_tensor(func, cls, types, args=(), kwargs=None):
    result = super(BaseTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )

    return result.tensor


@implements(
    [
        torch.Tensor.__getitem__,
        torch.take_along_dim,
        torch.Tensor.take_along_dim,
        torch.index_select,
        torch.Tensor.index_select,
    ]
)
def index_select(func, cls, types, args=(), kwargs=None):
    types = (torch.Tensor, torch.Tensor)
    result = super(BaseTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )
    return result


@implements([torch.cat])
def cat(func, cls, types, args=(), kwargs=None):
    result = super(BaseTensor, cls).__torch_function__(
        func, (torch.Tensor,) * len(types), args=args, kwargs=kwargs
    )

    # modify type if concatenating along last dimension
    tensors = args[0]
    ndim = result.ndim
    if (
        kwargs
        and "dim" in kwargs
        and kwargs["dim"] is not None
        and kwargs["dim"] % ndim == ndim - 1
    ):
        cumdims = [0] + list(itertools.accumulate(t.D for t in tensors[:-1]))
        w_dims = [
            w_dim + cumdim
            for t, cumdim in zip(tensors, cumdims)
            if isinstance(t, PeriodicTensor)
            for w_dim in t.w_dims
        ]
        extents = [
            extent
            for t in tensors
            if isinstance(t, PeriodicTensor)
            for extent in t.extents
        ]
        result = as_tensor(result, w_dims=w_dims, extents=extents)

    return result
