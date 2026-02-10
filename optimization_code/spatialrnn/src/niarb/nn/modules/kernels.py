from collections.abc import Callable, Sequence
from numbers import Number
from itertools import chain

import torch
from torch import Tensor
import hyclib as lib

from .functions import Function
from .frame import ParameterFrame
from ..functional import diff
from niarb.tensors.periodic import PeriodicTensor
from niarb.special.resolvent import laplace_r
from niarb.optimize import elementwise
from niarb import special
from niarb.utils import take_along_dims


__all__ = [
    "Scalar",
    "Matrix",
    "Gaussian",
    "AutapsedLaplace",
    "Laplace",
    "Monotonic",
    "Piecewise",
    "Tuning",
    "RankOne",
    "Norm",
    "Radial",
    # "radial",
]


class Kernel(Function):
    kernel: Callable[[*tuple[Tensor, ...]], Tensor]
    n: int

    def __init__(
        self,
        x_keys: Sequence[str] | str = (),
        y_keys: Sequence[str] | str | None = None,
        kernels: Sequence[Function] = (),
        validate_args: bool = True,
    ):
        super().__init__()
        if isinstance(x_keys, str):
            x_keys = (x_keys,)
        if isinstance(y_keys, str):
            y_keys = (y_keys,)
        if kernels is None:
            kernels = {}

        self.x_keys = tuple(x_keys)
        self.y_keys = self.x_keys if y_keys is None else tuple(y_keys)
        self.kernels = torch.nn.ModuleList(kernels)
        self.validate_args = validate_args

        if len(self.x_keys) != self.n:
            raise ValueError(f"Expected {self.n} x_keys, but got {len(self.x_keys)}.")
        if len(self.y_keys) != self.n:
            raise ValueError(f"Expected {self.n} y_keys, but got {len(self.y_keys)}.")

    def validate(self, x: ParameterFrame, y: ParameterFrame):
        if any(k not in x for k in self.x_keys):
            raise ValueError(f"x must contain all keys {self.x_keys}.")
        if any(k not in y for k in self.y_keys):
            raise ValueError(f"y must contain all keys {self.y_keys}.")

        try:
            torch.broadcast_shapes(x.shape, y.shape)
        except RuntimeError:
            raise ValueError(
                f"x and y must have broadcastable shapes, but {x.shape=} and {y.shape=}."
            )

    def forward(self, x: ParameterFrame, y: ParameterFrame) -> Tensor:
        if self.validate_args:
            self.validate(x, y)

        x_ = (x.data[k] for k in self.x_keys)
        y_ = (y.data[k] for k in self.y_keys)
        args = [k(x, y) for k in self.kernels]
        return self.kernel(*chain.from_iterable(zip(x_, y_)), *args)


class Radial(Kernel):
    def __init_subclass__(cls, abstract: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract and (not hasattr(cls, "n") or cls.n < 1):
            raise ValueError(f"Subclasses of Radial must have n >= 1, but {cls.n=}.")

    def validate(self, x: ParameterFrame, y: ParameterFrame):
        super().validate(x, y)

        x_rkey, y_rkey = self.x_keys[0], self.y_keys[0]
        if x.data[x_rkey].ndim != x.ndim + 1:
            raise ValueError(f"x['{x_rkey}'] must have one more dimension than x.")
        if y.data[y_rkey].ndim != y.ndim + 1:
            raise ValueError(f"y['{y_rkey}'] must have one more dimension than y.")
        if x.data[x_rkey].shape[-1] != y.data[y_rkey].shape[-1]:
            raise ValueError(
                f"Last dimension of x['{x_rkey}'] and y['{y_rkey}'] must be the same."
            )

    def forward(self, x: ParameterFrame, y: ParameterFrame) -> Tensor:
        if self.validate_args:
            self.validate(x, y)

        d = x.data[self.x_keys[0]].shape[-1]
        x_ = (x.data[k] for k in self.x_keys)
        y_ = (y.data[k] for k in self.y_keys)
        args = tuple(chain.from_iterable(zip(x_, y_)))
        other_args = [k(x, y) for k in self.kernels]
        return self.kernel(
            diff(args[0], args[1]).norm(dim=-1), *args[2:], *other_args, d=d
        )

    def __add__(self, g):
        if (
            isinstance(g, Radial)
            and self.x_keys == g.x_keys
            and self.y_keys == g.y_keys
        ):
            return Add(self, g, x_keys=g.x_keys, y_keys=g.y_keys)
        return super().__add__(g)

    def __radd__(self, g):
        if (
            isinstance(g, Radial)
            and self.x_keys == g.x_keys
            and self.y_keys == g.y_keys
        ):
            return Add(self, g, x_keys=g.x_keys, y_keys=g.y_keys)
        return super().__radd__(g)

    def __sub__(self, g):
        if (
            isinstance(g, Radial)
            and self.x_keys == g.x_keys
            and self.y_keys == g.y_keys
        ):
            return Sub(self, g, x_keys=g.x_keys, y_keys=g.y_keys)
        return super().__sub__(g)

    def __rsub__(self, g):
        if (
            isinstance(g, Radial)
            and self.x_keys == g.x_keys
            and self.y_keys == g.y_keys
        ):
            return Sub(self, g, x_keys=g.x_keys, y_keys=g.y_keys)
        return super().__rsub__(g)

    def __mul__(self, g):
        if (
            isinstance(g, Radial)
            and self.x_keys == g.x_keys
            and self.y_keys == g.y_keys
        ):
            return Mul(self, g, x_keys=g.x_keys, y_keys=g.y_keys)
        return super().__mul__(g)

    def __rmul__(self, g):
        if (
            isinstance(g, Radial)
            and self.x_keys == g.x_keys
            and self.y_keys == g.y_keys
        ):
            return Mul(self, g, x_keys=g.x_keys, y_keys=g.y_keys)
        return super().__rmul__(g)

    def __truediv__(self, g):
        if (
            isinstance(g, Radial)
            and self.x_keys == g.x_keys
            and self.y_keys == g.y_keys
        ):
            return TrueDiv(self, g, x_keys=g.x_keys, y_keys=g.y_keys)
        return super().__truediv__(g)

    def __rtruediv__(self, g):
        if (
            isinstance(g, Radial)
            and self.x_keys == g.x_keys
            and self.y_keys == g.y_keys
        ):
            return TrueDiv(self, g, x_keys=g.x_keys, y_keys=g.y_keys)
        return super().__rtruediv__(g)


class RadialBinOp(Radial, abstract=True):
    def __init__(
        self, f: Radial, g: Radial, *args, kernels: Sequence[Function] = (), **kwargs
    ):
        super().__init__(
            *args, kernels=kernels + tuple(f.kernels) + tuple(g.kernels), **kwargs
        )
        self.f = f
        self.g = g


class Add(RadialBinOp):
    n = 1

    def kernel(self, r: Tensor, *args: Tensor, **kwargs) -> Tensor:
        args_f, args_g = args[: len(self.f.kernels)], args[len(self.f.kernels) :]
        return self.f.kernel(r, *args_f, **kwargs) + self.g.kernel(r, *args_g, **kwargs)


class Sub(RadialBinOp):
    n = 1

    def kernel(self, r: Tensor, *args: Tensor, **kwargs) -> Tensor:
        args_f, args_g = args[: len(self.f.kernels)], args[len(self.f.kernels) :]
        return self.f.kernel(r, *args_f, **kwargs) - self.g.kernel(r, *args_g, **kwargs)


class Mul(RadialBinOp):
    n = 1

    def kernel(self, r: Tensor, *args: Tensor, **kwargs) -> Tensor:
        args_f, args_g = args[: len(self.f.kernels)], args[len(self.f.kernels) :]
        return self.f.kernel(r, *args_f, **kwargs) * self.g.kernel(r, *args_g, **kwargs)


class TrueDiv(RadialBinOp):
    n = 1

    def kernel(self, r: Tensor, *args: Tensor, **kwargs) -> Tensor:
        args_f, args_g = args[: len(self.f.kernels)], args[len(self.f.kernels) :]
        return self.f.kernel(r, *args_f, **kwargs) / self.g.kernel(r, *args_g, **kwargs)


class Scalar(Kernel):
    n = 0

    def __init__(self, scalar: Callable[[], Tensor] | Tensor | Number, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if callable(scalar):
            self.scalar = scalar
        else:
            self.register_buffer("scalar", torch.as_tensor(scalar), persistent=False)

    def kernel(self) -> Tensor:
        scalar = self.scalar() if callable(self.scalar) else self.scalar
        return scalar.squeeze()


class Matrix(Kernel):
    n = 1

    def __init__(
        self,
        matrix: Callable[[], Tensor] | Tensor | Sequence[Sequence[Number]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if callable(matrix):
            self.matrix = matrix
        else:
            self.register_buffer("matrix", torch.as_tensor(matrix), persistent=False)

    def kernel(self, idx_x: Tensor, idx_y: Tensor) -> Tensor:
        matrix = self.matrix() if callable(self.matrix) else self.matrix
        # identical to the code below, but much faster for large tensors during backprop
        ndim = max(idx_x.ndim, idx_y.ndim)
        if matrix.ndim < 2:
            raise ValueError(
                f"matrix must have at least 2 dimensions, but {matrix.ndim=}."
            )
        matrix = matrix[(..., *((None,) * ndim), slice(None), slice(None))]
        return take_along_dims(matrix, idx_x[..., None, None], idx_y[..., None, None])

        # if matrix.ndim > 2:
        #     # handle batch dimensions
        #     return matrix[..., idx_x, idx_y]
        # # Weird indexing due to vmap bug: https://github.com/pytorch/pytorch/issues/124423
        # return matrix[idx_x[None], idx_y[None]][0]


class Gaussian(Radial):
    n = 1

    def __init__(self, sigma: Function, *args, normalize: str = "none", **kwargs):
        if normalize not in {"none", "integral"}:
            raise ValueError(
                f"normalize must be 'none' or 'integral', but got {normalize}."
            )

        super().__init__(*args, kernels=(sigma,), **kwargs)
        self.normalize = normalize

    def kernel(self, r: Tensor, sigma: Tensor, d: int) -> Tensor:

        out = torch.exp(-(r**2) / (2 * sigma**2))
        if self.normalize == "integral":
            out = out / (2 * torch.pi * sigma**2).sqrt() ** d
        return out

class AutapsedLaplace(Radial):
    n = 2

    def __init__(
        self,
        sigma: Function,
        *args,
        d: int | None = None,
        normalize: str = "none",
        **kwargs,
    ):
        if normalize not in {"none", "integral", "origin"}:
            raise ValueError(
                f"normalize must be 'none', 'integral', or 'origin', but got {normalize}."
            )

        if normalize == "origin" and isinstance(d, int) and d > 1:
            raise ValueError(f"Normalization 'origin' only valid for d <= 1, but {d=}.")

        super().__init__(*args, kernels=(sigma,), **kwargs)
        self.d = d
        self.normalize = normalize

    def kernel(
        self,
        r: Tensor,
        _: Tensor | Number,
        space_dV: Tensor | Number,
        sigma: Tensor,
        d: int,
    ) -> Tensor:
        d = d if self.d is None else self.d
        dr = special.ball_radius(d, space_dV) if d > 1 else 0
        out = laplace_r(d, 1 / sigma, r, dr=dr, is_sqrt=True, validate_args=False)

        if self.normalize == "origin":
            if d > 1:
                raise ValueError(
                    f"Normalization 'origin' only valid for d <= 1, but {d=}."
                )
            zero = torch.tensor(0.0, dtype=r.dtype, device=r.device)
            out = out / laplace_r(d, 1 / sigma, zero, is_sqrt=True, validate_args=False)
        elif self.normalize == "integral":
            out = out / sigma**2
        return out


class Laplace(AutapsedLaplace):
    n = 1

    def kernel(self, r: Tensor, sigma: Tensor, d: int) -> Tensor:
        return super().kernel(r, 0, 0, sigma, d)


class Monotonic(Radial):
    n = 1

    def __init__(
        self, f: Radial, *args, x0: float = 1e-5, unique: bool = True, **kwargs
    ):
        super().__init__(*args, kernels=f.kernels, **kwargs)
        self.f = f
        self.x0 = x0
        self.unique = unique

    def kernel(self, r: Tensor, *args: Tensor, **kwargs) -> Tensor:
        if self.unique and len(args) > 0 and all(arg.ndim == r.ndim for arg in args):
            # in my use case there are very few unique elements in args,
            # so it is more efficient to minimize the function with respect to
            # only the unique elements
            uargs = torch.broadcast_tensors(*args)
            shape = uargs[0].shape
            uargs = torch.stack([arg.flatten() for arg in uargs])
            uargs, inverse = torch.unique(uargs, return_inverse=True, dim=1)
            uargs, inverse = tuple(uargs), inverse.reshape(shape)
            x0 = torch.full(uargs[0].shape, self.x0, dtype=r.dtype, device=r.device)
        elif len(args) == 0:
            uargs = ()
            x0 = torch.tensor(self.x0, dtype=r.dtype, device=r.device)
        else:
            uargs = args

        rmin = elementwise.minimize_newton(self.f.kernel, x0, args=uargs, kwargs=kwargs)

        if self.unique and len(args) > 0 and all(arg.ndim == r.ndim for arg in args):
            rmin = rmin[inverse]
        elif len(args) == 0:
            rmin = rmin.broadcast_to(r.shape)

        mask = r > rmin
        r_ = r.clone()  # avoid mutating r
        r_[mask] = rmin[mask]
        return self.f.kernel(r_, *args, **kwargs)


class Piecewise(RadialBinOp):
    n = 1

    def __init__(self, radius: Function, *args, continuous: bool = True, **kwargs):
        super().__init__(*args, kernels=(radius,), **kwargs)

        self.continuous = continuous

    def kernel(self, r: Tensor, radius: Tensor, *args: Tensor, **kwargs) -> Tensor:
        args_f, args_g = args[: len(self.f.kernels)], args[len(self.f.kernels) :]
        ratio = (
            self.f.kernel(radius, *args_f, **kwargs)
            / self.g.kernel(radius, *args_g, **kwargs)
            if self.continuous
            else 1
        )
        return torch.where(
            r < radius,
            self.f.kernel(r, *args_f, **kwargs),
            ratio * self.g.kernel(r, *args_g, **kwargs),
        )


class Tuning(Kernel):
    n = 1

    def __init__(self, kappa: Function, *args, normalize: bool = False, **kwargs):
        super().__init__(*args, kernels=(kappa,), **kwargs)
        self.normalize = normalize

    def kernel(self, x: PeriodicTensor, y: PeriodicTensor, kappa: Tensor) -> Tensor:
        theta = diff(x, y)
        out = 1 + 2 * kappa * torch.cos(theta.to_period(2 * torch.pi).norm(dim=-1))
        if self.normalize:
            out = out / theta.period
        return out


class RankOne(Kernel):
    n = 1

    def __init__(
        self,
        f: Callable[[Tensor], Tensor],
        g: Callable[[Tensor], Tensor] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.f = f
        self.g = f if g is None else g

    def kernel(self, x: Tensor, y: Tensor) -> Tensor:
        return self.f(x) * self.g(y)


class Norm(Kernel):
    n = 1

    def __init__(
        self, f: Callable[[Tensor], Tensor], *args, ord: int | float = 2, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.f = f
        self.ord = ord

    def forward(self, x: ParameterFrame, y: ParameterFrame) -> Tensor:
        if self.validate_args:
            self.validate(x, y)

        out = self.f(x, y)
        idx_x, idx_y = x.data[self.x_keys[0]], y.data[self.y_keys[0]]
        idx = idx_x * (idx_y.max() + 1) + idx_y
        norm = lib.pt.bincount(idx.flatten(), weights=out.flatten().abs() ** self.ord)
        return (norm ** (1 / self.ord))[idx]


# def radial(
#     *,
#     func: Callable[[ParameterFrame, ParameterFrame], Tensor] | None = None,
#     kernel: Callable[[*tuple[Tensor, ...]], Tensor] | None = None,
#     x_keys: Sequence[str] | str = (),
#     y_keys: Sequence[str] | str | None = None,
#     name: str = "CustomRadial",
#     **kwargs,
# ) -> Radial:
#     if bool(kernel) is bool(func):
#         raise ValueError("Exactly one of func and kernel must be provided.")

#     if kernel is None:

#         def kernel_(self, r: Tensor, *args: Tensor) -> Tensor:
#             x_keys, y_keys = self.x_keys, self.y_keys
#             zeros = torch.zeros(r.shape, dtype=r.dtype, device=r.device)[..., None]
#             r_ = r[..., None]
#             x = {x_keys[0]: zeros} | dict(zip(x_keys[1:], args[::2]))
#             y = {y_keys[0]: r_} | dict(zip(y_keys[1:], args[1::2]))
#             x, y = ParameterFrame(x, ndim=r.ndim), ParameterFrame(y, ndim=r.ndim)
#             return func(x, y)

#     else:

#         def kernel_(self, *args: Tensor) -> Tensor:
#             return kernel(*args)

#     n = 1 if isinstance(x_keys, str) else len(x_keys)

#     return type(name, (Radial,), {"kernel": kernel_, "n": n})(
#         x_keys=x_keys, y_keys=y_keys, **kwargs
#     )
