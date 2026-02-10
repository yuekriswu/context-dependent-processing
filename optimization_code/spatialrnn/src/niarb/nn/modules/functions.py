from collections.abc import Hashable, Callable
from numbers import Number
import math

import torch
from torch import Tensor

from niarb.tensors import categorical
from ..parameter import Parameter

__all__ = [
    "Identity",
    "Pow",
    "Add",
    "Sub",
    "Mul",
    "TrueDiv",
    "Sum",
    "Prod",
    "Compose",
    "Match",
]


class FunctionMixin:
    """
    A Mixin class that adds algebraic operations to torch.nn.Module
    """

    def __add__(self, g):
        if g == 0:
            return self
        return Add(self, g)

    def __radd__(self, g):
        if g == 0:
            return self
        return Add(self, g)

    def __sub__(self, g):
        if g == 0:
            return self
        return Sub(self, g)

    def __rsub__(self, g):
        if g == 0:
            return self
        return Sub(self, g)

    def __mul__(self, g):
        if g == 1:
            return self
        return Mul(self, g)

    def __rmul__(self, g):
        if g == 1:
            return self
        return Mul(self, g)

    def __truediv__(self, g):
        if g == 1:
            return self
        return TrueDiv(self, g)

    def __rtruediv__(self, g):
        if g == 1:
            return self
        return TrueDiv(self, g)

    def __pow__(self, p):
        if p == 1:
            return self
        return Compose(Pow(p), self)


class Function(FunctionMixin, torch.nn.Module):
    pass


class Identity(FunctionMixin, torch.nn.Identity):
    def inv(self):
        return Identity()


class Pow(Function):
    def __init__(self, p, optim=False, **kwargs):
        super().__init__()
        self.init_p = p
        self.optim = optim
        if optim:
            self.p = Parameter(torch.empty(()), **kwargs)
        else:
            self.p = p

    def reset_parameters(self):
        if self.optim:
            torch.nn.init.constant_(self.p, self.init_p)

    def forward(self, x):
        return x**self.p

    def inv(self):
        return Pow(1 / self.p)

    def nth_deriv(self, n, x):
        if isinstance(n, int) and isinstance(self.p, int) and n > self.p:
            return torch.zeros_like(x)

        # c = math.prod(range(self.p, self.p - n, -1))
        c = math.prod((self.p - k for k in range(n)))
        return c * x ** (self.p - n)


class BinOp(Function):
    def __init__(self, f, g):
        if not isinstance(f, torch.nn.Module):
            raise ValueError(
                f"f must be an instance of torch.nn.Module, but {type(f)=}."
            )

        super().__init__()
        self.f = f
        self.g = g


class Add(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) + self.g(*args, **kwargs)


class Sub(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) - self.g(*args, **kwargs)


class Mul(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) * self.g(*args, **kwargs)


class TrueDiv(BinOp):
    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs) / self.g(*args, **kwargs)


class Sum(Function):
    def __init__(self, funcs):
        super().__init__()
        self.funcs = torch.nn.ModuleDict(funcs)

    def forward(self, *args, **kwargs):
        return sum(func(*args, **kwargs) for func in self.funcs.values())


class Prod(Function):
    def __init__(self, funcs):
        super().__init__()
        self.funcs = torch.nn.ModuleDict(funcs)

    def forward(self, *args, **kwargs):
        return math.prod(func(*args, **kwargs) for func in self.funcs.values())


class Compose(Function):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.f = f
        self.args = torch.nn.ModuleList(args)
        self.kwargs = torch.nn.ModuleDict(kwargs)

    def forward(self, *args, **kwargs):
        return self.f(
            *[v(*args, **kwargs) for v in self.args],
            **{k: v(*args, **kwargs) for k, v in self.kwargs.items()},
        )

    def inv(self):
        if len(self.args) != 1 and len(self.kwargs) != 0:
            raise NotImplementedError()

        return Compose(self.args[0].inv(), self.f.inv())

    def nth_deriv(self, n, *args, **kwargs):
        # Compute the n-th derivative assuming all functions are scalar-valued and
        # only the first function has non-zero derivative.
        if len(self.args) < 1:
            raise ValueError("At least one function is required for the derivative.")

        if not (hasattr(self.f, "nth_deriv") and hasattr(self.args[0], "nth_deriv")):
            raise ValueError("Not all functions have a derivative method.")

        g = self.args[0]
        args_ = [v(*args, **kwargs) for v in self.args]
        kwargs_ = {k: v(*args, **kwargs) for k, v in self.kwargs.items()}

        if n == 1:
            return self.f.nth_deriv(1, *args_, **kwargs_) * g.nth_deriv(
                1, *args, **kwargs
            )

        if n == 2:
            return self.f.nth_deriv(2, *args_, **kwargs_) * g.nth_deriv(
                1, *args, **kwargs
            ) ** 2 + self.f.nth_deriv(1, *args_, **kwargs_) * g.nth_deriv(
                2, *args, **kwargs
            )

        raise NotImplementedError("only n=1 and n=2 are implemented for nth_deriv.")


class Match(Function):
    def __init__(
        self,
        cases: dict[str, torch.nn.Module | Callable[[Tensor], Tensor]],
        default: Callable[[Tensor], Tensor],
    ):
        super().__init__()
        self.cases = torch.nn.ModuleDict(cases) # HoYin added
        self.default = default

    def forward(self, x: Tensor, key: Tensor) -> Tensor:
        key, x = torch.broadcast_tensors(key, x)
        out = self.default(x)
        for k, v in self.cases.items():
            if not isinstance(k, Number):
                # this is not normally needed, but inside of torch.func.grad or
                # torch.func.vmap, the dispatch does not work properly, see
                # https://github.com/pytorch/pytorch/issues/149788
                k = categorical.tensor(0, categories=(k,), device=x.device)
            mask = key == k
            out = torch.where(mask, v(x), out)
        return out
