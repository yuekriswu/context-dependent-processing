from collections.abc import Hashable
import functools

import torch
import numpy as np
import pandas as pd


from .base import tensor_class_factory, _rebuild, BaseTensor


def tensor(
    data, dtype=None, device=None, requires_grad=False, pin_memory=False, categories=()
):
    data = torch.tensor(data, dtype=dtype, device=device, pin_memory=pin_memory)
    return _instantiate(data, requires_grad=requires_grad, categories=categories)


def as_tensor(data, dtype=None, device=None, categories=()):
    data = torch.as_tensor(data, dtype=dtype, device=device)
    return _instantiate(data, categories=categories)


def _instantiate(data, requires_grad=False, categories=()):
    # cast to tuple to ensure immutability
    categories = tuple(categories)
    cls = tensor_class_factory(CategoricalTensor, categories=categories)
    return cls(data, requires_grad=requires_grad)


class CategoricalTensor(BaseTensor):
    def __init__(self, *args, **kwargs):
        if torch.is_floating_point(self) or torch.is_complex(self):
            raise TypeError("CategoricalTensor must be an integer tensor.")

        if len(self.categories) < self.max().item() + 1:
            raise ValueError(
                "Must have at least as many categories as the 1 + maximum value in the tensor."
            )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in HANDLED_FUNCTIONS:
            result = HANDLED_FUNCTIONS[func](cls, types, args=args, kwargs=kwargs)

        else:
            result = super().__torch_function__(func, types, args, kwargs)

        return result

    def __ne__(self, other):
        # for some reason this is needed for __ne__ but not for __eq__
        if (
            not isinstance(other, torch.Tensor)  # for some reason Tensors are hashable
            and isinstance(other, Hashable)
            and other in self.categories
        ):
            other = self.categories.index(other)
        return super().__ne__(other)

    def __reduce_ex__(self, proto):
        # override __reduce_ex__ to support pickling
        func, args = super().__reduce_ex__(proto)
        args = list(args)
        args[1] = {"categories": self.categories}
        return _rebuild, (func, tuple(args), CategoricalTensor)

    def pandas(self, force: bool = False) -> pd.Categorical:
        data = self.tensor.numpy(force=force)
        return pd.Categorical.from_codes(data, categories=self.categories)


HANDLED_FUNCTIONS = {}


def implements(torch_functions):
    """Register a list of torch functions to override"""

    def decorator(func):
        for torch_function in torch_functions:
            HANDLED_FUNCTIONS[torch_function] = functools.partial(func, torch_function)

    return decorator


@implements(
    [
        torch.eq,
        torch.Tensor.eq,
        torch.ne,
        torch.Tensor.ne,
        torch.Tensor.__eq__,
        torch.Tensor.__ne__,
    ]
)
def comparison(func, cls, types, args=(), kwargs=None):
    if len(args) != 2:
        raise ValueError("Expected two arguments.")

    self, other = args
    if isinstance(other, torch.Tensor):
        # horrifying stuff to deal with torch.func.grad and torch.func.vmap,
        # uses torch internals to unwrap the gradtrackingtensor and batchedtensor
        # layers to get at the underlying tensor, might break in the future
        def unwrap(x):
            if torch._C._functorch.is_functorch_wrapped_tensor(x):
                return unwrap(torch._C._functorch.get_unwrapped(x))
            return x

        other = unwrap(other)

    if (
        not isinstance(other, torch.Tensor)  # for some reason Tensors are hashable
        and isinstance(other, Hashable)
        and other in self.categories
    ):
        other = self.categories.index(other)

    elif isinstance(self, CategoricalTensor) and isinstance(other, CategoricalTensor):
        mapping = [
            self.categories.index(v) if v in self.categories else len(self.categories)
            for v in other.categories
        ]
        mapping = torch.tensor(mapping, dtype=other.dtype, device=other.device)
        # workaround for torch.func.vmap bug with indexing, see
        # https://github.com/pytorch/pytorch/issues/124423
        other = mapping[args[1][None]][0]

    result = super(BaseTensor, cls).__torch_function__(
        func, (torch.Tensor,) * len(types), args=(self, other), kwargs=kwargs
    )
    return result.tensor


@implements([torch.where, torch.Tensor.where])
def where(func, cls, types, args=(), kwargs=None):
    result = super(BaseTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )
    return result.as_subclass(torch.Tensor)


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
    result = super(BaseTensor, cls).__torch_function__(
        func, (torch.Tensor,) * len(types), args=args, kwargs=kwargs
    )
    return result.as_subclass(type(args[0]))


@implements([torch.Tensor.bincount, torch.bincount])
def bincount(func, cls, types, args=(), kwargs=None):
    result = super(BaseTensor, cls).__torch_function__(
        func, (torch.Tensor,) * len(types), args=args, kwargs=kwargs
    )
    if len(args) == 2 and args[1] is not None:
        new_cls = type(args[1])
    elif (
        isinstance(kwargs, dict)
        and "weights" in kwargs
        and kwargs["weights"] is not None
    ):
        new_cls = type(kwargs["weights"])
    else:
        new_cls = torch.Tensor

    return result.as_subclass(new_cls)


@implements(
    [
        torch.Tensor.unique,
        torch.unique,
        torch.Tensor.unique_consecutive,
        torch.unique_consecutive,
    ]
)
def unique(func, cls, types, args=(), kwargs=None):
    # TODO: For the case dim=None, take advantage of the fact that the categories
    # are already unique and that the tensor itself is the inverse indices.
    # But this is not critical.
    result = super(BaseTensor, cls).__torch_function__(
        func, (torch.Tensor,) * len(types), args=args, kwargs=kwargs
    )

    if isinstance(result, tuple):
        result = (result[0], *(v.tensor for v in result[1:]))

    return result


@implements([torch.stack, torch.cat, torch.hstack, torch.vstack])
def combine(func, cls, types, args=(), kwargs=None):
    result = super(BaseTensor, cls).__torch_function__(
        func, (torch.Tensor,) * len(types), args=args, kwargs=kwargs
    )
    if not all(isinstance(v, cls) for v in args[0]):
        result = result.tensor

    return result


@implements([torch.Tensor.numpy])
def numpy(func, cls, types, args=(), kwargs=None):
    result = super(BaseTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )
    categories = np.array(cls.categories)
    return categories[result]


@implements([torch.broadcast_tensors])
def broadcast_tensors(func, cls, types, args=(), kwargs=None):
    shape = torch.broadcast_shapes(*(v.shape for v in args))
    result = tuple(v.broadcast_to(shape) for v in args)

    return result
