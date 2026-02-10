import math
import functools

import torch

from .base import tensor_class_factory, _rebuild, BaseTensor
from niarb import utils


def tensor(
    data, dtype=None, device=None, requires_grad=False, pin_memory=False, **kwargs
):
    data = torch.tensor(data, dtype=dtype, device=device, pin_memory=pin_memory)
    return _instantiate(data, requires_grad=requires_grad, **kwargs)


def as_tensor(data, dtype=None, device=None, **kwargs):
    data = torch.as_tensor(data, dtype=dtype, device=device)
    return _instantiate(data, **kwargs)


def _instantiate(data, requires_grad=False, nc=1, block=False):
    if not isinstance(nc, int):
        raise TypeError(f"nc must be an integer, but {type(nc)=}.")

    if nc < 0:
        raise ValueError(f"nc must be non-negative, but {nc=}.")

    if nc == 0 and not block:
        raise ValueError(
            "Must have at least one circulant dimension or a block dimension."
        )

    cls = tensor_class_factory(
        BaseCirculantTensor, nc=nc, block=block, orig_dtype=data.dtype
    )

    return cls(data, requires_grad=requires_grad)


def circulant(x, dim=-1):
    """
    Generalization of scipy.linalg.circulant that accepts an additional dim argument
    which specifies the dimension which is made circulant.
    Args:
        x: torch.Tensor. Represents the first column of the circulant tensor.
        dim: int. The dimension along which x is made circulant.
    Output:
        out: torch.Tensor with shape (..., n, n, ...), n = x.shape[dim]
    """
    dim = dim % x.ndim  # correctly handle negative dims

    # Generate index that performs the striding step used in scipy.linalg.circuit,
    # since PyTorch does not currently support negative strides
    n = x.shape[dim]
    i = torch.arange(n, device=x.device)
    i, j = torch.meshgrid(i, i, indexing="ij")  # (n, n), (n, n)
    idx = j - i + n - 1  # (n, n)

    # Form an extended tensor that could be indexed to give circulant version
    x = torch.cat(
        (x.flip([dim]), x[(slice(None),) * dim + (slice(1, None),)].flip([dim])),
        dim=dim,
    )

    return x[(slice(None),) * dim + (idx,)]


def n_circulant(x, dim=None):
    """
    Generalization of scipy.linalg.circulant to n-dimensions.
    Args:
        x: torch.Tensor. Represents the first 'column' of the circulant tensor.
        dim: int | Iterable[int] | None. The dimensions along which x is made circult.
    Output:
        out: torch.Tensor with shape (..., n_0, n_0, ..., n_1, n_1, ......, n_{-1}, n_{-1}, ...), n_i = x.shape[dim[i]]
    """
    if dim is None:
        dim = range(x.ndim)
    dim = utils.normalize_dim(dim, x.ndim)

    for d in reversed(sorted(dim)):
        x = circulant(x, dim=d)

    return x


class BaseCirculantTensor(BaseTensor):
    def __new__(cls, data, **kwargs):
        if data.ndim < cls.nc + 2 * cls.block:
            raise ValueError(
                f"{data.ndim=} must be at least {cls.nc + 2 * cls.block=}."
            )

        if cls.block and data.shape[-2] != data.shape[-1]:
            raise ValueError(
                f"Last two dimensions of data must be equal if block is True, but {data.shape=}."
            )

        cdim = tuple(range(-cls.nc - 2 * cls.block, -2 * cls.block))

        data = torch.fft.fftn(data, dim=cdim)

        return super().__new__(cls, data, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in HANDLED_FUNCTIONS:
            result = HANDLED_FUNCTIONS[func](cls, types, args=args, kwargs=kwargs)

        elif func.__name__ in ALLOWED_FUNCTIONS:
            result = super().__torch_function__(func, types, args, kwargs)

        else:
            raise NotImplementedError()

        return result

    def __reduce_ex__(self, proto):
        # override __reduce_ex__ to support pickling
        func, args = super().__reduce_ex__(proto)
        args = list(args)
        args[1] = {"nc": self.nc, "block": self.block, "orig_dtype": self.orig_dtype}
        return _rebuild, (func, tuple(args), BaseCirculantTensor)

    @property
    def nb(self):
        return int(self.block)

    @property
    def nbatch(self):
        return self.ndim - self.nc - 2 * self.nb

    @property
    def cdim(self):
        return tuple(range(self.nbatch, self.nbatch + self.nc))

    @property
    def bdim(self):
        return tuple(range(self.nbatch + self.nc + self.nb, self.ndim))

    @property
    def batchdim(self):
        return tuple(range(self.nbatch))

    @property
    def cshape(self):
        return torch.Size(self.shape[d] for d in self.cdim)

    @property
    def bshape(self):
        return torch.Size(self.shape[d] for d in self.bdim)

    @property
    def batchshape(self):
        return torch.Size(self.shape[d] for d in self.batchdim)

    def dense(self, keep_shape=True):
        c = torch.fft.ifftn(self.tensor, dim=self.cdim)
        out = n_circulant(c, dim=self.cdim)

        source_dim = tuple(d + i + 1 for i, d in enumerate(self.cdim))
        target_dim = tuple(d + self.nc + self.nb for d in self.cdim)
        out = out.movedim(source_dim, target_dim)

        if not torch.is_complex(torch.empty((), dtype=self.orig_dtype)):
            out = out.real

        out = out.to(self.orig_dtype).clone()

        if keep_shape:
            return out

        return out.reshape((*self.batchshape, math.prod(self.cshape + self.bshape), -1))


HANDLED_FUNCTIONS = {}
ALLOWED_FUNCTIONS = {"__get__"}


def implements(torch_functions):
    """Register a list of torch functions to override"""

    def decorator(func):
        for torch_function in torch_functions:
            HANDLED_FUNCTIONS[torch_function] = functools.partial(func, torch_function)

    return decorator


@implements([torch.stack])
def stack(func, cls, types, args=(), kwargs=None):
    if kwargs:
        raise NotImplementedError("kwargs is not currently supported.")

    return super(BaseCirculantTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )


@implements([torch.linalg.eigvals])
def eigvals(func, cls, types, args=(), kwargs=None):
    (data,) = args
    data = data.tensor.reshape(
        (*data.batchshape, -1, math.prod(data.bshape), math.prod(data.bshape))
    )
    eigvals = torch.linalg.eigvals(data)

    return eigvals.reshape(*eigvals.shape[:-2], -1)


@implements(
    [
        torch.Tensor.add,
        torch.add,
        torch.Tensor.sub,
        torch.sub,
        torch.Tensor.mul,
        torch.mul,
        torch.Tensor.div,
        torch.div,
    ]
)
def binop(func, cls, types, args=(), kwargs=None):
    if not all(
        isinstance(a, cls) or not isinstance(a, torch.Tensor) or a.shape in {(), (1,)}
        for a in args
    ):
        raise TypeError(f"{func.__name__} is only allowed between {cls} tensors.")

    return super(BaseCirculantTensor, cls).__torch_function__(
        func, types, args=args, kwargs=kwargs
    )


@implements(
    [
        torch.matmul,
        torch.linalg.matmul,
        torch.Tensor.matmul,
        torch.linalg.solve,
        torch.Tensor.solve,
    ]
)

def batched_matrix_op(func, cls, types, args=(), kwargs=None):
    """Batched matrix operation between a circulant tensor A and a tensor B.

    Args:
        A (BaseCirculantTensor): Circulant tensor.
        B (Tensor): Tensor with shape (*, *A.cshape, *A.bshape) or (*, *A.cshape, *A.bshape, k).

    Returns:
        Tensor with shape (*, *A.cshape, *A.bshape) or (*, *A.cshape, *A.bshape, k).

    """
    if not all(issubclass(cls, t) for t in types):
        return NotImplemented

    if kwargs:
        raise NotImplementedError("kwargs is not currently supported.")

    A, B = args
    if isinstance(B, cls):
        raise NotImplementedError(
            f"{func} with the second argument being a BaseCirculantTensor is not currently supported."
        )

    if A.orig_dtype != B.dtype:
        raise TypeError(f"{A.orig_dtype=} and {B.dtype=} must be the same.")

    is_complex = torch.is_complex(B)
    dtype = B.dtype

    # check shape of inputs
    ndim, shape = A.nc + A.nb, A.cshape + A.bshape
    if B.ndim < ndim:
        raise RuntimeError(f"{B.ndim=} must be at least {ndim=}.")

    # B.shape equals shape or (*, *shape, K)
    valid = B.shape == shape
    valid |= (B.ndim > ndim) and (B.shape[-ndim - 1 : -1] == shape)

    if "solve" in func.__name__:
        # B.shape equals (*A.batchshape, *shape)
        valid |= B.shape == (*A.batchshape, *shape)

    if not valid:
        raise RuntimeError(
            f"Incompatible shapes: {A.batchshape=}, {shape=} and {B.shape=}."
        )

    # check if B is vector-like
    is_vec = B.ndim == ndim
    if "solve" in func.__name__ and B.ndim > ndim:
        # B has shape (*, *shape) but not (*, *shape, K)
        is_vec |= (B.shape[-ndim:] == shape) and (B.shape[-ndim - 1 : -1] != shape)

    if is_vec:
        B = B.unsqueeze(-1)

    cdim = tuple(range(-A.nc - A.nb - 1, -A.nb - 1))
    B = torch.fft.fftn(B, dim=cdim)

    if A.block:
        out = func(A.tensor, B)
    elif "matmul" in func.__name__:
        out = A.tensor.unsqueeze(-1) * B
    elif "solve" in func.__name__:
        out = B / A.tensor.unsqueeze(-1)
    else:
        raise RuntimeError()  # not supposed to be here

    out = torch.fft.ifftn(out, dim=cdim)

    if is_vec:
        out = out.squeeze(-1)

    if not is_complex:
        out = out.real

    return out.to(dtype)
