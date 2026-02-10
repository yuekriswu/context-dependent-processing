from collections.abc import Sequence, Iterable
import math

import torch
from torch import Tensor

from niarb import utils
from niarb.tensors import circulant


def eye(shape: Sequence[int], **kwargs) -> Tensor:
    """Returns the identity matrix of shape (*shape, *shape).

    Args:
        shape: The shape of the identity matrix.
        **kwargs: Keyword arguments passed to torch.zeros.

    Returns:
        Tensor with shape (*shape, *shape).

    """
    I = torch.zeros((*shape, *shape), **kwargs)
    idx = torch.meshgrid(
        *[torch.arange(n, device=I.device) for n in shape], indexing="ij"
    )
    I[(*idx, *idx)] = 1.0
    return I


def eye_like(
    A: Tensor,
    dim1: int | Iterable[int] | None = None,
    dim2: int | Iterable[int] | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """
    Args:
        A: Tensor
        dim1 (optional): The output dimensions of the identity tensor.
        dim2 (optional): The input dimensions of the identity tensor.
        requires_grad (optional): Whether to require gradient for the output tensor.

    Returns:
        Identity tensor with shape that is broadcastable with A.

    """
    if isinstance(A, circulant.CirculantTensor):
        if dim1 is not None or dim2 is not None:
            raise ValueError(
                "CirculantTensor does not support dim1 and dim2 arguments."
            )

        return circulant.eye_like(A, requires_grad=requires_grad)

    if dim1 is None:
        dim1 = (-2,)
    if dim2 is None:
        dim2 = (-1,)
    dim1 = utils.normalize_dim(dim1, A.ndim, neg=True)
    dim2 = utils.normalize_dim(dim2, A.ndim, neg=True)

    kwargs = dict(
        dtype=A.dtype, layout=A.layout, device=A.device, requires_grad=requires_grad
    )

    if dim1 == (-2,) and dim2 == (-1,):
        return torch.eye(A.shape[-1], **kwargs)

    if set(dim1) & set(dim2):
        raise ValueError("dim1 and dim2 must be disjoint.")

    shape1 = tuple(A.shape[d] for d in dim1)
    shape2 = tuple(A.shape[d] for d in dim2)

    if shape1 != shape2:
        raise ValueError("A must have same shape along dim1 and dim2.")

    I = eye(shape1, **kwargs)
    I = I[(None,) * (A.ndim - len(dim1) - len(dim2))]
    I = I.movedim(tuple(range(-len(dim1) - len(dim2), 0)), dim1 + dim2)

    return I


def eigr(A, return_inverse=False):
    """
    Compute eigenvalues and eigenvectors of a real matrix A
    and remove redundant information by returning only the real
    part of the real eigenvalues/vectors and only one eigenvalue/vector
    from each complex conjugate pair.
    Note: this relies on the undocumented behavior of torch.linalg.eig
    where the complex conjugate pairs are adjacent.
    Args:
        A: torch.Tensor with shape (N, N)
    Returns:
        Let Nr be the number of real eigenvalues and Nc be the number of complex eigenvalues
        Lr: real eigenvalues with shape (Nr,) and real dtype
        Lc: complex eigenvalues with shape (Nc // 2,) and complex dype
        Vr: real eigenvectors with shape (N, Nr) and real dtype
        Vc: complex eigenvectors with shape (N, Nc // 2) and complex dype
        Vinvr: (optional) real rows of the inverse of the eigenvector matrix with shape (Nr, N) and real dtype
        Vinvc: (optional) complex rows of the inverse of the eigenvector matrix with shape (Nc // 2, N) and complex dtype
    """
    if A.is_complex():
        raise TypeError("A must be a real matrix.")

    L, V = torch.linalg.eig(A)
    isreal = L.isreal()

    Lr, Lc = L[isreal], L[~isreal]
    Lr, Lc = Lr.real, Lc[::2]
    Vr, Vc = V[:, isreal], V[:, ~isreal]
    Vr, Vc = Vr.real, Vc[:, ::2]

    if return_inverse:
        Vinv = torch.linalg.inv(V)
        Vinvr, Vinvc = Vinv[isreal], Vinv[~isreal]
        Vinvr, Vinvc = Vinvr.real, Vinvc[::2]
        return Lr, Lc, Vr, Vc, Vinvr, Vinvc

    return Lr, Lc, Vr, Vc


def is_diagonal(A: Tensor, tol: float = 1.0e-10) -> bool:
    """Determine whether or not A is a diagonal matrix

    Args:
        A: An arbitrary Tensor.
        tol: Tolerance for floating point equality.

    Returns:
        Whether or not A is a diagonal matrix.

    """
    if A.ndim < 2 or (A.shape[-2] != A.shape[-1]):
        return False

    shape = A.shape[:-2]
    n = A.shape[-1]
    # https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379
    A = A.reshape(*shape, -1)[..., 1:].reshape(*shape, n - 1, n + 1)[..., :-1]

    return (A.abs() < tol).all()


def nbmv(A, x, ndim1=1, ndim2=1):
    """
    Performs batched matrix-vector operation where
    outputs are elements of R^{N_1 x ... x N_{ndim1}}
    vectors are elements of R^{M_1 x ... x M_{ndim2}}
    Args:
        A: torch.Tensor with shape (*, N_1, ..., N_{ndim1}, M_1, ..., M_{ndim2})
        x: torch.Tensor with shape (**, M_1, ..., M_{ndim2})
    Returns:
        Ax: torch.Tensor with shape (broadcast(*, **), N_1, ..., N_{ndim1})
    """
    if A.ndim < ndim1 + ndim2 or x.ndim < ndim2:
        raise ValueError(
            f"A must be at least {ndim1+ndim2=} dimensional and x must be at least {ndim2=} dimensional."
        )

    if A.shape[-ndim2:] != x.shape[-ndim2:]:
        raise ValueError(f"Incompatible shapes: {A.shape=} and {x.shape=}.")

    shape1, shape2 = A.shape[-(ndim1 + ndim2) : -ndim2], A.shape[-ndim2:]
    N, M = math.prod(shape1), math.prod(shape2)
    A = A.reshape((*A.shape[: -(ndim1 + ndim2)], N, M))
    x = x.reshape((*x.shape[:-ndim2], M))

    out = torch.einsum("...ij,...j->...i", A, x)
    return out.reshape((*out.shape[:-1], *shape1))
