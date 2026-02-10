import importlib

import pytest
import torch
from scipy import linalg

from niarb import random
from niarb.tensors import circulant


@pytest.mark.parametrize(
    "cdim, bdim1, bdim2, ndim, vec_bdim, vec_bshape, vec_cdim",
    [
        ((5, 6), None, None, 2, (), (), (-2, -1)),
        ((3, 5), None, None, 4, (-3, -1), (2, 3), (-4, -2)),
        ((3, 5), (4, 6), None, None, (-4, -3), (2, 3), (-2, -1)),
        ((3, 5), None, (1, 2), None, (-4, -3), (2, 3), (-2, -1)),
        ((3, 5), (4, 6), (1, 2), None, (-4, -3), (2, 3), (-2, -1)),
    ],
)
def test_as_tensor(cdim, bdim1, bdim2, ndim, vec_bdim, vec_bshape, vec_cdim):
    A = torch.randn(5, 2, 3, 4, 2, 6, 3)
    A = circulant.as_tensor(A, cdim=cdim, bdim1=bdim1, bdim2=bdim2, ndim=ndim)
    assert A.vec_bdim == vec_bdim
    assert A.vec_bshape == vec_bshape
    assert A.vec_cdim == vec_cdim


def test_eye_like():
    A = torch.randn(5, 2, 3, 4, 2, 7, 3)
    A = circulant.as_tensor(A, cdim=(3, 5), bdim1=(4, 6), bdim2=(1, 2))
    I = circulant.eye_like(A)
    assert type(I) is type(A)
    assert I.shape == (4, 7, 6, 6)
    I = I.dense(keep_shape=False)
    assert I.shape == (168, 168)
    torch.testing.assert_close(I, torch.eye(168))

    A = torch.randn(5, 2, 3, 4, 2, 7, 3)
    A = circulant.as_tensor(A, cdim=(5, 6), ndim=2)
    I = circulant.eye_like(A)
    assert type(I) is type(A)
    assert I.shape == (7, 3)
    I = I.dense(keep_shape=False)
    assert I.shape == (21, 21)
    torch.testing.assert_close(I, torch.eye(21))


@pytest.mark.parametrize(
    "op",
    [
        ("torch.linalg", "matmul"),
        ("torch", "matmul"),
        ("torch.linalg", "solve"),
        ("operator", "matmul"),
    ],
)
@pytest.mark.parametrize("is_mat", [True, False])
@pytest.mark.parametrize("batched_c", [True, False])
@pytest.mark.parametrize("batched_B", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.complex64])
def test_matrix_op(op, is_mat, batched_c, batched_B, dtype):
    op = getattr(importlib.import_module(op[0]), op[1])

    batch_c = (4,) if batched_c else ()
    batch_B = (4,) if batched_B else ()
    mat_B = (5,) if is_mat else ()

    with random.set_seed(0):
        c = torch.randn((*batch_c, 2, 3, 3, 5, 6, 5, 6))
        B = torch.randn((*batch_B, 2, 3, 5, 5, 6, *mat_B))
        if dtype is torch.long:
            c = c.round().long()
            B = B.round().long()
        elif dtype is torch.complex64:
            c = c + 1.0j
            B = B + 2.0j

    A = circulant.as_tensor(c, cdim=(-2, -4, -7), bdim1=(-3, -5), bdim2=(-1, -6))
    A_, B_ = A.dense(keep_shape=False), B.reshape((*batch_B, -1, *mat_B))
    print(A.shape, B.shape, A_.shape, B_.shape)

    try:
        expected = op(A_, B_)
    except Exception as err:
        print(type(err), err)
        with pytest.raises(type(err)):
            out = op(A, B)
        return

    out = op(A, B)
    batch_shape = (4,) if batched_c or batched_B else ()
    expected = expected.reshape((*batch_shape, 2, 3, 5, 5, 6, *mat_B))

    torch.testing.assert_close(out, expected, rtol=1.3e-4, atol=5.0e-4)


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.complex64])
def test_eigvals(batched, dtype):
    batch = (4,) if batched else ()

    with random.set_seed(0):
        c = torch.randn((*batch, 3, 3, 5, 6, 5, 6))
        if dtype is torch.long:
            c = c.round().long()
        elif dtype in {torch.complex64, torch.complex128}:
            c = c + 1.0j

    A = circulant.as_tensor(c, cdim=(-2, -4), bdim1=(-3, -5), bdim2=(-1, -6))

    out = torch.linalg.eigvals(A)
    expected = torch.linalg.eigvals(A.dense(keep_shape=False))

    torch.testing.assert_close(
        out.real.sort()[0], expected.real.sort()[0], rtol=5.0e-6, atol=1.0e-4
    )
    torch.testing.assert_close(
        out.imag.sort()[0], expected.imag.sort()[0], rtol=5.0e-6, atol=1.0e-4
    )


@pytest.mark.parametrize("shape_B", [(), (20,)])
def test_solve(shape_B):
    shape_c = (100,)
    with random.set_seed(0):
        c = torch.randn(shape_c)
        B = torch.randn((*shape_c, *shape_B))
    A = circulant.as_tensor(c, ndim=1)

    out = torch.linalg.solve(A, B)
    expected = torch.from_numpy(linalg.solve_circulant(c.numpy(), B.numpy())).float()

    torch.testing.assert_close(out, expected)


def test_binop():
    A = torch.randn(2, 5, 4, 5, 6)
    A = circulant.as_tensor(A, cdim=(2, 4), ndim=3)
    B = torch.randn(2, 5, 4, 5, 6)
    B = circulant.as_tensor(B, cdim=(2, 4), ndim=3)
    assert ((A - B).tensor == A.tensor - B.tensor).all()
    with pytest.raises(NotImplementedError):
        A - B.tensor


def test_unhandled_functions():
    A = torch.randn(2, 5, 4, 5, 6)
    A = circulant.as_tensor(A, cdim=(2, 4), ndim=3)
    with pytest.raises(NotImplementedError):
        A[0]


def test_stack():
    C = torch.randn((3, 4, 5, 3, 3))
    C = circulant.as_tensor(C, ndim=2, cdim=(2,))
    out = torch.stack([C, C])
    expected = torch.stack([C.dense(), C.dense()])
    assert out.shape == (2, 3, 4, 5, 3, 3)
    assert type(out) is type(C)
    torch.testing.assert_close(out.dense(), expected)
