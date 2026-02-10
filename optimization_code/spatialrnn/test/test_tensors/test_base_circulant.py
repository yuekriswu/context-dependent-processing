import importlib

import pytest
import torch
from scipy import linalg
import numpy as np

from niarb import random
from niarb.tensors import base_circulant


@pytest.mark.parametrize("dim", [0, -1])
def test_circulant(dim):
    with random.set_seed(0):
        x = torch.randn(10)

    output = base_circulant.circulant(x, dim=dim)
    expected = torch.from_numpy(linalg.circulant(x.numpy()))
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2])
def test_batch_circulant(dim):
    with random.set_seed(0):
        x = torch.randn(3, 4, 5)

    output = base_circulant.circulant(x, dim=dim)

    x = x.movedim(dim, -1)
    expected = torch.empty((*x.shape[:-1], x.shape[-1], x.shape[-1]))
    for idx in np.ndindex(x.shape[:-1]):
        expected[idx] = torch.from_numpy(linalg.circulant(x[idx].numpy()))
    dim = dim % x.ndim
    expected = expected.movedim((-2, -1), (dim, dim + 1))

    torch.testing.assert_close(output, expected)


def test_n_circulant_shape():
    with random.set_seed(0):
        x = torch.randn(3, 4, 5)

    assert base_circulant.n_circulant(x).shape == (3, 3, 4, 4, 5, 5)
    assert base_circulant.n_circulant(x, dim=(0, -1)).shape == (3, 3, 4, 5, 5)
    assert base_circulant.n_circulant(x, dim=1).shape == (3, 4, 4, 5)
    assert base_circulant.n_circulant(x, dim=(2, 1)).shape == (3, 4, 4, 5, 5)


@pytest.mark.parametrize(
    "op",
    [
        ("torch.linalg", "matmul"),
        ("torch", "matmul"),
        ("torch.linalg", "solve"),
        ("operator", "matmul"),
    ],
)
@pytest.mark.parametrize("block", [True, False])
@pytest.mark.parametrize("nc", [0, 1, 2])
@pytest.mark.parametrize("is_mat", [True, False])
@pytest.mark.parametrize("batched_c", [True, False])
@pytest.mark.parametrize("batched_B", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.complex64])
def test_matrix_op(op, block, nc, is_mat, batched_c, batched_B, dtype):
    if nc == 0 and block is False:
        pytest.skip()

    op = getattr(importlib.import_module(op[0]), op[1])

    shape = tuple(3 * i for i in range(1, nc + 1))
    block_c = (2, 2) if block else ()
    block_B = (2,) if block else ()
    batch_c = (4,) if batched_c else ()
    batch_B = (4,) if batched_B else ()
    mat_B = (5,) if is_mat else ()

    with random.set_seed(0):
        c = torch.randn((*batch_c, *shape, *block_c))
        B = torch.randn((*batch_B, *shape, *block_B, *mat_B))
        if dtype is torch.long:
            c = c.round().long()
            B = B.round().long()
        elif dtype is torch.complex64:
            c = c + 1.0j
            B = B + 2.0j

    A = base_circulant.as_tensor(c, nc=nc, block=block)
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
    expected = expected.reshape(out.shape)

    torch.testing.assert_close(out, expected, rtol=1.3e-5, atol=1.0e-5)


@pytest.mark.parametrize("block", [True, False])
@pytest.mark.parametrize("nc", [0, 1, 2])
@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.complex64])
def test_eigvals(block, nc, batched, dtype):
    if nc == 0 and block is False:
        pytest.skip()

    shape = tuple(3 * i for i in range(1, nc + 1))
    block_c = (2, 2) if block else ()
    batch = (4,) if batched else ()

    with random.set_seed(0):
        c = torch.randn((*batch, *shape, *block_c))
        if dtype is torch.long:
            c = c.round().long()
        elif dtype is torch.complex64:
            c = c + 1.0j

    A = base_circulant.as_tensor(c, nc=nc, block=block)

    out = torch.linalg.eigvals(A)
    expected = torch.linalg.eigvals(A.dense(keep_shape=False))

    torch.testing.assert_close(
        out.real.sort()[0], expected.real.sort()[0], rtol=5.0e-6, atol=1.0e-5
    )
    torch.testing.assert_close(
        out.imag.sort()[0], expected.imag.sort()[0], rtol=5.0e-6, atol=1.0e-5
    )


@pytest.mark.parametrize("shape_B", [(), (20,)])
def test_solve(shape_B):
    shape_c = (100,)
    with random.set_seed(0):
        c = torch.randn(shape_c)
        B = torch.randn((*shape_c, *shape_B))
    A = base_circulant.as_tensor(c)

    out = torch.linalg.solve(A, B)
    expected = torch.from_numpy(linalg.solve_circulant(c.numpy(), B.numpy())).float()

    torch.testing.assert_close(out, expected)


def test_stack():
    C = torch.randn((3, 4, 5, 3, 3))
    C = base_circulant.as_tensor(C, nc=1, block=True)
    out = torch.stack([C, C])
    expected = torch.stack([C.dense(), C.dense()])
    assert out.shape == (2, 3, 4, 5, 3, 3)
    assert type(out) is type(C)
    torch.testing.assert_close(out.dense(), expected)
