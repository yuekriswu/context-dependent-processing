import math

import pytest
import torch
import hyclib as lib

from niarb import linalg


@pytest.mark.parametrize("shape", [(2,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float, torch.long])
def test_eye(shape, dtype):
    A = linalg.eye(shape, dtype=dtype)
    assert A.dtype == dtype
    assert A.shape == (*shape, *shape)
    N = math.prod(shape)
    assert (A.reshape(N, N) == torch.eye(N, dtype=dtype)).all()


def test_eigr():
    with lib.random.set_seed(0):
        A = torch.randn(41, 41)
    Lr, Lc, Vr, Vc, Vinvr, Vinvc = linalg.eigr(A, return_inverse=True)
    assert Lr.dtype == Vr.dtype == Vinvr.dtype == torch.float
    assert Lc.dtype == Vc.dtype == Vinvc.dtype == torch.complex64
    L = torch.cat([Lr, Lc, Lc.conj()])
    V = torch.cat([Vr, Vc, Vc.conj()], dim=1)
    Vinv = torch.cat([Vinvr, Vinvc, Vinvc.conj()], dim=0)
    out = V @ L.diag_embed() @ Vinv
    torch.testing.assert_close(out.imag, torch.zeros(out.shape))
    torch.testing.assert_close(out.real, A)


def test_nbmv():
    with lib.random.set_seed(0):
        A = torch.randn(2, 1, 4, 5, 4, 5, 6)
        x = torch.randn(1, 3, 4, 5, 6)
    output = linalg.nbmv(A, x, ndim1=2, ndim2=3)
    expected = (A.reshape(2, 1, 20, 120) @ x.reshape(1, 3, 120, 1)).squeeze(-1)
    expected = expected.reshape(2, 3, 4, 5)

    torch.testing.assert_close(output, expected)
