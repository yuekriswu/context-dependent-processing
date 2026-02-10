import pytest
import torch

from niarb import ft


@pytest.mark.parametrize("dim", [0, -1, 1])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("period", [2 * torch.pi, 1.0])
def test_ift(dim, norm, period):
    a = torch.tensor(
        [
            [2.0, 3.0, 1.5],
            [3.0 + 1.0j, 4.0 + 2.0j, 0.5j],
        ]
    )
    x = torch.linspace(-5, 5, steps=10)[:, None]

    output = ft.torus.ift(a, x, dim=dim, norm=norm, period=period)

    theta = 2 * torch.pi * x / period

    if dim == 0:
        a0, a1 = a[0, :], a[1, :]
        expected = a0 + a1 * torch.exp(1.0j * theta)
    elif dim in [-1, 1]:
        a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
        expected = a0 + a1 * torch.exp(1.0j * theta) + a2 * torch.exp(-1.0j * theta)
    else:
        raise RuntimeError()

    norm = {
        "backward": period,
        "ortho": period**0.5,
        "forward": 1.0,
    }[norm]
    expected = 1 / norm * expected

    expected_shape = (10, 3) if dim == 0 else (10, 2)
    assert expected.shape == expected_shape
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "a",
    [
        [[2.0, 3.0], [3.0 + 1.0j, 4.0 + 2.0j]],
        [[2.0, 3.0], [3.0, 4.0]],
    ],
)
@pytest.mark.parametrize("dim", [0, -1, 1])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("period", [2 * torch.pi, 1.0])
def test_irft(a, dim, norm, period):
    a = torch.tensor(a)
    x = torch.linspace(-5, 5, steps=10)[:, None]

    output = ft.torus.irft(a, x, dim=dim, norm=norm, period=period)

    theta = 2 * torch.pi * x / period

    if dim == 0:
        a0, a1 = a[0, :], a[1, :]
    elif dim in [-1, 1]:
        a0, a1 = a[:, 0], a[:, 1]
    else:
        raise RuntimeError()

    expected = (
        a0.real + a1 * torch.exp(1.0j * theta) + a1.conj() * torch.exp(-1.0j * theta)
    )

    norm = {
        "backward": period,
        "ortho": period**0.5,
        "forward": 1.0,
    }[norm]
    expected = 1 / norm * expected

    assert expected.shape == (10, 2)
    torch.testing.assert_close(torch.zeros_like(expected.imag), expected.imag)
    torch.testing.assert_close(output, expected.real)


@pytest.mark.parametrize("dim", [(0,), (-1,), (1,)])
@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("period", [2 * torch.pi, [4 * torch.pi], 1.0])
def test_irft1(dim, norm, period):
    a = torch.tensor(
        [
            [2.0, 3.0],
            [3.0 + 1.0j, 4.0 + 2.0j],
        ]
    )
    x = torch.linspace(-5, 5, steps=10)[:, None]

    output = ft.torus.irftn(a, x[..., None], dim=dim, norm=norm, period=period)

    period = torch.as_tensor(period)
    theta = 2 * torch.pi * x / period

    if dim[0] == 0:
        a0, a1 = a[0, :], a[1, :]
    elif dim[0] in [-1, 1]:
        a0, a1 = a[:, 0], a[:, 1]
    else:
        raise RuntimeError()

    expected = (
        a0.real + a1 * torch.exp(1.0j * theta) + a1.conj() * torch.exp(-1.0j * theta)
    )

    norm = {
        "backward": torch.prod(period),
        "ortho": torch.prod(period) ** 0.5,
        "forward": 1.0,
    }[norm]
    expected = 1 / norm * expected

    assert expected.shape == (10, 2)
    torch.testing.assert_close(torch.zeros_like(expected.imag), expected.imag)
    torch.testing.assert_close(output, expected.real)
