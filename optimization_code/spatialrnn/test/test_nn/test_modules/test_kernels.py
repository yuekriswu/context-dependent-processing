import torch
import pytest

from niarb import nn
from niarb.nn.modules import frame
from niarb.tensors import periodic


@pytest.fixture
def x():
    return frame.ParameterFrame(
        {
            "space": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
            "ori": periodic.tensor(
                [[-45.0], [0.0], [45.0], [90.0]], extents=[(-90.0, 90.0)]
            ),
            "cell_type": torch.tensor([0, 1, 1, 0]),
        }
    )


@pytest.fixture
def y():
    return frame.ParameterFrame(
        {
            "space": torch.tensor([[0.0], [2.0], [1.0], [5.0]]),
            "ori": periodic.tensor(
                [[0.0], [90.0], [45.0], [-45.0]], extents=[(-90.0, 90.0)]
            ),
            "cell_type": torch.tensor([0, 0, 1, 1]),
        }
    )


@pytest.fixture
def x2():
    return frame.ParameterFrame(
        {
            "space": torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
            "cell_type": torch.tensor([0, 1, 1, 0, 0]),
        }
    )


@pytest.fixture
def y2():
    return frame.ParameterFrame(
        {
            "space": torch.tensor([[0.5], [2.0], [1.0], [5.0], [5.0]]),
            "cell_type": torch.tensor([0, 0, 1, 1, 1]),
        }
    )


def test_matrix(x, y):
    W = nn.Matrix(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "cell_type")
    out = W(x, y)
    expected = torch.tensor([1.0, 3.0, 4.0, 2.0])
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("normalize", ["none", "integral"])
def test_gaussian(x, y, normalize):
    sigma = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2).sqrt()
    kernel = nn.Gaussian(nn.Matrix(sigma, "cell_type"), "space", normalize=normalize)
    out = kernel(x, y)
    # (x - y)^2 = [0, 1, 1, 4], 2 * sigma^2 = [1, 3, 4, 2]
    expected = torch.exp(-torch.tensor([0, 1 / 3, 1 / 4, 2]))
    if normalize == "integral":
        expected = expected / (torch.pi * torch.tensor([1, 3, 4, 2])) ** 0.5
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("d", [None, 2])
@pytest.mark.parametrize("normalize", ["none", "origin", "integral"])
def test_laplace(x, y, d, normalize):
    sigma = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sigma = nn.Matrix(sigma, "cell_type")

    if d == 2 and normalize == "origin":
        with pytest.raises(ValueError):
            kernel = nn.Laplace(sigma, "space", d=d, normalize=normalize)
        return

    kernel = nn.Laplace(sigma, "space", d=d, normalize=normalize)
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], sigma = [1, 3, 4, 2]
    if d is None:
        expected = torch.exp(-torch.tensor([0, 1 / 3, 1 / 4, 1]))
        if normalize == "none":
            expected = expected * torch.tensor([1, 3, 4, 2]) / 2
        elif normalize == "integral":
            expected = expected / (2 * torch.tensor([1, 3, 4, 2]))
    else:
        expected = torch.special.modified_bessel_k0(torch.tensor([0, 1 / 3, 1 / 4, 1]))
        expected[0] = 0.0
        if normalize == "none":
            expected = expected / (2 * torch.pi)
        else:
            expected = expected / (2 * torch.pi * torch.tensor([1, 3, 4, 2]) ** 2)
    torch.testing.assert_close(out, expected)


def test_monotonic(x, y):
    sigma1 = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    sigma2 = torch.tensor([[1.5, 1.0], [0.5, 2.0]])
    kernel1 = nn.Laplace(nn.Matrix(sigma1, "cell_type"), "space", d=2)
    kernel2 = nn.Gaussian(nn.Matrix(sigma2, "cell_type"), "space")
    kernel = nn.Monotonic(kernel1 / kernel2, "space")
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], sigma1 = [3, 1, 2, 4], sigma2 = [1.5, 0.5, 2, 1]
    expected = (kernel1 / kernel2)(x, y)
    expected[1] = 0.241606  # calculated with Mathematica
    expected[3] = 0.382324  # calculated with Mathematica

    torch.testing.assert_close(out, expected)


def test_monotonic_2(x, y):
    sigma1 = torch.tensor([[3.0, 1.5], [1.0, 2.0]])
    sigma2 = sigma1 / 2
    kernel1 = nn.Laplace(nn.Matrix(sigma1, "cell_type"), "space", d=2)
    kernel2 = nn.Laplace(nn.Matrix(sigma2, "cell_type"), "space", d=0)
    kernel = nn.Monotonic(kernel1 / kernel2, "space")
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], sigma1 = [3, 1, 2, 1.5], sigma2 = [1.5, 0.5, 1, 0.75]
    expected = (kernel1 / kernel2)(x, y)
    expected[1] = 0.934132  # calculated with Mathematica
    expected[3] = 0.41517  # calculated with Mathematica

    torch.testing.assert_close(out, expected)


# def test_radial(x, y):
#     kernel_ = lambda r: laplace_r(2, 1.0, r) / laplace_r(0, 0.5, r)
#     kernel = nn.radial(kernel=kernel_, x_keys="space")
#     out = kernel(x, y)
#     expected = kernel_((x["space"] - y["space"]).norm(dim=-1))
#     assert isinstance(kernel, nn.Radial)
#     torch.testing.assert_close(out, expected)


def test_piecewise(x, y):
    sigma1 = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2).sqrt()
    kernel1 = nn.Gaussian(nn.Matrix(sigma1, "cell_type"), "space")
    sigma2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    kernel2 = nn.Laplace(nn.Matrix(sigma2, "cell_type"), "space", d=2)
    radius = torch.tensor([[0.5, 1.5], [1.5, 0.5]])
    kernel = nn.Piecewise(nn.Matrix(radius, "cell_type"), kernel1, kernel2, "space")
    out = kernel(x, y)
    # norm(x - y) = [0, 1, 1, 2], 2 * sigma1^2 = [1, 3, 4, 2], sigma2 = [1, 3, 4, 2]
    # radius = [0.5, 1.5, 0.5, 1.5]
    ratio1 = torch.exp(-torch.tensor(0.5**2 / 4)) / torch.special.modified_bessel_k0(
        torch.tensor(0.5 / 4)
    )
    ratio2 = torch.exp(-torch.tensor(1.5**2 / 2)) / torch.special.modified_bessel_k0(
        torch.tensor(1.5 / 2)
    )
    expected = torch.tensor(
        [
            torch.exp(-torch.tensor(0)),
            torch.exp(-torch.tensor(1 / 3)),
            ratio1 * torch.special.modified_bessel_k0(torch.tensor(1 / 4)),
            ratio2 * torch.special.modified_bessel_k0(torch.tensor(1)),
        ]
    )
    torch.testing.assert_close(out, expected)


def test_tuning(x, y):
    kappa = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2
    kernel = nn.Tuning(nn.Matrix(kappa, "cell_type"), "ori")
    out = kernel(x, y)
    # Δ ori = [45, 90, 0, 45], cos(Δ ori) = [0, -1, 1, 0]
    # 2 * kappa = [1, 3, 4, 2]
    expected = torch.tensor([1.0, -2.0, 5.0, 1.0])
    torch.testing.assert_close(out, expected)


def test_norm(x2, y2):
    sigma = (torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 2).sqrt()
    kernel = nn.Gaussian(nn.Matrix(sigma, "cell_type"), "space")
    out = nn.Norm(kernel, "cell_type", ord=1)(x2, y2)
    # (x - y)^2 = [0.25, 1, 1, 4, 1], 2 * sigma^2 = [1, 3, 4, 2, 2]
    expected = torch.exp(-torch.tensor([0.25, 1 / 3, 1 / 4, 2, 1 / 2]))
    norm = expected[3] + expected[4]
    expected[3] = norm
    expected[4] = norm
    torch.testing.assert_close(out, expected)


def test_norm2(x2, y2):
    sigma1 = torch.tensor([[3.0, 1.5], [1.0, 2.0]])
    sigma2 = sigma1 / 2
    prod_kernel = nn.Laplace(nn.Matrix(sigma1, "cell_type"), "space", d=2)
    prob_kernel = nn.Laplace(nn.Matrix(sigma2, "cell_type"), "space", d=0)
    strength_kernel = nn.Monotonic(prod_kernel / prob_kernel, "space")
    kernel = (
        strength_kernel
        * nn.Norm(prod_kernel, "cell_type", ord=1)
        / nn.Norm(strength_kernel * prob_kernel, "cell_type", ord=1)
    )
    out = kernel(x2, y2)
    # norm(x - y) = [0.5, 1, 1, 2, 1], sigma1 = [3, 1, 2, 1.5, 1.5], sigma2 = [1.5, 0.5, 1, 0.75, 0.75]
    expected = strength_kernel(x2, y2)
    expected[1] = (
        expected[1] * prod_kernel(x2, y2)[1] / (expected[1] * prob_kernel(x2, y2)[1])
    )
    target_norm = prod_kernel(x2, y2)[3:].sum()
    current_norm = (expected[3:] * prob_kernel(x2, y2)[3:]).sum()
    expected[3:] = expected[3:] * target_norm / current_norm

    torch.testing.assert_close(out, expected)
