import contextlib
import pickle

import pytest
import torch
import numpy as np

from niarb.tensors import periodic


atan21 = np.arctan2(2.0, 1.0) / np.pi * 180
atan12 = np.arctan2(1.0, 2.0) / np.pi * 180
sqrt2 = 2**0.5
sqrt5 = 5**0.5


@pytest.fixture
def x():
    return periodic.tensor(
        [
            [[0.0, 1.0], [90.0, 3.0], [90.0, 5.0]],
            [[0.0, 2.0], [0.0, 6.0], [90.0, 10.0]],
            [[90.0, 3.0], [90.0, 9.0], [90.0, 15.0]],
        ],
        w_dims=[0],
        extents=[(-180.0, 180.0)],
    )


@pytest.fixture
def x2():
    return periodic.tensor(
        [
            [-90.0],
            [-45.0],
            [0.0],
            [45.0],
            [90.0],
        ],
        w_dims=[0],
        extents=[(-90.0, 90.0)],
    )


@pytest.fixture
def x3():
    return periodic.tensor(
        [
            [[0.0, 1.0, 0.0], [90.0, 3.0, 90.0], [90.0, 5.0, 90.0]],
            [[0.0, 2.0, 0.0], [0.0, 6.0, 0.0], [90.0, 10.0, 90.0]],
            [[90.0, 3.0, 90.0], [90.0, 9.0, 90.0], [90.0, 15.0, 90.0]],
        ],
        w_dims=[0, 2],
        extents=[(-180.0, 180.0), (-180.0, 180.0)],
    )


@pytest.fixture
def x4():
    return periodic.tensor(
        [
            [-45.0, 0.0, -135.0],
            [0.0, 1.0, -90.0],
            [45.0, 2.0, -45.0],
            [90.0, 3.0, 0.0],
            [135.0, 4.0, 45.0],
        ],
        w_dims=[0, 2],
        extents=[(-45.0, 135.0), (-135.0, 45.0)],
    )


@pytest.mark.parametrize(
    "dim, weight, expected, error",
    [
        (0, None, [[atan12, 2.0], [atan21, 6.0], [90.0, 10.0]], None),
        (1, None, [[atan21, 3.0], [atan12, 6.0], [90.0, 9.0]], None),
        (0, [[0.0], [1.0], [1.0]], [[45.0, 2.5], [45.0, 7.5], [90.0, 12.5]], None),
        (1, [[1.0, 1.0, 0.0]], [[45.0, 2.0], [0.0, 4.0], [90.0, 6.0]], None),
        ((0, 1), None, [atan21, 6.0], None),
        (-1, None, None, ValueError),
        (None, None, [atan21, 6.0], None),
    ],
)
@pytest.mark.parametrize(
    "device", pytest.non_mps_devices
)  # MPS does not support complex numbers
def test_cmean(x, dim, weight, expected, error, device):
    x = x.to(device)
    weight = torch.tensor(weight, device=device) if weight is not None else None

    with pytest.raises(error) if error is not None else contextlib.nullcontext():
        output = x.cmean(weight=weight, dim=dim)

    if error is None:
        expected = periodic.tensor(
            expected,
            device=device,
            w_dims=[0],
            extents=[(-180.0, 180.0)],
        ).float()

        assert output.w_dims == expected.w_dims
        assert output.extents == expected.extents
        assert type(output) is type(expected)
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    "dim, weight, expected, error",
    [
        (0, None, [[1 - sqrt5 / 3, 1.0], [1 - sqrt5 / 3, 9.0], [0.0, 25.0]], None),
        (1, None, [[1 - sqrt5 / 3, 4.0], [1 - sqrt5 / 3, 16.0], [0.0, 36.0]], None),
        (
            0,
            [[0.0], [1.0], [1.0]],
            [[1 - sqrt2 / 2, 0.5], [1 - sqrt2 / 2, 4.5], [0.0, 12.5]],
            None,
        ),
        (
            1,
            [[1.0, 1.0, 0.0]],
            [[1 - sqrt2 / 2, 2.0], [0.0, 8.0], [0.0, 18.0]],
            None,
        ),
        ((0, 1), None, [1 - sqrt5 / 3, 83 / 4], None),
        (-1, None, None, ValueError),
        (None, None, [1 - sqrt5 / 3, 83 / 4], None),
    ],
)
@pytest.mark.parametrize(
    "device", pytest.non_mps_devices
)  # MPS does not support complex numbers
def test_cvar(x, dim, weight, expected, error, device):
    x = x.to(device)
    weight = torch.tensor(weight, device=device) if weight is not None else None

    with pytest.raises(error) if error is not None else contextlib.nullcontext():
        output = x.cvar(weight=weight, dim=dim)

    if error is None:
        expected = torch.tensor(expected, device=device)
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("device", pytest.devices)
def test_gprod(x, device):
    x = x.to(device)

    expected = periodic.tensor(
        [
            [[0.0, 3.0], [-90.0, 9.0], [-90.0, 15.0]],
            [[0.0, 6.0], [0.0, 18.0], [-90.0, 30.0]],
            [[-90.0, 9.0], [-90.0, 27.0], [-90.0, 45.0]],
        ],
        device=device,
        w_dims=[0],
        extents=[(-180.0, 180.0)],
    )

    out = x.gprod(x).gprod(x)
    assert out.w_dims == expected.w_dims
    assert out.extents == expected.extents
    assert type(out) is type(expected)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("device", pytest.devices)
def test_ginv(x2, device):
    x2 = x2.to(device)

    expected = periodic.tensor(
        [[90.0], [45.0], [0.0], [-45.0], [-90.0]],
        device=device,
        w_dims=[0],
        extents=[(-90.0, 90.0)],
    )

    out = x2.ginv()
    assert out.w_dims == expected.w_dims
    assert out.extents == expected.extents
    assert type(out) is type(expected)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("device", pytest.devices)
def test_gprod2(x2, device):
    x2 = x2.to(device)

    expected = periodic.tensor(
        [[0.0], [45.0], [-90.0], [-45.0], [0.0]],
        device=device,
        w_dims=[0],
        extents=[(-90.0, 90.0)],
    )

    out = x2.gprod(-90.0)
    assert out.w_dims == expected.w_dims
    assert out.extents == expected.extents
    assert type(out) is type(expected)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("device", pytest.devices)
def test_gprod3(x3, device):
    x3 = x3.to(device)

    expected = periodic.tensor(
        [
            [[0.0, 3.0, 0.0], [-90.0, 9.0, -90.0], [-90.0, 15.0, -90.0]],
            [[0.0, 6.0, 0.0], [0.0, 18.0, 0.0], [-90.0, 30.0, -90.0]],
            [[-90.0, 9.0, -90.0], [-90.0, 27.0, -90.0], [-90.0, 45.0, -90.0]],
        ],
        device=device,
        w_dims=[0, 2],
        extents=[(-180.0, 180.0), (-180.0, 180.0)],
    )

    out = x3.gprod(x3).gprod(x3)
    assert out.w_dims == expected.w_dims
    assert out.extents == expected.extents
    assert type(out) is type(expected)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("device", pytest.devices)
def test_to_period(x2, device):
    x2 = x2.to(device)

    pi = torch.pi
    halfpi = torch.pi / 2

    expected = periodic.tensor(
        [
            [-pi],
            [-halfpi],
            [0.0],
            [halfpi],
            [pi],
        ],
        device=device,
        w_dims=[0],
        extents=[(-pi, pi)],
    )

    out = x2.to_period(2 * pi)

    assert out.w_dims == expected.w_dims
    assert out.extents == expected.extents
    assert type(out) is type(expected)
    torch.testing.assert_close(out, expected)

    out = out.to_period(torch.tensor([180.0], device=device))
    assert out.w_dims == x2.w_dims
    assert out.extents == x2.extents
    assert type(out) is type(x2)
    torch.testing.assert_close(out, x2)


@pytest.mark.parametrize("device", pytest.devices)
def test_to_period2(x4, device):
    x4 = x4.to(device)

    pi = torch.pi
    halfpi = torch.pi / 2
    half3pi = 3 * torch.pi / 2

    expected = periodic.tensor(
        [
            [-halfpi, 0.0, -half3pi],
            [0.0, 1.0, -pi],
            [halfpi, 2.0, -halfpi],
            [pi, 3.0, 0.0],
            [half3pi, 4.0, halfpi],
        ],
        device=device,
        w_dims=[0, 2],
        extents=[(-halfpi, half3pi), (-half3pi, halfpi)],
    )

    out = x4.to_period(2 * pi)

    assert out.w_dims == expected.w_dims
    assert out.extents == expected.extents
    assert type(out) is type(expected)
    torch.testing.assert_close(out, expected)

    out = out.to_period(torch.tensor([180.0, 180.0], device=device))
    assert out.w_dims == x4.w_dims
    assert out.extents == x4.extents
    assert type(out) is type(x4)
    torch.testing.assert_close(out, x4)


@pytest.mark.parametrize(
    "w_dims", [[], [0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]]
)
@pytest.mark.parametrize("device", pytest.devices)
def test_w_dims_slicing(w_dims, device):
    x = periodic.as_tensor(
        torch.rand(2, 4, 3, device=device),
        w_dims=w_dims,
        extents=[(0, 1)] * len(w_dims),
    )
    torch.testing.assert_close(x[..., x.w_dims], x[..., x._w_dims])
    if w_dims == [] or w_dims == [0, 2]:
        assert isinstance(x._w_dims, list)
    else:
        assert isinstance(x._w_dims, slice)


@pytest.mark.parametrize("device", pytest.devices)
def test_tensor_norm(device):
    x = periodic.as_tensor(torch.rand(2, 4, 3, device=device))
    torch.testing.assert_close(x.norm(dim=-1), x.tensor.norm(dim=-1))
    assert isinstance(x.norm(dim=-1), torch.Tensor)
    assert not isinstance(x.norm(dim=-1), periodic.PeriodicTensor)

    with pytest.raises(ValueError):
        x.norm()

    with pytest.raises(ValueError):
        x.norm(dim=0)


@pytest.mark.parametrize(
    "module, func",
    [("torch", "norm"), ("torch.linalg", "norm"), ("torch.linalg", "vector_norm")],
)
@pytest.mark.parametrize("device", pytest.devices)
def test_norm(module, func, device):
    if module == "torch":
        func = getattr(torch, func)
    else:
        func = getattr(torch.linalg, func)
    x = periodic.as_tensor(torch.rand(2, 4, 3, device=device))
    torch.testing.assert_close(func(x, dim=-1), func(x.tensor, dim=-1))
    assert isinstance(func(x, dim=-1), torch.Tensor)
    assert not isinstance(func(x, dim=-1), periodic.PeriodicTensor)

    with pytest.raises(ValueError):
        func(x)

    with pytest.raises(ValueError):
        func(x, dim=0)


@pytest.mark.parametrize("in_dtype", ["float", "long"])
@pytest.mark.parametrize("out_dtype", ["bool", "long", "int", "short"])
@pytest.mark.parametrize("to", [True, False])
@pytest.mark.parametrize("device", pytest.devices)
def test_floating_to_integer(in_dtype, out_dtype, to, device):
    x = periodic.tensor([[1], [2], [3]], device=device, dtype=getattr(torch, in_dtype))
    if to:
        out = x.to(getattr(torch, out_dtype))
    else:
        out = getattr(x, out_dtype)()
    is_periodic = isinstance(out, periodic.PeriodicTensor)
    assert is_periodic if in_dtype == "long" else not is_periodic

    if is_periodic:
        assert out.w_dims == x.w_dims
        assert out.extents == x.extents
        assert type(out) is type(x)
        out = out.tensor
    torch.testing.assert_close(out, getattr(x.tensor, out_dtype)())


@pytest.mark.parametrize("device", pytest.devices)
def test_cat(device):
    x = periodic.tensor(
        [[0.0, 1.0], [90.0, 3.0], [90.0, 5.0]],
        device=device,
        w_dims=[0],
        extents=[(-90.0, 90.0)],
    )
    y = periodic.tensor(
        [[0.0, 2.0], [0.0, 6.0], [90.0, 10.0]],
        device=device,
        w_dims=[0],
        extents=[(-180.0, 180.0)],
    )

    z = torch.cat([x, y], dim=-1)
    assert z.w_dims == (0, 2)
    assert z.extents == ((-90.0, 90.0), (-180.0, 180.0))
    assert z.shape == (3, 4)
    torch.testing.assert_close(z.tensor, torch.cat([x.tensor, y.tensor], dim=-1))

    # z = torch.cat([x, y])
    # assert isinstance(z, torch.Tensor) and not isinstance(z, periodic.PeriodicTensor)
    # torch.testing.assert_close(z, torch.cat([x.tensor, y.tensor]))

    y = periodic.tensor(
        [[0.0, 2.0], [0.0, 6.0], [90.0, 10.0]],
        device=device,
        w_dims=[0],
        extents=[(-90.0, 90.0)],
    )

    z = torch.cat([x, y])
    assert z.w_dims == x.w_dims
    assert z.extents == x.extents
    assert type(z) is type(x)
    torch.testing.assert_close(z.tensor, torch.cat([x.tensor, y.tensor]))


@pytest.mark.parametrize("func", ["__getitem__", "take_along_dim"])
def test_index_select(func):
    x = periodic.tensor([1.0, 2.0, 2.5], w_dims=[0], extents=[(-5.0, 5.0)])
    y = periodic.tensor([0, 2, 1], w_dims=[0], extents=[(0, 2)])
    if func == "__getitem__":
        out = x[y]
    else:
        out = getattr(x, func)(y)
    expected = periodic.tensor([1.0, 2.5, 2.0], w_dims=[0], extents=[(-5.0, 5.0)])
    assert out.w_dims == expected.w_dims
    assert out.extents == expected.extents
    assert type(out) is type(expected)
    assert (out == expected).all()


def test_pickle(tmp_path):
    x = periodic.tensor([[0.0, 1.0], [2.0, 0.5]], w_dims=[1], extents=[(-0.5, 3.0)])

    with open(tmp_path / "x.pkl", "wb") as f:
        pickle.dump(x, f)

    with open(tmp_path / "x.pkl", "rb") as f:
        y = pickle.load(f)

    assert type(x) is type(y)
    assert (x == y).all()
