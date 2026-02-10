import operator
import pickle

import pytest
import torch
import numpy as np
import pandas as pd

from niarb.tensors import categorical, periodic


def test_numpy():
    x = categorical.tensor([0, 2, 1, 2], categories=["a", "b", "c"])
    out = x.numpy()
    assert (out == np.array(["a", "c", "b", "c"])).all()


def test_pandas():
    x = categorical.tensor([0, 2, 1, 2], categories=["a", "b", "c"])
    out = x.pandas()
    expected = pd.Categorical(["a", "c", "b", "c"], categories=["a", "b", "c"])
    assert isinstance(out, pd.Categorical)
    assert list(out.categories) == ["a", "b", "c"]
    assert out.equals(expected)


@pytest.mark.parametrize(
    "op, kind, y, expected",
    [
        (("torch", "eq"), "category", 0, [False, True, False, True]),
        (("torch", "eq"), "category", [0, 0, 1, 2], [False, True, False, False]),
        (("torch", "eq"), "tensor", 0, [True, False, False, False]),
        (("torch", "eq"), "tensor", [0, 0, 1, 2], [True, False, True, True]),
        (("torch", "ne"), "category", 0, [True, False, True, False]),
        (("torch", "ne"), "category", [0, 0, 1, 2], [True, False, True, True]),
        (("torch", "ne"), "tensor", 0, [False, True, True, True]),
        (("torch", "ne"), "tensor", [0, 0, 1, 2], [False, True, False, False]),
        (("operator", "eq"), "category", 0, [False, True, False, True]),
        (("operator", "eq"), "category", [0, 0, 1, 2], [False, True, False, False]),
        (("operator", "eq"), "tensor", 0, [True, False, False, False]),
        (("operator", "eq"), "tensor", [0, 0, 1, 2], [True, False, True, True]),
        (("operator", "eq"), "hashable", 0, [False, True, False, True]),
        (("operator", "eq"), "scalar", 0, [True, False, False, False]),
        (("operator", "ne"), "category", 0, [True, False, True, False]),
        (("operator", "ne"), "category", [0, 0, 1, 2], [True, False, True, True]),
        (("operator", "ne"), "tensor", 0, [False, True, True, True]),
        (("operator", "ne"), "tensor", [0, 0, 1, 2], [False, True, False, False]),
        (("operator", "ne"), "hashable", 0, [True, False, True, False]),
        (("operator", "ne"), "scalar", 0, [False, True, True, True]),
    ],
)
@pytest.mark.parametrize("device", pytest.devices)
def test_comparison(op, kind, y, expected, device):
    y_cat = ["c", "d", "e"]
    x = categorical.tensor([0, 2, 1, 2], categories=["a", "b", "c"], device=device)
    if kind == "category":
        y = categorical.tensor(y, categories=y_cat, device=device)
    elif kind == "tensor":
        y = torch.tensor(y, device=device)
    elif kind == "hashable":
        y = y_cat[y]
    expected = torch.tensor(expected, device=device)

    op = getattr(torch, op[1]) if op[0] == "torch" else getattr(operator, op[1])

    out = op(x, y)
    assert (out == expected).all()

    out = op(y, x)
    assert (out == expected).all()


@pytest.mark.parametrize("func", ["stack", "cat", "hstack", "vstack"])
@pytest.mark.parametrize("y_is_categorical", [True, False])
def test_combine(func, y_is_categorical):
    x = categorical.tensor([0, 2, 1, 2], categories=["a", "b", "c"])
    if y_is_categorical:
        y = categorical.tensor([2, 1, 0, 0], categories=["a", "b", "c"])
    else:
        y = torch.tensor([2, 1, 0, 0])

    out = getattr(torch, func)([x, y])
    expected = getattr(torch, func)([x.tensor, y.as_subclass(torch.Tensor)])
    if y_is_categorical:
        expected = categorical.as_tensor(expected, categories=["a", "b", "c"])

    assert type(out) is type(expected)
    assert (out == expected).all()


@pytest.mark.parametrize("func", ["__getitem__", "take_along_dim"])
@pytest.mark.parametrize("x_is_periodic", [True, False])
def test_index_select(func, x_is_periodic):
    if x_is_periodic:
        x = periodic.tensor([1.0, 2.0, 2.5], w_dims=[0], extents=[(-5, 5)])
    else:
        x = torch.tensor([1.0, 2.0, 2.5])

    y = categorical.tensor([0, 2, 1], categories=["a", "b", "c"])
    if func == "__getitem__":
        out = x[y]
    else:
        out = getattr(x, func)(y)

    if x_is_periodic:
        expected = periodic.tensor([1.0, 2.5, 2.0], w_dims=[0], extents=[(-5, 5)])
        assert out.w_dims == expected.w_dims
        assert out.extents == expected.extents
    else:
        expected = torch.tensor([1.0, 2.5, 2.0])

    assert type(out) is type(expected)
    assert (out == expected).all()


@pytest.mark.parametrize("weights", [None, [0.5, 1.0, 2.0, 0.5]])
def test_bincount(weights):
    x = categorical.tensor([0, 2, 1, 2], categories=["a", "b", "c"])
    if weights:
        weights = torch.tensor(weights)
    out = torch.bincount(x, weights=weights)
    expected = torch.bincount(x.tensor, weights=weights)
    assert type(out) is type(expected) and (out == expected).all()


@pytest.mark.parametrize("return_inverse", [False, True])
@pytest.mark.parametrize("return_counts", [False, True])
@pytest.mark.parametrize("func", ["unique", "unique_consecutive"])
def test_unique(func, return_inverse, return_counts):
    x = categorical.tensor([0, 1, 1, 2], categories=["a", "b", "c"])

    out = getattr(torch, func)(
        x, return_inverse=return_inverse, return_counts=return_counts
    )

    expected = (categorical.tensor([0, 1, 2], categories=["a", "b", "c"]),)
    if return_inverse:
        expected = (*expected, torch.tensor([0, 1, 1, 2]))
    if return_counts:
        expected = (*expected, torch.tensor([1, 2, 1]))

    if len(expected) == 1:
        assert type(out) is type(expected[0]) and (out == expected[0]).all()
    else:
        assert all(
            type(v1) is type(v2) and (v1 == v2).all()
            for v1, v2 in zip(out, expected, strict=True)
        )


def test_pickle(tmp_path):
    x = categorical.tensor([0, 2, 1, 2], categories=["a", "b", "c"])

    with open(tmp_path / "x.pkl", "wb") as f:
        pickle.dump(x, f)

    with open(tmp_path / "x.pkl", "rb") as f:
        y = pickle.load(f)

    assert type(x) is type(y)
    assert (x == y).all()


@pytest.mark.parametrize("b_type", ["tensor", "parameter", "periodic"])
def test_broadcast_tensors(b_type):
    a = categorical.tensor([[0], [1], [2]], categories=["a", "b", "c"])
    if b_type in {"tensor", "parameter"}:
        b = torch.tensor([[1.0, 2.0]])
    elif b_type == "periodic":
        b = periodic.tensor([[1.0, 2.0]])
    if b_type == "parameter":
        b = torch.nn.Parameter(b)
    out = torch.broadcast_tensors(a, b)
    expected = a.broadcast_to((3, 2)), b.broadcast_to((3, 2))
    assert all(
        type(v1) is type(v2) and (v1 == v2).all()
        for v1, v2 in zip(out, expected, strict=True)
    )
