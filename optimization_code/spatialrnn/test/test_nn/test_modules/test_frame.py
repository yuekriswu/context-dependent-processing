import contextlib

import pytest
import pandas as pd
import numpy as np
import torch
import hyclib as lib
import tdfl

from niarb.nn.modules import frame
from niarb.tensors import categorical


@pytest.fixture
def a():
    with lib.random.set_seed(0):
        return torch.randn(2, 1, 4)


@pytest.fixture
def b():
    with lib.random.set_seed(0):
        return torch.randn(1, 3, 3, 2)


@pytest.fixture
def ab(a):
    return a.broadcast_to(2, 3, 4)


@pytest.fixture
def ba(b):
    return b.broadcast_to(2, 3, 3, 2)


@pytest.fixture
def mask():
    return torch.tensor([[True, False, True], [False, True, True]])


@pytest.fixture
def df(a, b):
    return frame.ParameterFrame({"a": a, "b": b}, ndim=2)


def test_broadcast():
    df0 = frame.ParameterFrame(
        {
            "a": torch.randn(3, 1, 4),
            "b": torch.randn(1, 1, 3, 2),
        },
        ndim=2,
    )
    df1 = frame.ParameterFrame(
        {
            "a": torch.randn(1, 1, 2, 6),
            "b": torch.randn(5, 1, 1),
        },
        ndim=3,
    )
    df0, df1 = frame.broadcast(df0, df1)
    assert df0.shape == (5, 3, 2)
    assert df1.shape == (5, 3, 2)
    shapes0 = {"a": (5, 3, 2, 4), "b": (5, 3, 2, 3, 2)}
    shapes1 = {"a": (5, 3, 2, 6), "b": (5, 3, 2)}
    assert all(df0.data[k].shape == v for k, v in shapes0.items())
    assert all(df1.data[k].shape == v for k, v in shapes1.items())


@pytest.mark.parametrize(
    "dims, sparse, shape0, shape1, shapes0, shapes1",
    [
        (
            None,
            False,
            (2, 3, 5, 2, 4),
            (2, 3, 5, 2, 4),
            {"a": (2, 1, 5, 2, 1, 4), "b": (1, 3, 5, 1, 4, 3, 2)},
            {"a": (2, 1, 5, 2, 1, 6), "b": (1, 3, 5, 1, 4)},
        ),
        (
            -1,
            False,
            (2, 5, 2, 3),
            (2, 5, 2, 4),
            {"a": (2, 5, 2, 1, 4), "b": (1, 5, 1, 3, 3, 2)},
            {"a": (2, 5, 2, 1, 6), "b": (1, 5, 1, 4)},
        ),
        (
            [(-1, None), (-1, None)],
            False,
            (2, 3, 4),
            (5, 2, 3, 4),
            {"a": (2, 1, 1, 4), "b": (1, 3, 4, 3, 2)},
            {"a": (5, 2, 1, 1, 6), "b": (5, 1, 3, 4)},
        ),
        (
            None,
            True,
            (2, 3, 1, 1, 1),
            (1, 1, 5, 2, 4),
            {"a": (2, 1, 1, 1, 1, 4), "b": (1, 3, 1, 1, 1, 3, 2)},
            {"a": (1, 1, 5, 2, 1, 6), "b": (1, 1, 5, 1, 4)},
        ),
        (
            -1,
            True,
            (2, 1, 1, 3),
            (1, 5, 2, 4),
            {"a": (2, 1, 1, 1, 4), "b": (1, 1, 1, 3, 3, 2)},
            {"a": (1, 5, 2, 1, 6), "b": (1, 5, 1, 4)},
        ),
        (
            [(-1, None), (-1, None)],
            True,
            (2, 3, 1),
            (5, 2, 1, 4),
            {"a": (2, 1, 1, 4), "b": (1, 3, 1, 3, 2)},
            {"a": (5, 2, 1, 1, 6), "b": (5, 1, 1, 4)},
        ),
    ],
)
def test_meshgrid(dims, sparse, shape0, shape1, shapes0, shapes1):
    df0 = frame.ParameterFrame(
        {
            "a": torch.randn(2, 1, 4),
            "b": torch.randn(1, 3, 3, 2),
        },
        ndim=2,
    )
    df1 = frame.ParameterFrame(
        {
            "a": torch.randn(5, 2, 1, 6),
            "b": torch.randn(5, 1, 4),
        },
        ndim=3,
    )
    df0, df1 = frame.meshgrid(df0, df1, dims=dims, sparse=sparse)
    assert df0.shape == shape0
    assert df1.shape == shape1
    assert all(df0.data[k].shape == v for k, v in shapes0.items())
    assert all(df1.data[k].shape == v for k, v in shapes1.items())


@pytest.mark.parametrize(
    "dims", [None, -1, 1, [(-1, None), (None, 1)], [(None, None), (0, None)]]
)
@pytest.mark.parametrize("sparse", [True, False])
def test_meshgrid_0d(dims, sparse):
    """
    Fix edge case bug when frame.ndim == 0. Should always return
    unmodified ParameterFrames.
    """
    df0 = frame.ParameterFrame(
        {
            "a": torch.randn(2, 1, 4),
            "b": torch.randn(1, 3, 3, 2),
        },
        ndim=0,
    )
    df1 = frame.ParameterFrame(
        {
            "a": torch.randn(5, 2, 1, 6),
            "b": torch.randn(5, 1, 4),
        },
        ndim=0,
    )
    shapes0 = dict(zip(df0.keys(), df0.shapes()))
    shapes1 = dict(zip(df1.keys(), df1.shapes()))
    df0, df1 = frame.meshgrid(df0, df1, dims=dims, sparse=sparse)
    assert df0.shape == ()
    assert df1.shape == ()
    assert all(df0.data[k].shape == v for k, v in shapes0.items())
    assert all(df1.data[k].shape == v for k, v in shapes1.items())


@pytest.mark.parametrize(
    "dim, shape, shapes",
    [
        (0, (5, 2, 3), {"a": (5, 2, 1, 4), "b": (5, 1, 3, 3, 2)}),
        (1, (2, 5, 3), {"a": (2, 5, 1, 4), "b": (1, 5, 3, 3, 2)}),
        (-2, (2, 5, 3), {"a": (2, 5, 1, 4), "b": (1, 5, 3, 3, 2)}),
        (-1, (2, 3, 5), {"a": (2, 1, 5, 4), "b": (1, 3, 5, 3, 2)}),
    ],
)
def test_stack(df, dim, shape, shapes):
    df = frame.stack([df] * 5, dim=dim)
    assert df.shape == shape and df.ndim == len(shape)
    assert all(df.data[k].shape == v for k, v in shapes.items())


@pytest.mark.parametrize(
    "columns, keys, delimiter, as_dict, out_columns, raises_error",
    [
        (["c", "d"], None, "_", False, ["a", "b", "c", "d"], False),
        (["a", "b"], ["0", "1"], "_", False, ["0_a", "0_b", "1_a", "1_b"], False),
        (["a", "b"], ["0", "1"], "-", False, ["0-a", "0-b", "1-a", "1-b"], False),
        (["a", "b"], None, "_", False, None, True),
        (["a", "b"], ["0", "0"], "_", False, None, True),
        (["a", "b"], ["0", "1"], "_", True, ["0_a", "0_b", "1_a", "1_b"], False),
        (["a", "b"], ["0", "1"], "-", True, ["0-a", "0-b", "1-a", "1-b"], False),
    ],
)
def test_concat_cols(df, columns, keys, delimiter, out_columns, raises_error, as_dict):
    a, b, c, d = (
        torch.randn(2, 1, 4),
        torch.randn(1, 3, 3, 2),
        torch.randn(1, 3, 6),
        torch.randn(2, 3, 4),
    )
    df0 = frame.ParameterFrame({"a": a, "b": b}, ndim=2)
    df1 = frame.ParameterFrame({columns[0]: c, columns[1]: d}, ndim=2)

    if as_dict:
        dfs = {keys[0]: df0, keys[1]: df1}
        keys = None
    else:
        dfs = [df0, df1]

    with pytest.raises(ValueError) if raises_error else contextlib.nullcontext():
        df = frame.concat(dfs, dim=-1, keys=keys, delimiter=delimiter)

    if not raises_error:
        values = [a, b, c, d]
        assert all(k0 == k1 for k0, k1 in zip(df.keys(), out_columns))
        assert all(
            v0.shape == v1.shape and (v0 == v1).all()
            for v0, v1 in zip(df._values(), values)
        )


def test_ndim(df):
    assert df.ndim == 2


def test_shape(df):
    assert df.shape == (2, 3)


def test_ndims(df):
    assert tuple(df.ndims()) == (1, 2)


def test_shapes(df):
    shapes = [(4,), (3, 2)]
    assert all(s0 == s1 for s0, s1 in zip(df.shapes(), shapes))


def test_getitem(ab, ba, mask, df):
    a_, b_ = df["a"], df["b"]
    assert a_.shape == ab.shape and (a_ == ab).all()
    assert b_.shape == ba.shape and (b_ == ba).all()
    sf = df[["a", "b"]]
    assert sf.shape == (2, 3)
    a_, b_ = sf.data["a"], sf.data["b"]
    assert a_.shape == ab.shape and (a_ == ab).all()
    assert b_.shape == ba.shape and (b_ == ba).all()
    sf = df[mask]
    assert sf.shape == (4,)
    assert (sf.data["a"] == ab[mask]).all()
    assert (sf.data["b"] == ba[mask]).all()


def test_iloc(ab, ba, df):
    sf = df.iloc[..., 0]
    a_, b_ = sf.data["a"], sf.data["b"]
    ab_, ba_ = ab[:, 0], ba[:, 0]
    assert a_.shape == ab_.shape and (a_ == ab_).all()
    assert b_.shape == ba_.shape and (b_ == ba_).all()


def test_data(a, b, df):
    a_, b_ = df.data["a"], df.data["b"]
    assert a_.shape == a.shape and (a_ == a).all()
    assert b_.shape == b.shape and (b_ == b).all()
    sf = df.data[["a", "b"]]
    assert sf.shape == (2, 3)
    a_, b_ = sf.data["a"], sf.data["b"]
    assert a_.shape == a.shape and (a_ == a).all()
    assert b_.shape == b.shape and (b_ == b).all()


def test_setitem(a, b, ab, ba):
    df = frame.ParameterFrame(ndim=2)
    df["a"] = a
    df["b"] = b
    assert df["a"].shape == ab.shape and (df["a"] == ab).all()
    assert df["b"].shape == ba.shape and (df["b"] == ba).all()


def test_keys(df):
    keys = ["a", "b"]
    assert all(k0 == k1 for k0, k1 in zip(df.keys(), keys))


def test__values(a, b, df):
    values = [a, b]
    assert all(
        v0.shape == v1.shape and (v0 == v1).all()
        for v0, v1 in zip(df._values(), values)
    )


def test_values(a, b, ab, ba, df):
    values = [ab, ba]
    assert all(
        v0.shape == v1.shape and (v0 == v1).all() for v0, v1 in zip(df.values(), values)
    )


def test__items(a, b, df):
    items = [("a", a), ("b", b)]
    assert all(
        k0 == k1 and v0.shape == v1.shape and (v0 == v1).all()
        for (k0, v0), (k1, v1) in zip(df._items(), items)
    )


def test_items(a, b, ab, ba, df):
    items = [("a", ab), ("b", ba)]
    assert all(
        k0 == k1 and v0.shape == v1.shape and (v0 == v1).all()
        for (k0, v0), (k1, v1) in zip(df.items(), items)
    )


def test_broadcast_to(a, b, df):
    df = df.broadcast_to(5, 2, 3)
    assert df.shape == (5, 2, 3)
    a_, b_ = df.data["a"], df.data["b"]
    assert a_.shape == (5, 2, 3, 4) and (a_ == a).all()
    assert b_.shape == (5, 2, 3, 3, 2) and (b_ == b).all()


@pytest.mark.parametrize("as_tuple", [True, False])
@pytest.mark.parametrize("shape", [(-1,), (2, 1, 3), (6, 1)])
def test_reshape(ab, ba, df, shape, as_tuple):
    df = df.reshape(shape) if as_tuple else df.reshape(*shape)
    ab, ba = ab.reshape(*shape, 4), ba.reshape(*shape, 3, 2)
    assert df.shape == ab.shape[:-1]
    assert df["a"].shape == ab.shape and (df["a"] == ab).all()
    assert df["b"].shape == ba.shape and (df["b"] == ba).all()


@pytest.mark.parametrize(
    "dim, expected",
    [
        (0, {"a": (1, 2, 1, 4), "b": (1, 1, 3, 3, 2)}),
        (2, {"a": (2, 1, 1, 4), "b": (1, 3, 1, 3, 2)}),
        (-1, {"a": (2, 1, 1, 4), "b": (1, 3, 1, 3, 2)}),
    ],
)
def test_unsqueeze(df, dim, expected):
    out = df.unsqueeze(dim)
    for k, v in out._items():
        assert v.shape == expected[k]


@pytest.mark.parametrize("cls", ["tdfl", "pd"])
@pytest.mark.parametrize("keep_indices", [True, False])
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("to_numpy", [True, False])
def test_to_framelike(cls, keep_indices, as_index, to_numpy):
    a = torch.tensor([[1.0], [2.0]])
    b = torch.tensor([[[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
    c = categorical.tensor([[0, 1, 2], [1, 0, 2]], categories=("d", "e", "f"))

    df = frame.ParameterFrame({"a": a, "b": b, "c": c}, ndim=2)
    cls = tdfl.DataFrame if cls == "tdfl" else pd.DataFrame
    out = df.to_framelike(cls, keep_indices, as_index, to_numpy)

    assert isinstance(out, cls)

    if cls is pd.DataFrame:
        expected = {
            "a": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "b[0]": [3.0, 5.0, 7.0, 3.0, 5.0, 7.0],
            "b[1]": [4.0, 6.0, 8.0, 4.0, 6.0, 8.0],
            "c": ["d", "e", "f", "e", "d", "f"],
        }
        index = None
        if keep_indices:
            if as_index:
                index = pd.MultiIndex.from_tuples(
                    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
                )
            else:
                expected["idx[0]"] = [0, 0, 0, 1, 1, 1]
                expected["idx[1]"] = [0, 1, 2, 0, 1, 2]
        expected = pd.DataFrame(expected, index=index)

        expected["c"] = expected["c"].astype("category")
        for k in ["a", "b[0]", "b[1]"]:
            expected[k] = expected[k].astype("float32")

        pd.testing.assert_frame_equal(out, expected)
    elif to_numpy:
        expected = tdfl.DataFrame(
            {
                "a": np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32),
                "b[0]": np.array([3.0, 5.0, 7.0, 3.0, 5.0, 7.0], dtype=np.float32),
                "b[1]": np.array([4.0, 6.0, 8.0, 4.0, 6.0, 8.0], dtype=np.float32),
                "c": np.array(["d", "e", "f", "e", "d", "f"]),
            }
        )
        if keep_indices and not as_index:
            expected["idx[0]"] = np.array([0, 0, 0, 1, 1, 1])
            expected["idx[1]"] = np.array([0, 1, 2, 0, 1, 2])
        assert out.columns == expected.columns
        for k in out.columns:
            assert isinstance(out[k], np.ndarray)
            assert (out[k] == expected[k]).all()
            assert out[k].dtype == expected[k].dtype
    else:
        bb = b.broadcast_to(2, 3, 2).reshape(-1, 2)
        expected = tdfl.DataFrame(
            {
                "a": torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
                "b[0]": bb[:, 0],
                "b[1]": bb[:, 1],
                "c": categorical.tensor([0, 1, 2, 1, 0, 2], categories=("d", "e", "f")),
            }
        )
        if keep_indices and not as_index:
            expected["idx[0]"] = torch.tensor([0, 0, 0, 1, 1, 1])
            expected["idx[1]"] = torch.tensor([0, 1, 2, 0, 1, 2])
        assert out.columns == expected.columns
        for k in out.columns:
            assert isinstance(out[k], torch.Tensor)
            assert (out[k] == expected[k]).all()
            assert out[k].dtype == expected[k].dtype
        assert out["b[0]"].requires_grad
        assert out["b[1]"].requires_grad
        assert isinstance(out["c"], categorical.CategoricalTensor)
        assert out["c"].categories == expected["c"].categories
