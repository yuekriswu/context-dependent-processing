import json

import pytest
import pandas as pd
import numpy as np

from niarb import utils, io


def test_compose():
    def f(x):
        return x + 1

    def g(x):
        return x * 2

    def h(x):
        return x**2

    assert utils.compose(f, g, h)(3) == f(g(h(3)))


def test_tree_map():
    d = {
        "a": 1,
        "b": {
            "c": {"__indices__": "1,3,5-7"},
            "d": [
                {"__indices__": "0-2"},
                {"e": 2},
            ],
        },
    }
    out = utils.tree_map(d, io.object_hook)
    expected = json.loads(json.dumps(d), object_hook=io.object_hook)
    assert out == expected


def test_tree_map_top():
    d = {
        "__matrix__": {
            "a": [1, 2],
            "b": [3, 4, 5],
        },
        "__include__": [
            {"c": {"__indices__": "1,3,5-7"}},
        ],
    }
    out = utils.tree_map(d, io.object_hook)
    expected = json.loads(json.dumps(d), object_hook=io.object_hook)
    assert out == expected


@pytest.mark.parametrize(
    "centers, window, expected",
    [
        ([1, 3, 5], 2, [[1, 3, 5]]),
        ([1, 3, 5], 3, [[1, 5], [3]]),
        ([1, 2, 3, 4, 5], 3, [[1, 4], [2, 5], [3]]),
    ],
)
def test_nonoverlapping_partition(centers, window, expected):
    assert utils.nonoverlapping_partition(centers, window) == expected


def test_rolling():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
    out = utils.rolling(df, "a", [1, 3, 5], 3)
    expected = pd.DataFrame(
        [
            [(-0.5, 2.5), "a"],
            [(-0.5, 2.5), "b"],
            [pd.NA, "c"],
            [(3.5, 6.5), "d"],
            [(3.5, 6.5), "e"],
            [pd.NA, "a"],
            [(1.5, 4.5), "b"],
            [(1.5, 4.5), "c"],
            [(1.5, 4.5), "d"],
            [pd.NA, "e"],
        ],
        columns=["a", "b"],
    )
    expected["a"] = pd.IntervalIndex.from_tuples(expected["a"], closed="left")
    expected["a"] = expected["a"].astype("category")
    expected["a"] = expected["a"].cat.reorder_categories(
        pd.IntervalIndex.from_tuples(
            [(-0.5, 2.5), (3.5, 6.5), (1.5, 4.5)], closed="left"
        )
    )
    pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


@pytest.mark.parametrize("is_dict", [False, True])
def test_concat(is_dict):
    df1 = pd.DataFrame({"a": pd.Categorical([0, 0, 1, 1]), "b": 1.0})
    df2 = pd.DataFrame({"a": pd.Categorical([2, 2, 3, 1]), "c": 2.0})

    if is_dict:
        out = utils.concat({"df1": df1, "df2": df2})
    else:
        out = utils.concat([df1, df2], ignore_index=True)

    index = None
    if is_dict:
        index = pd.MultiIndex.from_product([["df1", "df2"], range(4)])
    expected = pd.DataFrame(
        {
            "a": pd.Categorical([0, 0, 1, 1, 2, 2, 3, 1]),
            "b": [1.0] * 4 + [np.nan] * 4,
            "c": [np.nan] * 4 + [2.0] * 4,
        },
        index=index,
    )
    pd.testing.assert_frame_equal(out, expected)
