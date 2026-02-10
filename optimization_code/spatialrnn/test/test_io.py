import json

import pytest
import pandas as pd
from niarb import io


def test_load_config(tmp_path):
    filename1 = tmp_path / "config1.json"
    filename2 = tmp_path / "config2.json"

    config1 = {"b": [3, 4]}
    config2 = {
        "__matrix__": {"a": [1, 2], "b": {"__ref__": [str(filename1), "b"]}},
        "__include__": [{"c": {"__eval__": "a + b"}}],
    }

    with open(filename1, "w") as f:
        json.dump(config1, f)

    with open(filename2, "w") as f:
        json.dump(config2, f)

    out = io.load_config(filename2)

    expected = [
        {"a": 1, "b": 3, "c": 4},
        {"a": 1, "b": 4, "c": 5},
        {"a": 2, "b": 3, "c": 5},
        {"a": 2, "b": 4, "c": 6},
    ]
    assert out == expected


@pytest.mark.parametrize("multi", [False, True])
def test_load_data(tmp_path, multi):
    df = pd.DataFrame(
        {
            "a": ["b", "hi", "bye", pd.NA, "hihi"],
            "mean": [0.1, 5.0, float("nan"), 6.0, 7.0],
            "std": [0.2, float("nan"), 1.0, 2.0, 3.0],
        }
    )
    filename = tmp_path / "data.pkl"
    df.to_pickle(filename)

    if multi:
        df2 = df.copy()
        df2.loc[4, "mean"] = 10.0
        filename2 = tmp_path / "data2.pkl"
        df2.to_pickle(filename2)
        out = io.load_data(
            [filename, filename2], query="a == 'hihi'", y="mean", yerr="std"
        )
        expected = [
            pd.DataFrame({"a": ["hihi"], "dr": [7.0], "dr_se": [3.0]}, index=[4]),
            pd.DataFrame({"a": ["hihi"], "dr": [10.0], "dr_se": [3.0]}, index=[4]),
        ]
        for o, e in zip(out, expected):
            pd.testing.assert_frame_equal(o, e)

    else:
        out = io.load_data(filename, query="a == 'hihi'", y="mean", yerr="std")
        expected = pd.DataFrame({"a": ["hihi"], "dr": [7.0], "dr_se": [3.0]}, index=[4])
        pd.testing.assert_frame_equal(out, expected)


@pytest.mark.parametrize("indices", [0, [0, 2]])
def test_iterdir(tmp_path, indices):
    d = {"a": 1}
    filenames = ["0.29.json", "1e-4.pt", "1.25.json", "1.0e-3.json"]
    for filename in filenames:
        with open(tmp_path / filename, "w") as f:
            json.dump(d, f)

    out = io.iterdir(tmp_path, pattern="*.json", indices=indices)

    if indices == 0:
        expected = tmp_path / "1.0e-3.json"
    else:
        expected = [tmp_path / "1.0e-3.json", tmp_path / "1.25.json"]

    assert out == expected
