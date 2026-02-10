import torch
import pandas as pd

from niarb import dataframe


def test_from_tensor():
    tensor = torch.tensor([[1, 2], [3, 4]])
    dim = ["row", "column"]
    coord = [["A", "B"], ["X", "Y"]]
    out = dataframe.from_tensor(tensor, dim=dim, coord=coord)
    expected = pd.DataFrame(
        {
            "row": ["A", "A", "B", "B"],
            "column": ["X", "Y", "X", "Y"],
            "value": [1, 2, 3, 4],
        }
    )
    expected["row"] = expected["row"].astype("category")
    expected["column"] = expected["column"].astype("category")
    pd.testing.assert_frame_equal(out, expected)


def test_from_tensor_broadcasted():
    tensor = torch.tensor([[1, 2], [3, 4]])
    dim = ["column"]
    coord = [["X", "Y"]]
    out = dataframe.from_tensor(tensor, dim=dim, coord=coord)
    expected = pd.DataFrame(
        {
            "column": ["X", "Y", "X", "Y"],
            "value": [1, 2, 3, 4],
        }
    )
    expected["column"] = expected["column"].astype("category")
    pd.testing.assert_frame_equal(out, expected)


def test_from_tensor_empty():
    tensor = torch.tensor([[1, 2], [3, 4]])
    out = dataframe.from_tensor(tensor)
    expected = pd.DataFrame({"value": [1, 2, 3, 4]})
    pd.testing.assert_frame_equal(out, expected)


def test_from_tensor_no_coord():
    tensor = torch.tensor([[1, 2], [3, 4]])
    dim = ["row", "column"]
    out = dataframe.from_tensor(tensor, dim=dim)
    expected = pd.DataFrame(
        {
            "row": [0, 0, 1, 1],
            "column": [0, 1, 0, 1],
            "value": [1, 2, 3, 4],
        }
    )
    expected["row"] = expected["row"].astype("category")
    expected["column"] = expected["column"].astype("category")
    pd.testing.assert_frame_equal(out, expected)


def test_from_state_dict():
    state_dict = {
        "a": torch.tensor([1, 2]),
        "b": torch.tensor([[5, 6], [7, 8]]),
    }
    dims = {"b": ["column"]}
    coords = {"b": [["X", "Y"]]}
    out = dataframe.from_state_dict(state_dict, dims=dims, coords=coords)
    expected = pd.DataFrame(
        {
            "variable": ["a", "a", "b", "b", "b", "b"],
            "value": [1, 2, 5, 6, 7, 8],
            "column": [float("nan"), float("nan"), "X", "Y", "X", "Y"],
        }
    )
    expected["variable"] = expected["variable"].astype("category")
    expected["column"] = expected["column"].astype("category")
    pd.testing.assert_frame_equal(out, expected)
