from collections.abc import Sequence, Hashable
import logging

import numpy as np
from torch import Tensor
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


def from_tensor(
    tensor: Tensor,
    dim: Sequence[str] = (),
    coord: Sequence[Sequence[Hashable]] | None = None,
) -> DataFrame:
    """Convert a tensor to a dataframe.

    Args:
        tensor: A tensor.
        dim (optional): A sequence of dimension names.
        coord (optional): A sequence of coordinate names along each tensor dimension.

    Returns:
        A dataframe with columns [*dim, "value"].

    Raises:
        ValueError: If the dimensions and coordinates are not compatible with the tensor.

    Examples:
        >>> import torch
        >>> tensor = torch.tensor([[1, 2], [3, 4]])
        >>> from_tensor(tensor, dim=["row", "column"], coord=[["A", "B"], ["X", "Y"]])
          row column  value
        0   A      X      1
        1   A      Y      2
        2   B      X      3
        3   B      Y      4

        >>> from_tensor(tensor, dim=["column"], coord=[["X", "Y"]])
          column  value
        0      X      1
        1      Y      2
        2      X      3
        3      Y      4

        >>> from_tensor(tensor, dim=["column"])
          column  value
        0      0      1
        1      1      2
        2      0      3
        3      1      4

    """
    shape = tensor.shape[-len(dim) :] if dim else ()

    if coord is None:
        coord = [list(range(n)) for n in shape]

    if len(dim) != len(coord):
        raise ValueError(
            f"Lengths of dim and coord must be equal, but {len(dim)=}, {len(coord)=}."
        )

    if "value" in dim:
        raise ValueError("dim must not contain element 'value'.")

    if any(len(c) != n for c, n in zip(coord, shape)):
        raise ValueError(
            "Length of each element in coord must equal the size of the "
            f"corresponding dimension of tensor, but {coord=}, {shape=}."
        )

    indices = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")

    df = {}
    for d, c, idx in zip(dim, coord, indices):
        idx = np.broadcast_to(idx, tensor.shape).flatten()
        df[d] = pd.Categorical.from_codes(idx, categories=c)
    df["value"] = tensor.flatten().detach().cpu().numpy()

    return pd.DataFrame(df)


def from_state_dict(
    state_dict: dict[str, Tensor],
    dims: dict[str, Sequence[str]] | None = None,
    coords: dict[str, Sequence[Sequence[Hashable]]] | None = None,
) -> DataFrame:
    """Convert a state dict to a dataframe.

    Args:
        state_dict: A state dict.
        dims (optional): A dictionary of tensor dimension names.
        coords (optional): A dictionary of coordinate names along each tensor dimension.

    Returns:
        A dataframe with columns [*dim, "variable", "value"].

    Raises:
        ValueError: If the dimensions and coordinates are not compatible with the tensors.

    Examples:
        >>> state_dict = {
        ...     "a": torch.tensor([1, 2]),
        ...     "b": torch.tensor([[5, 6], [7, 8]]),
        ... }
        >>> dims = {"b": ["column"]}
        >>> coords = {"b": [["X", "Y"]]}
        >>> from_state_dict(state_dict, dims=dims, coords=coords)
          variable  value column
        0        a      1    NaN
        1        a      2    NaN
        2        b      5      X
        3        b      6      Y
        4        b      7      X
        5        b      8      Y

    """
    if dims is None:
        dims = {}

    if coords is None:
        coords = {}

    if any("variable" in dim for dim in dims.values()):
        raise ValueError("dims must not contain element 'variable'.")

    df = {
        k: from_tensor(v, dim=dims.get(k, ()), coord=coords.get(k))
        for k, v in state_dict.items()
    }
    df = pd.concat(df).reset_index(0, names=["variable"]).reset_index(drop=True)
    df["variable"] = pd.Categorical(df["variable"], categories=list(state_dict.keys()))

    return df


def pivot(
    df: DataFrame,
    columns: str | Sequence[str],
    values: str | Sequence[str],
    template: str | None = None,
    aux_columns: dict[str, str] | None = None,
) -> DataFrame:
    """Like pd.pivot, but does not require index argument and returns a single row.

    Args:
        df: Dataframe to pivot.
        columns: Column(s) to use to make new frame’s columns.
        values: Column(s) to use for populating new frame’s values. If not specified,
          all remaining columns will be used and the result will have hierarchically
          indexed columns.
        template (optional): Turn hierarchical column index into a string via template.
        aux_columns (optional): Add additional columns.

    Returns:
        Pivoted dataframe with a single row.

    """
    df.index = [0] * len(df)  # set a dummy index
    logger.debug(str(df))
    df = df.pivot(columns=columns, values=values)

    if template:
        df.columns = df.columns.to_flat_index()  # flatten MultiIndex to Index of tuples
        df.columns = [template.format(**dict(zip(columns, c))) for c in df.columns]

    if aux_columns:
        for k, v in aux_columns.items():
            df[k] = df.eval(v)

    return df
