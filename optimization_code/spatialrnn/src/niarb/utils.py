import importlib
import logging
from collections.abc import (
    Sequence,
    Mapping,
    Iterable,
    Callable,
    Collection,
    Mapping,
    Hashable,
)
from itertools import starmap, pairwise, chain
import operator
from typing import Any

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import hyclib as lib

logger = logging.getLogger(__name__)


def transpose(
    x: Iterable[Iterable], is_dict=(False, False), fillvalue=None
) -> Iterable[Iterable]:
    """Transpose an iterable of iterables

    Args:
        x: A iterable of iterables
        is_dict: a tuple of booleans indicating whether the outer and inner iterables
          are interpreted as dictionaries respectively.
        fillvalue: Value to fill missing values with. Only used if is_dict[1] is True.

    Returns:
        If is_dict[0] and is_dict[1], a dictionary of dictionaries.
        If is_dict[0] and not is_dict[1], a list of dictionaries.
        If not is_dict[0] and is_dict[1], a dictionary of lists.
        If not is_dict[0] and not is_dict[1], a list of lists.

    """
    if len(is_dict) != 2:
        raise ValueError("is_dict must have length 2.")

    if is_dict[0] and is_dict[1]:
        if not isinstance(x, Mapping):
            raise TypeError(f"Expected a dictionary, but got {type(x)=}.")

        fillvalue = object()
        x = {
            k: {kk: vv for kk, vv in zip(x.keys(), vs) if vv is not fillvalue}
            for k, *vs in lib.itertools.dict_zip(
                *x.values(), mode="union", fillvalue=fillvalue
            )
        }
    elif is_dict[1]:
        x = {
            k: list(vs)
            for k, *vs in lib.itertools.dict_zip(*x, mode="union", fillvalue=fillvalue)
        }
    elif is_dict[0]:
        if not isinstance(x, Mapping):
            raise TypeError(f"Expected a dictionary, but got {type(x)=}.")

        x = [dict(zip(x.keys(), vs)) for vs in zip(*x.values())]
    else:
        x = list(zip(*x))
    return x


def normalize_dim(
    dim: int | Iterable[int], ndim: int, neg: bool = False, sort: bool = True
) -> tuple[int, ...]:
    if not isinstance(dim, Iterable):
        if isinstance(dim, int):
            dim = (dim,)
        else:
            raise TypeError(f"dim must be int or iterable, but {type(dim)=}.")

    dim = tuple(dim)

    if not all(isinstance(d, int) for d in dim):
        raise TypeError(f"Invalid {dim=}, must be an iterable of integers.")

    if any(d >= ndim or d < -ndim for d in dim):
        raise ValueError(f"Invalid {dim=} given {ndim=}.")

    dim = (d % ndim for d in dim)
    if neg:
        dim = (d - ndim for d in dim)
    if sort:
        dim = sorted(dim)
    return tuple(dim)


def call(module, arg):
    if isinstance(module, str):
        module = importlib.import_module(module)

    if isinstance(arg, str):
        return getattr(module, arg)()

    elif isinstance(arg, Sequence):
        if len(arg) == 1:
            return getattr(module, arg[0])()
        elif len(arg) == 2:
            if isinstance(arg[1], Mapping):
                return getattr(module, arg[0])(**arg[1])
            elif isinstance(arg[1], Sequence):
                return getattr(module, arg[0])(*arg[1])
            else:
                raise TypeError(f"Invalid type for {arg[1]=}.")
        elif len(arg) == 3:
            return getattr(module, arg[0])(*arg[1], **arg[2])
        else:
            raise ValueError(f"Invalid number of arguments: {len(arg)=}.")

    raise TypeError(f"Invalid type for {arg=}.")


def take_along_dims(A, *indices, dims=None):
    if dims is None:
        dims = tuple(range(-len(indices), 0))

    ndim = max(A.ndim, *(idx.ndim for idx in indices))
    A = A[(None,) * (ndim - A.ndim)]

    for idx, dim in zip(indices, dims, strict=True):
        idx = idx[(None,) * (ndim - idx.ndim)]
        A = A.take_along_dim(idx, dim=dim)

    return A.squeeze(dims)


def compose(*funcs: Callable) -> Callable:
    if len(funcs) == 0:
        raise ValueError("compose() requires at least one function.")

    def composed(*args, **kwargs):
        out = funcs[-1](*args, **kwargs)
        for f in funcs[-2::-1]:
            out = f(out)
        return out

    return composed


def tree_map(
    node: dict | list, func: Callable[[dict], Any], exclude: Collection[str] = ()
) -> Any:
    out = []
    for k, v in enumerate(node) if isinstance(node, list) else node.items():
        if k not in exclude and isinstance(v, (dict, list)):
            logger.debug(f"Recursing into {k=}, {v=}.")
            v = tree_map(v, func, exclude=exclude)
        out.append(v)

    if isinstance(node, dict):
        out = func(dict(zip(node.keys(), out)))
    return out


def nonoverlapping_partition(
    centers: Sequence[int | float], window: int | float
) -> list[list[int | float]]:
    """Compute a partition of centers into subsequences without overlaps.

    Args:
        centers: The centers of the windows, assumed to be sorted.
        window: The width of the windows.

    Returns:
        A partition of centers into subsequences without overlaps.

    Examples:
        >>> nonoverlapping_partition([1, 3, 5], 2)
        [[1, 3, 5]]

        >>> nonoverlapping_partition([1, 3, 5], 3)
        [[1, 5], [3]]

        >>> nonoverlapping_partition([1, 2, 3, 4, 5], 3)
        [[1, 4], [2, 5], [3]]

    """
    if len(centers) == 0:
        return []

    if any(starmap(operator.gt, pairwise(centers))):
        raise ValueError("centers must be sorted.")

    partition = [[centers[0]]]
    for center in centers[1:]:
        for p in partition:
            if center - p[-1] >= window:
                p.append(center)
                break
        else:
            partition.append([center])

    return partition


def rolling(df, column, centers, window):
    """Apply a rolling window to a dataframe column.

    Args:
        df: The dataframe.
        column: The column to apply the rolling window to.
        centers: The centers of the windows.
        window: The width of the windows.

    Returns:
        The dataframe with the rolling window applied.

    Examples:
        >>> df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})
        >>> rolling(df, "a", [1, 3, 5], 3)
                     a  b
        0  [-0.5, 2.5)  a
        1  [-0.5, 2.5)  b
        2          NaN  c
        3   [3.5, 6.5)  d
        4   [3.5, 6.5)  e
        0          NaN  a
        1   [1.5, 4.5)  b
        2   [1.5, 4.5)  c
        3   [1.5, 4.5)  d
        4          NaN  e

    """
    partition = nonoverlapping_partition(centers, window)
    rolled = []
    logger.debug(f"{len(partition)=}")

    for p in partition:
        logger.debug(f"p:\n{p}")
        logger.debug(f"df[column]:\n{df[column]}")
        p = np.array(p)
        # Ideally we would use the code below, but there is a severe performance bug:
        # https://github.com/pandas-dev/pandas/issues/47614
        # bins = pd.IntervalIndex.from_arrays(
        #     p - window / 2, p + window / 2, closed="left"
        # )
        # rolled_i = pd.cut(df[column], bins=bins)
        # Fortunately the workaround is quite simple.
        bins = chain.from_iterable(zip(p - window / 2, p + window / 2))
        rolled_i = pd.cut(df[column], bins=bins, right=False)
        rolled_i = rolled_i.cat.remove_categories(rolled_i.cat.categories[1::2])
        rolled.append(rolled_i)
    rolled = pd.api.types.union_categoricals(rolled, ignore_order=True)
    df = pd.concat([df] * len(partition))
    df[column] = rolled
    return df


def concat(
    dfs: Sequence[DataFrame] | Mapping[Hashable, DataFrame], **kwargs
) -> DataFrame:
    dfs_ = dfs.values() if isinstance(dfs, Mapping) else dfs
    cat_cols = set.intersection(
        *[set(k for k, v in df.dtypes.items() if v.name == "category") for df in dfs_]
    )
    for col in cat_cols:
        cats = list(dict.fromkeys(chain(*[df[col].cat.categories for df in dfs_])))
        for df in dfs_:
            df[col] = df[col].cat.set_categories(cats)

    return pd.concat(dfs, **kwargs)


def is_interval_dtype(dtype):
    return isinstance(dtype, pd.IntervalDtype) or (
        isinstance(dtype, pd.CategoricalDtype)
        and isinstance(dtype.categories, pd.IntervalIndex)
    )


def get_interval_mid(series: Series) -> Series:
    dtype = series.dtype
    if isinstance(dtype, pd.IntervalDtype):
        return series.array.mid

    if isinstance(dtype, pd.CategoricalDtype) and isinstance(
        dtype.categories, pd.IntervalIndex
    ):
        return series.cat.rename_categories(series.cat.categories.mid)

    raise ValueError("series must be interval-like, but got {series=}.")
