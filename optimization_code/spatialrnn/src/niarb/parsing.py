import itertools
import logging
from functools import partial
import copy
import ast
from collections.abc import Sequence
from typing import Any

import numpy as np

from niarb.utils import tree_map

logger = logging.getLogger(__name__)


def eval_hook(d, locals):
    if "__eval__" in d:
        logger.debug(f"{d=}, {locals=}")
        out = eval(d.pop("__eval__"), locals.copy())  # note: eval mutates globals
        if len(d) > 0:
            raise ValueError(f"Unknown keys: {d.keys()}")
    else:
        out = d
    return out


def matrix(
    d: dict[str, Sequence] | Sequence[dict[str, Sequence]],
    include: Sequence[dict[str, Any]] = (),
    ignore: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Constructs a list of dictionaries.

    Construct a list of dictionaries using the same logic as
    github action matrix strategy configuation
    https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs,
    except that here we also allow for a list of dictionaries to be passed in.

    Args:
        d: A dictionary of sequences of values, or a sequence of dictionaries of sequences.
        include (optional): Sequence of dictionaries that modify/extend the output.
        ignore (optional): Sequence of keys to exclude from the output.

    Returns:
        A list of dictionaries.

    """
    if isinstance(d, dict):
        d = [d]

    rows = []
    for di in d:
        rows += [dict(zip(di.keys(), row)) for row in itertools.product(*di.values())]
    new_rows = copy.deepcopy(rows)

    for di in include:
        updated = False
        for new_row, row in zip(new_rows, rows):
            if any(k in row and row[k] != v for k, v in di.items()):
                continue

            new_row.update(tree_map(di, partial(eval_hook, locals=row)))
            updated = True

        if not updated:
            new_rows.append(di)

    for row in new_rows:
        for k in ignore:
            row.pop(k, None)

    logger.debug(new_rows)
    return new_rows


def array(string: str) -> np.ndarray:
    """Convert a string representation of an array into a numpy integer ndarray.

    Args:
        string: The string representation of the array.

    Returns:
        The numpy ndarray created from the string.

    Examples:
        >>> array("2,1:5,3")
        array([2, 1, 2, 3, 4, 3])

    """
    arr = []
    for ss in string.split(","):
        if ":" in ss:
            arr.append(slice(*map(ast.literal_eval, ss.split(":"))))
        else:
            arr.append(ast.literal_eval(ss))
    return np.r_[tuple(arr)]


def indices(string: str) -> list[int]:
    """Parse a string into a list of non-negative indices.

    The input string should be a comma-separated list of integers or ranges of integers
    separated by '-' (e.g. "2-5"). The ranges are start- and end-inclusive.

    Args:
        string: The input string.

    Returns:
        A list of indices parsed from the input string.

    Raises:
        ValueError: If the input string does not follow the expected format.

    Examples:
        >>> indices("1,3,5-7")
        [1, 3, 5, 6, 7]

    """
    if string == "":
        return []

    indices = []
    for s in string.split(","):
        ss = list(s.split("-"))
        if len(ss) == 1:
            indices.append(int(ss[0]))
        elif len(ss) == 2:
            indices += list(range(int(ss[0]), int(ss[1]) + 1))
        else:
            raise ValueError(
                "Each comma-separated substring must be either an integer or two"
                f"integers separated by '-', but got {ss}."
            )
    return indices
