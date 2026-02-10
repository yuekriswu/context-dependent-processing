import logging
import functools
import importlib
import inspect
import math
from collections.abc import Sequence, Callable
from itertools import accumulate

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.legend import Legend
from matplotlib.offsetbox import AnchoredText
import pandas as pd
from pandas import DataFrame
from scipy import stats
import seaborn as sns
from seaborn import FacetGrid
import statsmodels.api as sm

from niarb import utils

logger = logging.getLogger(__name__)


def mapped(func, mapping):
    @functools.wraps(func)
    def wrapper(data=None, **kwargs):
        keys = {"x", "y", "hue", "col", "row", "style"}
        data = data.rename(columns=mapping)
        kwargs = {
            k: mapping[v] if (k in keys) and (v in mapping) else v
            for k, v in kwargs.items()
        }
        return func(data, **kwargs)

    return wrapper


def figplot(
    data: DataFrame,
    func: Callable[[DataFrame], FacetGrid] | str | Sequence[str],
    *,
    stat: bool = False,
    stat_kws: dict | None = None,
    errordim: str | Sequence[str] | None = None,
    grid: str | None = None,
    mapping: dict[str, str] | None = None,
    tight_layout: bool = True,
    xlim: tuple[float | None, float | None] = (None, None),
    ylim: tuple[float | None, float | None] = (None, None),
    legend_loc: str | int | None = None,
    legend_title: bool = True,
    legend_kwargs: dict | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    **kwargs,
) -> FacetGrid:
    if mapping is None:
        mapping = {}

    if isinstance(func, str):
        func = {"relplot": relplot, "lmplot": lmplot, "displot": displot}.get(
            func, getattr(sns, func)
        )
    elif isinstance(func, Sequence):
        if len(func) != 2:
            raise ValueError("func must be a sequence of length 2.")
        func = getattr(importlib.import_module(func[0]), func[1])

    # some data cleaning:
    # 1. remove possibly unused categories
    for k, v in data.dtypes.items():
        if isinstance(v, pd.CategoricalDtype):
            data[k] = data[k].cat.remove_unused_categories()

    # 2. ensure unique index
    data = data.reset_index(drop=True)

    if errordim:
        if "y" not in kwargs:
            raise ValueError("errordim requires 'y' to be specified.")

        if isinstance(errordim, str):
            errordim = [errordim]

        estimator = kwargs.get("estimator", "mean")
        by = [
            v
            for k, v in kwargs.items()
            if k in {"x", "hue", "col", "row", "style"} and v is not None
        ]
        logger.debug(f"{by=}, {errordim=}, {estimator=}")

        data = data.groupby(
            by + list(errordim), observed=True, as_index=False, dropna=False
        )[kwargs["y"]].agg(estimator)

    logger.debug(f"data:\n{data}")

    g = mapped(func, mapping)(data, **kwargs)
    if stat:
        if "x" not in kwargs or "y" not in kwargs:
            raise ValueError("stat=True requires 'x' and 'y' to be specified.")

        statplot = {sns.catplot: catstatplot, lmplot: lmstatplot}.get(func)
        if statplot is None:
            raise ValueError(f"stat=True is not supported for {func}.")
        g.map_dataframe(statplot, x=kwargs["x"], y=kwargs["y"], **(stat_kws or {}))

    # more compact subplot titles
    if isinstance(g, sns.FacetGrid):
        row_temp = "{row_var}: {row_name}"
        col_temp = "{col_var}: {col_name}"
        temp = (
            f"{row_temp}\n{col_temp}" if "row" in kwargs and "col" in kwargs else None
        )
        g.set_titles(template=temp, row_template=row_temp, col_template=col_temp)

    # set xlim, ylim, xscale, yscale
    if (
        xlim != (None, None)
        or ylim != (None, None)
        or xscale != "linear"
        or yscale != "linear"
    ):
        g.set(xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale)

    # add gridlines, with options for adding x = 0 or y = 0 lines.
    if grid is not None:
        color, linewidth = rcParams["grid.color"], rcParams["grid.linewidth"]
        for ax in g.axes.flat:
            if grid in ["yzero", "xyzero"]:
                ymin, ymax = ax.get_ylim()
                if ymin < 0 < ymax:  # only add line if 0 is within the limits
                    ax.axhline(0, color=color, linewidth=linewidth)
            if grid in ["xzero", "xyzero"]:
                xmin, xmax = ax.get_xlim()
                if xmin < 0 < xmax:  # only add line if 0 is within the limits
                    ax.axvline(0, color=color, linewidth=linewidth)
            if grid not in ["xzero", "yzero", "xyzero"]:
                ax.grid(axis=grid)

    # format legend
    if g.legend:
        if legend_loc is not None:
            sns.move_legend(g, legend_loc, **(legend_kwargs or {}))
        if not legend_title:
            g.legend.set_title(None)

    # call tight_layout
    if tight_layout:
        g.tight_layout()

    return g


def relplot(data=None, *, x=None, y=None, **kwargs):
    if utils.is_interval_dtype(data[x].dtype):
        data = data.copy()
        data[x] = utils.get_interval_mid(data[x])
    logger.debug(f"data:\n{data}")
    logger.debug(f"data memory usage:\n{data.memory_usage()}")

    # for some reason seaborn is very memory-inefficient, so manually do groupby
    # if errorbar is "se", "sd", or None. This is important for plotting large
    # dataframes such as when plotting weights.
    if "errorbar" in kwargs and kwargs["errorbar"] in {"se", "sd", None}:
        errorbar = kwargs["errorbar"]
        by = [x] + [
            v
            for k, v in kwargs.items()
            if k in {"x", "hue", "col", "row", "style"} and v is not None
        ]
        agg = {y: "mean"}
        if errorbar == "se":
            agg[f"{y}_{errorbar}"] = "sem"
        elif errorbar == "sd":
            agg[f"{y}_{errorbar}"] = "std"
        data = data.groupby(by, observed=True, as_index=False)[y].agg(**agg)
        if errorbar:
            data = sample_df(data, errorbar=errorbar, y=y, yerr=f"{y}_{errorbar}")
    logger.debug(f"grouped data:\n{data}")

    return sns.relplot(data=data, x=x, y=y, **kwargs)


def lmplot(data=None, *, x=None, **kwargs):
    if utils.is_interval_dtype(data[x].dtype):
        data = data.copy()
        data[x] = utils.get_interval_mid(data[x])
    logger.debug(f"data:\n{data}")
    logger.debug(f"data memory usage:\n{data.memory_usage()}")

    return sns.lmplot(data=data, x=x, **kwargs)


def lmstatplot(
    data=None,
    *,
    x=None,
    y=None,
    loc="upper right",
    alpha=0.5,
    color=None,
    label=None,
    marker=None,
    **kwargs,
):
    fit = sm.OLS(data[y], sm.add_constant(data[x])).fit()
    stats = [
        f"Slope: {fit.params[1]:.1e}$\pm${fit.bse[1]:.1e}",
        f"Intercept: {fit.params[0]:.1e}$\pm${fit.bse[0]:.1e}",
        f"$R^2$: {fit.rsquared:.2g}, P-value: {fit.pvalues[1]:.1e}",
    ]
    text = AnchoredText("\n".join(stats), loc, **kwargs)
    text.patch.set_alpha(alpha)
    plt.gca().add_artist(text)
    return text


def displot(data=None, *, kind="hist", legend=True, **kwargs):
    kind = histplot if kind == "hist" else getattr(sns, f"{kind}plot")

    keys = set(inspect.signature(FacetGrid).parameters.keys())
    facet_kws = {k: kwargs.pop(k) for k in set(kwargs.keys()) if k in keys}
    facet_kws |= kwargs.pop("facet_kws", {})

    g = sns.FacetGrid(data, **facet_kws)
    g.map_dataframe(kind, **kwargs)
    if legend:
        g.add_legend()

    return g


def histplot(data=None, *, x=None, color=None, label=None, bins="auto", **kwargs):
    # sensible handling of data consisting of a single unique value
    if data[x].nunique() == 1:
        return plt.vlines(
            data[x].unique().item(), 0, len(data), color=color, label=label
        )

    if isinstance(bins, Sequence) and not isinstance(bins, str):
        if len(bins) < 1:
            raise ValueError(
                "If bins is a sequence, it must have at least one element, but "
                f"{len(bins)=}."
            )

        name, *args = bins
        if name == "zero":
            bins = histogram_bin_edges(data[x].min(), data[x].max(), *args)
        else:
            raise ValueError(f"Invalid binning scheme: {name}.")

    return sns.histplot(data, x=x, color=color, label=label, bins=bins, **kwargs)


def catstatplot(
    data: DataFrame | None = None,
    *,
    x: str | None = None,
    y: str | None = None,
    kind: str = "ind",
    test: Callable | str = "ttest_ind",
    test_kws: dict | None = None,
    alphas: Sequence[float] = (0.05, 0.01, 0.001),
    ax: Axes | None = None,
    ha: str = "center",
    **kwargs,
) -> Text:
    if kind not in {"ind", "1samp"}:
        raise ValueError(f"'kind' must be either 'ind' or '1samp', but got {kind}.")

    if test_kws is None:
        test_kws = {}

    if isinstance(test, str):
        test = getattr(stats, test)

    xs, samples = zip(*data.groupby(x)[y])
    logger.debug(f"xs:\n{xs}")
    logger.debug("samples:\n%s", "\n".join(str(s.tolist()) for s in samples))
    try:
        if kind == "ind":
            pvalues = [test(*samples, **test_kws).pvalue]
        else:
            pvalues = [test(s, **test_kws).pvalue for s in samples]
    except ValueError as err:
        logger.error(str(err))
        return

    logger.debug(f"{pvalues=}")

    texts = []
    for pvalue in pvalues:
        if pvalue >= alphas[0]:
            continue

        if np.isnan(pvalue):
            logger.warning("p-value is NaN.")
            continue

        if pvalue >= alphas[1]:
            text = "*"
        elif pvalue >= alphas[2]:
            text = "**"
        else:
            text = "***"
        texts.append(text)

    if not pd.api.types.is_numeric_dtype(data[x]):
        xs = range(len(xs))
    ys = (s.mean() for s in samples)

    if kind == "ind":
        xs = [sum(xs) / len(xs)]
        ys = [max(ys)]

    if ax is None:
        ax = plt.gca()

    objs = [ax.text(x, y, text, ha=ha, **kwargs) for x, y, text in zip(xs, ys, texts)]
    return objs


def histogram_bin_edges(min, max, bins):
    """
    Returns equally spaced histogram bin edges where
    0 is one of the edges if min < 0 < max, otherwise
    it just returns np.linspace(min, max, num=bins + 1)
    """
    if min >= max:
        raise ValueError(f"min must be smaller than max, but {min=}, {max=}.")

    if min >= 0 or max <= 0:
        return np.linspace(min, max, num=bins + 1)

    if not isinstance(bins, int) or bins < 2:
        raise ValueError(f"bins must be an integer that is at least 2, but {bins=}.")

    binwidth = (max - min) / bins
    amin = abs(min)
    N_pos = round(max / binwidth)
    N_neg = round(amin / binwidth)
    if math.isclose(amin, binwidth * N_neg) and math.isclose(max, binwidth * N_pos):
        return np.linspace(min, max, num=bins + 1)

    binwidth = (max - min) / (bins - 1)
    N_pos = math.ceil(max / binwidth)
    N_neg = bins - N_pos
    return np.linspace(-binwidth * N_neg, binwidth * N_pos, num=bins + 1)


def remove_legend_subtitles(ax: Axes, nums: Sequence[int], **kwargs) -> Legend:
    handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == len(labels)

    if sum(nums) + len(nums) != len(handles):
        raise ValueError(
            f"sum(nums) + len(nums) must equal the number of handles, but "
            f"{sum(nums) + len(nums)=}, {len(handles)=}."
        )

    cumnums = {0} | set(accumulate(n + 1 for n in nums))
    indices = [i for i in range(len(handles)) if i not in cumnums]
    return ax.legend(
        [handles[i] for i in indices], [labels[i] for i in indices], **kwargs
    )


def sample_df(
    df: DataFrame,
    estimator: str = "mean",
    errorbar: str | tuple[str, int] = "se",
    y: str = "y",
    yerr: str | tuple[str, str] = "yerr",
    index: str | None = None,
) -> DataFrame:
    """Generate 'samples' of dataframe

    This can be applied to dataframes with precomputed errorbars
    so that those errorbars can be plotted in seaborn.

    Args:
        df: Dataframe
        estimator (optional): {"mean", "median"}. Estimator for the target variable.
        errorbar (optional): {"se", "sd", ("pi", 100)}. Errorbar type. If ("pi", 100),
          estimator must be "median".
        y (optional): Target variable
        yerr (optional): Errorbar variable. If a tuple, the first element is the
          lower errorbar and the second element is the upper errorbar.
        index (optional): If not None, create a new column with this name containing
          the indices of the samples.

    Returns:
        Dataframe containing 'samples' of the original dataframe

    """
    df0, df1 = df.copy(), df.copy()
    if errorbar in {"se", "sd"}:
        if not isinstance(yerr, str):
            raise ValueError(
                f"yerr must be a string if errorbar is 'se' or 'sd', but {yerr=}."
            )
        scaling = {"se": 3**0.5, "sd": 1}[errorbar]
        df0[y] = df[y] - df[yerr] * scaling
        df1[y] = df[y] + df[yerr] * scaling
    else:
        if not isinstance(yerr, tuple) or len(yerr) != 2:
            raise ValueError(
                f"yerr must be a 2-tuple if errorbar == ('pi', 100), but {yerr=}."
            )
        if estimator != "median":
            raise ValueError(
                "estimator must be 'median' if errorbar == ('pi', 100), but "
                f"{estimator=}."
            )
        df0[y] = df[yerr[0]]
        df1[y] = df[yerr[1]]
        yerr = list(yerr)

    out = pd.concat(dict(enumerate([df, df0, df1]))).drop(columns=yerr)

    if index is not None:
        out = out.reset_index(0, names=index)

    return out.reset_index(drop=True)
