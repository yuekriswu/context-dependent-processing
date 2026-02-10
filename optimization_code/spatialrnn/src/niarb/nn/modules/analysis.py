import logging
from collections.abc import Iterable, Sequence
from itertools import chain

import torch
import pandas as pd
import tdfl

from niarb.tensors import categorical

logger = logging.getLogger(__name__)

__all__ = ["TensorDataFrameAnalysis", "Eigvals"]


class DataFrameAnalysis(torch.nn.Module):
    def __init__(
        self,
        x: str | Sequence[str] | pd.DataFrame,
        y: str | Sequence[str],
        query: str | None = None,
        evals: dict[str, str] | None = None,
        cuts: dict[str, Sequence[int | float]] | None = None,
        estimator: str = "mean",
        sem: str | Iterable[str] | None = None,
    ):
        r"""Perform groupby aggregations on a DataFrame.

        Args:
            x: Columns on which data is grouped on. If a DataFrame, groupby results
              are merged with `x`.
            y: Column(s) to which the estimator is applied.
            query (optional): Query to filter data.
            evals (optional): Dictionary of columns to evaluate.
            cuts (optional): Dictionary of columns to cut into bins. Must be None
              if `x` is a DataFrame.
            estimator (optional): {"mean", "median"}. Estimator to use for calculating
              statistics. If "median" and `sem` is not None, the standard error of the
              median is calculated by multiplying the standard error of the mean by
              $\sqrt{\pi/2}$ (normality assumption).
            sem (optional): If not None, computes SEM across specified columns.

        """
        if isinstance(x, pd.DataFrame):
            if len(x.drop_duplicates()) != len(x):
                raise ValueError("x must consist of unique rows.")
            if cuts is not None:
                raise ValueError("cuts must be None if x is a DataFrame.")

        if estimator not in {"mean", "median"}:
            raise ValueError(f"Unknown estimator: {estimator=}.")

        super().__init__()

        if evals is None:
            evals = {}

        if cuts is None:
            cuts = {}

        if isinstance(x, str):
            x = [x]
        elif isinstance(x, pd.DataFrame):
            x = x.copy()
            for k, v in x.items():
                if v.dtype.name == "category" and isinstance(
                    v.cat.categories, pd.IntervalIndex
                ):
                    cuts[k] = v.cat.categories
                elif v.dtype.name == "interval":
                    cuts[k] = pd.IntervalIndex(v)

        if isinstance(y, str):
            y = [y]

        groupby = list(x.columns if isinstance(x, pd.DataFrame) else x)

        if isinstance(sem, str):
            sem = [sem]
        if isinstance(sem, Iterable):
            sem = list(sem)
        if sem is not None and any(k in groupby for k in sem):
            raise ValueError(f"{sem=} must not be in {groupby=}.")

        self.x = x
        self.y = y
        self.groupby = groupby
        self.query = query
        self.evals = evals
        self.cuts = cuts
        self.estimator = estimator
        self.sem = sem


class TensorDataFrameAnalysis(DataFrameAnalysis):
    def forward(self, df: tdfl.DataFrame) -> tdfl.DataFrame:
        if self.query:
            df = df.query(self.query)

        df = df.copy()

        for k, v in self.evals.items():
            df[k] = df.eval(v)

        # Cut the interval columns for grouping
        for k, v in self.cuts.items():
            # Note: need to convert to float due to lib.pt._bin_numbers bug with Long input.
            # labels=False with missing_rep=-1 yields the codes of the categories.
            df[k] = tdfl.cut(
                df[k].float(), v, labels=False, right=False, missing_rep=-1
            )
            df[k] = categorical.as_tensor(df[k], categories=list(v) + [torch.nan])

        # Filter out all invalid groups first, which would make groupby faster
        df = df[:, (torch.stack([df[k] for k in self.groupby]) != -1).all(dim=0)]

        # Perform groupby operations
        agg = {k: (k, self.estimator) for k in self.y}
        if self.sem is not None:
            df = df.groupby(self.groupby + self.sem).agg(**agg)
            agg = ([(k, (k, self.estimator)), (f"{k}_se", (k, "sem"))] for k in self.y)
            out = df.groupby(self.groupby).agg(**dict(chain.from_iterable(agg)))
            if self.estimator == "median":
                for k in self.y:
                    out[f"{k}_se"] = out[f"{k}_se"] * (torch.pi / 2) ** 0.5
        else:
            # Note: df.groupby() is likely a bottleneck due to the slow torch.unqiue call
            out = df.groupby(self.groupby).agg(**agg)

        if isinstance(self.x, pd.DataFrame):
            # Convert CategoricalTensors to numpy arrays before merging
            for k, v in out.items():
                if isinstance(v, categorical.CategoricalTensor):
                    out[k] = v.detach().cpu().numpy()

            out = out.merge(self.x, how="right")

        for k in self.y:
            if out[k].isnan().any():
                logger.warning(
                    f"NaNs detected in analysis output\n:{out[:, out[k].isnan()]}"
                )
        return out


class Eigvals(torch.nn.Module):
    def forward(self, W: torch.Tensor) -> tdfl.DataFrame:
        logger.info(f"Calculating eigenvalues of W with shape {W.shape}...")
        eigvals = torch.linalg.eigvals(W)
        return tdfl.DataFrame(
            {"real": eigvals.real.flatten(), "imag": eigvals.imag.flatten()}
        )
