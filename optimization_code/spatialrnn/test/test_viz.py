import pandas as pd
import numpy as np
import pytest

from niarb import viz


@pytest.mark.parametrize(
    "min, max, bins, expected",
    [
        (-0.1, 0.8, 4, [-0.3, 0.0, 0.3, 0.6, 0.9]),
        (-0.4, 0.5, 4, [-0.6, -0.3, 0.0, 0.3, 0.6]),
        (-0.8, 0.1, 4, [-0.9, -0.6, -0.3, 0.0, 0.3]),
        (-0.5, 0.5, 4, [-0.5, -0.25, 0.0, 0.25, 0.5]),
        (0.1, 1.1, 4, [0.1, 0.35, 0.6, 0.85, 1.1]),
    ],
)
def test_histogram_bin_edges(min, max, bins, expected):
    out = viz.histogram_bin_edges(min, max, bins)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("estimator", ["mean", "median"])
@pytest.mark.parametrize("errorbar", ["se", "sd", ("pi", 100)])
def test_sample_df(estimator, errorbar):
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [0.5, 1.0, 1.5],
            "yerr": [0.1, 0.2, 0.3],
            "ylow": [0.4, 0.8, 1.2],
            "yhigh": [0.8, 1.4, 2.0],
        }
    )
    yerr = "yerr" if errorbar in {"se", "sd"} else ("ylow", "yhigh")

    if estimator == "mean" and errorbar == ("pi", 100):
        with pytest.raises(ValueError):
            viz.sample_df(df, yerr=yerr, estimator=estimator, errorbar=errorbar)
        return

    out = viz.sample_df(df, yerr=yerr, estimator=estimator, errorbar=errorbar)

    if errorbar in {"se", "sd"}:
        errorbar = {"se": "sem", "sd": "std"}[errorbar]
        out = out.groupby("x", observed=True, as_index=False)["y"].agg(
            y=estimator, yerr=errorbar
        )
        pd.testing.assert_frame_equal(out, df[["x", "y", "yerr"]])
    else:
        out = out.groupby("x", observed=True, as_index=False)["y"].agg(
            y=estimator, ylow="min", yhigh="max"
        )
        pd.testing.assert_frame_equal(out, df[["x", "y", "ylow", "yhigh"]])


def test_sample_df_index():
    df = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [0.5, 1.0],
            "yerr": [0.1, 0.2],
        }
    )
    out = viz.sample_df(df, errorbar="sd", index="idx")
    expected = pd.DataFrame(
        {
            "idx": [0, 0, 1, 1, 2, 2],
            "x": [1, 2, 1, 2, 1, 2],
            "y": [0.5, 1.0, 0.4, 0.8, 0.6, 1.2],
        }
    )
    pd.testing.assert_frame_equal(out, expected)
