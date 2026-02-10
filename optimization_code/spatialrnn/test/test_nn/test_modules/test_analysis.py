import pytest
import torch
import pandas as pd
import hyclib as lib
import tdfl

from niarb import nn
from niarb.tensors import categorical


@pytest.fixture
def x():
    shape = (2000,)
    with lib.random.set_seed(0):
        return tdfl.DataFrame(
            {
                "holo_id": torch.randint(0, 4, shape),
                "cell_type": categorical.as_tensor(
                    torch.randint(0, 3, shape), categories=["PYR", "PV", "SST"]
                ),
                "distance": torch.rand(shape) * 1000,
                "ens_size": torch.randint(0, 3, shape) * 10,
                "holo_osi": torch.randint(0, 2, shape),
                "dr": torch.randn(shape, requires_grad=True) * 2,
            }
        )


@pytest.fixture
def df():
    # fmt: off
    df = {
        "cell_type": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # noqa
        "holo_osi": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa
        "ens_size": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],  # noqa
        "distance": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],  # noqa
    }
    # fmt: on
    df = pd.DataFrame(df)
    df["cell_type"] = pd.Categorical.from_codes(
        df["cell_type"], categories=["PYR", "PV", "SST"]
    )
    df["ens_size"] = pd.Categorical.from_codes(
        df["ens_size"],
        categories=pd.IntervalIndex.from_breaks([0, 10, 30, 50], closed="left"),
    )
    df["distance"] = pd.Categorical.from_codes(
        df["distance"],
        categories=pd.IntervalIndex.from_breaks(
            [30, 60, 90, 120, 150, 180], closed="left"
        ),
    )
    return df


class TestTensorDataFrameAnalysis:
    @pytest.mark.parametrize("query_x", [None, 'cell_type != "SST"'])
    @pytest.mark.parametrize("query_df", [None, 'cell_type != "SST"'])
    @pytest.mark.parametrize("estimator", ["mean", "median"])
    @pytest.mark.parametrize("sem", [None, "holo_id"])
    def test_forward(self, x, df, query_x, query_df, estimator, sem):
        if query_df:
            df = df.query(query_df)

        model = nn.TensorDataFrameAnalysis(
            x=df, y="dr", query=query_x, estimator=estimator, sem=sem
        )
        out = model(x)
        if sem:
            out, out_sem = out["dr"], out["dr_se"]
        else:
            out = out["dr"]

        # check that we didn't modify the input
        assert not isinstance(x["distance"], categorical.CategoricalTensor)

        # calculate expected dataframe
        if query_x:
            x = x.query(query_x)
        x = x.to_pandas()
        x["ens_size"] = pd.cut(x["ens_size"], bins=df["ens_size"].cat.categories)
        x["distance"] = pd.cut(x["distance"], bins=df["distance"].cat.categories)
        by = list(df.columns) + ([sem] if sem else [])
        expected = x.groupby(by, as_index=False, observed=True)["dr"].agg(estimator)
        if sem:
            expected = expected.groupby(list(df.columns), as_index=False, observed=True)
            expected = expected["dr"].agg(dr=estimator, dr_se="sem")
            if estimator == "median":
                expected["dr_se"] = expected["dr_se"] * (torch.pi / 2) ** 0.5
        expected = expected.merge(df, how="right")
        if sem:
            expected, expected_sem = expected["dr"], expected["dr_se"]
            expected = torch.from_numpy(expected.to_numpy())
            expected_sem = torch.from_numpy(expected_sem.to_numpy())
        else:
            expected = torch.from_numpy(expected["dr"].to_numpy())

        # check that gradient successfully passes through groupby operations
        assert out.grad_fn is not None
        torch.testing.assert_close(out, expected, equal_nan=True)
        if sem:
            torch.testing.assert_close(out_sem, expected_sem, equal_nan=True)


class TestEigvals:
    def test_forward(self):
        W = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0 + 1.0j, 0.0], [0.0, 0.0, 3.0 - 2.0j]]
        )
        df = nn.Eigvals()(W).to_pandas()
        expected = pd.DataFrame(
            {"real": [1.0, 2.0, 3.0], "imag": [0.0, 1.0, -2.0]}
        ).astype("float32")
        pd.testing.assert_frame_equal(df, expected)
