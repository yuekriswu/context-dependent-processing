import pytest
import pandas as pd
from torch.utils.data import DataLoader
import torch

from niarb import nn
from niarb.dataset import Dataset, collate_fn


class TestPipeline:
    @pytest.fixture
    def data(self):
        data1 = {
            "distance": pd.cut(
                [150.0, 450.0, 700.0], bins=[0.0, 300.0, 600.0, 1000.0], right=False
            ),
            "dr": [4.0, 5.0, 6.0],
            "dr_se": [0.1, 0.2, 0.3],
        }
        data2 = {
            "perturbed_cell_type": ["PV", "PYR"],
            "dr": [1.0, 2.0],
        }
        return [pd.DataFrame(data1), pd.DataFrame(data2)]

    @pytest.fixture
    def dataset(self, data):
        neuron = {
            "N": 100,
            "variables": ["cell_type", "space"],
            "cell_types": ["PYR", "PV"],
            "space_extent": [1000.0],
        }
        inputs = {
            "configs": [
                {"N": 1, "perturbed_cell_type": "PYR"},
                {"N": 1, "perturbed_cell_type": "PV"},
            ],
            "mapping": {
                "PYR": {"cell_probs": {"PYR": 1.0}},
                "PV": {"cell_probs": {"PV": 1.0}},
            },
        }
        metrics = {"distance": "min_distance_to_ensemble"}
        return Dataset(neuron, inputs, data=data, metrics=metrics, seed=0)

    @pytest.mark.parametrize("estimator", ["mean", "median"])
    def test_forward(self, data, dataset, estimator):
        pipeline = nn.Pipeline(
            model=nn.V1(["cell_type", "space"]), data=data, estimator=estimator
        )
        assert list(pipeline.analysis.modules[0].x.columns) == ["distance"]
        assert list(pipeline.analysis.modules[1].x.columns) == ["perturbed_cell_type"]
        assert pipeline.analysis.modules[0].estimator == estimator
        assert pipeline.analysis.modules[0].estimator == estimator
        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        x, y, kwargs = next(iter(dataloader))
        y_pred = pipeline(x, **kwargs)
        assert y_pred.shape == y.shape == (5,)
        assert y_pred.grad_fn is not None

    @pytest.mark.parametrize("f", ["Identity", "SSN", "Ricciardi", "Match"])
    def test_scale_parameters(self, data, dataset, f):
        if f == "Match":
            f = nn.Match({"PV": nn.Ricciardi(tau=0.01)}, nn.Ricciardi())
        pipeline = nn.Pipeline(
            model=nn.V1(
                ["cell_type", "space"],
                cell_types=["PYR", "PV"],
                f=f,
                # nonlinear_kwargs={"assert_convergence": False},
                mode="numerical",
            ),
            data=data,
        )
        pipeline.load_state_dict(
            {"model.gW": torch.tensor([[0.01, 0.05], [0.075, 0.01]])}, strict=False
        )
        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        x, _, kwargs = next(iter(dataloader))
        y_pred = pipeline(x, **kwargs)

        pipeline.scale_parameters(2.0)

        new_y_pred = pipeline(x, **kwargs)

        assert not y_pred.isnan().all()
        torch.testing.assert_close(new_y_pred, 2.0 * y_pred, equal_nan=True)
