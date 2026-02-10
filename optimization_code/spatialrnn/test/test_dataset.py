import hyclib as lib

import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader

from niarb.nn import functional as F
from niarb.nn.modules import frame
from niarb import dataset, neurons, random, perturbation
from niarb.tensors import categorical


def test_get_tags_and_configs():
    configs = [
        {"N": 1, "cell_probs": {"PYR": 1.0}, "repeats": 2},
        {"N": 10, "cell_probs": {"PV": 1.0}, "density": "compact"},
        {"N": 10, "cell_probs": {"PV": 1.0}, "density": "spreadout"},
    ]
    mapping = {
        "compact": {"space": ("uniform", 100.0)},
        "spreadout": {"space": ("uniform", 200.0)},
    }
    out = dataset._get_tags_and_configs(configs, mapping)
    expected = (
        frame.ParameterFrame(
            {
                "N": torch.tensor([1, 1, 10, 10]),
                "density": categorical.tensor(
                    [0, 0, 1, 2], categories=(None, "compact", "spreadout")
                ),
            }
        ),
        [
            {"N": 1, "cell_probs": {"PYR": 1.0}},
            {"N": 1, "cell_probs": {"PYR": 1.0}},
            {"N": 10, "cell_probs": {"PV": 1.0}, "space": ("uniform", 100.0)},
            {"N": 10, "cell_probs": {"PV": 1.0}, "space": ("uniform", 200.0)},
        ],
    )
    assert all(
        (v1 == v2).all() for _, v1, v2 in lib.itertools.dict_zip(out[0], expected[0])
    )
    assert out[1] == expected[1]


class TestDataset:
    @pytest.fixture
    def dset_kwargs(self):
        return dict(
            neurons={"N": 500, "variables": ["space"], "space_extent": [1000.0]},
            inputs={
                "configs": [
                    {"N": 1, "repeats": 2},
                    {"N": 2, "density": "compact", "affine_space": True},
                    {"N": 2, "density": "spreadout", "affine_space": True},
                ],
                "mapping": {
                    "compact": {"space": ("uniform", 300.0)},
                    "spreadout": {"space": ("uniform", 600.0)},
                },
            },
            data=[
                pd.DataFrame(
                    {
                        "dr": [1.0, 2.0, 3.0],
                        "dr_se": [0.5, 0.5, 0.5],
                        "min_distance_to_ensemble": [0.0, 1.0, 2.0],
                    }
                )
            ],
            N_instantiations=3,
            window=500.0,
            abs_window=800.0,
            subsample=100,
            seed=0,
        )

    @pytest.fixture
    def grid_dset_kwargs(self):
        return dict(
            neurons=neurons.as_grid(n=2, N_space=(10,)),
            inputs={"configs": [{"N": 1, "repeats": 3}]},
            data=[pd.DataFrame({"dr": [1.0, 2.0, 3.0]})],
        )

    @pytest.mark.parametrize("psf", [None, (100.0, False)])
    def test_getitem(self, dset_kwargs, psf):
        dset = dataset.Dataset(**dset_kwargs, psf=psf)
        x, y, kwargs = dset[0]
        assert kwargs == {"model_kwargs": {"check_circulant": False, "ndim": 1}}
        assert (y == torch.tensor([1.0, 2.0, 3.0])).all()
        assert isinstance(x, frame.ParameterFrame)
        assert set(x.keys()) == {
            "space",
            "dV",
            "dh",
            "mask",
            "N",
            "density",
            "affine_space",
            "min_distance_to_ensemble",
        }
        assert x.shape == (4, 500)
        assert x.data["space"].shape == (1, 500, 1)
        assert x.data["dV"].shape == (1, 1)
        assert all(
            x.data[k].shape == (4, 500)
            for k in {"dh", "mask", "min_distance_to_ensemble"}
        )
        assert all(x.data[k].shape == (4, 1) for k in {"N", "density"})
        assert (x.data["N"].squeeze(-1) == torch.tensor([1, 1, 2, 2])).all()
        assert (
            x.data["density"].squeeze(-1)
            == categorical.tensor(
                [0, 0, 1, 2], categories=(None, "compact", "spreadout")
            )
        ).all()

        if psf is None:
            assert not (x["dh"] != 0).all()
            for i in range(x.shape[0]):
                xi = x.iloc[i]
                loc = xi["space"][xi["dh"] != 0].cmean()
                expected = F.diff(xi["space"], loc).norm(dim=-1) <= 250.0
                expected &= xi["space"].norm(dim=-1) <= 400.0
                with random.set_seed(0):
                    expected_ = perturbation.categorical_sample(
                        expected.float(), 100
                    ).bool()
                assert (expected_ & expected == expected_).all()  # check subsample
                assert (xi["mask"] == expected_).all()
        else:
            assert (x["dh"] != 0).all()

    @pytest.mark.parametrize("psf", [None, (10.0, False)])
    def test_getitem_grid(self, grid_dset_kwargs, psf):
        grid_dset = dataset.Dataset(**grid_dset_kwargs, psf=psf)
        x, _, kwargs = grid_dset[0]
        assert x.shape == (3, 2, 10)
        assert x.data["cell_type"].shape == (1, 2, 1)
        assert x.data["space"].shape == (1, 1, 10, 1)
        assert x.data["dh"].shape == (3, 2, 10)
        assert "mask" not in x
        assert kwargs == {"model_kwargs": {"ndim": 2}}

    def test_kwargs(self):
        dset_kwargs = {
            "neurons": {"N": 500, "variables": ["space"]},
            "inputs": {"configs": [{"N": 1}]},
        }

        dset = dataset.Dataset(**dset_kwargs, model_kwargs={"ndim": 2})
        _, kwargs = dset[0]
        assert kwargs == {"model_kwargs": {"ndim": 2, "check_circulant": False}}

        dset = dataset.Dataset(**dset_kwargs, model_kwargs={"output": "weight"})
        _, kwargs = dset[0]
        assert kwargs == {
            "model_kwargs": {"ndim": 1, "check_circulant": False, "output": "weight"}
        }

        dset = dataset.Dataset(**dset_kwargs, analysis_kwargs={"hi": "bye"})
        _, kwargs = dset[0]
        assert kwargs == {
            "model_kwargs": {"ndim": 1, "check_circulant": False},
            "analysis_kwargs": {"hi": "bye"},
        }

    def test_collate_fn(self, dset_kwargs):
        dset = dataset.Dataset(**dset_kwargs)
        dl = DataLoader(dset, batch_size=len(dset), collate_fn=dataset.collate_fn)
        x, y, kwargs = next(iter(dl))
        assert isinstance(x, frame.ParameterFrame)
        assert x.shape == (3, 4, 500)
        assert (y == torch.tensor([1.0, 2.0, 3.0])).all()
        assert kwargs == {"model_kwargs": {"ndim": 1, "check_circulant": False}}

    def test_collate_fn_grid(self, grid_dset_kwargs):
        grid_dset = dataset.Dataset(**grid_dset_kwargs)
        dl = DataLoader(grid_dset, batch_size=1, collate_fn=dataset.collate_fn)
        x, _, kwargs = next(iter(dl))
        assert isinstance(x, frame.ParameterFrame)
        assert x.shape == (1, 3, 2, 10)
        assert x.data["cell_type"].shape == (1, 1, 2, 1)
        assert x.data["space"].shape == (1, 1, 1, 10, 1)
        assert "mask" not in x
        assert kwargs == {"model_kwargs": {"ndim": 2}}

    def test_y(self, dset_kwargs):
        dset = dataset.Dataset(**dset_kwargs)
        dl = DataLoader(dset, batch_size=len(dset), collate_fn=dataset.collate_fn)
        _, y, _ = next(iter(dl))
        assert (y == torch.tensor([1.0, 2.0, 3.0])).all()

        dset.sample_target = True
        dset.reset_targets()
        _, y0, _ = next(iter(dl))
        dset.reset_targets()
        _, y1, _ = next(iter(dl))
        assert (y0 != y1).any() and (y0 != y).any() and (y1 != y).any()
        assert y0.shape == y1.shape == y.shape
        assert y0.dtype == y1.dtype == y.dtype

        dset.target_seed = 0
        dset.reset_targets()
        _, y0, _ = next(iter(dl))
        dset.reset_targets()
        _, y1, _ = next(iter(dl))
        torch.testing.assert_close(y0, y1)
        assert (y0 != y).any()
        assert y0.dtype == y1.dtype == y.dtype

        dl = iter(DataLoader(dset, batch_size=1, collate_fn=dataset.collate_fn))
        dset.target_seed = None
        dset.reset_targets()
        _, y0, _ = next(dl)
        _, y1, _ = next(dl)
        torch.testing.assert_close(y0, y1)
        assert (y0 != y).any()
        assert y0.dtype == y1.dtype == y.dtype
