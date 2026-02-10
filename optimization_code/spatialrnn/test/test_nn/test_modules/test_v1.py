from itertools import product
from collections.abc import Sequence
import math
from math import exp
import copy

import pytest
import torch
from torch.utils.data import DataLoader
import pandas as pd
import hyclib as lib

from niarb import neurons, special, nn, utils, perturbation, random
from niarb.nn.modules.v1 import compute_osi_scale, _cdim
from niarb.cell_type import CellType
from niarb.nn import functional
from niarb.nn.modules import frame
from niarb.tensors import periodic
from niarb.tensors.circulant import CirculantTensor
from niarb.dataset import Dataset, collate_fn


def spatial_component(d, s, r):
    if d == 1:
        return torch.exp(-r / s) / (2 * s)
    elif d == 2:
        return special.k0(r / s) / (2 * torch.pi * s**2)
    elif d == 3:
        return torch.exp(-r / s) / r / (4 * torch.pi * s**2)
    else:
        raise NotImplementedError()


def get_parameters(**kwargs):
    parameters = []
    for p in product(*kwargs.values()):
        p = dict(zip(kwargs.keys(), p))

        if "cell_type" not in p.get("variables", []):
            if "subcircuit_cell_types" in p and p["subcircuit_cell_types"]:
                continue

            if "osi_prob" in p and isinstance(p["osi_prob"][1], list):
                continue

        if "osi" not in p.get("variables", []):
            if "osi_func" in p and p["osi_func"] != ("Identity",):
                continue

            if "osi_prob" in p and p["osi_prob"] != ("Uniform", 0.0, 1.0):
                continue

        if "d" in p and "space" in p.get("variables", []):
            if ("ori" in p["variables"] or "osi" in p["variables"]) and p["d"] > 1:
                continue

        if "d" in p and p["d"] > 1 and "space" not in p.get("variables", []):
            continue

        if not {"cell_type", "space"}.issubset(p.get("variables", [])):
            if p.get("sigma_symmetry") != "pre":
                continue

        parameters.append(list(p.values()))
    return parameters


def get_state_dict(n, sigma_symmetry="pre", vf_symmetry=True):
    sigma = torch.tensor([[150.0, 220.0], [200.0, 175.0]])

    if n == 1:
        state_dict = {
            "gW": torch.tensor([[0.1]]),
            "sigma": sigma[:1, :1],
            "kappa": torch.tensor([[0.3]]),
        }
    elif n == 2:
        if sigma_symmetry == "pre":
            sigma = sigma[:1, :]
        elif sigma_symmetry == "post":
            sigma = sigma[:, :1]
        elif sigma_symmetry == "full":
            sigma = sigma[:1, :1]
        elif isinstance(sigma_symmetry, Sequence) and not isinstance(
            sigma_symmetry, str
        ):
            sigma = sigma[:, 0]
        state_dict = {
            "gW": torch.tensor(
                [
                    [0.1, 0.05],
                    [0.2, 0.15],
                ]
            ),
            "sigma": sigma,
            "kappa": torch.tensor(
                [
                    [0.5, 0.25],
                    [0.3, 0.15],
                ]
            ),
        }
    else:
        raise NotImplementedError()

    if not vf_symmetry:
        if n == 2:
            state_dict["vf"] = torch.tensor([0.8, 1.2])
        elif n != 1:
            raise NotImplementedError()
    return state_dict


class TestV1:
    @pytest.mark.parametrize(
        "sigma_symmetry", [[[0, 1], [1, 0]], "pre", "post", "full", None]
    )
    def test_sigma(self, sigma_symmetry):
        model = nn.V1(
            ["cell_type", "space"],
            cell_types=["PYR", "PV"],
            sigma_symmetry=sigma_symmetry,
        )
        sigma, S = model.sigma, model.S
        assert S.shape == (2, 2)
        if sigma_symmetry == "pre":
            assert sigma.shape == (1, 2)
        elif sigma_symmetry == "post":
            assert sigma.shape == (2, 1)
        elif sigma_symmetry == "full":
            assert sigma.shape == (1, 1)
        elif sigma_symmetry is None:
            assert sigma.shape == (2, 2)
        else:
            assert sigma.shape == (2,)
            assert S[0, 0] == S[1, 1]
            assert S[1, 0] == S[0, 1]
            assert S[0, 0] != S[0, 1]
        assert (S == model.sigma_**2).all()

    @pytest.mark.parametrize(
        "cell_types, null_connections, null_indices",
        [
            (["PYR", "PV"], [], []),
            (["PYR", "PV", "SST"], [("SST", "SST")], [(2, 2)]),
            (
                ["PYR", "PV", "SST", "VIP"],
                [("SST", "SST"), ("VIP", "VIP")],
                [(2, 2), (3, 3)],
            ),
        ],
    )
    def test_null_connections(self, cell_types, null_connections, null_indices):
        cell_types = [getattr(CellType, ct) for ct in cell_types]
        if null_connections is not None:
            null_connections = [
                (getattr(CellType, cti), getattr(CellType, ctj))
                for cti, ctj in null_connections
            ]

        model = nn.V1(
            ["cell_type", "space"],
            cell_types=cell_types,
            sigma_symmetry="pre",
            null_connections=null_connections,
        )

        if len(null_indices) > 0:
            null_indices = tuple(zip(*null_indices))
            assert (model.gW[null_indices] == 0.0).all()
            assert (model.gW.requires_optim[null_indices] == False).all()  # noqa
        else:
            assert (model.gW > 0.0).all()

    @pytest.mark.parametrize("d", [1, 2, 3])
    @pytest.mark.parametrize(
        "variables, osi_func, osi_prob, sigma_symmetry",
        get_parameters(
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
                ["cell_type", "space", "ori"],
                ["cell_type", "ori", "osi"],
                ["cell_type", "space", "ori", "osi"],
            ],
            osi_func=[("Identity",), ("Pow", (0.5,))],
            osi_prob=[
                ("Uniform", 0.0, 1.0),
                ("Beta", [2.0, 1.5], [3.0, 2.5]),
            ],
            sigma_symmetry=[[[0, 1], [1, 0]], "pre", "post", "full", None],
        ),
    )
    def test_weights(self, variables, d, osi_func, osi_prob, sigma_symmetry):
        model = nn.V1(
            variables,
            cell_types=(CellType.PYR, CellType.PV),
            osi_func=osi_func,
            osi_prob=osi_prob,
            f=(
                nn.Match({"PV": nn.SSN(3)}, nn.SSN(2))
                if "cell_type" in variables
                else nn.SSN(2)
            ),
            sigma_symmetry=sigma_symmetry,
        )
        state_dict = get_state_dict(model.n, sigma_symmetry)
        model.load_state_dict(state_dict, strict=False)

        if osi_func is None:
            osi_func = lambda x: x

        N_space, N_ori, N_osi = [5] * d, 4, 4
        x = neurons.as_grid(
            n=(model.n if "cell_type" in variables else 0),
            N_space=(N_space if "space" in variables else ()),
            N_ori=(N_ori if "ori" in variables else 0),
            N_osi=(N_osi if "osi" in variables else 0),
            space_extent=[1000.0] * d,
            ori_extent=(-90.0, 90.0),
            osi_prob=osi_prob,
        )

        with torch.no_grad():
            out = model(x, output="weight", ndim=x.ndim, to_dataframe=False)

        out = out.dense(keep_shape=False) if isinstance(out, CirculantTensor) else out
        x = x.reshape(-1)

        G = torch.atleast_1d(model.gain()).diag()  # (n, n)
        W = torch.linalg.inv(G) @ model.gW  # (n, n)
        sigma = model.S.clone() ** 0.5  # (n, n)
        kappa = model.kappa.clone()  # (n, n)

        if "cell_type" in variables:
            W[:, 1] = -W[:, 1]
            W = W[x["cell_type"], :][:, x["cell_type"]]  # (M, M)
            sigma = sigma[x["cell_type"], :][:, x["cell_type"]]  # (M, M)
            kappa = kappa[x["cell_type"], :][:, x["cell_type"]]  # (M, M)

        if "osi" in variables:
            osi_func = utils.call(nn, osi_func)
            kappa = (
                kappa * osi_func(x["osi"][:, None]) * osi_func(x["osi"][None, :])
            )  # (M, M)

        expected = W

        if "space" in variables:
            r = functional.diff(x["space"][:, None], x["space"][None, :])
            r = r.norm(dim=-1)  # (M, M)
            expected = (
                expected * spatial_component(d, sigma, r) * 1000**d / math.prod(N_space)
            )
            if d > 1:
                expected[r == 0] = 0.0

        if "ori" in variables:
            theta = functional.diff(x["ori"][:, None], x["ori"][None, :])
            theta = theta.tensor.squeeze(-1) / 90.0 * torch.pi  # (M, M)
            expected = expected * (1 + 2 * kappa * torch.cos(theta)) / N_ori

        if "osi" in variables:
            expected = expected / N_osi

        torch.testing.assert_close(out, expected)

    def test_weights_sparse(self):
        prob_kernel = nn.Gaussian(
            nn.Matrix([[100.0, 200.0], [200.0, 100.0]], "cell_type"), "space"
        )
        with random.set_seed(0):
            model = nn.V1(
                ["cell_type", "space"],
                cell_types=(CellType.PYR, CellType.PV),
                prob_kernel={"space": prob_kernel},
            )
        x = neurons.as_grid(
            2, (4,), cell_types=(CellType.PYR, CellType.PV), space_extent=(400,)
        )
        M = 50000
        x_ = frame.stack([x] * M)
        with random.set_seed(0):
            out = model(x_, output="weight", ndim=2, to_dataframe=False)

        prob1 = torch.tensor(
            [
                [1.0, exp(-0.5), exp(-2.0), exp(-0.5)],
                [exp(-0.5), 1.0, exp(-0.5), exp(-2.0)],
                [exp(-2.0), exp(-0.5), 1.0, exp(-0.5)],
                [exp(-0.5), exp(-2.0), exp(-0.5), 1.0],
            ]
        )
        prob2 = torch.tensor(
            [
                [1.0, exp(-0.125), exp(-0.5), exp(-0.125)],
                [exp(-0.125), 1.0, exp(-0.125), exp(-0.5)],
                [exp(-0.5), exp(-0.125), 1.0, exp(-0.125)],
                [exp(-0.125), exp(-0.5), exp(-0.125), 1.0],
            ]
        )
        prob = torch.stack([torch.stack([prob1, prob2]), torch.stack([prob2, prob1])])
        prob = prob.movedim(1, 2)
        sem = (prob * (1 - prob) / M).sqrt()
        out_prob = out.count_nonzero(dim=0) / M
        assert (out_prob >= prob - 2.5 * sem).all()
        assert (out_prob <= prob + 2.5 * sem).all()

        model.prob_kernel = nn.Prod([])
        expected = model(x, output="weight", ndim=2, to_dataframe=False).dense()
        torch.testing.assert_close(out.mean(dim=0), expected, atol=1e-5, rtol=5e-2)

    @pytest.mark.parametrize(
        "f, variables, d, osi_func, osi_prob, sigma_symmetry",
        get_parameters(
            f=[("Identity",)],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
                ["cell_type", "space", "ori"],
                ["cell_type", "ori", "osi"],
                ["cell_type", "space", "ori", "osi"],
            ],
            d=[1, 3],
            osi_func=[("Identity",), 0.5],
            osi_prob=[
                ("Uniform", 0.0, 1.0),
                ("Beta", [2.0, 1.5], [1.0, 1.0]),
            ],
            sigma_symmetry=[[[0, 1], [1, 0]], "pre", "post", "full", None],
        )
        + get_parameters(
            f=[("SSN", (2,))],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["ori", "osi"],
            ],
            d=[1],
            osi_func=[("Identity",)],
            osi_prob=[("Uniform", 0.0, 1.0)],
            sigma_symmetry=["pre"],
        ),
    )
    @pytest.mark.parametrize("vf_symmetry", [True, False])
    def test_forward(
        self, f, variables, d, osi_func, osi_prob, sigma_symmetry, vf_symmetry
    ):
        if not vf_symmetry and "cell_type" in variables and f != ("Identity",):
            with pytest.raises(NotImplementedError):
                model = nn.V1(variables, f=f, vf_symmetry=vf_symmetry)
            return

        model = nn.V1(
            variables,
            cell_types=(CellType.PYR, CellType.PV),
            f=f,
            osi_func=osi_func,
            osi_prob=osi_prob,
            sigma_symmetry=sigma_symmetry,
            vf_symmetry=vf_symmetry,
            autapse=True,
        )
        state_dict = get_state_dict(model.n, sigma_symmetry, vf_symmetry)
        model.load_state_dict(state_dict, strict=False)
        expected_model = copy.deepcopy(model)
        expected_model.mode = "matrix" if f == ("Identity",) else "numerical"

        N_ori, N_osi = 10, 30
        N_space = [350] if d == 1 else [125] * d
        space_extent = [4000.0] if d == 1 else [2000.0] * d
        x = neurons.as_grid(
            n=(model.n if "cell_type" in variables else 0),
            N_space=(N_space if "space" in variables else ()),
            N_ori=(N_ori if "ori" in variables else 0),
            N_osi=(N_osi if "osi" in variables else 0),
            space_extent=space_extent,
            ori_extent=(-90.0, 90.0),
            osi_prob=osi_prob,
        )

        prob = torch.ones(x.shape)

        if "cell_type" in variables:
            prob = prob * (x["cell_type"] == "PYR").float()

        if "osi" in variables:
            prob = prob * (x["osi"] > 0.3).float()

        x["dh"] = torch.zeros(x.shape)
        x["dh"][tuple(prob.nonzero()[0])] = 1.0

        # check unperturbed cells
        x["mask"] = torch.ones(x.shape, dtype=bool)
        x["mask"][x["dh"] != 0] = False

        if d == 1 or variables == ["space"]:
            model.to(torch.double)
            expected_model.to(torch.double)
            x.to(torch.double)

        with torch.no_grad():
            out = model(x, ndim=x.ndim)["dr"]
            expected = expected_model(x, ndim=x.ndim)["dr"]

        if variables != ["cell_type"]:
            assert (out != expected).any()  # check that the two methods are different

        torch.testing.assert_close(
            out, expected, rtol=1.3e-6, atol=expected.abs().max().item() * 2e-4
        )

        # check perturbed cells
        x["mask"] = torch.ones(x.shape, dtype=bool)
        x["mask"][x["dh"] == 0] = False

        with torch.no_grad():
            out = model(x, ndim=x.ndim)["dr"].float()
            expected = expected_model(x, ndim=x.ndim)["dr"].float()

        torch.testing.assert_close(out, expected)

    def test_forward_sample(self):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)
        model = nn.V1(variables, cell_types=cell_types)
        state_dict = get_state_dict(model.n, sigma_symmetry=None)
        model.load_state_dict(state_dict, strict=False)
        expected_model = copy.deepcopy(model)
        expected_model.mode = "matrix"

        data = pd.DataFrame(
            {
                "distance": pd.Categorical.from_codes(
                    list(range(10)) + list(range(0, 10, 2)),
                    categories=pd.interval_range(
                        start=10, end=300, periods=10, closed="left"
                    ),
                ),
                "cell_type": ["PYR"] * 10 + ["PV"] * 5,
            }
        )

        model = nn.Pipeline(model=model, data=[data])
        expected_model = nn.Pipeline(model=expected_model, data=[data])

        N_stim, N, space_extent = 15, 3000, 2000.0
        dataset = Dataset(
            neurons=dict(
                N=N,
                variables=variables,
                cell_types=cell_types,
                space_extent=[space_extent],
            ),
            inputs=dict(configs=[dict(N=2, cell_probs={"PYR": 1.0})]),
            metrics=dict(distance="min_distance_to_ensemble"),
            N_instantiations=N_stim,
            seed=0,
        )

        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        x, kwargs = next(iter(dataloader))
        assert x.shape == (N_stim, 1, N)

        model.to(torch.double)
        expected_model.to(torch.double)
        x.to(torch.double)

        with torch.no_grad():
            out = model(x, **kwargs)
            expected = expected_model(x, **kwargs)

        assert (out != expected).any()  # check that the two methods are different

        # print(out)
        # print(expected)
        # print((out - expected) / expected)
        torch.testing.assert_close(out, expected, rtol=5e-3, atol=0)

    @pytest.mark.parametrize(
        "mode", ["linear_approx", "quasi_linear_approx", "second_order_approx"]
    )
    @pytest.mark.parametrize("f", ["SSN", "Ricciardi", "Match"])
    def test_forward_approx(self, f, mode):
        if f == "Match":
            f = nn.Match({"PYR": nn.SSN(2)}, nn.Rectified())
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)
        model = nn.V1(variables, cell_types=cell_types, f=f, init_vf=1.2, mode=mode)
        state_dict = get_state_dict(model.n, sigma_symmetry=None)
        model.load_state_dict(state_dict, strict=False)
        expected_model = copy.deepcopy(model)
        expected_model.mode = "numerical"

        x = neurons.as_grid(n=model.n, N_space=[1000], space_extent=[4000.0])

        x["dh"] = torch.zeros(x.shape)
        indices = (
            torch.tensor(range(0, 1000, 10)) if mode == "second_order_approx" else 0
        )
        x["dh"][0, indices] = 0.1 if mode == "linear_approx" else 5.0
        if f == "Ricciardi" and mode != "linear_approx":
            # since Ricciardi is less 'nonlinear' than SSN, we can test with a larger perturbation
            x["dh"] = x["dh"] * 10

        # check unperturbed cells
        x["mask"] = torch.ones(x.shape, dtype=bool)
        x["mask"][x["dh"] != 0] = False

        model.to(torch.double)
        expected_model.to(torch.double)
        x.to(torch.double)

        with torch.no_grad():
            out = model(x, ndim=x.ndim)["dr"]
            expected = expected_model(x, ndim=x.ndim)["dr"]

        assert (out != expected).any()  # check that the two methods are different

        torch.testing.assert_close(
            out, expected, rtol=0.05, atol=expected.abs().max().item() * 1e-4
        )

    @pytest.mark.parametrize("f", ["Identity", "Ricciardi"])
    @pytest.mark.parametrize("mode", ["analytical", "numerical"])
    @pytest.mark.parametrize("masked", [True, False])
    @pytest.mark.parametrize("tau", [1.0, [1.0, 0.5]])
    def test_forward_double(self, f, mode, masked, tau):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)

        model = nn.V1(
            variables,
            cell_types=cell_types,
            tau=tau,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
        )
        state_dict = get_state_dict(model.n)
        model.load_state_dict(state_dict, strict=False)

        with lib.random.set_seed(0):
            x = neurons.sample(
                200, variables, cell_types=cell_types, space_extent=[1000.0]
            )

            x["dh"] = perturbation.sample(
                x, 10, cell_probs={"PYR": 1.0}, space=("uniform", 500.0)
            )

        if masked:
            x["mask"] = torch.ones(x.shape, dtype=bool)
            x["mask"][x["dh"] != 0] = False

        model = model.to(torch.double)
        x = x.to(torch.double)
        out = model(x)
        assert out["dr"].dtype == torch.double

    def test_forward_invalid_f(self):
        variables = ["space"]
        cell_types = (CellType.PYR, CellType.PV)
        with pytest.raises(ValueError):
            nn.V1(
                variables,
                cell_types=cell_types,
                f=nn.Match({"PV": nn.SSN(3)}, nn.SSN(2)),
            )

    @pytest.mark.parametrize("f", ["Identity", "Ricciardi", "Match"])
    @pytest.mark.parametrize(
        "mode",
        [
            "analytical",
            "linear_approx",
            "quasi_linear_approx",
            "second_order_approx",
            "numerical",
        ],
    )
    @pytest.mark.parametrize("tau", [1.0, [1.0, 0.5]])
    def test_forward_batched(self, f, mode, tau):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)

        if f == "Match":
            if mode == "analytical":
                pytest.skip("Match function is not implemented in analytical mode")
            f = nn.Match({"PV": nn.Ricciardi(tau=0.01)}, nn.Ricciardi())

        if f == "Identity" and mode.endswith("approx"):
            with pytest.raises(ValueError):
                model = nn.V1(variables, cell_types=cell_types, f=f, mode=mode)
            return

        model = nn.V1(
            variables,
            cell_types=cell_types,
            tau=tau,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
        )
        state_dict = get_state_dict(model.n)
        model.load_state_dict(state_dict, strict=False)

        dataset = Dataset(
            neurons=dict(
                N=200, variables=variables, cell_types=cell_types, space_extent=[1000.0]
            ),
            inputs=dict(
                configs=[
                    dict(N=10, cell_probs={"PYR": 1.0}, space=("uniform", 500.0)),
                    dict(N=10, cell_probs={"PV": 1.0}, space=("uniform", 500.0)),
                ]
            ),
            N_instantiations=5,
            seed=0,
        )

        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        x, kwargs = next(iter(dataloader))
        assert x.shape == (5, 2, 200)

        model = model.to(torch.double)
        x = x.to(torch.double)

        with torch.inference_mode():
            out = model(x, **kwargs["model_kwargs"]).to_pandas()

        expected = []
        for i, j in product(range(5), range(2)):
            xij = x.iloc[i, j]
            with torch.inference_mode():
                expected.append(model(xij, **kwargs["model_kwargs"]).to_pandas())
        expected = pd.concat(expected).reset_index(drop=True)

        torch.testing.assert_close(
            torch.from_numpy(out["dr"].to_numpy()),
            torch.from_numpy(expected["dr"].to_numpy()),
            rtol=1.3e-6,
            atol=1.0e-5,
        )
        out, expected = out.drop(columns="dr"), expected.drop(columns="dr")
        pd.testing.assert_frame_equal(out, expected)

    @pytest.mark.parametrize("f", ["Identity", "Ricciardi", "Match"])
    @pytest.mark.parametrize(
        "mode",
        [
            "analytical",
            "linear_approx",
            "quasi_linear_approx",
            "second_order_approx",
            "numerical",
        ],
    )
    @pytest.mark.parametrize("masked", [True, False])
    @pytest.mark.parametrize("tau", [1.0, [1.0, 0.5]])
    def test_batched_forward(self, f, mode, masked, tau):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)
        batch_shape = (4, 3)
        n = len(cell_types)

        if f == "Match":
            if mode == "analytical":
                pytest.skip("Match function is not implemented in analytical mode")
            f = nn.Match({"PV": nn.Ricciardi(tau=0.01)}, nn.Ricciardi())

        if f == "Identity" and mode.endswith("approx"):
            with pytest.raises(ValueError):
                model = nn.V1(variables, cell_types=cell_types, f=f, mode=mode)
            return

        model = nn.V1(
            variables,
            cell_types=cell_types,
            tau=tau,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
            init_gW_std=0.25,
        )
        batched_model = nn.V1(
            variables,
            cell_types=cell_types,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
            batch_shape=batch_shape,
        )

        with lib.random.set_seed(0):
            x = neurons.sample(
                200, variables, cell_types=cell_types, space_extent=[1000.0]
            )

            x["dh"] = perturbation.sample(
                x, 10, cell_probs={"PYR": 1.0}, space=("uniform", 500.0)
            )

        if masked:
            x["mask"] = torch.ones(x.shape, dtype=bool)
            x["mask"][x["dh"] != 0] = False

        model.to(torch.double)
        batched_model.to(torch.double)
        x = x.to(torch.double)
        state_dict = {
            "gW": torch.empty(*batch_shape, n, n, dtype=torch.double),
            "sigma": torch.empty(*batch_shape, 1, n, dtype=torch.double),
        }
        expected = []
        for seed, (i, j) in enumerate(product(*(range(s) for s in batch_shape))):
            with random.set_seed(seed):
                model.reset_parameters()

            for k in state_dict:
                state_dict[k][i, j] = model.state_dict()[k]

            with torch.inference_mode():
                expected.append(model(x).to_pandas())
        expected = pd.concat(expected).reset_index(drop=True)

        batched_model.load_state_dict(state_dict, strict=False)
        with torch.inference_mode():
            out = batched_model(x).to_pandas()

        torch.testing.assert_close(
            torch.from_numpy(out["dr"].to_numpy()),
            torch.from_numpy(expected["dr"].to_numpy()),
            rtol=1.3e-6,
            atol=1.0e-5,
        )
        out, expected = out.drop(columns="dr"), expected.drop(columns="dr")
        pd.testing.assert_frame_equal(out, expected)

    @pytest.mark.parametrize("f", ["Identity", "Ricciardi"])
    @pytest.mark.parametrize("mode", ["analytical", "numerical"])
    def test_batched_forward_batched(self, f, mode):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)
        batch_shape = (4,)
        n = len(cell_types)

        model = nn.V1(
            variables,
            cell_types=cell_types,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
            init_gW_std=0.25,
        )
        batched_model = nn.V1(
            variables,
            cell_types=cell_types,
            f=f,
            mode=mode,
            sigma_symmetry="pre",
            batch_shape=batch_shape,
        )

        dataset = Dataset(
            neurons=dict(
                N=200, variables=variables, cell_types=cell_types, space_extent=[1000.0]
            ),
            inputs=dict(
                configs=[
                    dict(N=10, cell_probs={"PYR": 1.0}, space=("uniform", 500.0)),
                    dict(N=10, cell_probs={"PV": 1.0}, space=("uniform", 500.0)),
                    dict(N=10, space=("uniform", 500.0)),
                ]
            ),
            seed=0,
            N_instantiations=2,
        )

        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        x, kwargs = next(iter(dataloader))
        assert x.shape == (2, 3, 200)
        x_batch_shape = (2, 3)

        model.to(torch.double)
        batched_model.to(torch.double)
        x = x.to(torch.double)

        state_dict = {
            "gW": torch.empty(*batch_shape, n, n, dtype=torch.double),
            "sigma": torch.empty(*batch_shape, 1, n, dtype=torch.double),
        }
        expected = []
        for seed, idx in enumerate(product(*(range(s) for s in batch_shape))):
            with random.set_seed(seed):
                model.reset_parameters()

            for k in state_dict:
                state_dict[k][idx] = model.state_dict()[k]

            for i, j in product(*(range(s) for s in x_batch_shape)):
                xij = x.iloc[i, j]
                with torch.inference_mode():
                    expected.append(model(xij, **kwargs["model_kwargs"]).to_pandas())
        expected = pd.concat(expected).reset_index(drop=True)

        batched_model.load_state_dict(state_dict, strict=False)
        with torch.inference_mode():
            out = batched_model(x, **kwargs["model_kwargs"]).to_pandas()

        torch.testing.assert_close(
            torch.from_numpy(out["dr"].to_numpy()),
            torch.from_numpy(expected["dr"].to_numpy()),
            rtol=1.3e-6,
            atol=1.0e-5,
        )
        out, expected = out.drop(columns="dr"), expected.drop(columns="dr")
        pd.testing.assert_frame_equal(out, expected)

    @pytest.mark.parametrize(
        "f, variables, subcircuit_cell_types, osi_func, osi_prob, sigma_symmetry",
        get_parameters(
            f=[("Identity",)],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
                ["cell_type", "space", "ori"],
                ["cell_type", "ori", "osi"],
            ],
            subcircuit_cell_types=[
                None,
                ["PYR"],
                ["PV"],
                ["SST"],
                ["PYR", "PV"],
                ["PYR", "SST"],
                ["PV", "SST"],
            ],
            osi_func=[("Identity",), ("Pow", (0.5,))],
            osi_prob=[
                ("Uniform", 0.0, 1.0),
                ("Beta", [2.0, 1.5, 2.0], [3.0, 2.5, 1.5]),
            ],
            sigma_symmetry=[
                [[0, 1, 2], [1, 0, 2], [2, 2, 0]],
                "pre",
                "post",
                "full",
                None,
            ],
        )
        + get_parameters(
            f=[("SSN", (2,))],
            variables=[
                ["cell_type"],
                ["space"],
                ["ori"],
                ["cell_type", "space"],
                ["cell_type", "ori"],
                ["space", "ori"],
                ["ori", "osi"],
            ],
            subcircuit_cell_types=[
                None,
                ["PYR"],
                ["PV"],
                ["SST"],
                ["PYR", "PV"],
                ["PYR", "SST"],
                ["PV", "SST"],
            ],
            osi_func=[("Identity",), ("Pow", (0.5,))],
            osi_prob=[("Uniform", 0.0, 1.0)],
            sigma_symmetry=["pre"],
        ),
    )
    def test_spectral_summary(
        self, f, variables, subcircuit_cell_types, osi_func, osi_prob, sigma_symmetry
    ):
        cell_types = (CellType.PYR, CellType.PV, CellType.SST)
        if subcircuit_cell_types:
            subcircuit_cell_types = tuple(CellType[ct] for ct in subcircuit_cell_types)
        n = len(cell_types)

        model_kwargs = dict(
            cell_types=cell_types,
            f=f,
            init_vf=0.75,
            init_sigma_bounds=(50.0, 150.0),
            osi_func=osi_func,
            osi_prob=osi_prob,
            sigma_symmetry=sigma_symmetry,
            autapse=True,
        )

        model = nn.V1(variables, **model_kwargs)

        for i in range(10):
            print(i)
            with lib.random.set_seed(i):
                model.reset_parameters()

            if "ori" in variables:
                with torch.no_grad():
                    model.kappa[:, 0] = 0.5
                    model.kappa[:, 1:] = -0.5

            output = model.spectral_summary(cell_types=subcircuit_cell_types)._asdict()

            N_space = (20,)
            N_ori = 10
            N_osi = 20
            for _ in range(6):
                print(N_space, N_ori, N_osi)
                neuron_kwargs = dict(
                    n=(n if "cell_type" in variables else 0),
                    N_space=(N_space if "space" in variables else ()),
                    N_ori=(N_ori if "ori" in variables else 0),
                    N_osi=(N_osi if "osi" in variables else 0),
                    space_extent=[2000.0] * len(N_space),
                    ori_extent=(-90.0, 90.0),
                    osi_prob=osi_prob,
                )
                x = neurons.as_grid(**neuron_kwargs)

                if subcircuit_cell_types:
                    indices = [cell_types.index(ct) for ct in subcircuit_cell_types]
                    x = x.iloc[indices]

                with torch.no_grad():
                    W = model(x, output="weight", ndim=x.ndim, to_dataframe=False)

                if f[0] == "SSN":
                    (p,) = f[1]
                    W = W * p * model_kwargs["init_vf"] ** (p - 1)
                spectrum = torch.linalg.eigvals(W)

                expected = {
                    "abscissa": spectrum.real.max(),
                    "radius": spectrum.abs().max(),
                }

                for k, v1, v2 in lib.itertools.dict_zip(output, expected):
                    print(k, v1.item(), v2.item())
                    passed = ((v1 == 0.0) & (v2.abs() < 5.0e-4)).item()
                    passed |= torch.allclose(
                        v1, v2, rtol=1.0e-2 * len(variables), atol=1.0e-5
                    )
                    if not passed:
                        N_space = tuple(Ni * 2 for Ni in N_space)
                        # N_ori = N_ori * 2
                        N_osi = N_osi * 2
                        break
                else:
                    break

            else:
                assert False

    @pytest.mark.parametrize(
        "subcircuit_cell_types",
        [None, ["PYR"], ["PV"], ["PYR", "PV"]],
    )
    def test_spectral_summary_2d(self, subcircuit_cell_types):
        variables = ["cell_type", "space"]
        cell_types = (CellType.PYR, CellType.PV)
        if subcircuit_cell_types:
            subcircuit_cell_types = tuple(CellType[ct] for ct in subcircuit_cell_types)
        n = len(cell_types)

        model_kwargs = dict(
            cell_types=cell_types,
            sigma_symmetry="pre",
            init_sigma_bounds=(50.0, 150.0),
            autapse=True,
        )

        model = nn.V1(variables, **model_kwargs)

        for i in range(1, 10):
            print(i)
            with lib.random.set_seed(i):
                model.reset_parameters()

            output = model.spectral_summary(cell_types=subcircuit_cell_types)._asdict()

            N_space = (60, 60)
            for _ in range(4):
                print(N_space)
                neuron_kwargs = dict(
                    n=n, N_space=N_space, space_extent=[2000.0] * len(N_space)
                )
                x = neurons.as_grid(**neuron_kwargs)

                if subcircuit_cell_types:
                    indices = [cell_types.index(ct) for ct in subcircuit_cell_types]
                    x = x.iloc[indices]

                with torch.no_grad():
                    W = model(x, output="weight", ndim=x.ndim, to_dataframe=False)

                spectrum = torch.linalg.eigvals(W)

                expected = {
                    "abscissa": spectrum.real.max(),
                    "radius": spectrum.abs().max(),
                }

                for k, v1, v2 in lib.itertools.dict_zip(output, expected):
                    print(k, v1.item(), v2.item())
                    passed = ((v1 == 0.0) & (v2.abs() < 5.0e-4)).item()
                    passed |= torch.allclose(
                        v1, v2, rtol=1.0e-2 * len(variables), atol=1.0e-5
                    )
                    if not passed:
                        N_space = tuple(Ni * 2 for Ni in N_space)
                        break
                else:
                    break

            else:
                assert False

    def test_spectral_summary_jacobian(self):
        cell_types = (CellType.PYR, CellType.PV, CellType.SST)
        tau = [1.0, 0.5, 0.75]

        model = nn.V1(["cell_type"], cell_types=cell_types, tau=tau)

        for i in range(10):
            print(i)
            with lib.random.set_seed(i):
                model.reset_parameters()

            output = model.spectral_summary(kind="J")._asdict()

            W = model.gW * model.sign[..., None, :]
            eye = torch.eye(model.n)
            spectrum = torch.linalg.eigvals((W - eye) / torch.tensor(tau)[:, None])

            expected = {
                "abscissa": spectrum.real.max(),
                "radius": spectrum.abs().max(),
            }

            for k, v1, v2 in lib.itertools.dict_zip(output, expected):
                print(k, v1.item(), v2.item())
                torch.testing.assert_close(v1, v2)

    def test_spectral_summary_jacobian_space(self):
        cell_types = (CellType.PYR, CellType.PV)
        tau = [1.0, 0.5]

        model = nn.V1(
            ["cell_type", "space"],
            cell_types=cell_types,
            tau=tau,
            sigma_symmetry="pre",
            init_sigma_bounds=(50.0, 150.0),
            autapse=True,
        )

        tau = torch.tensor(tau)[:, None].broadcast_to(2, 480).reshape(-1)

        for i in range(10):
            print(i)
            with lib.random.set_seed(i):
                model.reset_parameters()

            output = model.spectral_summary(kind="J")._asdict()

            x = neurons.as_grid(n=2, N_space=[480], space_extent=[2000.0])

            with torch.no_grad():
                W = model(x, output="weight", ndim=x.ndim, to_dataframe=False).dense(
                    keep_shape=False
                )

            eye = torch.eye(W.shape[-1])
            spectrum = torch.linalg.eigvals((W - eye) / tau[:, None])

            expected = {
                "abscissa": spectrum.real.max(),
                "radius": spectrum.abs().max(),
            }

            for k, v1, v2 in lib.itertools.dict_zip(output, expected):
                print(k, v1.item(), v2.item())
                torch.testing.assert_close(v1, v2, rtol=2e-3, atol=1e-5)

    @pytest.mark.parametrize("n", [2, 2.5, 3])
    def test_gain(self, n):
        model = nn.V1(
            variables=["cell_type", "space", "ori", "ori"],
            cell_types=(CellType.PYR, CellType.PV),
            f=nn.Rectified() ** n,
            init_vf=3.0,
        )
        torch.testing.assert_close(model.gain(), torch.tensor(n * 3.0 ** (n - 1)))

    def test_vector_gain(self):
        model = nn.V1(
            variables=["cell_type", "space"],
            cell_types=(CellType.PYR, CellType.PV, CellType.SST),
            f=nn.Match({"PV": nn.SSN(3)}, nn.SSN(2)),
            init_vf=3.0,
        )
        torch.testing.assert_close(model.gain(), torch.tensor([6.0, 27.0, 6.0]))

    @pytest.mark.parametrize(
        "osi_prob", [("Beta", 2.0, 3.0), ("Beta", [2.0, 1.0], [3.0, 2.0])]
    )
    def test_osi_func(self, osi_prob):
        model = nn.V1(
            variables=["cell_type", "space", "ori", "osi"],
            cell_types=(CellType.PYR, CellType.PV),
            osi_func=0.5,
            osi_prob=osi_prob,
        )
        x = torch.linspace(0, 1, 50)
        out = model.osi_func(x)
        expected = torch.distributions.Beta(2.0, 3.0).cdf(x) ** 0.5
        torch.testing.assert_close(out, expected)


def test_cdim():
    x = neurons.as_grid(
        n=2,
        N_space=[100] * 3,
        N_ori=10,
        N_osi=10,
        space_extent=[2000.0] * 3,
        ori_extent=(-90.0, 90.0),
    )
    out = _cdim(x, x.ndim)
    expected = (-5, -4, -3, -2)

    assert out == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        ([[0.0], [1.0], [2.0 + 1.0e-6]], (-1,)),
        ([[0.0], [1.5], [2.0 + 1.0e-6]], ()),
    ],
)
def test_cdim2(x, expected):
    x = frame.ParameterFrame({"space": periodic.tensor(x, extents=[(0.0, 3.0)])})
    out = _cdim(x, x.ndim)
    assert out == expected


@pytest.mark.parametrize(
    "osi_func, osi_prob, expected",
    [
        ("Identity", ("Beta", (2.0, 3.0)), 1 / 5),
        ("Identity", ("Beta", ([2.0, 1.0], [3.0, 2.0])), [1 / 5, 1 / 6]),
        (("Pow", (0.5,)), ("Beta", (2.0, 3.0)), 2 / 5),
        (("Pow", (0.5,)), ("Beta", ([2.0, 1.0], [3.0, 2.0])), [2 / 5, 1 / 3]),
        (1.0, ("Beta", (2.0, 3.0)), 1 / 3),
        (0.5, ("Beta", (2.0, 3.0)), 1 / 2),
        (1.0, ("Beta", ([2.0, 1.0], [3.0, 2.0])), [1 / 3, 4 / 15]),
        (0.5, ("Beta", ([2.0, 1.0], [3.0, 2.0])), [1 / 2, 2 / 5]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_compute_osi_scale(osi_func, osi_prob, expected, dtype):
    if not isinstance(osi_func, float):
        osi_func = utils.call(nn, osi_func)
    osi_prob = utils.call(
        torch.distributions, (osi_prob[0], [torch.tensor(x) for x in osi_prob[1]])
    )
    out = compute_osi_scale(osi_prob, osi_func=osi_func, dtype=dtype)
    expected = torch.tensor(expected, dtype=dtype)
    torch.testing.assert_close(out, expected)
