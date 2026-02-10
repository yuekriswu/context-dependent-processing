import torch
import hyclib as lib

from niarb import nn, perturbation, neurons
from niarb.cell_type import CellType


def get_state_dict(n):
    if n == 1:
        state_dict = {
            "gW": torch.tensor([[0.5]]),
            "sigma": torch.tensor([[150.0]]),
            "kappa": torch.tensor([[0.3]]),
        }
    elif n == 2:
        state_dict = {
            "gW": torch.tensor(
                [
                    [0.5, 0.25],
                    [0.3, 0.15],
                ]
            ),
            "sigma": torch.tensor([[150.0, 220.0]]),
            "kappa": torch.tensor(
                [
                    [0.5, 0.25],
                    [0.3, 0.15],
                ]
            ),
        }
    else:
        raise NotImplementedError()
    return state_dict


class V1:
    params = (
        [3],
        [1000],
        [10],
        ["Identity", "SSN", "Ricciardi"],
        [True, False],
        ["analytical", "numerical"],
    )
    param_names = ["d", "N", "P", "f", "masked", "mode"]

    def setup(self, d, N, P, f, masked, mode):
        variables = ["cell_type", "space", "ori", "osi"]
        cell_types = (CellType.PYR, CellType.PV)
        osi_prob = ("Beta", 2.0, 5.0)

        model = nn.V1(
            variables, cell_types=cell_types, f=f, mode=mode, sigma_symmetry="pre"
        )
        state_dict = get_state_dict(model.n)
        model.load_state_dict(state_dict, strict=False)

        with lib.random.set_seed(0):
            x = neurons.sample(
                N,
                variables,
                cell_types=cell_types,
                space_extent=[1000.0] * d,
                ori_extent=(-90.0, 90.0),
                osi_prob=osi_prob,
            )

            x["dh"] = perturbation.sample(
                x,
                P,
                cell_probs={"PYR": 1.0},
                space=("uniform", 500.0),
                ori=("von_mises", 1.0),
            )

        if masked:
            distance = x.apply(perturbation.min_distance_to_ensemble)
            x["mask"] = (distance > 0) & (distance < 250)

        self.model = model.to(torch.double)
        self.x = x.to(torch.double)

    def time_forward(self, d, N, P, f, masked, mode):
        with torch.no_grad():
            self.model(self.x)

    def peakmem_forward(self, d, N, P, f, masked, mode):
        with torch.no_grad():
            self.model(self.x)

    def time_forward_backward(self, d, N, P, f, masked, mode):
        y = self.model(self.x)
        y["dr"].mean().backward()

    def peakmem_forward_backward(self, d, N, P, f, masked, mode):
        y = self.model(self.x)
        y["dr"].mean().backward()
