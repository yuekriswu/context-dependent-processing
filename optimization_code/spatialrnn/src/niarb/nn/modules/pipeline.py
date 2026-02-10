from collections.abc import Sequence, Iterable

import torch
import pandas as pd
import tdfl

from .containers import FanOut, NamedSequential
from .analysis import TensorDataFrameAnalysis
from .frame import ParameterFrame
from ..parameter import Parameter
from .functions import Identity, Match
from .activations import Ricciardi


class Scaler(torch.nn.Module):
    def __init__(
        self,
        init_scale: float = 1.0,
        var: str = "dh",
        requires_optim: bool | Sequence[bool] | torch.Tensor = False,
        bounds: Sequence[float | Sequence] | torch.Tensor = (0.0, torch.inf),
        tag: str = "dh",
    ):
        super().__init__()
        self.scale = Parameter(
            torch.empty(()), requires_optim=requires_optim, bounds=bounds, tag=tag
        )
        self.var = var
        self.init_scale = init_scale
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.scale, self.init_scale)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()
        x[self.var] = self.scale * x[self.var]
        return x


class XInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.X_power = Parameter(
            torch.empty(()), bounds=[0.1, 1.0], tag="X_power"
        )
        self.X_sigma_k = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="X_slope")
        self.X_sigma_b = Parameter(torch.empty(()), bounds=[5.0, 20.0], tag="X_intercept")
        self.X_prefactor = Parameter(torch.empty(()), bounds=[0.1, 0.5], tag="X_prefactor")
        self.Y_power = Parameter(
            torch.empty(()), bounds=[-1.0, 1.0], tag="Y_power"
        )
        self.Y_sigma_k = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_slope")
        self.Y_sigma_b = Parameter(torch.empty(()), bounds=[5.0, 20.0], tag="Y_intercept")
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.1, 0.5], tag="Y_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.X_power, 0.5)
        torch.nn.init.constant_(self.X_sigma_k, 0.0)
        torch.nn.init.constant_(self.X_sigma_b, 15.0)
        torch.nn.init.constant_(self.X_prefactor, 0.2)
        torch.nn.init.constant_(self.Y_power, 0.5)
        torch.nn.init.constant_(self.Y_sigma_k, 0.0)
        torch.nn.init.constant_(self.Y_sigma_b, 15.0)
        torch.nn.init.constant_(self.Y_prefactor, 0.2)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()

        X_sigma = self.X_sigma_b + self.X_sigma_k * x["size"].clone()
        Y_sigma = self.Y_sigma_b + self.Y_sigma_k * x["size"].clone()
        X_dh = torch.sqrt(torch.tensor(0.5996) + self.X_prefactor * x["size"].clone() ** self.X_power * torch.exp(-x["distance"].clone() ** 2 / (2 * X_sigma ** 2))) - torch.sqrt(torch.tensor(0.5996))
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * x["size"].clone() ** self.Y_power * torch.exp(-x["distance"].clone() ** 2 / (2 * Y_sigma ** 2))) - torch.sqrt(torch.tensor(0.5996))

        X_mask = (x["cell_type"].clone() == "X") & (x["inverse"].clone() == 0)
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][X_mask] = X_dh[X_mask]
        x["dh"][Y_mask] = Y_dh[Y_mask]
        return x

class X2Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.X_power = Parameter(
            torch.empty(()), bounds=[0.1, 1.0], tag="X_power"
        )
        self.X_sigma_k = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="X_slope")
        self.X_sigma_b = Parameter(torch.empty(()), bounds=[5.0, 20.0], tag="X_intercept")
        self.X_prefactor = Parameter(torch.empty(()), bounds=[0.01, 0.5], tag="X_prefactor")
        self.Y_power = Parameter(
            torch.empty(()), bounds=[0.1, 1.0], tag="Y_power"
        )
        self.Y_sigma_k = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_slope")
        self.Y_sigma_b = Parameter(torch.empty(()), bounds=[5.0, 20.0], tag="Y_intercept")
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.01, 0.5], tag="Y_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.X_power, 0.5)
        torch.nn.init.constant_(self.X_sigma_k, 0.0)
        torch.nn.init.constant_(self.X_sigma_b, 15.0)
        torch.nn.init.constant_(self.X_prefactor, 0.2)
        torch.nn.init.constant_(self.Y_power, 0.5)
        torch.nn.init.constant_(self.Y_sigma_k, 0.0)
        torch.nn.init.constant_(self.Y_sigma_b, 15.0)
        torch.nn.init.constant_(self.Y_prefactor, 0.2)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()

        X_sigma = self.X_sigma_b + self.X_sigma_k * x["size"].clone()
        Y_sigma = self.Y_sigma_b + self.Y_sigma_k * x["size"].clone()
        X_dh = torch.sqrt(torch.tensor(0.5996) + self.X_prefactor * x["size"].clone() ** self.X_power * torch.exp(-x["distance"].clone() ** 2 / (2 * X_sigma ** 2))) - torch.sqrt(torch.tensor(0.5996))
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) ** self.Y_power * torch.exp(-x["distance"].clone() ** 2 / (2 * Y_sigma ** 2))) - torch.sqrt(torch.tensor(0.5996))

        X_mask = (x["cell_type"].clone() == "X") & (x["inverse"].clone() == 0)
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][X_mask] = X_dh[X_mask]
        x["dh"][Y_mask] = Y_dh[Y_mask]
        return x

class X4Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.X_power = Parameter(
            torch.empty(()), bounds=[0.1, 1.0], tag="X_power"
        )
        self.X_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.5], tag="X_prefactor")
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.5], tag="Y_prefactor")
        self.Z_power = Parameter(
            torch.empty(()), bounds=[0.1, 1.0], tag="Z_power"
        )
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.5], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.X_power, 0.5)
        torch.nn.init.constant_(self.X_prefactor, 1.0/8.0)
        torch.nn.init.constant_(self.Y_prefactor, 1.0/8.0)
        torch.nn.init.constant_(self.Z_power, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 1.0/8.0)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()

        X_dh = torch.sqrt(torch.tensor(0.5996) + self.X_prefactor * x["size"].clone() ** self.X_power * torch.exp(-x["distance"].clone() ** 2 / (2 * 15 ** 2))) - torch.sqrt(torch.tensor(0.5996))
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) / 10 * (3*torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2*torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + self.Z_prefactor * x["size"].clone() ** self.Z_power * torch.exp(-x["distance"].clone() ** 2 / (2 * 15 ** 2))) - torch.sqrt(torch.tensor(0.5996))
        X_mask = (x["cell_type"].clone() == "X") & (x["inverse"].clone() == 0)
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][X_mask] = X_dh[X_mask]
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x


class X5Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.5], tag="Y_prefactor")
        self.Z_power = Parameter(
            torch.empty(()), bounds=[0.1, 1.0], tag="Z_power"
        )
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.5], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.Y_prefactor, 1.0/8.0)
        torch.nn.init.constant_(self.Z_power, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 1.0/8.0)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()

        X_dh = torch.sqrt(torch.tensor(0.5996) + 1/8 * x["size"].clone() ** (1/2) * torch.exp(-x["distance"].clone() ** 2 / (2 * 15 ** 2))) - torch.sqrt(torch.tensor(0.5996))
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) / 10 * (3*torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2*torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + self.Z_prefactor * x["size"].clone() ** self.Z_power * torch.exp(-x["distance"].clone() ** 2 / (2 * 15 ** 2))) - torch.sqrt(torch.tensor(0.5996))
        X_mask = (x["cell_type"].clone() == "X") & (x["inverse"].clone() == 0)
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][X_mask] = X_dh[X_mask]
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x


class X6Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.X_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="X_prefactor")
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_prefactor")
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.X_prefactor, 0.5)
        torch.nn.init.constant_(self.Y_prefactor, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 0.5)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()

        X_dh = torch.sqrt(torch.tensor(0.5996) + self.X_prefactor * x["size"].clone() / 400 * torch.exp(-x["distance"].clone() ** 2 / (2 * 15 ** 2))) - torch.sqrt(torch.tensor(0.5996))
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) / 600 * (3 * torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2 * torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + self.Z_prefactor * x["size"].clone() / 400 * torch.exp(-x["distance"].clone() ** 2 / (2 * 15 ** 2))) - torch.sqrt(torch.tensor(0.5996))
        X_mask = (x["cell_type"].clone() == "X") & (x["inverse"].clone() == 0)
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][X_mask] = X_dh[X_mask]
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x

class X8Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_prefactor")
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.Y_prefactor, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 0.5)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) / 10 * (3*torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2*torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + self.Z_prefactor * x["size"].clone() / 10 * (torch.exp(-x["distance"].clone() ** 2 / (2 * 450)) - torch.exp(-x["distance"].clone() ** 2 / (2 * 150)))) - torch.sqrt(torch.tensor(0.5996))
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x

class X9Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_prefactor")
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.Y_prefactor, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 0.5)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) / 600 * (3*torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2*torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + self.Z_prefactor * x["size"].clone() / 160 * (torch.exp(-x["distance"].clone() ** 2 / (2 * 300)) - torch.exp(-x["distance"].clone() ** 2 / (2 * 100)))) - torch.sqrt(torch.tensor(0.5996))
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x


class X10Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_prefactor")
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.Y_prefactor, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 0.5)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) / 600 * (3*torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2*torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + self.Z_prefactor * x["size"].clone() / 400 * (torch.exp(-x["distance"].clone() ** 2 / (2 * 15**2)))) - torch.sqrt(torch.tensor(0.5996))
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x

class X11Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.0], tag="Y_prefactor")
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.Y_prefactor, 0.0)
        torch.nn.init.constant_(self.Z_prefactor, 0.5)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()
        Y_dh = torch.sqrt(torch.tensor(0.5996) + 0.0 * (torch.tensor(90) - x["size"].clone()) / 600 * (3*torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2*torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + self.Z_prefactor * x["size"].clone() / 400 * (torch.exp(-x["distance"].clone() ** 2 / (2 * 15**2)))) - torch.sqrt(torch.tensor(0.5996))
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x

class X12Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_prefactor")
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.0], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.Y_prefactor, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 0.5)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) / 600 * (3*torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2*torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + 0.0 * x["size"].clone() / 400 * (torch.exp(-x["distance"].clone() ** 2 / (2 * 15**2)))) - torch.sqrt(torch.tensor(0.5996))
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x

class X13Input(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_prefactor")
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.0], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.Y_prefactor, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 0.5)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * x["size"].clone() / 400 * (torch.exp(-x["distance"].clone() ** 2 / (2 * 15**2)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + (x["size"].clone() - 45 > 0) / 400 * (torch.exp(-x["distance"].clone() ** 2 / (2 * 15**2)))) - torch.sqrt(torch.tensor(0.5996))
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x



class YInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Y_prefactor = Parameter(torch.empty(()), bounds=[0.0, 1.0], tag="Y_prefactor")
        self.Z_prefactor = Parameter(torch.empty(()), bounds=[0.0, 0.0], tag="Z_prefactor")
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.Y_prefactor, 0.5)
        torch.nn.init.constant_(self.Z_prefactor, 0.5)

    def forward(self, x: ParameterFrame) -> ParameterFrame:
        x = x.copy()
        Y_dh = torch.sqrt(torch.tensor(0.5996) + self.Y_prefactor * (torch.tensor(90) - x["size"].clone()) / 60 * (3*torch.exp(-x["distance"].clone() ** 2 / (2 * 210)) - 2*torch.exp(-x["distance"].clone() ** 2 / (2 * 70)))) - torch.sqrt(torch.tensor(0.5996))
        Z_dh = torch.sqrt(torch.tensor(0.5996) + 0.0 * x["size"].clone() / 400 * (torch.exp(-x["distance"].clone() ** 2 / (2 * 15**2)))) - torch.sqrt(torch.tensor(0.5996))
        Y_mask = (x["cell_type"].clone() == "Y") & (x["inverse"].clone() == 1)
        Z_mask = (x["cell_type"].clone() == "Z") & (x["inverse"].clone() == 1)
        x["dh"] = x["dh"].clone()
        x["dh"][Y_mask] = Y_dh[Y_mask]
        x["dh"][Z_mask] = Z_dh[Z_mask]
        return x


class ToTensor(torch.nn.Module):
    def __init__(self, var: str = "dr"):
        super().__init__()
        self.var = var

    def forward(self, x: Iterable[tdfl.DataFrame]) -> torch.Tensor:
        return torch.cat([xi[self.var] for xi in x])


class Pipeline(NamedSequential):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        data: Iterable[pd.DataFrame] | None = None,
        scaler: torch.nn.Module | dict | None = None,
        analysis: torch.nn.Module | None = None,
        y: str = "dr",
        yerr: str = "dr_se",
        estimator: str = "mean",
    ):
        if data is not None and analysis is not None:
            raise ValueError("`data` and `analysis` cannot both be provided.")

        if scaler is None:
            scaler = {}

        if isinstance(scaler, dict):
            scaler = Scaler(**scaler)

        if data:
            data = list(data).copy()
            for i, df in enumerate(data):
                if y in df.columns:
                    df = df.drop(columns=y)
                if yerr in df.columns:
                    df = df.drop(columns=yerr)
                data[i] = df

        modules = {}
        modules["scaler"] = scaler
        modules["model"] = model

        if data:
            modules["analysis"] = FanOut(
                [TensorDataFrameAnalysis(x=df, y=y, estimator=estimator) for df in data]
            )
            modules["to_tensor"] = ToTensor(var=y)
        elif analysis:
            modules["analysis"] = analysis

        super().__init__(modules)

    def scale_parameters(self, scale: float):
        with torch.no_grad():
            if isinstance(self.model.f, Identity):
                self.scaler.scale *= scale
            elif hasattr(self.model.f, "inv"):
                k = self.model.f.inv()(torch.tensor(scale))
                self.scaler.scale *= k
                self.model.vf *= k
            elif isinstance(self.model.f, Ricciardi):
                self.model.f.scale *= scale
            elif isinstance(self.model.f, Match):
                if not isinstance(self.model.f.default, Ricciardi) or not all(
                    isinstance(v, Ricciardi) for v in self.model.f.cases.values()
                ):
                    raise NotImplementedError()
                self.model.f.default.scale *= scale
                for v in self.model.f.cases.values():
                    v.scale *= scale
            else:
                raise RuntimeError()
