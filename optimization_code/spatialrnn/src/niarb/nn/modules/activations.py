from numbers import Number

import torch
from ricciardi import ricciardi

from .functions import FunctionMixin, Function
from ..parameter import Parameter

__all__ = ["Threshold", "Rectified", "Ricciardi", "SSN"]


class Threshold(FunctionMixin, torch.nn.Threshold):
    """
    f(x) = x if x > threshold else value
    """

    def inv(self):
        if self.inplace:
            raise NotImplementedError()

        return InvThreshold(self.threshold, self.value)

    def nth_deriv(self, n, x):
        if self.inplace:
            raise NotImplementedError()

        if n == 1:
            return (x > self.threshold).float()
        return torch.zeros_like(x)


class InvThreshold(Function):
    def __init__(self, threshold, value):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x):
        if (x <= self.threshold).any() or (x == self.value).any():
            raise ValueError(
                f"""Function is not invertible for inputs less than or
                equal to {self.threshold}, or inputs equal to {self.value},
                but got {x.min()=} and {x[x == self.value].numel()=}."""
            )
        return x


class Rectified(Threshold):
    """
    f(x) = x if x > threshold else threshold
    """

    def __init__(self, threshold=0.0, inplace=False):
        super().__init__(threshold, threshold, inplace=inplace)


class Ricciardi(Function):
    """Ricciardi nonlinearity with an optimizable scale parameter.

    Args:
        scale (optional): Initial value of the scale parameter.
        requires_optim (optional): Whether the scale parameter requires optimization.
        bounds (optional): Lower and upper bounds of the scale parameter.
        tag (optional): Tag of the scale parameter.
        unitless (optional): If True, then the function input is expected to be a unitless
          quantity where a value of 1.0 corresponds to an input with the same magnitude as
          the standard deviation of the input noise, sigma. This is good for simulation
          since we generally want to keep quantities around 1 to avoid numerical issues.
        sigma (optional): Standard deviation of the input noise.
        V_r (optional): Resting potential.
        theta (optional): Threshold potential.
        **kwargs: Additional arguments to pass to ricciardi.

    """

    def __init__(
        self,
        scale=0.025,
        requires_optim=False,
        bounds=(0.0, torch.inf),
        tag="alpha",
        unitless=True,
        sigma=0.01,  # in volts
        V_r=0.01,  # in volts
        theta=0.02,  # in volts
        **kwargs,
    ):
        super().__init__()
        self.init_scale = scale
        self.scale = Parameter(
            torch.empty(()), requires_optim=requires_optim, bounds=bounds, tag=tag
        )
        if unitless:
            V_r = V_r / sigma
            theta = theta / sigma
            sigma = 1.0
        self.kwargs = kwargs | {"sigma": sigma, "V_r": V_r, "theta": theta}
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.scale, self.init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * ricciardi(x, **self.kwargs)


def SSN(p: Number = 2, **kwargs) -> torch.nn.Module:
    return Rectified(**kwargs) ** p
