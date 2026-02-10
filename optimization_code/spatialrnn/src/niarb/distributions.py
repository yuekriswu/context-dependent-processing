from torch import Tensor
from scipy import special


def beta_cdf(self, value: Tensor) -> Tensor:
    if (
        value.requires_grad
        or self.concentration1.requires_grad
        or self.concentration0.requires_grad
    ):
        raise NotImplementedError(
            "The beta CDF is neither differentiable with respect to its inputs nor "
            "parameters."
        )
    return special.betainc(
        self.concentration1.cpu(), self.concentration0.cpu(), value.cpu()
    ).to(value.device)
