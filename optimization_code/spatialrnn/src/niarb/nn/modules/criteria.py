import logging

import torch

logger = logging.getLogger(__name__)

__all__ = ["WeightedNormalizedLoss"]


class WeightedNormalizedLoss(torch.nn.Module):
    def __init__(self, w: torch.Tensor | None = None, normalized: bool = True):
        super().__init__()
        if w is not None:
            self.register_buffer("w", w)
        else:
            self.w = 1.0
        self.normalized = normalized

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not x.isfinite().all():
            N_finite = x.isfinite().count_nonzero()
            N = x.numel()
            raise ValueError(
                f"{N - N_finite}/{N} non-finite values (NaN, +inf, or -inf) "
                "detected in model output."
            )

        if self.normalized and x.norm() > 0:
            # Normalize x and y if x is not a zero vector (since it will result in NaN)
            # in which case the loss is 1
            x, y = x / x.norm(), y / y.norm()

        numerator = ((x - y) ** 2 * self.w).mean() ** 0.5
        denominator = (y**2 * self.w).mean() ** 0.5
        loss = numerator / denominator
        logger.debug(f"loss={loss.item()}")
        return loss
