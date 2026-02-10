import torch


def W_(tensor, std, a, b, mask=None):
    """Initialize a tensor with values drawn from a truncated normal distribution with mean 0.

    Args:
        tensor (torch.Tensor): Tensor to be initialized.
        std (float): Standard deviation of the normal distribution.
        a, b (float): Lower and upper bounds of the truncated normal distribution.
        mask (torch.Tensor | None):
            Boolean mask for which elements to initialize with the truncated normal distribution.
            False values are initialized with 0.

    """
    torch.nn.init.trunc_normal_(tensor, 0.0, std, a, b)
    if mask is not None:
        with torch.no_grad():
            tensor[~mask] = 0.0


def uniform_(tensor, bounds, mask=None):
    """Initialize a tensor with values drawn from a uniform distribution.

    Args:
        tensor (torch.Tensor): Tensor to be initialized.
        bounds (tuple[float] | torch.Tensor):
            Lower and upper bounds for the uniform distribution. If a Tensor,
            it must be broadcastable to (*tensor.shape, 2).
        mask (torch.Tensor | None):
            Boolean mask for which elements to initialize with uniform distribution.
            False values are initialized with the lower bound.

    """
    if mask is None:
        mask = torch.ones_like(tensor, dtype=torch.bool)

    bounds = torch.as_tensor(bounds, device=tensor.device, dtype=tensor.dtype)
    bounds = bounds.broadcast_to((*tensor.shape, 2))
    values = torch.distributions.Uniform(
        bounds[..., 0][mask], bounds[..., 1][mask]
    ).sample()

    with torch.no_grad():
        tensor[mask] = values
        tensor[~mask] = bounds[..., 0][~mask]
