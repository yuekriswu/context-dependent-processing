import torch
from scipy import stats
import pytest

# We need to import niarb to inject cdf function to torch.distributions.Beta
import niarb  # noqa


def test_beta_cdf():
    a, b = 2.0, 3.0
    value = torch.rand(10)
    beta = torch.distributions.Beta(a, b)

    torch.testing.assert_close(
        beta.cdf(value), torch.from_numpy(stats.beta(a, b).cdf(value)).float()
    )


def test_beta_cdf_grad():
    value = torch.rand(10, requires_grad=True)
    beta = torch.distributions.Beta(2.0, 3.0)
    with pytest.raises(NotImplementedError):
        beta.cdf(value / 2)
