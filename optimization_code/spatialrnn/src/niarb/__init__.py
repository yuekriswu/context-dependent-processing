import torch
from niarb.distributions import beta_cdf

torch.distributions.Beta.cdf = beta_cdf
