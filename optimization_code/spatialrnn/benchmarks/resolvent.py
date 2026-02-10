import torch

from niarb import special


class LaplaceR:
    params = [2, 3], [1.0, 1.0 + 1.0j], [10_000_000], [True, False]
    param_names = ["d", "l", "N", "requires_grad"]

    def setup(self, d, l, N, requires_grad):
        self.l = 2 * torch.tensor(l, requires_grad=requires_grad)
        self.r = torch.linspace(0.0, 10.0, steps=N)

    def time_laplace_r(self, d, l, N, requires_grad):
        special.resolvent.laplace_r(d, self.l, self.r)

    def peakmem_laplace_r(self, d, l, N, requires_grad):
        special.resolvent.laplace_r(d, self.l, self.r)
