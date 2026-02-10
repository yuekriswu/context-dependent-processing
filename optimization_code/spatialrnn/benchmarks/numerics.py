import torch

from niarb import numerics, nn


class ComputeGain:
    params = [nn.SSN(), nn.Ricciardi()], [1, 3], [False, True]
    param_names = ["f", "N", "requires_grad"]

    def setup(self, f, N, requires_grad):
        self.vf = torch.randn(N, requires_grad=requires_grad)

    def time_compute_gain(self, f, N, requires_grad):
        numerics.compute_gain(f, self.vf)

    def peakmem_compute_gain(self, f, N, requires_grad):
        numerics.compute_gain(f, self.vf)


class ComputeNthDeriv:
    params = [nn.SSN(), nn.Ricciardi()], [1, 2], [1, 3, 100, 10000], [False, True]
    param_names = ["f", "n", "N", "requires_grad"]

    def setup(self, f, n, N, requires_grad):
        self.vf = torch.randn(N, requires_grad=requires_grad)

    def time_compute_nth_deriv(self, f, n, N, requires_grad):
        numerics.compute_nth_deriv(f, self.vf, n=n)

    def peakmem_compute_nth_deriv(self, f, n, N, requires_grad):
        numerics.compute_nth_deriv(f, self.vf, n=n)
