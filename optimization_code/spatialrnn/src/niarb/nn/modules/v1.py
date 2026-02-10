import functools
import itertools
import logging
import math
from typing import NamedTuple
from collections.abc import Sequence, Callable, Iterable
from numbers import Number

import torch
from torch import Tensor
import numpy as np
import pandas as pd
import hyclib as lib
import tdfl

from niarb import special, ft, nn, weights, linalg, numerics, exceptions, utils, random
from niarb.nn import functional
from niarb.nn.modules import frame
from niarb.nn.modules.frame import ParameterFrame
from niarb.cell_type import CellType
from niarb.tensors import categorical
from niarb.tensors.periodic import PeriodicTensor
from niarb.tensors.circulant import CirculantTensor
from niarb.nn.modules.kernels import Kernel, Radial

logger = logging.getLogger(__name__)


class SpectralSummary(NamedTuple):
    abscissa: Tensor
    radius: Tensor


def compute_osi_scale(
    osi_prob: torch.distributions.Distribution,
    osi_func: float | Callable[[Tensor], Tensor] = 1.0,
    **kwargs,
) -> Tensor | float:
    r"""Compute $\mathbb{E}[f^2]$, where $f$ is osi_func and expectation is over osi_prob.

    Args:
        osi_prob: Distribution of OSI.
        osi_func: Amplitude of of cosine-tuned weights as a function of OSI.
          f should map [0, 1] to [0, 1], with f(0) = 0 and f(1) = 1.
          If a float, f is given by CDF(x) ** osi_func, where CDF is the cumulative
          distribution of osi_prob. If osi_prob has batch_shape (n,), CDF is taken
          to be the CDF of the first element of osi_prob.
        **kwargs: Optional arguments passed to torch.zeros and torch.as_tensor

    Returns:
        Tensor with shape osi_prob.batch_shape

    """
    if isinstance(osi_func, float):
        # we need to integrate P(x)F(x)^(2a) from 0 to 1
        # where P(x) is the probability density of x and F(x) is the CDF of x
        # so let u = F(x) so that du = P(x)dx. Also, since F(0) = 0 and F(1) = 1
        # by assumption, the integration limits remain the same.
        # Thus we have int_0^1 u^(2a) du = 1 / (2a + 1)
        if math.prod(osi_prob.batch_shape) == 1:
            return torch.tensor(1 / (2 * osi_func + 1), **kwargs)

        def osi_func(x, alpha=osi_func):
            return osi_prob.cdf(x)[0] ** alpha

    if isinstance(osi_func, torch.nn.Identity):
        return torch.as_tensor(osi_prob.mean**2 + osi_prob.variance, **kwargs)

    integrand = lambda x: osi_func(x) ** 2 * osi_prob.log_prob(x).exp()
    lb, ub = torch.zeros(osi_prob.batch_shape, **kwargs), 1
    return lib.pt.integrate.fixed_quad(integrand, lb, ub, n=10)[0]


def UV_decomposition(W: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose W*sigma**-2 into U*V.

    Decomposition is based on the shape of sigma, taking advantage of the implicit
    symmetries in sigma defined by its shape.

    Args:
        W: Tensor with shape (*BW, n, n)
        sigma: Tensor with shape broadcastable to (*BW, n, n).

    Returns:
        Tuple of tensors (U, V) with shapes (*BW, n, m), (*BW, m, n).

    """
    n = W.shape[-1]
    A = W * sigma**-2  # (*BW, n, n)

    if sigma.shape[-2:] == (n, n):
        U = torch.zeros((*A.shape[:-2], n, n**2), dtype=A.dtype, device=A.device)
        V = torch.zeros((n**2, n), dtype=A.dtype, device=A.device)
        for i, j in itertools.product(range(n), repeat=2):
            U[..., i, i * n + j] = A[..., i, j]
            V[i * n + j, j] = 1
    else:
        eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
        U, V = (A, eye) if sigma.shape[-2] == 1 else (eye, A)

    return U, V


def resolvent(
    l: Number,
    x: ParameterFrame,
    y: ParameterFrame,
    W: Tensor,
    sigma: Tensor,
    kappa: Tensor | float,
    osi_func: Callable[[Tensor], Tensor],
    osi_scale: Tensor | float,
    autapse: bool = False,
    order: int = 1,
    mode: str = "parallel",
) -> Tensor:
    """Computes the linear response and connectivity function of the V1 model.

    Plugging in l = -1 yields the linear response (minus a delta function), while
    plugging in l = 0 yeidls the connectivity function. For other values of l,
    this returns l^{-1}(I - (I + lW)^{-1})

    Args:
        l: regular value of resolvent.
        x, y:
            ParameterFrames with any combination of columns ['space', 'cell_type', 'ori', 'osi'], with shapes Bx, By.
            x and y must share the same columns. If "osi" is present, "ori" must also be present.
        W: Tensor with shape (*BW, n, n)
        sigma: Shape must be broadcastable to (*BW, n, n).
        kappa: If a tensor, shape must be broadcastable to (*BW, n, n)
        osi_func: Amplitude of of cosine-tuned weights as a function of OSI.
        osi_scale:
            Expectation of osi_func ** 2 over some distribution of OSI. If a tensor, shape must be
            broadcastable to (n,).
        order (optional): Optional argument passed to functional.wrapped.
        mode (optional): {'parallel', 'sequential'}. Optional argument passed to functional.wrapped.

    Returns:
        Tensor with shape BWxy = broadcast(BW, Bx, By).

    """
    if set(x.keys()) != set(y.keys()):
        raise ValueError(
            f"Expected x and y to have the same keys, but got {x.keys()=} and {y.keys()=}."
        )

    if "osi" in x and "ori" not in x:
        raise ValueError("If 'osi' is present in x, 'ori' must also be present.")

    batch_shape = W.shape[:-2]

    if isinstance(osi_scale, float):
        osi_scale = torch.tensor(osi_scale, device=W.device, dtype=W.dtype)
    osi_scale = osi_scale.broadcast_to((W.shape[-1],))  # (n,)

    if "cell_type" in x:
        i, j = x.data["cell_type"], y.data["cell_type"]  # Bx, By
    else:
        i = torch.zeros(x.shape, device=x.device, dtype=torch.long)  # Bx
        j = torch.zeros(y.shape, device=y.device, dtype=torch.long)  # By

    if "ori" in x:
        W = torch.stack([W, W * kappa], dim=-3)  # (*BW, 2, n, n)
        i, j = i[..., None], j[..., None]  # (*Bx, 1), (*By, 1)

    if "osi" in x:
        l = (
            l * torch.stack([torch.ones_like(osi_scale), osi_scale])[..., None]
        )  # (2, n, 1)

    if "space" in x:
        U, V = UV_decomposition(W, sigma)  # (*BW, [2,], n, m), (*BW, [2,], m, n)
        S = (sigma**2).reshape(*batch_shape, -1)
        S = S.broadcast_to(*batch_shape, U.shape[-1]).diag_embed()  # (*BW, m, m)
        r = functional.diff(x.data["space"], y.data["space"])  # (*Bxy, d)
        if autapse:
            dr = special.ball_radius(r.shape[-1], y.data["space_dV"])  # (*By)
        else:
            dr = 0.0
        if "ori" in x:
            S = S[..., None, :, :]  # (*BW, 1, m, m)
            r = r[..., None, :]  # (*Bxy, 1, d)
            if isinstance(dr, Tensor):
                dr = dr[..., None]  # (*By, 1)
        func = functools.partial(special.resolvent.laplace, dr=dr, validate_args=False)
        func = functools.partial(special.resolvent.mixture, S, U, V, func, l, i, j)
        func = functional.wrapped(func, order=order, mode=mode)
        out = func(r).real  # (*BWxy, [2])
    else:
        eye = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)  # (n,)
        out = W @ torch.linalg.inv(l * W + eye)  # (*BW, [2], n, n)
        out = utils.take_along_dims(
            out, i[..., None, None], j[..., None, None]
        )  # (*BWxy, [2])

    if "osi" in x:
        out1 = out[..., 1] * osi_func(x.data["osi"]) * osi_func(y.data["osi"])  # BWxy
        out = torch.stack(
            torch.broadcast_tensors(out[..., 0], out1), dim=-1
        )  # (*BWxy, 2)

    if "ori" in x:
        theta = functional.diff(x.data["ori"], y.data["ori"])  # (*BWxy, 1)
        out = ft.torus.irftn(
            out, theta, dim=(-1,), period=theta.period.tolist()
        )  # BWxy

    return out.broadcast_to(torch.broadcast_shapes(batch_shape, x.shape, y.shape))


def spectrum(
    W: Tensor,
    S: Tensor | None = None,
    tau: Tensor | None = None,
    kmax: float = 100.0,
    ksteps: int = 1000,
    cell_types: Iterable[int] | None = None,
) -> Tensor:
    """Compute a finite subset of the spectrum of the connectivity or jacobian operator.

    Args:
        W: Tensor with shape (*, n, n)
        S (optional): Tensor with shape (*, n, n). Connectivity width squared.
        tau (optional): Tensor with shape (*, n). Time constants of the model. If None,
          computes spectrum of the connectivity operator W. Otherwise, computes the
          spectrum of the jacobian T^{-1}(W - I)
        kmax (optional): Maximum Fourier frequency.
        ksteps (optional): Number of steps in Fourier frequency.
        cell_types (optional): Spectrum of the subcircuit composed of only the
          specified cell-types. If None, computes spectrum of the full circuit.

    Returns:
        Tensor with shape (*, ksteps, m), where m = len(cell_types) if cell_types else n.

    """
    if S is not None:
        # normalize by mean of S since the spectrum should be invariant to the scaling of S
        S = S / S.mean(dim=(-2, -1), keepdim=True)  # (*, n, n)

        k = torch.linspace(0, kmax, steps=ksteps, device=S.device, dtype=S.dtype)
        k = k[:, None, None]  # (ksteps, 1, 1)
        W = W[..., None, :, :]  # (*, 1, n, n)
        S = S[..., None, :, :]  # (*, 1, n, n)

        op = W * (1 + S * k**2).reciprocal()  # (*, ksteps, n, n)
    else:
        op = W[..., None, :, :]  # (*, 1, n, n)

    if tau is not None:
        eye = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)
        op = tau[..., None, :, None].reciprocal() * (op - eye)  # (*, ksteps/1, n, n)

    if cell_types:
        cell_types = list(cell_types)
        op = op[..., cell_types, :][..., :, cell_types]

    return torch.linalg.eigvals(op)  # (*, ksteps, m)


def spectral_summary(
    W: Tensor,
    S: Tensor | None = None,
    kappa: Tensor | float | None = None,
    osi_scale: Tensor | float | None = None,
    tau: Tensor | None = None,
    **kwargs,
) -> SpectralSummary:
    """Compute spectral abscissa and radius of the connectivity or jacobian operator.

    Args:
        W: Tensor with shape (*, n, n).
        S (optional): If a tensor, shape must be broadcastable to (*, n, n).
        kappa (optional): If a tensor, shape must be broadcastable to (*, n, n).
        osi_scale (optional): Expectation of osi_func ** 2 over some distribution of
          OSI. If a tensor, shape must be broadcastable to (n,).
        tau (optional): Tensor with shape (*, n). Time constants of the model. If None,
          computes spectrum of the connectivity operator W. Otherwise, computes the
          spectrum of the jacobian T^{-1}(W - I)
        **kwargs: Optional arguments passed to spectrum.

    Returns:
        Namedtuple with fields 'abscissa' and 'radius', both of which are Tensors with shape (*)

    """
    if osi_scale is not None:
        kappa = kappa * osi_scale

    if kappa is not None:
        W = torch.stack([W, W * kappa])  # (2, *, n, n)

    eigvals = spectrum(W, S=S, tau=tau, **kwargs)  # ([2], *, ksteps, m)
    eigvals = eigvals.reshape(*eigvals.shape[:-2], -1)  # ([2], *, ksteps*m)

    abscissa = eigvals.real.max(dim=-1).values  # ([2], *)
    radius = eigvals.abs().max(dim=-1).values  # ([2], *)

    if S is not None and tau is None:
        # we have a lower bound for abscissa by taking frequency k to infinity
        abscissa = torch.clip(abscissa, 0, torch.inf)  # ([2], *)

    if kappa is not None:
        abscissa = abscissa.max(dim=0).values  # (*)
        radius = radius.max(dim=0).values  # (*)

    return SpectralSummary(abscissa=abscissa, radius=radius)


class V1(torch.nn.Module):
    r"""Firing rate model of perturbation response of a single layer of mouse V1.

    Model connectivity function is given by
    \[
        W_{\alpha\beta}(x, y, \theta, \phi, \mu, \nu)
        = \frac{w_{\alpha\beta}}{2\pi\sigma_{\alpha\beta}^2}
          G_d(r;\sigma_{\alpha\beta}^{-2})
          (1 + 2\kappa_{\alpha\beta}f(\mu)f(\nu)\cos(\theta - \phi))
    \]
    where $d$ is the number of spatial dimensions, and $G_d(r;\sigma^{-2})$ is defined by
    \[
        G_d(r;\sigma^{-2}) = (2\pi)^{-\frac{d}{2}}(\sigma r)^-\nu K_\nu(\frac{r}{\sigma})
    \]
    where $\nu = \frac{d}{2} - 1$, and $K_\nu$ is the modified Bessel function of the
    second kind of order $\nu$. Distribution of OSI $\mu \in [0, 1]$ is allowed to be
    non-uniform and dependent on cell type with probability distribution $P_\alpha$.
    Number of spatial dimensions is determined by the shape of the model input x, with
    d = x["space"].shape[-1].

    """

    def __init__(
        self,
        variables: Sequence[str],
        *,
        cell_types: Sequence[CellType | str] = tuple(CellType),
        tau: Sequence[float] | float = 1.0,
        osi_func: float | Callable[[Tensor], Tensor] | str | Sequence = "Identity",
        osi_prob: torch.distributions.Distribution | Sequence = ("Uniform", 0.0, 1.0),
        f: float | Callable[[Tensor], Tensor] | str | Sequence = "Identity",
        sigma_symmetry: str | Sequence[Sequence[int]] | Tensor | None = None,
        vf_symmetry: bool = True,
        null_connections: Iterable[Sequence[CellType]] | None = None,
        autapse: bool = False,
        sigma_optim: bool | None = None,
        kappa_optim: bool | None = None,
        vf_optim: bool | None = None,
        gW_bounds: Sequence[float] = (1e-5, 1e3),
        sigma_bounds: Sequence[float | Sequence] | Tensor = (3, 20), # Tensor = (1e0, 1e3),
        kappa_bounds: Sequence[float | Sequence] | Tensor = (-0.5, 0.5),
        vf_bounds: Sequence[float] | Tensor = (1.0e-5, 1e3),
        init_gW_std: float = 0.5,
        init_gW_bounds: Sequence[float] | None = None,
        init_sigma_bounds: Sequence[float | Sequence] | Tensor = (25.0, 300.0),
        init_kappa_bounds: Sequence[float | Sequence] | Tensor | None = None,
        init_vf: float | Tensor = 1.0,
        init_stable: bool = False,
        mode: str = "analytical",
        space_strength_kernel: str | type[Kernel] | None = None,
        space_strength_kernel2: str | type[Kernel] | None = None, # Kris: difference of Gaussians
        prob_kernel: dict[str, Kernel] | None = None,
        monotonic_strength: bool = False,
        keep_monotonic_norm: bool = False,
        monotonic_norm_ord: int | float = 1,
        dense: bool = False,
        N_synapses: float | int = None,
        W_std: float = 0.0,
        seed: int | None = None,
        sparsify_kwargs: dict[str] | None = None,
        nonlinear_kwargs: dict[str] | None = None,
        simulation_kwargs: dict[str] | None = None,
        monotonic_kwargs: dict[str] | None = None,
        wrapped_kwargs: dict[str] | None = None,
        batch_shape: Sequence[int] = (),
        diff_Gaussian_mask: Tensor | None = None, # Kris: difference of Gaussians
        diff_Gaussian_mask2: Tensor | None = None,  # Kris: difference of Gaussians
        gW_requires_optim: Tensor | None = None,  # Kris: requires optimization for gW
    ):
        """Initialize V1 model.

        Args:
            variables:
                {"cell_type", "space", "ori", "osi"}. Dependent variables of the connectivity function.
            cell_types (optional): Cell types in the model.
            tau (optional): Time constants of the model.
            osi_func (optional):
                A function f(x) that determines the scaling of cosine-tuned weights
                as a function of OSI. If a float, f is given by CDF(x) ** osi_func,
                where CDF is the cumulative distribution of osi_prob. If osi_prob
                has batch_shape (n,), CDF is taken to be that of the first element.
                Defaults to f(x) = x.
            osi_prob (optional):
                Distribution of OSI. If a tuple, the first argument is the name of the distribution,
                and the rest are the distribution parameters. batch_shape must be either () or (n,).
                Ignored if "osi" not in variables.
            f (optional): Model nonlinearity. If None, model is linear.
            sigma_symmetry (optional): Symmetries in sigma. If a string, must be one of
                {"pre", "post", "full"}. If tensor-like, must have shape (n, n)
                consisting of consecutive integers starting from 0. If None, no symmetry
                is assumed.
            vf_symmetry (optional): If True, vf is the same for different cell types.
            null_connections (optional):
                Sequence of (cell_type_i, cell_type_j) pairs where the connectivity from
                cell_type_j to cell_type_i is fixed to zero. If None, defaults to the connectivity
                specified by the 'targets' field of each cell type in cell_types.
            autapse (optional): Whether or not to allow autapses. Ignored if space_strength_kernel is
                not None.
            sigma_optim (optional): Whether or not to optimize sigma. Defaults to True if "space" in variables.
            kappa_optim (optional): Whether or not to optimize kappa. Defaults to True if "ori" in variables.
            vf_optim (optional): Whether or not to optimize vf. Defaults to True if f is not Identity.
            gW_bounds (optional): Lower and upper bound for all elements of gW.
            sigma_bounds (optional):
                Lower and upper bound for all elements or elementwise lower and upper bounds of sigma.
            kappa_bounds (optional):
                Lower and upper bound for all elements or elementwise lower and upper bounds of kappa.
            vf_bounds (optional):
                Lower and upper bound for all elements or elementwise lower and upper bounds of vf.
                Defaults to (1.0e-5, inf), since for most typical choices of nonlinearity a negative vf
                would typically result in 0 gain due to rectification.
            init_gW_std (optional): Standard deviation of initial gW.
            init_gW_bounds (optional): Lower and upper bound for initialization of gW.
            init_sigma_bounds (optional):
                Initialization lower and upper bound for all elements or elementwise initialization
                lower and upper bounds of sigma.
            init_kappa_bounds (optional):
                Initialization lower and upper bound for all elements or elementwise initialization
                lower and upper bounds of kappa.
            init_vf (optional): Initial baseline voltage ('v'oltage 'f'ixed-point).
            init_stable (optional): If True, resample initial parameters until the model is stable.
            mode (optional): {'analytical', 'matrix', 'numerical', 'linear_approx', 'quasi_linear_approx',
                'second_order_approx'}. Method for computing the perturbation response.
            wrapped_kwargs (optional): keyword arguments passed to functional.wrapped.
            batch_shape (optional): Shape of model batch dimensions.
            space_strength_kernel (optional): Kernel class for computing spatial connectivity strength.
                Assumed to be translationally invariant. If None, this is the Laplace kernel divided by
                prob_kernel["space"]. If not None, `mode` must be "numerical" or "matrix".
            The remaining options are ignored if `mode` == 'analytical':
            prob_kernel (optional): Dict of kernel functions for computing connection probabilities.
                space and ori kernels are assumed to be translationally invariant. Keys must be ones of
                a subset of {"cell_type"} | set(variables). If "cell_type" is not in variables, use
                "cell_type" to specify the overall probability amplitude.
            monotonic_strength (optional): If True, connectivity strength is modified to be
                monotonically decreasing with distance, and prob_kernel["space"] must be a Radial kernel
                if provided.
            keep_monotonic_norm (optional): If True, the monotonic kernel is scaled such that the
                norm of the product of the monotonic strength kernel and the probability kernel
                is equal to the norm of the product of the non-monotonic strength kernel and the
                probability kernel. Ignored if monotonic_strength is False.
            monotonic_norm_ord (optional): Order of the vector norm used for `keep_monotonic_norm`.
            dense (optional): If True, connectivity is dense, and `N_synapses` must be None.
            N_synapses (optional):
                Expected number of synapses per neuron, must be non-negative. If not None, `dense` must be False.
            W_std (optional): Standard deviation (as a fraction of the mean) of connection weight distribution.
            seed (optional):
                Random seed for generating connection weight matrix, only relevant if N_synapses is not None or W_std > 0.
            sparsify_kwargs (optional): keyword arguments passed to weights.sparsify.
            nonlinear_kwargs (optional): keyword arguments passed to numerics.perturbed_steady_state_approx.
            simulation_kwargs (optional): keyword arguments passed to numerics.perturbed_steady_state.
            wrapped_kwargs (optional): keyword arguments passed to functional.wrapped.
            batch_shape (optional): Shape of model batch dimensions.

        """
        # validate inputs
        if any(v not in {"cell_type", "space", "ori", "osi"} for v in variables):
            raise ValueError(
                f"variables must be a subset of ['cell_type', 'space', 'ori', 'osi'], but got {variables=}."
            )

        if "osi" in variables and "ori" not in variables:
            raise ValueError(
                "If 'osi' is present in variables, 'ori' must also be present."
            )

        if not (
            isinstance(sigma_symmetry, (Tensor, Sequence))
            or sigma_symmetry in {"pre", "post", "full", None}
        ):
            raise ValueError(
                "sigma_symmetry must be a Tensor, Sequence, 'pre', 'post', 'full', or"
                f"None, but got {sigma_symmetry=}."
            )

        if not isinstance(init_vf, float):
            raise NotImplementedError(
                f"Currently only accepts float vf, but got {type(init_vf)=}."
            )

        if mode not in {
            "analytical",
            "matrix",
            "numerical",
            "linear_approx",
            "quasi_linear_approx",
            "second_order_approx",
        }:
            raise ValueError(
                "mode must be either 'analytical', 'matrix', 'numerical', "
                "'linear_approx', 'quasi_linear_approx', or 'second_order_approx', "
                f"but {mode=}."
            )

        if W_std < 0.0:
            raise ValueError(f"W_std must be non-negative, but got {W_std=}.")

        if space_strength_kernel is not None and mode not in {"numerical", "matrix"}:
            raise ValueError(
                "space_strength_kernel must be None if mode is not 'numerical' or 'matrix', "
                f"but got {space_strength_kernel=}, {mode=}."
            )

        if prob_kernel and not set(prob_kernel.keys()).issubset(
            ["cell_type"] + variables
        ):
            raise ValueError(
                f"prob_kernel keys must be a subset of {'cell_type'} | set(variables), "
                f"but got {prob_kernel.keys()=}."
            )

        if (
            monotonic_strength
            and "space" in prob_kernel
            and not isinstance(prob_kernel["space"], Radial)
        ):
            raise ValueError(
                "If monotonic_strength is True and prob_kernel['space'] is provided, "
                "prob_kernel['space'] must be a Radial kernel, but "
                f"{type(prob_kernel['space'])=}."
            )

        if dense and N_synapses is not None:
            raise ValueError("`N_synapses` must be none if `dense` is True.")

        if wrapped_kwargs is not None:
            raise NotImplementedError("wrapped_kwargs is not currently implemented.")

        cell_types = tuple(
            CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
        )

        # initialize defaults
        if "cell_type" not in variables:
            cell_types = [cell_types[0]]
        n = len(cell_types)

        if isinstance(tau, float):
            tau = [tau] * n

        if not isinstance(osi_func, float) and not callable(osi_func):
            osi_func = utils.call(nn, osi_func)

        if isinstance(osi_prob, Sequence):
            osi_prob = getattr(torch.distributions, osi_prob[0])(
                *[torch.as_tensor(v) for v in osi_prob[1:]]
            )
        if len(osi_prob.batch_shape) not in {0, 1}:
            raise ValueError(
                f"osi_prob must have 0/1D batch_shape, but got {osi_prob.batch_shape=}."
            )

        if not callable(f):
            f = utils.call(nn, f)

        if isinstance(f, nn.Match) and "cell_type" not in variables:
            raise ValueError("nn.Match nonlinearity requires 'cell_type' variable.")

        if (
            not vf_symmetry
            and "cell_type" in variables
            and mode == "analytical"
            and not isinstance(f, nn.Identity)
        ):
            raise NotImplementedError(
                "The combination vf_symmetry == False and 'cell_type' in variables "
                "mode == 'analytical' and f is not Identity is not yet implemented."
            )

        if isinstance(sigma_symmetry, Sequence) and not isinstance(sigma_symmetry, str):
            sigma_symmetry = torch.tensor(sigma_symmetry)

        if isinstance(sigma_symmetry, Tensor):
            if sigma_symmetry.shape != (n, n):
                raise ValueError(
                    f"sigma_symmetry must have shape (n, n), but got {sigma_symmetry.shape=}."
                )
            if sigma_symmetry.dtype != torch.long:
                raise ValueError(
                    f"sigma_symmetry must have dtype torch.long, but got {sigma_symmetry.dtype=}."
                )
            m = sigma_symmetry.max().item() + 1
            if set(sigma_symmetry.reshape(-1).tolist()) != set(range(m)):
                raise ValueError(
                    "sigma_symmetry must contain consecutive integers starting from 0."
                )

        if mode == "matrix" and not isinstance(f, nn.Identity):
            raise ValueError("f must be nn.Identity when mode = 'matrix'.")

        if mode.endswith("approx") and isinstance(f, nn.Identity):
            raise ValueError("f must be nonlinear when using approximation modes.")

        if null_connections is None:
            null_connections = filter(
                lambda ct: ct[0] not in ct[1].targets,
                itertools.product(cell_types, cell_types),
            )

        if sigma_optim is None:
            sigma_optim = "space" in variables

        if kappa_optim is None:
            kappa_optim = "ori" in variables

        if vf_optim is None:
            vf_optim = not isinstance(f, nn.Identity)

        if prob_kernel is None:
            prob_kernel = {}

        if monotonic_kwargs is None:
            monotonic_kwargs = {}

        super().__init__()

        self.variables = list(variables)
        self.cell_types = cell_types
        self._osi_func = osi_func
        self.osi_prob = osi_prob
        self.f = f
        self.autapse = autapse
        self.mode = mode
        self.N_synapses = N_synapses
        self.dense = dense
        self.W_std = W_std
        self.seed = seed
        self.sparsify_kwargs = sparsify_kwargs if sparsify_kwargs else {}
        self.nonlinear_kwargs = nonlinear_kwargs if nonlinear_kwargs else {}
        self.simulation_kwargs = simulation_kwargs if simulation_kwargs else {}
        self.wrapped_kwargs = wrapped_kwargs if wrapped_kwargs else {}
        self.diff_Gaussian_mask = diff_Gaussian_mask
        self.diff_Gaussian_mask2 = diff_Gaussian_mask2

        # define model parameters
        # note that gW refers to the matrix GW, where G = diag(gain). The lowercase
        # g is due to the fact that in older versions of this code, gain is a scalar.

        self.gW = nn.Parameter(
            torch.empty((*batch_shape, n, n)),
            bounds=torch.as_tensor(gW_bounds),
            tag="gW",
        )  # (*, n, n)


        for cti, ctj in null_connections:
            self.gW.requires_optim[
                ..., cell_types.index(cti), cell_types.index(ctj)
            ] = False

        if gW_requires_optim is not None:
            self.gW.requires_optim = gW_requires_optim

        sigma_shape = {"pre": (1, n), "post": (n, 1), "full": (1, 1), None: (n, n)}
        sigma_shape = (
            (m,) if isinstance(sigma_symmetry, Tensor) else sigma_shape[sigma_symmetry]
        )
        self.sigma = nn.Parameter(
            torch.empty((*batch_shape, *sigma_shape)),
            requires_optim=sigma_optim,
            bounds=torch.as_tensor(sigma_bounds),
            tag="sigma",
        )  # (*, 1 or n, 1 or n) or (*, m)

        # Kris: different of Gaussians
        self.sigma2 = nn.Parameter(
            torch.empty((*batch_shape, *sigma_shape)),
            requires_optim=sigma_optim,
            bounds=torch.as_tensor(sigma_bounds),
            tag="sigma2",
        )

        self.kappa = nn.Parameter(
            torch.empty((*batch_shape, n, n)),
            requires_optim=kappa_optim,
            bounds=torch.as_tensor(kappa_bounds),
            tag="kappa",
        )  # (*, n, n)

        self.vf = nn.Parameter(
            torch.empty(() if vf_symmetry or n == 1 else (n,)),
            requires_optim=vf_optim,
            bounds=torch.as_tensor(vf_bounds),
            tag="vf",
        )

        self.register_buffer("tau", torch.tensor(tau), persistent=False)  # (n,)
        self.register_buffer(
            "sign",
            torch.tensor([ct.sign for ct in cell_types]).float(),
            persistent=False,
        )  # (*, n)
        if isinstance(sigma_symmetry, Tensor):
            self.register_buffer("sigma_symmetry", sigma_symmetry, persistent=False)
        else:
            self.sigma_symmetry = sigma_symmetry

        # define model kernels
        if isinstance(space_strength_kernel, str):
            space_strength_kernel = getattr(nn, space_strength_kernel)

        if space_strength_kernel is not None and not issubclass(
            space_strength_kernel, Kernel
        ):
            raise TypeError("space_strength_kernel must be a Kernel subclass or None.")

        has_ct = "cell_type" in self.variables

        space_prob_kernel = prob_kernel.get("space", 1)
        sqrtS = nn.Matrix(self.sqrtS, "cell_type") if has_ct else nn.Scalar(self.sqrtS)

        # Kris: difference of Gaussians
        sqrtS2 = nn.Matrix(self.sqrtS2, "cell_type") if has_ct else nn.Scalar(self.sqrtS2)

        if space_strength_kernel is None:
            if autapse:
                k = nn.AutapsedLaplace(
                    sqrtS, ["space", "space_dV"], normalize="integral"
                )
            else:
                k = nn.Laplace(sqrtS, "space", normalize="integral")
            if monotonic_strength:
                space_strength_kernel = nn.Monotonic(
                    k / space_prob_kernel, "space", **monotonic_kwargs
                )
                if keep_monotonic_norm:
                    space_strength_kernel = (
                        space_strength_kernel
                        * nn.Norm(k, "cell_type", ord=monotonic_norm_ord)
                        / nn.Norm(
                            space_strength_kernel * space_prob_kernel,
                            "cell_type",
                            ord=monotonic_norm_ord,
                        )
                    )
                space_product_kernel = space_strength_kernel * space_prob_kernel
            else:
                space_product_kernel = k
                space_strength_kernel = space_product_kernel / space_prob_kernel
        else:
            space_strength_kernel = space_strength_kernel(
                sqrtS, "space", normalize="integral"
            )
            space_product_kernel = space_strength_kernel * space_prob_kernel

            # Kris: difference of Gaussians
            if self.diff_Gaussian_mask is not None:
                space_strength_kernel2 = space_strength_kernel2(
                    sqrtS2, "space", normalize="integral"
                )
                space_product_kernel2 = space_strength_kernel2 * space_prob_kernel

        kappa_kernel = (
            nn.Matrix(self.kappa_, "cell_type") if has_ct else nn.Scalar(self.kappa_)
        )
        if "osi" in variables:
            kappa_kernel = kappa_kernel * nn.RankOne(self.osi_func, x_keys="osi")

        product_kernel = {
            "cell_type": (
                nn.Matrix(self.W, "cell_type") if has_ct else nn.Scalar(self.W)
            ),
            "space": space_product_kernel,
            "ori": nn.Tuning(kappa_kernel, "ori", normalize=True),
        }

        # Kris: difference of Gaussians
        if self.diff_Gaussian_mask is not None:
            product_kernel2 = {
                "cell_type": (
                    nn.Matrix(self.W, "cell_type") if has_ct else nn.Scalar(self.W)
                ),
                "space": space_product_kernel2,
                "ori": nn.Tuning(kappa_kernel, "ori", normalize=True),
            }

        strength_kernel = {
            "cell_type": product_kernel["cell_type"] / prob_kernel.get("cell_type", 1),
            "space": space_strength_kernel,
            "ori": product_kernel["ori"] / prob_kernel.get("ori", 1),
        }

        def filt(x):
            return x[0] in {"cell_type"} | set(self.variables)

        self.product_kernel = nn.Prod(dict(filter(filt, product_kernel.items())))
        if self.diff_Gaussian_mask is not None:
            self.product_kernel2 = nn.Prod(dict(filter(filt, product_kernel2.items())))
        self.prob_kernel = nn.Prod(dict(filter(filt, prob_kernel.items())))
        self.strength_kernel = nn.Prod(dict(filter(filt, strength_kernel.items())))

        # initialize model parameters
        self.init_gW_std = init_gW_std
        self.init_gW_bounds = gW_bounds if init_gW_bounds is None else init_gW_bounds
        self.init_sigma_bounds = init_sigma_bounds
        self.init_kappa_bounds = (
            kappa_bounds if init_kappa_bounds is None else init_kappa_bounds
        )
        self.init_vf = init_vf
        self.init_stable = init_stable
        self.reset_parameters() # After initialization, assign values to gW, therefore W by the W function.

    @property
    def n(self):
        return self.gW.shape[-1]

    @property
    def batch_shape(self):
        return self.gW.shape[:-2]

    @property
    def batch_ndim(self):
        return len(self.batch_shape)

    @property
    def sigma_(self):
        return (
            self.sigma[self.sigma_symmetry]
            if isinstance(self.sigma_symmetry, Tensor)
            else self.sigma
        )

    # Kris: different of Gaussians
    @property
    def sigma2_(self):
        return (
            self.sigma2[self.sigma_symmetry]
            if isinstance(self.sigma_symmetry, Tensor)
            else self.sigma2
        )

    @property
    def S(self):
        batch_shape, n = self.batch_shape, self.n
        return (self.sigma_**2).broadcast_to(*batch_shape, n, n)

    @property
    def osi_scale(self):
        if "osi" not in self.variables:
            return 1.0

        return compute_osi_scale(
            self.osi_prob,
            osi_func=self._osi_func,
            device=self.gW.device,
            dtype=self.gW.dtype,
        )  # osi_prob.batch_shape

    @property
    def osi_func(self):
        if isinstance(self._osi_func, float):
            if self.osi_prob.batch_shape != ():
                if not isinstance(self.osi_prob, torch.distributions.Beta):
                    raise NotImplementedError(
                        "Only supports Beta distribution for now."
                    )
                osi_prob = type(self.osi_prob)(
                    self.osi_prob.concentration1[0], self.osi_prob.concentration0[0]
                )
            else:
                osi_prob = self.osi_prob
            return lambda x: osi_prob.cdf(x) ** self._osi_func
        return self._osi_func

    def kappa_(self) -> Tensor:
        return self.kappa

    def W(self, with_gain: bool = False, **kwargs) -> Tensor:
        W = self.gW * self.sign[..., None, :]  # (*batch_shape, n, n)
        if not with_gain:
            W = W / self.gain(**kwargs)[..., None]  # (*batch_shape, n, n)
        return W

    def sqrtS(self) -> Tensor:
        batch_shape, n = self.batch_shape, self.n
        return self.sigma_.broadcast_to(*batch_shape, n, n)

    # Kris: difference of Gaussians
    def sqrtS2(self) -> Tensor:
        batch_shape, n = self.batch_shape, self.n
        return self.sigma2_.broadcast_to(*batch_shape, n, n)

    def reset_parameters(self):
        torch.nn.init.constant_(self.vf, self.init_vf)
        nn.init.W_(
            self.gW, self.init_gW_std, *self.init_gW_bounds, self.gW.requires_optim
        )
        nn.init.uniform_(self.sigma, self.init_sigma_bounds, self.sigma.requires_optim)

        # Kris: different of Gaussians
        nn.init.uniform_(self.sigma2, self.init_sigma_bounds, self.sigma.requires_optim)

        nn.init.uniform_(self.kappa, self.init_kappa_bounds, self.kappa.requires_optim)

        if self.init_stable and self.spectral_summary().abscissa.item() >= 1.0:
            self.reset_parameters()

    def gain(self, **kwargs) -> Tensor:
        """Compute gain of the model.

        Args:
            **kwargs: keyword arguments to torch.autograd.functional.jacobian

        Returns:
            torch.Tensor: Tensor with shape () or (n,)

        """
        f, vf = self.f, self.vf
        if isinstance(f, nn.Match):
            f = functools.partial(
                f.forward,
                key=categorical.tensor(
                    list(range(self.n)),
                    categories=[ct.name for ct in self.cell_types],
                    device=self.vf.device,
                ),
            )
            vf = vf.broadcast_to((self.n,))
        return numerics.compute_gain(f, vf, **kwargs)  # () or (n,)

    def resolvent(
        self,
        l: Number,
        x: ParameterFrame,
        y: ParameterFrame,
        with_gain: bool = True,
        **kwargs,
    ) -> Tensor:
        W = self.W(with_gain=with_gain, **kwargs)  # (*batch_shape, n, n)
        if self.batch_ndim > 0:
            idx = (slice(None),) * self.batch_ndim + (None,) * x.ndim
            W, sigma, kappa = W[idx], self.sigma_[idx], self.kappa[idx]
        else:
            sigma, kappa = self.sigma_, self.kappa

        return resolvent(
            l,
            x,
            y,
            W,
            sigma,
            kappa,
            self.osi_func,
            self.osi_scale,
            autapse=self.autapse,
            **self.wrapped_kwargs,
        )  # (*self.batch_shape, *broadcast_shapes(x.shape, y.shape))

    def spectral_summary(
        self, cell_types: Iterable[CellType | str] | None = None, kind="W", **kwargs
    ) -> SpectralSummary:
        """Compute spectral abscissa and radius of the model's connectivity or jacobian operator.

        Args:
            cell_types (optional): If not None, restrict to subcircuit composed of only
              the specified cell-types.
            kind (optional): {'W', 'J'}. Whether to compute the spectrum of the
              connectivity or jacobian operator.
            **kwargs: Optional arguments passed to v1.spectral_summary

        Returns:
            Named tuple with fields 'abscissa' and 'radius':
              abscissa: Tensor with shape self.batch_shape
              radius: Tensor with shape self.batch_shape

        """
        if kind not in {"W", "J"}:
            raise ValueError(f"kind must be either 'W' or 'J', but got {kind=}.")

        if cell_types:
            cell_types = (
                CellType[ct] if isinstance(ct, str) else ct for ct in cell_types
            )
            cell_types = (self.cell_types.index(ct) for ct in cell_types)

        return spectral_summary(
            self.gW * self.sign[..., None, :],
            S=(self.S if "space" in self.variables else None),
            kappa=(self.kappa if "ori" in self.variables else None),
            osi_scale=(self.osi_scale if "osi" in self.variables else None),
            tau=(self.tau if kind == "J" else None),
            cell_types=cell_types,
            **kwargs,
        )

    def forward(
        self,
        x: ParameterFrame,
        output: str = "response",
        ndim: int = 1,
        in_var: str = "dh",
        out_var: str = "dr",
        mask_var: str = "mask",
        gain_kwargs: dict[str] | None = None,
        check_circulant: bool = True,
        assert_finite: bool = True,
        to_dataframe: str | bool = True,
    ) -> ParameterFrame | tdfl.DataFrame | pd.DataFrame | Tensor:
        """Compute perturbation response or connectivity weights of the model.

        Output is computed for each set of model parameters, i.e. it is equivalent
        to computing output for each param[*idx] where idx in np.ndindex(self.batch_shape)
        and param is a model parameter ("gW", "sigma", "kappa")

        Args:
            x: ParameterFrame containing zero-dimensional columns {in_var, 'dV'}
            output (optional): {'response', 'weight'}. Whether to compute perturbation
                response or connectivity weights.
            ndim (optional): Number of non-batch dimensions (assumed to be trailing).
            in_var (optional): Column name of input. Ignored if output == 'weight'.
            out_var (optional): Column name of output. Ignored if output == 'weight'.
            mask_var (optional):
                Column name of output mask. Ignored if output == 'weight'. If mask_var in x,
                only returns neurons where x[mask] == True. This improves efficiency of computing
                analytic linear response when only the responses of a subset of neurons are desired.
            gain_kwargs (optional): Keyword arguments to numerics.compute_gain. Ignored if output == 'weight'.
            check_circulant (optional):
                Check if the connectivity is circulant and use circulant matrix optimizations if so.
                Set to False if it is known to not be circulant to improve performance.
            assert_finite (optional): Raise error if Infs or NaNs are present in model output.
                Ignored if output == 'weight'.
            to_dataframe (optional): If True, returns a tdfl.DataFrame. If a str, must
                be either 'tdfl' or 'pandas'. If 'pandas', returns a pandas.DataFrame.

        Returns:
            If output == 'response', a DataFrame with column out_var added. If
            to_dataframe is False, then output is instead a ParamaterFrame.
            Otherwise, a DataFrame with columns ['W', 'presynaptic_cell_type',
            'postsynaptic_cell_type', 'distance', 'rel_ori', 'presynaptic_osi',
            'postsynaptic_osi'] (excluding columns incompatible with model
            variables, e.g. if 'ori' is not in self.variables, then 'rel_ori'
            is not an output column). If to_dataframe is False, then output
            is instead a Tensor of connectivity weights.

        """

        if output not in {"response", "weight"}:
            raise ValueError(
                f"output must be either 'response' or 'weight', but got {output=}."
            )

        if isinstance(to_dataframe, str) and to_dataframe not in {"tdfl", "pandas"}:
            raise ValueError(
                "to_dataframe must be either 'tdfl', 'pandas', or bool, but got "
                f"{to_dataframe=}."
            )
        # don't keep indices if output is connectivity weights to save memory
        framelike_kwargs = {
            "cls": pd.DataFrame if to_dataframe == "pandas" else tdfl.DataFrame,
            "keep_indices": to_dataframe == "pandas" and output == "response",
            "to_numpy": to_dataframe == "pandas",
        }

        if (
            output == "response"
            and self.mode != "numerical"
            and (self.N_synapses or self.W_std)
        ):
            raise ValueError(
                "N_synapses must be None and W_std must be 0 when output == 'response' and mode != 'numerical'."
            )

        if gain_kwargs is None:
            gain_kwargs = {}

        keys = self.variables + ["dV"]
        if "space" in self.variables and "space_dV" in x.keys():
            keys += ["space_dV"]

        if check_circulant:
            cdim = _cdim(x.data[self.variables], ndim, rtol=1.0e-2, atol=1.0e-8)
        else:
            cdim = ()
        logger.debug(f"Assuming circulant dimensions are {cdim}, with {x.shape=}")

        f, f_, f_args = self.f, self.f, ()
        if isinstance(f, nn.Match):
            # ensure nn.Match nonlinearities gets called with cell type information
            f_ = functools.partial(f, key=x.data["cell_type"])
            f_args = (x.data["cell_type"],)

        # expand vf to shape of x if vf is not a scalar
        vf = self.vf[x["cell_type"]] if self.vf.ndim > 0 else self.vf

        if output == "response" and self.mode in {
            "analytical",
            "linear_approx",
            "quasi_linear_approx",
        }:
            if isinstance(f, nn.Identity) or self.mode.endswith("linear_approx"):
                x_ = x.data[keys]

                # For convenience, move batch, non-broadcasted dimensions to the front.
                # These are the "model instance" dimensions.
                nbcastdims = 0
                for d, (n0, n1) in enumerate(zip(x_.shape[:-ndim], x.shape[:-ndim])):
                    if n0 != n1:
                        # these are the broadcasted dimensions ("stimuli" dimensions)
                        assert n0 == 1
                        x = x.movedim(d, -ndim - 1)
                        x_ = x_.movedim(d, -ndim - 1)
                        nbcastdims += 1
                nbatchdims = x.ndim - ndim - nbcastdims
                bcastdim = tuple(range(nbcastdims))

                # Iterate over batch, non-broadcasted dimensions.
                # Although for-loop is slow, this approach has O(N*P) rather than O(N^2)
                # complexity, where P is the number of perturbed neurons.
                dr = torch.empty(
                    (*self.batch_shape, *x.shape),
                    dtype=x.data[in_var].dtype,
                    device=x.device,
                )
                for idx in np.ndindex(x.shape[:nbatchdims]):
                    dh = x[in_var][idx]  # shape[nbatchdim:]

                    # ugly code to get the correct nonlinearity and vf for current idx
                    if isinstance(f, nn.Match):
                        f_ = functools.partial(f, key=x["cell_type"][idx])
                        f_args = (x["cell_type"][idx],)
                    if vf.ndim > 0:
                        vf = vf[idx]

                    if self.mode == "linear_approx":
                        dh = numerics.compute_nth_deriv(f, vf, f_args) * dh
                    elif self.mode == "quasi_linear_approx":
                        dh = f_(vf + dh) - f_(vf)
                    in_mask = (dh != 0).any(dim=bcastdim)  # shape[-ndim:]

                    x_post = x_.iloc[idx].squeeze(dim=bcastdim)  # shape[-ndim:]
                    x_pre = x_post[in_mask]  # (P_in,)
                    if mask_var in x:
                        out_mask = x[mask_var][idx].any(dim=bcastdim)  # shape[-ndim:]
                        x_post = x_post.iloc[out_mask]  # (P_out,)

                    x_post, x_pre = frame.meshgrid(x_post, x_pre, sparse=True)
                    R = self.resolvent(-1, x_post, x_pre) * x_pre.data["dV"]
                    # R = self.resolvent(-1, x_post, x_pre) * x_post.data["dV"]
                    # currently R has shape (*bshape, ?, ?) but dh has shape (*, ?)
                    R = R[(slice(None),) * self.batch_ndim + (None,) * nbcastdims]

                    if mask_var in x:
                        dr[(slice(None),) * self.batch_ndim + (*idx, ..., out_mask)] = (
                            linalg.nbmv(R, dh[..., in_mask]) + dh[..., out_mask]
                        )
                    else:
                        dr[(slice(None),) * self.batch_ndim + idx] = (
                            linalg.nbmv(R, dh[..., in_mask], ndim1=ndim) + dh
                        )

            else:
                kernel = functools.partial(self.resolvent, -1)
                J = weights.discretize(
                    kernel, x.data[keys], ndim=ndim, dim=cdim
                )  # (I - gW)^{-1} - I, (*bshape, *, *shape, *shape)
                I = linalg.eye_like(J)  # (*bshape, *, *shape, *shape)
                gain = self.gain(**gain_kwargs)  # () or (n,)
                if gain.ndim != 0:
                    raise NotImplementedError("Currently only supports scalar gain.")
                J = (J + I) * gain  # (g^{-1} - W)^{-1}, (*bshape, *, *shape, *shape)
                dh = x.data[in_var]  # (*, *shape)
                dr = numerics.perturbed_steady_state_approx(
                    vf, J, f_, dh, gain=gain, **self.nonlinear_kwargs
                )  # (*bshape, *, *shape)

        elif output == "response" and self.mode == "second_order_approx":
            dh = x.data[in_var]  # (*, *shape)

            G = numerics.compute_nth_deriv(f, vf, f_args)  # ()
            dG = numerics.compute_nth_deriv(f, vf + dh, f_args) - G  # (*, *shape)
            H = numerics.compute_nth_deriv(f, vf + dh, f_args, n=2)  # (*, *shape)
            # print(G.shape)
            # # print(G[0, :, :])
            # print(G[:, 0, 0])

            dh = f_(vf + dh) - f_(vf)  # (*, *shape)
            kernel = functools.partial(self.resolvent, -1)
            tL = weights.discretize(
                kernel, x.data[keys], ndim=ndim, dim=cdim
            )  # (I - gW)^{-1} - I, (*bshape, *, *shape, *shape)

            GinvtLdh = (tL @ dh.unsqueeze(-1)).squeeze(-1) / G  # (*bshape, *, *shape)
            dh = dh + dG * GinvtLdh + 0.5 * H * GinvtLdh**2  # (*bshape, *, *shape)
            dr = dh + (tL @ dh.unsqueeze(-1)).squeeze(-1)  # (*bshape, *, *shape)

        elif output == "response" and self.mode == "matrix":
            if not isinstance(f, nn.Identity):
                raise ValueError("f must be nn.Identity when mode = 'matrix'.")

            W = self.forward(x, output="weight", ndim=ndim, to_dataframe=False)
            I = linalg.eye_like(W)  # (*bshape, *, *shape, *shape)
            dh = x.data[in_var]  # (*, *shape)
            dr = torch.linalg.solve(I - W, dh)  # (*bshape, *, *shape)

        else:
            if self.dense or len(self.prob_kernel.funcs) == 0:
                if self.diff_Gaussian_mask is not None: # Kris: difference of Gaussians
                    W = weights.discretize(
                        self.product_kernel, x.data[keys], diff_Gaussian_mask=self.diff_Gaussian_mask, ndim=ndim, dim=cdim
                    )  # (*bshape, *, *shape, *shape)
                    W2 = weights.discretize(
                        self.product_kernel2, x.data[keys], diff_Gaussian_mask=self.diff_Gaussian_mask2, ndim=ndim, dim=cdim
                    )  # (*bshape, *, *shape, *shape)
                    W = W - W2 # Kris: new

                else:
                    W = weights.discretize(
                        self.product_kernel, x.data[keys], ndim=ndim, dim=cdim
                    )  # (*bshape, *, *shape, *shape)

            else:
                # Note: this is non-differentiable.
                prob = weights.discretize(
                    self.prob_kernel, x.data[keys], ndim=ndim, dim=cdim, mul_dV=False
                )
                W = weights.discretize(
                    self.strength_kernel, x.data[keys], ndim=ndim, dim=cdim
                )  # (*bshape, *, *shape, *shape)

                if isinstance(prob, CirculantTensor):
                    prob = prob.dense()
                if isinstance(W, CirculantTensor):
                    W = W.dense()

                if (prob < -1e-5).any() or (prob > 1 + 1e-5).any():
                    raise ValueError("Connection probability must be in [0, 1].")
                prob = prob.clip(min=0.0, max=1.0)

                W = W * torch.bernoulli(prob)  # (*bshape, *, *shape, *shape)

            if self.N_synapses is not None:
                # Note: this is currently non-differentiable.
                mean_p = self.N_synapses / math.prod(x.shape[-ndim:])
                with random.set_seed(self.seed):
                    W = weights.sparsify(W, mean_p=mean_p, **self.sparsify_kwargs)
                logger.debug(
                    f"{W.count_nonzero().item()}/{W.numel()} = {W.count_nonzero().item()/W.numel()} "
                    "fraction of connected neurons"
                )

            if self.W_std > 0.0:
                with random.set_seed(self.seed):
                    W = weights.sample_log_normal(W, self.W_std)

            if output == "weight":
                if not to_dataframe:
                    return W

                dims = [(-ndim, None), (-ndim, None)]
                x = x.data[self.variables + ([mask_var] if mask_var in x else [])]
                x_post, x_pre = frame.meshgrid(x, x, dims=dims, sparse=True)

                # TODO: Think about how to make this more flexible, allowing for
                # arbitrary columns instead of hardcoding everything.
                x = {}
                for k in ["cell_type", "osi"]:
                    if k in self.variables:
                        x[f"presynaptic_{k}"] = x_pre[k]
                        x[f"postsynaptic_{k}"] = x_post[k]
                for k, v in [("space", "distance"), ("ori", "rel_ori")]:
                    if k in self.variables:
                        x[v] = functional.diff(x_post[k], x_pre[k]).norm(dim=-1)
                if mask_var in x_pre:
                    x[mask_var] = x_post[mask_var] & x_pre[mask_var]
                x = frame.ParameterFrame(x, ndim=x_pre.ndim)

                if isinstance(W, CirculantTensor):
                    W = W.dense()
                x = x.datailoc[(None,) * self.batch_ndim] | {"W": W}
                logger.debug(f"x:\n{x}")

                if mask_var in x:
                    x = x.iloc[x[mask_var]]
                    del x[mask_var]
                logger.debug(f"x:\n{x}")
                x = x.to_framelike(**framelike_kwargs)

                return x

            if ndim > 1 and not isinstance(W, CirculantTensor):
                raise NotImplementedError(
                    "Currently only supports 1D inputs for non-circulant tensors."
                )

            dh = x.data[in_var]  # (*, *shape)

            if "cell_type" in self.variables:
                tau = self.tau[x.data["cell_type"]]  # (*, *shape)
            else:
                tau = 1.0

            # print("print x here")
            # print(x.shape)
            # print(x["dh"][0, 6, :, :])
            # print(torch.max(x["dh"][0, 6, :, :]))

            dr = numerics.perturbed_steady_state(
                vf, W, f_, dh, tau=tau, **self.simulation_kwargs
            ).x  # (*bshape, *, *shape) # Kris: new

        if assert_finite and not dr.isfinite().all():
            N_finite = dr.isfinite().count_nonzero()
            N = dr.numel()
            raise exceptions.SimulationError(
                f"{N - N_finite}/{N} non-finite values (NaN, +inf, or -inf) "
                "detected in V1 response."
            )

        x = x.datailoc[(None,) * self.batch_ndim] | {out_var: dr}

        if mask_var in x:
            x = x.iloc[x[mask_var]]

        if to_dataframe:
            x = x.to_framelike(**framelike_kwargs)

        return x


def _allconst(x, dim, **kwargs):
    n = x.shape[dim]
    if n == 1:
        return True
    return torch.allclose(x.narrow(dim, 0, n - 1), x.narrow(dim, 1, n - 1), **kwargs)


def _cdim(x, ndim, **kwargs):
    cdim = []
    for dim in range(x.ndim - ndim, x.ndim):
        if all(
            _allconst(v, dim, **kwargs)
            or (
                (k in {"space", "ori"})
                and isinstance(v, PeriodicTensor)
                and _allconst(v.diff(dim=dim), dim, **kwargs)
                # note: this is currently not exhaustive since it does not check
                # that the circular difference between the first and last element is
                # the same as that between all other consecutive pairs of elements
            )
            for k, v in x._items()
        ):
            cdim.append(dim)
    return utils.normalize_dim(cdim, x.ndim, neg=True)
