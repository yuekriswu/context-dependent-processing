import functools
import logging
import time
from collections.abc import Sequence

import torch
import numpy as np
import scipy.optimize as sp_opt
import hyclib as lib

from niarb import exceptions
from niarb.nn.parameter import Parameter
from .constraint import Constraint

logger = logging.getLogger(__name__)


def _ndarray_to_tuple(f):
    @functools.wraps(f)
    def wrapped_f(x):
        return f(tuple(x.tolist()))

    return wrapped_f


class Optimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        regularizer: torch.nn.Module | None = None,
        method: str | None = None,
        constraints: Sequence[Constraint] = (),
        tol: float | None = None,
        options: dict | None = None,
        use_autograd: bool = True,
        timeout: float = 1000.0,
        cache_size: int = 128,
    ):
        """Initialize optimizer.

        Args:
            model: Model to optimize.
            criterion: Loss function.
            regularizer (optional): Regularization term.
            method (optional): 'method' argument of scipy.optimize.minimize.
            constraints (optional): Constraints.
            tol (optional): 'tol' argument of scipy.optimize.minimize.
            options (optional): 'options' argument of scipy.optimize.minimize.
            use_autograd (optional): Whether to use autograd.
            timeout (optional): Time (in seconds) before optimization times out.
            cache_size (optional): Cache size (in number of calls).

        """
        self.model = model
        self.criterion = criterion
        self.regularizer = regularizer
        self.method = method
        self.tol = tol
        self.options = options
        self.use_autograd = use_autograd
        self.timeout = timeout
        self.cache_size = cache_size

        self.constraints = []
        self.constraint_caches = []

        if constraints:
            for c in constraints:
                if not isinstance(c, Constraint):
                    raise TypeError(
                        f"constraint must be a list of Constraint objects, but got {type(c)=}."
                    )

                constraint, constraint_cache = self.make_constraint(c)
                self.constraints.append(constraint)
                self.constraint_caches.append(constraint_cache)
        self.constraint_names = [repr(constraint) for constraint in constraints]

        if len(set(self.constraint_names)) != len(self.constraint_names):
            raise ValueError(
                f"constraints must be unique, but got {self.constraint_names}."
            )

    @property
    def params(self):
        """
        Returns a 1D-tensor of optimizable parameters
        """
        params = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter) and torch.any(param.requires_optim):
                params.append(param[param.requires_optim])
        return torch.cat(params)

    @property
    def params_grad(self):
        """
        Returns a 1D-tensor of optimizable parameters gradients
        """
        params_grad = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter) and torch.any(param.requires_optim):
                if (
                    param.grad is None
                ):  # parameter has no gradient, probably because the objective function is independent of the parameter
                    params_grad.append(
                        torch.zeros(
                            len(param.requires_optim.nonzero()), device=param.device
                        )
                    )
                else:
                    params_grad.append(param.grad[param.requires_optim])
        return torch.cat(params_grad)

    @params.setter
    def params(self, params):
        """
        Updates the optimizable parameters with params, which is a 1D-tensor.
        """
        assert len(params) == len(self.params)

        i = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if isinstance(param, Parameter) and torch.any(param.requires_optim):
                    N = param[param.requires_optim].numel()
                    param[param.requires_optim] = torch.tensor(
                        params[i : i + N], dtype=param.dtype, device=param.device
                    )
                    i += N

    @property
    def bounds(self):
        bounds = []
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter):
                bounds += param.bounds[param.requires_optim].tolist()
        return bounds

    def state_dict(self):
        state_dict = {}
        for name, param in self.model.named_parameters():
            if isinstance(param, Parameter):
                state_dict[name] = param.data
        return state_dict

    def make_compute_constraint_quantities(self, c):

        @_ndarray_to_tuple
        @functools.lru_cache(maxsize=self.cache_size)
        def compute_constraint_quantities(params):
            with torch.set_grad_enabled(self.use_autograd):
                self.params = params

                constraint_val = c(self.model)

                if self.use_autograd:
                    self.model.zero_grad()
                    constraint_val.backward()

                    grad = self.params_grad.tolist()

                    return {"constraint_val": constraint_val.item(), "grad": grad}

                return {"constraint_val": constraint_val.item()}
        return compute_constraint_quantities

    def make_constraint(self, c):
        compute_constraint_quantities = self.make_compute_constraint_quantities(c)
        cached_func = compute_constraint_quantities.__wrapped__

        constraint_func = lambda params: compute_constraint_quantities(params)[
            "constraint_val"
        ]
        constraint_type = "eq" if c.is_equality else "ineq"

        if self.use_autograd:
            constraint_jac = lambda params: compute_constraint_quantities(params)[
                "grad"
            ]
            return {
                "type": constraint_type,
                "fun": constraint_func,
                "jac": constraint_jac,
            }, cached_func

        return {"type": constraint_type, "fun": constraint_func}, cached_func

    def make_compute_quantities(self, x, y, **kwargs):

        def compute_quantities(params, x, y, **kwargs):
            with torch.set_grad_enabled(self.use_autograd):
                logger.debug(f"Computing quantities with {params=}...")
                self.params = params

                try:
                    y_pred = self.model(x, **kwargs)

                except exceptions.SimulationError as err:
                    logger.debug(
                        f"Exception encountered when running model with params {self.params}: {err}"
                    )
                    loss = torch.tensor(np.inf)
                else:
                    loss = self.criterion(y_pred, y)

                pure_loss_item = loss.item()

                if self.regularizer is not None:
                    loss += self.regularizer(self.model)

                loss_item = loss.item()

                if self.use_autograd:
                    if loss.grad_fn is None:  # non-differentiable
                        grad = [np.nan for _ in range(len(params))]

                    else:
                        self.model.zero_grad()
                        loss.backward() # loss.backward(retain_graph=True)

                        grad = self.params_grad
                        if (~grad.isfinite()).any():
                            raise exceptions.OptimizationError(
                                f"Gradient contains non-finite values: {grad.tolist()}"
                            )
                        grad = grad.tolist()

                    return {
                        "pure_loss": pure_loss_item,
                        "loss": loss_item,
                        "grad": grad,
                    }

                return {"pure_loss": pure_loss_item, "loss": loss_item}

        func = functools.partial(compute_quantities, x=x, y=y, **kwargs)
        func = functools.lru_cache(maxsize=self.cache_size)(func)
        return _ndarray_to_tuple(func)

    def make_loss_func(self, compute_quantities):
        def loss_func(params):
            return compute_quantities(params)["pure_loss"]

        return loss_func

    def make_minimizer_func(self, compute_quantities):
        def minimizer_func(params):
            quantities = compute_quantities(params)
            if self.use_autograd:
                return quantities["loss"], quantities["grad"]
            return quantities["loss"]

        return minimizer_func

    def make_callback(self, loss_func, cache_func, hist, start_time):

        def callback(params):
            if (time_taken := time.time() - start_time) > self.timeout:
                raise exceptions.OptimizationError(
                    f"self.optimize has been running for {time_taken} seconds and timed out."
                )

            # Compute loss, hopefully result is in cache
            loss_item = loss_func(params)

            # Compute constraints, hopefully results are in cache
            all_satisfied = True
            constraint_values_dict = {}

            for constraint_name, constraint in zip(
                self.constraint_names, self.constraints
            ):
                val = constraint["fun"](params)
                if constraint["type"] == "eq":
                    satisfied = val == 0
                elif constraint["type"] == "ineq":
                    satisfied = val >= 0
                else:
                    raise RuntimeError()

                constraint_values_dict[constraint_name] = val
                all_satisfied = all_satisfied and satisfied

            hist["loss"].append(loss_item)
            hist["satisfied"].append(all_satisfied)
            hist["params"].append(params)

            logger.info(f"Loss: {loss_item}")
            logger.debug(f"Constraints:\n{lib.pprint.pformat(constraint_values_dict)}")
            logger.debug(f"Loss func cache info: {cache_func.cache_info()}")
            for constraint_name, constraint_cache in zip(
                self.constraint_names, self.constraint_caches
            ):
                logger.debug(
                    f"{constraint_name} cache info: {constraint_cache.cache_info()}"
                )
            logger.debug(f"self.params={self.params.detach().cpu()}")

        return callback

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, **kwargs
    ) -> tuple[bool, float]:
        logger.info("Started optimizing...")
        logger.debug(lib.pprint.pformat(self.state_dict()))

        hist = {"loss": [], "satisfied": [], "params": []}
        start_time = time.time()

        compute_quantities = self.make_compute_quantities(x, y, **kwargs)
        loss_func = self.make_loss_func(compute_quantities)
        minimizer_func = self.make_minimizer_func(compute_quantities)
        callback = self.make_callback(
            loss_func, compute_quantities.__wrapped__, hist, start_time
        )

        try:
            result = sp_opt.minimize(
                minimizer_func,
                self.params.tolist(),
                method=self.method,
                jac=self.use_autograd,
                bounds=self.bounds,
                constraints=self.constraints,
                tol=self.tol,
                callback=callback,
                options=self.options,
            )
            if not result.success:
                raise exceptions.OptimizationError(result.message)

        except exceptions.OptimizationError as err:
            logger.info(f"Optimization failed due to exception: {err}")

            logger.debug(f"Loss hist: {hist['loss']}")
            logger.debug(f"Satisfied hist: {hist['satisfied']}")

            if len(hist["loss"]) == 0:
                logger.info("No result returned since len(hist['loss']) = 0.")
                return False, np.inf

            loss_hist = np.array(hist["loss"])
            satisfied_hist = np.array(hist["satisfied"])
            params_hist = np.array(hist["params"])

            satisfied_loss_hist = loss_hist[
                satisfied_hist
            ]  # get only the losses where constraint is satisfied
            satisfied_params_hist = params_hist[satisfied_hist]

            logger.debug(f"Satisfied loss hist: {satisfied_loss_hist}")

            if len(satisfied_loss_hist) == 0:
                logger.info("No result returned since len(satisfied_loss_hist) = 0.")
                return False, np.inf

            idx = np.argmin(satisfied_loss_hist)

            self.params = satisfied_params_hist[idx]
            loss = satisfied_loss_hist[idx]

            logger.info(f"Returning result during optimization. Loss: {loss}.")
            logger.debug(self.params.detach().cpu())

            return True, loss

        self.params = result.x
        loss = loss_func(result.x)

        logger.info(f"Finished optimization successfully. Loss: {loss}")
        logger.debug(self.params.detach().cpu())

        return True, loss
