from pathlib import Path
import logging
from collections.abc import Sequence
import copy
import pprint
import traceback

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from niarb import nn, optimize, exceptions, random
from niarb.dataset import Dataset, collate_fn

from niarb.optimize import constraint

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_parser_arguments(parser):
    parser.add_argument(
        "-N", type=int, help="number of optimization trials (default: 10)"
    )
    parser.add_argument("--out", "-o", type=Path, help="path to output directory")
    parser.add_argument(
        "--ignore-errors", action="store_true", help="ignore errors during optimization"
    )

    return parser


def run_from_conf_args(conf, args):
    conf = conf.copy()
    for k in ["N", "out", "ignore_errors", "progress"]:
        if v := getattr(args, k):
            conf[k] = v

    logger.debug(f"config:\n{pprint.pformat(conf)}")
    run(**conf)


def run(
    data: Sequence[pd.DataFrame],
    dataset: Dataset | dict,
    pipeline: nn.Pipeline | dict,
    optimizer: dict | None = None,
    constraints: Sequence[constraint.Constraint] = (),
    validation_dataset: Dataset | dict | None = None,
    validation_pipeline: nn.Pipeline | dict | None = None,
    init_state_dict: dict[str, Tensor] | str | Path | None = None,
    seed: int | None = None,
    loss_threshold: float = 0.75,
    weighted_loss: bool = False,
    equal_loss: bool | Sequence[float] = False,
    normalized_loss: bool = True,
    dtype: str | torch.dtype | None = None,
    validation_dtype: str | torch.dtype | None = None,
    validation_batch_size: int | None = None,
    N: int = 10,
    out: Path | str | None = None,
    ignore_errors: bool = False,
    progress: bool = False,
) -> tuple[list[float], list[torch.Tensor]]:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if isinstance(validation_dtype, str):
        validation_dtype = getattr(torch, validation_dtype)

    validate = (
        (validation_dataset is not None)
        or (validation_pipeline is not None)
        or (dtype is not validation_dtype)
    )

    # initialize dataset
    if isinstance(dataset, dict):
        dataset = Dataset(**dataset, data=data)
    dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)

    # initialize validation dataset
    if isinstance(validation_dataset, dict):
        validation_dataset = Dataset(**validation_dataset, data=data)
    elif validation_dataset is None:
        validation_dataset = copy.deepcopy(dataset)
    if validation_batch_size is None:
        validation_batch_size = len(validation_dataset)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=validation_batch_size, collate_fn=collate_fn
    )

    # initialize fitting pipeline
    if isinstance(pipeline, dict):
        pipeline = nn.Pipeline(data=data, **pipeline)
    pipeline.to(device, dtype=dtype)
    logger.debug(str(pipeline))
    logger.debug(pprint.pformat(nn.param_dict(pipeline)))

    # initialize validation pipeline
    if isinstance(validation_pipeline, dict):
        validation_pipeline = nn.Pipeline(data=data, **validation_pipeline)
    elif validation_pipeline is None:
        validation_pipeline = copy.deepcopy(pipeline)
    validation_pipeline.to(device, dtype=validation_dtype)

    # initialize fitting criterion
    w = None
    if weighted_loss and equal_loss:
        raise ValueError("Only one of weighted_loss and equal_loss can be True.")
    if weighted_loss:
        if not all("dr_se" in df for df in data):
            raise ValueError(
                "If weighted_loss is True, data must contain standard errors."
            )
        # loss is weighted inversely proportional to standard error of data
        w = [torch.tensor(df["dr_se"].to_numpy()).float() for df in data]
        w = torch.cat(w) ** -2

    if equal_loss:
        if isinstance(equal_loss, bool):
            equal_loss = [1.0] * len(data)
        if len(equal_loss) != len(data):
            raise ValueError(f"{len(equal_loss)=}, but {len(data)=}.")

        # weight loss from each dataset roughly equally by scaling weighing
        # the loss of each dataset inversely proportional to the variance of the
        # dataset and the number of data points in the dataset
        alen = sum(len(df) for df in data) / len(data)
        w = [torch.full((len(df),), alen / (len(df) * df["dr"].var())) for df in data]
        w = torch.cat([scale * wi for scale, wi in zip(equal_loss, w)])
    criterion = nn.WeightedNormalizedLoss(w=w, normalized=normalized_loss)
    criterion.to(device, dtype=dtype)

    # initialize optimizer
    if optimizer is None:
        optimizer = {}

    optimizer = optimize.Optimizer(pipeline, criterion, constraints=constraints, **optimizer)

    # load initial state_dict
    if isinstance(init_state_dict, (str, Path)):
        init_state_dict = torch.load(init_state_dict, map_location="cpu")

    # initialize output directory
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=N, desc="fit", disable=not progress)
    N_successes, losses, state_dicts = 0, [], []
    while N_successes < N:
        # sample x and y
        dataset.reset_targets()
        x, y, kwargs = next(iter(dataloader))

        x, y = x.to(device, dtype=dtype), y.to(device, dtype=dtype)
        logger.debug(str(x))

        # initialize pipeline parameters
        logger.info(f"Initializing pipeline with {seed=}")
        with random.set_seed(seed):
            pipeline.apply(nn.reset_parameters)

        if init_state_dict:
            logger.info("Loading state_dict...")
            nn.load_state_dict(pipeline, init_state_dict, strict=False)

        pipeline.zero_grad()

        # optimize pipeline and handle exceptions
        try:
            success, loss = optimizer(x, y, **kwargs)

            if normalized_loss:
                with torch.inference_mode():
                    y_pred = pipeline(x, **kwargs)
        except (exceptions.OptimizationError, exceptions.SimulationError) as err:
            logger.info(str(err))
            success, loss = False, np.inf
        except Exception as err:
            if not ignore_errors:
                raise
            logger.error(traceback.format_exc())
            success, loss = False, np.inf

        validation_pipeline.load_state_dict(pipeline.state_dict())
        if success and loss < loss_threshold and normalized_loss:
            validation_pipeline.scale_parameters((y.norm() / y_pred.norm()).item())

        if success and loss < loss_threshold and validate:
            # optionally 'validate' fitted parameters on an altered (presumably more
            # accuarate but computationally expensive) pipeline and validation dataset
            logger.info("Validating model...")
            try:
                y_pred = []
                for i, (x, _, kwargs) in enumerate(validation_dataloader):
                    logger.info(f"Running validation batch {i}...")
                    x = x.to(device, dtype=validation_dtype)
                    with torch.inference_mode():
                        y_pred.append(validation_pipeline(x, **kwargs))
                y_pred = torch.stack(y_pred).mean(dim=0)
                loss = criterion(y_pred, y).item()
            except exceptions.SimulationError as err:
                logger.info(str(err))
                success, loss = False, np.inf
            except Exception as err:
                if not ignore_errors:
                    raise
                logger.error(traceback.format_exc())
                success, loss = False, np.inf
            logger.info(f"Validation loss: {loss}")

        # save loss and state_dict if successful and loss is below threshold
        if success and loss < loss_threshold:
            state_dict = nn.state_dict(validation_pipeline)
            if out:
                torch.save(state_dict, out / f"{loss:.10e}.pt")

            # Note: deepcopy is necessary since module.state_dict() only
            # returns a shallow copy
            state_dicts.append(copy.deepcopy(state_dict))
            losses.append(loss)

            pbar.update()
            N_successes += 1

        # increment random seed whether successful or not
        seed = seed + 1 if seed is not None else None

    return losses, state_dicts
