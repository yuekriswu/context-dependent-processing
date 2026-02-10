import logging
import pprint
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pandas import DataFrame
import pandas as pd
from tqdm import tqdm

from niarb import nn, exceptions
from niarb.dataset import Dataset, collate_fn

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_parser_arguments(parser):
    parser.add_argument("--out", "-o", type=Path, help="output filename")

    return parser


def run_from_conf_args(conf, args):
    conf = conf.copy()
    for k in ["out"]:
        if v := getattr(args, k):
            conf[k] = v

    logger.debug(f"config:\n{pprint.pformat(conf)}")
    run(**conf)


def run(
    dataset: Dataset | dict,
    pipeline: nn.Pipeline | dict,
    state_dict: dict[str, Tensor] | str | Path,
    dtype: torch.dtype | str | None = None,
    batch_size: int | None = None,
    out: Path | str | None = None,
    progress: bool = False,
) -> DataFrame | Tensor:
    # handle alternative input types
    if isinstance(state_dict, (str, Path)):
        state_dict = torch.load(state_dict, map_location="cpu", weights_only=True)

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    # initialize dataset
    if isinstance(dataset, dict):
        dataset = Dataset(**dataset)

    if batch_size is None:
        batch_size = len(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # initialize pipeline
    if isinstance(pipeline, dict):
        pipeline = nn.Pipeline(**pipeline)
    logger.debug(f"pipeline:\n{pipeline}")

    # load state dict
    logger.debug(f"state dict:\n{pprint.pformat(state_dict)}")
    nn.load_state_dict(pipeline, state_dict)
    pipeline.to(device, dtype=dtype)

    # run model
    logger.info("Running model...")
    rets = []
    for x, kwargs in tqdm(dataloader, desc="batch", disable=not progress):
        x = x.to(device, dtype=dtype)
        logger.debug(f"x:\n{x}")

        try:
            with torch.inference_mode():
                ret = pipeline(x, **kwargs)
        except exceptions.SimulationError as err:
            logger.warning(f"Simulation failed: {err}.")
        else:
            rets.append(ret if isinstance(ret, Tensor) else ret.to_pandas())

    if all(isinstance(ret, Tensor) for ret in rets):
        ret = torch.cat(rets)
    else:
        ret = pd.concat(rets)

    # save output
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(ret, Tensor):
            torch.save(ret, out)
        else:
            ret.to_pickle(out)

    return ret
