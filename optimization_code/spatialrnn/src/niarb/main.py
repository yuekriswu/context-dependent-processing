import logging
import argparse
import pprint
from pathlib import Path

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from niarb import io
from .cli import fit, plot, run

logger = logging.getLogger(__name__)


class ConstWithMultiArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        obj = values if values else self.const if option_string else self.default
        setattr(namespace, self.dest, obj)


def add_shared_arguments(parser):
    parser.add_argument("confs", type=Path, nargs="*", help="configuration filename(s)")
    parser.add_argument(
        "--progress", action="store_true", help="display tqdm progress bar"
    )
    parser.add_argument(
        "--matmul-precision",
        "--mp",
        dest="matmul_precision",
        choices=["high", "medium"],
        help=(
            "Set precision of float32 matmul operation. "
            "If not provided, defaults to 'highest'."
        ),
    )
    parser.add_argument(
        "--log-debug",
        "--ldebug",
        dest="log_debug",
        nargs="*",
        action=ConstWithMultiArgs,
        const=[None],
        default=[],
        help=(
            "Set log level of specified modules to DEBUG. If no argument is provided, "
            "set log level of the root logger to DEBUG."
        ),
    )
    parser.add_argument(
        "--log-info",
        "--linfo",
        dest="log_info",
        nargs="*",
        action=ConstWithMultiArgs,
        const=[None],
        default=[],
        help=(
            "Set log level of specified modules to INFO. If no argument is provided, "
            "set log level of the root logger to INFO."
        ),
    )
    parser.add_argument(
        "--log-warning",
        "--lwarn",
        dest="log_warning",
        nargs="*",
        action=ConstWithMultiArgs,
        const=[None],
        default=[],
        help=(
            "Set log level of specified modules to WARNING. If no argument is provided, "
            "set log level of the root logger to WARNING."
        ),
    )
    parser.add_argument(
        "--log-error",
        "--lerr",
        dest="log_error",
        nargs="*",
        action=ConstWithMultiArgs,
        const=[None],
        default=[],
        help=(
            "Set log level of specified modules to ERROR. If no argument is provided, "
            "set log level of the root logger to ERROR."
        ),
    )
    return parser


def main():
    parser = argparse.ArgumentParser("niarb")
    subparsers = parser.add_subparsers()
    for name, module in {"fit": fit, "plot": plot, "run": run}.items():
        subparser = subparsers.add_parser(name)
        subparser = module.add_parser_arguments(subparser)
        subparser = add_shared_arguments(subparser)
        subparser.set_defaults(func=module.run_from_conf_args)

    args = parser.parse_args()

    # configure logging
    logging.basicConfig(format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")
    logging.captureWarnings(True)
    for module in args.log_debug:
        logging.getLogger(module).setLevel(logging.DEBUG)
    for module in args.log_info:
        logging.getLogger(module).setLevel(logging.INFO)
    for module in args.log_warning:
        logging.getLogger(module).setLevel(logging.WARNING)
    for module in args.log_error:
        logging.getLogger(module).setLevel(logging.ERROR)

    logger.debug(f"args:\n{pprint.pformat(vars(args))}")

    # set matmul precision, highest by default
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
        logger.info(f"float32 matmul precision set to {args.matmul_precision}")

    # resolve Unix-style filename patterns
    paths = [path for conf in args.confs for path in Path(".").glob(str(conf))]
    logger.debug(f"{paths=}")

    # parse config files. If the top-level object is a list, flatten it.
    confs = []
    for path in paths:
        conf = io.load_config(path)
        if isinstance(conf, list):
            confs += [(f"{path.stem}[{i}]", c) for i, c in enumerate(conf)]
        else:
            confs.append((path.stem, conf))

    disable = len(confs) == 1 or not args.progress
    with logging_redirect_tqdm():
        for name, conf in tqdm(confs, desc="conf", disable=disable):
            logger.info(f"Running config '{name}'...")
            logger.debug(f"config:\n{pprint.pformat(conf)}")

            args.func(conf, args)
