import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        nargs="+",
        choices=["cpu", "cuda", "mps"],
        help="Device(s) to run tests on.",
    )


def pytest_make_parametrize_id(config, val, argname):
    if config.option.verbose >= 2:  # -vv or -vvv
        return f"{argname}={val}"
    return repr(val)  # the default


def pytest_configure(config):
    devices = config.getoption("--device")
    if devices is None:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            devices.append("mps")

    pytest.devices = devices
    pytest.non_mps_devices = [device for device in devices if device != "mps"]
