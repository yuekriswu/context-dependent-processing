from typing import Self

import torch

__all__ = ["ParameterDict", "FanOut", "NamedSequential"]


class FanOut(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules = modules

    def forward(self, x):
        return [module(x) for module in self.modules]


class NamedSequential(torch.nn.Module):
    _modules: dict[str, torch.nn.Module]  # type: ignore[assignment]

    def __init__(self, modules: dict[str, torch.nn.Module]):
        super().__init__()
        if len(modules) == 0:
            raise ValueError("`modules` must contain at least one module.")

        for name, module in modules.items():
            self.add_module(name, module)
        self._allowed_keywords = {f"{name}_kwargs" for name in modules.keys()}

    def forward(self, input, **kwargs):

        if any(k not in self._allowed_keywords for k in kwargs.keys()):
            raise ValueError(
                f"unexpected kwargs: {set(kwargs.keys()) - self._allowed_keywords}"
            )

        for name, module in self._modules.items():
            input = module(input, **kwargs.get(f"{name}_kwargs", {}))

        return input

    def __getitem__(self, key: str | int | slice) -> torch.nn.Module | Self:
        if isinstance(key, str):
            out = self._modules[key]

        elif isinstance(key, int):
            out = list(self._modules.values())[key]

        elif isinstance(key, slice):
            names = list(self._modules.keys())
            start = names.index(key.start) if isinstance(key.start, str) else key.start
            # stop is inclusive if it is a string, consistent with pandas .loc slicing
            stop = names.index(key.stop) + 1 if isinstance(key.stop, str) else key.stop
            key = slice(start, stop, key.step)

            out = NamedSequential(dict(list(self._modules.items())[key]))
        else:
            raise TypeError(f"key must be str, int, or slice, but {type(key)=}.")

        return out


class ParameterDict(torch.nn.ParameterDict):
    """
    Stores Tensors that are not Parameters using
    module.register_buffer(attr, tensor, persistent=False)
    instead of as Parameters.
    """

    def __setitem__(self, key, value):
        self._keys[key] = None
        attr = self._key_to_attr(key)
        if isinstance(value, torch.Tensor) and not isinstance(
            value, torch.nn.Parameter
        ):
            self.register_buffer(attr, value, persistent=False)
        setattr(self, attr, value)
