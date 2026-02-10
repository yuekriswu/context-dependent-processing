from collections.abc import Sequence, Callable, Iterable, Hashable
from numbers import Number
import logging
import copy

from torch import Tensor
import torch
import pandas as pd

from niarb import perturbation, special, random, utils, visual_stim
from niarb.neurons import sample as sample_neurons
from niarb.nn.modules import frame
from niarb.nn.modules.frame import ParameterFrame
from niarb.nn import functional as F
from niarb.tensors import categorical

logger = logging.getLogger()

FUNC_MAPPING: dict[str, Callable[..., Tensor]] = {
    "perturbation": perturbation.sample,
    "visual_stim": visual_stim.grating,
}


def _get_tags_and_configs(
    configs: Iterable[dict], mapping: dict[Hashable, dict] | None = None
) -> tuple[ParameterFrame, list[dict]]:
    if mapping is None:
        mapping = {}

    tags, new_configs = [], []
    for config in copy.deepcopy(configs):
        repeats = config.pop("repeats", 1)

        tag = {}
        for k, v in config.copy().items():
            if isinstance(v, Hashable) and v in mapping:
                tag[k] = config.pop(k)
                config.update(mapping[v])
            elif isinstance(v, (Number, str)):
                # Automatically add a tag if v is a number of string. Could also
                # consider adding tag if v is a Hashable, but let's be more restrictive
                # to ensure existing tests pass for now.
                tag[k] = v

        new_configs += [config] * repeats
        tags += [tag] * repeats

    tags = utils.transpose(tags, is_dict=(False, True))
    for k, v in tags.items():
        categories = tuple(dict.fromkeys(v))
        if all(isinstance(cat, Number) for cat in categories):
            # note that boolean values are also instances of Number
            tags[k] = torch.tensor(v)
        else:
            v = [categories.index(vi) for vi in v]
            tags[k] = categorical.tensor(v, categories=categories)
    tags = frame.ParameterFrame(tags, ndim=1)
    return tags, new_configs


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        neurons: ParameterFrame | dict,
        inputs: Tensor | dict | Sequence[dict],
        input_funcs: (
            str | Callable[..., Tensor] | Sequence[str | Callable[..., Tensor]]
        ) = "perturbation",
        tags=None,
        data: Sequence[pd.DataFrame] = (),
        N_instantiations: int = 1,
        window: float | Sequence[float] = torch.inf,
        abs_window: float | Sequence[float] = torch.inf,
        subsample: int | None = None,
        metrics: dict[str, Callable[[ParameterFrame], Tensor] | str] | None = None,
        psf: tuple[float, bool] | None = None,
        y: str = "dr",
        yerr: str = "dr_se",
        sample_target: bool = False,
        seed: int | None = None,
        target_seed: int | None = None,
        **forward_kwargs,
    ):
        if isinstance(neurons, ParameterFrame):
            if N_instantiations != 1:
                raise ValueError(
                    "N_instantiations must be 1 when both neurons "
                    f"have already been instantiated, but {N_instantiations=}."
                )

            if isinstance(inputs, Tensor):
                if inputs.ndim != neurons.ndim + 1:
                    raise ValueError(
                        "inputs must have one more dimension than neurons, "
                        f"but {inputs.ndim=}, {neurons.ndim=}."
                    )

                if seed is not None:
                    raise ValueError(
                        "seed must be None when both neurons and inputs "
                        f"have already been instantiated, but {seed=}."
                    )

        if metrics is None:
            metrics = {
                k: k for df in data for k in df.columns if hasattr(perturbation, k)
            }

        metrics = metrics.copy()
        for k, v in metrics.items():
            kwargs = {}
            if not callable(v):
                if v != "min_distance_to_ensemble":
                    kwargs["keepdim"] = True
                v = getattr(perturbation, v)
            metrics[k] = (v, kwargs)

        self.seed = seed
        if len(data) == 0:
            self.y = None
        else:
            # note: to_numpy() call is necessary when Series index is not trivial
            self.y = torch.cat([torch.tensor(df[y].to_numpy()).float() for df in data])

            self.yerr = [
                (
                    torch.tensor(df[yerr].to_numpy()).float()
                    if yerr in df.columns
                    else torch.zeros(len(df))
                )
                for df in data
            ]
            self.yerr = torch.cat(self.yerr)

        self.neurons = neurons
        if isinstance(inputs, Tensor):
            self.tags, self.inputs = tags, inputs
        else:
            if isinstance(inputs, dict):
                inputs = [inputs]
            self.tags, input_configs = [], []
            for input in inputs:
                tag, input_config = _get_tags_and_configs(**input)
                self.tags.append(tag)
                input_configs.append(input_config)
            self.tags = frame.concat(self.tags)

            if isinstance(input_funcs, str) or callable(input_funcs):
                input_funcs = [input_funcs] * len(input_configs)

            if len(input_funcs) != len(input_configs):
                raise ValueError(
                    "input_funcs must have the same length as input_configs, "
                    f"but {len(input_funcs)=}, {len(input_configs)=}."
                )

            if not all(
                callable(func) or func in {"perturbation", "visual_stim"}
                for func in input_funcs
            ):
                raise ValueError(
                    "input_funcs must be either callable or 'perturbation' or 'visual_stim', "
                    f"but {input_funcs=}"
                )

            if psf is not None and any(func != "perturbation" for func in input_funcs):
                raise ValueError(
                    "psf can only be applied to perturbation inputs, "
                    f"but {input_funcs=}, {psf=}"
                )

            input_funcs = (
                func if callable(func) else FUNC_MAPPING[func] for func in input_funcs
            )
            self.inputs = list(zip(input_configs, input_funcs, strict=True))

        self.N_instantiations = N_instantiations
        self.window = window
        self.abs_window = abs_window
        self.subsample = subsample
        self.metrics = metrics
        self.psf = psf
        self.sample_target = sample_target
        self.target_seed = target_seed

        if isinstance(neurons, ParameterFrame):
            model_kwargs = {"ndim": neurons.ndim}
        else:
            model_kwargs = {"ndim": 1, "check_circulant": False}
        model_kwargs |= forward_kwargs.pop("model_kwargs", {})
        self.kwargs = {"model_kwargs": model_kwargs} | forward_kwargs

        self.reset_targets()

    def __len__(self):
        return self.N_instantiations

    def reset_targets(self):
        if self.sample_target:
            if self.y is None:
                raise ValueError(
                    "sample_target cannot be True when target data is not provided."
                )
            with random.set_seed(self.target_seed):
                self.sampled_y = self.y + self.yerr * torch.randn_like(self.y)

    def __getitem__(self, index):
        seed = self.seed + index if self.seed is not None else None

        if isinstance(self.neurons, ParameterFrame):
            x = self.neurons
        else:
            with random.set_seed(seed):
                x = sample_neurons(**self.neurons)  # (*shape)

        if isinstance(self.inputs, Tensor):
            dh = self.inputs # self.inputs.clone() # self.inputs #
        else:
            dh = []
            with random.set_seed(seed):
                for confs, func in self.inputs:
                    for conf in confs:
                        dh.append(func(x, **conf))
            dh = torch.stack(dh)

        x = x.unsqueeze(0)  # (1, *shape)
        x["dh"] = dh  # (N_stims, *shape)
        x["mask"] = torch.ones_like(x["dh"], dtype=torch.bool)

        if "space" in x:
            if self.window != torch.inf:

                def func(xi):
                    cmean = xi["space"][xi["dh"] != 0].cmean()  # (D,)
                    return cmean[(None,) * xi.ndim]  # (1, ..., 1, D)

                loc = x.apply(func, dim=range(1, x.ndim))
                x["mask"] &= (
                    special.uniform(self.window, F.diff(x["space"], loc))
                    .prod(dim=-1)
                    .bool()
                )

            if self.abs_window != torch.inf:
                x["mask"] &= (
                    special.uniform(self.abs_window, x["space"]).prod(dim=-1).bool()
                )

        if self.subsample is not None:
            for i, m in enumerate(x["mask"]):
                with random.set_seed(seed):
                    x["mask"][i] = perturbation.categorical_sample(
                        m.float(), self.subsample
                    ).bool()

        if x["mask"].all():
            del x["mask"]  # no need to store mask if all True

        for k, (func, kwargs) in self.metrics.items():
            x[k] = x.apply(func, dim=range(1, x.ndim), **kwargs)

        # Convolve perturbations with Gaussian PSF
        # Note that this must be done after calculating metrics
        if self.psf is not None:
            x["dh"] = x.apply(
                perturbation.convolve, dim=range(1, x.ndim), args=self.psf
            )

        if len(self.tags) > 0:
            x = frame.concat(
                [x, self.tags.datailoc[(...,) + (None,) * (x.ndim - 1)]], dim=-1
            )

        if self.y is None:
            return x, self.kwargs

        y = self.sampled_y if self.sample_target else self.y
        return x, y, self.kwargs


def collate_fn(batch):
    out = tuple(zip(*batch))
    x = frame.stack(out[0])
    out = (x,) + tuple(v[0] for v in out[1:])
    return out
