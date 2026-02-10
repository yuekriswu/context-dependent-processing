from collections.abc import Iterable, Hashable, Sequence
import copy

import torch
import numpy as np
import pandas as pd
import hyclib as lib

from .containers import ParameterDict
from niarb.tensors.categorical import CategoricalTensor


class ParameterFrame(ParameterDict):
    """
    ParameterDict with tensors that are at least n-dimensional
    and whose first n dimensions are broadcastable with each other.
    Indexing returns the broadcasted tensors, althought the original
    tensors can be recovered with the indexer .data.
    The iterators .values() and .items() return broadcasted tensors,
    while ._values() and ._items() return the original tensors.
    Under the hood, this is achieved by storing tensors in their
    original form, only broadcasting them when they are returned.
    Every method should work the same as pd.DataFrame, except for
    __len__, which returns the number of columns, consistent with
    the __len__ method of ParameterDict.
    """

    def __init__(self, parameters=None, ndim=1):
        if not isinstance(ndim, int):
            raise TypeError(f"ndim must be an int, but {type(ndim)=}.")

        if ndim < 0:
            raise ValueError(f"ndim must be non-negative, but {ndim=}.")

        super().__init__()
        self._ndim = ndim

        if parameters is not None:
            self.update(parameters)

    def __setitem__(self, key, value):
        if value.ndim < self.ndim:
            raise ValueError(
                f"""value must be at least {self.ndim} dimensional,
                but got {value.ndim=}."""
            )

        if len(self._keys) != 0:
            try:
                torch.broadcast_shapes(self.shape, value.shape[: self.ndim])
            except RuntimeError as err:
                raise ValueError(
                    f"Unable to set ParameterFrame item due to shape mismatch: {self.shape=}, but {value.shape[:self.ndim]=}."
                ) from err

        super().__setitem__(key, value)

    def __getitem__(self, key):
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            return ParameterFrame({k: self[k] for k in key}, ndim=self.ndim)

        if (
            isinstance(key, torch.Tensor)
            and key.shape == self.shape
            and key.dtype == torch.bool
        ):
            return self.iloc[key]

        value = super().__getitem__(key)
        return value.broadcast_to((*self.shape, *value.shape[self.ndim :]))

    def _values(self):
        return (self.data[k] for k in self._keys)

    def _items(self):
        return ((k, self.data[k]) for k in self._keys)

    @property
    def iloc(self):
        return ILocIndexer(self)

    @property
    def data(self):
        return DataIndexer(self)

    @property
    def datailoc(self):
        return ILocIndexer(self, safe=False)  # used for indexing the underlying tensors

    @property
    def device(self):
        if len(self) > 0 and lib.itertools.isconst(v.device for v in self._values()):
            return next(iter(self._values())).device

        raise ValueError(
            "ParameterFrame must have at least one tensor and all tensors must be on the same device."
        )

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return torch.broadcast_shapes(*[v.shape[: self.ndim] for v in self._values()])

    def ndims(self):
        yield from (v.ndim - self.ndim for v in self._values())

    def shapes(self):
        yield from (v.shape[self.ndim :] for v in self._values())

    def rename(self, columns, copy=True):
        df = self.copy() if copy else self
        for k, v in columns.items():
            df[v] = df[k]
            del df[k]
        return df

    def drop(self, labels=None, *, dim=0, index=None, columns=None):
        if sum(v is not None for v in [labels, index, columns]) != 1:
            raise ValueError(
                "Exactly one of {labels, index, columns} must be provided."
            )

        if dim not in {0, 1, "index", "columns"}:
            raise ValueError(
                f"dim must be one of {0, 1, 'index', 'columns'}, but got {dim}."
            )

        if index is not None:
            labels = index
            dim = 0

        if columns is not None:
            labels = columns
            dim = 1

        if not isinstance(labels, list):
            labels = [labels]

        if dim == 0:
            raise NotImplementedError()
        else:
            df = self.copy()
            for label in labels:
                del df[label]
            return df

    def broadcast_to(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        elif not all(isinstance(s, int) for s in shape):
            types = [type(s) for s in shape]
            raise TypeError(
                f"input must be either a sequence of ints or a tuple, but got {types=}."
            )
        return ParameterFrame(
            {
                k: v.broadcast_to((*shape, *v.shape[self.ndim :]))
                for k, v in self._items()
            },
            ndim=len(shape),
        )

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        elif not all(isinstance(s, int) for s in shape):
            types = [type(s) for s in shape]
            raise TypeError(
                f"shape must be either a sequence of ints or a tuple, but got {types=}."
            )
        return ParameterFrame(
            {k: v.reshape(*shape, *v.shape[self.ndim :]) for k, v in self.items()},
            ndim=len(shape),
        )

    def squeeze(self, dim=None):
        if dim is None:
            dim = tuple(range(self.ndim))
        elif isinstance(dim, int):
            dim = (dim,)

        dim = tuple(d % self.ndim for d in dim if self.shape[d] == 1)
        return ParameterFrame(
            {k: v.squeeze(dim) for k, v in self._items()}, ndim=self.ndim - len(dim)
        )

    def unsqueeze(self, dim):
        dim = dim % (self.ndim + 1)
        return ParameterFrame(
            {k: v.unsqueeze(dim) for k, v in self._items()}, ndim=self.ndim + 1
        )

    def movedim(self, source, destination):
        if isinstance(source, int):
            source = (source,)

        if isinstance(destination, int):
            destination = (destination,)

        source = tuple(d % self.ndim for d in source)
        destination = tuple(d % self.ndim for d in destination)

        return ParameterFrame(
            {k: v.movedim(source, destination) for k, v in self._items()},
            ndim=self.ndim,
        )

    def reset_index(self, levels=None, *, drop=False, names=None):
        """
        Resulting shape is self.shape[[i for i in range(self.ndim - 1) if i not in levels]] + [-1]
        """
        if levels is None:
            levels = range(self.ndim)

        if not isinstance(levels, Iterable):
            levels = {levels}
        else:
            levels = set(levels)

        df = self.copy()
        if not drop:
            if names is None:
                if df.ndim == 1:
                    names = ["index"]
                else:
                    names = (f"level_{i}" for i in range(len(levels)))
            elif isinstance(names, str):
                names = [names]

            try:
                device = self.device
            except AttributeError:
                device = None

            indices = list(lib.np.meshndim(*([1] * self.ndim)))

            for level, name in zip(levels, names, strict=True):
                df[name] = torch.arange(self.shape[level], device=device)[
                    indices[level]
                ]

        shape = [s for i, s in enumerate(self.shape) if i not in levels]
        if len(shape) == 0:
            shape = (-1,)
        else:
            shape[-1] = -1
        df = df.reshape(*shape)

        return df

    def explode(self, column=None, dim=0, to_pandas=False):
        if column is None:
            column = self.keys()

        if isinstance(column, Hashable):
            column = [column]

        if dim == 0:
            if set(column) != set(self.keys()):
                raise NotImplementedError()

            if not lib.itertools.isconst(self.shapes()):
                raise ValueError(
                    f"Shapes of exploded columns must be equal, but got {list(self.shapes())=}."
                )

            d = {k: v.reshape(-1) for k, v in self.items()}
            ndim = 1

        elif dim == 1:
            d = {}
            for k, v in self._items():
                if k in column and v.ndim > 1:
                    for idx in np.ndindex(v.shape[self.ndim :]):
                        key = (
                            (k, *idx)
                            if to_pandas
                            else f'{k}[{",".join(map(str, idx))}]'
                        )
                        d[key] = v[(..., *idx)]
                else:
                    d[k] = v
            ndim = self.ndim

        else:
            raise ValueError(f"dim must be 0 or 1, but got {dim}.")

        if to_pandas:
            if not ndim == 1:
                raise ValueError(
                    f"to_pandas=True is allowed only when resulting ndim=1, but {ndim=}."
                )

            return pd.DataFrame(d)

        return ParameterFrame(d, ndim=ndim)

    def to_framelike(
        self,
        cls=pd.DataFrame,
        keep_indices=True,
        as_index=True,
        to_numpy=True,
        **kwargs,
    ):
        """Convert to a DataFrame-like object.

        Args:
            cls (optional): Type of the resulting DataFrame-like object.
            keep_indices (optional): Whether or not to keep track of the indices
                of the original multi-dimensional ParameterFrame. If the original
                ParameterFrame is one-dimensional, then the indices are discarded
                regardless.
            as_index (optional): If True, sets the indices as an 'index' attribute.
                Otherwise, adds columns 'idx[0]', 'idx[1]', ... to the resulting
                DataFrame-like object. Ignored if keep_indices=False.
            to_numpy (optional): Whether to convert tensors to numpy arrays. If
                cls is pd.DataFrame, then this conversion is performed regardless.
            **kwargs: Additional keyword arguments to pass to cls.

        Returns:
            DataFrame-like object.

        """
        if keep_indices and not as_index and self.ndim > 1:
            if "idx" in self:
                raise ValueError(
                    "ParameterFrame must not contain the column 'idx' if "
                    "keep_indices=True, as_index=False, and ndim > 1."
                )

            df = self.copy()
            df["idx"] = torch.stack(
                torch.meshgrid(*(torch.arange(s) for s in self.shape), indexing="ij"),
                dim=-1,
            )

        else:
            df = self

        df = df.reshape(-1).explode(dim=1)
        if cls is pd.DataFrame:
            # take advantage of the more memory-efficient categorical dtype
            df = {
                k: (
                    v.pandas(force=True)
                    if isinstance(v, CategoricalTensor)
                    else v.numpy(force=True)
                )
                for k, v in df.items()
            }
        else:
            df = {k: v.numpy(force=True) if to_numpy else v for k, v in df.items()}
        df = cls(df, **kwargs)

        if keep_indices and as_index and self.ndim > 1:
            df.index = pd.MultiIndex.from_product([np.arange(s) for s in self.shape])

        return df

    def _repr_html_(self):
        return self.to_framelike()._repr_html_()

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        return ParameterFrame(
            {k: self.data[k] for k in self._keys}, ndim=self.ndim
        )  # shallow copy

    def apply(self, func, dim=-1, args=(), **kwargs):
        """
        Similar to dataframe.apply, but with axis = (*dim, 'columns') and result_type = 'expand'
        Args:
            func: function applied to each subframe that returns tensor/frame.
                  Assumes that output has at least subframe.ndim number of dimensions.
                  For the output to make sense, output should have the same shape as/
                  is broadcastable to subframe.shape, although this is not required.
            dim: int | Sequence[int]. dimensions along which func is applied
            args: Additional positional arguments to pass to func
            **kwargs: Additional keyword arguments to pass to func
        Returns:
            Tensor/frame containing outputs of func applied to all subframes.
        """
        if isinstance(dim, int):
            dim = (dim,)

        if not all(-self.ndim <= d < self.ndim for d in dim):
            raise ValueError(
                f"dim must satisfy -self.ndim <= dim < self.ndim, but {dim=}."
            )

        dim = tuple(d % self.ndim for d in dim)
        newdim = tuple(range(self.ndim - len(dim), self.ndim))

        x = self.movedim(dim, newdim)  # move batch dimensions to leading dimensions
        bshape = x.shape[: -len(dim)]

        out = [func(x.iloc[idx], *args, **kwargs) for idx in np.ndindex(bshape)]
        out = torch.stack(out) if isinstance(out[0], torch.Tensor) else stack(out)
        out = out.reshape(*bshape, *out.shape[1:])

        return out.movedim(newdim, dim)


class ILocIndexer:
    def __init__(self, df, safe=True):
        self.df = df
        self.safe = safe

    def __getitem__(self, key):
        if len(self.df) == 0:
            raise ValueError("Cannot index an empty ParameterFrame.")

        if not isinstance(key, tuple):
            key = (key,)

        items = self.df.items() if self.safe else self.df._items()
        df = {
            k: v[key + (slice(None),) * ndim]
            for (k, v), ndim in zip(items, self.df.ndims())
        }
        k0 = next(iter(self.df.keys()))
        pre_v, post_v = self.df[k0], df[k0]

        ndim = post_v.ndim - (pre_v.ndim - self.df.ndim)
        return ParameterFrame(df, ndim=ndim)


class DataIndexer:
    def __init__(self, df):
        self.df = df

    def __getitem__(self, key):
        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            return ParameterFrame({k: self.df.data[k] for k in key}, ndim=self.df.ndim)

        return super(ParameterFrame, self.df).__getitem__(key)


def broadcast(*frames):
    if len(frames) == 0:
        return

    shape = torch.broadcast_shapes(*[f.shape for f in frames])
    frames = [f.broadcast_to(shape) for f in frames]

    if len(frames) == 1:
        return frames[0]
    return frames


def meshgrid(*frames, dims=None, **kwargs):
    if len(frames) == 0:
        return

    if not all(isinstance(frame, ParameterFrame) for frame in frames):
        types = [type(frame) for frame in frames]
        raise TypeError(f"Each frame must be a ParameterFrame, but got {types=}.")

    if dims is None or isinstance(dims, int):
        dims = [(None, dims)] * len(frames)
    else:
        dims = list(dims)

    # convert negative indices and endpoint None to non-negative indices
    for i, (dim, frame) in enumerate(zip(dims, frames, strict=True)):
        dim = list(dim)
        if isinstance(dim[0], int):
            dim[0] = dim[0] % frame.ndim if frame.ndim > 1 else 0
        if isinstance(dim[1], int):
            dim[1] = dim[1] % frame.ndim if frame.ndim > 1 else 0
        elif dim[1] is None:
            dim[1] = frame.ndim
        dims[i] = tuple(dim)

    ndims = [
        len(frame.shape[dim[0] : dim[1]])
        for dim, frame in zip(dims, frames, strict=True)
    ]
    new_frames = tuple(
        ParameterFrame(ndim=sum(ndims) - ndim + frame.ndim)
        for ndim, frame in zip(ndims, frames, strict=True)
    )

    n = len(frames[0])
    if any(len(f) != n for f in frames):
        raise ValueError("frames must have the same length")

    for key in frames[0].keys():
        new_values = lib.pt.meshgrid(
            *[f.data[key] for f in frames], dims=dims, **kwargs
        )

        for new_frame, new_value in zip(new_frames, new_values, strict=True):
            new_frame[key] = new_value

    if len(new_frames) == 1:
        return new_frames[0]

    return new_frames


def stack(frames, dim=0):
    if not isinstance(frames, (tuple, list)):
        raise NotImplementedError()

    if len(frames) == 0:
        raise ValueError("Must provide a non-empty list of ParameterFrames.")

    if not lib.itertools.isconst(set(frame.keys()) for frame in frames):
        raise ValueError("ParameterFrames must contain the same columns.")

    if not lib.itertools.isconst(frame.ndim for frame in frames):
        raise ValueError("ParameterFrames must have the same ndim.")

    columns = list(frames[0].keys())
    ndim = frames[0].ndim
    dim = dim % (ndim + 1)
    return ParameterFrame(
        {c: torch.stack([f.data[c] for f in frames], dim=dim) for c in columns},
        ndim=ndim + 1,
    )


def concat(
    frames: dict[ParameterFrame] | Sequence[ParameterFrame],
    keys: Iterable[str] | None = None,
    delimiter: str = "_",
    dim: int = 0,
) -> ParameterFrame:
    """
    Similar to pd.concat, but with verify_integrity=True, and
    instead of a heirachical index, hierarchy is represented by
    using the delimiter. dim = -1 represents concatenating
    along the 'columns'. keys and delimiter are ignored when dim == 0.
    keys must be None if frames is a dict.
    """
    if isinstance(frames, dict):
        if keys is not None:
            raise ValueError(f"keys must be None if frames is a dict, but {keys=}.")

        return concat(
            list(frames.values()), keys=frames.keys(), delimiter=delimiter, dim=dim
        )

    if len(frames) == 0:
        raise ValueError(
            f"Must have non-zero number of frames, but got {len(frames)=}."
        )

    if not all(isinstance(frame, ParameterFrame) for frame in frames):
        types = [type(frame) for frame in frames]
        raise TypeError(f"Each frame must be a ParameterFrame, but got {types=}.")

    if not all(frame.ndim == frames[0].ndim for frame in frames):
        raise ValueError("All frames must have the same ndim.")

    ndim = frames[0].ndim
    if dim > ndim:
        raise ValueError(
            f"dim must be less than or equal to the number of dimensions {ndim}, but {dim=}."
        )
    dim = dim % (ndim + 1)

    if dim == ndim:
        if keys is None:
            keys = [None] * len(frames)

        new_frame = ParameterFrame(ndim=ndim)

        for key, frame in zip(keys, frames, strict=True):
            for k, v in frame._items():
                new_frame[delimiter.join(filter(None, (key, k)))] = v

        if len(new_frame) < sum(len(f) for f in frames):
            raise ValueError("ParameterFrames has overlapping keys.")

        return new_frame
    else:
        if not all(list(f.keys()) == list(frames[0].keys()) for f in frames):
            raise ValueError("All frames must have the same keys.")

        if keys:
            raise NotImplementedError()

        frame = {k: torch.cat(v, dim=dim) for k, *v in lib.itertools.dict_zip(*frames)}
        return ParameterFrame(frame, ndim=ndim)
