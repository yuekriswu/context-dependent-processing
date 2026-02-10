import functools

import torch


@functools.cache
def tensor_class_factory(*bases, **kwargs):
    if len(bases) == 0:
        raise ValueError("Must have at least one base class.")

    # cache created classes to ensure equivalence of classes with the same metadata
    metadata = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return type(f"{bases[0].__name__}({metadata})", bases, kwargs)


# for pickling
def _rebuild(func, args, cls):
    args = list(args)
    args[1] = tensor_class_factory(cls, **args[1])
    return func(*args)


class BaseTensor(torch.Tensor):
    def __new__(cls, data, requires_grad=False):
        if requires_grad:
            tensor = cls._make_subclass(cls, data, requires_grad)
        else:
            tensor = super().__new__(cls, data)

        return tensor

    @property
    def tensor(self):
        # return a pure torch.Tensor without all the extra attribute
        return self.as_subclass(torch.Tensor)

    def new_empty(self, *args, **kwargs):
        # needed for copy.deepcopy to work
        return super().new_empty(*args, **kwargs).as_subclass(type(self))
