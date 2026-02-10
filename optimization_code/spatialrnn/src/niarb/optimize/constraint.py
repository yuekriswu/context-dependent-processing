import abc
from collections.abc import Iterable

import torch

from niarb.cell_type import CellType
from niarb import nn

__all__ = ["Constraint", "StabilityCon"]

class Constraint(abc.ABC):
    def __init__(self, is_equality: bool):
        if not isinstance(is_equality, bool):
            raise ValueError(f"is_equality must be a bool, but {type(is_equality)=}.")

        super().__init__()
        self.is_equality = is_equality

    def __repr__(self):
        properties = [
            f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")
        ]
        return f'{type(self).__name__}({", ".join(properties)})'

    @abc.abstractmethod
    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        pass

class EqConstraint1(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[0, 6] - m.gW[0, 7]

class EqConstraint2(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[1, 6] - m.gW[1, 7]

class EqConstraint3(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[2, 6] - m.gW[2, 7]

class EqConstraint4(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[3, 6] - m.gW[3, 7]

class EqConstraint5(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[0, 8] - m.gW[0, 9]

class EqConstraint6(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[1, 8] - m.gW[1, 9]

class EqConstraint7(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[2, 8] - m.gW[2, 9]

class EqConstraint8(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[3, 8] - m.gW[3, 9]

class EqConstraint9(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[0, 10] - m.gW[0, 11]

class EqConstraint10(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[1, 10] - m.gW[1, 11]

class EqConstraint11(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[2, 10] - m.gW[2, 11]

class EqConstraint12(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[3, 10] - m.gW[3, 11]

class EqConstraint13(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[0, 12] - m.gW[0, 13]

class EqConstraint14(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[1, 12] - m.gW[1, 13]

class EqConstraint15(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[2, 12] - m.gW[2, 13]

class EqConstraint16(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[3, 12] - m.gW[3, 13]


class EqConstraintACA(Constraint):
    def __init__(self):
        super().__init__(is_equality=True)

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        return m.gW[2, 6] - m.gW[2, 7]

class StabilityCon(Constraint):
    def __init__(
        self,
        eps: float = 0.1,
        cell_types: Iterable[CellType | str] = None,
        stable: bool = True,
    ):
        super().__init__(is_equality=False)
        self.eps = eps
        self.cell_types = cell_types
        self.stable = stable

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        v1_modules = list(filter(lambda m: isinstance(m, nn.V1), model.modules()))
        if len(v1_modules) != 1:
            raise ValueError(
                f"model must have exactly one V1 module, but got {len(v1_modules)=}."
            )
        m = v1_modules[0]
        a = m.spectral_summary(cell_types=self.cell_types, kind="J").abscissa
        out = -a if self.stable else a
        return out - self.eps
