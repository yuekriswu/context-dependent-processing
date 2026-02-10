from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class _CellType:
    sign: int
    prob: float
    _targets: tuple[str]

    @property
    def targets(self):
        return set(CellType[k] for k in self._targets)


# V1 L2/3 statistics, probabilities approximately equal to Allen Institute's mouse V1 L2/3 model (2020 Billeh)
# (here we taken VIP to be synonymous with the broader class of Htr3a inhibitory neurons)
# TODO: Redesign API to allow for different animals, areas, and layers, loading constants from json/toml files
# class CellType(_CellType, Enum):
#     PYR = (1, 0.85, ("PYR", "PV", "SST", "VIP"))
#     PV = (-1, 0.043, ("PYR", "PV", "SST", "VIP"))
#     SST = (-1, 0.032, ("PYR", "PV", "VIP"))
#     VIP = (-1, 0.075, ("SST",))
#     L4 = (1, 0.075, ("PYR", "PV"))
#     LM = (1, 0.075, ("PYR", "PV", "SST"))
#     X = (1, 0.075, ("SST",))
#     Y = (-1, 0.05, ("SST",))
#
#     # OPTO = (1, 0.06, ("PYR", "PV"))
#     # INE = (1, 0.06, ("PYR",))
#     # INP = (1, 0.06, ("PV",))
#     # INS = (1, 0.06, ("SST",))
#     # INV = (1, 0.06, ("VIP",))


# classical 4
class CellType(_CellType, Enum):
    PYR = (1, 0.8, ("PYR", "PV", "SST", "VIP"))
    PV = (-1, 0.04, ("PYR", "PV", "VIP"))
    SST = (-1, 0.03, ("PYR", "PV", "VIP"))
    VIP = (-1, 0.07, ("SST",))
    L4 = (1, 0.06, ("PYR", "PV",))
    LM = (1, 0.09, ("PYR", "PV", "SST", "VIP"))
    X = (1, 0.05, ("PYR", "PV", "SST", "VIP"))
    X2 = (-1, 0.05, ("PYR", "PV", "SST", "VIP"))
    Y = (1, 0.065, ("PYR", "PV", "SST", "VIP"))
    Y2 = (-1, 0.065, ("PYR", "PV", "SST", "VIP"))
    Z = (-1, 0.075, ("PYR", "PV", "SST", "VIP"))
    Z2 = (1, 0.075, ("PYR", "PV", "SST", "VIP"))

# classical 3
# class CellType(_CellType, Enum):
#     PYR = (1, 0.8, ("PYR", "PV", "SST", "VIP"))
#     PV = (-1, 0.04, ("PYR", "PV", "VIP"))
#     SST = (-1, 0.03, ("PYR", "PV", "VIP"))
#     VIP = (-1, 0.07, ("SST",))
#     L4 = (1, 0.06, ("PYR", "PV",))
#     LM = (1, 0.09, ("PYR", "PV", "SST", "VIP"))
#     X = (1, 0.05, ("SST",))
#     X2 = (-1, 0.05, ("SST",))
#     Y = (1, 0.065, ("PYR", "PV", "SST", "VIP"))
#     Y2 = (-1, 0.065, ("PYR", "PV", "SST", "VIP"))
#     Z = (-1, 0.075, ("PYR", "PV", "SST", "VIP"))
#     Z2 = (1, 0.075, ("PYR", "PV", "SST", "VIP"))

# # # classical 2
# class CellType(_CellType, Enum):
#     PYR = (1, 0.8, ("PYR", "PV", "SST", "VIP"))
#     PV = (-1, 0.04, ("PYR", "PV", "SST", "VIP"))
#     SST = (-1, 0.03, ("PYR", "PV", "SST", "VIP"))
#     VIP = (-1, 0.07, ("PYR", "PV", "SST", "VIP"))
#     L4 = (1, 0.06, ("PYR", "PV", "SST", "VIP"))
#     LM = (1, 0.09, ("PYR", "PV", "SST", "VIP"))
#     X = (1, 0.05, ("SST",))
#     X2 = (-1, 0.05, ("SST",))
#     Y = (1, 0.065, ("PYR", "PV", "SST", "VIP"))
#     Y2 = (-1, 0.065, ("PYR", "PV", "SST", "VIP"))
#     Z = (-1, 0.075, ("PYR", "PV", "SST", "VIP"))
#     Z2 = (1, 0.075, ("PYR", "PV", "SST", "VIP"))

# class CellType(_CellType, Enum):
#    PYR = (1, 0.8, ("PYR", "PV", "SST", "VIP"))
#    PV = (-1, 0.04, ("PYR", "PV", "SST", "VIP"))
#    SST = (-1, 0.03, ("PYR", "PV", "SST", "VIP"))
#    VIP = (-1, 0.07, ("PYR", "PV", "SST", "VIP"))
#    L4 = (1, 0.06, ("PYR", "PV", "SST", "VIP"))
#    LM = (1, 0.09, ("PYR", "PV", "SST", "VIP"))
#    X = (1, 0.05, ("SST",))
#    Y = (-1, 0.05, ("SST",))

### no disinhibition
# class CellType(_CellType, Enum):
#    PYR = (1, 0.8, ("PYR", "PV", "SST", "VIP"))
#    PV = (-1, 0.04, ("PYR", "PV", "SST", "VIP"))
#    SST = (-1, 0.03, ("PYR", "PV", "SST", "VIP"))
#    VIP = (-1, 0.07, ("PYR", "PV", "VIP"))
#    L4 = (1, 0.06, ("PYR", "PV", "SST", "VIP"))
#    LM = (1, 0.09, ("PYR", "PV", "SST", "VIP"))
#    X = (1, 0.05, ("PYR", "PV", "SST", "VIP"))
#    Y = (-1, 0.05, ("PYR", "PV", "SST", "VIP"))


### no recurrence
# class CellType(_CellType, Enum):
#    PYR = (1, 0.8, ())
#    PV = (-1, 0.04, ())
#    SST = (-1, 0.03, ())
#    VIP = (-1, 0.07, ())
#    L4 = (1, 0.06, ("PYR", "PV", "SST", "VIP"))
#    LM = (1, 0.09, ("PYR", "PV", "SST", "VIP"))
#    X = (1, 0.05, ("PYR", "PV", "SST", "VIP"))
#    Y = (-1, 0.05, ("PYR", "PV", "SST", "VIP"))

## no recurrent excitation
# class CellType(_CellType, Enum):
#    PYR = (1, 0.8, ())
#    PV = (-1, 0.04, ("PYR", "PV", "SST", "VIP"))
#    SST = (-1, 0.03, ("PYR", "PV", "SST", "VIP"))
#    VIP = (-1, 0.07, ("PYR", "PV", "SST", "VIP"))
#    L4 = (1, 0.06, ("PYR", "PV", "SST", "VIP"))
#    LM = (1, 0.09, ("PYR", "PV", "SST", "VIP"))
#    X = (1, 0.05, ("PYR", "PV", "SST", "VIP"))
#    Y = (-1, 0.05, ("PYR", "PV", "SST", "VIP"))
#

# class CellType(_CellType, Enum):
#     PYR = (1, 0.8, ("PYR", "PV", "SST", "VIP"))
#     PV = (-1, 0.04, ("PYR", "PV", "SST", "VIP"))
#     SST = (-1, 0.03, ("PYR", "PV", "SST", "VIP"))
#     VIP = (-1, 0.07, ("PYR", "PV", "SST", "VIP"))
#     L4 = (1, 0.06, ("PYR", "PV", "SST", "VIP"))
#     LM = (1, 0.09, ("PYR", "PV", "SST", "VIP"))
#     X = (1, 0.05, ("PYR", "PV", "SST", "VIP"))
#     Y = (-1, 0.05, ("PYR", "PV", "SST", "VIP"))

# class CellType(_CellType, Enum):
#     PYR = (1, 0.85, ("SST",))
#     PV = (-1, 0.043, ("SST",))
#     SST = (-1, 0.032, ("SST",))
#     VIP = (-1, 0.075, ("SST",))
#     L4 = (-1, 0.075, ("SST",))
#     LM = (-1, 0.05, ("SST",))
#     # Z = (-1, 0.05, ("SST",))
