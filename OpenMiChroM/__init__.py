__version__ = '1.1.1'

from importlib import import_module

from .ChromDynamics import MiChroM
from .Optimization import FullTraining, CustomMiChroMTraining
from .CndbTools import CndbTools, cndbTools
from .Integrators import ActiveBrownianIntegrator
from .CustomReporter import *


_CONVERTER_EXPORTS = {
    "convert",
    "ndb_to_cndb",
    "cndb_to_ndb",
    "ndb_to_pdb",
    "pdb_to_ndb",
    "ndb_to_spw",
    "spw_to_ndb",
    "gro_to_ndb",
    "csv_to_ndb",
}


def __getattr__(name):
    """Load converter exports lazily to keep ``python -m ...Converters`` clean."""

    if name in _CONVERTER_EXPORTS:
        value = getattr(import_module(".Converters", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _CONVERTER_EXPORTS)
