"""
pyfive : a pure python HDF5 file reader.
This is the public API exposed by pyfive,
which is a small subset of the H5PY API.
"""

from .high_level import File, Group, Dataset
from .h5t import check_enum_dtype, check_string_dtype, check_dtype, opaque_dtype, check_opaque_dtype
from .h5py import Datatype, Empty
from importlib.metadata import version
from .inspect import p5ncdump

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyfive")
except PackageNotFoundError:
    __version__ = "1.0.1+hdf5_indexed_reader"
