"""
XuPy: A library with authomatic handling of GPU and CPU arrays.

The namespace is resolved dynamically (PEP 562) against the active backend;
see ``xupy._core``.
"""

from . import _core
from .__version__ import __version__
from ._core import use_cpu, use_gpu, backend

# Import xupy.ma so that sys.modules['xupy.ma'] exists, then drop the package
# attribute: `xp.ma`, `import xupy.ma as m` and `from xupy import ma` then go
# through __getattr__ and follow the backend (numpy.ma on CPU).
# `from xupy.ma import ...` still loads XuPy's module through sys.modules.
from . import ma as _ma  # noqa: F401

del ma, _ma


def __getattr__(name):
    if name == "__all__":
        return _core._public_names()
    return _core.__getattr__(name)


def __dir__():
    return sorted(set(_core.__dir__()) | {"__version__"})
