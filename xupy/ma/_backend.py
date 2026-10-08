"""
Backend resolution for ``xupy.ma``.

The array module (numpy or cupy) used by every masked-array method and
function is resolved from the *operands*, never from the global backend
switch: if any operand lives on a CUDA device, cupy is used, else numpy.
The global/context backend (``xupy._core._active_gpu``) is consulted only
when *creating* an array from Python data (lists, scalars), which carries no
device of its own.
"""
from __future__ import annotations

import sys as _sys

import numpy as _np

from .. import _core


def cupy_module():
    """Return the cupy module if it is usable/imported, else ``None``."""
    cp = _core._cupy
    if cp is not None:
        return cp
    cp = _sys.modules.get("cupy")  # imported by the user; may be None (blocked)
    return cp


def is_cupy_array(x) -> bool:
    """True if ``x`` is a cupy ndarray."""
    cp = cupy_module()
    return cp is not None and isinstance(x, cp.ndarray)


def _operand_array(x):
    """Return the raw array held by an operand, or ``x`` itself."""
    # XuPy masked arrays (and the `masked` constant) expose `_data`.
    d = getattr(x, "_data", None)
    return x if d is None else d


def get_xp(*operands):
    """Array module for ``operands``: cupy if any operand is a cupy array
    (or an XuPy masked array holding one), else numpy."""
    cp = cupy_module()
    if cp is None:
        return _np
    nd = cp.ndarray
    for x in operands:
        if isinstance(x, nd) or isinstance(_operand_array(x), nd):
            return cp
    return _np


def default_xp():
    """Array module for creating arrays from Python data: cupy when the active
    (global or context) backend is the GPU, else numpy."""
    cp = _core._cupy
    if cp is not None and _core._active_gpu():
        return cp
    return _np


def creation_xp(raw, keep_device=False):
    """Array module for *creating* a masked array from ``raw``.

    cupy data stays on the GPU.  Host data (numpy arrays, ``numpy.ma`` arrays,
    Python data) goes to the active XuPy backend, so it is transferred to the
    GPU in GPU mode (the XuPy <= 1.7 contract), unless ``keep_device`` is set
    (an existing numpy-backed XuPy masked array never moves implicitly).
    """
    if is_cupy_array(raw):
        return cupy_module()
    if keep_device and isinstance(raw, _np.ndarray):
        return _np
    return default_xp()


def is_cupy_module(xp) -> bool:
    return xp is not _np


def asarray(x, xp, dtype=None, copy=None):
    """``xp.asarray(x, dtype)``, moving ``x`` between devices when needed."""
    if xp is _np:
        if is_cupy_array(x):
            x = x.get()
        return _np.asarray(x, dtype=dtype) if copy is None else _np.array(x, dtype=dtype, copy=copy)
    if copy:
        return xp.array(x, dtype=dtype)
    return xp.asarray(x, dtype=dtype)


def to_host(x):
    """Host numpy ndarray for a numpy/cupy array (no copy for numpy input)."""
    if is_cupy_array(x):
        return x.get()
    return _np.asarray(x)


def host_scalar(x):
    """Numpy scalar (``np.generic``) from a 0-d numpy/cupy array or scalar."""
    if is_cupy_array(x):
        x = x.get()
    if isinstance(x, _np.generic):
        return x
    return _np.asarray(x)[()]
