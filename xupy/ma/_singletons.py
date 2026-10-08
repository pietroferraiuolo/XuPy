"""
Singletons of ``xupy.ma``: ``nomask``, ``masked`` and the mask type/errors.

``nomask`` and ``masked`` are XuPy's own objects (not numpy's), so they work
the same way on numpy and cupy data.  This module does not import ``core``.
"""
from __future__ import annotations

import operator as _op
import warnings as _warnings

import numpy as _np

MaskType = _np.bool_
MAError = _np.ma.MAError
MaskError = _np.ma.MaskError

_FALSE = _np.False_


class _NoMask:
    """The "no masked value" marker: behaves like ``numpy.ma.nomask`` (``np.False_``).

    Shape-manipulation methods return ``nomask`` itself, so mask code does not
    need to special-case it.  Interaction with numpy and cupy arrays
    (``arr | nomask``, ``nomask & arr``, ``np.logical_or(arr, nomask)``)
    goes through ``__array_ufunc__``/the reflected dunders, which substitute
    ``np.False_`` for the singleton.
    """

    __slots__ = ()
    shape = ()
    ndim = 0
    size = 1
    itemsize = 1
    dtype = _np.dtype(bool)

    # --- identity / protocols -------------------------------------------------
    def __repr__(self):
        return "False"

    __str__ = __repr__

    def __bool__(self):
        return False

    def __hash__(self):
        return hash(_FALSE)

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __array__(self, dtype=None, copy=None):
        return _np.array(False, dtype=dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(_FALSE if i is self else i for i in inputs)
        if kwargs.get("out") is not None:
            kwargs["out"] = tuple(_FALSE if i is self else i for i in kwargs["out"])
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __reduce__(self):
        return "nomask"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __getitem__(self, index):
        return _FALSE[index]

    # --- operators: exactly those of np.False_ ---------------------------------
    def __invert__(self):
        return ~_FALSE

    def _bitop(name):
        fn = getattr(_op, name)

        def method(self, other):
            return fn(_FALSE if other is self else other, _FALSE)

        method.__name__ = f"__{name.rstrip('_')}__"
        return method

    # all three are commutative, so the reflected form is the same function
    __and__ = __rand__ = _bitop("and_")
    __or__ = __ror__ = _bitop("or_")
    __xor__ = __rxor__ = _bitop("xor")
    del _bitop

    def __eq__(self, other):
        return _op.eq(_FALSE if other is self else other, _FALSE)

    def __ne__(self, other):
        return _op.ne(_FALSE if other is self else other, _FALSE)

    # --- arithmetic / comparisons / conversions: exactly those of np.False_ -----
    def _false_op(name):
        fn = getattr(_FALSE, name, None)
        if fn is None:  # not provided by np.False_ either
            return None

        def method(self, *args):
            args = tuple(_FALSE if a is self else a for a in args)
            return fn(*args)

        method.__name__ = name
        return method

    for _n in (
        "add sub mul truediv floordiv mod divmod pow matmul lshift rshift "
        "lt le gt ge"
    ).split():
        if _false_op(f"__{_n}__"):
            locals()[f"__{_n}__"] = _false_op(f"__{_n}__")
    for _n in "add sub mul truediv floordiv mod divmod pow matmul lshift rshift".split():
        if _false_op(f"__r{_n}__"):
            locals()[f"__r{_n}__"] = _false_op(f"__r{_n}__")
    for _n in "neg pos abs complex index".split():
        if _false_op(f"__{_n}__"):
            locals()[f"__{_n}__"] = _false_op(f"__{_n}__")
    del _n, _false_op

    # --- array-like surface ----------------------------------------------------
    def _self(self, *args, **kwargs):
        return self

    reshape = ravel = flatten = copy = astype = transpose = squeeze = view = _self
    swapaxes = _self
    T = property(_self)
    mT = property(_self)

    def any(self, *args, **kwargs):
        return _FALSE.any(*args, **kwargs)

    def all(self, *args, **kwargs):
        return _FALSE.all(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return _FALSE.sum(*args, **kwargs)

    def item(self, *args):
        return False

    def tolist(self):
        return False


nomask = _NoMask()


class _MaskedConstant:
    """The masked element: behaves like ``numpy.ma.masked``.

    Arithmetic and comparisons with anything return ``masked``.  It is a 0-d,
    fully-masked float64 operand in array operations (``_data``/``_mask``);
    ``asmarray()`` maps it to ``numpy.ma.masked``.
    """

    _is_xupy_masked_constant = True
    __array_priority__ = 1000  # cupy defers `cupy_array <op> masked` to our reflected dunders
    __slots__ = ()

    shape = ()
    ndim = 0
    size = 1
    dtype = _np.dtype("float64")
    itemsize = 8
    _data = _np.array(0.0)
    _mask = _np.array(True)
    _data.flags.writeable = False
    _mask.flags.writeable = False
    data = _data
    mask = _mask
    _fill_value = _np.array(_np.ma.default_fill_value(dtype))
    _hardmask = False
    _sharedmask = False

    # --- values ----------------------------------------------------------------
    @property
    def fill_value(self):
        return self._fill_value[()]

    def filled(self, fill_value=None):
        """The fill value as a 0-d array (the only element is masked)."""
        if fill_value is None:
            return self._fill_value.copy()
        return _np.asarray(fill_value, dtype=self.dtype)

    def asmarray(self, **kwargs):
        """``numpy.ma.masked``."""
        return _np.ma.masked

    def copy(self, *args, **kwargs):
        return self

    __copy__ = copy

    def __deepcopy__(self, memo=None):
        return self

    def __reduce__(self):
        return "masked"

    def tolist(self, fill_value=None):
        return None if fill_value is None else fill_value

    def item(self, *args):
        return 0.0

    def __array__(self, dtype=None, copy=None):
        return _np.array(0.0, dtype=dtype)

    # --- numpy protocols and methods: those of the 0-d fully-masked float64 array ---
    # (``np.sqrt(masked)`` is ``masked``, ``masked.sum()`` is ``masked``, ... as numpy.ma)
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return getattr(ufunc, method)(*_as_arrays(inputs), **kwargs)

    def __getitem__(self, index):
        return self._as_array()[index]

    @property
    def T(self):
        return self._as_array().T

    mT = T

    def _delegate(name):
        def method(self, *args, **kwargs):
            return getattr(self._as_array(), name)(*args, **kwargs)

        method.__name__ = name
        return method

    for _n in (
        "sum prod mean std var min max ptp any all argmin argmax cumsum cumprod count "
        "astype reshape ravel flatten squeeze transpose swapaxes repeat take view "
        "round conj conjugate clip compressed anom"
    ).split():
        locals()[_n] = _delegate(_n)
    del _n, _delegate

    # --- Python scalar conversions (as numpy 2.5) --------------------------------
    def __bool__(self):
        return False

    def __float__(self):
        _warnings.warn("Warning: converting a masked element to nan.", stacklevel=2)
        return _np.nan

    def __int__(self):
        raise MaskError("Cannot convert masked element to a Python int.")

    def __complex__(self):
        return 0j

    def __hash__(self):
        return id(self)

    # --- text --------------------------------------------------------------------
    def __repr__(self):
        return "masked"

    def __str__(self):
        return str(_np.ma.masked_print_option)

    def __format__(self, format_spec):
        return str(self)

    # --- operators --------------------------------------------------------------
    # Against scalars (and `masked`) everything yields `masked`.  Against arrays
    # the constant behaves as numpy.ma's 0-d fully-masked float64 array broadcast
    # against the operand, giving an all-masked ARRAY: XuPy arrays are left to
    # their own (reflected) dunder; plain numpy/cupy/list operands go through the
    # operator of the 0-d masked array equivalent of `masked`.
    def _as_array(self):
        from . import core

        return core._wrap(self._data, self._mask)

    def _binop(name):
        dunder = f"__{name}__"

        def method(self, other):
            if (
                isinstance(other, _SCALAR_TYPES)
                or getattr(other, "_is_xupy_masked_constant", False)
            ):
                if name in _NO_FLOAT:
                    return getattr(self._as_array(), dunder)(other)
                return (self, self) if name == "divmod" else self
            if getattr(other, "_is_xupy_masked", False):
                return NotImplemented
            return getattr(self._as_array(), dunder)(other)

        method.__name__ = dunder
        return method

    def _rbinop(name):
        dunder = f"__r{name}__"

        def method(self, other):
            if (
                isinstance(other, _SCALAR_TYPES)
                or getattr(other, "_is_xupy_masked_constant", False)
            ):
                if name in _NO_FLOAT:
                    return getattr(self._as_array(), dunder)(other)
                return (self, self) if name == "divmod" else self
            if getattr(other, "_is_xupy_masked", False):
                return NotImplemented
            return getattr(self._as_array(), dunder)(other)

        method.__name__ = dunder
        return method

    for _n in "add sub mul truediv floordiv mod divmod pow matmul lshift rshift and or xor".split():
        locals()[f"__{_n}__"] = _binop(_n)
        locals()[f"__r{_n}__"] = _rbinop(_n)
    # numpy.ma has in-place `+ - * // / **` returning the constant itself; the
    # others fall back to the plain operators
    for _n in "add sub mul truediv floordiv pow".split():
        locals()[f"__i{_n}__"] = lambda self, other: self
    # comparisons: the reflected comparison of the other operand is used for arrays
    for _n in "eq ne lt le gt ge".split():
        locals()[f"__{_n}__"] = _binop(_n)
    del _n, _binop, _rbinop

    def __neg__(self):
        return self

    __pos__ = __abs__ = __neg__

    def __invert__(self):
        return _np.invert(self._data)  # TypeError for float, as numpy.ma


def _as_arrays(obj):
    """Replace ``masked`` by its 0-d masked-array form in a tuple of ufunc inputs."""
    return tuple(o._as_array() if getattr(o, "_is_xupy_masked_constant", False) else o for o in obj)


_SCALAR_TYPES = (bool, int, float, complex, _np.generic)
# operations numpy.ma rejects for a float64 `masked` (no float loop / 0-d matmul)
_NO_FLOAT = frozenset("matmul lshift rshift and or xor".split())


masked = _MaskedConstant()
