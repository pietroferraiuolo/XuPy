"""
Masked-array extras (port of ``numpy.ma.extras`` for numpy and cupy data).

Every function resolves the array module from its operands (cupy if any
operand lives on the GPU, numpy otherwise); Python lists and scalars follow
the active backend.  Results stay on the device of the input.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as _np
from numpy.lib.array_utils import normalize_axis_tuple as _normalize_axis_tuple

from . import _backend
from . import core as _c
from ._backend import get_xp as _get_xp
from ._backend import is_cupy_array as _is_cupy_array
from ._ops import concatenate as _concatenate
from ._singletons import MAError, nomask

_NV = _np._NoValue

__all__ = [
    "issequence", "count_masked", "masked_all", "masked_all_like",
    "compress_nd", "compress_rowcols", "compress_rows", "compress_cols",
    "mask_rowcols", "mask_rows", "mask_cols", "flatten_inplace",
    "sum", "mean", "prod", "product", "average", "std", "var", "min", "max",
    "empty_like", "zeros_like", "ones_like",
    "atleast_1d", "atleast_2d", "atleast_3d",
    "vstack", "hstack", "column_stack", "dstack", "stack", "row_stack",
    "hsplit", "diagflat", "ediff1d", "mr_",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_arraylike(x):
    """True for operands that carry their own device (arrays, masked arrays)."""
    return isinstance(x, _np.ndarray) or _is_cupy_array(x) or getattr(x, "_data", None) is not None


def _xp_of(*ops):
    """Array module of ``ops``; Python lists/scalars alone follow the active backend."""
    xp = _get_xp(*ops)
    if xp is _np and not any(_is_arraylike(o) for o in ops):
        return _backend.default_xp()
    return xp


def _parts(a, xp=None):
    """``(xp, data, mask)`` of ``a`` on ``xp``; ``mask`` is ``nomask`` or a bool array."""
    xp = xp or _xp_of(a)
    d, m = _c._unpack(a)
    d = _backend.asarray(d, xp)
    return xp, d, (m if m is nomask else _backend.asarray(m, xp))


def _full_mask(d, m, xp):
    return xp.zeros(d.shape, dtype=bool) if m is nomask else m


def _asma(a, xp=None):
    """``a`` as a XuPy masked array (no copy when it already is one)."""
    if getattr(a, "_is_xupy_masked", False) and (xp is None or _get_xp(a) is xp):
        return a
    xp, d, m = _parts(a, xp)
    return _c._wrap(d, m, like=a if _c.isMaskedArray(a) else None)


def _copy_ma(a):
    """Independent masked copy of ``a`` (data and mask)."""
    xp, d, m = _parts(a)
    return _c._wrap(d.copy(), m if m is nomask else m.copy(), like=a if _c.isMaskedArray(a) else None)


def _scalar_or_array(res):
    return _backend.host_scalar(res) if res.ndim == 0 else res


# ---------------------------------------------------------------------------
# sequences, counting, creation
# ---------------------------------------------------------------------------
def issequence(seq):
    """Is ``seq`` a sequence (ndarray, cupy array, masked array, list or tuple)?"""
    return bool(isinstance(seq, (_np.ndarray, tuple, list)) or _is_cupy_array(seq)
                or getattr(seq, "_is_xupy_masked", False))


def count_masked(arr, axis=None):
    """Count the masked elements along ``axis`` (all of them if ``axis`` is None).

    Examples
    --------
    >>> import numpy as np
    >>> from xupy.ma import masked_array, count_masked
    >>> count_masked(masked_array(np.array([1., 2., 3.]), mask=np.array([0, 1, 1], bool)))
    np.int64(2)
    """
    xp, d, m = _parts(arr)
    return _scalar_or_array(_full_mask(d, m, xp).sum(axis))


def masked_all(shape, dtype=float):
    """Empty masked array of ``shape`` with every element masked.

    Examples
    --------
    >>> from xupy.ma import masked_all
    >>> masked_all((2, 3)).dtype
    dtype('float64')
    """
    xp = _backend.default_xp()
    return _c._wrap(xp.empty(shape, dtype), xp.ones(shape, dtype=bool))


def _like_xp(a):
    """Device for ``*_like`` results: as the constructor, host input that is not
    already a XuPy masked array goes to the active backend."""
    return _backend.creation_xp(_c._unpack(a)[0], keep_device=getattr(a, "_is_xupy_masked", False))


def masked_all_like(arr):
    """Empty fully-masked array with the shape and dtype of ``arr``."""
    xp, d, _ = _parts(arr, _like_xp(arr))
    like = arr if _c.isMaskedArray(arr) else None
    return _c._wrap(xp.empty_like(d), xp.ones(d.shape, dtype=bool), like=like)


def _make_like(name):
    def func(a, dtype=None, order="K", subok=True, shape=None):
        xp, d, m = _parts(a, _like_xp(a))
        data = getattr(xp, name)(d, dtype=dtype, order=order, shape=shape)
        if m is not nomask:  # numpy's __array_finalize__: keep the mask when sizes agree
            try:
                m = m.copy().reshape(data.shape)
            except ValueError:
                m = xp.zeros(data.shape, dtype=bool)
        return _c._wrap(data, m, like=a if _c.isMaskedArray(a) else None)

    func.__name__ = func.__qualname__ = name
    func.__doc__ = (f"Masked ``{name}``: new array like ``a`` (the mask of ``a`` is kept when "
                    "the number of elements matches), on the device of ``a``.")
    return func


empty_like = _make_like("empty_like")
zeros_like = _make_like("zeros_like")
ones_like = _make_like("ones_like")


def flatten_inplace(seq):
    """Flatten a (nested) sequence in place and return it."""
    k = 0
    while k != len(seq):
        while hasattr(seq[k], "__iter__"):
            seq[k:(k + 1)] = seq[k]
        k += 1
    return seq


# ---------------------------------------------------------------------------
# compress / mask rows and columns
# ---------------------------------------------------------------------------
def compress_nd(x, axis=None):
    """Suppress the slices along ``axis`` (default all axes) that contain masked values.

    Returns the plain data array (a data-dependent shape forces a device sync).
    """
    xp, d, m = _parts(x)
    axis = _normalize_axis_tuple(tuple(range(d.ndim)) if axis is None else axis, d.ndim)
    if m is nomask or not bool(m.any()):
        return d
    if bool(m.all()):
        return xp.array([])
    for ax in axis:
        others = tuple(i for i in range(d.ndim) if i != ax)
        d = d[(slice(None),) * ax + (~m.any(axis=others),)]
    return d


def _check_2d(a, what):
    if _np.ndim(a) != 2:
        raise NotImplementedError(f"{what} works for 2D arrays only.")


def compress_rowcols(x, axis=None):
    """Suppress the rows and/or columns of a 2-D array that contain masked values."""
    _check_2d(x, "compress_rowcols")
    return compress_nd(x, axis=axis)


def compress_rows(a):
    """Suppress whole rows of a 2-D array that contain masked values."""
    _check_2d(a, "compress_rows")
    return compress_rowcols(a, 0)


def compress_cols(a):
    """Suppress whole columns of a 2-D array that contain masked values."""
    _check_2d(a, "compress_cols")
    return compress_rowcols(a, 1)


def mask_rowcols(a, axis=None):
    """Mask the rows (``axis`` None or 0) and/or columns (None, 1, -1) containing masked values.

    Returns a masked copy; any other ``axis`` masks nothing, as in numpy.
    """
    _check_2d(a, "mask_rowcols")
    a = _copy_ma(a)
    a._hardmask = False  # numpy copies through ``array(a, subok=False)``
    m = a._mask
    if m is nomask:
        return a
    new = m
    if not axis:
        new = new | m.any(axis=1)[:, None]
    if axis is None or axis in (1, -1):
        new = new | m.any(axis=0)[None, :]
    a._mask = new
    return a


def _no_axis(axis):
    if axis is not _NV:
        _warnings.warn("The axis argument has always been ignored, in future passing it "
                       "will raise TypeError", DeprecationWarning, stacklevel=3)


def mask_rows(a, axis=_NV):
    """Mask whole rows of a 2-D array that contain masked values."""
    _no_axis(axis)
    return mask_rowcols(a, 0)


def mask_cols(a, axis=_NV):
    """Mask whole columns of a 2-D array that contain masked values."""
    _no_axis(axis)
    return mask_rowcols(a, 1)


# ---------------------------------------------------------------------------
# reductions: thin wrappers over the class methods
# ---------------------------------------------------------------------------
def _frommethod(name, doc):
    def func(a, *args, **kwargs):
        return getattr(_asma(a), name)(*args, **kwargs)

    func.__name__ = func.__qualname__ = name
    func.__doc__ = f"{doc}\n\nFunction form of ``MaskedArray.{name}``; ``a`` may be any array-like."
    return func


sum = _frommethod("sum", "Sum of the unmasked elements over the given axis.")
mean = _frommethod("mean", "Mean of the unmasked elements over the given axis.")
prod = product = _frommethod("prod", "Product of the unmasked elements over the given axis.")
std = _frommethod("std", "Standard deviation of the unmasked elements (``ddof``, ``mean`` supported).")
var = _frommethod("var", "Variance of the unmasked elements (``ddof``, ``mean`` supported).")
min = _frommethod("min", "Minimum of the unmasked elements (``fill_value`` supported).")
max = _frommethod("max", "Maximum of the unmasked elements (``fill_value`` supported).")


def average(a, axis=None, weights=None, returned=False, *, keepdims=_NV):
    """Weighted average over ``axis`` (numpy.ma semantics, masked values excluded).

    With ``returned=True`` the sum of the weights is returned too.  The result
    dtype follows numpy.ma (floating promotion of ``a`` and ``weights``).
    """
    xp = _xp_of(a, weights) if weights is not None else _xp_of(a)
    a = _asma(a, xp)
    m = a._mask
    if axis is not None:
        axis = _normalize_axis_tuple(axis, a.ndim, argname="axis")
    kw = {} if keepdims is _NV else {"keepdims": keepdims}

    if weights is None:
        avg = a.mean(axis, **kw)
        cnt = a.count(axis)
        scl = cnt.astype(avg.dtype) if hasattr(cnt, "astype") else avg.dtype.type(cnt)
    else:
        wgt = _asma(weights, xp)
        if issubclass(a.dtype.type, (_np.integer, _np.bool_)):
            result_dtype = _np.result_type(a.dtype, wgt.dtype, "f8")
        else:
            result_dtype = _np.result_type(a.dtype, wgt.dtype)
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError("Axis must be specified when shapes of a and weights differ.")
            if wgt.shape != tuple(a.shape[ax] for ax in axis):
                raise ValueError("Shape of weights must be consistent with "
                                 "shape of a along specified axis.")
            wgt = wgt.transpose(*(int(i) for i in _np.argsort(axis)))
            wgt = wgt.reshape(tuple(s if ax in axis else 1 for ax, s in enumerate(a.shape)))
        if m is not nomask:
            wgt = _c._wrap(wgt._data * ~m, _c._mask_or(wgt._mask, m, xp))
        scl = wgt.sum(axis=axis, dtype=result_dtype, **kw)
        avg = _np.multiply(a, wgt, dtype=result_dtype).sum(axis, **kw) / scl

    if not returned:
        return avg
    if scl.shape != avg.shape:  # np.broadcast_to drops the mask (subok=False)
        scl = xp.broadcast_to(_c.getdata(scl), avg.shape).copy()
    return avg, scl


# ---------------------------------------------------------------------------
# functions applied to both the data and the mask
# ---------------------------------------------------------------------------
def _single(name, a, *args, **kwargs):
    xp, d, m = _parts(a)
    fn = getattr(xp, name)
    return fn(d, *args, **kwargs), fn(_full_mask(d, m, xp), *args, **kwargs)


def _allargs(name):
    def func(*arys, **kwargs):
        out = tuple(_c._wrap(*_single(name, a, **kwargs)) for a in arys)
        return out[0] if len(out) == 1 else out

    func.__name__ = func.__qualname__ = name
    func.__doc__ = f"Masked ``{name}``: applied to the data and the mask of each input."
    return func


atleast_1d = _allargs("atleast_1d")
atleast_2d = _allargs("atleast_2d")
atleast_3d = _allargs("atleast_3d")


def _joined(name, arrays, kwargs, mask_kwargs=None):
    """``xp.<name>`` of the data and (without ``kwargs``) of the full masks of ``arrays``."""
    arrays = list(arrays)
    xp = _xp_of(*arrays) if arrays else _np
    ps = [_parts(a, xp) for a in arrays]
    if name == "column_stack":  # cupy's version rejects arrays with more than 2 dimensions
        def fn(t):
            return xp.concatenate([xp.atleast_2d(a).T if a.ndim < 2 else a for a in t], 1)
    else:
        fn = getattr(xp, name)
    d = fn(tuple(p[1] for p in ps), **kwargs)
    m = fn(tuple(_full_mask(p[1], p[2], xp) for p in ps), **(mask_kwargs or {}))
    return _c._wrap(d, m.astype(bool, copy=False))


def _stacker(name):
    def func(tup, **kwargs):
        return _joined(name, tup, kwargs)

    func.__name__ = func.__qualname__ = name
    func.__doc__ = f"Masked ``{name}``: applied to the data and the masks of the inputs."
    return func


vstack = row_stack = _stacker("vstack")
hstack = _stacker("hstack")
column_stack = _stacker("column_stack")
dstack = _stacker("dstack")


def stack(arrays, axis=0, out=None, *, dtype=None, casting="same_kind"):
    """Join masked arrays along a new ``axis`` (``out`` receives the data only)."""
    return _joined("stack", arrays, {"axis": axis, "out": out, "dtype": dtype, "casting": casting},
                   {"axis": axis})


def hsplit(ary, indices_or_sections):
    """Split horizontally like ``numpy.ma.hsplit``.

    As in numpy 2.5 the pieces are stacked in one masked array, so uneven
    splits raise ``ValueError``.
    """
    d, m = _single("hsplit", ary, indices_or_sections)
    xp = _get_xp(*d)
    if len({p.shape for p in d}) > 1:
        raise ValueError("setting an array element with a sequence. The requested array has an "
                         "inhomogeneous shape")
    return _c._wrap(xp.stack(d), xp.stack(m))


def diagflat(v, k=0):
    """Create a 2-D masked array with the flattened input as the ``k``-th diagonal."""
    return _c._wrap(*_single("diagflat", v, k=k))


def ediff1d(arr, to_end=None, to_begin=None):
    """Differences between consecutive elements of the flattened array.

    Masked where either operand is masked; ``to_begin``/``to_end`` are
    prepended/appended (their masks, if any, are kept).
    """
    flat = _asma(arr).ravel()
    ed = flat[1:] - flat[:-1]
    arrays = [ed]
    if to_begin is not None:
        arrays.insert(0, to_begin)
    if to_end is not None:
        arrays.append(to_end)
    return ed if len(arrays) == 1 else hstack(arrays)


# ---------------------------------------------------------------------------
# mr_
# ---------------------------------------------------------------------------
class _MRClass:
    """Translate slices, scalars and arrays into a masked concatenation along axis 0.

    Use as ``mr_[a, b, 0, 1:4]`` (see ``numpy.ma.mr_``).  Matrix and axis
    directives (strings) are not supported.
    """

    __slots__ = ()

    def __len__(self):
        return 0

    def __getitem__(self, key):
        if isinstance(key, str):
            raise MAError("Unavailable for masked array.")
        if not isinstance(key, tuple):
            key = (key,)
        xp = _xp_of(*(k for k in key if _is_arraylike(k) or isinstance(k, (list, tuple))))
        objs, rt = [], []
        for item in key:
            if isinstance(item, slice):
                start = 0 if item.start is None else item.start
                step = 1 if item.step is None else item.step
                if isinstance(step, (complex, _np.complexfloating)):
                    obj = xp.linspace(start, item.stop, num=int(abs(step)))
                else:
                    obj = xp.arange(start, item.stop, step)
                m = nomask
            elif isinstance(item, str):
                raise ValueError("unknown special directive")
            elif type(item) in _np.ScalarType:
                obj, m = _np.asarray(item), nomask
            else:
                _, obj, m = _parts(item, xp)
            rt.append(item if type(item) in _np.ScalarType else obj.dtype)
            objs.append((obj, m))
        dtype = _np.result_type(*rt) if rt else None
        pieces = []
        for obj, m in objs:
            obj = _backend.asarray(obj, xp, dtype=dtype)
            if obj.ndim == 0:
                obj = obj.reshape(1)
            if m is not nomask and m.ndim == 0:
                m = m.reshape(1)
            pieces.append(_c._wrap(obj, m))
        return _concatenate(pieces, axis=0)


mr_ = _MRClass()
