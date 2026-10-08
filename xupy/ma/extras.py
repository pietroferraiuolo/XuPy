"""
Masked-array extras (port of ``numpy.ma.extras`` for numpy and cupy data).

Every function resolves the array module from its operands (cupy if any
operand lives on the GPU, numpy otherwise); Python lists and scalars follow
the active backend.  Results stay on the device of the input.
"""
from __future__ import annotations

import itertools as _itertools
import warnings as _warnings

import numpy as _np
from numpy.lib.array_utils import normalize_axis_index as _normalize_axis_index
from numpy.lib.array_utils import normalize_axis_tuple as _normalize_axis_tuple

from . import _backend
from . import core as _c
from ._backend import get_xp as _get_xp
from ._backend import is_cupy_array as _is_cupy_array
from ._ops import concatenate as _concatenate
from ._ops import dot
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
    "dot", "anom", "anomalies", "apply_along_axis", "apply_over_axes", "clump_masked",
    "clump_unmasked", "corrcoef", "cov", "flatnotmasked_contiguous",
    "flatnotmasked_edges", "in1d", "intersect1d", "isin", "median", "ndenumerate",
    "notmasked_contiguous", "notmasked_edges", "polyfit", "setdiff1d", "setxor1d",
    "union1d", "unique", "vander",
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
anom = anomalies = _frommethod("anom", "Anomalies (deviations from the mean) of the unmasked elements.")


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=_NV, mean=_NV):
    """Standard deviation of the unmasked elements.

    Function form of ``MaskedArray.std`` (``ddof`` and ``mean`` supported, as in
    numpy 2.5 ``numpy.ma.std``; numpy.ma has no ``correction`` keyword).
    """
    return _asma(a).std(axis, dtype, out, ddof, keepdims, mean)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=_NV, mean=_NV):
    """Variance of the unmasked elements.

    Function form of ``MaskedArray.var`` (``ddof`` and ``mean`` supported, as in
    numpy 2.5 ``numpy.ma.var``; numpy.ma has no ``correction`` keyword).
    """
    return _asma(a).var(axis, dtype, out, ddof, keepdims, mean)


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


# ---------------------------------------------------------------------------
# apply_along_axis / apply_over_axes
# ---------------------------------------------------------------------------
def _res_parts(res, xp):
    """``(data, mask)`` of a ``func1d`` result (scalar, array or masked)."""
    d, m = _c._unpack(res)
    return _backend.asarray(d, xp), (m if m is nomask else _backend.asarray(m, xp))


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply ``func1d(a, *args, **kwargs)`` to the 1-D slices of ``arr`` along ``axis``.

    Like ``numpy.ma.apply_along_axis``: ``func1d`` receives masked-array slices and the
    result is a masked array (mask ``nomask`` if no result carried a mask), with the
    largest dtype among the results.  One ``func1d`` call per slice (host loop); the
    results are stacked on the device of ``arr`` without host synchronisation.
    """
    xp = _xp_of(arr)
    arr = _asma(arr, xp)
    nd = arr.ndim
    axis = _normalize_axis_index(axis, nd)
    others = [i for i in range(nd) if i != axis]
    lane_shape = tuple(arr.shape[i] for i in others)
    parts, dtypes = [], []
    for ind in _itertools.product(*(range(n) for n in lane_shape)):
        idx = [0] * nd
        idx[axis] = slice(None)
        for i, k in zip(others, ind):
            idx[i] = k
        parts.append(_res_parts(func1d(arr[tuple(idx)], *args, **kwargs), xp))
        dtypes.append(parts[-1][0].dtype)
    if not parts:  # numpy fails indexing the first (non-existent) lane
        raise IndexError("apply_along_axis: index 0 is out of bounds, no slice to apply func1d to.")
    rshape = parts[0][0].shape
    data = xp.stack([p[0] for p in parts]).reshape(lane_shape + rshape)
    anym = any(p[1] is not nomask for p in parts)
    mask = nomask
    if anym:
        mask = xp.stack([xp.broadcast_to(_full_mask(p[0], p[1], xp), rshape) for p in parts])
        mask = mask.reshape(lane_shape + rshape)
    r = len(rshape)
    if r:  # the results' dimensions replace ``axis``
        perm = list(range(axis)) + list(range(nd - 1, nd - 1 + r)) + list(range(axis, nd - 1))
        data = data.transpose(perm)
        if anym:
            mask = mask.transpose(perm)
    data = data.astype(_np.dtype(_np.asarray(dtypes).max()), copy=False)
    return _c._wrap(data, mask, fill_value=_c.default_fill_value(data))


def apply_over_axes(func, a, axes):
    """Apply ``func(a, axis)`` repeatedly over ``axes`` (masked-array version of ``numpy.apply_over_axes``)."""
    val = _asma(a)
    N = a.ndim
    if _np.ndim(axes) == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        res = func(val, axis)
        if res.ndim == val.ndim:
            val = res
        else:
            if isinstance(res, _np.generic):  # scalar result: plain array on the device of ``a``
                res = _get_xp(val).asarray(res)
            elif getattr(res, "_is_xupy_masked_constant", False):
                res = _asma(res, _get_xp(val))
            res = _c.expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning an array of the correct shape")
    return val


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------
def _fill_max(dtype):
    """Value that sorts after every valid value of ``dtype`` (+inf, int max, True)."""
    dtype = _np.dtype(dtype)
    if dtype.kind in "fc":
        return _np.inf
    if dtype.kind == "b":
        return True
    return _np.iinfo(dtype).max


def _median_lanes(x, mk, xp):
    """Median of the unmasked values along the LAST axis of ``x``; returns ``(data, empty_lane_mask)``."""
    L = x.shape[-1]
    cnt = (~mk).sum(-1, keepdims=True)
    s = xp.sort(xp.where(mk, _fill_max(x.dtype), x), axis=-1)
    h = cnt // 2
    lo = xp.maximum(xp.where(cnt % 2 == 1, h, h - 1), 0)
    hi = xp.minimum(h, L - 1)
    low = xp.take_along_axis(s, lo, -1)
    high = xp.take_along_axis(s, hi, -1)
    if x.dtype.kind in "fc":
        res = (low + high) / 2
        res = xp.where(xp.isnan(s[..., -1:]), xp.asarray(_np.nan, dtype=res.dtype), res)  # unmasked NaN wins
    else:
        res = (low.astype(_np.float64) + high.astype(_np.float64)) / 2
    return res[..., 0], (cnt == 0)[..., 0]


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Median of the unmasked elements along ``axis`` (numpy.ma semantics).

    Computed on the device with a sort and the per-lane counts of unmasked
    values.  ``axis`` may be an int, a tuple or None.  Lanes with no unmasked
    value are masked; an unmasked NaN gives NaN.  ``overwrite_input`` is
    accepted but the input is never modified (numpy treats it as a permission).
    A scalar result (``axis=None`` without ``keepdims``) is a host
    synchronisation; array results do not synchronise.
    """
    if not hasattr(a, "mask"):
        xp = _xp_of(a)
        d = _backend.asarray(_c._unpack(a)[0], xp)
        m = xp.median(d, axis=axis, out=out, keepdims=keepdims)
        if out is None and m.ndim == 0 and not keepdims:
            return _backend.host_scalar(m)
        return _c._wrap(m, nomask) if m.ndim >= 1 else m
    xp, d, mk = _parts(a)
    if d.ndim == 0:
        d, mk = d.reshape(1), (mk if mk is nomask else mk.reshape(1))
    nd = d.ndim
    axes = tuple(range(nd)) if axis is None else _normalize_axis_tuple(axis, nd)
    rest = tuple(i for i in range(nd) if i not in axes)
    perm = rest + axes
    lane = 1
    for i in axes:
        lane *= d.shape[i]
    rshape = tuple(d.shape[i] for i in rest)
    x = d.transpose(perm).reshape(rshape + (lane,))
    full = _full_mask(d, mk, xp).transpose(perm).reshape(rshape + (lane,))
    if lane == 0:  # mean of an empty slice, as numpy
        fdt = x.dtype if x.dtype.kind in "fc" else _np.float64
        data = xp.full(rshape, _np.nan, dtype=fdt)
        mask = nomask if mk is nomask else xp.ones(rshape, dtype=bool)
    else:
        data, empty = _median_lanes(x, full, xp)
        mask = nomask if mk is nomask else empty
    shape = tuple(1 if i in axes else d.shape[i] for i in range(nd)) if keepdims else rshape
    if shape != data.shape:
        data = data.reshape(shape)
        mask = mask if mask is nomask else mask.reshape(shape)
    if out is not None:
        if _c.isMaskedArray(out):
            out[...] = _c._wrap(data, mask)
        else:
            out[...] = data
        return out
    if data.ndim == 0:
        return _c._maybe_scalar(data, mask)
    if keepdims and not rest:  # numpy reshapes the scalar result: plain array unless masked (float64 then)
        if mask is not nomask and bool(mask.any()):
            return _c._wrap(data if data.dtype.kind == "c" else data.astype(_np.float64), mask)
        return data
    return _c._wrap(data, mask)


# ---------------------------------------------------------------------------
# unique and set operations (flattened inputs; sorted, masked values last)
# ---------------------------------------------------------------------------
def _stable_argsort(x, xp):
    return _np.argsort(x, kind="stable") if xp is _np else xp.argsort(x)


def _flat(a, xp):
    """Flattened ``(data, mask)`` of ``a``; ``mask`` is ``nomask`` or a bool array."""
    _, d, m = _parts(a, xp)
    return d.ravel(), (m if m is nomask else m.ravel())


def _cat(arrays, xp, shrink=False):
    """Concatenation of the flattened inputs: ``(data, mask)``.

    With ``shrink`` a mask without any True becomes ``nomask`` (``numpy.ma.concatenate``; host sync).
    """
    ps = [_flat(a, xp) for a in arrays]
    d = xp.concatenate([p[0] for p in ps])
    if all(p[1] is nomask for p in ps):
        return d, nomask
    m = xp.concatenate([_full_mask(p[0], p[1], xp) for p in ps])
    return d, (nomask if shrink and not bool(m.any()) else m)


def _sort_order(d, m, xp):
    """Stable ordering with masked values last (all masked values tie)."""
    if m is nomask:
        return _stable_argsort(d, xp)
    o1 = _stable_argsort(xp.where(m, xp.zeros((), d.dtype), d), xp)
    return o1[_stable_argsort(m[o1], xp)]


def _neq(a, b, xp):
    """``a != b`` with NaN == NaN (``np.unique(equal_nan=True)``)."""
    ne = a != b
    if a.dtype.kind == "f":
        ne = ne & ~(xp.isnan(a) & xp.isnan(b))
    return ne


def unique(ar1, return_index=False, return_inverse=False):
    """Sorted unique elements of ``ar1`` (flattened); all masked values collapse into one, last.

    Returns a masked array (mask ``nomask`` when ``ar1`` has none), plus
    ``index`` (first occurrences) and ``inverse`` arrays on request.  The
    output size is data-dependent: this function synchronises with the device.
    """
    xp = _xp_of(ar1)
    d, m = _flat(ar1, xp)
    if m is not nomask and not bool(m.any()):
        m = nomask
    order = _sort_order(d, m, xp)
    sd = d[order]
    sm = nomask if m is nomask else m[order]
    n = sd.size
    flag = xp.ones(n, dtype=bool)
    if n > 1:
        if m is nomask:  # NaNs collapse (``np.unique``) only without masked values
            ne = _neq(sd[1:], sd[:-1], xp)
        else:
            ne = (sm[1:] != sm[:-1]) | (~sm[1:] & (sd[1:] != sd[:-1]))
        flag[1:] = ne
    idx = xp.flatnonzero(flag)
    res = _c._wrap(sd[idx], nomask if sm is nomask else sm[idx])
    if not (return_index or return_inverse):
        return res
    out = [res]
    if return_index:
        out.append(order[idx])
    if return_inverse:
        inv = xp.empty(n, dtype=_np.intp)
        inv[order] = xp.cumsum(flag, dtype=_np.intp) - 1
        out.append(inv.reshape(_np.shape(_c._unpack(ar1)[0])))
    return tuple(out)


def _sorted_cat(arrays, xp):
    d, m = _cat(arrays, xp, shrink=True)
    order = _sort_order(d, m, xp)
    return d[order], (nomask if m is nomask else m[order])


def intersect1d(ar1, ar2, assume_unique=False):
    """Sorted values present in both inputs (flattened); masked values match each other.

    The output size is data-dependent: this function synchronises with the device.
    """
    xp = _xp_of(ar1, ar2)
    if not assume_unique:
        ar1, ar2 = unique(ar1), unique(ar2)
    sd, sm = _sorted_cat((ar1, ar2), xp)
    eq = sd[1:] == sd[:-1]
    if sm is not nomask:
        eq = (sm[1:] & sm[:-1]) | (~sm[1:] & ~sm[:-1] & eq)
    return _c._wrap(sd[:-1][eq], nomask if sm is nomask else sm[:-1][eq])


def setxor1d(ar1, ar2, assume_unique=False):
    """Sorted values present in exactly one of the inputs (flattened).

    The output size is data-dependent: this function synchronises with the device.
    """
    xp = _xp_of(ar1, ar2)
    if not assume_unique:
        ar1, ar2 = unique(ar1), unique(ar2)
    sd, sm = _sorted_cat((ar1, ar2), xp)
    if sd.size == 0:
        return _c._wrap(sd, sm)
    f = sd
    if sm is not nomask:  # numpy compares ``aux.filled()`` (default fill value for masked)
        f = xp.where(sm, xp.asarray(_np.array(_c.default_fill_value(sd.dtype)).astype(sd.dtype)), sd)
    t = xp.ones(1, dtype=bool)
    flag = xp.concatenate([t, f[1:] != f[:-1], t])
    keep = flag[1:] == flag[:-1]
    return _c._wrap(sd[keep], nomask if sm is nomask else sm[keep])


def in1d(ar1, ar2, assume_unique=False, invert=False):
    """Test whether each element of ``ar1`` (flattened) is also in ``ar2``; always a masked array.

    Masks follow ``numpy.ma.in1d``.  (numpy 2.5's ``numpy.ma.in1d`` emits no
    DeprecationWarning, so neither does this function.)  Without
    ``assume_unique`` it synchronises with the device (via ``unique``).
    """
    xp = _xp_of(ar1, ar2)
    if not assume_unique:
        u1, rev_idx = unique(ar1, return_inverse=True)
        ar1, ar2 = u1, unique(ar2)
    n1 = _flat(ar1, xp)[0].size
    d, m = _cat((ar1, ar2), xp)
    order = _sort_order(d, m, xp)
    sd = d[order]
    n = sd.size
    ne = sd[1:] != sd[:-1]
    t = xp.asarray([bool(invert)])
    if m is None or m is nomask:
        pair = xp.logical_not(ne) if not invert else ne
        flag_m = nomask
    else:
        sm = m[order]
        mpair = sm[1:] | sm[:-1]
        base = sm[1:] != sm[:-1]
        pair = xp.where(mpair, base, ne) if invert else xp.where(mpair, ~base, ~ne)
        flag_m = xp.concatenate([mpair, xp.zeros(1, dtype=bool)])
    flag = xp.concatenate([pair, t])
    pos = xp.empty(n, dtype=_np.intp)
    pos[order] = xp.arange(n, dtype=_np.intp)
    indx = pos[:n1]
    if not assume_unique:
        indx = indx[rev_idx]
    return _c._wrap(flag[indx], flag_m if flag_m is nomask else flag_m[indx])


def isin(element, test_elements, assume_unique=False, invert=False):
    """Element-wise ``in`` test keeping the shape of ``element`` (see ``in1d``)."""
    element = _asma(element, _xp_of(element, test_elements))
    return in1d(element, test_elements, assume_unique=assume_unique, invert=invert).reshape(element.shape)


def union1d(ar1, ar2):
    """Sorted union of the (flattened) inputs; masked values collapse into one.

    The output size is data-dependent: this function synchronises with the device.
    """
    xp = _xp_of(ar1, ar2)
    d, m = _cat((ar1, ar2), xp, shrink=True)
    return unique(_c._wrap(d, m))


def setdiff1d(ar1, ar2, assume_unique=False):
    """Sorted values of ``ar1`` not in ``ar2`` (flattened).

    The output size is data-dependent: this function synchronises with the device.
    """
    xp = _xp_of(ar1, ar2)
    if assume_unique:
        ar1 = _asma(ar1, xp).ravel()
    else:
        ar1, ar2 = unique(ar1), unique(ar2)
    keep = _c.getdata(in1d(ar1, ar2, assume_unique=True, invert=True))
    return ar1[keep]


# ---------------------------------------------------------------------------
# covariance
# ---------------------------------------------------------------------------
def _ndmin2(a, xp):
    """Float64 masked copy of ``a`` with at least two dimensions (leading axes added)."""
    _, d, m = _parts(a, xp)
    d = d.astype(_np.float64)
    if d.ndim < 2:
        shape = (1,) * (2 - d.ndim) + d.shape
        d = d.reshape(shape)
        m = m if m is nomask else m.reshape(shape)
    return d, m


def _covhelper(x, y=None, rowvar=True, allow_masked=True):
    xp = _xp_of(x, y)
    xd, xm = _ndmin2(x, xp)
    xmask = _full_mask(xd, xm, xp)
    if not allow_masked and bool(xmask.any()):
        raise ValueError("Cannot process masked data.")
    if xd.shape[0] == 1:
        rowvar = True
    rowvar = int(bool(rowvar))
    axis = 1 - rowvar
    tup = (slice(None), None) if rowvar else (None, slice(None))
    if y is None:
        big = xd.shape[0] > 2 ** 24 or xd.shape[1] > 2 ** 24
        xnotmask = xp.logical_not(xmask).astype(_np.float64 if big else _np.float32)
    else:
        yd, ym = _ndmin2(y, xp)
        ymask = _full_mask(yd, ym, xp)
        if not allow_masked and bool(ymask.any()):
            raise ValueError("Cannot process masked data.")
        if yd.shape == xd.shape and (xm is not nomask or ym is not nomask):
            xmask = ymask = xmask | ymask
            xm = ym = xmask
        xd = xp.concatenate((xd, yd), axis)
        xm = xp.concatenate((xmask, ymask), axis) if (xm is not nomask or ym is not nomask) else nomask
        xmask = _full_mask(xd, xm, xp)
        big = xd.shape[0] > 2 ** 24 or xd.shape[1] > 2 ** 24
        xnotmask = xp.logical_not(xmask).astype(_np.float64 if big else _np.float32)
    xm_ = _c._wrap(xd, xm)
    xm_ = xm_ - xm_.mean(axis=rowvar)[tup]
    return xm_, xnotmask, rowvar


def cov(x, y=None, rowvar=True, bias=False, allow_masked=True, ddof=None):
    """Covariance matrix of masked data (numpy.ma semantics).

    Entries estimated from no sample (``N - ddof <= 0``) are masked.  Runs
    on the device of the inputs without host synchronisation (except for the
    ``allow_masked=False`` check).
    """
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be an integer")
    if ddof is None:
        ddof = 0 if bias else 1
    x, xnotmask, rowvar = _covhelper(x, y, rowvar, allow_masked)
    xp = _get_xp(x)
    z = x.filled(0)
    with _np.errstate(divide="ignore", invalid="ignore"):  # as numpy.ma: masked entries (fact <= 0) are expected
        if not rowvar:
            fact = xp.dot(xnotmask.T, xnotmask) - ddof
            data = xp.dot(z.T, z.conj()) / fact
        else:
            fact = xp.dot(xnotmask, xnotmask.T) - ddof
            data = xp.dot(z, z.T.conj()) / fact
    return _c._wrap(data, fact <= 0).squeeze()


def corrcoef(x, y=None, rowvar=True, allow_masked=True):
    """Pearson correlation coefficients of masked data (numpy.ma semantics).

    A scalar covariance gives ``masked``, as numpy.ma does.
    """
    corr = cov(x, y, rowvar, allow_masked=allow_masked)
    if corr.ndim < 2:
        return _c.masked
    with _np.errstate(divide="ignore", invalid="ignore"):
        std = _np.sqrt(corr.diagonal())
        corr /= std[:, None] * std[None, :]
    return corr


# ---------------------------------------------------------------------------
# enumeration, edges and clumps of unmasked values
# ---------------------------------------------------------------------------
def ndenumerate(a, compressed=True):
    """Iterate over ``(index, value)`` of the elements of ``a`` (C order, host iteration).

    Masked elements are skipped, or yielded as ``masked`` if ``compressed``
    is False.  The data and mask are copied to the host once (a synchronisation).
    """
    _, d, m = _parts(a)
    hd = _backend.to_host(d)
    hm = _backend.to_host(_full_mask(d, m, _get_xp(d)))
    for it, mask in zip(_np.ndenumerate(hd), hm.flat):
        if not mask:
            yield it
        elif not compressed:
            yield it[0], _c.masked


def flatnotmasked_edges(a):
    """First and last unmasked flat indices of ``a`` as a 2-array, or None if all are masked.

    Synchronises with the device.  The array lives on the device of ``a``.
    """
    xp, d, m = _parts(a)
    if m is nomask or not bool(m.any()):
        return xp.array([0, d.size - 1])
    unmasked = xp.flatnonzero(~m)
    if len(unmasked) > 0:
        return unmasked[[0, -1]]
    return None


def notmasked_edges(a, axis=None):
    """Indices of the first and last unmasked values along ``axis`` (see ``numpy.ma.notmasked_edges``).

    Synchronises with the device (data-dependent output size).
    """
    a = _asma(a)
    if axis is None or a.ndim == 1:
        return flatnotmasked_edges(a)
    xp, d, m = _parts(a)
    m = _full_mask(d, m, xp)
    valid = ~m.all(axis=axis)
    ind = xp.indices(d.shape)
    big = _np.iinfo(ind.dtype).max
    low = tuple(xp.where(m, big, ind[i]).min(axis=axis)[valid] for i in range(d.ndim))
    high = tuple(xp.where(m, -1, ind[i]).max(axis=axis)[valid] for i in range(d.ndim))
    return [low, high]


def flatnotmasked_contiguous(a):
    """List of slices of the contiguous unmasked runs of the flattened ``a``.

    Synchronises with the device (the mask is copied to the host).
    """
    xp, d, m = _parts(a)
    if m is nomask:
        return [slice(0, d.size)]
    i = 0
    result = []
    for k, g in _itertools.groupby(_backend.to_host(m).ravel()):
        n = len(list(g))
        if not k:
            result.append(slice(i, i + n))
        i += n
    return result


def notmasked_contiguous(a, axis=None):
    """Contiguous unmasked runs along ``axis`` of a 1-D or 2-D array (one list per slice).

    Synchronises with the device.
    """
    a = _asma(a)
    nd = a.ndim
    if nd > 2:
        raise NotImplementedError("Currently limited to at most 2D array.")
    if axis is None or nd == 1:
        return flatnotmasked_contiguous(a)
    result = []
    other = (axis + 1) % 2
    idx = [0, 0]
    idx[axis] = slice(None, None)
    for i in range(a.shape[other]):
        idx[other] = i
        result.append(flatnotmasked_contiguous(a[tuple(idx)]))
    return result


def _ezclump(mask):
    """Slices of the runs of True in ``mask`` (flattened); copies the run boundaries to the host."""
    mask = mask.ravel()
    idx = _backend.to_host((mask[1:] ^ mask[:-1]).nonzero()[0]) + 1
    first, last = bool(mask[0]), bool(mask[-1])
    if first:
        if len(idx) == 0:
            return [slice(0, mask.size)]
        r = [slice(0, idx[0])]
        r.extend(slice(left, right) for left, right in zip(idx[1:-1:2], idx[2::2]))
    else:
        if len(idx) == 0:
            return []
        r = [slice(left, right) for left, right in zip(idx[:-1:2], idx[1::2])]
    if last:
        r.append(slice(idx[-1], mask.size))
    return r


def clump_unmasked(a):
    """Slices of the contiguous unmasked runs of the flattened ``a`` (synchronises with the device)."""
    mask = getattr(a, "_mask", nomask)
    if mask is nomask:
        return [slice(0, a.size)]
    return _ezclump(~mask)


def clump_masked(a):
    """Slices of the contiguous masked runs of the flattened ``a`` (synchronises with the device)."""
    mask = _c._unpack(a)[1]
    if mask is nomask:
        return []
    return _ezclump(_backend.asarray(mask, _get_xp(a, mask)))


# ---------------------------------------------------------------------------
# polynomial fit
# ---------------------------------------------------------------------------
def vander(x, n=None):
    """Vandermonde matrix of ``x``; rows of masked entries are zero.

    Returns a plain array (as ``numpy.ma.vander``) on the device of ``x``.
    """
    xp, d, m = _parts(x)
    if d.ndim != 1:
        raise ValueError("x must be a one-dimensional array or sequence.")
    N = d.shape[0] if n is None else n
    if N < 0:
        raise ValueError("N must be non-negative")
    dt = _np.promote_types(d.dtype, int)
    # same accumulation as numpy (cupy's own vander uses ``pow`` and differs in the last bits)
    cols = xp.ones((d.shape[0], int(N > 0)), dt)
    if N > 1:
        cols = xp.concatenate([cols, xp.cumprod(xp.broadcast_to(d.astype(dt)[:, None], (d.shape[0], N - 1)), axis=1)], 1)
    v = cols[:, ::-1]
    if m is not nomask:
        v = xp.where(m[:, None], xp.zeros((), v.dtype), v)
    return v


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """Least-squares polynomial fit ignoring masked points (``numpy.ma.polyfit``).

    Points masked in ``x``, ``y`` (any column of a row for 2-D ``y``) or ``w``
    are removed, then ``polyfit`` of the device array module is used.  Removing
    points is a host synchronisation when any mask is present.
    """
    xp = _xp_of(x, y, w)
    x, y = _asma(x, xp), _asma(y, xp)
    m = x._mask
    if y.ndim == 1:
        m = _c._mask_or(m, y._mask, xp)
    elif y.ndim == 2:
        my = mask_rows(y)._mask
        if my is not nomask:
            m = _c._mask_or(m, my[:, 0], xp)
    else:
        raise TypeError("Expected a 1D or 2D array for y!")
    if w is not None:
        w = _asma(w, xp)
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        m = _c._mask_or(m, w._mask, xp)
    xd, yd, wd = x._data, y._data, None if w is None else w._data
    if m is not nomask:
        keep = ~m
        xd, yd = xd[keep], yd[keep]
        wd = None if wd is None else wd[keep]
    return xp.polyfit(xd, yd, deg, rcond, full, wd, cov)
