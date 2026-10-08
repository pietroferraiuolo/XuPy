"""
Reductions, scans and sorting of XuPy masked arrays (``numpy.ma`` semantics).

Every reduction fills the masked slots with the operation identity (or the
min/max fill value), reduces on the data's own backend and computes the result
mask as ``mask.all(axis, keepdims)``.  There is no boolean compaction and no
data-dependent host synchronisation; the only sync is the final scalar
conversion of 0-d results (a numpy scalar, or ``masked`` when masked).

Known differences from numpy.ma (all about *nomask-ness*, never values):
``var`` of a partly masked array along an axis keeps an all-False mask array
where numpy.ma shrinks it to ``nomask`` (that needs a sync); ``ptp(out=...)``
likewise leaves ``out`` with an all-False mask.  ``var``/``std`` of unmasked
integer data with ``dtype=int64`` (or complex) and ``ddof >= count`` mask the
lanes where numpy returns a garbage/unmasked nan.

Notes
-----
cupy sorts accept neither ``order`` nor a meaningful ``kind``: those are only
forwarded to numpy data (cupy's sort is not guaranteed stable).
"""
from __future__ import annotations

import warnings as _warnings

import numpy as _np

from . import _backend
from ._singletons import MaskError, masked as _masked, nomask

__all__ = ["sort", "argsort"]

_NV = _np._NoValue


def _core():
    from . import core as _c

    return _c


def _out_array(out, xp):
    """Raw ``out`` array to pass to a numpy reduction, else None.

    cupy reductions are not given ``out`` (cupy 14's cub min/max reject it):
    `_emit` copies the result there instead.
    """
    if out is None or xp is not _np:
        return None
    arr = getattr(out, "_data", out)
    return arr if _backend.get_xp(arr) is xp else None


def _kw(keepdims):
    """``keepdims`` as a bool (``_NoValue`` -> False)."""
    return False if keepdims is _NV else bool(keepdims)


def _check_sort_args(dtype, kind, order):
    """Validate ``kind``/``order`` as ndarray.sort does (cupy ignores them): same exceptions."""
    if kind is not None or order is not None:
        _np.empty(0, dtype=dtype).sort(kind=kind, order=order)


def _nd_mean(xp, arr, axis, dtype, keepdims, items):
    """``ndarray.mean`` (numpy ``_methods._mean``) on a non-numpy backend (cupy's differs for ``dtype=``)."""
    half = False
    if dtype is None:
        if arr.dtype.kind in "iub":
            dtype = _np.dtype("f8")
        elif arr.dtype == _np.float16:
            dtype, half = _np.dtype("f4"), True
    ret = xp.asarray(xp.sum(arr, axis=axis, dtype=dtype, keepdims=keepdims))
    wide = xp.result_type(ret.dtype, _np.float64)
    out = xp.true_divide(ret.astype(wide), items)
    return out.astype(arr.dtype if half else ret.dtype)


def _nd_var(xp, arr, axis, dtype, ddof, keepdims, mean, items):
    """``ndarray.var`` (numpy ``_methods._var``) on a non-numpy backend; ``items`` is the lane size."""
    kind = arr.dtype.kind
    if dtype is None and kind in "iub":
        dtype = _np.dtype("f8")

    def _div(a, n):  # true_divide(a, intp-scalar, out=a, casting="unsafe")
        wide = xp.result_type(a.dtype, _np.float64)
        return xp.true_divide(a.astype(wide), n).astype(a.dtype)

    if mean is not _NV:
        arrmean = _backend.asarray(_core()._unpack(mean)[0], xp)
    else:
        arrmean = _div(xp.sum(arr, axis=axis, dtype=dtype, keepdims=True), items)
    if arr.dtype == _np.bool_ and arrmean.dtype == _np.bool_:
        raise TypeError("numpy boolean subtract, the `-` operator, is not supported, "
                        "use the bitwise_xor, the `^` operator, or the logical_xor function instead.")
    x = arr - arrmean
    if kind in "fiu":
        x = x * x
    elif x.dtype.kind == "c":
        x = x.real * x.real + x.imag * x.imag
    else:
        x = (x * x.conj()).real
    ret = xp.asarray(xp.sum(x, axis=axis, dtype=dtype, keepdims=keepdims))
    return _div(ret, max(items - ddof, 0))


class _ReductionsMixin:
    """Reductions, scans, sorting and counting for ``_XupyMaskedArray``."""

    def _ax(self, axis):
        """0-d data accept ``axis`` 0 / -1 as ``None`` (numpy does; cupy raises AxisError)."""
        if self._data.ndim == 0 and isinstance(axis, int) and not isinstance(axis, bool) and axis in (0, -1):
            return None
        return axis

    # ---- internals -------------------------------------------------------
    def _filled_with(self, value):
        """Data with the masked slots set to ``value`` (data itself if no mask)."""
        if self._mask is nomask:
            return self._data
        xp = self._xp
        return xp.where(self._mask, xp.asarray(value, dtype=self._data.dtype), self._data)

    def _check_fv(self, fill_value, default):
        if fill_value is None:
            return default
        return _core()._check_fill_value(fill_value, self._data.dtype)

    def _reduce(self, name, axis, dtype, keepdims, fill, out=None):
        """Reduce the filled data with ``xp.<name>``; return ``(result, mask)``.

        ``out`` is written directly for numpy data (numpy then applies its own
        casting/shape rules); otherwise `_emit` copies.
        """
        xp = self._xp
        axis = self._ax(axis)
        kw = {} if dtype is None else {"dtype": dtype}
        tgt = _out_array(out, xp)
        if tgt is not None:
            kw["out"] = tgt
        res = xp.asarray(
            getattr(xp, name)(self._filled_with(fill), axis=axis, keepdims=keepdims, **kw)
        )
        m = self._mask
        if m is not nomask:
            m = xp.asarray(m.all(axis=axis, keepdims=keepdims))
        return res, m

    def _emit(self, res, mask, out=None, *, like=None, plain_out=None, keep_nomask=False):
        """Package ``(res, mask)`` as scalar / masked array, or store into ``out``.

        ``plain_out="nan"``: NaN where masked, MaskError for integer ``out``
        (non-masked ``out``, numpy.ma min/max).  ``keep_nomask``: a ``nomask``
        result leaves a ``nomask`` ``out`` untouched (numpy.ma ``__setmask__``).
        """
        c = _core()
        if out is None:
            if res.ndim:
                return c._wrap(res, mask, like=like)
            return c._maybe_scalar(res, mask)
        is_ma = isinstance(out, c._XupyMaskedArray)
        target = out._data if is_ma else out
        oxp = _backend.get_xp(target)
        if res is not target:
            if res.shape != target.shape:
                raise ValueError(f"output array has shape {target.shape}, expected {res.shape}")
            oxp.copyto(target, _backend.asarray(res, oxp), casting="unsafe")
        if is_ma:
            if mask is nomask and (keep_nomask and out._mask is nomask):
                return out
            if out._mask is nomask:
                out._mask = oxp.zeros(target.shape, dtype=bool)
            out._mask[...] = False if mask is nomask else _backend.asarray(mask, oxp)
        elif plain_out == "nan":
            if out.dtype.kind in "biu":
                raise MaskError("Masked data information would be lost in one or more location.")
            if mask is not nomask:
                oxp.copyto(out, _np.nan, where=_backend.asarray(mask, oxp))
        return out

    def _items(self, axis, keepdims):
        """Static ``(items per lane, lane shape)`` of a reduction (no device work)."""
        shape = self._data.shape
        if shape == ():
            if axis not in (None, 0):
                raise _np.exceptions.AxisError(axis, 0)
            return 1, ()
        axes = tuple(range(len(shape))) if axis is None else _np.lib.array_utils.normalize_axis_tuple(axis, len(shape))
        items = 1
        for a in axes:
            items *= shape[a]
        return items, tuple(1 if i in axes else n for i, n in enumerate(shape) if keepdims or i not in axes)

    def _count_raw(self, axis, keepdims):
        """Unmasked count as an ``intp`` array of the data's backend."""
        if self._mask is not nomask:
            axis = self._ax(axis)
            return (~self._mask).sum(axis=axis, dtype=_np.intp, keepdims=keepdims)
        items, shape = self._items(axis, keepdims)
        return self._xp.full(shape, items, dtype=_np.intp)

    # ---- sum / prod / any / all -----------------------------------------
    def sum(self, axis=None, dtype=None, out=None, keepdims=_NV):
        """Sum of the unmasked elements over the given axis."""
        res, m = self._reduce("sum", axis, dtype, _kw(keepdims), 0, out)
        return self._emit(res, m, out)

    def prod(self, axis=None, dtype=None, out=None, keepdims=_NV):
        """Product of the unmasked elements over the given axis."""
        res, m = self._reduce("prod", axis, dtype, _kw(keepdims), 1, out)
        return self._emit(res, m, out)

    product = prod

    def any(self, axis=None, out=None, keepdims=_NV):
        """True where any unmasked element is true (masked count as False)."""
        res, m = self._reduce("any", axis, None, _kw(keepdims), False, out)
        return self._emit(res, m, out, keep_nomask=True)

    def all(self, axis=None, out=None, keepdims=_NV):
        """True where all unmasked elements are true (masked count as True)."""
        res, m = self._reduce("all", axis, None, _kw(keepdims), True, out)
        return self._emit(res, m, out, keep_nomask=True)

    # ---- count -----------------------------------------------------------
    def count(self, axis=None, keepdims=_NV):
        """Number of unmasked elements (numpy ``intp`` scalar, or array)."""
        kd = _kw(keepdims)
        if self._mask is nomask and (self._data.ndim == 0 or (axis is None and not kd)):
            return self._items(axis, kd)[0]  # python int, as numpy.ma
        res = self._count_raw(axis, kd)
        if self._mask is nomask:
            return res  # numpy.ma: a 0-d array when `axis` is given
        return _backend.host_scalar(res) if res.ndim == 0 else res

    def count_unmasked(self, axis=None):
        """Number of unmasked elements along ``axis`` (alias of `count`)."""
        return self.count(axis)

    def count_masked(self, axis=None):
        """Number of masked elements along ``axis``."""
        if self._mask is nomask:
            res = self._xp.zeros_like(self._count_raw(axis, False))
        else:
            res = self._mask.sum(axis=self._ax(axis), dtype=_np.intp)
        return _backend.host_scalar(res) if res.ndim == 0 else res

    # ---- mean / var / std / anom ----------------------------------------
    def _mean_raw(self, axis, dtype, keepdims):
        xp = self._xp
        if self._mask is nomask:  # ndarray.mean: axis 0 of a 0-d array is an AxisError (not normalised)
            if xp is _np:
                return xp.asarray(xp.mean(self._data, axis=axis, dtype=dtype, keepdims=keepdims)), nomask
            return _nd_mean(xp, self._data, axis, dtype, keepdims, self._items(axis, keepdims)[0]), nomask
        half = False
        if dtype is None:
            kind = self._data.dtype.kind
            if kind in "biu":
                dtype = _np.dtype("f8")
            elif self._data.dtype == _np.float16:
                dtype, half = _np.dtype("f4"), True
        dsum, m = self._reduce("sum", axis, dtype, keepdims, 0)
        cnt = self._count_raw(axis, keepdims)
        res = xp.true_divide(xp.multiply(dsum, 1.0), xp.where(cnt == 0, 1, cnt))
        if half:
            res = res.astype(self._data.dtype)
        if res.ndim:  # numpy.ma divides with a masked divide: non-finite results are masked
            m = xp.logical_or(m, ~xp.isfinite(res))
        return res, m

    def mean(self, axis=None, dtype=None, out=None, keepdims=_NV):
        """Mean of the unmasked elements (masked if there are none)."""
        res, m = self._mean_raw(axis, dtype, _kw(keepdims))
        return self._emit(res, m, out)

    def _var_raw(self, axis, dtype, ddof, keepdims, mean):
        xp = self._xp
        data, mask = self._data, self._mask
        if mask is nomask:
            if axis is not None:  # invalid axes: AxisError as ndarray.var (cupy raises IndexError)
                _np.lib.array_utils.normalize_axis_tuple(axis, data.ndim)
            items, _ = self._items(axis, keepdims)
            if xp is _np:
                kw = {} if mean is _NV else {"mean": _core()._unpack(mean)[0]}
                with _np.errstate(all="ignore"):
                    res = xp.asarray(xp.var(data, axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kw))
            else:  # cupy.var differs from ndarray.var for dtype=/mean=: follow numpy's algorithm
                res = _nd_var(xp, data, axis, dtype, ddof, keepdims, mean, items)
            # ddof >= count: numpy.ma masks the lanes (a static fact here), except for
            # bool data where ndarray.var's own division leaves an unmasked nan
            left = items - ddof
            masked_all = items and left <= 0 and not (data.dtype.kind == "b" and (dtype is None or _np.dtype(dtype).kind in "fc"))
            return res, (xp.ones(res.shape, dtype=bool) if masked_all else nomask)
        axis = self._ax(axis)
        c = _core()
        cnt = self._count_raw(axis, keepdims) - ddof
        if mean is not _NV:
            mdata, mmask = c._unpack(mean)[0], nomask
        else:
            mdata, mmask = self._mean_raw(axis, dtype, True)
        danom = data - _backend.asarray(mdata, xp)
        danom = xp.abs(danom) ** 2 if data.dtype.kind == "c" else danom * danom
        # danom is masked where the data or the mean is
        dmask = mask if mmask is nomask else xp.logical_or(mask, mmask)
        zero = xp.zeros((), dtype=danom.dtype)
        scalar = cnt.ndim == 0
        bad = (cnt == 0) if scalar else (cnt <= 0)
        with _np.errstate(all="ignore"):
            res = xp.true_divide(xp.where(dmask, zero, danom).sum(axis=axis, keepdims=keepdims),
                                 xp.where(bad, 1, cnt))
        if scalar:  # divide of scalars: all-masked, count == 0 or non-finite masks
            newmask = xp.logical_or(xp.logical_or(dmask.all(axis=axis, keepdims=keepdims), bad), ~xp.isfinite(res))
            return xp.asarray(res), xp.asarray(newmask)
        # numpy.ma shrinks an all-False mask to nomask here (a host sync): we keep the array
        return xp.asarray(res), xp.asarray(xp.logical_or(mask.all(axis=axis, keepdims=keepdims), bad))

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=_NV, mean=_NV):
        """Variance of the unmasked elements; masked where ``count <= ddof``."""
        res, m = self._var_raw(axis, dtype, ddof, _kw(keepdims), mean)
        return self._emit(res, m, out, like=self, plain_out=None if m is nomask else "nan", keep_nomask=True)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=_NV, mean=_NV):
        """Standard deviation of the unmasked elements (``sqrt`` of `var`).

        ``mean`` is accepted but ignored, as in numpy.ma 2.5.
        """
        xp = self._xp
        res, m = self._var_raw(axis, dtype, ddof, _kw(keepdims), _NV)
        if out is not None:  # numpy.ma: sqrt in place, mask as set by var
            root = xp.sqrt(res if m is nomask else xp.where(m, xp.zeros((), dtype=res.dtype), res))
            return self._emit(root, m, out, like=self, plain_out="nan", keep_nomask=True)
        with _np.errstate(all="ignore"):
            root = xp.sqrt(res)
        bad = ~xp.isfinite(root)  # sqrt is a domained op: non-finite results are masked
        m = bad if m is nomask else xp.logical_or(m, bad)
        return self._emit(xp.where(bad, res, root), m, None, like=self)

    def anom(self, axis=None, dtype=None):
        """Deviations from the arithmetic mean along ``axis``."""
        m = self.mean(axis, dtype)
        if not axis or not hasattr(m, "expand_dims"):
            return self - m
        return self - m.expand_dims(axis)

    # ---- min / max / ptp -------------------------------------------------
    def _extremum(self, name, axis, out, fill_value, keepdims):
        dt = self._data.dtype
        default = (_np.ma.minimum_fill_value if name == "min" else _np.ma.maximum_fill_value)(dt)
        fv = self._check_fv(fill_value, default)
        res, m = self._reduce(name, axis, None, keepdims, fv, out)
        if out is None and m is not nomask and res.ndim:
            xp = self._xp
            res = xp.where(m, xp.asarray(_core()._check_fill_value(None, dt), dtype=dt), res)
        return res, m

    def min(self, axis=None, out=None, fill_value=None, keepdims=_NV):
        """Minimum of the unmasked elements (masked lanes: fully masked)."""
        res, m = self._extremum("min", axis, out, fill_value, _kw(keepdims))
        return self._emit(res, m, out, plain_out="nan")

    def max(self, axis=None, out=None, fill_value=None, keepdims=_NV):
        """Maximum of the unmasked elements (masked lanes: fully masked)."""
        res, m = self._extremum("max", axis, out, fill_value, _kw(keepdims))
        return self._emit(res, m, out, plain_out="nan")

    def ptp(self, axis=None, out=None, fill_value=None, keepdims=False):
        """Peak-to-peak (``max - min``) of the unmasked elements."""
        hi, m = self._extremum("max", axis, None, fill_value, keepdims)
        lo, _ = self._extremum("min", axis, None, fill_value, keepdims)
        try:
            diff = hi - lo
        except TypeError:  # bool data
            # numpy.ma: an array result raises; a fully masked scalar is still `masked`
            # (decided at the final scalar conversion, which syncs anyway)
            if hi.ndim or out is not None or self._emit(hi, m) is not _masked:
                raise
            return _masked
        return self._emit(diff, m, out, keep_nomask=True)

    # ---- arg-reductions ---------------------------------------------------
    def _arg(self, name, axis, fill_value, out, keepdims):
        dt = self._data.dtype
        default = (_np.ma.minimum_fill_value if name == "argmin" else _np.ma.maximum_fill_value)(dt)
        fv = self._check_fv(fill_value, default)
        xp = self._xp
        axis = self._ax(axis)
        res = xp.asarray(getattr(xp, name)(self._filled_with(fv), axis=axis, keepdims=_kw(keepdims)))
        if res.dtype.type is not _np.intp:  # cupy returns 'q' (longlong) for e.g. bool data
            res = res.astype(_np.intp)
        if out is not None:
            _backend.get_xp(out).copyto(out, _backend.asarray(res, _backend.get_xp(out)), casting="unsafe")
            return out
        return _backend.host_scalar(res) if res.ndim == 0 else res

    def argmin(self, axis=None, fill_value=None, out=None, *, keepdims=_NV):
        """Index of the minimum, masked values filled with ``fill_value``."""
        return self._arg("argmin", axis, fill_value, out, keepdims)

    def argmax(self, axis=None, fill_value=None, out=None, *, keepdims=_NV):
        """Index of the maximum, masked values filled with ``fill_value``."""
        return self._arg("argmax", axis, fill_value, out, keepdims)

    # ---- scans ------------------------------------------------------------
    def _scan(self, name, fill, axis, dtype, out):
        xp = self._xp
        axis = self._ax(axis)
        res = xp.asarray(getattr(xp, name)(self._filled_with(fill), axis=axis, dtype=dtype, out=_out_array(out, xp)))
        m = self._mask
        if m is not nomask:
            m = xp.array(m.ravel() if axis is None else m, copy=True)
        return self._emit(res, m, out, keep_nomask=True)

    def cumsum(self, axis=None, dtype=None, out=None):
        """Cumulative sum (masked slots count as 0; the mask is kept)."""
        return self._scan("cumsum", 0, axis, dtype, out)

    def cumprod(self, axis=None, dtype=None, out=None):
        """Cumulative product (masked slots count as 1; the mask is kept)."""
        return self._scan("cumprod", 1, axis, dtype, out)

    # ---- sorting ----------------------------------------------------------
    @staticmethod
    def _no_stable_descending(stable, descending):
        if stable:
            raise ValueError("`stable` parameter is not supported for masked arrays.")
        if descending:
            raise ValueError("`descending` parameter is not supported for masked arrays.")

    def argsort(self, axis=_NV, kind=None, order=None, endwith=True, fill_value=None,
                *, stable=False, descending=False):
        """Indices that sort the array, masked values last (``endwith``)."""
        self._no_stable_descending(stable, descending)
        if axis is _NV:
            if self.ndim <= 1:
                axis = -1
            else:
                _warnings.warn(
                    "In the future the default for argsort will be axis=-1, not the "
                    "current None, to match its documentation and np.argsort. "
                    "Explicitly pass -1 or None to silence this warning.",
                    _np.ma.core.MaskedArrayFutureWarning, stacklevel=2)
                axis = None
        dt = self._data.dtype
        if fill_value is None:
            if not endwith:
                fill_value = _np.ma.maximum_fill_value(dt)
            elif _np.issubdtype(dt, _np.floating):
                fill_value = _np.nan
            else:
                fill_value = _np.ma.minimum_fill_value(dt)
        d = self._filled_with(self._check_fv(fill_value, None))
        xp = self._xp
        if xp is _np:
            return d.argsort(axis=axis, kind=kind, order=order)
        _check_sort_args(dt, kind, order)
        return xp.argsort(d.ravel() if axis is None else d, axis=-1 if axis is None else axis)

    def sort(self, axis=-1, kind=None, order=None, endwith=True, fill_value=None,
             *, stable=False, descending=False):
        """Sort in place along ``axis``; the mask moves with the data."""
        self._no_stable_descending(stable, descending)
        xp = self._xp
        if self._mask is nomask:
            if xp is _np:
                self._data.sort(axis=axis, kind=kind, order=order)
            else:
                _check_sort_args(self._data.dtype, kind, order)
                self._data.sort(axis=axis)
            return
        if axis is None:  # numpy.ma: take_along_axis(self, idx, None) fails
            raise ValueError("`indices` and `arr` must have the same number of dimensions")
        sidx = self.argsort(axis=axis, kind=kind, order=order, fill_value=fill_value, endwith=endwith)
        c = _core()
        self[...] = c._wrap(xp.take_along_axis(self._data, sidx, axis=axis),
                            xp.take_along_axis(self._mask, sidx, axis=axis))

    # ---- misc ---------------------------------------------------------------
    def nonzero(self):
        """Indices of the unmasked, non-zero elements (tuple of arrays)."""
        return self._filled_with(0).nonzero()

    def clip(self, min=None, max=None, out=None, **kwargs):
        """Clip the data to ``[min, max]``; masks of the bounds are merged in."""
        c = _core()
        xp = _backend.get_xp(self, min, max)
        mask = nomask
        bounds = []
        for b in (min, max):
            if b is not None and not isinstance(b, (int, float, complex, _np.generic)):
                b, bm = c._unpack(b)  # python/numpy scalars stay weakly typed, as in numpy
                b = _backend.asarray(b, xp)
                mask = self._mask_union(mask, bm, xp)
            bounds.append(b)
        data = _backend.asarray(self._data, xp)
        if xp is _np:
            res = xp.asarray(xp.clip(data, *bounds, **kwargs))
        else:  # cupy.clip ignores NaN bounds; numpy's propagates them (minimum/maximum do)
            lo, hi = bounds
            if data.dtype.kind in "iu":  # numpy's `_clip`: python-int bounds beyond the dtype are no-ops
                info = _np.iinfo(data.dtype)
                lo = None if type(lo) is int and lo <= info.min else lo
                hi = None if type(hi) is int and hi >= info.max else hi
            if lo is None and hi is None and data.dtype.kind == "b":
                _np.positive(_np.empty(0, bool))  # numpy's identity clip is ufunc positive: raises for bool
            res = data
            if lo is not None:
                res = xp.maximum(res, lo)
            if hi is not None:
                res = xp.minimum(res, hi)
            res = xp.array(res, copy=True) if res is data else xp.asarray(res)
        mask = self._mask_union(mask, self._mask, xp)
        if mask is nomask and min is None and max is None:
            mask = xp.zeros(res.shape, dtype=bool)  # numpy.ma materialises it here
        if mask is not nomask:
            mask = xp.broadcast_to(mask, res.shape).copy()
        if out is not None:
            return self._emit(res, mask, out)
        if res.ndim == 0 and mask is not nomask and _backend.host_scalar(mask):
            return _masked  # numpy.ma: a fully masked 0-d result is the constant
        return c._wrap(res, mask, like=self)

    @staticmethod
    def _mask_union(m1, m2, xp):
        """``m1 | m2`` for masks of possibly different shapes (nomask-aware)."""
        if m2 is nomask:
            return m1
        m2 = _backend.asarray(m2, xp)
        return m2 if m1 is nomask else xp.logical_or(m1, m2)


# ---------------------------------------------------------------------------
# module-level sort / argsort (numpy.ma.sort, numpy.ma.argsort)
# ---------------------------------------------------------------------------
def _asanyarray(a):
    """Masked arrays pass; host/cupy arrays stay on their device; lists go to the active backend."""
    c = _core()
    if isinstance(a, c._XupyMaskedArray):
        return a
    if isinstance(a, _np.ma.MaskedArray) or getattr(a, "_is_xupy_masked_constant", False):
        return c.MaskedArray(a)
    if isinstance(a, _np.ndarray) or _backend.is_cupy_array(a):
        return a
    return _backend.default_xp().asarray(a)


def sort(a, axis=-1, kind=None, order=None, endwith=True, fill_value=None, *,
         stable=None, descending=None):
    """Return a sorted copy of ``a``, masked values last (``endwith``); see `MaskedArray.sort`.

    A plain array gives a plain sorted array, a masked array a masked one.  ``axis=None``
    sorts the flattened array.  No host sync.
    """
    a = _asanyarray(a)
    c = _core()
    xp = _backend.get_xp(a)
    a = a.copy() if isinstance(a, c._XupyMaskedArray) else xp.array(a, copy=True)
    if axis is None:
        a = a.flatten()
        axis = 0
    if isinstance(a, c._XupyMaskedArray):
        a.sort(axis=axis, kind=kind, order=order, endwith=endwith, fill_value=fill_value,
               stable=stable, descending=descending)
    elif xp is _np:
        kw = {k: v for k, v in (("stable", stable), ("descending", descending)) if v is not None}
        a.sort(axis=axis, kind=kind, order=order, **kw)
    else:
        _check_sort_args(a.dtype, kind, order)
        a.sort(axis=axis)
        if descending:
            a[...] = xp.flip(a, axis)
    return a


def argsort(a, axis=_NV, kind=None, order=None, endwith=True, fill_value=None, *,
            stable=None, descending=None):
    """Indices that sort ``a`` (masked values last); see `MaskedArray.argsort`.

    With ``axis`` omitted a 2-d or larger array is flattened (FutureWarning), as numpy.ma.
    """
    a = _asanyarray(a)
    if axis is _NV:
        if a.ndim <= 1:
            axis = -1
        else:
            _warnings.warn(
                "In the future the default for argsort will be axis=-1, not the "
                "current None, to match its documentation and np.argsort. "
                "Explicitly pass -1 or None to silence this warning.",
                _np.ma.core.MaskedArrayFutureWarning, stacklevel=2)
            axis = None
    if isinstance(a, _core()._XupyMaskedArray):
        return a.argsort(axis=axis, kind=kind, order=order, endwith=endwith,
                         fill_value=fill_value, stable=stable, descending=descending)
    xp = _backend.get_xp(a)
    if xp is _np:
        kw = {k: v for k, v in (("stable", stable), ("descending", descending)) if v is not None}
        return a.argsort(axis=axis, kind=kind, order=order, **kw)
    if descending:
        raise NotImplementedError("argsort(descending=True) of a plain cupy array is not supported")
    _check_sort_args(a.dtype, kind, order)
    return xp.argsort(a.ravel() if axis is None else a, axis=-1 if axis is None else axis)
