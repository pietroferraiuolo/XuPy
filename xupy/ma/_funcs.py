"""
Module-level ``numpy.ma.core`` functions that are not masked ufuncs
(``numpy.ma.masked_where``, ``numpy.ma.arange``, ``numpy.ma.diff``, ...).

Same rules as the rest of ``xupy.ma``: the device follows the operands (host
input and creation from Python data go to the active backend, see
``_backend``), shared helpers for masks/fill values, no host sync unless the
output size is data-dependent or the result is a scalar.  The names listed in
``__all__`` are re-exported by ``xupy.ma.core`` and ``xupy.ma``.

Known differences from numpy.ma (nomask-ness, never values): ``masked_where``
and the ``masked_<cmp>`` family, ``masked_invalid``, ``fix_invalid`` and
``outer`` do not shrink an all-False mask to ``nomask`` (that needs a device
sync); ``masked_values(shrink=True)`` does, with one sync.  ``indices(sparse=True)``
returns a tuple of masked arrays (numpy.ma raises ``AttributeError``).
Structured, record and object features raise ``NotImplementedError``.
"""
from __future__ import annotations

import math as _math
import warnings as _warnings

import numpy as _np

from . import _backend, _ops
from ._singletons import masked as _masked, nomask

__all__: list = []

_NV = _np._NoValue
_UNSUPPORTED = "not supported by xupy.ma: numeric and bool dtypes only"
_MAFutureWarning = _np.ma.core.MaskedArrayFutureWarning


def _core():
    from . import core

    return core


def _is_xma(x):
    return getattr(x, "_is_xupy_masked", False)


def _check_dtype(dtype):
    if _np.dtype(dtype).kind not in "biufc":
        raise NotImplementedError(f"dtype {_np.dtype(dtype)} is not supported: only numeric and boolean data.")


def _no_like(like, device=None):
    if like is not None:
        raise NotImplementedError("the `like` argument is not supported by xupy.ma.")
    if device is not None:
        raise NotImplementedError("the `device` argument is not supported by xupy.ma.")


def _to_dev(m, xp):
    """Masked array ``m`` moved to the device of ``xp`` (a view when already there)."""
    if m._xp is xp:
        return m
    c = _core()
    mk = m._mask
    return c._wrap(c._to_xp(m._data, xp), mk if mk is nomask else c._to_xp(mk, xp), like=m)


def _common(*ops):
    """``(xp, [masked arrays])``: host operands go to the active backend, cupy wins."""
    mas = [_coerce(o) for o in ops]
    xp = _backend.get_xp(*mas)
    return xp, [_to_dev(m, xp) for m in mas]


def _full_mask(m, xp):
    return xp.zeros(m.shape, dtype=bool) if m._mask is nomask else m._mask


def _mask_or(a, b, xp):
    """``mask_or`` of the masks of two masked arrays: `nomask` if nothing is masked (one sync)."""
    m = _core()._mask_or(a._mask, b._mask, xp)
    return nomask if m is nomask or not bool(m.any()) else m


def _finish(res, fill_value=None, hardmask=None):
    """Apply the ``fill_value``/``hardmask`` keywords of numpy's ``_convert2ma``."""
    if fill_value is not None:
        res.fill_value = fill_value
    if hardmask is not None:
        res._hardmask = bool(hardmask)
    return res


def _unsupported(name):
    raise NotImplementedError(f"{name} is {_UNSUPPORTED}")


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------
def asarray(a, dtype=None, order=None):
    """Convert ``a`` to a masked array (base class, never a copy unless needed)."""
    return _core().MaskedArray(a, dtype=dtype, copy=False, keep_mask=True, subok=False, order=order or "C")


def asanyarray(a, dtype=None, order=None):
    """Convert ``a`` to a masked array; masked arrays are returned as they are."""
    c = _core()
    if (
        isinstance(a, c._XupyMaskedArray)
        and (dtype is None or dtype == a.dtype)
        and (
            order in (None, "A", "K")
            or (order == "C" and a.flags.c_contiguous)
            or (order == "F" and a.flags.f_contiguous)
        )
    ):
        return a
    return c.MaskedArray(a, dtype=dtype, copy=False, keep_mask=True, subok=True, order=order)


def _hostwrap(a):
    """Operand coercion: host ``ndarray``/``numpy.ma`` operands become numpy-backed masked
    arrays *where they are* (functions follow the device of their operands, they do not
    move host data to the active backend).  Everything else is returned unchanged."""
    c = _core()
    if isinstance(a, _np.ndarray) and not isinstance(a, c._XupyMaskedArray):
        _check_dtype(a.dtype)
        data, mask = c._unpack(a)
        return c._wrap(
            data, mask, fill_value=getattr(a, "_fill_value", None), hardmask=getattr(a, "_hardmask", None)
        )
    return a


def _coerce(a, dtype=None, order=None):
    """Operand-style ``asanyarray``: like it, but host ndarrays stay on the host and
    lists/scalars follow the active backend (cupy and XuPy arrays never move)."""
    return asanyarray(_hostwrap(a), dtype, order)


def isarray(x):
    """True if ``x`` is a masked array (alias of ``isMaskedArray``)."""
    return _core().isMaskedArray(x)


bool_ = _np.bool_
masked_singleton = _masked


# ---------------------------------------------------------------------------
# function forms of MaskedArray methods
# ---------------------------------------------------------------------------
def _frommethod(name, doc=None):
    def func(a, *args, **kwargs):
        return getattr(_coerce(a), name)(*args, **kwargs)

    func.__name__ = func.__qualname__ = name
    func.__doc__ = doc or f"Function form of ``MaskedArray.{name}`` (``a`` is converted to a masked array)."
    return func


all = _frommethod("all")
any = _frommethod("any")
argmax = _frommethod("argmax")
argmin = _frommethod("argmin")
count = _frommethod("count")
cumprod = _frommethod("cumprod")
cumsum = _frommethod("cumsum")
copy = _frommethod("copy")
diagonal = _frommethod("diagonal")
harden_mask = _frommethod("harden_mask")
soften_mask = _frommethod("soften_mask")
ravel = _frommethod("ravel")
repeat = _frommethod("repeat")
swapaxes = _frommethod("swapaxes")
nonzero = _frommethod(
    "nonzero",
    "Indices of the unmasked non-zero elements (output size is data-dependent: one device sync).",
)


def compress(condition, a, axis=None, out=None):
    """Function form of ``MaskedArray.compress`` (output size is data-dependent: one device sync)."""
    return _coerce(a).compress(condition, axis=axis, out=out)


def compressed(x):
    """1-D array of the unmasked values (output size is data-dependent: one device sync)."""
    return _coerce(x).compressed()


def take(a, indices, axis=None, out=None, mode="raise"):
    """Take elements of ``a`` along ``axis`` (see ``MaskedArray.take``)."""
    return _coerce(a).take(indices, axis=axis, out=out, mode=mode)


def put(a, indices, values, mode="raise"):
    """Set storage-indexed locations of ``a`` to ``values`` (in place; see ``MaskedArray.put``)."""
    try:
        return a.put(indices, values, mode=mode)
    except AttributeError:
        return _np.asarray(a).put(indices, values, mode=mode)


def ptp(obj, axis=None, out=None, fill_value=None, keepdims=_NV):
    """Peak-to-peak (maximum - minimum) along ``axis``, masked values ignored."""
    kwargs = {} if keepdims is _NV else {"keepdims": keepdims}
    try:
        return obj.ptp(axis, out=out, fill_value=fill_value, **kwargs)
    except (AttributeError, TypeError):
        return _coerce(obj).ptp(axis=axis, fill_value=fill_value, out=out, **kwargs)


def _plain_reduction(a, name, axis, out, kw):
    """``numpy.amax``/``amin`` of anything that is not a masked array: a plain result."""
    xp = _backend.get_xp(a)
    a = a if isinstance(a, _np.ndarray) or xp is not _np else _np.asarray(a)
    if xp is _np:
        return getattr(_np, name)(a, axis=axis, out=out, **kw)
    res = getattr(xp, name)(a, axis=axis, **{k: v for k, v in kw.items() if k == "keepdims"})
    if out is not None:
        out[...] = res
        return out
    return _backend.host_scalar(res) if res.ndim == 0 else res


def _amaxmin(name, meth):
    def func(a, axis=None, out=None, keepdims=_NV, initial=_NV, where=_NV):
        kw = {k: v for k, v in (("keepdims", keepdims), ("initial", initial), ("where", where)) if v is not _NV}
        if _core().isMaskedArray(a):
            return getattr(_coerce(a), meth)(axis=axis, out=out, **kw)
        return _plain_reduction(a, name, axis, out, kw)

    func.__name__ = func.__qualname__ = name
    func.__doc__ = (
        f"``numpy.{name}`` as in ``numpy.ma``: masked arrays use ``MaskedArray.{meth}`` "
        "(masked values ignored), other inputs give a plain array/scalar."
    )
    return func


amax = _amaxmin("amax", "max")
amin = _amaxmin("amin", "min")


def angle(a, *args, **kwargs):
    """Masked ``numpy.angle`` (the mask is kept)."""
    return _ops._munary("angle", a, *args, **kwargs)


# ---------------------------------------------------------------------------
# shape functions
# ---------------------------------------------------------------------------
def ndim(obj):
    """Number of dimensions of ``obj``."""
    return _core().getdata(obj).ndim


def shape(obj):
    """Shape of ``obj``."""
    return _core().getdata(obj).shape


def size(obj, axis=None):
    """Number of elements of ``obj`` (along ``axis`` if given)."""
    d = _core().getdata(obj)
    if axis is None:
        return d.size
    axes = _np.lib.array_utils.normalize_axis_tuple(axis, d.ndim, allow_duplicate=False)
    return _math.prod(d.shape[ax] for ax in axes)


def reshape(a, new_shape, order="C"):
    """Reshape ``a`` (masked arrays keep their mask, lists become masked arrays)."""
    try:
        return a.reshape(new_shape, order=order)
    except AttributeError:
        return _core().MaskedArray(_np.asarray(a).reshape(new_shape, order=order))


def transpose(a, axes=None):
    """Permute the axes of ``a`` (masked arrays keep their mask, lists become masked arrays)."""
    try:
        return a.transpose(axes)
    except AttributeError:
        return _core().MaskedArray(_np.asarray(a).transpose(axes))


def squeeze(a, axis=None, *, fill_value=_NV, hardmask=_NV):
    """Remove length-1 axes; ``fill_value``/``hardmask`` override those of the result."""
    res = _coerce(a).squeeze(axis)
    if res is a:
        res = res.view()
    if fill_value is not _NV:
        res.fill_value = fill_value
    if hardmask is not _NV:
        res._hardmask = bool(hardmask)
    return res


def resize(x, new_shape):
    """New masked array of ``new_shape``, repeating the data (and mask) of ``x`` as needed."""
    c = _core()
    x = _coerce(x)
    xp = x._xp
    d = xp.resize(x._data, new_shape)
    m = x._mask if x._mask is nomask else xp.resize(x._mask, new_shape)
    return c._wrap(d, m if d.ndim else nomask)


def diag(v, k=0):
    """Extract a diagonal or build a diagonal masked array (the mask follows)."""
    c = _core()
    v = _coerce(v)
    if v.ndim not in (1, 2):
        raise ValueError("Input must be 1- or 2-d.")
    xp = v._xp
    d = xp.diag(v._data, k)
    m = v._mask if v._mask is nomask else xp.diag(v._mask, k)
    return c._wrap(d, m, like=v if v.ndim == 2 else None)


def flatten_mask(mask):
    """Flattened bool version of ``mask`` (a plain array, on the device of ``mask``)."""
    c = _core()
    m = c.getdata(mask)
    if m.dtype.names is not None:
        _unsupported("flatten_mask of structured masks")
    xp = _backend.get_xp(m)
    if xp is _np and not _is_xma(mask):  # host input goes to the active backend
        xp = _backend.default_xp()
    return xp.asarray(m, dtype=bool).reshape(-1)


# ---------------------------------------------------------------------------
# creation (on the active backend, like creation from Python data)
# ---------------------------------------------------------------------------
def _new(data, fill_value=None, hardmask=False):
    return _finish(_core()._wrap(data, nomask), fill_value, hardmask)


def arange(*args, fill_value=None, hardmask=False, **kwargs):
    """Masked ``numpy.arange`` created on the active backend."""
    if kwargs.get("dtype") is not None:
        _check_dtype(kwargs["dtype"])
    _no_like(kwargs.pop("like", None), kwargs.pop("device", None))
    return _new(_backend.default_xp().arange(*args, **kwargs), fill_value, hardmask)


def _filled_shape(name):
    def func(shape, dtype=float, order="C", *, device=None, like=None, fill_value=None, hardmask=False):
        _no_like(like, device)
        _check_dtype(dtype)
        return _new(getattr(_backend.default_xp(), name)(shape, dtype=dtype, order=order), fill_value, hardmask)

    func.__name__ = func.__qualname__ = name
    func.__doc__ = f"Masked ``numpy.{name}`` created on the active backend (no mask)."
    return func


empty = _filled_shape("empty")
ones = _filled_shape("ones")
zeros = _filled_shape("zeros")


def identity(n, dtype=None, *, like=None, fill_value=None, hardmask=False):
    """Masked identity matrix created on the active backend."""
    _no_like(like)
    dtype = float if dtype is None else dtype
    _check_dtype(dtype)
    return _new(_backend.default_xp().identity(n, dtype=dtype), fill_value, hardmask)


def indices(dimensions, dtype=int, sparse=False, *, fill_value=None, hardmask=False):
    """Masked ``numpy.indices`` (a tuple of masked arrays if ``sparse``), on the active backend."""
    _check_dtype(dtype)
    xp = _backend.default_xp()
    if not sparse:
        return _new(xp.indices(dimensions, dtype=dtype), fill_value, hardmask)
    dimensions = tuple(dimensions)
    n = len(dimensions)
    return tuple(
        _new(xp.arange(dim, dtype=dtype).reshape((1,) * i + (dim,) + (1,) * (n - i - 1)), fill_value, hardmask)
        for i, dim in enumerate(dimensions)
    )


def fromfunction(function, shape, *, dtype=float, like=None, **kwargs):
    """Masked ``numpy.fromfunction`` (``function`` gets index arrays of the active backend)."""
    _no_like(like)
    _check_dtype(dtype)
    xp = _backend.default_xp()
    res = function(*xp.indices(shape, dtype=dtype), **kwargs)
    if hasattr(res, "dtype"):
        _check_dtype(res.dtype)
    return _core().MaskedArray(res)


def frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None):
    """Masked ``numpy.frombuffer``: read on the host, then moved to the active backend."""
    _no_like(like)
    _check_dtype(dtype)
    return _core().MaskedArray(_np.frombuffer(buffer, dtype=dtype, count=count, offset=offset))


# ---------------------------------------------------------------------------
# fill values
# ---------------------------------------------------------------------------
def _dtype_of(obj):
    return obj if isinstance(obj, _np.dtype) else getattr(obj, "dtype", obj)


def maximum_fill_value(obj):
    """Minimum value representable by the dtype of ``obj`` (the fill for a maximum)."""
    return _np.ma.maximum_fill_value(_dtype_of(obj))


def minimum_fill_value(obj):
    """Maximum value representable by the dtype of ``obj`` (the fill for a minimum)."""
    return _np.ma.minimum_fill_value(_dtype_of(obj))


def common_fill_value(a, b):
    """The fill value shared by ``a`` and ``b``, else None."""
    c = _core()

    def fv(x):
        return x.fill_value if c.isMaskedArray(x) else c.default_fill_value(x)

    t1, t2 = fv(a), fv(b)
    return t1 if t1 == t2 else None


# ---------------------------------------------------------------------------
# masking functions
# ---------------------------------------------------------------------------
def _cond(condition, xp):
    """Condition as a bool array of ``xp`` (masked entries count as True)."""
    if condition is nomask:
        return xp.zeros((), dtype=bool)
    return _backend.asarray(_core().filled(condition, True), xp, dtype=bool)


def masked_where(condition, a, copy=True):
    """Mask ``a`` where ``condition`` is True (OR-ed with the mask of ``a``).

    Unlike numpy.ma the all-False result mask is not shrunk to `nomask` (no sync).
    """
    c = _core()
    res = c.MaskedArray(_hostwrap(a), copy=copy)
    xp = _backend.get_xp(res, condition)  # a cupy operand never moves: a host ``a`` follows it
    if res._xp is not xp:
        res = _to_dev(res, xp)
    cond = _cond(condition, xp)
    if cond.shape and cond.shape != res.shape:
        raise IndexError(
            f"Inconsistent shape between the condition and the input (got {cond.shape} and {res.shape})"
        )
    if cond.shape != res.shape:
        cond = xp.broadcast_to(cond, res.shape)
    m0 = res._mask
    if m0 is nomask:
        res._mask = xp.array(cond, dtype=bool, copy=True)
        res._sharedmask = False
        if not copy and _is_xma(a) and a._mask is nomask:
            a._mask = res._mask.view()
    else:
        m0[...] = xp.logical_or(cond, m0)
    return res


def _masked_cmp(name):
    def func(x, value, copy=True):
        x = _coerce(x)
        return masked_where(getattr(_ops, name)(x, value), x, copy=copy)

    func.__name__ = func.__qualname__ = f"masked_{name}"
    func.__doc__ = f"Mask ``x`` where ``x`` is {name.replace('_', ' ')} ``value`` (see ``masked_where``)."
    return func


masked_greater = _masked_cmp("greater")
masked_greater_equal = _masked_cmp("greater_equal")
masked_less = _masked_cmp("less")
masked_less_equal = _masked_cmp("less_equal")
masked_not_equal = _masked_cmp("not_equal")


def masked_equal(x, value, copy=True):
    """Mask ``x`` where equal to ``value``; ``value`` becomes the fill value of the result."""
    x = _coerce(x)
    out = masked_where(_ops.equal(x, value), x, copy=copy)
    out.fill_value = value
    return out


def masked_inside(x, v1, v2, copy=True):
    """Mask ``x`` inside the closed interval ``[v1, v2]``."""
    if v2 < v1:
        v1, v2 = v2, v1
    x = _coerce(x)
    xf = x.filled()
    return masked_where((xf >= v1) & (xf <= v2), x, copy=copy)


def masked_outside(x, v1, v2, copy=True):
    """Mask ``x`` outside the closed interval ``[v1, v2]``."""
    if v2 < v1:
        v1, v2 = v2, v1
    x = _coerce(x)
    xf = x.filled()
    return masked_where((xf < v1) | (xf > v2), x, copy=copy)


def masked_invalid(a, copy=True):
    """Mask the invalid (NaN, inf) values of ``a``; the mask is always an array."""
    a = _coerce(a)
    return masked_where(~a._xp.isfinite(a._data), a, copy=copy)


def masked_values(x, value, rtol=1e-5, atol=1e-8, copy=True, shrink=True):
    """Mask ``x`` where close to ``value`` (floats, ``rtol``/``atol``) or equal (other dtypes).

    ``value`` becomes the fill value.  With ``shrink=True`` an all-False mask is
    replaced by `nomask` (one device sync).
    """
    c = _core()
    x = _coerce(x)
    xp = x._xp
    xnew = x.filled(value)
    if value is None:
        # as numpy: ``isclose`` fails on floats, ``== None`` is False elsewhere
        if _np.issubdtype(xnew.dtype, _np.floating):
            raise TypeError(f"unsupported operand type(s) for -: '{xnew.dtype.name}' and 'NoneType'")
        mask = xp.zeros(xnew.shape, dtype=bool)
    elif _np.issubdtype(xnew.dtype, _np.floating):
        mask = xp.isclose(xnew, value, rtol=rtol, atol=atol)
    else:
        mask = xp.equal(xnew, value)
    if copy and x._mask is nomask:      # `filled` already returned a new array otherwise
        xnew = xp.array(xnew, copy=True)
    ret = c._wrap(xnew, mask, fill_value=value)
    if shrink:
        ret.shrink_mask()
    return ret


def fix_invalid(a, mask=nomask, copy=True, fill_value=None):
    """Mask the invalid (NaN, inf) values of ``a`` and replace them by ``fill_value``.

    The mask of a floating-point result is always an array (no sync).
    """
    a = _core().MaskedArray(_hostwrap(a), copy=copy, mask=mask, subok=True)
    if a.dtype.kind not in "fc":
        return a
    xp = a._xp
    invalid = ~xp.isfinite(a._data)
    if a._mask is nomask:
        a._mask = invalid
    else:
        a._mask |= invalid
    if fill_value is None:
        fill_value = a.fill_value
    xp.copyto(a._data, xp.asarray(fill_value, dtype=a.dtype), where=invalid)
    return a


# ---------------------------------------------------------------------------
# bit shifts, products, correlation
# ---------------------------------------------------------------------------
def _shift(name):
    def func(a, n):
        c = _core()
        a = _coerce(a)
        xp = a._xp
        if _is_xma(n) or isinstance(n, _np.ndarray) or _backend.is_cupy_array(n):
            n = c._to_xp(c.getdata(n), xp)
        d = getattr(xp, name)(a.filled() if a._mask is nomask else a.filled(0), n)
        return c._wrap(xp.asarray(d), nomask if a._mask is nomask else a._mask.copy())

    func.__name__ = func.__qualname__ = name
    func.__doc__ = f"Masked ``{name}`` of ``a`` by ``n`` bits (the mask of ``a`` is kept)."
    return func


left_shift = _shift("left_shift")
right_shift = _shift("right_shift")


def inner(a, b):
    """Inner product, masked values counting as 0 (not masked; a scalar result needs one sync)."""
    c = _core()
    xp, (a, b) = _common(a, b)
    fa, fb = a.filled(0), b.filled(0)
    fa = fa.reshape((1,)) if fa.ndim == 0 else fa
    fb = fb.reshape((1,)) if fb.ndim == 0 else fb
    res = xp.asarray(xp.inner(fa, fb))
    return _backend.host_scalar(res) if res.ndim == 0 else c._wrap(res, nomask)


innerproduct = inner


def outer(a, b):
    """Outer product of the flattened inputs, masked values counting as 0.

    The result is masked where either factor is; an all-False mask is not shrunk (no sync).
    """
    c = _core()
    xp, (a, b) = _common(a, b)
    d = xp.outer(a.filled(0).ravel(), b.filled(0).ravel())
    if a._mask is nomask and b._mask is nomask:
        return c._wrap(d, nomask)
    return c._wrap(d, xp.logical_or.outer(_full_mask(a, xp).ravel(), _full_mask(b, xp).ravel()))


outerproduct = outer


def _convolve_or_correlate(name, a, v, mode, propagate_mask):
    c = _core()
    if mode not in ("valid", "same", "full"):
        raise ValueError(f"mode must be one of 'valid', 'same', or 'full' (got {mode!r})")
    xp, (a, v) = _common(a, v)
    if a.ndim > 1 or v.ndim > 1:
        raise ValueError("object too deep for desired array")
    f = getattr(xp, name)
    ma, mv = _full_mask(a, xp), _full_mask(v, xp)
    if propagate_mask:
        mask = f(ma, xp.ones(v.shape, dtype=bool), mode=mode) | f(xp.ones(a.shape, dtype=bool), mv, mode=mode)
        data = f(a._data, v._data, mode=mode)
    else:
        mask = ~f(~ma, ~mv, mode=mode)
        data = f(a.filled(0), v.filled(0), mode=mode)
    return c._wrap(data, mask)


def correlate(a, v, mode="valid", propagate_mask=True):
    """Cross-correlation of two 1-D masked sequences (see ``numpy.ma.correlate``)."""
    return _convolve_or_correlate("correlate", a, v, mode, propagate_mask)


def convolve(a, v, mode="full", propagate_mask=True):
    """Discrete linear convolution of two 1-D masked sequences (see ``numpy.ma.convolve``)."""
    return _convolve_or_correlate("convolve", a, v, mode, propagate_mask)


def append(a, b, axis=None):
    """Append ``b`` to a copy of ``a`` along ``axis`` (flattened if None), keeping masks."""
    return _ops.concatenate([_coerce(a), _coerce(b)], axis)


def _edge(extra, a, axis):
    """``prepend``/``append`` operand of ``diff`` (a 0-d one is broadcast, keeping the data only as numpy.ma)."""
    extra = _coerce(extra)
    if extra.ndim == 0:
        shp = list(a.shape)
        shp[axis] = 1
        return _backend.get_xp(extra).broadcast_to(extra._data, tuple(shp))
    return extra


def diff(a, /, n=1, axis=-1, prepend=_NV, append=_NV):
    """Calculate the n-th discrete difference along ``axis`` (masks OR-ed)."""
    if n == 0:
        return a
    if n < 0:
        raise ValueError("order must be non-negative but got " + repr(n))
    a = _coerce(a)
    if a.ndim == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    combined = []
    if prepend is not _NV:
        combined.append(_edge(prepend, a, axis))
    combined.append(a)
    if append is not _NV:
        combined.append(_edge(append, a, axis))
    if len(combined) > 1:
        a = _ops.concatenate(combined, axis)
    nd = a.ndim
    axis = _np.lib.array_utils.normalize_axis_index(axis, nd)
    s1 = [slice(None)] * nd
    s2 = [slice(None)] * nd
    s1[axis] = slice(1, None)
    s2[axis] = slice(None, -1)
    s1, s2 = tuple(s1), tuple(s2)
    op = _np.not_equal if a.dtype == _np.bool_ else _np.subtract
    for _ in range(n):
        a = op(a[s1], a[s2])
    return a


# ---------------------------------------------------------------------------
# comparison to a scalar
# ---------------------------------------------------------------------------
def allequal(a, b, fill_value=True):
    """True if all elements are equal, masked ones counting as equal if ``fill_value`` (one sync)."""
    c = _core()
    xp, (a, b) = _common(a, b)
    m = _mask_or(a, b, xp)
    if m is nomask:
        return _backend.host_scalar(xp.equal(a._data, b._data).all())
    if fill_value:
        d = xp.where(m, True, xp.equal(a._data, b._data))
        return _backend.host_scalar(d.all())
    return False


def allclose(a, b, masked_equal=True, rtol=1e-5, atol=1e-8):
    """True if two masked arrays are element-wise equal within tolerance (one sync).

    With infinities present the result needs boolean indexing (data-dependent size).
    """
    c = _core()
    xp, (x, y) = _common(a, b)
    if y.dtype.kind != "m":
        dtype = _np.result_type(y.dtype, 1.0)
        if y.dtype != dtype:
            y = c.MaskedArray(y, dtype=dtype, copy=False)
    m = _mask_or(x, y, xp)
    xinf = _np.isinf(c.MaskedArray(x, copy=False, mask=m)).filled(False)
    if not bool(xp.all(xinf == c.filled(_np.isinf(y), False))):
        return False

    def close(x, y):
        d = c.filled(_ops.less_equal(_ops.absolute(x - y), atol + rtol * _ops.absolute(y)), masked_equal)
        return _backend.host_scalar(xp.all(xp.asarray(d)))

    if not bool(xp.any(xinf)):
        return close(x, y)
    if not bool(xp.all(xp.asarray(c.filled(x[xinf] == y[xinf], masked_equal)))):
        return False
    return close(x[~xinf], y[~xinf])


# ---------------------------------------------------------------------------
# reductions of masked logical operations
# ---------------------------------------------------------------------------
def _logical_reduce(name, filly):
    def func(target, axis=0, dtype=None):
        c = _core()
        t = _coerce(target)
        xp = t._xp
        m = t._mask
        d = t.filled(filly)
        if dtype is not None and m is not nomask:  # as numpy.ma, ``dtype`` only counts with a mask
            getattr(_np, name).reduce(_np.zeros(1, d.dtype), dtype=dtype)
        if d.ndim == 0:
            d = d.reshape(1)
            if m is not nomask:
                m = m.reshape(1)
        f = getattr(xp, "all" if name == "logical_and" else "any")
        tr = xp.asarray(f(d, axis=axis))
        if m is nomask:
            mr = nomask
        else:
            mr = xp.asarray(m.all(axis=axis))
        if tr.ndim == 0:
            if mr is not nomask and bool(mr):
                return _masked
            return _backend.host_scalar(tr)
        return c._wrap(tr, mr)

    return func


alltrue = _logical_reduce("logical_and", 1)
alltrue.__name__ = alltrue.__qualname__ = "alltrue"
alltrue.__doc__ = "Masked logical and reduction along ``axis`` (masked values count as True)."
sometrue = _logical_reduce("logical_or", 0)
sometrue.__name__ = sometrue.__qualname__ = "sometrue"
sometrue.__doc__ = "Masked logical or reduction along ``axis`` (masked values count as False)."


# ---------------------------------------------------------------------------
# putmask
# ---------------------------------------------------------------------------
def putmask(a, mask, values):
    """Set ``a`` to ``values`` where ``mask`` is True, in place (data and mask)."""
    c = _core()
    if not _is_xma(a):
        if not (isinstance(a, _np.ndarray) or _backend.is_cupy_array(a)):
            raise AttributeError(f"'{type(a).__name__}' object has no attribute 'view'")
        a = c._wrap(a, nomask)
    xp = a._xp
    valdata, valmask = c._unpack(values)
    valdata = c._to_xp(valdata, xp)
    valmask = valmask if valmask is nomask else c._to_xp(valmask, xp)
    where = _backend.asarray(c._unpack(mask)[0], xp, dtype=bool)
    if a._mask is nomask:
        if valmask is not nomask:
            a._sharedmask = True
            a._mask = xp.zeros(a.shape, dtype=bool)
            xp.copyto(a._mask, valmask, where=where)
    elif a._hardmask:
        if valmask is not nomask:
            m = a._mask.copy()
            xp.copyto(m, valmask, where=where)
            a._mask |= m
    else:
        if valmask is nomask:
            valmask = xp.zeros(_np.shape(valdata), dtype=bool)
        xp.copyto(a._mask, valmask, where=where)
    xp.copyto(a._data, valdata, where=where)


# ---------------------------------------------------------------------------
# maximum / minimum
# ---------------------------------------------------------------------------
class _extrema_operation:
    """``numpy.ma.maximum``/``minimum``: callable with ``reduce`` and ``outer``."""

    def __init__(self, name, compare):
        self.f = getattr(_np, name)
        self.__name__ = self.__qualname__ = name
        self.__doc__ = self.f.__doc__
        self._compare = compare

    def __str__(self):
        return f"Masked version of {self.f}"

    def __call__(self, a, b):
        return _ops.where(getattr(_ops, self._compare)(a, b), a, b)

    def _fill(self, t):
        fn = maximum_fill_value if self.__name__ == "maximum" else minimum_fill_value
        return fn(t)

    def reduce(self, target, axis=_NV):
        """Reduce ``target`` along ``axis`` (masked values ignored; plain input gives a plain result)."""
        c = _core()
        is_ma = c.isMaskedArray(target)
        if is_ma:
            t = _coerce(target)
            m = t._mask
            d, xp = t._data, t._xp
        else:
            xp = _backend.get_xp(target)
            d = target if isinstance(target, _np.ndarray) or xp is not _np else _np.array(target)
            m = nomask
        if axis is _NV and d.ndim > 1:
            _warnings.warn(
                f"In the future the default for ma.{self.__name__}.reduce will be axis=0, "
                "not the current None, to match np.%s.reduce. "
                "Explicitly pass 0 or None to silence this warning." % self.__name__,
                _MAFutureWarning,
                stacklevel=2,
            )
            axis = None
        ax = 0 if axis is _NV else axis
        red = "max" if self.__name__ == "maximum" else "min"
        if m is not nomask:
            d = xp.where(m, xp.asarray(self._fill(d.dtype), dtype=d.dtype), d)
        res = xp.asarray(self.f.reduce(d, axis=ax) if xp is _np else getattr(xp, red)(d, axis=ax))
        mr = nomask if m is nomask else xp.asarray(m.all(axis=ax))
        if not is_ma:  # plain input gives a plain result
            return _backend.host_scalar(res) if res.ndim == 0 else res
        return c._wrap(res, mr)

    def outer(self, a, b):
        """Outer application of the operation (masks OR-ed, masked values filled by default)."""
        c = _core()
        xp, (a, b) = _common(a, b)
        d = xp.asarray(getattr(xp, self.__name__).outer(a.filled(), b.filled()))
        if a._mask is nomask and b._mask is nomask:
            return c._wrap(d, nomask)
        return c._wrap(d, xp.logical_or.outer(_full_mask(a, xp), _full_mask(b, xp)))


maximum = _extrema_operation("maximum", "greater")
minimum = _extrema_operation("minimum", "less")


# ---------------------------------------------------------------------------
# structured / record / object features: not supported
# ---------------------------------------------------------------------------
def flatten_structured_array(a):
    """Not supported (structured arrays)."""
    _unsupported("flatten_structured_array")


def fromflex(fxarray):
    """Not supported (flexible records)."""
    _unsupported("fromflex")


def make_mask_descr(ndtype):
    """Not supported (structured mask dtypes)."""
    _unsupported("make_mask_descr")


def masked_object(x, value, copy=True, shrink=True):
    """Not supported (object arrays)."""
    _unsupported("masked_object")


class mvoid:
    """Not supported (structured scalars): constructing one raises ``NotImplementedError``."""

    def __init__(self, *args, **kwargs):
        _unsupported("mvoid")


__all__ = [
    "all", "allclose", "allequal", "alltrue", "amax", "amin", "angle", "any", "append", "arange",
    "argmax", "argmin", "asanyarray", "asarray", "bool_", "common_fill_value", "compress", "compressed",
    "convolve", "copy", "correlate", "count", "cumprod", "cumsum", "diag", "diagonal", "diff", "empty",
    "fix_invalid", "flatten_mask", "flatten_structured_array", "fromflex", "frombuffer", "fromfunction",
    "harden_mask", "identity", "indices", "inner", "innerproduct", "isarray", "left_shift",
    "make_mask_descr", "masked_equal", "masked_greater", "masked_greater_equal", "masked_inside",
    "masked_invalid", "masked_less", "masked_less_equal", "masked_not_equal", "masked_object",
    "masked_outside", "masked_singleton", "masked_values", "masked_where", "maximum", "maximum_fill_value",
    "minimum", "minimum_fill_value", "mvoid", "ndim", "nonzero", "ones", "outer", "outerproduct", "ptp",
    "put", "putmask", "ravel", "repeat", "reshape", "resize", "right_shift", "shape", "size",
    "soften_mask", "sometrue", "squeeze", "swapaxes", "take", "transpose", "zeros",
]
