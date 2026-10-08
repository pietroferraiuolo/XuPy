"""
Operators, masked ufuncs and numpy protocols of ``xupy.ma`` (numpy.ma semantics).

``_OpsMixin`` provides the operator dunders, the unary ufunc methods and the
``__array_ufunc__``/``__array_function__`` protocols.  Three code paths mirror
numpy.ma:

* *table path* (``_mbin``, ``_dbin``, ``_munary``): ``numpy.ma``'s
  ``_MaskedBinaryOperation``/``_DomainedBinaryOperation``/
  ``_MaskedUnaryOperation``; used by ``+ - * / // **``, ``==`` ... and by the
  module-level functions (``add``, ``sqrt``, ...);
* *wrap path* (``_ufunc_call``): what ``ndarray`` ufuncs do through
  ``MaskedArray.__array_wrap__`` (masks OR-ed, domain from ``ufunc_domain``);
  used by ``np.<ufunc>(...)``, ``%``, ``divmod``, bitwise operators, unary
  ``- + abs ~`` and in-place ``%= &= |= ^= <<= >>=``;
* in-place table operators (``+= -= *= /= //= **=``).

All paths resolve the array module from the operands (cupy wins) and avoid
data-dependent host syncs: masks are computed unconditionally.  Consequently the
mask of an array result may be an all-False array where numpy.ma shrinks it to
``nomask`` (only empty and 0-d masks are shrunk).  Mixins must not import ``core``
at module level, hence the lazy ``_core()``.

Documented divergence from numpy.ma: ``np.power(ndarray, xupy_masked)`` (reflected ufunc call) behaves
like the operator ``ndarray ** xupy_masked`` (non-finite results are masked), whereas numpy.ma goes
through its wrap path and does not mask them.

Known cupy limitations (the right answer would need a host sync or is not
representable on the device):

* integer ``**`` negative integer raises ``ValueError`` only for host exponents
  (scalars, lists, numpy arrays), not for cupy exponents or ``np.power(...)``;
* float ``%``/``//``/``divmod`` follow numpy (`_cp_divmod`); other float ufuncs and
  complex transcendental functions can differ from numpy by an ulp;
* ``ufunc.reduce`` ignores masked slots (numpy.ma reduces the raw data);
  ``bitwise_*``/``logical_xor`` reductions and ``accumulate`` other than
  ``add``/``multiply`` are not implemented on cupy;
* ``==``/``!=`` against text is answered (all False/True), ordering comparisons
  and string/object *arrays* raise ``TypeError`` on the GPU.
"""
from __future__ import annotations

import builtins as _builtins
import functools as _functools
import operator as _operator

import inspect as _inspect

import numpy as _np

from . import _backend
from ._backend import get_xp as _get_xp
from ._backend import host_scalar as _host_scalar
from ._backend import is_cupy_array as _is_cupy_array
from ._domains import (
    BINARY_OPS,
    DOMAINED_BINARY_OPS,
    UNARY_OPS,
    _DomainSafeDivide,
    ufunc_domain,
    ufunc_fills,
)
from ._singletons import MaskError, masked, nomask

__all__ = ["concatenate", "where", "dot", "power", "round", "round_", "clip", "choose", "ids", "trace"]

_NV = _np._NoValue
_NP_MASKED = _np.ma.MaskedArray


# ---------------------------------------------------------------------------
# domain / fill tables (as numpy.ma fills them when building its masked ops)
# ---------------------------------------------------------------------------
for _n, _f, _d in UNARY_OPS.values():
    ufunc_domain[_n], ufunc_fills[_n] = _d, _f
for _n, _fx, _fy in BINARY_OPS.values():
    ufunc_domain[_n], ufunc_fills[_n] = None, (_fx, _fy)
for _n, _d, _fx, _fy in DOMAINED_BINARY_OPS.values():
    ufunc_domain[_n], ufunc_fills[_n] = _d, (_fx, _fy)
del _n, _f, _d, _fx, _fy


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _core():
    from . import core

    return core


def _is_xma(x):
    return getattr(x, "_is_xupy_masked", False)


def _is_mconst(x):
    return getattr(x, "_is_xupy_masked_constant", False)


def _like(*ops):
    """First operand carrying fill_value/hardmask (numpy's ``_update_from`` source)."""
    for o in ops:
        if _is_xma(o) or _is_mconst(o) or isinstance(o, _NP_MASKED):
            return o
    return None


def _uname(ufunc):
    n = getattr(ufunc, "__name__", None) or ufunc.name
    return n[len("cupy_"):] if n.startswith("cupy_") else n


def _es():
    return _np.errstate(divide="ignore", invalid="ignore")


def _to_dev(x, xp):
    """Move a plain array to the device of ``xp`` (numpy -> cupy only)."""
    if xp is _np:
        return x.get() if _is_cupy_array(x) else x
    if isinstance(x, _np.ndarray):
        if x.dtype.kind not in "biufc":
            raise TypeError(f"unsupported operand dtype {x.dtype} on the GPU")
        return xp.asarray(x)
    return x


def _operands(*ops, keep_py=False):
    """``(xp, [(data, mask), ...])`` with data and masks on the common device.

    ``keep_py`` leaves Python scalars as they are (weak scalars, as ndarray
    ufuncs see them); otherwise they become strong 0-d arrays like
    ``numpy.ma.getdata`` does.
    """
    c = _core()
    xp = _get_xp(*ops)
    out = []
    for o in ops:
        if keep_py and type(o) in (bool, int, float, complex):
            out.append((o, nomask))
            continue
        d, m = c._unpack(o)
        d = _to_dev(d, xp)
        out.append((d, m if m is nomask else _to_dev(m, xp)))
    return xp, out


def _fit(m, shape, xp):
    """Mask broadcast to ``shape`` (owning its memory); ``nomask`` passes."""
    if m is nomask:
        return m
    m = xp.asarray(m)  # 0-d numpy ops return scalars
    return m if m.shape == shape else xp.broadcast_to(m, shape).copy()


def _shrink_cheap(m):
    """numpy's ``shrink=True`` where it is free: empty and 0-d masks (no sync for arrays)."""
    if m is nomask or m.size == 0 or (m.ndim == 0 and not bool(m)):
        return nomask
    return m


def _or(ma, mb, shape, xp):
    return _fit(_core()._mask_or(ma, mb, xp), shape, xp)


def _full(m, shape, xp):
    """Full boolean mask (zeros for ``nomask``)."""
    return xp.zeros(shape, dtype=bool) if m is nomask else m


def _shape(d):
    return _np.shape(d) if not hasattr(d, "shape") else d.shape


def _put(xp, dst, src, where, casting="unsafe"):
    xp.copyto(dst, src, casting=casting, where=where)


def _fillarr(xp, fv, dtype):
    return xp.asarray(_np.asarray(fv).astype(dtype))


def _scalar_or(c, res, m, xp):
    """``masked`` / numpy scalar for a 0-d result (decision 1)."""
    return c._maybe_scalar(xp.asarray(res), m)


def _check_cast(ufunc, arrs, out_dtype, **kw):
    """Raise numpy's own casting error for an in-place/``out=`` ufunc (cupy path)."""
    dummies = [_np.zeros((), a.dtype) if hasattr(a, "dtype") else a for a in arrs]
    with _np.errstate(all="ignore"):
        ufunc(*dummies, out=_np.zeros((), out_dtype), **kw)


# ---------------------------------------------------------------------------
# table path: numpy.ma _MaskedUnary/_MaskedBinary/_DomainedBinary operations
# ---------------------------------------------------------------------------
def _munary(name, a, *args, **kwargs):
    """``numpy.ma`` masked unary operation (domain -> new masked values)."""
    c = _core()
    xp, ((d, m),) = _operands(a)
    f = getattr(xp, name)
    dom = ufunc_domain.get(name)
    with _es():
        res = xp.asarray(f(d, *args, **kwargs))
    if dom is not None:
        nm = ~xp.isfinite(res)
        nm |= dom(d)
        if m is not nomask:
            nm |= m
        m = _fit(nm, res.shape, xp)
    elif m is not nomask:
        m = m.copy()
    if res.ndim == 0:
        return _scalar_or(c, res, m, xp)
    if m is not nomask:
        try:
            _put(xp, res, d, m, casting="same_kind")
        except (TypeError, ValueError):
            pass
    return c._wrap(res, m, like=_like(a))


def _raise_numpy_error(name, a, b):
    """Re-raise numpy's own TypeError for ufunc ``name`` on operands of unsupported dtype.

    Evaluated on empty host arrays of the operand dtypes (no device access); returns if numpy accepts them.
    """
    c = _core()
    e = [_np.empty(0, dtype=c._unpack(o)[0].dtype) for o in (a, b)]
    try:
        getattr(_np, name)(*e)
    except TypeError:
        raise
    except Exception:  # noqa: BLE001
        pass


def _mbin(name, a, b, *args, **kwargs):
    """``numpy.ma`` masked binary operation (masks OR-ed, data kept under masks)."""
    c = _core()
    try:
        xp, ((da, ma), (db, mb)) = _operands(a, b)
    except TypeError:
        _raise_numpy_error(name, a, b)
        raise
    with _es():
        res = xp.asarray(getattr(xp, name)(da, db, *args, **kwargs))
    m = _or(ma, mb, res.shape, xp)
    if res.ndim == 0:
        return _scalar_or(c, res, m, xp)
    if m is not nomask:
        try:
            _put(xp, res, da, m)
        except Exception:  # noqa: BLE001  (impossible to guarantee masked values)
            pass
    return c._wrap(res, m, like=_like(a, b), sharedmask=True)


def _dbin(name, a, b):
    """``numpy.ma`` domained binary operation (division-like)."""
    c = _core()
    xp, ((da, ma), (db, mb)) = _operands(a, b)
    with _es():
        res = xp.asarray(_xp_fn(xp, name)(da, db))
    m = xp.asarray(~xp.isfinite(res))
    for mk in (ma, mb):
        if mk is not nomask:
            m |= mk
    dom = ufunc_domain.get(name)
    if dom is not None:
        m |= dom(da, db)
    if res.ndim == 0:
        return _scalar_or(c, res, m, xp)
    res = xp.where(m, xp.zeros((), res.dtype), res)
    masked_da = xp.multiply(m, da)
    if _np.can_cast(masked_da.dtype, res.dtype, casting="safe"):
        res += masked_da
    return c._wrap(res, m, like=_like(a, b))


def _check_int_pow(xp, a, b):
    """numpy raises for integer ``**`` negative integer; on cupy only when it is free to know.

    Checked (no device sync) when the exponent is a host scalar/array; a cupy
    exponent would need a sync and silently gives cupy's result.
    """
    if xp is _np or _is_cupy_array(b) or getattr(b, "_is_xupy_masked", False) \
            or isinstance(b, _NP_MASKED):
        return
    try:
        eb = _np.asarray(b)
        ea = a.dtype if hasattr(a, "dtype") else _np.asarray(a).dtype
    except Exception:  # noqa: BLE001
        return
    if ea.kind in "biu" and eb.dtype.kind in "iu" and eb.size and (eb < 0).any():
        raise ValueError("Integers to negative integer powers are not allowed.")


def power(a, b, third=None):
    """Element-wise ``a ** b`` with masks OR-ed and non-finite results masked."""
    if third is not None:
        raise MaskError("3-argument power not supported.")
    c = _core()
    xp, ((fa, ma), (fb, mb)) = _operands(a, b)
    _check_int_pow(xp, fa, b)
    with _es():
        p = xp.asarray(xp.power(fa, fb))
    rt = _np.result_type(fa.dtype, p.dtype)
    m = _or(ma, mb, p.shape, xp)
    res = p.astype(rt, copy=False) if m is nomask else xp.where(m, fa, p)
    like = a if _like(a) is a else None
    if res.dtype.kind in "biu":  # integer results are always finite
        if res.ndim == 0:  # numpy.ma: `masked` or a 0-d masked array (not a scalar)
            if m is not nomask and bool(m):
                return masked
            return c._wrap(res, nomask, like=like)
        return c._wrap(res, m, like=like)
    with _es():
        invalid = ~xp.isfinite(res)
    mm = xp.asarray(invalid if m is nomask else (m | invalid))
    if res.ndim == 0:
        if bool(mm):
            return masked
        return c._wrap(res, nomask if m is nomask else mm, like=like)
    out = c._wrap(res, mm, like=like)
    out._data = xp.where(invalid, _fillarr(xp, out.fill_value, res.dtype), res)
    return out


def _is_text_operand(x):
    """``None``, text scalars and host string arrays (not representable on the GPU)."""
    return x is None or isinstance(x, (str, bytes)) or (
        isinstance(x, _np.ndarray) and x.dtype.kind in "US")


def _compare(self, other, op):
    """``numpy.ma`` ``MaskedArray._comparison`` (masked==masked is equal, else unequal)."""
    c = _core()
    if op in (_operator.eq, _operator.ne) and _is_xma(self) and self._xp is not _np \
            and _is_text_operand(other):
        # numeric device data never equals text (numpy: all False / all True)
        shape = _np.broadcast_shapes(self.shape, getattr(other, "shape", ()))
        check = self._xp.full(shape, op is _operator.ne, dtype=bool)
        m = self._mask
        return c._wrap(check, m if m is nomask else _fit(m, shape, self._xp), like=self)
    xp, ((ds, ms), (do, mo)) = _operands(self, other)
    mask = c._mask_or(ms, mo, xp)
    check = op(ds, do)
    if check is NotImplemented:
        return NotImplemented
    if isinstance(check, (bool, _np.bool_)):  # incomparable objects: python semantics
        return masked if (mask is not nomask and bool(mask.any())) else check
    check = xp.asarray(check)
    if check.ndim == 0:
        return _scalar_or(c, check, mask, xp)
    if mask is not nomask:
        if op in (_operator.eq, _operator.ne):
            zero = xp.zeros((), dtype=bool)
            check = xp.where(mask, op(zero if ms is nomask else ms, zero if mo is nomask else mo), check)
        mask = _shrink_cheap(_fit(mask, check.shape, xp))
    return c._wrap(check, mask, like=self)


# ---------------------------------------------------------------------------
# wrap path: ndarray ufuncs through MaskedArray.__array_wrap__
# ---------------------------------------------------------------------------
def _fill_value_of(name):
    fv = ufunc_fills[name]
    return fv[-1] if isinstance(fv, tuple) else fv


def _cp_divmod(xp, a, b, which="both", **kw):
    """Floating ``divmod`` with numpy's ``npy_divmod`` semantics, on device.

    cupy's own ``remainder``/``divmod`` return nan for a finite dividend and an
    infinite divisor (numpy: ``5.5 % inf == 5.5``) and inf for ``inf // 2``
    (numpy: nan).  Integer and other dtypes use cupy's functions unchanged.
    """
    dt = xp.result_type(a, b)  # before asarray: Python scalars stay weak
    if kw or dt.kind != "f":
        fn = {"both": xp.divmod, "mod": xp.remainder, "floor": xp.floor_divide}[which]
        return fn(a, b, **kw)
    a, b = xp.asarray(a), xp.asarray(b)
    a, b = a.astype(dt, copy=False), b.astype(dt, copy=False)
    fm = xp.fmod(a, b)
    div = (a - fm) / b
    nonzero = fm != 0  # True for nan
    adj = nonzero & ((b < 0) != (fm < 0))
    zero = xp.zeros((), dt)
    mod = xp.where(adj, fm + b, xp.where(nonzero, fm, xp.copysign(zero, b)))
    div = xp.where(adj, div - 1, div)
    floor = xp.floor(div)
    floor = xp.where(div - floor > 0.5, floor + 1, floor)
    quo = xp.where(div != 0, floor, xp.copysign(zero, a / b))
    bz = b == 0  # numpy: modulus nan (fmod), quotient a / b
    mod = xp.where(bz, fm, mod)
    quo = xp.where(bz, a / b, quo)
    if which == "mod":
        return mod
    return quo if which == "floor" else (quo, mod)


def _xp_fn(xp, name):
    """``xp.<name>``; cupy's ``remainder``/``mod`` get numpy's float semantics."""
    if xp is not _np and name in ("remainder", "mod"):
        return _functools.partial(_cp_divmod, xp, which="mod")
    if xp is not _np and name == "floor_divide":
        return _functools.partial(_cp_divmod, xp, which="floor")
    return getattr(xp, name)


def _xp_ufunc(xp, ufunc):
    if xp is _np:
        return ufunc
    if ufunc is _np.positive:  # cupy rejects booleans (numpy raises UFuncTypeError)
        def _positive(x, **kw):
            if getattr(x, "dtype", None) is not None and x.dtype.kind == "b":
                _np.positive(_np.zeros((), bool))  # raises numpy's own error
            return xp.array(x)

        return _positive
    if ufunc is _np.divmod:
        return _functools.partial(_cp_divmod, xp)
    if ufunc is _np.remainder:
        return _functools.partial(_cp_divmod, xp, which="mod")
    if ufunc is _np.floor_divide:
        return _functools.partial(_cp_divmod, xp, which="floor")
    return getattr(xp, _uname(ufunc), None)


def _ufunc_kwargs(xp, kw):
    if xp is _np:
        return kw
    if _builtins.set(kw) - {"dtype", "casting", "subok", "order"}:
        return None
    return {k: v for k, v in kw.items() if k in ("dtype", "casting")}


def _ufunc_call(ufunc, inputs, out=None, where=True, kw=None):
    """Apply ``ufunc`` with ``__array_wrap__`` mask semantics (see module doc)."""
    c = _core()
    kw = dict(kw or {})
    name = _uname(ufunc)
    xp, ops = _operands(*inputs, keep_py=True)
    f = _xp_ufunc(xp, ufunc)
    xkw = _ufunc_kwargs(xp, kw)
    if f is None or xkw is None:
        return NotImplemented
    datas = [d for d, _ in ops]
    masks = [m for _, m in ops]
    nout = ufunc.nout
    outs = [None] * nout if out is None else list(out)
    if len(outs) != nout:
        return NotImplemented
    odata = []
    for o in outs:
        if o is None:
            odata.append(None)
        elif _is_xma(o) and _get_xp(o) is xp:
            odata.append(o._data)
        else:  # plain arrays would silently drop the mask
            return NotImplemented
    if where is not True:
        where = _to_dev(_arr(c.filled(where, False)), xp)
    given = [o for o in odata if o is not None]
    with _es():
        if given and xp is _np:
            res = f(*datas, out=tuple(odata), **({} if where is True else {"where": where}), **xkw)
        else:
            if given:
                _check_cast(ufunc, datas, [o.dtype for o in given][0], **xkw)
            res = f(*datas, **xkw)
            if nout == 1:
                res = (res,)
            res = tuple(xp.asarray(r) for r in res)
            if given:
                cast = xkw.get("casting", "same_kind")
                for o, r in _builtins.zip(odata, res):
                    if o is not None:
                        xp.copyto(o, r, casting=cast, where=True if where is True else where)
                res = tuple(r if o is None else o for o, r in _builtins.zip(odata, res))
            elif where is not True:
                res = tuple(xp.where(where, r, xp.zeros((), r.dtype)) for r in res)
    if not isinstance(res, tuple):
        res = (res,)
    res = tuple(xp.asarray(r) for r in res)
    shape = res[0].shape
    # mask: OR of the input masks; unary ufuncs always get a full mask
    if len(datas) == 1:
        mk = masks[0]
        m = xp.zeros(_shape(datas[0]), bool) if mk is nomask else mk.copy()
    else:
        m = _functools.reduce(lambda x, y: c._mask_or(x, y, xp), masks)
    # domain: invalid (or already masked) positions take the fill value
    dom = ufunc_domain.get(name)
    if dom is not None:
        with _es():
            d = xp.asarray(dom(*datas)).astype(bool, copy=False)
        if len(datas) == 1:  # ma-aware unary domains count masked values as invalid
            d = d | masks[0] if masks[0] is not nomask else d
        for r in res:
            _put(xp, r, _fillarr(xp, _fill_value_of(name), r.dtype), d)
        m = d if m is nomask else (m | d)
    m = _fit(m, shape, xp)
    if len(datas) > 1:
        m = _shrink_cheap(m)
    like = _like(*inputs)
    results = []
    for i, (r, o) in enumerate(_builtins.zip(res, outs)):
        mi = m if (i == 0 or m is nomask) else m.copy()
        if o is not None:  # numpy sets the mask of `out` (in-place semantics)
            o._mask, o._sharedmask = mi, False
            results.append(o)
        elif r.ndim == 0 and mi is not nomask and bool(mi):
            results.append(masked)
        else:
            results.append(c._wrap(r, mi, like=like))
    return results[0] if nout == 1 else tuple(results)


# --- reductions of ufuncs (masked values replaced by the neutral element) ---
_REDUCE_FN = {  # numpy ufunc name -> array-module function (cupy has no ufunc.reduce)
    "add": "sum", "multiply": "prod", "maximum": "max", "minimum": "min",
    "logical_and": "all", "logical_or": "any", "fmax": "nanmax", "fmin": "nanmin",
}
_REDUCE_UFUNCS = {
    "add", "multiply", "maximum", "minimum", "fmax", "fmin", "logical_and", "logical_or",
    "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor",
}


def _neutral(name, dtype):
    if name in ("add", "logical_or", "logical_xor", "bitwise_or", "bitwise_xor"):
        return 0
    if name in ("multiply", "logical_and"):
        return 1
    if name == "bitwise_and":
        return True if dtype.kind == "b" else -1
    if name in ("maximum", "fmax"):
        return _np.ma.maximum_fill_value(dtype)
    return _np.ma.minimum_fill_value(dtype)


def _xreduce(xp, name, t, axis, dtype, keepdims, method="reduce"):
    kw = {"axis": axis, "keepdims": keepdims} if method == "reduce" else {"axis": axis}
    if dtype is not None:
        kw["dtype"] = dtype
    if xp is _np:
        return getattr(getattr(_np, name), method)(t, **kw)
    if method == "accumulate":
        fn = {"add": "cumsum", "multiply": "cumprod"}.get(name)
    else:
        fn = _REDUCE_FN.get(name)
    if fn is None:
        raise NotImplementedError
    if fn in ("max", "min", "all", "any", "nanmax", "nanmin"):
        kw.pop("dtype", None)
    return getattr(xp, fn)(t, **kw)


def _ufunc_scan(ufunc, method, inputs, out, where, kw):
    """``reduce`` / ``accumulate`` of a ufunc over the unmasked values."""
    name = _uname(ufunc)
    if (out is not None and any(o is not None for o in out)) or where is not True or ufunc.nin != 2 \
            or name not in _REDUCE_UFUNCS or len(inputs) != 1 or _builtins.set(kw) - {"axis", "dtype", "keepdims"}:
        return NotImplemented
    c = _core()
    axis, dtype, keepdims = kw.get("axis", 0), kw.get("dtype"), kw.get("keepdims", False)
    xp, ((d, m),) = _operands(inputs[0])
    t = d if m is nomask else xp.where(m, xp.asarray(_neutral(name, d.dtype), dtype=d.dtype), d)
    try:
        if method == "reduce":
            res = xp.asarray(_xreduce(xp, name, t, axis, dtype, keepdims))
            mr = nomask if m is nomask else xp.asarray(_xreduce(xp, "logical_and", m, axis, None, keepdims))
            if res.ndim == 0:
                return c._maybe_scalar(res, mr)
            return c._wrap(res, mr)
        res = xp.asarray(_xreduce(xp, name, t, axis, dtype, False, "accumulate"))
    except NotImplementedError:
        return NotImplemented
    return c._wrap(res, nomask if m is nomask else m.copy())


def _ufunc_outer(ufunc, inputs, kw):
    if len(inputs) != 2 or kw or ufunc.nin != 2:
        return NotImplemented
    c = _core()
    xp, ((da, ma), (db, mb)) = _operands(*inputs)
    f = _xp_ufunc(xp, ufunc)
    if f is None:
        return NotImplemented
    res = xp.asarray(f.outer(da, db))
    if ma is nomask and mb is nomask:
        return c._wrap(res, nomask)
    m = xp.logical_or.outer(_full(ma, da.shape, xp), _full(mb, db.shape, xp))
    return c._wrap(res, m)


# ---------------------------------------------------------------------------
# matmul / dot
# ---------------------------------------------------------------------------
def _filled0(xp, d, m):
    return d if m is nomask else xp.where(m, xp.zeros((), d.dtype), d)


def _prop(m, ndim, axis, xp):
    """Mask whole 1-d vectors (along ``axis``) that contain masked values."""
    if m is nomask:
        return m
    new = m.copy()
    for ax in _np.lib.array_utils.normalize_axis_tuple(axis, ndim):
        new |= m.any(axis=ax, keepdims=True)
    return new


def _product(a, b, fn, strict=False, out=None):
    c = _core()
    xp, ((da, ma), (db, mb)) = _operands(a, b)
    if _builtins.any(x.dtype.kind not in "biufc" for x in (da, db)):
        return NotImplemented
    if fn == "matmul" and (da.ndim == 0 or db.ndim == 0):
        raise ValueError(
            "matmul: Input operand %d does not have enough dimensions "
            "(has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)" % (0 if da.ndim == 0 else 1)
        )
    if strict and da.ndim and db.ndim:
        ka, kb = da.ndim - 1, db.ndim - (1 if db.ndim == 1 else 2)
        ma, mb = _prop(ma, da.ndim, ka, xp), _prop(mb, db.ndim, kb, xp)
    f = getattr(xp, fn)
    if out is not None:
        d = f(_filled0(xp, da, ma), _filled0(xp, db, mb), out=out._data)
    else:
        d = xp.asarray(f(_filled0(xp, da, ma), _filled0(xp, db, mb)))
    if ma is nomask and mb is nomask and not strict and da.shape[-1:] != (0,):
        m = nomask
    else:
        am = ~_full(ma, da.shape, xp)
        bm = ~_full(mb, db.shape, xp)
        m = ~xp.asarray(f(am, bm))
    if out is not None:
        out._mask, out._sharedmask = (m if m is nomask else _fit(m, d.shape, xp)), False
        return out
    if d.ndim == 0:
        return c._maybe_scalar(d, m)
    return c._wrap(d, m)


def dot(a, b, strict=False, out=None):
    """Dot product of two arrays, masked values counting as 0 (as ``numpy.ma.dot``).

    With ``strict=True`` a masked value masks the whole row/column it belongs to.
    """
    return _product(a, b, "dot", strict, out)


# ---------------------------------------------------------------------------
# concatenate / where
# ---------------------------------------------------------------------------
def concatenate(arrays, axis=0):
    """Concatenate arrays along ``axis`` keeping masks (cupy wins over numpy)."""
    c = _core()
    arrays = list(arrays)
    xp = _get_xp(*arrays)
    datas = [_to_dev(c.getdata(a), xp) for a in arrays]
    if xp is not _np and axis is not None and datas:
        # numpy raises ValueError for 0-d / mismatching ndim; cupy raises TypeError / another message
        if datas[0].ndim == 0:
            raise ValueError("zero-dimensional arrays cannot be concatenated")
        for i, a in enumerate(datas):
            if a.ndim != datas[0].ndim:
                raise ValueError("all the input arrays must have same number of dimensions, but the array at "
                                 f"index 0 has {datas[0].ndim} dimension(s) and the array at index {i} has "
                                 f"{a.ndim} dimension(s)")
    d = xp.concatenate(datas, axis)
    if _builtins.all(c.getmask(a) is nomask for a in arrays):
        return c._wrap(d, nomask)
    dm = xp.concatenate([_to_dev(c.getmaskarray(a), xp) for a in arrays], axis)
    return c._wrap(d, dm.reshape(d.shape))


def _arr(x):
    return x if isinstance(x, _np.ndarray) or _is_cupy_array(x) else _np.asarray(x)


def _nonzero(a):
    """Indices of the unmasked non-zero elements (masked values count as 0)."""
    c = _core()
    f = _arr(c.filled(a, 0))
    return _get_xp(f).nonzero(f)


def where(condition, x=_NV, y=_NV):
    """Masked ``numpy.where``: masked where the condition or the chosen value is masked."""
    n_missing = (x is _NV, y is _NV).count(True)
    if n_missing == 1:
        raise ValueError("Must provide both 'x' and 'y' or neither.")
    if n_missing == 2:
        return _nonzero(condition)
    c = _core()
    xp = _get_xp(condition, x, y)
    cf = _to_dev(_arr(c.filled(condition, False)), xp)
    xd, yd = (_to_dev(c.getdata(v), xp) for v in (x, y))
    cm, xm, ym = (_to_dev(c.getmaskarray(v), xp) for v in (condition, x, y))
    if x is masked and y is not masked:
        xd, xm = xp.zeros((), yd.dtype), xp.ones((), bool)
    elif y is masked and x is not masked:
        yd, ym = xp.zeros((), xd.dtype), xp.ones((), bool)
    data = xp.where(cf, xd, yd)
    mask = xp.where(cm, xp.ones((), bool), xp.where(cf, xm, ym))
    return c._wrap(data, mask)


# ---------------------------------------------------------------------------
# in-place operators
# ---------------------------------------------------------------------------
def _inplace_prepare(self, other):
    """``(xp, other_data, other_mask)`` or ``None`` (cupy operand, numpy target)."""
    xp = _get_xp(self, other)
    if xp is not _np and _get_xp(self) is _np:
        return None
    _, (_, (d, m)) = _operands(self, other)
    return xp, d, m


def _own_mask(self, xp):
    """The target's mask array, created (all False) when it is ``nomask``."""
    if self._mask is nomask:
        self._mask = xp.zeros(self.shape, dtype=bool)
    return self._mask


def _iop_data(self, op, other_data, npuf, xp):
    if xp is not _np:
        _check_cast(npuf, [self._data, other_data], self._data.dtype)
    with _es():
        op(self._data, other_data)


def _iop_basic(self, other, op, npuf, fill):
    """``+= -= *=``: data under masked positions is left unchanged."""
    prep = _inplace_prepare(self, other)
    if prep is None:
        return NotImplemented
    xp, od, om = prep
    if om is not nomask:
        _own_mask(self, xp)
        self._mask |= om
    if self._mask is not nomask:
        od = xp.where(self._mask, xp.asarray(fill, dtype=od.dtype), od)
    _iop_data(self, op, od, npuf, xp)
    return self


def _iop_div(self, other, op, npuf):
    """``/= //=``: division by (nearly) zero is masked and divides by 1."""
    prep = _inplace_prepare(self, other)
    if prep is None:
        return NotImplemented
    xp, od, om = prep
    dom = _DomainSafeDivide()(self._data, od)
    new = dom if om is nomask else (dom | om)
    od = xp.where(dom, xp.asarray(1, dtype=od.dtype), od)
    if self._mask is nomask:
        self._mask = new
    else:
        self._mask |= new
    od = xp.where(self._mask, xp.asarray(1, dtype=od.dtype), od)
    _iop_data(self, op, od, npuf, xp)
    return self


def _iop_pow(self, other):
    prep = _inplace_prepare(self, other)
    if prep is None:
        return NotImplemented
    xp, od, om = prep
    if self.dtype.kind != "b":  # bool targets fail on the output cast first (as numpy)
        _check_int_pow(xp, self._data, other)
    if self._mask is not nomask:
        od = xp.where(self._mask, xp.asarray(1, dtype=od.dtype), od)
    _iop_data(self, _operator.ipow, od, _np.power, xp)
    with _es():
        invalid = ~xp.isfinite(self._data)
    _put(xp, self._data, _fillarr(xp, self.fill_value, self.dtype), invalid, casting="same_kind")
    new = invalid if om is nomask else (invalid | om)
    self._mask = new if self._mask is nomask else (self._mask | new)
    return self


def _iop_wrap(self, other, ufunc):
    """``%= &= |= ^= <<= >>=``: ``out=self`` ufunc with wrap-path masks."""
    if _get_xp(self, other) is not _get_xp(self):
        return NotImplemented
    res = _ufunc_call(ufunc, (self, other), out=(self,))
    return res if res is NotImplemented else self


# ---------------------------------------------------------------------------
# __array_function__ table
# ---------------------------------------------------------------------------
def _meth(name, params, defaults=None):
    def impl(a, *args, **kwargs):
        if not _is_xma(a):
            return NotImplemented
        kw = dict(defaults or {})
        kw.update(_builtins.zip(params, args))
        kw.update(kwargs)
        return getattr(a, name)(**{k: v for k, v in kw.items() if v is not _NV})

    return impl


_P_SUM = ("axis", "dtype", "out", "keepdims", "initial", "where")
_P_MEAN = ("axis", "dtype", "out", "keepdims", "where")
_P_STD = ("axis", "dtype", "out", "ddof", "keepdims", "where")
_P_MAX = ("axis", "out", "keepdims", "initial", "where")
_P_ANY = ("axis", "out", "keepdims", "where")
_P_ARG = ("axis", "out", "keepdims")
_P_CUM = ("axis", "dtype", "out")

_METHOD_FUNCS = {
    "sum": ("sum", _P_SUM), "prod": ("prod", _P_SUM), "mean": ("mean", _P_MEAN),
    "std": ("std", _P_STD), "var": ("var", _P_STD),
    "max": ("max", _P_MAX), "amax": ("max", _P_MAX), "min": ("min", _P_MAX), "amin": ("min", _P_MAX),
    "any": ("any", _P_ANY), "all": ("all", _P_ANY),
    "argmax": ("argmax", _P_ARG), "argmin": ("argmin", _P_ARG), "ptp": ("ptp", _P_ARG),
    "cumsum": ("cumsum", _P_CUM), "cumprod": ("cumprod", _P_CUM),
    "argsort": ("argsort", ("axis", "kind", "order")), "ravel": ("ravel", ("order",)),
    "squeeze": ("squeeze", ("axis",)), "swapaxes": ("swapaxes", ("axis1", "axis2")),
    "repeat": ("repeat", ("repeats", "axis")), "take": ("take", ("indices", "axis", "out", "mode")),
    "round": ("round", ("decimals", "out")), "around": ("round", ("decimals", "out")),
}


_DEFAULTS = {"argsort": {"axis": -1}}


def _f_reshape(a, shape=None, order="C", *, newshape=None, copy=None):
    return a.reshape(shape if shape is not None else newshape, order=order) if _is_xma(a) else NotImplemented


def _f_transpose(a, axes=None):
    if not _is_xma(a):
        return NotImplemented
    return a.transpose() if axes is None else a.transpose(axes)


def _f_clip(a, a_min=_NV, a_max=_NV, out=None, **kw):
    if not _is_xma(a):
        return NotImplemented
    lo, hi = kw.pop("min", a_min), kw.pop("max", a_max)
    return a.clip(None if lo is _NV else lo, None if hi is _NV else hi, out=out, **kw)


def _f_sort(a, axis=-1, kind=None, order=None, **kw):
    if not _is_xma(a) or not hasattr(a, "sort"):
        return NotImplemented
    c = a.copy() if axis is not None else a.ravel().copy()
    c.sort(axis=-1 if axis is None else axis, kind=kind, order=order, **kw)
    return c


def _diag_parts(a, offset, axis1, axis2):
    xp = a._xp
    d = xp.diagonal(a._data, offset, axis1, axis2)
    m = a._mask
    return xp, d, (m if m is nomask else xp.diagonal(m, offset, axis1, axis2).copy())


def _f_diagonal(a, offset=0, axis1=0, axis2=1):
    if not _is_xma(a):
        return NotImplemented
    _, d, m = _diag_parts(a, offset, axis1, axis2)
    return _core()._wrap(d, m, like=a)


def _f_trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    return a.trace(offset, axis1, axis2, dtype, out) if _is_xma(a) else NotImplemented


def _f_expand_dims(a, axis):
    return a.expand_dims(axis) if _is_xma(a) else NotImplemented


def _f_copy(a, order="K", subok=False):
    return a.copy(order if order != "K" else "C") if _is_xma(a) else NotImplemented


def _f_count_nonzero(a, axis=None, *, keepdims=False):
    if not _is_xma(a):
        return NotImplemented
    xp = a._xp
    if xp is _np:
        r = xp.count_nonzero(a._data, axis=axis, keepdims=keepdims)
    else:  # cupy's count_nonzero has no ``keepdims``
        r = xp.asarray(xp.count_nonzero(a._data, axis=axis))
        if keepdims:
            r = r.reshape((1,) * a.ndim) if axis is None else xp.expand_dims(
                r, tuple(_np.lib.array_utils.normalize_axis_tuple(axis, a.ndim)))
    return _host_scalar(r) if r.ndim == 0 else r


def _f_nonzero(a):
    return _nonzero(a) if _is_xma(a) else NotImplemented


def _f_where(condition, x=_NV, y=_NV):
    return where(condition, x, y)


def _seq_stack(name):
    """``np.<name>`` applied to the data and to the masks (as numpy.ma's)."""

    def impl(tup, *args, **kwargs):
        c = _core()
        tup = list(tup)
        xp = _get_xp(*tup)
        fn = getattr(xp, name)
        d = fn([_to_dev(c.getdata(a), xp) for a in tup], *args, **kwargs)
        if _builtins.all(c.getmask(a) is nomask for a in tup):
            return c._wrap(d, nomask)
        return c._wrap(d, fn([_to_dev(c.getmaskarray(a), xp) for a in tup], *args, **kwargs))

    return impl


def _f_like(name):
    """``np.<name>_like(xma, ...)``: masked result, mask kept as numpy.ma does."""

    def impl(a, *args, device=None, **kwargs):
        if not _is_xma(a):
            return NotImplemented
        from . import extras

        return getattr(extras, name)(a, *args, **kwargs)

    return impl


def _f_full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None, *, device=None):
    if not _is_xma(a):
        return NotImplemented
    c = _core()
    xp, m = a._xp, a._mask
    data = xp.full_like(a._data, fill_value, dtype=dtype, order=order, shape=shape)
    if m is not nomask:  # numpy's __array_finalize__: keep the mask when sizes agree
        try:
            m = m.copy().reshape(data.shape)
        except ValueError:
            m = xp.zeros(data.shape, dtype=bool)
    return c._wrap(data, m, like=a)


_FUNC_TABLE = None


def _func_table():
    global _FUNC_TABLE
    if _FUNC_TABLE is None:
        t = {}
        for npname, (meth, params) in _METHOD_FUNCS.items():
            if hasattr(_np, npname):
                t[getattr(_np, npname)] = _meth(meth, params, _DEFAULTS.get(npname))
        custom = {
            "reshape": _f_reshape, "transpose": _f_transpose, "clip": _f_clip, "sort": _f_sort,
            "diagonal": _f_diagonal, "trace": _f_trace, "expand_dims": _f_expand_dims, "copy": _f_copy,
            "count_nonzero": _f_count_nonzero, "nonzero": _f_nonzero, "where": _f_where,
            "concatenate": lambda arrays, axis=0, out=None, **kw: concatenate(arrays, axis)
            if out is None and not kw else NotImplemented,
            "dot": lambda a, b, out=None: dot(a, b, out=out),
            "ones_like": _f_like("ones_like"), "zeros_like": _f_like("zeros_like"),
            "empty_like": _f_like("empty_like"), "full_like": _f_full_like,
            "shape": lambda a: a.shape, "ndim": lambda a: a.ndim,
            "size": lambda a, axis=None: a.size if axis is None else a.shape[axis],
        }
        for n in ("stack", "vstack", "hstack", "dstack", "column_stack"):
            custom[n] = _seq_stack(n)
        for npname, fn in custom.items():
            if hasattr(_np, npname):
                t[getattr(_np, npname)] = fn
        _FUNC_TABLE = t
    return _FUNC_TABLE


def _acceptable_type(t):
    return (
        getattr(t, "_is_xupy_masked", False)
        or issubclass(t, _np.ndarray)
        or getattr(t, "__module__", "").startswith("cupy")
    )


# ---------------------------------------------------------------------------
# dunder factories
# ---------------------------------------------------------------------------
def _defer(self, other):
    """numpy's ``_delegate_binop``: True if ``other`` wants to handle the operation."""
    if _is_xma(other) or _is_mconst(other):
        return False
    au = getattr(other, "__array_ufunc__", False)
    if au is False:
        return self.__array_priority__ < getattr(other, "__array_priority__", -1000000)
    return au is None


# python name -> (kind, implementation key)
_TABLE_OPS = {
    "add": ("m", "add"), "sub": ("m", "subtract"), "mul": ("m", "multiply"),
    "truediv": ("d", "divide"), "floordiv": ("d", "floor_divide"), "pow": ("p", None),
}
_WRAP_OPS = {
    "mod": _np.remainder, "divmod": _np.divmod, "and": _np.bitwise_and, "or": _np.bitwise_or,
    "xor": _np.bitwise_xor, "lshift": _np.left_shift, "rshift": _np.right_shift,
}
_UNARY_WRAP = {"neg": _np.negative, "pos": _np.positive, "abs": _np.absolute, "invert": _np.invert}
_CMP = {"eq": _operator.eq, "ne": _operator.ne, "lt": _operator.lt, "le": _operator.le,
        "gt": _operator.gt, "ge": _operator.ge}
_INPLACE_NP = {"add": (_operator.iadd, _np.add, 0), "sub": (_operator.isub, _np.subtract, 0),
               "mul": (_operator.imul, _np.multiply, 1)}


def _table_impl(kind, key):
    if kind == "m":
        return lambda a, b: _mbin(key, a, b)
    if kind == "d":
        return lambda a, b: _dbin(key, a, b)
    return power


_REFLECTED = {
    "add": lambda a, b: _mbin("add", a, b), "subtract": lambda a, b: _mbin("subtract", a, b),
    "multiply": lambda a, b: _mbin("multiply", a, b), "divide": lambda a, b: _dbin("divide", a, b),
    "floor_divide": lambda a, b: _dbin("floor_divide", a, b), "power": lambda a, b: power(a, b),
    "equal": lambda a, b: _compare(b, a, _operator.eq), "not_equal": lambda a, b: _compare(b, a, _operator.ne),
    "less": lambda a, b: _compare(b, a, _operator.gt), "less_equal": lambda a, b: _compare(b, a, _operator.ge),
    "greater": lambda a, b: _compare(b, a, _operator.lt), "greater_equal": lambda a, b: _compare(b, a, _operator.le),
}


def _reflected(ufunc, left, right):
    fn = _REFLECTED.get(_uname(ufunc))
    return NotImplemented if fn is None else fn(left, right)


def _make_ops(ns):
    for pyname, (kind, key) in _TABLE_OPS.items():
        impl = _table_impl(kind, key)

        def fwd(self, other, impl=impl):
            return NotImplemented if _defer(self, other) else impl(self, other)

        ns[f"__{pyname}__"] = fwd
        ns[f"__r{pyname}__"] = lambda self, other, impl=impl: impl(other, self)
    for pyname, uf in _WRAP_OPS.items():
        def fwd(self, other, uf=uf):
            return NotImplemented if _defer(self, other) else _ufunc_call(uf, (self, other))

        ns[f"__{pyname}__"] = fwd
        ns[f"__r{pyname}__"] = lambda self, other, uf=uf: _ufunc_call(uf, (other, self))
    for pyname, uf in _UNARY_WRAP.items():
        ns[f"__{pyname}__"] = lambda self, uf=uf: _ufunc_call(uf, (self,))
    for pyname, cmp_ in _CMP.items():
        ns[f"__{pyname}__"] = lambda self, other, cmp_=cmp_: _compare(self, other, cmp_)
    for pyname, (op, npuf, fill) in _INPLACE_NP.items():
        ns[f"__i{pyname}__"] = lambda self, other, op=op, npuf=npuf, fill=fill: _iop_basic(self, other, op, npuf, fill)
    ns["__itruediv__"] = lambda self, other: _iop_div(self, other, _operator.itruediv, _np.divide)
    ns["__ifloordiv__"] = lambda self, other: _iop_div(self, other, _operator.ifloordiv, _np.floor_divide)
    ns["__ipow__"] = lambda self, other: _iop_pow(self, other)
    for pyname in ("mod", "and", "or", "xor", "lshift", "rshift"):
        ns[f"__i{pyname}__"] = lambda self, other, uf=_WRAP_OPS[pyname]: _iop_wrap(self, other, uf)
    for name, fn in list(ns.items()):
        if callable(fn) and name.startswith("__") and getattr(fn, "__name__", "") == "<lambda>":
            fn.__name__ = name


def _unary_method(name):
    def method(self):
        return _munary(name, self)

    method.__name__ = name
    method.__doc__ = f"Masked ``{name}`` (domain errors are masked), as ``numpy.ma.{name}``."
    return method


class _OpsMixin:
    """Operators, unary ufunc methods and numpy protocols of the masked array."""

    __hash__ = None
    _make_ops(locals())

    def __matmul__(self, other):
        if _defer(self, other):
            return NotImplemented
        return _product(self, other, "matmul")

    def __rmatmul__(self, other):
        return _product(other, self, "matmul")

    def __imatmul__(self, other):
        res = _product(self, other, "matmul")
        if res is NotImplemented or not _is_xma(res) or res.shape != self.shape \
                or not _np.can_cast(res.dtype, self.dtype, "same_kind") or _get_xp(res) is not self._xp:
            return res
        self._data[...] = res._data
        m = res._mask
        self._mask, self._sharedmask = m, False
        return self

    def dot(self, b, out=None, strict=False):
        """Masked dot product (see `dot`)."""
        return dot(self, b, strict=strict, out=out)

    for _name in ("sqrt exp log log10 sin cos tan arcsin arccos arctan sinh cosh tanh floor ceil").split():
        locals()[_name] = _unary_method(_name)
    del _name

    def conjugate(self):
        """Complex conjugate, element-wise."""
        if self.dtype == bool:  # numpy.ma keeps bool (np.conjugate would give int8)
            return self.copy()
        return _ufunc_call(_np.conjugate, (self,))

    conj = conjugate

    def round(self, decimals=0, out=None):
        """Round to ``decimals`` decimals (mask unchanged); ``out`` is a masked or plain array."""
        c = _core()
        xp = self._xp
        if out is not None:
            if _is_xma(out):
                self._data.round(decimals, out=out._data)
                out.__setmask__(self._mask)
            elif hasattr(out, "shape") and hasattr(out, "dtype") and not _is_mconst(out):
                self._data.round(decimals, out=out)  # plain ndarray: filled with the data, as numpy.ma
            else:
                raise TypeError(f"'out' must be an array, not {type(out).__name__}")
            return out
        res = xp.asarray(self._data.round(decimals))
        if res.ndim == 0:
            return c._maybe_scalar(res, self._mask)
        m = self._mask
        return c._wrap(res, m if m is nomask else m.copy(), like=self)

    def apply_ufunc(self, ufunc, *args, **kwargs):
        """Apply ``ufunc(self, *args, **kwargs)`` with masked-array semantics.

        Non-ufunc callables (``xp.round``) are applied to the data, keeping the mask.
        """
        if hasattr(ufunc, "nin"):
            return _ufunc_call(ufunc, (self,) + args, kw=kwargs)
        res = ufunc(self._data, *args, **kwargs)
        m = self._mask
        return _core()._wrap(res, m if m is nomask else m.copy(), like=self)

    # ---- numpy protocols -----------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, where=True, **kwargs):
        if method == "__call__":
            if len(inputs) == 2 and out is None and where is True and not kwargs \
                    and not _is_xma(inputs[0]) and _is_xma(inputs[1]):
                # `ndarray <op> masked_array`: numpy.ma answers with its reflected operator
                res = _reflected(ufunc, inputs[0], inputs[1])
                if res is not NotImplemented:
                    return res
            if ufunc is _np.matmul:
                if out is not None or where is not True or kwargs:
                    return NotImplemented
                return _product(inputs[0], inputs[1], "matmul")
            return _ufunc_call(ufunc, inputs, out, where, kwargs)
        if method in ("reduce", "accumulate"):
            return _ufunc_scan(ufunc, method, inputs, out, where, kwargs)
        if method == "outer":
            return _ufunc_outer(ufunc, inputs, kwargs)
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        impl = _func_table().get(func)
        if impl is None or not _builtins.all(_acceptable_type(t) for t in types):
            return NotImplemented
        return impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# module-level masked functions (numpy.ma.add, numpy.ma.sqrt, ...)
# ---------------------------------------------------------------------------
def round(a, decimals=0, out=None):
    """Round an array to the given number of decimals (masked-array aware)."""
    c = _core()
    return (a if _is_xma(a) else c.MaskedArray(a)).round(decimals, out)


def round_(a, decimals=0, out=None):
    """Deprecated alias of `round` (deprecated in numpy.ma 2.5 too)."""
    import warnings

    warnings.warn(
        "numpy.ma.round_ is deprecated. Use numpy.ma.round instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return round(a, decimals, out)


def _module_func(name, kind):
    if kind == "unary":
        def func(a, *args, **kwargs):
            return _munary(name, a, *args, **kwargs)
    elif kind == "binary":
        def func(a, b, *args, **kwargs):
            return _mbin(name, a, b, *args, **kwargs)
    else:
        def func(a, b):
            return _dbin(name, a, b)
    func.__name__ = func.__qualname__ = name
    func.__doc__ = f"Masked version of ``numpy.{name}`` (see ``numpy.ma.{name}``)."
    return func


for _k, (_n, _f, _d) in UNARY_OPS.items():
    globals()[_k] = _module_func(_n, "unary")
    __all__.append(_k)
for _k, (_n, _fx, _fy) in BINARY_OPS.items():
    globals()[_k] = _module_func(_n, "binary")
    __all__.append(_k)
for _k, (_n, _d, _fx, _fy) in DOMAINED_BINARY_OPS.items():
    globals()[_k] = _module_func(_n, "domained")
    __all__.append(_k)
del _k, _n, _f, _d, _fx, _fy


# ---------------------------------------------------------------------------
# clip / choose / ids / trace (numpy.ma module-level functions)
# ---------------------------------------------------------------------------
def _as_xma(a):
    """``asanyarray``: masked arrays pass; host/cupy arrays are wrapped where they are (no copy)."""
    c = _core()
    if _is_xma(a):
        return a
    if isinstance(a, _np.ndarray) or _is_cupy_array(a):
        return c._wrap(a, nomask)
    return c.MaskedArray(a)


def clip(a, a_min=_NV, a_max=_NV, out=None, *, min=_NV, max=_NV, **kwargs):
    """Clip the values of an array (``numpy.ma.clip``); returns a masked array.

    Mask handling is that of `MaskedArray.clip`.  ``fill_value`` and ``hardmask``
    (keyword-only) set those attributes of the result.
    """
    extra = {k: kwargs.pop(k) for k in ("fill_value", "hardmask") if k in kwargs}
    if a_min is _NV and a_max is _NV:
        a_min = None if min is _NV else min
        a_max = None if max is _NV else max
    elif a_min is _NV:
        raise TypeError("clip() missing 1 required positional argument: 'a_min'")
    elif a_max is _NV:
        raise TypeError("clip() missing 1 required positional argument: 'a_max'")
    elif min is not _NV or max is not _NV:
        raise ValueError("Passing `min` or `max` keyword argument when "
                         "`a_min` and `a_max` are provided is forbidden.")
    res = _as_xma(a).clip(a_min, a_max, out=out, **kwargs)
    if extra and _is_xma(res):
        if out is not None:  # numpy sets them on a view, not on `out`
            res = res.view()
        if "fill_value" in extra:
            res.fill_value = extra["fill_value"]
        if "hardmask" in extra:
            res._hardmask = bool(extra["hardmask"])
    return res


_clip_params = list(_inspect.signature(clip).parameters.values())
clip.__signature__ = _inspect.Signature(
    _clip_params[:-1]
    + [_inspect.Parameter("fill_value", _inspect.Parameter.KEYWORD_ONLY, default=None),
       _inspect.Parameter("hardmask", _inspect.Parameter.KEYWORD_ONLY, default=False)]
    + _clip_params[-1:]
)
del _clip_params


def _choose_raw(c, data, mode, out=None):
    """``numpy.choose(c, data, mode=mode, out=out)`` for numpy or cupy arrays (``data``: a list).

    cupy has no ``choose`` for a list of arrays, so it is done with ``take_along_axis`` on the
    stacked, broadcast choices.  ``mode='raise'`` must look at the index values: that syncs
    on cupy (``wrap`` and ``clip`` do not).
    """
    xp = _get_xp(c, out, *data)
    if xp is _np:
        return _np.choose(c, data, mode=mode, out=out)
    if mode not in ("raise", "wrap", "clip"):
        raise ValueError("clipmode not understood")
    if not data:
        raise ValueError("0-length sequence.")
    if c.dtype.kind not in "biu":
        raise TypeError(f"Cannot cast array data from dtype('{c.dtype}') to dtype('int64') "
                        "according to the rule 'safe'")
    n = len(data)
    shape = _np.broadcast_shapes(c.shape, *(d.shape for d in data))
    dt = xp.result_type(*data)
    stacked = xp.stack([xp.broadcast_to(d.astype(dt, copy=False), shape) for d in data])
    idx = xp.broadcast_to(c.astype(_np.intp, copy=False), shape)
    if mode == "raise":
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n):
            raise ValueError("invalid entry in choice array")
    elif mode == "wrap":
        idx = idx % n
    else:
        idx = xp.clip(idx, 0, n - 1)
    res = xp.take_along_axis(stacked, idx[None], axis=0)[0]
    if out is None:
        return res
    if out.shape != res.shape:
        raise ValueError(f"output array has shape {out.shape}, expected {res.shape}")
    xp.copyto(out, res, casting="same_kind")
    return out


def _choose_method(self, choices, out, mode):
    """``MaskedArray.choose`` (ndarray semantics): raw data of the choices, own mask kept."""
    c = _core()
    xp = _get_xp(self, out, *list(choices))
    data = [_to_dev(_arr(c.getdata(x)), xp) for x in list(choices)]
    raw = None if out is None else getattr(out, "_data", out)
    res = _choose_raw(_to_dev(self._data, xp), data, mode, raw)
    if out is not None:
        return out
    if res.ndim == 0:
        return _host_scalar(res)
    keep = self._mask is not nomask and res.shape == self._data.shape
    return c._wrap(res, xp.array(_to_dev(self._mask, xp), copy=True) if keep else nomask, like=self)


def choose(indices, choices, out=None, mode="raise"):
    """Use an index array to construct a new array from a list of choices (``numpy.ma.choose``).

    The result is masked where the chosen element or the index is masked.  An all-False
    result mask is kept as an array (numpy.ma shrinks it to `nomask`, which would sync).
    ``mode='raise'`` checks the index range on the device's data: that syncs on cupy.
    """
    c = _core()
    choices = list(choices)
    xp = _get_xp(indices, out, *choices)
    idx = _to_dev(_arr(c.filled(indices, 0)), xp)
    masks, data = [], []
    for x in choices:
        if x is masked:
            masks.append(_np.ones((), bool))
            data.append(_np.ones((), bool))
        else:
            m = c.getmask(x)
            masks.append(_np.zeros((), bool) if m is nomask else m)
            data.append(_arr(c.filled(x)))
    masks = [_to_dev(_arr(m), xp) for m in masks]
    data = [_to_dev(d, xp) for d in data]
    om = _choose_raw(idx, masks, mode)
    d = _choose_raw(idx, data, mode, None if out is None else getattr(out, "_data", out))
    om = _shrink_cheap(_or(om, c.getmask(indices), _shape(d), xp))
    if out is not None:
        if _is_xma(out):
            out.__setmask__(om)
        return out
    return c._wrap(d, om)


def ids(a):
    """Addresses of the data and mask areas of ``a`` (the id of `nomask` if unmasked)."""
    return _as_xma(a).ids()


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Sum along a diagonal of ``a``, masked values counting as 0 (see `MaskedArray.trace`)."""
    return _as_xma(a).trace(offset, axis1, axis2, dtype, out)
