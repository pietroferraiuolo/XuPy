"""
GPU-mode shims for NumPy 2.x names that CuPy does not provide.

Only :func:`build` is meant to be used (by ``xupy._core``, lazily, the first
time the GPU namespace is needed).  It returns two dicts:

* ``fill``: used only for names that CuPy lacks (native CuPy always wins);
* ``override``: replaces the CuPy attribute of the same name (CuPy's version
  rejects NumPy 2 keywords).

Compute modules without a CuPy equivalent (``emath``, ``strings``, ``char``,
``rec``, ``ctypeslib``, ...) are deliberately *not* forwarded: mixing
host-only modules with device arrays is a trap.
"""

import contextlib as _contextlib
import types as _types

import numpy as _np

#: Pure dtype/metadata objects: CuPy uses NumPy dtypes, so NumPy's are forwarded.
_FORWARD_FROM_NUMPY = (
    "False_", "True_", "ScalarType", "bytes_", "character", "clongdouble",
    "complex256", "datetime64", "datetime_data", "dtype", "dtypes", "exceptions",
    "finfo", "flexible", "float128", "iinfo", "isdtype", "little_endian",
    "longdouble", "object_", "sctypeDict", "str_", "timedelta64", "typecodes",
    "void",
)

_ARRAY_API_VERSION_GPU = "2023.12"


class _LinalgProxy(_types.ModuleType):
    """``cupy.linalg`` plus the NumPy 2 functions it lacks."""

    def __init__(self, base, extras):
        super().__init__("xupy.linalg", getattr(base, "__doc__", None))
        self._base = base
        self._extras = extras

    def __getattr__(self, name):  # only reached when normal lookup fails
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._extras[name]
        except KeyError:
            return getattr(self._base, name)

    def __dir__(self):
        return sorted(set(dir(self._base)) | set(self._extras))


def build(cp, cupyx):
    """Create the shims for CuPy module ``cp`` (and ``cupyx``)."""
    fill = {}
    override = {}

    # -- dtype / metadata forwarding ---------------------------------------
    for name in _FORWARD_FROM_NUMPY:
        if hasattr(_np, name):
            fill[name] = getattr(_np, name)

    # -- floating-point error state -----------------------------------------
    # cupyx.errstate/seterr/geterr only implement `linalg` and raise
    # NotImplementedError for the FP flags.  GPU kernels never trap or warn,
    # i.e. they always behave as "ignore": we accept "ignore"/"warn" (no-ops)
    # so portable code such as `with xp.errstate(divide="ignore")` works, and
    # refuse the modes that cannot be honoured.  seterrcall/geterrcall are
    # unsupported on GPU (left out on purpose).
    def _check_fp(**flags):
        for k, v in flags.items():
            if v not in (None, "ignore", "warn"):
                raise NotImplementedError(
                    f"{k}={v!r} is not supported on GPU (FP errors never trap or warn)"
                )

    def geterr():
        """FP error state; on GPU the FP flags are always 'ignore'."""
        old = dict(cupyx.geterr())
        return {k: ("ignore" if v is None else v) for k, v in old.items()}

    def seterr(all=None, divide=None, over=None, under=None, invalid=None, linalg=None):
        old = geterr()
        _check_fp(all=all, divide=divide, over=over, under=under, invalid=invalid)
        if linalg is None and all in ("ignore", "raise"):
            linalg = all
        cupyx.seterr(linalg=linalg)
        return old

    @_contextlib.contextmanager
    def errstate(*, call=None, all=None, divide=None, over=None, under=None,
                 invalid=None, linalg=None):
        if call is not None:
            raise NotImplementedError("call= is not supported on GPU")
        old = seterr(all, divide, over, under, invalid, linalg)
        try:
            yield
        finally:
            cupyx.seterr(linalg=old.get("linalg"))

    fill.update(errstate=errstate, seterr=seterr, geterr=geterr)

    # -- NumPy 2.x vector/matrix products ----------------------------------
    def vecdot(x1, x2, /, *, axis=-1):
        """sum(conj(x1) * x2, axis) after broadcasting (numpy.vecdot)."""
        x1, x2 = cp.broadcast_arrays(cp.asarray(x1), cp.asarray(x2))
        dt = cp.result_type(x1, x2)
        if dt == cp.bool_:
            return cp.any(x1 & x2, axis=axis)
        return cp.sum(cp.conj(x1) * x2, axis=axis, dtype=dt)

    def matvec(x1, x2, /):
        """Matrix (..., m, n) times vector (..., n) -> (..., m)."""
        return cp.matmul(cp.asarray(x1), cp.asarray(x2)[..., None])[..., 0]

    def vecmat(x1, x2, /):
        """conj(vector) (..., n) times matrix (..., n, m) -> (..., m)."""
        return cp.matmul(cp.conj(cp.asarray(x1))[..., None, :], cp.asarray(x2))[..., 0, :]

    def unstack(x, /, *, axis=0):
        """Split an array into a tuple of arrays along ``axis``."""
        return tuple(cp.moveaxis(cp.asarray(x), axis, 0))

    fill.update(vecdot=vecdot, matvec=matvec, vecmat=vecmat, unstack=unstack)

    # -- sort / argsort / unique with NumPy 2.5 keywords --------------------
    def _no_order(order):
        if order is not None:
            raise NotImplementedError("order= (structured arrays) is not supported on GPU")

    def _desc_key(a):
        # An ascending *stable* sort of this key is a descending stable sort of
        # `a`: ties keep their original order and NaNs stay last, as in NumPy.
        return -a if a.dtype.kind in "fc" else ~a

    # CuPy's sort/argsort are stable in practice (verified against NumPy), so
    # stable=True needs no special handling.
    def sort(a, axis=-1, kind=None, order=None, *, stable=None, descending=None):
        _no_order(order)
        a = cp.asarray(a)
        if descending:
            if axis is None:
                a, axis = a.ravel(), -1
            idx = cp.argsort(_desc_key(a), axis=axis)
            return cp.take_along_axis(a, idx, axis=axis)
        return cp.sort(a, axis=axis, kind=kind)

    def argsort(a, axis=-1, kind=None, order=None, *, stable=None, descending=None):
        _no_order(order)
        a = cp.asarray(a)
        if descending:
            if axis is None:
                a, axis = a.ravel(), -1
            return cp.argsort(_desc_key(a), axis=axis)
        return cp.argsort(a, axis=axis, kind=kind)

    def unique(ar, return_index=False, return_inverse=False, return_counts=False,
               axis=None, *, equal_nan=True, sorted=True):
        # CuPy always returns sorted output, which is valid for sorted=False.
        return cp.unique(ar, return_index, return_inverse, return_counts, axis,
                         equal_nan=equal_nan)

    override.update(sort=sort, argsort=argsort, unique=unique)

    # -- linalg -------------------------------------------------------------
    la = {}

    def matrix_norm(x, /, *, keepdims=False, ord="fro"):
        return cp.linalg.norm(cp.asarray(x), ord=ord, axis=(-2, -1), keepdims=keepdims)

    def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
        x = cp.asarray(x)
        if x.dtype.kind not in "fc":
            x = x.astype(cp.float64)
        if axis is None:
            axis = tuple(range(x.ndim))
        if ord is None:
            ord = 2
        ax = cp.abs(x)
        if ord == float("inf"):
            return cp.max(ax, axis=axis, keepdims=keepdims)
        if ord == float("-inf"):
            return cp.min(ax, axis=axis, keepdims=keepdims)
        if ord == 0:
            return cp.sum(x != 0, axis=axis, keepdims=keepdims, dtype=ax.dtype)
        if ord == 1:
            return cp.sum(ax, axis=axis, keepdims=keepdims)
        if ord == 2:
            sq = x.real ** 2 + x.imag ** 2 if x.dtype.kind == "c" else x * x
            return cp.sqrt(cp.sum(sq, axis=axis, keepdims=keepdims))
        return cp.sum(ax ** ord, axis=axis, keepdims=keepdims) ** (1.0 / ord)

    def diagonal(x, /, *, offset=0):
        return cp.diagonal(cp.asarray(x), offset=offset, axis1=-2, axis2=-1)

    def trace(x, /, *, offset=0, dtype=None):
        return cp.trace(cp.asarray(x), offset=offset, axis1=-2, axis2=-1, dtype=dtype)

    def outer(x1, x2, /):
        return cp.outer(cp.asarray(x1), cp.asarray(x2))

    def svdvals(x, /):
        return cp.linalg.svd(cp.asarray(x), compute_uv=False)

    def tensordot(x1, x2, /, *, axes=2):
        return cp.tensordot(cp.asarray(x1), cp.asarray(x2), axes=axes)

    def multi_dot(arrays, *, out=None):
        if len(arrays) < 2:
            raise ValueError("Expecting at least two arrays.")
        arrays = [cp.asarray(a) for a in arrays]
        res = arrays[0]
        for a in arrays[1:]:
            res = cp.dot(res, a)
        if out is not None:
            out[...] = res
            return out
        return res

    def matrix_transpose(x, /):
        return cp.swapaxes(cp.asarray(x), -1, -2)

    for fn in (matrix_norm, vector_norm, diagonal, trace, outer, svdvals, tensordot,
               multi_dot, matrix_transpose):
        la[fn.__name__] = fn
    la["vecdot"] = vecdot
    # Only the functions cupy.linalg really lacks are added.
    la = {k: v for k, v in la.items() if not hasattr(cp.linalg, k)}
    override["linalg"] = _LinalgProxy(cp.linalg, la)

    # -- Array API metadata -------------------------------------------------
    api_version, api_info = _ARRAY_API_VERSION_GPU, None
    try:  # optional dependency
        import array_api_compat.cupy as _aac  # type: ignore

        api_version = getattr(_aac, "__array_api_version__", api_version)
        api_info = getattr(_aac, "__array_namespace_info__", None)
    except Exception:
        pass
    fill["__array_api_version__"] = api_version
    if api_info is not None:
        fill["__array_namespace_info__"] = api_info

    return fill, override
