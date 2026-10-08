"""
Helpers for the differential tests of ``xupy.ma`` against ``numpy.ma``.

Convention: a *pair* is (xupy_result, numpy_result) produced from the same
numpy input.  ``assert_same`` compares them structurally (result kind, dtype,
shape, mask, unmasked data, fill_value, device).
"""
import sys

import numpy as np
import pytest

import xupy  # noqa: F401  (populates sys.modules["xupy.ma"])
from xupy import _core

XMA = sys.modules["xupy.ma"]          # XuPy's masked module (never numpy.ma)
cp = _core._cupy                      # cupy module when usable, else None
GPU_OK = cp is not None

DEVICES = ["cpu"] + (
    [pytest.param("gpu", id="gpu")] if GPU_OK else
    [pytest.param("gpu", id="gpu", marks=pytest.mark.skip(reason="no usable GPU"))]
)


# numpy.ma's default fill_value for unsigned dtypes is int64 before numpy 2.2.
NP_LT_22 = np.lib.NumpyVersion(np.__version__) < "2.2.0"
_NP_OLD_UFILL = NP_LT_22


# numpy.ma.sort/argsort(descending=...) arrived in numpy 2.5; XuPy follows 2.5 (descending=True
# raises ValueError "not supported for masked arrays", a falsy value is a no-op).
NP_LT_21 = np.lib.NumpyVersion(np.__version__) < "2.1.0"
NP_LT_25 = np.lib.NumpyVersion(np.__version__) < "2.5.0"


def np_kw(kw):
    """Keywords for the installed numpy.ma: drop a falsy ``descending=``.

    numpy < 2.5 rejects the keyword (TypeError); a falsy value is a no-op on 2.5, so
    dropping it keeps the comparison meaningful.  Identity on numpy >= 2.5.
    """
    if NP_LT_25 and "descending" in kw and not kw["descending"]:
        return {k: v for k, v in kw.items() if k != "descending"}
    return kw


def expected_exc(e, kw=None):
    """Exception type XuPy must raise when numpy.ma raised ``e``.

    Identity on numpy >= 2.5.  On older numpy, two numpy quirks that 2.5 changed are mapped
    to XuPy's (2.5) behaviour, ValueError:
      * ``descending=True`` -> TypeError "unexpected keyword" (2.5: ValueError, unsupported);
      * ``MaskedArray.sort(axis=None)`` on masked data -> TypeError "MaskedIterator has no
        len()" (numpy <= 2.2; 2.5 raises ValueError "indices and arr must have the same
        number of dimensions").
    """
    if NP_LT_25 and isinstance(e, TypeError):
        msg = str(e)
        if "descending" in msg or "MaskedIterator" in msg:
            return ValueError
    return type(e)


def np_repr(n):
    """``repr`` of a numpy.ma result, with the numpy < 2.2 unsigned default fill_value normalised.

    numpy < 2.2 prints the default fill_value of an unsigned array as ``np.int64(999999)``;
    2.2+ and XuPy print ``np.uint64(999999)`` (see ``_NP_OLD_UFILL``).  Identity otherwise.
    """
    r = repr(n)
    if _NP_OLD_UFILL and isinstance(n, np.ma.MaskedArray) and n.dtype.kind == "u":
        fv = np.asarray(n.fill_value)
        if fv.dtype == np.int64:
            r = r.replace(f"fill_value=np.int64({fv.item()})", f"fill_value=np.uint64({fv.item()})")
    return r


def is_np_uint_fill_bug(e):
    """True if numpy.ma raised its unsigned default-fill_value bug (numpy < 2.2).

    With an int64 default fill_value on uint data, ``min``/``max``/``ptp`` of a fully masked lane do
    ``np.copyto(result, fill_value, where=...)`` which raises "Cannot cast scalar from dtype('int64')
    to dtype('uint8')" on numpy 2.1 (observed on 2.1.3; 2.0.2 and 2.2+ are fine).  XuPy returns the
    masked result instead, so there is no oracle: XuPy must merely not raise.
    """
    return NP_LT_22 and isinstance(e, TypeError) and "Cannot cast scalar from dtype('int64') to dtype('uint" in str(e)


def xp_of(dev):
    return cp if dev == "gpu" else np


def to_dev(a, dev):
    """numpy array -> array on ``dev`` (cupy for "gpu")."""
    a = np.asarray(a)
    return cp.asarray(a) if dev == "gpu" else a


def host(x):
    """Any numpy/cupy array (or scalar) -> numpy."""
    if cp is not None and isinstance(x, cp.ndarray):
        return x.get()
    return np.asarray(x)


def on_dev(x, dev):
    """True if the array ``x`` lives on ``dev``."""
    if dev == "gpu":
        return isinstance(x, cp.ndarray)
    return isinstance(x, np.ndarray)


def make(data, mask=None, dev="cpu", **kw):
    """Build the (xupy, numpy.ma) pair from numpy ``data``/``mask``.

    ``mask=None`` -> no mask given (nomask).  Extra keywords (dtype,
    fill_value, hard_mask, keep_mask, ...) go to both constructors.
    """
    data = np.asarray(data)
    mk = {} if mask is None else {"mask": mask}
    n = np.ma.masked_array(data.copy(), **{k: (np.array(v) if k == "mask" else v) for k, v in mk.items()}, **kw)
    xm = None if mask is None else to_dev(np.asarray(mask), dev)
    # Host data goes to the active backend on construction, so build the "cpu"
    # array under the CPU backend (it then stays numpy-backed for its lifetime).
    with xupy.backend(dev):
        x = XMA.masked_array(to_dev(data.copy(), dev), **({} if xm is None else {"mask": xm}), **kw)
    return x, n


def mka(dev, *args, **kw):
    """``XMA.masked_array(*args, **kw)`` built under the ``dev`` backend.

    Host input (numpy/numpy.ma/list) goes to the active backend, so an array
    meant to be numpy-backed is built under the CPU backend.
    """
    with xupy.backend(dev):
        return XMA.masked_array(*args, **kw)


def _tol(dt):
    dt = np.dtype(dt)
    if dt.kind in "fc":
        return float(np.finfo(dt).eps) * 64
    return 0.0


def _assert_values(hx, hn, ctx):
    assert hx.shape == hn.shape, f"{ctx}: shape {hx.shape} != {hn.shape}"
    if hn.dtype.kind in "fc":
        np.testing.assert_allclose(hx, hn, rtol=_tol(hn.dtype), atol=0, equal_nan=True, err_msg=ctx)
    else:
        np.testing.assert_array_equal(hx, hn, err_msg=ctx)


def assert_same(x, n, dev=None, strict_nomask=True, check_fill=True, ctx=""):
    """Assert the XuPy result ``x`` matches the numpy.ma result ``n``.

    Result kinds are distinguished: ``masked`` singleton, numpy scalar, masked
    array, plain ndarray, python object, tuple/list (recursive).
    ``dev`` ("cpu"/"gpu"), when given, checks array results live on that device.
    ``strict_nomask`` also requires ``nomask``-ness of the mask to agree.
    """
    ctx = ctx or "result"
    if n is np.ma.masked:
        assert x is XMA.masked, f"{ctx}: expected xupy.ma.masked, got {type(x)!r}: {x!r}"
    elif isinstance(n, np.ma.MaskedArray):
        assert isinstance(x, XMA.MaskedArray), f"{ctx}: expected XuPy masked array, got {type(x)!r}"
        assert x.dtype == n.dtype and isinstance(x.dtype, np.dtype), f"{ctx}: dtype {x.dtype!r} != {n.dtype!r}"
        assert x.shape == n.shape, f"{ctx}: shape {x.shape} != {n.shape}"
        if dev is not None:
            assert on_dev(x.data, dev), f"{ctx}: data not on {dev}: {type(x.data)}"
            if x.mask is not XMA.nomask:
                assert on_dev(x.mask, dev), f"{ctx}: mask not on {dev}"
        if strict_nomask:
            assert (x.mask is XMA.nomask) == (n.mask is np.ma.nomask), (
                f"{ctx}: nomask mismatch (xupy mask is nomask: {x.mask is XMA.nomask}, "
                f"numpy: {n.mask is np.ma.nomask})")
        mn = np.ma.getmaskarray(n)
        mx = host(XMA.getmaskarray(x))
        assert mx.dtype == np.bool_
        np.testing.assert_array_equal(mx, mn, err_msg=f"{ctx}: mask")
        keep = ~mn
        _assert_values(host(x.data)[keep], np.asarray(n.data)[keep], f"{ctx}: data")
        if check_fill:
            fx, fn = np.asarray(x.fill_value), np.asarray(n.fill_value)
            if fn.dtype != n.dtype and fx.dtype == n.dtype:
                # numpy < 2.5.3 can keep an operand's fill_value dtype on a result of
                # another dtype (int fill on an int / int -> float64 result); XuPy
                # follows newer numpy and casts it to the result dtype.
                fn = fn.astype(n.dtype)
            if (_NP_OLD_UFILL and fx.dtype != fn.dtype and fx.dtype.kind in "iu"
                    and fn.dtype.kind in "iu" and fx.shape == fn.shape and (fx == fn).all()):
                # numpy < 2.2 (checked on 2.0.2) stores the default fill_value of an
                # unsigned-int array as int64 (999999); 2.2+ (and XuPy) keep the
                # array's own signedness.  Integer kinds with equal values only.
                fn = fn.astype(fx.dtype)
            assert fx.dtype == fn.dtype, f"{ctx}: fill_value dtype {fx.dtype} != {fn.dtype}"
            np.testing.assert_array_equal(fx, fn, err_msg=f"{ctx}: fill_value")
        assert bool(x.hardmask) == bool(n.hardmask), f"{ctx}: hardmask"
    elif isinstance(n, np.generic):
        assert isinstance(x, np.generic) and type(x) is type(n), (
            f"{ctx}: expected numpy scalar {type(n).__name__}, got {type(x)!r}: {x!r}")
        _assert_values(np.asarray(x), np.asarray(n), ctx)
    elif isinstance(n, np.ndarray):
        if dev is not None:
            assert on_dev(x, dev), f"{ctx}: expected array on {dev}, got {type(x)!r}"
        else:
            assert hasattr(x, "shape") and not isinstance(x, XMA.MaskedArray), f"{ctx}: got {type(x)!r}"
        assert host(x).dtype == n.dtype, f"{ctx}: dtype {host(x).dtype} != {n.dtype}"
        _assert_values(host(x), n, ctx)
    elif isinstance(n, (tuple, list)):
        assert isinstance(x, (tuple, list)) and len(x) == len(n), f"{ctx}: seq mismatch {x!r}"
        for i, (xi, ni) in enumerate(zip(x, n)):
            assert_same(xi, ni, dev=dev, strict_nomask=strict_nomask, check_fill=check_fill, ctx=f"{ctx}[{i}]")
    else:  # python scalar / None / str / bool
        assert type(x) is type(n) and x == n, f"{ctx}: {x!r} != {n!r}"


def both(fx, fn, *args_pair, **kw):
    """Call ``fx(*xupy_args)`` and ``fn(*numpy_args)``.

    ``args_pair`` is a sequence of (xupy_arg, numpy_arg) pairs.  Returns
    (result_x, result_n).  If numpy raises, XuPy must raise the same
    exception type (returns (None, None) after checking).
    """
    xa = [a for a, _ in args_pair]
    na = [b for _, b in args_pair]
    try:
        rn = fn(*na, **np_kw(kw))
    except Exception as e:   # noqa: BLE001
        if NP_LT_21 and isinstance(e, TypeError) and "a_min" in str(e) and "missing" in str(e):
            # numpy < 2.1: np.ma.clip needs both a_min and a_max positionally; the
            # `min=`/`max=` keywords (and clip(a) alone) arrived in 2.1.  XuPy supports the
            # keywords (superset): there is no oracle, so only require that XuPy accepts the
            # call.  With no bound at all it forwards to ndarray.clip, which on numpy < 2.1
            # raises ValueError("One of max or min must be given") (2.1+: identity).
            if all(kw.get(k) is None for k in ("min", "max")):
                with pytest.raises(ValueError, match="One of max or min"):
                    fx(*xa, **kw)
            else:
                fx(*xa, **kw)
            return None, None
        with pytest.raises(expected_exc(e, kw)):
            fx(*xa, **kw)
        return None, None
    return fx(*xa, **kw), rn
