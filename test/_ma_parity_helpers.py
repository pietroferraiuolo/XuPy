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
    x = XMA.masked_array(to_dev(data.copy(), dev), **({} if xm is None else {"mask": xm}), **kw)
    return x, n


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
        rn = fn(*na, **kw)
    except Exception as e:   # noqa: BLE001
        with pytest.raises(type(e)):
            fx(*xa, **kw)
        return None, None
    return fx(*xa, **kw), rn
