"""
Differential tests of the XuPy masked reductions / sort family against
``numpy.ma`` (numpy 2.5 semantics), on numpy data ("cpu") and cupy data
("gpu").  numpy.ma is the truth: its quirks are mirrored, and if numpy.ma
raises, XuPy must raise the same exception type.
"""
# NO-SYNC: numpy.ma shrinks the mask to nomask via a host-side .any(); xupy.ma never syncs
# (DESIGN decision 6), so a result mask may be a full all-False array where numpy.ma has
# nomask.  Calls that pass `strict_nomask=False` accept exactly that difference (see NO-SYNC note).

import contextlib
import functools
import warnings  # noqa: F401

import numpy as np
import pytest

from . import _ma_parity_helpers as _H
from ._ma_parity_helpers import XMA, assert_same, host, make, mka, on_dev, to_dev

pytestmark = [
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),   # numpy.ma argsort default-axis notice
]

DTYPES = ["int8", "int32", "float32", "float64", "bool", "complex128"]      # representative set
MASKS = ["none", "partial", "col", "full", "false"]                          # row only where needed


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------
def sample(dtype, shape, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if dt.kind == "b":
        return rng.integers(0, 2, shape).astype(bool)
    if dt.kind == "u":
        return rng.integers(1, 5, shape).astype(dt)
    if dt.kind == "i":
        return rng.integers(-3, 5, shape).astype(dt)
    if dt.kind == "f":
        return rng.normal(size=shape).astype(dt)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dt)


def mask_for(kind, shape, seed=1):
    if kind == "none":
        return None
    rng = np.random.default_rng(seed)
    if kind == "partial":
        return np.asarray(rng.random(shape) < 0.4)
    if kind == "full":
        return np.ones(shape, bool)
    if kind == "false":
        return np.zeros(shape, bool)
    m = np.zeros(shape, bool)
    if m.size == 0 or m.ndim == 0:
        return m
    if kind == "col":          # last-index-0 lane(s) fully masked
        m[..., 0] = True
    elif kind == "row":        # first row fully masked
        m[0] = True
    else:  # pragma: no cover
        raise ValueError(kind)
    return m


def pair(dtype, shape, mkind, dev, seed=0, **kw):
    return make(sample(dtype, shape, seed), mask_for(mkind, shape), dev, **kw)


def call(x, n, name, args=(), kw=None, kwx=None, argsx=None):
    """Call method ``name`` on both; if numpy raises, XuPy must raise the same type."""
    kwn = dict(kw or {})
    kwx = kwn if kwx is None else kwx
    argsx = args if argsx is None else argsx
    try:
        rn = getattr(n, name)(*args, **kwn)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            getattr(x, name)(*argsx, **kwx)
        return None, None
    return getattr(x, name)(*argsx, **kwx), rn


def callf(fx, fn, x, n, args=(), kw=None, kwx=None):
    kwn = dict(kw or {})
    kwx = kwn if kwx is None else kwx
    try:
        rn = fn(n, *args, **kwn)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            fx(x, *args, **kwx)
        return None, None
    return fx(x, *args, **kwx), rn


# GPU accumulation order (cupy reduces float32/complex64/float16 in a different order / precision than
# numpy's pairwise sum): sum-like reductions are compared with a looser rtol, ONLY on gpu and ONLY when a
# low-precision float is involved.  Everything else keeps the helpers' default tolerance.
F32_RTOL = float(np.finfo(np.float32).eps) * 256
F16_RTOL = float(np.finfo(np.float16).eps) * 64
BIG_RTOL = 1e-12   # 1e6-element float64 lanes: gpu tree reduction vs numpy pairwise sum differ by ~1e-13
SUMLIKE = ("sum", "prod", "mean", "var", "std", "cumsum", "cumprod", "average", "anom")


def acc_rtol(dev, name, *dtypes):
    """Looser rtol for gpu sum-like reductions on float32/complex64/float16 data or ``dtype=``."""
    if dev != "gpu" or name not in SUMLIKE:
        return None
    kinds = {np.dtype(d).name for d in dtypes if d is not None}
    if "float16" in kinds:
        return F16_RTOL
    if kinds & {"float32", "complex64"}:
        return F32_RTOL
    return None


@contextlib.contextmanager
def loose(rtol):
    orig = _H._tol
    _H._tol = lambda dt: max(orig(dt), rtol) if np.dtype(dt).kind in "fc" else orig(dt)
    try:
        yield
    finally:
        _H._tol = orig


def check(rx, rn, dev, ctx="", strict_nomask=True, rtol=None):
    if rn is None and rx is None:
        return
    with loose(rtol) if rtol else contextlib.nullcontext():
        assert_same(rx, rn, dev=dev, ctx=ctx, strict_nomask=strict_nomask)


def sn(name):
    """``strict_nomask`` for reduction ``name``.

    numpy.ma's ``var(axis=...)`` (and what is built on it: ``out=`` results of std/var/ptp) shrinks the
    all-False result mask to ``nomask`` through a data-dependent ``mask_or(..., shrink=True)``.  XuPy does not
    sync the host (DESIGN.md decision 6), so it returns an all-False mask array instead: relax ONLY the
    nomask-ness for those; mask values, data, dtype and fill_value are still compared.
    """
    return name != "var"


def inplace(x, n, name, kw):
    """In-place method on both; True if it ran (and returned None) on both."""
    try:
        r = getattr(n, name)(**kw)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            getattr(x, name)(**kw)
        return False
    assert r is None
    assert getattr(x, name)(**kw) is None
    return True


@functools.lru_cache(maxsize=None)
def _big(dtype, kind):
    rng = np.random.default_rng(7)
    shape = (1000, 1000)
    if np.dtype(dtype).kind == "f":
        d = (rng.integers(-8, 9, shape) / 8.0).astype(dtype)
    else:
        d = rng.integers(0, 4, shape).astype(dtype)
    m = rng.random(shape) < 0.3
    if kind == "col":
        m[:, 5] = True
    return d, m



def run(dtype, shape, mkind, dev, name, ctx="", seed=0, **kw):
    """pair() + method call + check, with a descriptive context."""
    x, n = pair(dtype, shape, mkind, dev, seed=seed)
    rx, rn = call(x, n, name, kw=kw)
    check(rx, rn, dev, f"{name}({dtype},{mkind},{shape},{kw}) {ctx}", strict_nomask=sn(name),
          rtol=acc_rtol(dev, name, dtype, kw.get("dtype")))
    return rx, rn


AXES_3D = [None, 0, -1, (0, 2), (1, 2), (0, 1, 2)]
REDUCE = ["sum", "prod", "mean", "var", "std", "min", "max", "any", "all", "count", "ptp"]
REDUCE8 = ["sum", "prod", "mean", "var", "std", "min", "max", "ptp"]


# --------------------------------------------------------------------------
# the reduction matrices (hand-trimmed: loops over small tables inside tests)
# --------------------------------------------------------------------------
class TestReductionMatrix2D:
    @pytest.mark.parametrize("name", REDUCE)
    @pytest.mark.parametrize("dtypes", [("int8", "int32", "bool"), ("float32", "float64", "complex128")], ids=str)
    def test_method(self, dev, name, dtypes):
        for dtype in dtypes:
            for mkind in MASKS:
                for axis in (None, 0, -1, (0, 1)):
                    run(dtype, (3, 4), mkind, dev, name, axis=axis)

    @pytest.mark.parametrize("name", REDUCE)
    def test_invalid_and_positional_axis(self, dev, name):
        for axis in (2, -3, (0, 0), (0, 2)):
            run("float64", (3, 4), "partial", dev, name, axis=axis)
        x, n = pair("float64", (3, 4), "partial", dev)
        rx, rn = call(x, n, name, args=(1,))
        check(rx, rn, dev, f"{name} positional axis", strict_nomask=sn(name))

    @pytest.mark.parametrize("name", REDUCE)
    def test_keepdims(self, dev, name):
        for dtype in ("int32", "float64", "bool"):
            for mkind in ("none", "partial", "col", "full"):
                for axis in (None, 0, 1, (0, 1)):
                    for kd in (True, False):
                        run(dtype, (3, 4), mkind, dev, name, axis=axis, keepdims=kd)


class TestReductionMatrix3D:
    @pytest.mark.parametrize("name", REDUCE)
    def test_method(self, dev, name):
        for dtype in ("int32", "float64"):
            for mkind in ("none", "partial", "full", "col", "row"):
                for axis in AXES_3D:
                    for kd in (False, True):
                        run(dtype, (2, 3, 4), mkind, dev, name, axis=axis, keepdims=kd)

    @pytest.mark.parametrize("name", ["sum", "prod", "min", "max", "count", "any", "all"])
    def test_tuple_axis_with_fully_masked_lanes(self, dev, name):
        d = sample("int32", (2, 3, 4))
        m = np.zeros((2, 3, 4), bool)
        m[:, 1, :] = True        # lane for axis (0, 2) at index 1 is fully masked
        m[0, 2, :] = True
        x, n = make(d, m, dev)
        for ax in [(0, 2), (0, 1), (1, 2)]:
            for kd in (False, True):
                rx, rn = call(x, n, name, kw={"axis": ax, "keepdims": kd})
                check(rx, rn, dev, f"{name} axis={ax} kd={kd}")


class TestEdgeShapes:
    SHAPES = [(), (1,), (5,), (0,), (0, 3), (3, 0), (2, 0, 2), (1, 1)]

    @pytest.mark.parametrize("name", REDUCE)
    def test_method(self, dev, name):
        for shape in self.SHAPES:
            for mkind in ("none", "partial", "full"):
                for axis in (None, 0, -1, 1):
                    for dtype in ("int32", "float64"):
                        run(dtype, shape, mkind, dev, name, axis=axis)

    @pytest.mark.parametrize("name", ["argmin", "argmax", "cumsum", "cumprod"])
    def test_scan_and_arg(self, dev, name):
        for shape in self.SHAPES:
            for mkind in ("none", "partial", "full"):
                for axis in (None, 0, -1):
                    run("float64", shape, mkind, dev, name, axis=axis)

    def test_zero_d_scalar_kinds(self, dev):
        for name in REDUCE8:
            for m in (None, True, False):
                x, n = make(np.array(2.5), None if m is None else np.array(m), dev)
                rx, rn = call(x, n, name)
                check(rx, rn, dev, f"{name} mask={m}")


# --------------------------------------------------------------------------
# result KIND (scalar types, masked singleton, no nan leakage)
# --------------------------------------------------------------------------
class TestResultKind:
    TYPES = [
        ("int32", "sum", np.int64), ("int8", "sum", np.int64), ("uint8", "sum", np.uint64),
        ("float32", "sum", np.float32), ("float64", "sum", np.float64),
        ("bool", "sum", np.int64), ("complex128", "sum", np.complex128),
        ("int32", "prod", np.int64), ("float32", "prod", np.float32),
        ("int32", "mean", np.float64), ("float32", "mean", np.float64),
        ("int8", "min", np.int8), ("float32", "max", np.float32),
        ("int32", "any", np.bool_), ("float64", "all", np.bool_),
        ("float32", "std", np.float64), ("int32", "var", np.float64),
        ("complex128", "std", np.float64),
    ]

    def test_scalar_type(self, dev):
        for dtype, name, etype in self.TYPES:
            x, n = pair(dtype, (4, 5), "partial", dev)
            rx, rn = call(x, n, name)
            check(rx, rn, dev, f"{name}({dtype})", rtol=acc_rtol(dev, name, dtype))
            assert type(rx) is etype, (name, dtype, type(rx), type(rn))

    def test_count_returns_numpy_int(self, dev):
        x, n = pair("float64", (4, 5), "partial", dev)
        c = x.count()
        assert isinstance(c, (int, np.integer)) and not isinstance(c, bool)
        assert type(c) is type(n.count()) and int(c) == int(n.count())
        check(*call(x, n, "count", kw={"axis": 0}), dev)

    def test_fully_masked_scalar_is_masked_singleton(self, dev):
        for name in REDUCE8:
            for dtype in ("int32", "float64", "float32", "uint8"):
                x, _ = pair(dtype, (3, 4), "full", dev)
                assert getattr(x, name)() is XMA.masked, (name, dtype)
                x0, _ = pair(dtype, (5,), "full", dev)
                assert getattr(x0, name)(axis=0) is XMA.masked, (name, dtype)

    def test_int_axis_with_masked_lane_no_nan_leak(self, dev):
        for name in REDUCE8:
            for dtype in ("int8", "int32", "uint8", "bool"):
                x, n = pair(dtype, (4, 3), "col", dev)       # column 0 fully masked
                for axis in (0, 1, -1):
                    rx, rn = call(x, n, name, kw={"axis": axis})
                    check(rx, rn, dev, f"{name} {dtype} axis={axis}", strict_nomask=sn(name))
                    if isinstance(rx, XMA.MaskedArray):
                        assert isinstance(rx.data, type(x.data))
                        assert rx.dtype == rn.dtype
                        if rx.dtype.kind in "iub":
                            assert not np.issubdtype(rx.dtype, np.floating)

    def test_mean_fully_masked_int_axis_keeps_float64(self, dev):
        x, n = pair("int32", (3, 4), "col", dev)
        r = x.mean(axis=0)
        assert isinstance(r, XMA.MaskedArray)
        assert r.dtype == np.float64
        assert bool(host(XMA.getmaskarray(r))[0]) is True
        assert not host(XMA.getmaskarray(r))[1:].any()
        check(r, n.mean(axis=0), dev)

    @pytest.mark.parametrize("name", ["any", "all"])
    def test_any_all_axis_are_masked_arrays(self, dev, name):
        x, n = pair("float64", (3, 4), "partial", dev)
        r = getattr(x, name)(axis=0)
        assert isinstance(r, XMA.MaskedArray)
        check(r, getattr(n, name)(axis=0), dev)
        r = getattr(x, name)(axis=1, keepdims=True)
        assert r.shape == (3, 1)
        check(r, getattr(n, name)(axis=1, keepdims=True), dev)

    def test_sum_dtype_f4(self, dev):
        x, n = pair("float64", (3, 4), "partial", dev)
        r = x.sum(dtype="f4")
        assert type(r) is np.float32
        check(r, n.sum(dtype="f4"), dev)
        check(x.sum(axis=0, dtype="f4"), n.sum(axis=0, dtype="f4"), dev)
        check(x.sum(dtype=np.float32), n.sum(dtype=np.float32), dev)

    def test_std_ddof1_axis_none(self, dev):
        x, n = pair("float64", (6, 7), "partial", dev)
        for ddof in (0, 1, 2):
            check(x.std(ddof=ddof), n.std(ddof=ddof), dev)
            check(x.var(ddof=ddof), n.var(ddof=ddof), dev)
        x, n = pair("float64", (6, 7), "none", dev)
        check(x.std(ddof=1), n.std(ddof=1), dev)

    def test_keepdims_not_ignored(self, dev):
        x, n = pair("float64", (3, 4), "partial", dev)
        for name in REDUCE:
            r = getattr(x, name)(axis=1, keepdims=True)
            assert r.shape == (3, 1), name
            r = getattr(x, name)(keepdims=True)
            assert np.shape(r) == (1, 1), name

    def test_nan_inf_propagate_unmasked(self, dev):
        d = np.array([[1.0, np.nan, 3.0], [np.inf, 2.0, -np.inf]])
        m = np.array([[False, False, True], [False, False, False]])
        x, n = make(d, m, dev)
        for name in REDUCE8 + ["any", "all"]:
            for axis in (None, 0, 1):
                rx, rn = call(x, n, name, kw={"axis": axis})
                check(rx, rn, dev, f"{name} axis={axis}", strict_nomask=sn(name))

    def test_nan_in_masked_position_ignored(self, dev):
        d = np.array([1.0, np.nan, 3.0, np.inf])
        m = np.array([False, True, False, True])
        x, n = make(d, m, dev)
        for name in REDUCE8:
            rx, rn = call(x, n, name)
            check(rx, rn, dev, name)
            assert np.isfinite(float(rx))

    def test_reductions_do_not_mutate(self, dev):
        x, n = pair("float64", (3, 4), "partial", dev)
        for name in REDUCE + ["argmin", "argmax", "cumsum", "cumprod"]:
            getattr(x, name)()
            getattr(x, name)(axis=0)
        x.nonzero()
        x.compressed()
        assert_same(x, n, dev=dev)   # n (numpy.ma, never called) is the pristine reference


# --------------------------------------------------------------------------
# dtype= / out= / ddof / mean= / fill_value
# --------------------------------------------------------------------------
class TestDtypeKeyword:
    @pytest.mark.parametrize("name", ["sum", "prod", "mean", "var", "std", "cumsum", "cumprod"])
    def test_dtype(self, dev, name):
        for src in ("int32", "float64", "bool", "uint8", "float32"):
            for dt in ("f4", "f8", "i8", "c16", "u1", "bool", "float16"):
                for axis in (None, 0):
                    for mkind in ("none", "partial"):
                        if dev == "gpu" and dt == "u1" and np.dtype(src).kind == "f":
                            # cupy limitation: float -> unsigned casts of negative values saturate to 0,
                            # numpy wraps (255); C-level undefined behaviour either way
                            continue
                        x, n = pair(src, (3, 4), mkind, dev)
                        rx, rn = call(x, n, name, kw={"axis": axis, "dtype": dt})
                        check(rx, rn, dev, f"{name}({src},{mkind},axis={axis},dtype={dt})", strict_nomask=sn(name),
                              rtol=acc_rtol(dev, name, src, dt))

    def test_dtype_bogus(self, dev):
        for name in ("sum", "mean"):
            x, n = pair("float64", (3, 4), "partial", dev)
            rx, rn = call(x, n, name, kw={"dtype": "not-a-dtype"})
            assert rn is None


class TestOutKeyword:
    @pytest.mark.parametrize("name", ["sum", "prod", "mean", "std", "var", "min", "max", "any", "all", "ptp",
                                      "cumsum", "cumprod"])
    def test_out_masked_array(self, dev, name):
        for axis in (0, -1):
            for dtype in ("float64", "int32"):
                for mkind in ("none", "partial", "col"):
                    x, n = pair(dtype, (3, 4), mkind, dev)
                    try:
                        ref = getattr(n, name)(axis=axis)
                    except Exception as e:  # noqa: BLE001
                        with pytest.raises(type(e)):
                            getattr(x, name)(axis=axis)
                        continue
                    shape = np.shape(ref)
                    dt = np.asarray(ref).dtype
                    on = np.ma.masked_array(np.zeros(shape, dt))
                    ox = mka(dev, to_dev(np.zeros(shape, dt), dev))
                    rn = getattr(n, name)(axis=axis, out=on)
                    rx = getattr(x, name)(axis=axis, out=ox)
                    ctx = f"{name} {dtype} {mkind} axis={axis}"
                    assert (rn is on) == (rx is ox), ctx
                    check(rx, rn, dev, ctx + " returned", strict_nomask=name not in ("std", "var", "ptp"))
                    assert_same(ox, on, dev=dev, ctx=ctx + " out", strict_nomask=False)  # see NO-SYNC note

    def test_out_wrong_shape(self, dev):
        for name in ("sum", "mean", "min", "max", "cumsum"):
            for shape in [(3,), (4, 4), (2, 5), ()]:
                x, n = pair("float64", (3, 4), "partial", dev)
                on = np.ma.masked_array(np.zeros(shape))
                ox = mka(dev, to_dev(np.zeros(shape), dev))
                try:
                    getattr(n, name)(axis=0, out=on)
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        getattr(x, name)(axis=0, out=ox)
                else:
                    getattr(x, name)(axis=0, out=ox)
                    assert_same(ox, on, dev=dev, strict_nomask=False, ctx=f"{name} {shape}")  # see NO-SYNC note

    def test_out_int_dtype_for_float_result(self, dev):
        for name in ("sum", "mean", "min", "max", "prod"):
            x, n = pair("float64", (3, 4), "partial", dev)
            on = np.ma.masked_array(np.zeros(4, dtype="int64"))
            ox = mka(dev, to_dev(np.zeros(4, dtype="int64"), dev))
            try:
                getattr(n, name)(axis=0, out=on)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    getattr(x, name)(axis=0, out=ox)
            else:
                getattr(x, name)(axis=0, out=ox)
                if dev == "gpu" and name in ("min", "max"):
                    # cupy limitation: a non-finite float cast to int64 gives 0 on the device, INT64_MIN on x86
                    nd = np.ma.getdata(on)
                    nd[nd == np.iinfo(np.int64).min] = 0
                assert_same(ox, on, dev=dev, strict_nomask=False, ctx=name)  # see NO-SYNC note

    def test_out_zero_d_axis_none(self, dev):
        for name in ("sum", "min", "max", "mean", "any"):
            x, n = pair("float64", (3, 4), "partial", dev)
            on = np.ma.masked_array(np.zeros(()))
            ox = mka(dev, to_dev(np.zeros(()), dev))
            try:
                rn = getattr(n, name)(out=on)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    getattr(x, name)(out=ox)
                continue
            rx = getattr(x, name)(out=ox)
            assert (rn is on) == (rx is ox), name
            assert_same(ox, on, dev=dev, strict_nomask=False, ctx=name)  # see NO-SYNC note

    def test_out_plain_ndarray(self, dev):
        for name in ("sum", "min", "mean"):
            x, n = pair("float64", (3, 4), "partial", dev)
            on = np.zeros(4)
            ox = to_dev(np.zeros(4), dev)
            try:
                rn = getattr(n, name)(axis=0, out=on)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    getattr(x, name)(axis=0, out=ox)
                continue
            rx = getattr(x, name)(axis=0, out=ox)
            np.testing.assert_allclose(host(ox), on)
            assert type(rx) is type(ox) or isinstance(rx, XMA.MaskedArray) == isinstance(rn, np.ma.MaskedArray)


class TestDdofAndMean:
    @pytest.mark.parametrize("name", ["std", "var"])
    def test_ddof(self, dev, name):
        for ddof in (0, 1, 2, 5, 12, 13, -1, 0.5):
            for axis in (None, 0, 1):
                for mkind in ("none", "partial", "col", "full"):
                    run("float64", (3, 4), mkind, dev, name, axis=axis, ddof=ddof)

    @pytest.mark.parametrize("name", ["std", "var"])
    def test_ddof_dtypes(self, dev, name):
        for dtype in ("int32", "float32", "complex128", "bool"):
            for axis in (None, 0, 1):
                for ddof in (0, 1, 7):
                    run(dtype, (4, 5), "partial", dev, name, axis=axis, ddof=ddof)

    def test_ddof_ge_count_masks(self, dev):
        d = sample("float64", (3, 3))
        m = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], bool)
        for name in ("std", "var"):
            x, n = make(d, m, dev)
            for ddof in (0, 1, 2, 3):
                rx, rn = call(x, n, name, kw={"axis": 1, "ddof": ddof})
                check(rx, rn, dev, f"{name} ddof={ddof}", strict_nomask=sn(name))

    @pytest.mark.parametrize("name", ["std", "var"])
    def test_mean_keyword(self, dev, name):
        for axis in (None, 0, 1, -1):
            for mkind in ("none", "partial", "col"):
                for ddof in (0, 1):
                    for plain in (False, True):
                        x, n = pair("float64", (3, 4), mkind, dev)
                        mn = n.mean(axis=axis, keepdims=True)
                        mx = x.mean(axis=axis, keepdims=True)
                        if plain:
                            mn = np.asarray(np.ma.filled(mn, 0.0))
                            mx = to_dev(mn, dev)
                        rx, rn = call(x, n, name,
                                      kw={"axis": axis, "ddof": ddof, "keepdims": True, "mean": mn},
                                      kwx={"axis": axis, "ddof": ddof, "keepdims": True, "mean": mx})
                        check(rx, rn, dev, f"{name} axis={axis} {mkind} ddof={ddof} plain={plain}", strict_nomask=sn(name))

    def test_mean_keyword_scalar_and_shifted(self, dev):
        for name in ("std", "var"):
            x, n = pair("float64", (3, 4), "partial", dev)
            for m in (0.0, 1.5):
                check(*call(x, n, name, kw={"mean": m}), dev, f"{name} mean={m}", strict_nomask=sn(name))
                check(*call(x, n, name, kw={"axis": 0, "mean": m}), dev, f"{name} axis0 mean={m}", strict_nomask=sn(name))


class TestFillValueKeyword:
    FILLS = [None, 0, -100, 1e30, np.float64(2.5)]

    @pytest.mark.parametrize("name", ["min", "max", "ptp", "argmin", "argmax"])
    def test_fill_value(self, dev, name):
        for dtype in ("float64", "int32", "float32"):
            for fill in self.FILLS:
                for axis in (None, 0, 1):
                    for mkind in ("partial", "col", "full"):
                        run(dtype, (3, 4), mkind, dev, name, axis=axis, fill_value=fill)

    @pytest.mark.parametrize("name", ["argmin", "argmax"])
    def test_argminmax_respect_mask(self, dev, name):
        for dtype in ("int8", "int32", "float64", "bool", "complex128"):
            for mkind in ("none", "partial", "col", "row", "full"):
                for axis in (None, 0, 1, -1):
                    for kd in (False, True):
                        kw = {"axis": axis, "keepdims": True} if kd else {"axis": axis}
                        run(dtype, (4, 5), mkind, dev, name, **kw)

    def test_argminmax_masked_extreme(self, dev):
        d = np.array([5.0, 100.0, -100.0, 3.0, 7.0])
        m = np.array([False, True, True, False, False])
        for name in ("argmin", "argmax"):
            x, n = make(d, m, dev)
            rx, rn = call(x, n, name)
            check(rx, rn, dev, name)
            assert int(rx) == (4 if name == "argmax" else 3)

    def test_argminmax_out(self, dev):
        for name in ("argmin", "argmax"):
            x, n = pair("float64", (3, 4), "partial", dev)
            on = np.zeros(4, dtype=np.intp)
            ox = to_dev(on.copy(), dev)
            try:
                getattr(n, name)(axis=0, out=on)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    getattr(x, name)(axis=0, out=ox)
                continue
            getattr(x, name)(axis=0, out=ox)
            np.testing.assert_array_equal(host(ox), on)


# --------------------------------------------------------------------------
# sort family
# --------------------------------------------------------------------------
SORT_KW = [
    {}, {"axis": 0}, {"axis": 1}, {"axis": -1}, {"axis": -2}, {"axis": None}, {"axis": 5},
    {"endwith": False}, {"axis": 0, "endwith": False}, {"axis": 1, "endwith": True},
    {"fill_value": 0.0}, {"fill_value": 100.0, "endwith": False}, {"fill_value": -100.0, "endwith": True},
    {"kind": "quicksort"}, {"kind": "mergesort"}, {"kind": "stable"}, {"kind": "heapsort"}, {"kind": "bogus"},
    {"stable": True}, {"stable": False}, {"descending": True}, {"descending": False},
    {"descending": True, "endwith": False}, {"stable": True, "descending": True, "axis": 0},
    {"order": "a"}, {"order": ["a"]},
]


def _gather_equal(x, n, rx, rn, axis, dev):
    """argsort results equal up to ties: gathering must give the same sorted data."""
    fx = host(x.filled(1e9) if x.dtype.kind in "fiu" else x.filled())
    fn = np.ma.filled(n, 1e9) if n.dtype.kind in "fiu" else np.ma.filled(n)
    ix, inn = host(rx), np.asarray(rn)
    if axis is None:
        fx, fn = fx.ravel(), fn.ravel()
        ax = 0
    else:
        ax = axis
    if ix.shape != fx.shape and axis is not None:
        return
    gx = np.take_along_axis(fx, ix, ax)
    gn = np.take_along_axis(fn, inn, ax)
    np.testing.assert_array_equal(gx, gn)


class TestSort:
    @pytest.mark.parametrize("dtype", ["float64", "int32", "bool"])
    def test_sort_inplace(self, dev, dtype):
        for kw in SORT_KW:
            for mkind in ("none", "partial", "full", "col"):
                for shape in ((5,), (4, 5)):
                    x, n = pair(dtype, shape, mkind, dev)
                    if inplace(x, n, "sort", kw):
                        assert_same(x, n, dev=dev, ctx=f"sort {kw} {mkind} {shape}")

    def test_sort_edge_shapes(self, dev):
        for mkind in ("none", "partial", "full"):
            for kw in ({}, {"axis": 0}, {"axis": 0, "endwith": False}, {"descending": True}):
                for shape in ((), (0,), (1,), (0, 3), (3, 0)):
                    x, n = pair("float64", shape, mkind, dev)
                    if inplace(x, n, "sort", kw):
                        assert_same(x, n, dev=dev, ctx=f"sort {kw} {mkind} {shape}")

    def test_sort_mask_moves_with_data(self, dev):
        d = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.0])
        m = np.array([False, True, False, False, True, False])
        for endwith in (True, False):
            for descending in (False, True):
                x, n = make(d, m, dev)
                kw = {"endwith": endwith, "descending": descending}
                if inplace(x, n, "sort", kw):
                    assert_same(x, n, dev=dev, ctx=str(kw))
            x, n = make(d, m, dev)
            assert inplace(x, n, "sort", {"endwith": endwith})
            assert_same(x, n, dev=dev)
            mm = host(XMA.getmaskarray(x))
            assert (mm[-2:].all() and not mm[:-2].any()) if endwith else (mm[:2].all() and not mm[2:].any())

    def test_sort_with_nan_in_data(self, dev):
        d = np.array([3.0, np.nan, 1.0, 2.0, np.nan, 0.0])
        m = np.array([False, False, True, False, False, False])
        for kw in ({}, {"endwith": False}, {"descending": True}, {"stable": True}):
            x, n = make(d, m, dev)
            if inplace(x, n, "sort", kw):
                assert_same(x, n, dev=dev, ctx=str(kw))

    def test_sort_structured_order(self, dev):
        dt = np.dtype([("a", "i4"), ("b", "f8")])
        d = np.array([(2, 0.5), (1, 0.7), (2, 0.1), (1, 0.2)], dtype=dt)
        # structured dtypes are unsupported by design (DESIGN.md: numeric and bool data only)
        with pytest.raises((NotImplementedError, TypeError)):
            mka(dev, to_dev(d, dev))

    def test_sort_returns_none_and_is_inplace(self, dev):
        x, n = pair("float64", (6,), "partial", dev)
        assert x.sort() is None
        n.sort()
        assert_same(x, n, dev=dev)

    def test_sort_function_form(self, dev):
        f = XMA.sort
        for kw in SORT_KW[:12]:
            x, n = pair("float64", (4, 5), "partial", dev)
            rx, rn = callf(f, np.ma.sort, x, n, kw=kw)
            check(rx, rn, dev, str(kw))
            assert_same(x, n, dev=dev)   # input untouched

    @pytest.mark.parametrize("shape", [(7,), (4, 5), (2, 3, 4)], ids=str)
    def test_argsort(self, dev, shape):
        for kw in (k for k in SORT_KW if "order" not in k):
            for mkind in ("none", "partial", "full", "col"):
                x, n = pair("float64", shape, mkind, dev)
                rx, rn = call(x, n, "argsort", kw=kw)
                if rn is None:
                    continue
                ctx = f"argsort {kw} {mkind}"
                assert isinstance(rn, np.ndarray) and not isinstance(rn, np.ma.MaskedArray)
                assert on_dev(rx, dev) and host(rx).dtype == rn.dtype and rx.shape == rn.shape, ctx
                if kw.get("stable") or kw.get("kind") in ("stable", "mergesort"):
                    np.testing.assert_array_equal(host(rx), rn, err_msg=ctx)
                else:
                    _gather_equal(x, n, rx, rn, kw.get("axis", -1), dev)

    def test_argsort_stable_ties(self, dev):
        for dtype in ("int32", "uint8", "bool", "float32", "complex128"):
            for mkind in ("none", "partial", "col"):
                x, n = pair(dtype, (4, 6), mkind, dev)
                for ax in (0, 1, -1):
                    for kw in ({"axis": ax, "kind": "stable"}, {"axis": ax, "stable": True},
                               {"axis": ax, "kind": "stable", "endwith": False}):
                        rx, rn = call(x, n, "argsort", kw=kw)
                        check(rx, rn, dev, f"{dtype} {mkind} {kw}")

    def test_argsort_function_form(self, dev):
        f = XMA.argsort
        x, n = pair("int32", (4, 5), "partial", dev)
        rx, rn = callf(f, np.ma.argsort, x, n, kw={"axis": 1, "kind": "stable"})
        check(rx, rn, dev)


# --------------------------------------------------------------------------
# misc array methods
# --------------------------------------------------------------------------
class TestNonzeroClipCompressed:
    SHAPES6 = [(), (6,), (3, 4), (2, 3, 2), (0,), (0, 2)]

    @pytest.mark.parametrize("name", ["nonzero", "compressed"])
    def test_nonzero_compressed(self, dev, name):
        for dtype in DTYPES:
            for mkind in MASKS:
                for shape in self.SHAPES6:
                    x, n = pair(dtype, shape, mkind, dev)
                    rx, rn = call(x, n, name)
                    check(rx, rn, dev, f"{name} {dtype} {mkind} {shape}")

    @pytest.mark.parametrize("dtype", ["float64", "int32", "float32", "uint8"])
    def test_clip_scalars(self, dev, dtype):
        bounds_list = [
            (-0.5, 0.5), (None, 0.5), (-0.5, None), (0, 2), (2, 0), (1, 1), (np.nan, 1.0), (-1, np.inf),
            (1000, 2000), (-2000, -1000), (None, None), (-1.5, 3),
        ]
        for mkind in ("none", "partial", "full", "col"):
            for bounds in bounds_list:
                x, n = pair(dtype, (3, 4), mkind, dev)
                check(*call(x, n, "clip", args=bounds), dev, f"clip {dtype} {mkind} {bounds}")
                check(*call(x, n, "clip", kw={"min": bounds[0], "max": bounds[1]}), dev,
                      f"clip kw {dtype} {mkind} {bounds}")

    def test_clip_array_bounds(self, dev):
        lo = np.array([-1.0, -0.5, 0.0, 0.5])
        hi = np.array([0.5, 1.0, 1.5, 2.0])
        for mkind in ("none", "partial", "col"):
            x, n = pair("float64", (3, 4), mkind, dev)
            check(*call(x, n, "clip", kw={"min": lo, "max": hi},
                        kwx={"min": to_dev(lo, dev), "max": to_dev(hi, dev)}), dev)
            check(*call(x, n, "clip", kw={"min": lo}, kwx={"min": to_dev(lo, dev)}), dev)
            rx, rn = call(x, n, "clip", kw={"min": np.zeros(7)}, kwx={"min": to_dev(np.zeros(7), dev)})
            assert rn is None  # broadcast error in both

    def test_clip_does_not_modify_input_and_mask_kept(self, dev):
        x, n = pair("float64", (3, 4), "partial", dev)
        r = x.clip(-0.2, 0.2)
        assert_same(x, n, dev=dev)
        np.testing.assert_array_equal(host(XMA.getmaskarray(r)), np.ma.getmaskarray(n))

    def test_clip_out(self, dev):
        x, n = pair("float64", (3, 4), "partial", dev)
        on = np.ma.masked_array(np.zeros((3, 4)))
        ox = mka(dev, to_dev(np.zeros((3, 4)), dev))
        try:
            rn = n.clip(-0.2, 0.2, out=on)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x.clip(-0.2, 0.2, out=ox)
            return
        rx = x.clip(-0.2, 0.2, out=ox)
        assert (rn is on) == (rx is ox)
        assert_same(ox, on, dev=dev, strict_nomask=False)  # see NO-SYNC note

    def test_clip_function_form(self, dev):
        f = XMA.clip
        x, n = pair("float64", (3, 4), "partial", dev)
        rx, rn = callf(f, np.ma.clip, x, n, args=(-0.3, 0.3))
        check(rx, rn, dev)


class TestScans:
    @pytest.mark.parametrize("name", ["cumsum", "cumprod"])
    def test_scan(self, dev, name):
        for dtype in DTYPES:
            for mkind in MASKS:
                for axis in (None, 0, 1, -1, 2):
                    run(dtype, (3, 4), mkind, dev, name, axis=axis)

    @pytest.mark.parametrize("name", ["cumsum", "cumprod"])
    def test_scan_shapes(self, dev, name):
        for shape in ((5,), (2, 3, 4)):
            for axis in (None, 0, 1, 2, -1):
                for mkind in ("none", "partial", "col"):
                    run("int32", shape, mkind, dev, name, axis=axis)

    def test_masked_filled_with_identity_and_mask_stays(self, dev):
        d = np.array([2.0, 3.0, 4.0, 5.0])
        m = np.array([False, True, False, False])
        for name in ("cumsum", "cumprod"):
            x, n = make(d, m, dev)
            rx, rn = call(x, n, name)
            check(rx, rn, dev, name)
            ref = np.cumsum([2.0, 0, 4, 5]) if name == "cumsum" else np.cumprod([2.0, 1, 4, 5])
            np.testing.assert_allclose(host(rx.data)[~m], ref[~m])
            np.testing.assert_array_equal(host(XMA.getmaskarray(rx)), m)

    def test_dtype_scan_int8_promote(self, dev):
        for name in ("cumsum", "cumprod"):
            x, n = pair("int8", (3, 4), "partial", dev)
            rx, rn = call(x, n, name, kw={"axis": 1})
            check(rx, rn, dev, name)
            assert rx.dtype == rn.dtype


class TestTakeTolistRealImag:
    INDICES = [0, 2, -1, 11, [0, 1], [], [3, 3, 0], np.array([[0, 1], [2, 3]]), np.array([5]),
               12, -13, [0, 12], 1.5, "a", None]

    @pytest.mark.parametrize("mkind", ["none", "partial", "col"])
    def test_take(self, dev, mkind):
        for idx in self.INDICES:
            idx_x = to_dev(idx, dev) if isinstance(idx, np.ndarray) else idx
            for mode in ("raise", "wrap", "clip", "bogus"):
                for axis in (None, 0, 1):
                    x, n = pair("float64", (3, 4), mkind, dev)
                    rx, rn = call(x, n, "take", args=(idx,), argsx=(idx_x,), kw={"axis": axis, "mode": mode})
                    check(rx, rn, dev, f"take {idx!r} {mode} {axis}")

    def test_take_dtypes_1d(self, dev):
        for dtype in ("int32", "bool", "complex128", "float32"):
            for mkind in ("none", "partial", "full"):
                x, n = pair(dtype, (6,), mkind, dev)
                for idx in (0, 3, [1, 2, 5], [5, 0]):
                    check(*call(x, n, "take", args=(idx,)), dev, f"{dtype} {mkind} {idx}")

    def test_take_scalar_index_on_masked_element(self, dev):
        x, n = make(np.arange(5.0), np.array([0, 1, 0, 0, 0], bool), dev)
        assert x.take(1) is XMA.masked and n.take(1) is np.ma.masked
        check(x.take(2), n.take(2), dev)

    def test_take_out(self, dev):
        x, n = pair("float64", (6,), "partial", dev)
        on = np.ma.masked_array(np.zeros(3))
        ox = mka(dev, to_dev(np.zeros(3), dev))
        try:
            n.take([0, 1, 2], out=on)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x.take([0, 1, 2], out=ox)
            return
        x.take([0, 1, 2], out=ox)
        assert_same(ox, on, dev=dev, strict_nomask=False)  # see NO-SYNC note

    def test_tolist(self, dev):
        for dtype in DTYPES:
            for mkind in MASKS:
                for shape in ((), (4,), (2, 3), (0,), (2, 2, 2)):
                    x, n = pair(dtype, shape, mkind, dev)
                    rx, rn = call(x, n, "tolist")
                    if rn is not None or rx is not None:
                        ctx = f"tolist {dtype} {mkind} {shape}"
                        assert rx == rn, ctx
                        assert _types(rx) == _types(rn), ctx
                        assert (None in _flat(rx)) == (None in _flat(rn)), ctx

    def test_tolist_fill_value(self, dev):
        for fill in (-1, 0.5, 7):
            x, n = pair("float64", (2, 3), "partial", dev)
            rx, rn = call(x, n, "tolist", kw={"fill_value": fill})
            assert rx == rn

    def test_tolist_masked_is_none(self, dev):
        x, _ = make(np.arange(4), np.array([0, 1, 0, 1], bool), dev)
        assert x.tolist() == [0, None, 2, None]

    @pytest.mark.parametrize("attr", ["real", "imag"])
    def test_real_imag(self, dev, attr):
        for dtype in ("complex128", "float64", "int32", "bool", "float32"):
            for mkind in ("none", "partial", "full", "col"):
                for shape in ((), (5,), (3, 2)):
                    x, n = pair(dtype, shape, mkind, dev)
                    try:
                        rn = getattr(n, attr)
                    except Exception as e:  # noqa: BLE001
                        with pytest.raises(type(e)):
                            getattr(x, attr)
                        continue
                    rx = getattr(x, attr)
                    assert isinstance(rx, XMA.MaskedArray) == isinstance(rn, np.ma.MaskedArray)
                    check(rx, rn, dev, f"{attr} {dtype} {mkind} {shape}")


def _flat(v):
    if isinstance(v, list):
        return [e for s in v for e in _flat(s)]
    return [v]


def _types(v):
    return [type(e) for e in _flat(v)]


class TestCounts:
    def test_count_masked_function(self, dev):
        for mkind in MASKS:
            for axis in (None, 0, 1, -1):
                for dtype in ("float64", "int32", "bool"):
                    x, n = pair(dtype, (3, 4), mkind, dev)
                    rx, rn = callf(XMA.count_masked, np.ma.count_masked, x, n, kw={"axis": axis})
                    check(rx, rn, dev, f"{dtype} {mkind} {axis}")
                    if axis is None:
                        assert isinstance(rx, (int, np.integer))

    def test_count_masked_plain_array_and_scalars(self, dev):
        d = sample("float64", (3, 4))
        check(XMA.count_masked(to_dev(d, dev)), np.ma.count_masked(d), dev)
        check(XMA.count_masked(to_dev(d, dev), axis=0), np.ma.count_masked(d, axis=0), dev)

    def test_count_masked_zero_d_and_empty(self, dev):
        for shape in ((), (0,), (0, 3)):
            for mk in ("none", "partial", "full"):
                x, n = pair("float64", shape, mk, dev)
                rx, rn = callf(XMA.count_masked, np.ma.count_masked, x, n)
                check(rx, rn, dev, f"{shape} {mk}")

    def test_count_masked_method_if_present(self, dev):
        # numpy.ma has no MaskedArray.count_masked method; if XuPy offers one it
        # must agree with the function.
        x, n = pair("float64", (3, 4), "partial", dev)
        m = getattr(x, "count_masked", None)
        if m is None:
            pytest.skip("no count_masked method (as numpy.ma)")
        for ax in (None, 0, 1):
            check(m(axis=ax), np.ma.count_masked(n, axis=ax), dev)

    def test_count_unmasked_if_present(self, dev):
        x, n = pair("float64", (3, 4), "partial", dev)
        f = getattr(XMA, "count_unmasked", None)
        m = getattr(x, "count_unmasked", None)
        if f is None and m is None:
            pytest.skip("count_unmasked not provided (numpy.ma has none)")
        for ax in (None, 0, 1):
            ref = n.count(axis=ax)
            if f is not None:
                check(f(x, axis=ax), ref, dev)
            if m is not None:
                check(m(axis=ax), ref, dev)

    def test_count_plus_count_masked_is_size(self, dev):
        for mkind in MASKS:
            x, n = pair("int32", (3, 4), mkind, dev)
            assert int(x.count()) + int(XMA.count_masked(x)) == 12, mkind
            for axis, length in ((0, 3), (1, 4), (-1, 4)):
                tot = host(x.count(axis=axis)) + host(XMA.count_masked(x, axis=axis))
                np.testing.assert_array_equal(tot, np.full(np.shape(tot), length), err_msg=f"{mkind} {axis}")


# --------------------------------------------------------------------------
# removed forwarding
# --------------------------------------------------------------------------
class TestNoAttributeForwarding:
    def test_hasattr_false(self, dev):
        x, n = pair("float64", (3, 4), "partial", dev)
        for name in ("itemset", "newbyteorder", "tostring", "__cuda_array_interface__",
                     "definitely_not_an_attribute", "_bogus", "nope_xyz", "get", "toarray"):
            if not hasattr(n, name):
                assert not hasattr(x, name), name
                with pytest.raises(AttributeError):
                    getattr(x, name)

    def test_ptp_is_mask_aware(self, dev):
        for dtype in ("float64", "int32", "uint8"):
            for mkind in ("none", "partial", "col", "full"):
                for kw in ({}, {"axis": 0}, {"axis": 1}, {"axis": -1, "keepdims": True},
                           {"fill_value": 3}, {"axis": (0, 1)}):
                    run(dtype, (3, 4), mkind, dev, "ptp", **kw)

    def test_ptp_values(self, dev):
        d = np.array([1.0, 50.0, 3.0, -40.0, 9.0])
        m = np.array([0, 1, 0, 1, 0], bool)
        x, _ = make(d, m, dev)
        assert x.ptp() == 8.0
        assert type(x.ptp()) is np.float64

    def test_methods_exist_on_class(self, dev):
        x, _ = pair("float64", (2, 2), "none", dev)
        for name in ("sum", "mean", "min", "max", "std", "var", "prod", "ptp", "count", "any", "all",
                     "argmin", "argmax", "cumsum", "cumprod", "sort", "argsort", "clip", "nonzero",
                     "compressed", "take", "tolist"):
            assert callable(getattr(x, name)), name
            assert any(name in vars(c) for c in type(x).__mro__), name


# --------------------------------------------------------------------------
# module-level (extras) function forms
# --------------------------------------------------------------------------
FUNCS = ["sum", "mean", "prod", "product", "std", "var", "min", "max"]


class TestFunctionForms:
    @pytest.mark.parametrize("name", FUNCS)
    def test_function(self, dev, name):
        for dtype in ("int32", "float64", "float32", "bool", "complex128"):
            for mkind in MASKS:
                for axis in (None, 0, 1, (0, 1)):
                    x, n = pair(dtype, (3, 4), mkind, dev)
                    rx, rn = callf(getattr(XMA, name), getattr(np.ma, name), x, n, kw={"axis": axis})
                    check(rx, rn, dev, f"{name} {dtype} {mkind} {axis}", strict_nomask=sn(name),
                          rtol=acc_rtol(dev, name, dtype))

    @pytest.mark.parametrize("name", FUNCS)
    def test_function_kwargs(self, dev, name):
        kws = [
            {"keepdims": True}, {"axis": 0, "keepdims": True}, {"axis": 1, "keepdims": False},
            {"dtype": "f4"}, {"axis": 0, "dtype": "f8"}, {"dtype": "i8"},
            {"ddof": 1}, {"axis": 0, "ddof": 1}, {"ddof": 20}, {"axis": 1, "ddof": 2, "keepdims": True},
            {"fill_value": 3}, {"axis": 1, "fill_value": -5},
        ]
        for kw in kws:
            for mkind in ("none", "partial", "col"):
                x, n = pair("float64", (3, 4), mkind, dev)
                rx, rn = callf(getattr(XMA, name), getattr(np.ma, name), x, n, kw=kw)
                check(rx, rn, dev, f"{name} {kw} {mkind}", strict_nomask=sn(name),
                      rtol=acc_rtol(dev, name, "float64", kw.get("dtype")))

    def test_function_on_plain_arrays(self, dev):
        for name in FUNCS:
            for dtype in ("int32", "float64", "float32"):
                d = sample(dtype, (3, 4))
                for axis in (None, 0, 1):
                    rx, rn = callf(getattr(XMA, name), getattr(np.ma, name), to_dev(d, dev), d, kw={"axis": axis})
                    check(rx, rn, dev, f"{name} {dtype} {axis}")

    def test_function_on_lists_and_scalars(self, dev):
        for name in FUNCS:
            for obj in ([1.0, 2.0, 3.0], [[1, 2], [3, 4]], 3.5):
                try:
                    rn = getattr(np.ma, name)(obj)
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        getattr(XMA, name)(obj)
                    continue
                rx = getattr(XMA, name)(obj)
                if isinstance(rn, np.generic):
                    assert type(rx) is type(rn), (name, obj)
                    np.testing.assert_allclose(rx, rn)

    def test_prod_nomask_returns_masked_array_like_numpy(self, dev):
        for name in ("prod", "product"):
            for dtype in ("int32", "float64"):
                x, n = pair(dtype, (3, 4), "none", dev)
                rx, rn = callf(getattr(XMA, name), getattr(np.ma, name), x, n, kw={"axis": 0})
                check(rx, rn, dev, f"{name} {dtype}")
                assert isinstance(rx, XMA.MaskedArray) == isinstance(rn, np.ma.MaskedArray)

    def test_minmax_function_fill_value(self, dev):
        for name in ("min", "max"):
            for fill in (None, 0, 99, -99):
                for axis in (None, 0, 1):
                    for mkind in ("partial", "col", "full"):
                        x, n = pair("int32", (3, 4), mkind, dev)
                        rx, rn = callf(getattr(XMA, name), getattr(np.ma, name), x, n,
                                       kw={"axis": axis, "fill_value": fill})
                        check(rx, rn, dev, f"{name} {fill} {axis} {mkind}")

    def test_std_var_function_ddof_and_mean(self, dev):
        for name in ("std", "var"):
            for ddof in (0, 1, 3):
                x, n = pair("float64", (3, 4), "partial", dev)
                rx, rn = callf(getattr(XMA, name), getattr(np.ma, name), x, n, kw={"ddof": ddof})
                check(rx, rn, dev, f"{name} ddof={ddof}", strict_nomask=sn(name))
                mn = n.mean(axis=0, keepdims=True)
                mx = x.mean(axis=0, keepdims=True)
                rx, rn = callf(getattr(XMA, name), getattr(np.ma, name), x, n,
                               kw={"axis": 0, "ddof": ddof, "keepdims": True, "mean": mn},
                               kwx={"axis": 0, "ddof": ddof, "keepdims": True, "mean": mx})
                check(rx, rn, dev, f"{name} mean= ddof={ddof}", strict_nomask=sn(name))

    def test_function_out(self, dev):
        for name in ("sum", "mean", "min", "max"):
            x, n = pair("float64", (3, 4), "partial", dev)
            on = np.ma.masked_array(np.zeros(4))
            ox = mka(dev, to_dev(np.zeros(4), dev))
            try:
                rn = getattr(np.ma, name)(n, axis=0, out=on)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    getattr(XMA, name)(x, axis=0, out=ox)
                continue
            rx = getattr(XMA, name)(x, axis=0, out=ox)
            assert (rn is on) == (rx is ox), name
            assert_same(ox, on, dev=dev, strict_nomask=False, ctx=name)  # see NO-SYNC note


class TestAverage:
    def test_unweighted(self, dev):
        for dtype in ("int32", "int8", "uint8", "float32", "float64", "bool", "complex128"):
            for mkind in ("none", "partial", "full", "col", "row"):
                for axis in (None, 0, 1, -1):
                    for returned in (False, True):
                        x, n = pair(dtype, (3, 4), mkind, dev)
                        rx, rn = callf(XMA.average, np.ma.average, x, n, kw={"axis": axis, "returned": returned})
                        check(rx, rn, dev, f"average {dtype} {mkind} {axis} {returned}", rtol=acc_rtol(dev, "average", dtype))

    @pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
    def test_weighted(self, dev, dtype):
        rng = np.random.default_rng(3)
        ws = {
            "same": rng.random((3, 4)), "axis0": rng.random(3), "axis1": rng.random(4),
            "ones": np.ones(4), "zeros": np.zeros((3, 4)), "int": rng.integers(1, 4, (3, 4)),
            "wrong": rng.random(5), "wrong2d": rng.random((4, 3)),
        }
        for mkind in ("none", "partial", "full", "col"):
            for wkind, w in ws.items():
                for axis in (None, 0, 1):
                    for returned in (False, True):
                        x, n = pair(dtype, (3, 4), mkind, dev)
                        rx, rn = callf(XMA.average, np.ma.average, x, n,
                                       kw={"weights": w, "axis": axis, "returned": returned},
                                       kwx={"weights": to_dev(w, dev), "axis": axis, "returned": returned})
                        check(rx, rn, dev, f"average {mkind} w={wkind} {axis} {returned}")

    def test_result_dtype_not_forced_float32(self, dev):
        for dtype in ("float32", "float64", "int32"):
            for wdtype in ("float32", "float64", "int32", "int8"):
                x, n = pair(dtype, (3, 4), "partial", dev)
                w = np.random.default_rng(0).integers(1, 5, (3, 4)).astype(wdtype)
                rx, rn = callf(XMA.average, np.ma.average, x, n, kw={"weights": w, "axis": 0},
                               kwx={"weights": to_dev(w, dev), "axis": 0})
                check(rx, rn, dev, f"{dtype}/{wdtype} axis0")
                rx, rn = callf(XMA.average, np.ma.average, x, n, kw={"weights": w},
                               kwx={"weights": to_dev(w, dev)})
                check(rx, rn, dev, f"{dtype}/{wdtype}")
                assert np.asarray(rx).dtype == np.asarray(rn).dtype

    def test_float64_not_downcast(self, dev):
        x, n = pair("float64", (3, 4), "partial", dev)
        r = XMA.average(x, axis=0)
        assert r.dtype == np.float64
        r = XMA.average(x)
        assert type(r) is np.float64

    def test_keepdims(self, dev):
        for keepdims in (True, False):
            for axis in (None, 0, 1):
                for weighted in (False, True):
                    x, n = pair("float64", (3, 4), "partial", dev)
                    w = np.random.default_rng(1).random((3, 4)) if weighted else None
                    kwn = {"axis": axis, "keepdims": keepdims, "weights": w}
                    kwx = dict(kwn, weights=None if w is None else to_dev(w, dev))
                    rx, rn = callf(XMA.average, np.ma.average, x, n, kw=kwn, kwx=kwx)
                    check(rx, rn, dev, f"keepdims={keepdims} axis={axis} weighted={weighted}")

    def test_edge_shapes(self, dev):
        for shape in ((), (0,), (1,), (5,), (0, 3)):
            for mkind in ("none", "partial", "full"):
                x, n = pair("float64", shape, mkind, dev)
                rx, rn = callf(XMA.average, np.ma.average, x, n)
                check(rx, rn, dev, f"{shape} {mkind}")

    def test_average_weighted_masked_weights(self, dev):
        d = sample("float64", (6,))
        x, n = make(d, np.array([0, 1, 0, 0, 0, 0], bool), dev)
        w = np.arange(1.0, 7.0)
        wm = np.ma.masked_array(w, mask=[0, 0, 1, 0, 0, 0])
        wx = mka(dev, to_dev(w, dev), mask=to_dev(np.array([0, 0, 1, 0, 0, 0], bool), dev))
        try:
            rn = np.ma.average(n, weights=wm)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                XMA.average(x, weights=wx)
            return
        check(XMA.average(x, weights=wx), rn, dev)

    def test_average_zero_weights_sum(self, dev):
        x, n = pair("float64", (4,), "none", dev)
        w = np.array([1.0, -1.0, 2.0, -2.0])
        rx, rn = callf(XMA.average, np.ma.average, x, n, kw={"weights": w, "returned": True},
                       kwx={"weights": to_dev(w, dev), "returned": True})
        check(rx, rn, dev)


# --------------------------------------------------------------------------
# mid-size arrays (1000 x 1000): right answers and device stays
# --------------------------------------------------------------------------
class TestLarge:
    @pytest.mark.parametrize("dtype", ["float64", "int32"])
    def test_big(self, dev, dtype):
        for kind in ("rand", "col"):
            d, m = _big(dtype, kind)
            x, n = make(d, m.copy(), dev)
            for name in ("sum", "mean", "min", "max", "any", "all", "count", "ptp", "std", "var"):
                for axis in (None, 0, 1):
                    rx, rn = call(x, n, name, kw={"axis": axis})
                    check(rx, rn, dev, f"{name} {dtype} {kind} axis={axis}", strict_nomask=sn(name),
                          rtol=BIG_RTOL if dev == "gpu" and name in SUMLIKE else None)
                    if isinstance(rx, XMA.MaskedArray):
                        assert on_dev(rx.data, dev)

    def test_big_scan_and_arg(self, dev):
        d, m = _big("int32", "rand")
        d = np.ones_like(d)      # prod stays 1 (no overflow)
        x, n = make(d, m.copy(), dev)
        for name in ("cumsum", "cumprod"):
            for axis in (0, 1):
                check(*call(x, n, name, kw={"axis": axis}), dev, f"{name} axis={axis}")
        d, m = _big("float64", "rand")
        x, n = make(d, m.copy(), dev)
        for name in ("argmax", "argmin"):
            for axis in (None, 0, 1):
                check(*call(x, n, name, kw={"axis": axis}), dev, f"{name} axis={axis}")

    def test_big_sort_and_compressed(self, dev):
        rng = np.random.default_rng(9)
        d = rng.permutation(1_000_000).astype("float64").reshape(1000, 1000)
        m = rng.random(d.shape) < 0.3
        x, n = make(d, m.copy(), dev)
        x.sort(axis=1)
        n.sort(axis=1)
        assert_same(x, n, dev=dev)
        x, n = make(d, m.copy(), dev)
        check(*call(x, n, "compressed"), dev)
        check(*call(x, n, "nonzero"), dev)
        check(*call(x, n, "count", kw={"axis": 0}), dev)

    def test_big_function_forms(self, dev):
        d, m = _big("float64", "col")
        x, n = make(d, m.copy(), dev)
        for name in ("sum", "mean", "std"):
            check(*callf(getattr(XMA, name), getattr(np.ma, name), x, n, kw={"axis": 0}), dev, name,
                  rtol=BIG_RTOL if dev == "gpu" else None)
        w = np.random.default_rng(2).random(d.shape[1])
        check(*callf(XMA.average, np.ma.average, x, n, kw={"weights": w, "axis": 1},
                     kwx={"weights": to_dev(w, dev), "axis": 1}), dev, rtol=BIG_RTOL if dev == "gpu" else None)
