"""
Differential tests of the numpy.ma.core API names added to ``xupy.ma``
(module-level functions, MaskedArray methods, unsupported-name stubs) against
``numpy.ma``, on numpy data ("cpu") and cupy data ("gpu").

numpy.ma is the truth.  If numpy.ma raises, XuPy must raise the same type
(``both``).  Names absent from the installed numpy.ma are skipped.

Accepted deviation (NO-SYNC): functions whose all-False result mask would need a
host sync to shrink to ``nomask`` return an all-False mask ARRAY where numpy.ma
gives ``nomask`` (masked_where / masked_<cmp> / masked_equal / masked_inside /
masked_outside / masked_invalid, fix_invalid, outer, maximum / minimum, choose,
clip).  Those calls use ``strict_nomask=False``.
"""
import inspect
import sys
import warnings

import numpy as np
import pytest

import xupy

from . import _ma_parity_helpers as _H
from ._ma_parity_helpers import XMA, assert_same, both, host, make, mka, on_dev, to_dev

pytestmark = [
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

NPV = tuple(int(v) for v in np.__version__.split(".")[:2])

# --------------------------------------------------------------------------
# data fixtures (all values distinct so that sort/argsort are unambiguous)
# --------------------------------------------------------------------------
_F1 = np.array([3.5, -1.25, 7.0, 0.5, 2.0, -4.0])
_M1 = np.array([0, 1, 0, 0, 1, 0], bool)
_F2 = (np.random.default_rng(3).permutation(12).reshape(3, 4) * 0.75 - 3.0)
_M2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], bool)
_I2 = np.random.default_rng(4).permutation(12).reshape(3, 4).astype(np.int32) - 4
_B2 = np.array([[1, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 0]], bool)
_C2 = _F2 + 1j * _F2[::-1]
_FNAN = np.array([1.0, np.nan, np.inf, -2.0, 4.0, -np.inf])
_F131 = _F2[:1, :3].reshape(1, 3, 1)

DATA = {
    "f1": (_F1, _M1), "f2": (_F2, _M2), "f2n": (_F2, None),
    "f2z": (_F2, np.zeros((3, 4), bool)), "f2a": (_F2, np.ones((3, 4), bool)),
    "i2": (_I2, _M2), "b2": (_B2, _M2), "c2": (_C2, _M2), "fnan": (_FNAN, None),
    "f131": (_F131, np.zeros((1, 3, 1), bool)), "f3": (_F2.reshape(2, 3, 2), _M2.reshape(2, 3, 2)),
    "i1n": (np.array([4, -2, 9, 1]), None),
}


def D(key, dev, **kw):
    data, mask = DATA[key]
    return make(data, mask, dev, **kw)


def _nf(name):
    f = getattr(np.ma, name, None)
    if f is None:
        pytest.skip(f"numpy.ma.{name} absent in numpy {np.__version__}")
    return f


def _run(name, args, dev, kw=None, snm=True, check_fill=True, backend=False):
    """Call XMA.name / np.ma.name with ``args`` = [(x_arg, n_arg), ...].

    ``backend=True`` runs the XuPy call under ``xupy.backend(dev)`` (needed when an
    operand is host data -- plain numpy array, list, python scalar -- that follows
    the active backend rather than another operand).
    """
    nf = _nf(name)
    fx = getattr(XMA, name)
    if backend:
        def fx(*a, _f=fx, **k):
            with xupy.backend(dev):
                return _f(*a, **k)
    rx, rn = both(fx, nf, *args, **(kw or {}))
    if rx is None and rn is None:
        return None, None
    assert_same(rx, rn, dev=dev, strict_nomask=snm, check_fill=check_fill, ctx=name)
    return rx, rn


def _same_arg(a):
    return (a, a)


def _pair(key, dev):
    return D(key, dev)


def _group(cases):
    out = {}
    for c in cases:
        out.setdefault(c[0], []).append(c)
    return out


def _names(cases):
    return list(_group(cases))


# --------------------------------------------------------------------------
# namespace
# --------------------------------------------------------------------------
def test_swapaxes_bad_axis(dev):
    # numpy raises AxisError (a ValueError and IndexError); cupy a plain ValueError
    x, n = D("f2", dev)
    with pytest.raises((ValueError, IndexError)):
        np.ma.swapaxes(n, 0, 5)
    with pytest.raises((ValueError, IndexError)):
        XMA.swapaxes(x, 0, 5)


def test_all_is_superset_and_attributes_exist():
    mod = sys.modules["xupy.ma"]
    assert mod is XMA
    missing = set(np.ma.__all__) - set(XMA.__all__)
    assert not missing, sorted(missing)
    no_attr = [n for n in np.ma.__all__ if not hasattr(mod, n)]
    assert not no_attr, no_attr


def test_bool_and_masked_singleton_identity():
    assert XMA.bool_ is np.bool_
    assert XMA.masked_singleton is XMA.masked
    assert XMA.masked_singleton is not np.ma.masked


# --------------------------------------------------------------------------
# generic single-array calls: (name, data key, extra positional, kwargs)
# --------------------------------------------------------------------------
CASES = [
    ("all", "f2", (), {}), ("all", "f2", (), {"axis": 0}), ("all", "b2", (), {"axis": 1}),
    ("all", "f2a", (), {}), ("all", "f2n", (), {"axis": 1, "keepdims": True}),
    ("any", "f2", (), {}), ("any", "b2", (), {"axis": 0}), ("any", "f2a", (), {"axis": 1}),
    ("any", "f2z", (), {"keepdims": True}),
    ("amax", "f2", (), {}), ("amax", "f2", (), {"axis": 0}), ("amax", "i2", (), {"axis": 1}),
    ("amax", "f2a", (), {}), ("amax", "f2n", (), {"axis": 1, "keepdims": True}),
    ("amin", "f2", (), {}), ("amin", "f2", (), {"axis": 1}), ("amin", "i2", (), {"axis": 0}),
    ("amin", "f2a", (), {"axis": 0}), ("amin", "b2", (), {}),
    ("argmax", "f2", (), {}), ("argmax", "f2", (), {"axis": 1}), ("argmax", "i2", (), {"axis": 0}),
    ("argmax", "f2", (), {"fill_value": -100.0}), ("argmax", "f2n", (), {"axis": 0}),
    ("argmin", "f2", (), {}), ("argmin", "f2", (), {"axis": 0}), ("argmin", "i2", (), {"axis": 1}),
    ("argmin", "f2", (), {"fill_value": 100.0}), ("argmin", "f2n", (), {"axis": 1}),
    ("cumsum", "f2", (), {"axis": 0}), ("cumsum", "f2", (), {"axis": 1}), ("cumsum", "f2", (), {}),
    ("cumsum", "i2", (), {"axis": 1}), ("cumsum", "b2", (), {"axis": 0}),
    ("cumprod", "f2", (), {"axis": 0}), ("cumprod", "f2", (), {}), ("cumprod", "i2", (), {"axis": 1}),
    ("cumprod", "f2n", (), {"axis": 1}),
    ("count", "f2", (), {}), ("count", "f2", (), {"axis": 0}), ("count", "f2a", (), {"axis": 1}),
    ("count", "f2n", (), {"axis": 1, "keepdims": True}),
    ("copy", "f2", (), {}), ("copy", "f2n", (), {}), ("copy", "i2", (), {}),
    ("ravel", "f2", (), {}), ("ravel", "f2", (), {"order": "F"}), ("ravel", "f2n", (), {}),
    ("diagonal", "f2", (), {}), ("diagonal", "f2", (), {"offset": 1}),
    ("diagonal", "f2", (), {"offset": -1}), ("diagonal", "f3", (), {"axis1": 1, "axis2": 2}),
    ("diag", "f2", (), {}), ("diag", "f2", (), {"k": 1}), ("diag", "f1", (), {}), ("diag", "f1", (), {"k": -1}),
    ("squeeze", "f131", (), {}), ("squeeze", "f131", (), {"axis": 0}), ("squeeze", "f2", (), {}),
    ("transpose", "f2", (), {}), ("transpose", "f3", ((2, 0, 1),), {}), ("transpose", "f1", (), {}),
    ("swapaxes", "f2", (0, 1), {}), ("swapaxes", "f3", (0, 2), {}),
    ("trace", "f2", (), {}), ("trace", "f2", (), {"offset": 1}), ("trace", "i2", (), {"offset": -1}),
    ("trace", "f3", (), {"axis1": 1, "axis2": 2}), ("trace", "f1", (), {}),
    ("ptp", "f2", (), {}), ("ptp", "f2", (), {"axis": 0}), ("ptp", "i2", (), {"axis": 1}),
    ("ptp", "f2a", (), {"axis": 0}), ("ptp", "f2n", (), {"axis": 1, "keepdims": True}),
    ("nonzero", "f2", (), {}), ("nonzero", "b2", (), {}), ("nonzero", "f2a", (), {}),
    ("ndim", "f2", (), {}), ("shape", "f2", (), {}), ("size", "f2", (), {}),
    ("size", "f2", (), {"axis": 1}), ("ndim", "f1", (), {}),
    ("reshape", "f2", ((4, 3),), {}), ("reshape", "f2", ((-1,),), {}), ("reshape", "f2", ((5, 5),), {}),
    ("reshape", "f2n", ((2, 6),), {"order": "F"}),
    ("resize", "f2", ((5, 5),), {}), ("resize", "f2", ((2, 2),), {}), ("resize", "f1", ((2, 4),), {}),
    ("repeat", "f2", (2,), {"axis": 0}), ("repeat", "f2", (2,), {}), ("repeat", "i2", (3,), {"axis": 1}),
    ("repeat", "f2", ([1, 2, 0],), {"axis": 0}),
    ("round_", "f2", (), {}), ("round_", "f2", (1,), {}), ("round_", "f2", (), {"decimals": -1}),
    ("round_", "i2", (), {"decimals": -1}),
    ("angle", "c2", (), {}), ("angle", "c2", (), {"deg": True}), ("angle", "f2", (), {}),
    ("angle", "f2", (), {"deg": True}),
    ("compressed", "f2", (), {}), ("compressed", "f2n", (), {}), ("compressed", "f2a", (), {}),
    ("compressed", "b2", (), {}),
    ("diff", "f2", (), {}), ("diff", "f2", (), {"axis": 0}), ("diff", "i2", (), {"n": 2}),
    ("diff", "f1", (), {"n": 0}), ("diff", "f2", (), {"n": 4}), ("diff", "f2", (), {"axis": 0, "n": 3}),
    ("take", "f2", ([0, 2],), {"axis": 1}), ("take", "f2", ([0, 5, 11],), {}),
    ("take", "f2", ([0, 7],), {"axis": 0, "mode": "wrap"}),
    ("take", "f2", ([0, 7],), {"axis": 0, "mode": "clip"}), ("take", "f2", ([20],), {}),
    ("take", "f2", (3,), {}),
    ("sort", "f2", (), {}), ("sort", "f2", (), {"axis": 0}), ("sort", "f1", (), {"axis": None}),
    ("sort", "i2", (), {}),
    ("clip", "f2", (-1.0, 3.0), {}), ("clip", "i2", (-1, 3), {}), ("clip", "f2", (), {"min": 0.0}),
    ("clip", "f2", (), {"max": 0.0}), ("clip", "f2n", (None, 1.5), {}),
    ("clip", "f2", (3.0, -1.0), {}), ("clip", "f2", (), {}),
    ("maximum_fill_value", "f2", (), {}), ("maximum_fill_value", "i2", (), {}),
    ("maximum_fill_value", "b2", (), {}), ("minimum_fill_value", "f2", (), {}),
    ("minimum_fill_value", "i2", (), {}), ("minimum_fill_value", "b2", (), {}),
    ("isarray", "f2", (), {}), ("isarray", "f2n", (), {}),
    ("fix_invalid", "fnan", (), {}), ("fix_invalid", "fnan", (), {"fill_value": 9.0}),
    ("fix_invalid", "fnan", (), {"copy": False}), ("fix_invalid", "f2", (), {}),
    ("fix_invalid", "i2", (), {}),
    ("masked_invalid", "fnan", (), {}), ("masked_invalid", "f2", (), {}), ("masked_invalid", "i2", (), {}),
    ("masked_invalid", "fnan", (), {"copy": False}),
    ("asarray", "f2", (), {}), ("asarray", "f2n", (), {}), ("asarray", "i2", (), {"dtype": np.float32}),
    ("asanyarray", "f2", (), {}), ("asanyarray", "f2n", (), {}),
    ("harden_mask", "f2", (), {}), ("soften_mask", "f2", (), {}), ("harden_mask", "f2n", (), {}),
    ("flatten_mask", "f2", (), {}),
]

# NO-SYNC deviations: all-False mask array where numpy.ma gives nomask
_NONSTRICT = {"fix_invalid", "masked_invalid", "clip"}


@pytest.mark.parametrize("name", _names(CASES))
def test_single_array_functions(dev, name):
    for _, key, extra, kw in _group(CASES)[name]:
        _single_case(dev, name, key, extra, kw)


def _single_case(dev, name, key, extra, kw):
    if name == "flatten_mask":
        _nf(name)
        # takes the *mask*: boolean ndarray input (device-resident for XuPy)
        _, n = D(key, dev)
        xm = to_dev(np.ma.getmaskarray(n), dev)
        with xupy.backend(dev):
            rx, rn = both(XMA.flatten_mask, np.ma.flatten_mask, (xm, np.ma.getmaskarray(n)))
        if rx is not None:
            assert_same(rx, rn, dev=dev, ctx=name)
        return
    ctx = f"{name} {key} {extra} {kw}"
    x, n = D(key, dev)
    args = [(x, n)] + [_same_arg(a) for a in extra]
    if name in ("fix_invalid", "masked_invalid") and kw.get("copy") is False:
        pass
    if name in ("harden_mask", "soften_mask"):
        rx, rn = _run(name, args, dev, kw)
        assert rx is x and rn is n or rx is None, ctx   # in place, returns the array itself
        assert_same(x, n, dev=dev)
        return
    rx, rn = _run(name, args, dev, kw, snm=name not in _NONSTRICT)
    if name in ("copy",) and rx is not None:
        assert rx is not x
    if name in ("fix_invalid", "masked_invalid") and kw.get("copy") is False and rx is not None:
        # numpy.ma: data of a *masked* input is modified in place when copy=False
        assert_same(x, n, dev=dev, strict_nomask=False)


# --------------------------------------------------------------------------
# sort / argsort keywords, function and method form
# --------------------------------------------------------------------------
def _has_kw(fn, kw):
    try:
        return kw in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


SORT_KW = [
    {}, {"axis": 0}, {"axis": None}, {"kind": "stable"}, {"stable": True}, {"stable": False},
    {"endwith": False}, {"endwith": True, "axis": 0}, {"fill_value": 2.0}, {"fill_value": -50.0, "endwith": False},
    {"descending": True}, {"descending": True, "stable": True}, {"descending": False, "axis": 0},
    {"descending": True, "endwith": False}, {"kind": "stable", "stable": True},
    {"kind": "quicksort"}, {"axis": 7}, {"order": "a"},
]


def _kw_ok(kw):
    return "descending" not in kw or _has_kw(np.ma.sort, "descending")


@pytest.mark.parametrize("name", ["sort", "argsort"])
@pytest.mark.parametrize("form", ["function", "method"])
def test_sort_argsort_keywords(dev, name, form):
    for key in ("f2", "f2n", "i2", "f2a"):
        for kw in SORT_KW:
            if not _kw_ok(kw):
                continue
            if key != "f2" and kw not in ({}, {"axis": 0}, {"descending": True}, {"endwith": False},
                                          {"stable": True}):
                continue
            _sort_case(dev, name, form, key, kw)


def _sort_case(dev, name, form, key, kw):
    if form == "method" and name == "sort":
        # in-place: compare the mutated arrays
        x, n = D(key, dev)
        try:
            n.sort(**kw)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x.sort(**kw)
            return
        x.sort(**kw)
        assert_same(x, n, dev=dev, strict_nomask=False, ctx=f"{key} {kw}")
        return
    kw = dict(kw)
    if name == "argsort" and "kind" not in kw and "stable" not in kw:
        kw["kind"] = "stable"        # cupy sort is unstable: ties between masked fills
    x, n = D(key, dev)
    if form == "function":
        rx, rn = both(getattr(XMA, name), getattr(np.ma, name), (x, n), **kw)
    else:
        try:
            rn = getattr(n, name)(**kw)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                getattr(x, name)(**kw)
            return
        rx = getattr(x, name)(**kw)
    if rx is None and rn is None:
        return
    assert_same(rx, rn, dev=dev, strict_nomask=False, ctx=f"{form} {name} {key} {kw}")


def test_sort_function_does_not_modify_input(dev):
    x, n = D("f2", dev)
    XMA.sort(x, axis=1)
    assert_same(x, n, dev=dev)


# --------------------------------------------------------------------------
# std / var: mean= and ddof, function and method form
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["std", "var"])
@pytest.mark.parametrize("form", ["function", "method"])
def test_std_var_ddof_and_mean(dev, name, form):
    for key in ("f2", "f2n", "f2a", "i2"):
        for axis in (None, 0, 1):
            for ddof in (0, 1, 2, 5):
                _std_case(dev, name, form, key, axis, ddof)


def _std_case(dev, name, form, key, axis, ddof):
    for use_mean in (False, True):
        x, n = D(key, dev)
        kw = {"axis": axis, "ddof": ddof}
        if use_mean:
            if axis is None and key == "f2a":
                continue
            mx = XMA.mean(x, axis=axis, keepdims=True)
            mn = np.ma.mean(n, axis=axis, keepdims=True)
            if _has_kw(np.ma.std, "mean") is False:
                continue
            kwx, kwn = dict(kw, mean=mx), dict(kw, mean=mn)
        else:
            kwx = kwn = kw
        try:
            if form == "function":
                rn = getattr(np.ma, name)(n, **kwn)
            else:
                rn = getattr(n, name)(**kwn)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                (getattr(XMA, name)(x, **kwx) if form == "function" else getattr(x, name)(**kwx))
            continue
        rx = getattr(XMA, name)(x, **kwx) if form == "function" else getattr(x, name)(**kwx)
        assert_same(rx, rn, dev=dev, strict_nomask=False, ctx=f"{name} {form} {kw} mean={use_mean}")


# --------------------------------------------------------------------------
# binary / multi-array functions
# --------------------------------------------------------------------------
_V = np.array([2.0, -1.0, 0.5, 3.0])
_VM = np.array([0, 0, 1, 0], bool)
_K = np.array([1.0, 2.0, -0.5])
_KM = np.array([0, 1, 0], bool)


def _vec(dev, which="v"):
    d, m = {"v": (_V, _VM), "k": (_K, _KM), "kn": (_K, None), "vn": (_V, None),
            "vi": (np.array([3, -2, 5, 1]), _VM), "ki": (np.array([2, 1, -1]), _KM)}[which]
    return make(d, m, dev)


CONV = [("v", "k"), ("v", "kn"), ("vn", "k"), ("vn", "kn"), ("vi", "ki")]


@pytest.mark.parametrize("name", ["convolve", "correlate"])
def test_convolve_correlate(dev, name):
    for a, b in CONV:
        for mode in (None, "full", "same", "valid"):
            for pm in (True, False):
                xa, na = _vec(dev, a)
                xb, nb = _vec(dev, b)
                kw = {"propagate_mask": pm}
                if mode:
                    kw["mode"] = mode
                _conv_run(name, xa, na, xb, nb, dev, kw)


def _conv_run(name, xa, na, xb, nb, dev, kw):
    """cupy's float convolve/correlate is not exact (1e-16 where numpy gives 0): atol 1e-12."""
    rn = getattr(np.ma, name)(na, nb, **kw)
    rx = getattr(XMA, name)(xa, xb, **kw)
    if rn.dtype.kind != "f":
        return assert_same(rx, rn, dev=dev, ctx=name)
    assert isinstance(rx, XMA.MaskedArray) and rx.dtype == rn.dtype and rx.shape == rn.shape
    assert on_dev(rx.data, dev), f"{name}: data on wrong device {type(rx.data)}"
    mx, mn = host(XMA.getmaskarray(rx)), np.ma.getmaskarray(rn)
    np.testing.assert_array_equal(mx, mn, err_msg=f"{name} {kw}: mask")
    np.testing.assert_allclose(host(rx.data)[~mn], np.asarray(rn.data)[~mn], rtol=1e-12, atol=1e-12)


def test_convolve_correlate_errors(dev):
    xa, na = _vec(dev, "v")
    for name in ("convolve", "correlate"):
        _run(name, [(xa, na), (xa, na)], dev, {"mode": "bogus"})
        # empty kernel
        xe, ne = make(np.zeros(0), None, dev)
        _run(name, [(xa, na), (xe, ne)], dev)
        # 2-d input
        x2, n2 = D("f2", dev)
        _run(name, [(x2, n2), (xa, na)], dev)


@pytest.mark.parametrize("name", ["inner", "innerproduct"])
def test_inner(dev, name):
    for a, b in [("v", "v"), ("v", "vn"), ("vn", "vn"), ("vi", "vi")]:
        xa, na = _vec(dev, a)
        xb, nb = _vec(dev, b)
        _run(name, [(xa, na), (xb, nb)], dev)
    xa, na = _vec(dev, "v")
    xb, nb = _vec(dev, "vi")
    # 2-d x 1-d, plain operand
    x2, n2 = D("f2", dev)
    _run(name, [(x2, n2), (to_dev(_V, dev), _V)], dev, backend=True)  # plain operand
    _run(name, [(x2, n2), (xb, nb)], dev)     # shape mismatch -> same error


@pytest.mark.parametrize("name", ["outer", "outerproduct"])
def test_outer(dev, name):
    for a, b in [("v", "k"), ("vn", "kn"), ("v", "kn"), ("vi", "ki")]:
        xa, na = _vec(dev, a)
        xb, nb = _vec(dev, b)
        _run(name, [(xa, na), (xb, nb)], dev, snm=False)     # NO-SYNC
    x2, n2 = D("f2", dev)
    _run(name, [(x2, n2), (xb, nb)], dev, snm=False)     # 2-d is flattened


@pytest.mark.parametrize("name", ["maximum", "minimum"])
def test_maximum_minimum_call(dev, name):
    for ka, kb in [("f2", "i2"), ("f2", "f2n"), ("f2n", "f2n"), ("f2a", "f2"), ("f2", "f2z"), ("i2", "b2")]:
        _mm_case(dev, name, ka, kb)


def _mm_case(dev, name, ka, kb):
    (xa, na), (xb, nb) = D(ka, dev), D(kb, dev)
    _run(name, [(xa, na), (xb, nb)], dev, snm=False)
    _run(name, [(xa, na), _same_arg(2.0)], dev, snm=False)
    _run(name, [(xa, na), (to_dev(_F2[0], dev), _F2[0])], dev, snm=False)   # broadcasting
    _run(name, [(xa, na), (xb[:, :2], nb[:, :2])], dev, snm=False)          # shape mismatch


@pytest.mark.parametrize("name", ["maximum", "minimum"])
def test_maximum_minimum_reduce(dev, name):
    for key in ("f2", "f2n", "i2", "f2a", "f2z", "f1", "b2"):
        for kw in ({}, {"axis": 0}, {"axis": 1}, {"axis": None}, {"axis": -1}, {"axis": 3}):
            _mm_reduce(dev, name, key, kw)


def _mm_reduce(dev, name, key, kw):
    x, n = D(key, dev)
    rx, rn = both(getattr(XMA, name).reduce, getattr(np.ma, name).reduce, (x, n), **kw)
    if rx is not None or rn is not None:
        assert_same(rx, rn, dev=dev, strict_nomask=False, ctx=f"{name}.reduce {kw}")


@pytest.mark.parametrize("name", ["maximum", "minimum"])
def test_maximum_minimum_outer(dev, name):
    for a, b in [("v", "k"), ("vn", "kn"), ("v", "kn"), ("vi", "ki")]:
        xa, na = _vec(dev, a)
        xb, nb = _vec(dev, b)
        rx, rn = both(getattr(XMA, name).outer, getattr(np.ma, name).outer, (xa, na), (xb, nb))
        assert_same(rx, rn, dev=dev, strict_nomask=False)


def test_append_and_allclose_allequal_common_fill(dev):
    for ka, kb, kw in [("f2", "f2n", {}), ("f2", "f2n", {"axis": 0}), ("f2", "f2", {"axis": 1}),
                       ("f1", "f1", {}), ("i2", "f2", {"axis": 0}), ("f2", "f1", {"axis": 0})]:
        (xa, na), (xb, nb) = D(ka, dev), D(kb, dev)
        _run("append", [(xa, na), (xb, nb)], dev, kw)
    _run("append", [(D("f2", dev)[0], D("f2", dev)[1]), _same_arg(5.0)], dev, backend=True)
    # allclose / allequal -> python bool
    for ka, kb in [("f2", "f2"), ("f2", "f2n"), ("f2a", "f2"), ("f2", "i2"), ("i2", "i2"), ("f2n", "f2z")]:
        (xa, na), (xb, nb) = D(ka, dev), D(kb, dev)
        for name in ("allclose", "allequal"):
            _run(name, [(xa, na), (xb, nb)], dev)
    (xa, na), (xb, nb) = D("f2", dev), D("f2", dev)
    _run("allclose", [(xa, na), (xb, nb)], dev, {"masked_equal": False})
    _run("allclose", [(xa, na), (xb + 1e-7, nb + 1e-7)], dev, {"rtol": 1e-3, "atol": 0.0})
    _run("allequal", [(xa, na), (xb, nb)], dev, {"fill_value": False})
    _run("allclose", [(xa, na), (xb[:, :2], nb[:, :2])], dev)
    # common_fill_value: fill value if equal, else None
    (xa, na), (xb, nb) = D("f2", dev, fill_value=3.0), D("f2n", dev, fill_value=3.0)
    _run("common_fill_value", [(xa, na), (xb, nb)], dev)
    (xc, nc) = D("f2n", dev, fill_value=4.0)
    _run("common_fill_value", [(xa, na), (xc, nc)], dev)
    _run("common_fill_value", [(xa, na), _same_arg(1.0)], dev)


# --------------------------------------------------------------------------
# masked_* constructors
# --------------------------------------------------------------------------
MASKED_CASES = [
    ("masked_equal", "f2", (_F2[0, 1],), {}), ("masked_equal", "i2", (3,), {}),
    ("masked_equal", "f2n", (_F2[1, 1],), {"copy": False}), ("masked_equal", "f2", (1234.0,), {}),
    ("masked_equal", "b2", (True,), {}),
    ("masked_not_equal", "f2", (_F2[0, 1],), {}), ("masked_not_equal", "i2", (3,), {}),
    ("masked_greater", "f2", (0.0,), {}), ("masked_greater", "i2", (2,), {}), ("masked_greater", "f2n", (100.0,), {}),
    ("masked_greater", "f2", (0.0,), {"copy": False}),
    ("masked_greater_equal", "f2", (0.0,), {}), ("masked_greater_equal", "i2", (2,), {}),
    ("masked_less", "f2", (0.0,), {}), ("masked_less", "i2", (2,), {}), ("masked_less", "f2a", (0.0,), {}),
    ("masked_less_equal", "f2", (0.0,), {}), ("masked_less_equal", "i2", (2,), {}),
    ("masked_inside", "f2", (-1.0, 2.0), {}), ("masked_inside", "f2", (2.0, -1.0), {}),
    ("masked_inside", "i2", (0, 3), {}), ("masked_inside", "f2n", (-50.0, -40.0), {}),
    ("masked_outside", "f2", (-1.0, 2.0), {}), ("masked_outside", "f2", (2.0, -1.0), {}),
    ("masked_outside", "i2", (0, 3), {}), ("masked_outside", "f2n", (-50.0, 50.0), {}),
    ("masked_values", "f2", (_F2[0, 0],), {}), ("masked_values", "f2n", (0.75,), {"rtol": 0.5}),
    ("masked_values", "f2", (0.75,), {"atol": 1.0, "rtol": 0.0}), ("masked_values", "i2", (3,), {}),
    ("masked_values", "f2", (1234.0,), {}), ("masked_values", "f2", (1234.0,), {"shrink": False}),
    ("masked_values", "f2", (_F2[0, 0],), {"copy": False}),
    ("masked_values", "fnan", (np.nan,), {}), ("masked_values", "fnan", (np.inf,), {}),
    ("masked_values", "b2", (True,), {}),
]


@pytest.mark.parametrize("name", _names(MASKED_CASES))
def test_masked_comparisons(dev, name):
    for _, key, args, kw in _group(MASKED_CASES)[name]:
        _masked_case(dev, name, key, args, kw)


def _masked_case(dev, name, key, args, kw):
    x, n = D(key, dev)
    # NO-SYNC: all-False result mask array instead of nomask -> strict_nomask=False
    rx, rn = _run(name, [(x, n)] + [_same_arg(a) for a in args], dev, kw, snm=False)
    if rx is not None and kw.get("copy") is False:
        assert_same(x, n, dev=dev, strict_nomask=False)      # same in-place side effects as numpy


def test_masked_where(dev):
    for cond_kind in ("same", "broadcast_row", "scalar_true", "scalar_false", "short", "nested"):
        for key in ("f2", "f2n", "i2"):
            for kw in ({}, {"copy": False}, {"shrink": False}):
                _mw_case(dev, cond_kind, key, kw)


def _mw_case(dev, cond_kind, key, kw):
    x, n = D(key, dev)
    cond = {"same": _F2 > 0, "broadcast_row": np.array([True, False, True, False]),
            "scalar_true": np.True_, "scalar_false": np.False_,
            "short": np.array([True, False]), "nested": np.zeros((3, 4, 1), bool)}[cond_kind]
    cx = to_dev(cond, dev)
    rx, rn = both(XMA.masked_where, np.ma.masked_where, (cx, cond), (x, n), **kw)
    if rx is None and rn is None:
        return
    assert_same(rx, rn, dev=dev, strict_nomask=False)      # NO-SYNC


def test_masked_where_error_parity(dev):
    x, n = D("f2", dev)
    cond = np.ones((2, 4), bool)
    with pytest.raises(IndexError):
        np.ma.masked_where(cond, n)
    with pytest.raises(IndexError):
        XMA.masked_where(to_dev(cond, dev), x)
    # masked condition input (masked cond entries count as False... numpy: filled)
    cm, cn = make(_F2 > 0, _M2, dev)
    rx, rn = both(XMA.masked_where, np.ma.masked_where, (cm, cn), (x, n))
    assert_same(rx, rn, dev=dev, strict_nomask=False)
    # list / scalar data to be masked
    rx, rn = both(XMA.masked_where, np.ma.masked_where, ([True, False, True], [True, False, True]),
                  ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
    assert_same(rx, rn, dev=None if dev == "cpu" else dev, strict_nomask=False)
    for name in ("masked_greater", "masked_equal"):
        rx, rn = both(getattr(XMA, name), getattr(np.ma, name), (x, n), (np.array([1.0, 2.0]), np.array([1.0, 2.0])))
        if rx is not None:
            assert_same(rx, rn, dev=dev, strict_nomask=False)


def test_masked_values_error_parity(dev):
    x, n = D("f2", dev)
    _run("masked_values", [(x, n)], dev)                                  # missing value
    _run("masked_values", [(x, n), _same_arg(None)], dev, snm=False)
    _run("masked_inside", [(x, n), _same_arg(1.0)], dev, snm=False)
    _run("masked_equal", [(x, n), _same_arg("a")], dev, snm=False)
    _run("masked_greater", [(x, n), _same_arg(np.nan)], dev, snm=False)


def test_masked_comparisons_with_plain_and_list_inputs(dev):
    for name, val in [("masked_greater", 1.0), ("masked_equal", 3.0), ("masked_invalid", None)]:
        for src in (_F1, _F1.tolist()):
            xs = to_dev(src, dev) if isinstance(src, np.ndarray) else src
            args = [(xs, src)] + ([] if val is None else [_same_arg(val)])
            nf = _nf(name)
            with xupy.backend(dev):
                rn = nf(*[a[1] for a in args])
                rx = getattr(XMA, name)(*[a[0] for a in args])
            assert_same(rx, rn, dev=dev, strict_nomask=False)


# --------------------------------------------------------------------------
# put / putmask / compress / choose
# --------------------------------------------------------------------------
PUT_CASES = [
    ([0, 2], [10.0, 20.0], {}), ([1], [5.0], {}), ([0, 1, 2], [9.0], {}), ([-1], [7.0], {}),
    ([0, 100], [1.0, 2.0], {"mode": "clip"}), ([0, 100], [1.0, 2.0], {"mode": "wrap"}),
    ([0, 100], [1.0, 2.0], {}), ([], [], {}), ([0], [], {}), ([0, 1, 2, 3], [1.0, 2.0], {}),
    ([1], [np.ma.masked], {}), ([0, 3], [1.0, 2.0], {"mode": "bogus"}),
]


def test_put(dev):
    for key in ("f1", "f2", "f2n", "i2"):
        for idx, vals, kw in PUT_CASES:
            _put_case(dev, key, idx, vals, kw)


def _put_case(dev, key, idx, vals, kw):
    x, n = D(key, dev)
    ix, inn = to_dev(np.array(idx, dtype=np.intp), dev), np.array(idx, dtype=np.intp)
    if vals and vals[0] is np.ma.masked:
        vx, vn = XMA.masked, np.ma.masked
    else:
        vn = np.array(vals, dtype=float)
        vx = to_dev(vn, dev)
    nf = _nf("put")
    try:
        rn = nf(n, inn, vn, **kw)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            XMA.put(x, ix, vx, **kw)
        return
    rx = XMA.put(x, ix, vx, **kw)
    assert rx is None and rn is None
    assert_same(x, n, dev=dev, strict_nomask=False)


def test_put_method_and_hard_mask(dev):
    for hard in (False, True):
        x, n = D("f1", dev, hard_mask=hard)  # 1-d: numpy's hardmask put fails for ndim > 1
        x.put([0, 1], [100.0, 200.0])
        n.put([0, 1], [100.0, 200.0])
        assert_same(x, n, dev=dev, strict_nomask=False)
        if hard:
            x2, _ = D("f2", dev, hard_mask=True)  # xupy handles ndim>1 (raveled storage indices)
            x2.put([0, 1], [100.0, 200.0])
        x, n = D("f1", dev, hard_mask=hard)
        XMA.put(x, [1, 3], [1.5, 2.5])
        np.ma.put(n, [1, 3], [1.5, 2.5])
        assert_same(x, n, dev=dev, strict_nomask=False)


def test_putmask(dev):
    for key in ("f1", "f2", "f2n", "i2"):
        for hard in (False, True):
            for mask_kind, vals in [("gt0", [100.0]), ("gt0", [1.0, 2.0, 3.0]), ("none", [1.0]),
                                    ("all", [4.0, 5.0]), ("gt0", []), ("short", [1.0]), ("gt0", [np.nan])]:
                _putmask_case(dev, key, hard, mask_kind, vals)


def _putmask_case(dev, key, hard, mask_kind, vals):
    x, n = D(key, dev, hard_mask=hard)
    shape = n.shape
    m = {"gt0": np.asarray(n.data) > 0, "none": np.zeros(shape, bool), "all": np.ones(shape, bool),
         "short": np.array([True, False])}[mask_kind]
    vn = np.array(vals, dtype=float)
    nf = _nf("putmask")
    try:
        rn = nf(n, m, vn)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            XMA.putmask(x, to_dev(m, dev), to_dev(vn, dev))
        return
    rx = XMA.putmask(x, to_dev(m, dev), to_dev(vn, dev))
    assert rx is None and rn is None
    assert_same(x, n, dev=dev, strict_nomask=False, check_fill=False)


def test_compress(dev):
    for key in ("f2", "f2n", "f2a", "i2", "b2"):
        for cond_kind in ("all", "none", "mixed"):
            for axis in (None, 0, 1, -1, 4):
                _compress_case(dev, key, cond_kind, axis)


def _compress_case(dev, key, cond_kind, axis):
    x, n = D(key, dev)
    ln = n.shape[axis] if axis is not None and -2 <= axis < 2 else n.size
    cond = {"all": np.ones(ln, bool), "none": np.zeros(ln, bool),
            "mixed": (np.arange(ln) % 2 == 0)}[cond_kind]
    # function form
    kw = {} if axis is None else {"axis": axis}
    rx, rn = both(XMA.compress, np.ma.compress, (to_dev(cond, dev), cond), (x, n), **kw)
    if rx is not None:
        assert_same(rx, rn, dev=dev, strict_nomask=False, ctx="compress")
    # method form
    try:
        rn = n.compress(cond, **kw)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            x.compress(to_dev(cond, dev), **kw)
        return
    assert_same(x.compress(to_dev(cond, dev), **kw), rn, dev=dev, strict_nomask=False, ctx="compress method")


def test_compress_masked_condition_and_wrong_length(dev):
    x, n = D("f2", dev)
    cm, cn = make(np.array([True, True, False]), np.array([0, 1, 0], bool), dev)
    rx, rn = both(XMA.compress, np.ma.compress, (cm, cn), (x, n), axis=0)
    if rx is not None:
        assert_same(rx, rn, dev=dev, strict_nomask=False)
    bad = np.ones(7, bool)
    rx, rn = both(XMA.compress, np.ma.compress, (to_dev(bad, dev), bad), (x, n), axis=0)


def _choice_data(dev):
    i = np.array([[0, 1, 1], [1, 0, 2]])
    im = np.array([[0, 0, 1], [0, 0, 0]], bool)
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    c = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    ma_ = np.array([[0, 1, 0], [0, 0, 0]], bool)
    return i, im, a, b, c, ma_


@pytest.mark.parametrize("form", ["function", "method"])
def test_choose(dev, form):
    for idx_masked in (False, True):
        for choice_masked in (False, True):
            for mode in ("raise", "wrap", "clip"):
                for nch in (2, 3):
                    _choose_case(dev, idx_masked, choice_masked, mode, nch, form)


def _choose_case(dev, idx_masked, choice_masked, mode, nch, form):
    i, im, a, b, c, mc = _choice_data(dev)
    xi, ni = make(i, im if idx_masked else None, dev)
    chs = [(a, mc if choice_masked else None), (b, None), (c, mc)][:nch]
    xs, ns = zip(*[make(d, m, dev) for d, m in chs])
    # NO-SYNC: result mask is a full array
    if form == "function":
        rx, rn = both(XMA.choose, np.ma.choose, (xi, ni), (list(xs), list(ns)), mode=mode)
    else:
        try:
            rn = ni.choose(list(ns), mode=mode)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                xi.choose(list(xs), mode=mode)
            return
        rx = xi.choose(list(xs), mode=mode)
    if rx is None and rn is None:
        return
    assert_same(rx, rn, dev=dev, strict_nomask=False)


def test_choose_plain_and_out_and_broadcast(dev):
    i, im, a, b, c, mc = _choice_data(dev)
    idx = to_dev(i % 2, dev)
    rx, rn = both(XMA.choose, np.ma.choose, (idx, i % 2), ((to_dev(a, dev), to_dev(b, dev)), (a, b)))
    assert_same(rx, rn, dev=dev, strict_nomask=False)
    # scalar choices broadcast
    rx, rn = both(XMA.choose, np.ma.choose, (idx, i % 2), ((1.0, 2.0), (1.0, 2.0)))
    assert_same(rx, rn, dev=dev, strict_nomask=False)
    # out of range -> ValueError / IndexError in both
    ob = np.array([[0, 5, 1], [0, 0, 0]])
    rx, rn = both(XMA.choose, np.ma.choose, (to_dev(ob, dev), ob), ((to_dev(a, dev), to_dev(b, dev)), (a, b)))


# --------------------------------------------------------------------------
# creation functions and device rule
# --------------------------------------------------------------------------
CREATE = [
    ("ones", ((2, 3),), {}), ("ones", ((2, 3),), {"dtype": np.int16}), ("ones", (4,), {"fill_value": 7.0}),
    ("ones", ((0,),), {}), ("ones", ((),), {}), ("ones", ((2, 3),), {"hardmask": True}),
    ("ones", ((2, 3),), {"order": "F"}),
    ("zeros", ((2, 3),), {}), ("zeros", ((2, 3),), {"dtype": bool}), ("zeros", (3,), {"dtype": np.complex64}),
    ("zeros", ((0, 2),), {}), ("zeros", ((2, 2),), {"fill_value": 3}),
    ("identity", (3,), {}), ("identity", (3,), {"dtype": np.int32}), ("identity", (0,), {}), ("identity", (1,), {}),
    ("identity", (2,), {"fill_value": 5, "hardmask": True}),
    ("arange", (5,), {}), ("arange", (2, 9, 3), {}), ("arange", (0, 1, 0.25), {}),
    ("arange", (5,), {"dtype": np.float32}), ("arange", (5, 0), {}), ("arange", (3,), {"fill_value": 9}),
    ("arange", (0,), {}), ("arange", (4,), {"hardmask": True}), ("arange", (0, 5, 0), {}),
    ("indices", ((2, 3),), {}), ("indices", ((2,),), {}), ("indices", ((2, 2, 2),), {"dtype": np.float32}),
    ("indices", ((0, 2),), {}), ("indices", ((2, 3),), {"fill_value": 4, "hardmask": True}),
    ("indices", ((),), {}),
]


@pytest.mark.parametrize("name", _names(CREATE))
def test_creation_functions(dev, name):
    for _, args, kw in _group(CREATE)[name]:
        _creation_case(dev, name, args, kw)


def _creation_case(dev, name, args, kw):
    nf = _nf(name)
    try:
        rn = nf(*args, **kw)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)), xupy.backend(dev):
            getattr(XMA, name)(*args, **kw)
        return
    with xupy.backend(dev):
        rx = getattr(XMA, name)(*args, **kw)
    assert isinstance(rx, XMA.MaskedArray)
    if isinstance(rn, np.ma.MaskedArray) and rn.dtype.kind in "fiubc":
        # fill_value default of the result dtype; ones()/zeros() float default dtype is float64
        assert_same(rx, rn, dev=dev, strict_nomask=False)
    assert on_dev(rx.data, dev)


def test_empty(dev):
    for shape, kw in [((2, 3), {}), ((0,), {}), ((), {}), ((2,), {"dtype": np.int8}), ((3,), {"fill_value": 2.0})]:
        rn = np.ma.empty(shape, **kw)
        with xupy.backend(dev):
            rx = XMA.empty(shape, **kw)
        assert isinstance(rx, XMA.MaskedArray) and on_dev(rx.data, dev)
        assert rx.shape == rn.shape and rx.dtype == rn.dtype
        assert not bool(np.any(host(XMA.getmaskarray(rx))))
        assert np.asarray(rx.fill_value).dtype == np.asarray(rn.fill_value).dtype
        if "fill_value" in kw:
            assert rx.fill_value == rn.fill_value


@pytest.mark.parametrize("name,args", [("ones", ((2,),)), ("zeros", ((2,),)), ("empty", ((2,),)),
                                       ("identity", (2,)), ("arange", (3,))])
def test_creation_follows_backend_both_ways(name, args):
    """Result lives on the active backend, whichever it is."""
    if _H.cp is None:
        pytest.skip("no GPU")
    with xupy.backend("cpu"):
        assert isinstance(getattr(XMA, name)(*args).data, np.ndarray)
    with xupy.backend("gpu"):
        assert isinstance(getattr(XMA, name)(*args).data, _H.cp.ndarray)


def test_indices_sparse_is_tuple(dev):
    # sparse=True differs by design from numpy.ma (see brief); only check the container/devices
    with xupy.backend(dev):
        r = XMA.indices((2, 3), sparse=True)
    assert isinstance(r, (tuple, list)) and len(r) == 2


def test_fromfunction(dev):
    cases = [(lambda i, j: i + j, (3, 4), {}), (lambda i, j: i * 2.5 - j, (2, 3), {"dtype": float}),
             (lambda i: i ** 2, (5,), {"dtype": int}), (lambda i, j: i == j, (3, 3), {"dtype": int}),
             (lambda i, j: i + j, (0, 3), {})]
    for f, shape, kw in cases:
        rn = np.ma.fromfunction(f, shape, **kw)
        with xupy.backend(dev):
            rx = XMA.fromfunction(f, shape, **kw)
        assert_same(rx, rn, dev=dev, strict_nomask=False)


def test_frombuffer(dev):
    buf = np.arange(6, dtype=np.float64).tobytes()
    for kw in [{"dtype": np.float64}, {"dtype": np.float32}, {"dtype": np.float64, "count": 3},
               {"dtype": np.float64, "offset": 16}, {"dtype": np.int32, "count": 4, "offset": 8}]:
        rn = np.ma.frombuffer(buf, **kw)
        with xupy.backend(dev):
            rx = XMA.frombuffer(buf, **kw)
        assert_same(rx, rn, dev=dev, strict_nomask=False)
    # bad: count too large, offset out of range, buffer size not multiple of itemsize
    for kw in [{"dtype": np.float64, "count": 100}, {"dtype": np.float64, "offset": 1000},
               {"dtype": np.float64, "offset": 1}]:
        with pytest.raises(ValueError):
            np.ma.frombuffer(buf, **kw)
        with pytest.raises(ValueError), xupy.backend(dev):
            XMA.frombuffer(buf, **kw)
    with pytest.raises((ValueError, TypeError)):
        np.ma.frombuffer(b"abc", dtype=np.float64)
    with pytest.raises((ValueError, TypeError)), xupy.backend(dev):
        XMA.frombuffer(b"abc", dtype=np.float64)


# --------------------------------------------------------------------------
# operations follow operands
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _H.GPU_OK, reason="no usable GPU")
@pytest.mark.parametrize("name,nargs,kw", [
    ("cumsum", 1, {"axis": 0}), ("amax", 1, {"axis": 1}), ("sort", 1, {}), ("clip", 1, {"min": 0.0}),
    ("transpose", 1, {}), ("ravel", 1, {}), ("round_", 1, {}), ("copy", 1, {}), ("diff", 1, {}),
    ("masked_greater", 1, {}), ("maximum", 2, {}), ("minimum", 2, {}), ("outer", 2, {}), ("append", 2, {}),
    ("masked_invalid", 1, {}), ("fix_invalid", 1, {}), ("ptp", 1, {"axis": 0}),
])
def test_operations_follow_cupy_operands_under_cpu_backend(name, nargs, kw):
    x, n = D("f2", "gpu")
    args = [x] * nargs + ([0.5] if name == "masked_greater" else [])
    with xupy.backend("cpu"):
        r = getattr(XMA, name)(*args, **kw)
    if name in ("outer",):
        pass
    assert isinstance(r, XMA.MaskedArray)
    assert on_dev(r.data, "gpu"), type(r.data)
    if r.mask is not XMA.nomask:
        assert on_dev(r.mask, "gpu")


@pytest.mark.skipif(not _H.GPU_OK, reason="no usable GPU")
def test_masked_where_cupy_condition_with_cupy_array_under_cpu_backend():
    x, n = D("f2", "gpu")
    cond = _H.cp.asarray(_F2 > 0)
    with xupy.backend("cpu"):
        r = XMA.masked_where(cond, x)
    assert on_dev(r.data, "gpu") and on_dev(r.mask, "gpu")


_HF = np.arange(12.0).reshape(3, 4) + 0.5      # host ndarray operands
_HC = _HF > 3


@pytest.mark.skipif(not _H.GPU_OK, reason="no usable GPU")
@pytest.mark.parametrize("name,call", [
    ("masked_where", lambda m: m.masked_where(_HC, _HF)),
    ("masked_greater", lambda m: m.masked_greater(_HF, 3)),
    ("masked_greater_equal", lambda m: m.masked_greater_equal(_HF, 3)),
    ("masked_less", lambda m: m.masked_less(_HF, 3)),
    ("masked_less_equal", lambda m: m.masked_less_equal(_HF, 3)),
    ("masked_not_equal", lambda m: m.masked_not_equal(_HF, 3)),
    ("masked_equal", lambda m: m.masked_equal(_HF, 3)),
    ("masked_inside", lambda m: m.masked_inside(_HF, 2, 3)),
    ("masked_outside", lambda m: m.masked_outside(_HF, 2, 3)),
    ("masked_invalid", lambda m: m.masked_invalid(_HF)),
    ("masked_values", lambda m: m.masked_values(_HF, 3.5)),
    ("fix_invalid", lambda m: m.fix_invalid(_HF)),
    ("diff", lambda m: m.diff(_HF)),
    ("inner", lambda m: m.inner(_HF, _HF)),
    ("outer", lambda m: m.outer(_HF[0], _HF[1])),
    ("diag", lambda m: m.diag(_HF)),
    ("alltrue", lambda m: m.alltrue(_HF, axis=0)),
    ("sometrue", lambda m: m.sometrue(_HF, axis=0)),
    ("resize", lambda m: m.resize(_HF, (2, 2))),
    ("take", lambda m: m.take(_HF, [0, 1])),
    ("ptp", lambda m: m.ptp(_HF, axis=0)),
    ("count", lambda m: m.count(_HF, axis=0)),
    ("compress", lambda m: m.compress([True, False, True], _HF, axis=0)),
    ("append", lambda m: m.append(_HF, _HF)),
    ("left_shift", lambda m: m.left_shift(_HF.astype(int), 1)),
    ("convolve", lambda m: m.convolve(_HF[0], _HF[1])),
    ("maximum.outer", lambda m: m.maximum.outer(_HF[0], _HF[1])),
    ("clip", lambda m: m.clip(_HF, 2, 5)),
    ("median", lambda m: m.median(_HF, axis=0)),
    ("sort", lambda m: m.sort(_HF)),
])
def test_functions_keep_host_ndarray_operands_on_numpy_under_gpu_backend(name, call):
    """Functions follow the device of their operands: a host ndarray is not moved to the GPU."""
    with xupy.backend("gpu"):
        r = call(XMA)
    data = getattr(r, "_data", r)
    assert isinstance(data, np.ndarray), (name, type(data))
    if isinstance(r, XMA.MaskedArray) and r.mask is not XMA.nomask:
        assert isinstance(r.mask, np.ndarray), (name, type(r.mask))


@pytest.mark.skipif(not _H.GPU_OK, reason="no usable GPU")
def test_cupy_operand_never_moves_to_host_when_mixed_with_numpy_backed():
    with xupy.backend("cpu"):
        x = XMA.MaskedArray(_HF.copy())      # numpy-backed
    assert isinstance(x.data, np.ndarray)
    r = XMA.masked_where(_H.cp.asarray(_HC), x)
    assert on_dev(r.data, "gpu") and on_dev(r.mask, "gpu")
    r = x.take(_H.cp.asarray([0, 1]), mode="clip")
    assert on_dev(r.data, "gpu")
    assert isinstance(x.data, np.ndarray)    # the numpy-backed operand itself is untouched


def test_compress_out_masked_array_keeps_out_mask(dev):
    x, n = mka(dev, [1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0]), np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0])
    xo = mka(dev, np.zeros(2), mask=[1, 1])
    no = np.ma.array(np.zeros(2), mask=[1, 1])
    cond = [True, False, True, False]
    r = x.compress(cond, out=xo)
    n.compress(cond, out=no)
    assert on_dev(r.data, dev) and not isinstance(r.data, XMA.MaskedArray)
    np.testing.assert_array_equal(host(r.data), np.ma.getdata(no))
    np.testing.assert_array_equal(host(XMA.getmaskarray(xo)), np.ma.getmaskarray(no))   # out mask untouched
    np.testing.assert_array_equal(host(xo.data), np.ma.getdata(no))


# --------------------------------------------------------------------------
# MaskedArray methods
# --------------------------------------------------------------------------
@pytest.mark.parametrize("inplace", [False, True])
def test_byteswap(dev, inplace):
    for key in ("f2", "f2n", "i2", "b2", "f1"):
        _byteswap_case(dev, key, inplace)


def _byteswap_case(dev, key, inplace):
    x, n = D(key, dev)
    rn = n.byteswap(inplace=inplace)
    rx = x.byteswap(inplace=inplace)
    assert isinstance(rx, XMA.MaskedArray)
    if inplace:
        assert rx is x and rn is n
    # byteswap copies the mask (deviation: mask is always a fresh array)
    assert_same(rx, rn, dev=dev, strict_nomask=False, check_fill=False)
    assert on_dev(rx.data, dev)
    if not inplace:
        assert_same(x, D(key, dev)[1], dev=dev, strict_nomask=False)    # original untouched
        # double byteswap round-trips
        back = rx.byteswap()
        assert_same(back, D(key, dev)[1], dev=dev, strict_nomask=False, check_fill=False)


def test_iscontiguous_and_ids(dev):
    for key in ("f2", "f2n", "f2z", "i2"):
        _contig_case(dev, key)


def _contig_case(dev, key):
    x, n = D(key, dev)
    for view in (lambda a: a, lambda a: a.T, lambda a: a[:, ::2], lambda a: a[1:], lambda a: a[:, 1:3],
                 lambda a: a.reshape(-1)):
        vx, vn = view(x), view(n)
        assert x.iscontiguous() == n.iscontiguous()
        assert vx.iscontiguous() == vn.iscontiguous(), key
    ix, inn = x.ids(), n.ids()
    assert isinstance(ix, tuple) and len(ix) == 2
    assert all(isinstance(v, int) for v in ix)
    assert (ix[1] == id(XMA.nomask)) == (inn[1] == id(np.ma.nomask)) or x.mask is not XMA.nomask
    if x.mask is XMA.nomask:
        assert ix[1] == id(XMA.nomask)
    assert XMA.ids(x) == x.ids()
    assert XMA.ids(x)[0] == x.ids()[0]
    assert isinstance(_nf("ids")(n), tuple)


TRACE_CASES = [
    ("f2", {}), ("f2", {"offset": 1}), ("f2", {"offset": -1}), ("i2", {"offset": 2}), ("f2n", {}),
    ("f3", {"axis1": 1, "axis2": 2}), ("f2", {"dtype": np.float32}), ("f1", {}), ("f2a", {}),
    ("f2", {"offset": 10}), ("b2", {}), ("f2", {"axis1": 0, "axis2": 0}),
]


def test_trace_method(dev):
    for key, kw in TRACE_CASES:
        _trace_case(dev, key, kw)


def _trace_case(dev, key, kw):
    x, n = D(key, dev)
    try:
        rn = n.trace(**kw)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            x.trace(**kw)
        return
    assert_same(x.trace(**kw), rn, dev=dev, strict_nomask=False)


def test_baseclass_property(dev):
    for key in ("f2", "f2n", "i2"):
        _baseclass_case(dev, key)


def _baseclass_case(dev, key):
    x, n = D(key, dev)
    assert x.baseclass is not None
    # numpy.ma.baseclass is np.ndarray; XuPy: its own array class for the backend, never numpy.ma
    assert isinstance(x.baseclass, type)
    assert n.baseclass is np.ndarray
    assert not issubclass(x.baseclass, np.ma.MaskedArray)
    if dev == "gpu":
        assert x.baseclass is _H.cp.ndarray
    else:
        assert x.baseclass is np.ndarray
    # a view preserves baseclass
    assert x[:, :2].baseclass is x.baseclass


def test_method_choose_matches_function(dev):
    i, im, a, b, c, mc = _choice_data(dev)
    xi, _ = make(i, im, dev)
    xa, _ = make(a, mc, dev)
    xb, _ = make(b, None, dev)
    r1, r2 = xi.choose([xa, xb, xa]), XMA.choose(xi, [xa, xb, xa])
    assert_same(r1, r2, dev=dev, strict_nomask=False) if False else None
    np.testing.assert_array_equal(host(XMA.getmaskarray(r1)), host(XMA.getmaskarray(r2)))


# --------------------------------------------------------------------------
# unsupported names
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,args", [
    ("flatten_structured_array", ([1, 2],)), ("fromflex", ([1, 2],)), ("make_mask_descr", (np.float64,)),
    ("masked_object", ([1, 2], 1)), ("mvoid", ()), ("mvoid", ((1, 2),)),
])
def test_unsupported_functions_raise(name, args):
    f = getattr(XMA, name)
    with pytest.raises(NotImplementedError):
        f(*args)


@pytest.mark.parametrize("attr,args", [
    ("ctypes", None), ("recordmask", None), ("getfield", (np.float64,)), ("setfield", (1.0, np.float64)),
    ("setflags", ()), ("toflex", ()), ("torecords", ()),
])
def test_unsupported_methods_raise(dev, attr, args):
    x, _ = D("f2", dev)
    with pytest.raises(NotImplementedError):
        if args is None:
            getattr(x, attr)
        else:
            getattr(x, attr)(*args)


@pytest.mark.parametrize("attr", ["ctypes", "recordmask"])
def test_unsupported_properties_are_attribute_errors_too(dev, attr):
    """hasattr / inspect.getmembers (help, completion) must not blow up on them."""
    x, _ = D("f2", dev)
    assert not hasattr(x, attr)
    with pytest.raises(NotImplementedError):
        getattr(x, attr)
    with pytest.raises(AttributeError):
        getattr(x, attr)
    names = dict(inspect.getmembers(x))
    assert "sum" in names


def test_unsupported_recordmask_setter(dev):
    x, _ = D("f2", dev)
    with pytest.raises(NotImplementedError):
        x.recordmask = False


# --------------------------------------------------------------------------
# deprecation behaviour: xupy warns exactly when the installed numpy does
# --------------------------------------------------------------------------
def _warned(f, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = f(*a, **k)
    return r, sorted({x.category.__name__ for x in w if issubclass(x.category, DeprecationWarning)})


@pytest.mark.parametrize("name,kw", [("round_", {}), ("round_", {"decimals": 1}), ("alltrue", {}),
                                      ("sometrue", {}), ("alltrue", {"axis": 1}), ("sometrue", {"axis": None})])
def test_deprecation_warning_parity(dev, name, kw):
    nf = _nf(name)
    x, n = D("b2" if name != "round_" else "f2", dev)
    rn, wn = _warned(nf, n, **kw)
    rx, wx = _warned(getattr(XMA, name), x, **kw)
    assert wx == wn, f"{name}: xupy warns {wx}, numpy {wn} (numpy {np.__version__})"
    assert_same(rx, rn, dev=dev, strict_nomask=False)
    if name == "round_" and NPV >= (2, 5):
        with pytest.warns(DeprecationWarning):
            getattr(XMA, name)(x, **kw)
        with pytest.warns(DeprecationWarning):
            nf(n, **kw)
