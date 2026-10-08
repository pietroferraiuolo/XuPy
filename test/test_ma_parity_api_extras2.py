"""
Differential tests of the newer ``numpy.ma.extras`` names of ``xupy.ma``
against ``numpy.ma``:

anom anomalies apply_along_axis apply_over_axes clump_masked clump_unmasked
corrcoef cov flatnotmasked_contiguous flatnotmasked_edges in1d intersect1d
isin median ndenumerate notmasked_contiguous notmasked_edges polyfit
setdiff1d setxor1d union1d unique vander, module-level std / var and the
``numpy.ma.extras.__all__`` coverage.

Accepted deviations (mirrored by the arguments of the tests):
* BOOL data with masks: numpy's set operations are buggy -> set ops / unique
  are tested on int and float data only (bool without mask is fine).
* in1d / isin with ``assume_unique=True`` return ``nomask`` only if both
  inputs are nomask -> ``strict_nomask=False`` there.
* intersect1d / setxor1d with ``assume_unique=True`` flatten N-d inputs ->
  N-d inputs are not tested with ``assume_unique``.
* ``median(overwrite_input=...)`` never modifies the input.
* ``var/std(axis=..., ddof>0)``: the mask may be an all-False array where
  numpy gives ``nomask`` -> ``strict_nomask=False``.
"""
import inspect
import warnings

import numpy as np
import pytest

import xupy
from ._ma_parity_helpers import (
    XMA, GPU_OK, cp, make, mka, assert_same, both, to_dev, host, on_dev,
)

_RNG = np.random.default_rng(424242)


def _fixed_corrcoef(x, y=None, rowvar=True, allow_masked=True):
    """numpy >= 2.2's ``numpy.ma.corrcoef`` (covariance / outer(std, std)), on the installed numpy."""
    corr = np.ma.cov(x, y, rowvar, allow_masked=allow_masked)
    try:
        std = np.ma.sqrt(np.ma.diagonal(corr))
    except ValueError:
        return np.ma.core.MaskedConstant()
    corr /= np.ma.multiply.outer(std, std)
    return corr


def _corrcoef_is_cov_based():
    probe = np.ma.masked_array([[100.0, 101.0, 102.0, 105.0], [101.0, 103.0, 106.0, 99.0]],
                               mask=[[0, 0, 0, 1], [0, 0, 0, 0]])
    return bool(np.allclose(np.ma.corrcoef(probe), _fixed_corrcoef(probe), rtol=1e-9, atol=0))


# numpy 2.0's numpy.ma.corrcoef (numpy < 2.2 as far as checked: 2.0.2) estimates each
# coefficient from the pair-wise complete samples; numpy >= 2.2 divides the covariance by
# outer(std, std) of its diagonal (and 2.5 dropped the deprecated bias/ddof).  XuPy follows the
# newer algorithm, so on an old numpy the oracle is the newer algorithm above, built on that
# numpy's own ``cov``.  On numpy >= 2.2 this is numpy.ma.corrcoef itself.
_np_corrcoef = np.ma.corrcoef if _corrcoef_is_cov_based() else _fixed_corrcoef


def _ref(name):
    if name == "corrcoef":
        return _np_corrcoef
    fn = getattr(np.ma, name, None)
    if fn is None:
        fn = getattr(np.ma.extras, name, None)
    if fn is None:
        pytest.skip(f"numpy.ma has no {name} in numpy {np.__version__}")
    return fn


def _accepts(fn, kw):
    try:
        return kw in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _data(dtype, shape, lo=-6, hi=7):
    """Integer-valued data with many duplicates (halves for floats)."""
    a = _RNG.integers(lo, hi, size=shape)
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return (a / 2.0).astype(dt)
    return a.astype(dt)


def _rnd(dtype, shape):
    return _RNG.standard_normal(shape).astype(dtype)


def _rmask(shape, p=0.35):
    m = _RNG.random(shape) < p
    m.flat[0] = True    # always partly masked: keeps the nomask-ness of the
    m.flat[-1] = False  # references independent of the random draw
    return m


DTYPES = [np.int64, np.float64, np.float32]
SHAPES = [(7,), (4, 5), (3, 4, 5)]


def _axes(nd):
    return [None] + list(range(nd)) + [-1] + ([tuple(range(nd))[:2]] if nd > 1 else [])


def _pair(data, mask, dev, kind="masked"):
    """-> (xupy, numpy.ma) pair for the input kind."""
    if kind == "masked":
        return make(data, mask, dev)
    if kind == "nomask":
        return make(data, None, dev)
    if kind == "allmasked":
        return make(data, np.ones(np.shape(data), bool), dev)
    if kind == "mask_false":
        return make(data, np.zeros(np.shape(data), bool), dev)
    raise AssertionError(kind)


KINDS = ["masked", "nomask", "allmasked", "mask_false"]


def _ctx(*label):
    class _C:
        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if ev is not None and hasattr(ev, "add_note"):
                ev.add_note(f"case: {label!r}")
            return False
    return _C()


def _cmp(name, args, dev, strict_nomask=True, check_fill=True, **kw):
    fx, fn = getattr(XMA, name), _ref(name)
    rx, rn = both(fx, fn, *args, **kw)
    if rx is None and rn is None:
        return None
    assert_same(rx, rn, dev=dev, strict_nomask=strict_nomask, check_fill=check_fill, ctx=name)
    return rx


def _close(x, n, dev=None, rtol=1e-10, check_mask=True):
    """Looser comparison: structure/kind of ``assert_same`` is replaced by
    host values (allclose) plus separate mask checks."""
    if n is np.ma.masked:
        assert x is XMA.masked
        return
    mn = np.ma.getmaskarray(n)
    if isinstance(x, XMA.MaskedArray):
        mx = host(XMA.getmaskarray(x))
        dx = host(x.data)
    else:
        mx = np.zeros(np.shape(n), bool)
        dx = host(x)
    assert dx.shape == np.shape(n)
    if check_mask:
        np.testing.assert_array_equal(mx, mn)
    keep = ~mn
    np.testing.assert_allclose(dx[keep], np.asarray(n.data if hasattr(n, "data") and isinstance(n, np.ma.MaskedArray) else n)[keep],
                               rtol=rtol, atol=1e-12 if rtol < 1e-6 else 1e-6, equal_nan=True)
    if dev is not None and not np.isscalar(x) and hasattr(x, "shape") and x.ndim:
        arr = x.data if isinstance(x, XMA.MaskedArray) else x
        assert on_dev(arr, dev)


def _warn_kinds(fn, *a, **k):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn(*a, **k)
        except Exception:  # noqa: BLE001
            pass
    return sorted({x.category.__name__ for x in w})


# ---------------------------------------------------------------------------
# __all__ coverage
# ---------------------------------------------------------------------------
NEW_NAMES = (
    "anom anomalies apply_along_axis apply_over_axes clump_masked clump_unmasked "
    "corrcoef cov flatnotmasked_contiguous flatnotmasked_edges in1d intersect1d isin "
    "median ndenumerate notmasked_contiguous notmasked_edges polyfit setdiff1d setxor1d "
    "union1d unique vander std var"
).split()


class TestCoverage:
    def test_all_covered(self):
        missing = set(np.ma.extras.__all__) - set(XMA.extras.__all__)
        assert not missing, sorted(missing)

    def test_new_names_present_and_exported(self):
        for name in NEW_NAMES:
            assert callable(getattr(XMA.extras, name)), name
            assert name in XMA.extras.__all__, name
            if hasattr(np.ma, name):
                assert callable(getattr(XMA, name)), name

    def test_signatures_follow_numpy(self):
        for name in NEW_NAMES:
            try:
                fn = _ref(name)
            except pytest.skip.Exception:
                continue
            pn = list(inspect.signature(fn).parameters)
            if name == "corrcoef":
                # numpy < 2.5 still has the deprecated, ignored `bias`/`ddof` parameters
                # (default np._NoValue); numpy 2.5 removed them and so does XuPy.
                pn = [q for q in pn if q not in ("bias", "ddof")]
            if name in ("std", "var") and pn == ["a", "args", "params"]:
                pn = ["a"]  # numpy < 2.5: opaque (a, *args, **params) forwarder; 2.5 spells it out
            sx = inspect.signature(getattr(XMA.extras, name))
            if any(p.kind is p.VAR_POSITIONAL for p in sx.parameters.values()):
                continue  # generic *args/**kwargs forwarders (anom/anomalies)
            px = list(sx.parameters)
            assert px[:len(pn)] == pn, (name, pn, px)


# ---------------------------------------------------------------------------
# std / var
# ---------------------------------------------------------------------------
class TestStdVar:
    @pytest.mark.parametrize("name", ["std", "var"])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_axes(self, dev, name, dtype):
        for shape in SHAPES:
            data = _data(dtype, shape)
            mask = _rmask(shape)
            for kind in KINDS:
                x, n = _pair(data, mask, dev, kind)
                for axis in _axes(len(shape)):
                    for ddof in (0, 1):
                        with _ctx(name, dtype, shape, kind, axis, ddof):
                            _cmp(name, [(x, n)], dev, strict_nomask=False,  # all-False mask array where numpy shrinks (needs a sync)
                                 axis=axis, ddof=ddof)

    @pytest.mark.parametrize("name", ["std", "var"])
    def test_keepdims_dtype(self, dev, name):
        data = _data(np.float64, (4, 5))
        x, n = make(data, _rmask((4, 5)), dev)
        for axis in (None, 0, 1):
            with _ctx(axis):
                _cmp(name, [(x, n)], dev, strict_nomask=False, axis=axis, keepdims=True)
                _cmp(name, [(x, n)], dev, strict_nomask=False, axis=axis, keepdims=False)
        xi, ni = make(_data(np.int64, (4, 5)), _rmask((4, 5)), dev)
        for dt in (np.float32, np.float64):
            _cmp(name, [(xi, ni)], dev, strict_nomask=False, axis=0, dtype=dt)
            _cmp(name, [(xi, ni)], dev, strict_nomask=False, dtype=dt)

    @pytest.mark.parametrize("name", ["std", "var"])
    def test_out(self, dev, name):
        data = _data(np.float64, (4, 5))
        mask = _rmask((4, 5))
        x, n = make(data, mask, dev)
        on = np.ma.masked_array(np.zeros(5), mask=np.zeros(5, bool))
        ox = mka(dev, to_dev(np.zeros(5), dev), mask=to_dev(np.zeros(5, bool), dev))
        rx = getattr(XMA, name)(x, axis=0, out=ox)
        rn = _ref(name)(n, axis=0, out=on)
        assert rx is ox and rn is on
        assert_same(ox, on, dev=dev, strict_nomask=False)

    @pytest.mark.parametrize("name", ["std", "var"])
    def test_mean_kw(self, dev, name):
        if not _accepts(_ref(name), "mean"):
            pytest.skip("no mean= in this numpy.ma")
        data = _data(np.float64, (4, 5))
        x, n = make(data, _rmask((4, 5)), dev)
        for axis in (None, 0, 1):
            mn = n.mean(axis=axis, keepdims=True)
            mx = x.mean(axis=axis, keepdims=True)
            with _ctx(axis):
                rn = _ref(name)(n, axis=axis, mean=mn)
                rx = getattr(XMA, name)(x, axis=axis, mean=mx)
                assert_same(rx, rn, dev=dev, strict_nomask=False)

    @pytest.mark.parametrize("name", ["std", "var"])
    def test_edge(self, dev, name):
        # ddof >= count, fully masked, single element, size 0, 0-d, plain ndarray / list inputs
        x, n = make(_data(np.float64, (5,)), np.array([1, 1, 1, 1, 0], bool), dev)
        _cmp(name, [(x, n)], dev, ddof=1)
        _cmp(name, [(x, n)], dev, ddof=2)
        x, n = make(np.arange(4.0), np.ones(4, bool), dev)
        _cmp(name, [(x, n)], dev)
        _cmp(name, [(x, n)], dev, axis=0)
        x, n = make(np.zeros((0, 3)), None, dev)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cmp(name, [(x, n)], dev, axis=0)
        a = _data(np.float64, (3, 4))
        _cmp(name, [(to_dev(a, dev), a.copy())], dev, axis=1)
        _cmp(name, [(to_dev(a, dev), a.copy())], dev)
        _cmp(name, [(x, n)], dev, axis=5)

    @pytest.mark.parametrize("name", ["std", "var"])
    def test_bad_axis_and_ddof(self, dev, name):
        x, n = make(_data(np.float64, (4, 5)), _rmask((4, 5)), dev)
        for axis in (2, -3, (0, 0)):
            _cmp(name, [(x, n)], dev, axis=axis)


# ---------------------------------------------------------------------------
# anom / anomalies
# ---------------------------------------------------------------------------
class TestAnom:
    @pytest.mark.parametrize("name", ["anom", "anomalies"])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_basic(self, dev, name, dtype):
        for shape in SHAPES:
            data = _data(dtype, shape)
            mask = _rmask(shape)
            for kind in KINDS:
                x, n = _pair(data, mask, dev, kind)
                for axis in [None] + list(range(len(shape))):
                    with _ctx(name, dtype, shape, kind, axis):
                        _cmp(name, [(x, n)], dev, axis=axis)

    @pytest.mark.parametrize("name", ["anom", "anomalies"])
    def test_dtype_and_method(self, dev, name):
        x, n = make(_data(np.int64, (4, 5)), _rmask((4, 5)), dev)
        for dt in (np.float32, np.float64):
            _cmp(name, [(x, n)], dev, axis=1, dtype=dt)
        for axis in (None, 0, 1):
            assert_same(x.anom(axis=axis), n.anom(axis=axis), dev=dev)

    def test_bad_axis(self, dev):
        x, n = make(_data(np.float64, (4, 5)), _rmask((4, 5)), dev)
        _cmp("anom", [(x, n)], dev, axis=3)


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------
class TestMedian:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_axes_kinds(self, dev, dtype):
        for shape in SHAPES:
            data = _data(dtype, shape)
            mask = _rmask(shape)
            for kind in KINDS:
                x, n = _pair(data, mask, dev, kind)
                for axis in _axes(len(shape)):
                    for keepdims in (False, True):
                        with _ctx(dtype, shape, kind, axis, keepdims):
                            _cmp("median", [(x, n)], dev, axis=axis, keepdims=keepdims)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_complex_keeps_imaginary_part(self, dev, axis):
        c = np.array([[1 + 2j, 3 - 1j, 2 + 0.5j, 5j], [0 + 1j, 4 + 4j, 1 - 1j, 2 + 2j]])
        mask = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], bool)
        x, n = make(c, mask, dev)
        try:
            rn = np.ma.median(n, axis=axis)
        except Exception as e:  # numpy.ma without complex support
            pytest.skip(f"numpy.ma.median(complex) fails: {type(e).__name__}")
        rx = XMA.median(x, axis=axis)
        assert np.dtype(getattr(rx, "dtype", None) or np.asarray(rx).dtype).kind == "c"
        np.testing.assert_allclose(host(getattr(rx, "data", rx)), np.ma.getdata(rn))

    def test_odd_even_counts(self, dev):
        for k in range(0, 9):  # k unmasked values among 8, in 1-d
            data = _data(np.float64, (8,))
            mask = np.ones(8, bool)
            mask[_RNG.permutation(8)[:k]] = False
            x, n = make(data, mask, dev)
            with _ctx(k):
                _cmp("median", [(x, n)], dev)
            xi, ni = make(data.astype(np.int64), mask, dev)
            with _ctx("int", k):
                _cmp("median", [(xi, ni)], dev)

    def test_lanes_with_some_all_masked(self, dev):
        data = _data(np.float64, (4, 5))
        mask = _rmask((4, 5))
        mask[1, :] = True
        mask[:, 2] = True
        x, n = make(data, mask, dev)
        for axis in (0, 1, None):
            for keepdims in (False, True):
                _cmp("median", [(x, n)], dev, axis=axis, keepdims=keepdims)

    def test_nan_and_inf(self, dev):
        data = np.array([[1.0, np.nan, 3.0, 4.0], [np.inf, 2.0, -np.inf, 5.0]])
        mask = np.array([[0, 0, 0, 1], [0, 0, 0, 1]], bool)
        x, n = make(data, mask, dev)
        for axis in (None, 0, 1):
            _cmp("median", [(x, n)], dev, axis=axis)
        masked_nan = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], bool)
        x, n = make(data, masked_nan, dev)
        for axis in (None, 0, 1):
            _cmp("median", [(x, n)], dev, axis=axis)

    def test_out(self, dev):
        data = _data(np.float64, (4, 5))
        mask = _rmask((4, 5))
        x, n = make(data, mask, dev)
        for axis in (0, 1):
            sh = (5,) if axis == 0 else (4,)
            on = np.ma.masked_array(np.zeros(sh), mask=np.zeros(sh, bool))
            ox = mka(dev, to_dev(np.zeros(sh), dev), mask=to_dev(np.zeros(sh, bool), dev))
            rx = XMA.median(x, axis=axis, out=ox)
            rn = np.ma.median(n, axis=axis, out=on)
            assert rx is ox and rn is on
            assert_same(ox, on, dev=dev, strict_nomask=False)
        # ndarray out for an unmasked input
        a = _data(np.float64, (4, 5))
        xo, no = to_dev(np.zeros(5), dev), np.zeros(5)
        rx = XMA.median(to_dev(a, dev), axis=0, out=xo)
        rn = np.ma.median(a, axis=0, out=no)
        # numpy wraps the filled ndarray ``out`` in a masked array; only values are compared
        np.testing.assert_allclose(host(xo), no)
        np.testing.assert_allclose(host(XMA.getdata(rx)), np.ma.getdata(rn))

    def test_overwrite_input_never_modifies(self, dev):
        data = _data(np.float64, (4, 5))
        mask = _rmask((4, 5))
        x, n = make(data, mask, dev)
        d0, m0 = host(x.data).copy(), host(XMA.getmaskarray(x)).copy()
        for axis in (None, 0, 1):
            rx = XMA.median(x, axis=axis, overwrite_input=True)
            rn = np.ma.median(make(data, mask, "cpu")[1], axis=axis, overwrite_input=True)
            assert_same(rx, rn, dev=dev)
        np.testing.assert_array_equal(host(x.data), d0)
        np.testing.assert_array_equal(host(XMA.getmaskarray(x)), m0)

    @pytest.mark.parametrize("inkind", ["plain", "list", "npma"])
    def test_non_masked_inputs(self, dev, inkind):
        a = _data(np.float64, (4, 5))
        if inkind == "plain":
            x, n = to_dev(a, dev), a.copy()
        elif inkind == "list":
            x, n = a.tolist(), a.tolist()
        else:
            n = np.ma.masked_array(a, mask=_rmask((4, 5)))
            x = n.copy()
        for axis in (None, 0, 1):
            fx, fn = XMA.median, np.ma.median
            rn = fn(n, axis=axis)
            rx = fx(x, axis=axis)
            if inkind == "npma":
                _close(rx, rn)
            elif inkind == "list":
                # host list -> any array of the right values; no kind assertion
                np.testing.assert_allclose(host(rx), np.asarray(rn))
            else:
                np.testing.assert_allclose(host(rx), np.asarray(rn))
                if np.ndim(rn):
                    assert on_dev(rx.data if isinstance(rx, XMA.MaskedArray) else rx, dev)

    def test_empty_and_scalar(self, dev):
        for shape in ((0,), (0, 3), (3, 0)):
            x, n = make(np.zeros(shape), None, dev)
            for axis in (None, 0):
                with _ctx(shape, axis), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _cmp("median", [(x, n)], dev, axis=axis)
        x, n = make(np.array(3.5), None, dev)
        _cmp("median", [(x, n)], dev)
        x, n = make(np.array(3.5), True, dev)
        _cmp("median", [(x, n)], dev)

    def test_bad_axis(self, dev):
        x, n = make(_data(np.float64, (4, 5)), _rmask((4, 5)), dev)
        for axis in (2, -3):
            _cmp("median", [(x, n)], dev, axis=axis)

    def test_hardmask_fill_value_preserved(self, dev):
        x, n = make(_data(np.float64, (4, 5)), _rmask((4, 5)), dev, fill_value=-9.0)
        _cmp("median", [(x, n)], dev, axis=0)


# ---------------------------------------------------------------------------
# unique and set operations (int / float only for masked data)
# ---------------------------------------------------------------------------
SET_DTYPES = [np.int64, np.float64, np.float32]


class TestUnique:
    @pytest.mark.parametrize("dtype", SET_DTYPES)
    def test_unique(self, dev, dtype):
        for shape in SHAPES:
            data = _data(dtype, shape)
            mask = _rmask(shape)
            for kind in KINDS:
                x, n = _pair(data, mask, dev, kind)
                # an all-False mask array: numpy keeps it, xupy collapses it to nomask
                sn = kind != "mask_false"
                with _ctx(dtype, shape, kind):
                    _cmp("unique", [(x, n)], dev, strict_nomask=sn)
                    rx, rn = XMA.unique(x, return_index=True), np.ma.unique(n, return_index=True)
                    assert_same(rx, rn, dev=dev, strict_nomask=sn)
                    rx, rn = XMA.unique(x, return_inverse=True), np.ma.unique(n, return_inverse=True)
                    assert_same(rx, rn, dev=dev, strict_nomask=sn)
                    rx = XMA.unique(x, True, True)
                    rn = np.ma.unique(n, True, True)
                    assert_same(rx, rn, dev=dev, strict_nomask=sn)

    def test_unique_bool_unmasked(self, dev):
        a = _RNG.random(9) > 0.5
        x, n = make(a, None, dev)
        _cmp("unique", [(x, n)], dev)
        assert_same(XMA.unique(x, return_inverse=True), np.ma.unique(n, return_inverse=True), dev=dev)

    def test_unique_edge(self, dev):
        for data, mask in [
            (np.zeros(0), None),
            (np.zeros(0), np.zeros(0, bool)),
            (np.array([5.0]), None),
            (np.array([3, 3, 3]), None),
            (np.array([3, 3, 3]), [True, True, True]),
            (np.array([3, 3, 3]), [False, True, False]),
        ]:
            m = None if mask is None else np.array(mask, bool)
            x, n = make(data, m, dev)
            with _ctx(data.tolist(), mask):
                _cmp("unique", [(x, n)], dev, strict_nomask=False)  # all-False mask -> nomask in xupy
                assert_same(XMA.unique(x, True, True), np.ma.unique(n, True, True), dev=dev, strict_nomask=False)

    def test_unique_single_masked(self, dev):
        x, n = make(np.array([5.0]), np.array([True]), dev)
        _cmp("unique", [(x, n)], dev, strict_nomask=False)
        assert_same(XMA.unique(x, True, True), np.ma.unique(n, True, True), dev=dev, strict_nomask=False)

    def test_unique_nan_inf(self, dev):
        data = np.array([np.nan, 1.0, np.nan, np.inf, -np.inf, 1.0, np.inf])
        x, n = make(data, None, dev)
        _cmp("unique", [(x, n)], dev)
        x, n = make(data, np.array([0, 0, 0, 0, 0, 1, 0], bool), dev)
        _cmp("unique", [(x, n)], dev, strict_nomask=False)

    def test_unique_plain_and_list(self, dev):
        a = _data(np.int64, (3, 4))
        rx = XMA.unique(to_dev(a, dev))
        assert_same(rx, np.ma.unique(a), dev=dev)
        rx = XMA.unique(a.tolist())
        assert_same(rx, np.ma.unique(a.tolist()))


def _set_pair(dev, dtype, kind1, kind2, shape1=(9,), shape2=(7,)):
    d1, d2 = _data(dtype, shape1), _data(dtype, shape2)
    return (_pair(d1, _rmask(shape1), dev, kind1), _pair(d2, _rmask(shape2), dev, kind2))


class TestSetOps:
    @pytest.mark.parametrize("name", ["intersect1d", "setxor1d", "union1d", "setdiff1d"])
    @pytest.mark.parametrize("dtype", SET_DTYPES)
    def test_kinds(self, dev, name, dtype):
        for k1 in ("masked", "nomask", "allmasked"):
            for k2 in ("masked", "nomask", "allmasked"):
                for shapes in [((9,), (7,)), ((3, 4), (2, 5))]:
                    (x1, n1), (x2, n2) = _set_pair(dev, dtype, k1, k2, *shapes)
                    with _ctx(name, dtype, k1, k2, shapes):
                        _cmp(name, [(x1, n1), (x2, n2)], dev)

    @pytest.mark.parametrize("name", ["intersect1d", "setxor1d", "setdiff1d"])
    @pytest.mark.parametrize("dtype", SET_DTYPES)
    def test_assume_unique_1d(self, dev, name, dtype):
        # strictly increasing / unique 1-d inputs (N-d + assume_unique flattens in numpy)
        for k1 in ("masked", "nomask"):
            for k2 in ("masked", "nomask"):
                d1 = np.unique(_data(dtype, (12,)))
                d2 = np.unique(_data(dtype, (10,), -4, 9))
                x1, n1 = _pair(d1, _rmask(d1.shape, 0.2), dev, k1)
                x2, n2 = _pair(d2, _rmask(d2.shape, 0.2), dev, k2)
                with _ctx(name, dtype, k1, k2):
                    _cmp(name, [(x1, n1), (x2, n2)], dev, assume_unique=True,
                         strict_nomask=False)

    def test_bool_unmasked(self, dev):
        a, b = _RNG.random(8) > 0.5, _RNG.random(6) > 0.5
        for name in ("intersect1d", "setxor1d", "union1d", "setdiff1d"):
            (x1, n1), (x2, n2) = make(a, None, dev), make(b, None, dev)
            with _ctx(name):
                _cmp(name, [(x1, n1), (x2, n2)], dev)

    @pytest.mark.parametrize("name", ["intersect1d", "setxor1d", "union1d", "setdiff1d"])
    def test_edge(self, dev, name):
        cases = [
            (np.zeros(0), None, np.arange(3.0), None),
            (np.arange(3.0), None, np.zeros(0), None),
            (np.zeros(0), None, np.zeros(0), None),
            (np.arange(4.0), None, np.arange(4.0), None),
            (np.arange(4.0), np.ones(4, bool), np.arange(4.0), np.ones(4, bool)),
            (np.arange(4.0), np.ones(4, bool), np.arange(4.0), None),
            (np.arange(4.0), None, np.arange(4.0) + 10, None),
            (np.array([1, 2, 2, 3]), [0, 1, 0, 0], np.array([2, 3, 3]), [1, 0, 0]),
        ]
        for d1, m1, d2, m2 in cases:
            m1 = None if m1 is None else np.array(m1, bool)
            m2 = None if m2 is None else np.array(m2, bool)
            x1, n1 = make(d1, m1, dev)
            x2, n2 = make(d2, m2, dev)
            with _ctx(name, d1.tolist(), m1, d2.tolist(), m2):
                _cmp(name, [(x1, n1), (x2, n2)], dev, strict_nomask=False)

    @pytest.mark.parametrize("name", ["intersect1d", "setxor1d", "union1d", "setdiff1d"])
    def test_mixed_dtypes(self, dev, name):
        d1, d2 = _data(np.int64, (8,)), _data(np.float32, (6,))
        x1, n1 = make(d1, _rmask(8), dev)
        x2, n2 = make(d2, _rmask(6), dev)
        _cmp(name, [(x1, n1), (x2, n2)], dev)

    @pytest.mark.parametrize("name", ["intersect1d", "setxor1d", "union1d", "setdiff1d"])
    def test_plain_inputs(self, dev, name):
        a, b = _data(np.int64, (3, 4)), _data(np.int64, (5,))
        rx = getattr(XMA, name)(to_dev(a, dev), to_dev(b, dev))
        rn = _ref(name)(a, b)
        assert_same(rx, rn, dev=dev, strict_nomask=False)


class TestIn1dIsin:
    @pytest.mark.parametrize("name", ["in1d", "isin"])
    @pytest.mark.parametrize("dtype", SET_DTYPES)
    def test_kinds(self, dev, name, dtype):
        _ref(name)
        for k1 in ("masked", "nomask", "allmasked"):
            for k2 in ("masked", "nomask", "allmasked"):
                for shapes in [((9,), (7,)), ((3, 4), (2, 5))]:
                    for invert in (False, True):
                        (x1, n1), (x2, n2) = _set_pair(dev, dtype, k1, k2, *shapes)
                        with _ctx(name, dtype, k1, k2, shapes, invert):
                            _cmp(name, [(x1, n1), (x2, n2)], dev, invert=invert)

    @pytest.mark.parametrize("name", ["in1d", "isin"])
    @pytest.mark.parametrize("dtype", SET_DTYPES)
    def test_assume_unique(self, dev, name, dtype):
        _ref(name)
        for k1 in ("masked", "nomask"):
            for k2 in ("masked", "nomask"):
                for invert in (False, True):
                    d1 = np.unique(_data(dtype, (12,)))
                    d2 = np.unique(_data(dtype, (10,), -4, 9))
                    x1, n1 = _pair(d1, _rmask(d1.shape, 0.2), dev, k1)
                    x2, n2 = _pair(d2, _rmask(d2.shape, 0.2), dev, k2)
                    with _ctx(name, dtype, k1, k2, invert):
                        _cmp(name, [(x1, n1), (x2, n2)], dev, assume_unique=True,
                             invert=invert, strict_nomask=False)

    @pytest.mark.parametrize("name", ["in1d", "isin"])
    def test_edge_and_plain(self, dev, name):
        _ref(name)
        x1, n1 = make(np.arange(5.0), None, dev)
        x2, n2 = make(np.zeros(0), None, dev)
        _cmp(name, [(x1, n1), (x2, n2)], dev)
        _cmp(name, [(x2, n2), (x1, n1)], dev)
        x3, n3 = make(np.arange(5.0), np.ones(5, bool), dev)
        _cmp(name, [(x3, n3), (x1, n1)], dev)
        a = _data(np.int64, (3, 4))
        b = _data(np.int64, (5,))
        rx = getattr(XMA, name)(to_dev(a, dev), to_dev(b, dev))
        rn = _ref(name)(a, b)
        assert_same(rx, rn, dev=dev, strict_nomask=False)

    def test_isin_keeps_shape_and_bool_unmasked(self, dev):
        a = _RNG.random((3, 4)) > 0.5
        x, n = make(a, None, dev)
        t, tn = make(np.array([True]), None, dev)
        _cmp("isin", [(x, n), (t, tn)], dev)

    @pytest.mark.parametrize("name", ["in1d"])
    def test_deprecation_warning_parity(self, dev, name):
        _ref(name)
        (x1, n1), (x2, n2) = _set_pair(dev, np.int64, "masked", "masked")
        assert _warn_kinds(getattr(XMA, name), x1, x2) == _warn_kinds(_ref(name), n1, n2)

    @pytest.mark.parametrize("name", ["product", "row_stack"])
    def test_other_deprecations_parity(self, dev, name):
        fn = _ref(name)
        x, n = make(_data(np.float64, (3, 4)), _rmask((3, 4)), dev)
        if name == "product":
            assert _warn_kinds(getattr(XMA, name), x) == _warn_kinds(fn, n)
        else:
            assert _warn_kinds(getattr(XMA, name), (x, x)) == _warn_kinds(fn, (n, n))


# ---------------------------------------------------------------------------
# cov / corrcoef
# ---------------------------------------------------------------------------
class TestCov:
    @pytest.mark.parametrize("name", ["cov", "corrcoef"])
    @pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
    def test_variants(self, dev, name, dtype):
        rtol = 1e-4 if dtype == np.float32 else 1e-10
        data = (_rnd(np.float64, (4, 8)) * 10).astype(dtype)
        for mask in (_rmask((4, 8), 0.2), None, np.ones((4, 8), bool)):
            x, n = make(data, mask, dev)
            for rowvar in (True, False):
                with _ctx(name, dtype, None if mask is None else mask.all(), rowvar):
                    fx, fn = getattr(XMA, name), _ref(name)
                    rx, rn = both(fx, fn, (x, n), rowvar=rowvar)
                    if rn is None:
                        continue
                    _close(rx, rn, dev, rtol=rtol)
                    assert type(rx) is type(rn) or (isinstance(rx, XMA.MaskedArray) and isinstance(rn, np.ma.MaskedArray)) \
                        or (rx is XMA.masked and rn is np.ma.masked)

    @pytest.mark.parametrize("name", ["cov", "corrcoef"])
    def test_no_runtime_warnings_for_empty_estimates(self, dev, name):
        """Entries estimated from no sample are masked silently (numpy.ma uses errstate)."""
        data = _rnd(np.float64, (3, 4))
        mask = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]], bool)
        x, _ = make(data, mask, dev)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            getattr(XMA, name)(x)

    def test_xy_bias_ddof(self, dev):
        a = _rnd(np.float64, (3, 9))
        b = _rnd(np.float64, (2, 9))
        ma_, mb_ = _rmask((3, 9), 0.2), _rmask((2, 9), 0.2)
        xt, nt = make(a.T.copy(), ma_.T.copy(), dev)
        yt, nyt = make(b.T.copy(), mb_.T.copy(), dev)
        x, n = make(a, ma_, dev)
        y, ny = make(b, mb_, dev)
        for bias in (False, True):
            for ddof in (None, 0, 1, 2):
                with _ctx(bias, ddof):
                    rx, rn = XMA.cov(x, y, bias=bias, ddof=ddof), np.ma.cov(n, ny, bias=bias, ddof=ddof)
                    _close(rx, rn, dev)
        _close(XMA.corrcoef(x, y), _np_corrcoef(n, ny), dev)
        _close(XMA.corrcoef(xt, yt, rowvar=False), _np_corrcoef(nt, nyt, rowvar=False), dev)
        _close(XMA.cov(xt, yt, rowvar=False), np.ma.cov(nt, nyt, rowvar=False), dev)

    def test_1d_and_scalar(self, dev):
        a = _rnd(np.float64, (9,))
        x, n = make(a, _rmask((9,), 0.2), dev)
        _close(XMA.cov(x), np.ma.cov(n), dev)
        rx, rn = XMA.corrcoef(x), _np_corrcoef(n)
        assert rn is np.ma.masked
        assert rx is XMA.masked
        y, ny = make(_rnd(np.float64, (9,)), _rmask((9,), 0.2), dev)
        _close(XMA.cov(x, y), np.ma.cov(n, ny), dev)
        _close(XMA.corrcoef(x, y), _np_corrcoef(n, ny), dev)

    def test_few_samples_masked_entries(self, dev):
        # entries estimated from <= ddof samples are masked
        a = _rnd(np.float64, (3, 4))
        mask = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0]], bool)
        x, n = make(a, mask, dev)
        for ddof in (None, 0, 1):
            with _ctx(ddof):
                _close(XMA.cov(x, ddof=ddof), np.ma.cov(n, ddof=ddof), dev)
        _close(XMA.corrcoef(x), _np_corrcoef(n), dev)

    def test_allow_masked_false_and_errors(self, dev):
        a = _rnd(np.float64, (3, 6))
        x, n = make(a, _rmask((3, 6), 0.3), dev)
        for fx, fn in ((XMA.cov, np.ma.cov), (XMA.corrcoef, _np_corrcoef)):
            rx, rn = both(fx, fn, (x, n), allow_masked=False)
        x, n = make(a, None, dev)
        _close(XMA.cov(x, allow_masked=False), np.ma.cov(n, allow_masked=False), dev)
        # ddof not integer, bad shapes
        rx, rn = both(XMA.cov, np.ma.cov, (x, n), ddof=1.5)
        rx, rn = both(XMA.cov, np.ma.cov, (x, n), (x[:, :4], n[:, :4]))
        rx, rn = both(XMA.cov, np.ma.cov, (to_dev(np.zeros((2, 2, 2)), dev), np.zeros((2, 2, 2))))
        with _ctx("plain"):
            p = _rnd(np.float64, (3, 6))
            _close(XMA.cov(to_dev(p, dev)), np.ma.cov(p), dev)
            _close(XMA.corrcoef(to_dev(p, dev)), _np_corrcoef(p), dev)


# ---------------------------------------------------------------------------
# ndenumerate / edges / contiguous / clumps
# ---------------------------------------------------------------------------
MASK_PATTERNS = {
    "partial1d": ((9,), np.array([1, 0, 0, 1, 1, 0, 1, 0, 0], bool)),
    "first_masked": ((6,), np.array([1, 1, 0, 0, 0, 0], bool)),
    "last_masked": ((6,), np.array([0, 0, 0, 1, 1, 1], bool)),
    "alt": ((6,), np.array([1, 0, 1, 0, 1, 0], bool)),
    "none": ((6,), np.zeros(6, bool)),
    "all": ((6,), np.ones(6, bool)),
    "single_ok": ((1,), np.zeros(1, bool)),
    "single_masked": ((1,), np.ones(1, bool)),
    "2d": ((4, 5), _rmask((4, 5), 0.4)),
    "2d_row_masked": ((4, 5), np.repeat(np.array([[0], [1], [0], [0]], bool), 5, 1)),
    "2d_all": ((4, 5), np.ones((4, 5), bool)),
    "2d_none": ((4, 5), np.zeros((4, 5), bool)),
    "2d_col": ((3, 6), np.tile(np.array([0, 0, 1, 1, 0, 1], bool), (3, 1))),
    "3d": ((3, 4, 5), _rmask((3, 4, 5), 0.4)),
}


class TestIterationAndRuns:
    @pytest.mark.parametrize("compressed", [True, False])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_ndenumerate(self, dev, compressed, dtype):
        for key, (shape, mask) in MASK_PATTERNS.items():
            data = _data(dtype, shape)
            for kind in ("masked", "nomask"):
                x, n = make(data, mask if kind == "masked" else None, dev)
                with _ctx(key, kind, compressed):
                    lx = list(XMA.ndenumerate(x, compressed=compressed))
                    ln = list(_ref("ndenumerate")(n, compressed=compressed))
                    assert len(lx) == len(ln)
                    for (ix, vx), (inn, vn) in zip(lx, ln):
                        assert ix == inn and all(type(i) is int for i in ix)
                        if vn is np.ma.masked:
                            assert vx is XMA.masked
                        else:
                            assert type(vx) is type(vn) and vx == vn

    def test_ndenumerate_plain_and_empty(self, dev):
        a = _data(np.int64, (2, 3))
        lx = list(XMA.ndenumerate(to_dev(a, dev)))
        ln = list(np.ma.ndenumerate(a))
        assert [(i, int(v)) for i, v in lx] == [(i, int(v)) for i, v in ln]
        x, n = make(np.zeros((0, 3)), None, dev)
        assert list(XMA.ndenumerate(x)) == list(np.ma.ndenumerate(n)) == []
        x, n = make(np.array(4.0), None, dev)
        assert [(i, float(v)) for i, v in XMA.ndenumerate(x)] == [(i, float(v)) for i, v in np.ma.ndenumerate(n)]

    def test_flatnotmasked_edges(self, dev):
        for key, (shape, mask) in MASK_PATTERNS.items():
            for kind in ("masked", "nomask"):
                x, n = make(_data(np.float64, shape), mask if kind == "masked" else None, dev)
                with _ctx(key, kind):
                    _cmp("flatnotmasked_edges", [(x, n)], dev)

    def test_flatnotmasked_contiguous(self, dev):
        for key, (shape, mask) in MASK_PATTERNS.items():
            for kind in ("masked", "nomask"):
                x, n = make(_data(np.float64, shape), mask if kind == "masked" else None, dev)
                with _ctx(key, kind):
                    rx = XMA.flatnotmasked_contiguous(x)
                    rn = np.ma.flatnotmasked_contiguous(n)
                    assert isinstance(rx, list)
                    assert [(s.start, s.stop, s.step) for s in rx] == [(s.start, s.stop, s.step) for s in rn]

    def test_notmasked_edges(self, dev):
        for key, (shape, mask) in MASK_PATTERNS.items():
            for kind in ("masked", "nomask"):
                x, n = make(_data(np.float64, shape), mask if kind == "masked" else None, dev)
                for axis in [None] + list(range(len(shape))):
                    with _ctx(key, kind, axis):
                        _cmp("notmasked_edges", [(x, n)], dev, axis=axis)

    def test_notmasked_contiguous(self, dev):
        for key, (shape, mask) in MASK_PATTERNS.items():
            for kind in ("masked", "nomask"):
                x, n = make(_data(np.float64, shape), mask if kind == "masked" else None, dev)
                for axis in [None] + list(range(len(shape))):
                    with _ctx(key, kind, axis):
                        fx, fn = XMA.notmasked_contiguous, np.ma.notmasked_contiguous
                        rx, rn = both(fx, fn, (x, n), axis=axis)
                        if rn is None:
                            continue
                        assert _slices(rx) == _slices(rn)

    @pytest.mark.parametrize("name", ["clump_masked", "clump_unmasked"])
    def test_clumps(self, dev, name):
        for key, (shape, mask) in MASK_PATTERNS.items():
            for kind in ("masked", "nomask"):
                x, n = make(_data(np.float64, shape), mask if kind == "masked" else None, dev)
                with _ctx(key, kind):
                    rx, rn = getattr(XMA, name)(x), getattr(np.ma, name)(n)
                    assert isinstance(rx, list)
                    assert _slices(rx) == _slices(rn)

    def test_clumps_empty_and_plain(self, dev):
        for name in ("clump_masked", "clump_unmasked", "flatnotmasked_contiguous", "flatnotmasked_edges"):
            x, n = make(np.zeros(0), np.zeros(0, bool), dev)
            fx, fn = getattr(XMA, name), _ref(name)
            rx, rn = both(fx, fn, (x, n))
            if rn is not None:
                if isinstance(rn, list):
                    assert _slices(rx) == _slices(rn)

    def test_clumps_int_dtype_and_ndarray_input(self, dev):
        d = _data(np.int64, (10,))
        m = _RNG.random(10) < 0.5
        x, n = make(d, m, dev)
        assert _slices(XMA.clump_masked(x)) == _slices(np.ma.clump_masked(n))
        # masked data that is a view / slice
        assert _slices(XMA.clump_unmasked(x[2:8])) == _slices(np.ma.clump_unmasked(n[2:8]))
        assert _slices(XMA.clump_masked(x[::2])) == _slices(np.ma.clump_masked(n[::2]))

    def test_notmasked_contiguous_3d_error(self, dev):
        x, n = make(_data(np.float64, (2, 3, 4)), _rmask((2, 3, 4)), dev)
        both(XMA.notmasked_contiguous, np.ma.notmasked_contiguous, (x, n), axis=0)


def _slices(r):
    if isinstance(r, list):
        return [_slices(i) for i in r]
    if isinstance(r, slice):
        return (int(r.start), int(r.stop), r.step)
    return r


# ---------------------------------------------------------------------------
# vander / polyfit
# ---------------------------------------------------------------------------
class TestVanderPolyfit:
    @pytest.mark.parametrize("dtype", [np.int64, np.float64, np.float32])
    def test_vander(self, dev, dtype):
        d = _data(dtype, (6,), -3, 4)
        for kind in ("masked", "nomask"):
            x, n = make(d, _rmask((6,)), dev) if kind == "masked" else make(d, None, dev)
            for N in (None, 0, 1, 2, 5):
                with _ctx(dtype, kind, N):
                    rx, rn = both(XMA.vander, np.ma.vander, (x, n), n=N)
                    if rn is None:
                        continue
                    assert_same(rx, rn, dev=dev)

    def test_vander_bad(self, dev):
        x, n = make(_data(np.float64, (3, 3)), None, dev)
        both(XMA.vander, np.ma.vander, (x, n))
        x, n = make(_data(np.float64, (3,)), None, dev)
        both(XMA.vander, np.ma.vander, (x, n), n=-1)
        xp, npl = to_dev(np.arange(4.0), dev), np.arange(4.0)
        assert_same(XMA.vander(xp, 3), np.ma.vander(npl, 3), dev=dev)

    def test_polyfit_basic(self, dev):
        xs = np.linspace(-2, 2, 15)
        ys = 1.5 * xs ** 2 - xs + 0.3 + _rnd(np.float64, 15) * 0.05
        mx, my = _rmask((15,), 0.2), _rmask((15,), 0.2)
        for kx in (None, mx):
            for ky in (None, my):
                x, n = make(xs, kx, dev)
                y, ny = make(ys, ky, dev)
                for deg in (0, 1, 2, 3):
                    with _ctx(deg, kx is None, ky is None):
                        rx, rn = XMA.polyfit(x, y, deg), np.ma.polyfit(n, ny, deg)
                        assert on_dev(rx, dev)
                        np.testing.assert_allclose(host(rx), rn, rtol=1e-8, atol=1e-10)

    def test_polyfit_options(self, dev):
        xs = np.linspace(0, 3, 12)
        ys = np.stack([xs ** 2, 2 * xs + 1], 1) + _rnd(np.float64, (12, 2)) * 0.01
        ws = np.abs(_rnd(np.float64, 12)) + 0.5
        x, n = make(xs, _rmask((12,), 0.2), dev)
        y, ny = make(ys, _rmask((12, 2), 0.15), dev)
        w, nw = make(ws, _rmask((12,), 0.1), dev)
        rx, rn = XMA.polyfit(x, y, 2, w=w), np.ma.polyfit(n, ny, 2, w=nw)
        np.testing.assert_allclose(host(rx), rn, rtol=1e-8, atol=1e-10)
        rx, rn = XMA.polyfit(x, y[:, 0], 1, full=True), np.ma.polyfit(n, ny[:, 0], 1, full=True)
        assert len(rx) == len(rn) == 5
        for a, b in zip(rx, rn):
            np.testing.assert_allclose(host(a), b, rtol=1e-6, atol=1e-10)
        rx, rn = XMA.polyfit(x, y[:, 0], 1, cov=True), np.ma.polyfit(n, ny[:, 0], 1, cov=True)
        for a, b in zip(rx, rn):
            np.testing.assert_allclose(host(a), b, rtol=1e-6, atol=1e-10)
        rx, rn = XMA.polyfit(x, y[:, 0], 1, rcond=1e-3), np.ma.polyfit(n, ny[:, 0], 1, rcond=1e-3)
        np.testing.assert_allclose(host(rx), rn, rtol=1e-8, atol=1e-10)

    def test_polyfit_errors(self, dev):
        x, n = make(np.arange(6.0), None, dev)
        y3, ny3 = make(np.zeros((6, 2, 2)), None, dev)
        with pytest.raises(TypeError):
            np.ma.polyfit(n, ny3, 1)
        with pytest.raises(TypeError):
            XMA.polyfit(x, y3, 1)
        w2, nw2 = make(np.ones((6, 2)), None, dev)
        y, ny = make(np.arange(6.0) ** 2, None, dev)
        with pytest.raises(TypeError):
            np.ma.polyfit(n, ny, 1, w=nw2)
        with pytest.raises(TypeError):
            XMA.polyfit(x, y, 1, w=w2)
        ws, nws = make(np.ones(5), None, dev)
        with pytest.raises(TypeError):
            np.ma.polyfit(n, ny, 1, w=nws)
        with pytest.raises(TypeError):
            XMA.polyfit(x, y, 1, w=ws)
        # plain device arrays
        a = np.arange(8.0)
        rx = XMA.polyfit(to_dev(a, dev), to_dev(a * 2 + 1, dev), 1)
        np.testing.assert_allclose(host(rx), np.ma.polyfit(a, a * 2 + 1, 1), rtol=1e-8, atol=1e-10)
        assert on_dev(rx, dev)


# ---------------------------------------------------------------------------
# apply_along_axis / apply_over_axes
# ---------------------------------------------------------------------------
def _f_scalar_sum(v):
    return v.sum()


def _f_mean(v):
    return v.mean()


def _f_vec(v):
    return v * 2


def _f_pair(v):
    return np.ma.array([v.sum(), v.min()]) if isinstance(v, np.ma.MaskedArray) else XMA.masked_array([v.sum(), v.min()])


def _f_plain_sum(v):
    return np.sum(v.filled(0)) if hasattr(v, "filled") else np.sum(v)


def _f_arg(v, k, shift=0):
    return v[:k] + shift


class TestApply:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_along_axis(self, dev, dtype):
        for shape in SHAPES:
            data = _data(dtype, shape)
            mask = _rmask(shape)
            for kind in ("masked", "nomask", "allmasked"):
                x, n = _pair(data, mask, dev, kind)
                for axis in list(range(len(shape))) + [-1]:
                    for f in (_f_scalar_sum, _f_mean, _f_vec):
                        with _ctx(dtype, shape, kind, axis, f.__name__):
                            rx = XMA.apply_along_axis(f, axis, x)
                            rn = np.ma.apply_along_axis(f, axis, n)
                            assert_same(rx, rn, dev=dev, strict_nomask=False)

    def test_along_axis_args_kwargs(self, dev):
        x, n = make(_data(np.float64, (4, 6)), _rmask((4, 6)), dev)
        rx = XMA.apply_along_axis(_f_arg, 1, x, 3, shift=1.5)
        rn = np.ma.apply_along_axis(_f_arg, 1, n, 3, shift=1.5)
        assert_same(rx, rn, dev=dev, strict_nomask=False)

    def test_along_axis_plain_and_dtype_promotion(self, dev):
        a = _data(np.int64, (3, 4))
        rx = XMA.apply_along_axis(_f_scalar_sum, 0, to_dev(a, dev))
        rn = np.ma.apply_along_axis(_f_scalar_sum, 0, a)
        assert_same(rx, rn, dev=dev, strict_nomask=False)
        # results of different dtypes -> the largest one
        def f(v):
            return v.sum() if int(v.filled(0)[0]) % 2 else v.sum() * 1.5
        x, n = make(a, _rmask((3, 4)), dev)
        assert_same(XMA.apply_along_axis(f, 1, x), np.ma.apply_along_axis(f, 1, n), dev=dev, strict_nomask=False)

    def test_along_axis_errors(self, dev):
        x, n = make(_data(np.float64, (3, 4)), _rmask((3, 4)), dev)
        for axis in (2, -3):
            with pytest.raises(Exception) as ei:
                np.ma.apply_along_axis(_f_mean, axis, n)
            with pytest.raises(type(ei.value)):
                XMA.apply_along_axis(_f_mean, axis, x)
        # empty lanes: numpy raises ValueError
        x, n = make(np.zeros((0, 3)), None, dev)
        # no lane to apply func1d to: same exception type as numpy
        with pytest.raises(Exception) as ei:
            np.ma.apply_along_axis(_f_mean, 1, n)
        with pytest.raises(type(ei.value)):
            XMA.apply_along_axis(_f_mean, 1, x)

    def test_over_axes(self, dev):
        for shape in SHAPES:
            data = _data(np.float64, shape)
            for kind in ("masked", "nomask"):
                x, n = _pair(data, _rmask(shape), dev, kind)
                axes_list = [0, -1, [0], (0, shape and len(shape) - 1)]
                if len(shape) > 1:
                    axes_list += [[0, 1], (1, 0)]
                for axes in axes_list:
                    for f in ("sum", "mean", "max"):
                        with _ctx(shape, kind, axes, f):
                            rx = XMA.apply_over_axes(getattr(XMA, f), x, axes)
                            rn = np.ma.apply_over_axes(getattr(np.ma, f), n, axes)
                            assert_same(rx, rn, dev=dev, strict_nomask=False)

    def test_over_axes_all_masked_and_errors(self, dev):
        x, n = make(_data(np.float64, (3, 4)), np.ones((3, 4), bool), dev)
        rx = XMA.apply_over_axes(XMA.sum, x, [0, 1])
        rn = np.ma.apply_over_axes(np.ma.sum, n, [0, 1])
        assert_same(rx, rn, dev=dev, strict_nomask=False)
        # func collapsing dimensions wrongly
        with pytest.raises(ValueError):
            np.ma.apply_over_axes(lambda a, ax: np.ma.sum(a), n, [0])
        with pytest.raises(ValueError):
            XMA.apply_over_axes(lambda a, ax: XMA.sum(a), x, [0])
        x, n = make(_data(np.float64, (3, 4)), _rmask((3, 4)), dev)
        for axes in (5, [3]):
            with pytest.raises(Exception) as ei:
                np.ma.apply_over_axes(np.ma.sum, n, axes)
            with pytest.raises(type(ei.value)):
                XMA.apply_over_axes(XMA.sum, x, axes)


# ---------------------------------------------------------------------------
# cupy operands follow their own device, whatever the active backend
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not GPU_OK, reason="no usable GPU")
class TestCupyOperandsUnderCpuBackend:
    def _gpu_pair(self, shape=(4, 5), dtype=np.float64, mask=True):
        data = _data(dtype, shape)
        m = _rmask(shape) if mask else None
        with xupy.backend("gpu"):
            x = XMA.masked_array(cp.asarray(data), **({} if m is None else {"mask": cp.asarray(m)}))
        n = np.ma.masked_array(data, **({} if m is None else {"mask": m}))
        return x, n

    def test_results_stay_on_gpu(self):
        x, n = self._gpu_pair()
        y, ny = self._gpu_pair((6,))
        z, nz = self._gpu_pair((3,))
        with xupy.backend("cpu"):
            for name, args in [
                ("median", [(x, n)]), ("unique", [(x, n)]), ("anom", [(x, n)]),
                ("std", [(x, n)]), ("var", [(x, n)]),
                ("intersect1d", [(y, ny), (z, nz)]), ("union1d", [(y, ny), (z, nz)]),
                ("setdiff1d", [(y, ny), (z, nz)]), ("setxor1d", [(y, ny), (z, nz)]),
                ("in1d", [(y, ny), (z, nz)]), ("isin", [(y, ny), (z, nz)]),
                ("vander", [(y, ny)]),
            ]:
                fx, fn = getattr(XMA, name), _ref(name)
                kw = {"axis": 0} if name in ("median", "anom", "std", "var") else {}
                rx, rn = both(fx, fn, *args, **kw)
                with _ctx(name):
                    assert_same(rx, rn, dev="gpu", strict_nomask=False)
            rx = XMA.cov(x)
            assert isinstance(rx.data, cp.ndarray)
            rx = XMA.corrcoef(x)
            assert isinstance(rx.data, cp.ndarray)
            e = XMA.flatnotmasked_edges(x)
            assert isinstance(e, cp.ndarray)
            ra = XMA.apply_along_axis(_f_scalar_sum, 0, x)
            assert isinstance(ra.data, cp.ndarray)
            p = XMA.polyfit(y, y, 1)
            assert isinstance(p, cp.ndarray)

    def test_plain_cupy_input_under_cpu_backend(self):
        a = cp.asarray(_data(np.float64, (4, 5)))
        with xupy.backend("cpu"):
            r = XMA.median(a, axis=0)
            assert on_dev(r.data if isinstance(r, XMA.MaskedArray) else r, "gpu")
            u = XMA.unique(a)
            assert isinstance(u.data, cp.ndarray)
            v = XMA.vander(a[0], 3)
            assert isinstance(v, cp.ndarray)
