"""
Differential tests of ``xupy.ma`` against ``numpy.ma`` (numpy 2.5):

* A. ``repr`` / ``str`` / ``format`` parity (incl. summarisation, large arrays,
  no host copy of big GPU arrays);
* B. the functions of ``xupy.ma.extras`` versus ``numpy.ma.extras``;
* C. function-form helpers of the core namespace
  (getmask, getdata, mask_or, concatenate, where, ...).

numpy.ma is the reference; its quirks are mirrored on purpose.
"""
# NO-SYNC: numpy.ma shrinks the mask to nomask via a host-side .any(); xupy.ma never syncs
# (DESIGN decision 6), so a result mask may be a full all-False array where numpy.ma has
# nomask.  Calls that pass `strict_nomask=False` accept exactly that difference (see NO-SYNC note).

import contextlib
import time

import numpy as np
import pytest

import xupy
from ._ma_parity_helpers import (
    XMA, GPU_OK, cp, make, assert_same, both, to_dev, host, on_dev, np_repr, NP_LT_25,
)

NOMASK = np.ma.nomask


def _ref(name):
    fn = getattr(np.ma, name, None)
    return fn if fn is not None else getattr(np.ma.extras, name)


# ---------------------------------------------------------------------------
# A. repr / str / format
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(20250607)


def _rand(shape, dtype=np.float64):
    r = _RNG.standard_normal(shape) * 100
    if np.dtype(dtype).kind == "c":
        return (r + 1j * _RNG.standard_normal(shape)).astype(dtype)
    if np.dtype(dtype).kind == "b":
        return r > 0
    if np.dtype(dtype).kind in "iu":
        return np.abs(r).astype(dtype) % 100
    return r.astype(dtype)


def _rmask(shape, p=0.3):
    return _RNG.random(shape) < p


def _arange(shape, dtype=np.float64):
    return (np.arange(int(np.prod(shape))) * 1.5 - 7).reshape(shape).astype(dtype)


_SPECIAL = np.array([-1.5, np.nan, np.inf, -np.inf, 1e-300, 1e300, 1e-8, 123456789.123,
                     0.0, -0.0, 1e16, 3.0], dtype=np.float64)

# id -> (data, mask, kwargs)
_REPR_CASES = {
    "0d_float": (np.float64(3.5), None, {}),
    "0d_masked": (np.float64(3.5), True, {}),
    "0d_unmasked_mask": (np.float64(3.5), False, {}),
    "0d_int": (np.int64(7), True, {}),
    "1d_f64": (_rand(7), _rmask(7), {}),
    "1d_f32": (_rand(7, np.float32), _rmask(7), {}),
    "1d_i8": (_rand(9, np.int8), _rmask(9), {}),
    "1d_i64": (_rand(9, np.int64), _rmask(9), {}),
    "1d_u8": (_rand(9, np.uint8), _rmask(9), {}),
    "1d_bool": (_rand(6, np.bool_), _rmask(6), {}),
    "1d_c128": (_rand(5, np.complex128), _rmask(5), {}),
    "1d_nomask_none": (_rand(6), None, {}),
    "1d_nomask_false": (_rand(6), np.zeros(6, bool), {}),
    "1d_all_masked": (_rand(5), np.ones(5, bool), {}),
    "1d_all_masked_int": (_rand(5, np.int64), np.ones(5, bool), {}),
    "1d_fill_float": (_rand(5), _rmask(5), {"fill_value": -9.5}),
    "1d_fill_int": (_rand(5, np.int64), _rmask(5), {"fill_value": 77}),
    "1d_fill_bool": (_rand(5, np.bool_), _rmask(5), {"fill_value": True}),
    "1d_hard": (_rand(6), _rmask(6), {"hard_mask": True}),
    "1d_hard_nomask": (_rand(6), None, {"hard_mask": True}),
    "1d_special": (_SPECIAL, np.arange(12) % 5 == 0, {}),
    "1d_special_nomask": (_SPECIAL, None, {}),
    "1d_neg_int": (np.array([-5, 10, -300, 4]), np.array([0, 1, 0, 0], bool), {}),
    "1d_long_wrap": (_arange((100,)), _rmask(100), {}),
    "1d_long_wrap_int": (_arange((100,), np.int64), _rmask(100), {}),
    "2d_f64": (_rand((3, 4)), _rmask((3, 4)), {}),
    "2d_i8_fill": (_rand((3, 4), np.int8), _rmask((3, 4)), {"fill_value": 5}),
    "2d_wide": (_arange((4, 30)), _rmask((4, 30)), {}),
    "2d_nomask": (_rand((3, 4)), None, {}),
    "2d_one_row": (_rand((1, 6)), _rmask((1, 6)), {}),
    "2d_one_col": (_rand((6, 1)), _rmask((6, 1)), {}),
    "2d_all_masked": (_rand((2, 3)), np.ones((2, 3), bool), {}),
    "2d_c128": (_rand((2, 3), np.complex128), _rmask((2, 3)), {}),
    "3d_f64": (_rand((2, 3, 4)), _rmask((2, 3, 4)), {}),
    "3d_bool": (_rand((2, 3, 2), np.bool_), _rmask((2, 3, 2)), {}),
    "4d_f64": (_rand((2, 2, 3, 2)), _rmask((2, 2, 3, 2)), {}),
    "4d_hard": (_rand((2, 2, 2, 2), np.int64), _rmask((2, 2, 2, 2)), {"hard_mask": True}),
    "size0_1d": (np.zeros(0), None, {}),
    "size0_1d_mask": (np.zeros(0), np.zeros(0, bool), {}),
    "size0_2d": (np.zeros((0, 3)), None, {}),
    "size0_2d_b": (np.zeros((2, 0), np.int64), np.zeros((2, 0), bool), {}),
    "size0_3d": (np.zeros((2, 0, 3), np.float32), None, {}),
}

_SUMMARY_CASES = {
    "1d": (_arange((60,)), _rmask((60,), 0.4), {}),
    "1d_edge_masked": (_arange((60,)), np.arange(60) % 7 == 0, {}),
    "1d_int": (_arange((60,), np.int64), _rmask((60,), 0.4), {}),
    "2d": (_arange((9, 9)), _rmask((9, 9), 0.4), {}),
    "2d_edge_masked": (_arange((9, 9)), np.indices((9, 9)).sum(0) % 4 == 0, {}),
    "3d": (_arange((5, 6, 7)), _rmask((5, 6, 7), 0.4), {}),
    "3d_c128": (_arange((4, 5, 6), np.complex128), _rmask((4, 5, 6), 0.4), {}),
    "4d": (_arange((4, 4, 5, 4)), _rmask((4, 4, 5, 4), 0.4), {}),
    "2d_fill": (_arange((9, 9), np.int8), _rmask((9, 9), 0.4), {"fill_value": 3}),
    "2d_nomask": (_arange((9, 9)), None, {}),
    "2d_all_masked": (_arange((9, 9)), np.ones((9, 9), bool), {}),
    "2d_hard": (_arange((9, 9)), _rmask((9, 9)), {"hard_mask": True}),
}

_OPTIONS = {
    "default": {},
    "thr_edge": {"threshold": 10, "edgeitems": 2},
    "thr_edge1": {"threshold": 5, "edgeitems": 1},
    "thr_edge5": {"threshold": 20, "edgeitems": 5},
    "linewidth": {"linewidth": 30},
    "linewidth_wide": {"linewidth": 200},
    "precision": {"precision": 2},
    "precision0": {"precision": 0},
    "combo": {"threshold": 12, "edgeitems": 3, "linewidth": 40, "precision": 3},
    "thr_high": {"threshold": 100000},
    "fixed": {"floatmode": "fixed", "precision": 3},
    "sign": {"sign": "+"},
    "suppress": {"suppress": True},
    "legacy": {"legacy": "1.25"},
}


def _check_repr(x, n):
    rn, rx = np_repr(n), repr(x)
    assert rx == rn
    assert str(x) == str(n)
    assert format(x, "") == format(n, "")
    # explicit properties (checked on the numpy reference first)
    for s in (rn, rx):
        assert "'--'" not in s and '"--"' not in s
    if n.mask is NOMASK and n.ndim > 0:
        assert "mask=False" in rn and "mask=False" in rx
    assert ("fill_value=" in rn) == ("fill_value=" in rx)
    if n.ndim > 0 or n.mask is not NOMASK:
        assert "fill_value=" in rn and "fill_value=" in rx




@contextlib.contextmanager
def _note(*label):
    """Tag any failure raised inside a loop body with the case that triggered it."""
    try:
        yield
    except BaseException as e:  # noqa: BLE001
        note = f"case: {label!r}"
        if hasattr(e, "add_note"):      # BaseException.add_note is Python 3.11+
            e.add_note(note)
        else:
            e.args = (f"{e.args[0] if e.args else ''}\n{note}",) + tuple(e.args[1:])
        raise


_ALL_CASES = list(_REPR_CASES)
_1D_DTYPES = ["1d_f64", "1d_f32", "1d_i8", "1d_i64", "1d_u8", "1d_bool", "1d_c128"]
_DEFAULT_GROUPS = {
    "0d": [c for c in _ALL_CASES if c.startswith("0d_")],
    "1d_dtypes": _1D_DTYPES,
    "1d_other": [c for c in _ALL_CASES if c.startswith("1d_") and c not in _1D_DTYPES],
    "2d": [c for c in _ALL_CASES if c.startswith("2d_")],
    "3d_4d": [c for c in _ALL_CASES if c[:2] in ("3d", "4d")],
    "size0": [c for c in _ALL_CASES if c.startswith("size0")],
}
# option sets chosen per shape class (ndim of the summary case)
_OPTS_BY_NDIM = {
    1: ["default", "thr_edge", "thr_edge1", "linewidth", "precision0", "combo", "fixed", "legacy"],
    2: ["default", "thr_edge", "thr_edge5", "linewidth_wide", "precision", "combo", "sign",
        "thr_high"],
    3: ["default", "thr_edge", "thr_edge1", "combo", "thr_high", "suppress"],
    4: ["default", "thr_edge", "thr_edge1", "combo", "thr_high", "suppress"],
}
_SMALL_OPTS = ["thr_edge", "combo", "linewidth", "precision"]


class TestReprParity:
    @pytest.mark.parametrize("group", list(_DEFAULT_GROUPS))
    def test_default_options(self, dev, group):
        for case in _DEFAULT_GROUPS[group]:
            data, mask, kw = _REPR_CASES[case]
            x, n = make(data, mask, dev, **kw)
            with _note(case):
                _check_repr(x, n)

    @pytest.mark.parametrize("case", list(_SUMMARY_CASES))
    def test_printoptions(self, dev, case):
        data, mask, kw = _SUMMARY_CASES[case]
        x, n = make(data, mask, dev, **kw)
        for opt in _OPTS_BY_NDIM[np.ndim(data)]:
            with _note(case, opt), np.printoptions(**_OPTIONS[opt]):
                _check_repr(x, n)

    @pytest.mark.parametrize("case", ["0d_float", "1d_f64", "1d_special", "1d_c128",
                                      "2d_wide", "3d_f64", "size0_2d", "1d_all_masked"])
    def test_printoptions_on_small_cases(self, dev, case):
        data, mask, kw = _REPR_CASES[case]
        x, n = make(data, mask, dev, **kw)
        for opt in _SMALL_OPTS:
            with _note(case, opt), np.printoptions(**_OPTIONS[opt]):
                _check_repr(x, n)

    def test_summarisation_ellipsis_present(self, dev):
        for shp in [(60,), (9, 9), (5, 6, 7)]:
            x, n = make(_arange(shp), _rmask(shp), dev)
            with np.printoptions(threshold=10, edgeitems=2):
                assert "..." in repr(n) and "..." in repr(x)
                assert "..." in str(n) and "..." in str(x)

    def test_format_spec_on_arrays_matches(self, dev):
        x, n = make(_rand((3,)), _rmask((3,)), dev)
        for spec in ["", ">10", "<12", "^14", "10"]:
            with _note(spec):
                rx, rn = both(lambda a: format(a, spec), lambda a: format(a, spec), (x, n))
                if rn is not None:
                    assert rx == rn

    def test_repr_after_state_changes(self, dev):
        x, n = make(_rand((4, 3)), _rmask((4, 3)), dev)
        for a in (x, n):
            a.harden_mask()
            a.fill_value = 7.25
        assert repr(x) == repr(n) and str(x) == str(n)
        for a in (x, n):
            a[0, 0] = XMA.masked if a is x else np.ma.masked
        assert repr(x) == repr(n) and str(x) == str(n)

    def test_view_slices_and_reshape_repr(self, dev):
        for dt in [np.float32, np.float64, np.int8, np.int64, np.uint8, np.bool_,
                   np.complex128]:
            x, n = make(_rand((4, 6), dt), _rmask((4, 6)), dev)
            with _note(dt):
                for sl in (np.s_[1:3], np.s_[:, ::2], np.s_[::-1, 1], np.s_[0, :]):
                    assert repr(x[sl]) == np_repr(n[sl])
                    assert str(x[sl]) == str(n[sl])
                assert repr(x.reshape(2, 12)) == np_repr(n.reshape(2, 12))

    def test_result_of_operations_repr(self, dev):
        x, n = make(_rand((3, 3)), _rmask((3, 3)), dev)
        assert repr(x + 1) == repr(n + 1)
        assert repr(x * x) == repr(n * n)
        assert repr(x.astype(np.float32)) == repr(n.astype(np.float32))
        assert str(-x) == str(-n)

    def test_masked_singleton_repr_str_format(self):
        assert repr(XMA.masked) == repr(np.ma.masked) == "masked"
        assert str(XMA.masked) == str(np.ma.masked) == "--"
        assert format(XMA.masked, "") == format(np.ma.masked, "")
        # DESIGN.md decision 3 pins the repr of XuPy's nomask to "False"; numpy 2.5 prints "np.False_"
        assert repr(XMA.nomask) == "False"
        assert str(XMA.nomask) == str(NOMASK)

    def test_unmasked_str_has_dashes_not_quotes(self, dev):
        x, n = make(np.arange(4.0), np.array([0, 1, 0, 1], bool), dev)
        assert str(x) == str(n) == "[0.0 -- 2.0 --]"
        assert repr(x) == repr(n)


def _big_pair(shape, dtype, dev, p=0.1, seed=3):
    rng = np.random.default_rng(seed)
    d = rng.random(shape).astype(dtype)
    m = rng.random(shape) < p
    # make sure edges contain masked values
    m.flat[0] = True
    m.flat[-1] = True
    return make(d, m, dev), d, m


class TestReprLargeArrays:
    # the only moderately large repr test of the default run (9e6 float32 elements)
    def test_large_matches_numpy(self, dev):
        (x, n), _, _ = _big_pair((3000, 3000), np.float32, dev)
        assert repr(x) == repr(n)
        assert str(x) == str(n)
        assert "..." in repr(x)

    @pytest.mark.slow
    @pytest.mark.parametrize("shape,dtype", [((3000, 3000), np.float64),
                                             ((200, 200, 200), np.float32),
                                             ((1_000_000,), np.int64)])
    def test_large_more_shapes_match_numpy(self, dev, shape, dtype):
        (x, n), _, _ = _big_pair(shape, dtype, dev)
        assert repr(x) == repr(n)
        assert str(x) == str(n)
        assert "..." in repr(x)

    @pytest.mark.slow
    @pytest.mark.parametrize("shape", [(3000, 3000), (200, 200, 200)])
    def test_large_fully_masked_and_nomask(self, dev, shape):
        d = np.zeros(shape, np.float32)
        x, n = make(d, np.ones(shape, bool), dev)
        assert repr(x) == repr(n) and str(x) == str(n)
        x, n = make(d, None, dev)
        assert repr(x) == repr(n) and str(x) == str(n)

    @pytest.mark.slow
    def test_large_with_custom_options(self, dev):
        (x, n), _, _ = _big_pair((3000, 3000), np.float64, dev)
        with np.printoptions(edgeitems=1, precision=2, linewidth=50):
            assert repr(x) == repr(n)
        with np.printoptions(threshold=10**7, edgeitems=2):
            assert str(x[:20, :20]) == str(n[:20, :20])

    @pytest.mark.slow
    def test_huge_1d_5e7(self, dev):
        (x, n), _, _ = _big_pair((50_000_000,), np.float64, dev, seed=11)
        assert repr(x) == repr(n)
        assert str(x) == str(n)
        del x, n


@pytest.mark.skipif(not GPU_OK, reason="no usable GPU")
class TestReprNoHostCopy:
    LIMIT = 5000

    @pytest.fixture
    def transfers(self, monkeypatch):
        rec = []
        orig_get = cp.ndarray.get
        orig_asnumpy = cp.asnumpy

        def get(self, *a, **k):
            rec.append(("get", int(self.size)))
            return orig_get(self, *a, **k)

        def asnumpy(a, *args, **k):
            rec.append(("asnumpy", int(getattr(a, "size", 0))))
            return orig_asnumpy(a, *args, **k)

        monkeypatch.setattr(cp.ndarray, "get", get)
        monkeypatch.setattr(cp, "asnumpy", asnumpy)
        return rec

    def test_recorder_intercepts_transfers(self, transfers):
        big = cp.zeros(10000)
        big.get()
        cp.asnumpy(big)
        assert all(sz == 10000 for _, sz in transfers)
        assert any(k == "get" for k, _ in transfers)
        assert any(k == "asnumpy" for k, _ in transfers)

    @pytest.mark.parametrize("shape,dtype", [
        pytest.param((2000, 2000), np.float64),
        pytest.param((3000, 3000), np.float64, marks=pytest.mark.slow),
        pytest.param((200, 200, 200), np.float32, marks=pytest.mark.slow),
        pytest.param((4_000_000,), np.float64, marks=pytest.mark.slow)])
    def test_no_large_transfer_and_fast(self, shape, dtype, transfers):
        (x, n), _, _ = _big_pair(shape, dtype, "gpu")
        for fn in (repr, str):
            fn(make(np.zeros(3), np.array([0, 1, 0], bool), "gpu")[0])  # warm up
            transfers.clear()
            t0 = time.perf_counter()
            out = fn(x)
            dt = time.perf_counter() - t0
            big = [t for t in transfers if t[1] > self.LIMIT]
            assert not big, f"large host transfer during {fn.__name__}: {big}"
            assert dt < 2.0, f"{fn.__name__} took {dt:.2f}s"
            assert out == fn(n)

    def test_no_large_transfer_with_printoptions(self, transfers):
        (x, n), _, _ = _big_pair((1000, 1000), np.float32, "gpu")
        transfers.clear()
        with np.printoptions(threshold=100, edgeitems=2, precision=2):
            out = repr(x)
        assert not [t for t in transfers if t[1] > self.LIMIT]
        with np.printoptions(threshold=100, edgeitems=2, precision=2):
            assert out == repr(n)


# ---------------------------------------------------------------------------
# B. extras
# ---------------------------------------------------------------------------
_A2 = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
_M2 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]], bool)
_A23 = np.arange(6.).reshape(2, 3)
_M23 = np.array([[0, 0, 1], [0, 0, 0]], bool)
_A3 = np.arange(24.).reshape(2, 3, 4)
_M3 = _RNG.random((2, 3, 4)) < 0.15
_A1 = np.array([1., 2., 3., 4.])
_M1 = np.array([0, 1, 0, 0], bool)

_KINDS = ["masked", "nomask", "mask_false", "plain", "list", "npma"]


def _inp(kind, data, mask, dev):
    """-> (xupy_arg, numpy_arg, device_to_check)."""
    data = np.asarray(data)
    if kind == "masked":
        x, n = make(data, mask, dev)
        return x, n, dev
    if kind == "nomask":
        x, n = make(data, None, dev)
        return x, n, dev
    if kind == "mask_false":
        x, n = make(data, np.zeros(data.shape, bool), dev)
        return x, n, dev
    if kind == "plain":
        return to_dev(data, dev), data.copy(), dev
    if kind == "list":
        return data.tolist(), data.tolist(), None
    if kind == "npma":
        a = np.ma.masked_array(data.copy(), mask=mask)
        return a, a.copy(), "cpu"
    raise AssertionError(kind)


def _cmp(name, args, dev, check_dev=True, strict_nomask=True, **kw):
    """Call the XuPy function and the numpy one on ``args`` [(x, n), ...]."""
    fx, fn = getattr(XMA, name), _ref(name)
    rx, rn = both(fx, fn, *args, **kw)
    if rn is None and rx is None:
        return None
    assert_same(rx, rn, dev=dev if check_dev else None, strict_nomask=strict_nomask,
                ctx=name)
    return rx




class TestIssequenceCountMasked:
    def test_issequence_host(self):
        for obj in [[1], (1,), np.arange(3), 3, "ab", None, {1: 2}, 2.5, {1}]:
            with _note(obj):
                assert XMA.issequence(obj) == np.ma.extras.issequence(obj)
                assert isinstance(XMA.issequence(obj), bool)

    def test_issequence_masked_and_cupy(self, dev):
        x, n = make(_A1, _M1, dev)
        assert XMA.issequence(x) == np.ma.extras.issequence(n)
        assert XMA.issequence(to_dev(_A1, dev)) is True

    @pytest.mark.parametrize("kind", _KINDS)
    def test_count_masked_2d(self, dev, kind):
        x, n, d = _inp(kind, _A2, _M2, dev)
        for axis in [None, 0, 1, -1, -2]:
            with _note(kind, axis):
                r = _cmp("count_masked", [(x, n)], d, axis=axis)
                if axis is None and r is not None:
                    assert type(r) is type(np.ma.count_masked(n)) is np.int64

    def test_count_masked_1d(self, dev):
        x, n = make(_A1, _M1, dev)
        r = _cmp("count_masked", [(x, n)], dev, axis=None)
        assert type(r) is np.int64
        _cmp("count_masked", [(x, n)], dev, axis=0)

    def test_count_masked_3d(self, dev):
        x, n = make(_A3, _M3, dev)
        for axis in [None, 0, 1, 2, -1, (0, 1)]:
            with _note(axis):
                _cmp("count_masked", [(x, n)], dev, axis=axis)

    def test_count_masked_bad_axis(self, dev):
        for axis in [3, -4, 5]:
            for mask in (_M2, None):
                x, n = make(_A2, mask, dev)
                with _note(axis, mask is None):
                    _cmp("count_masked", [(x, n)], dev, axis=axis)

    def test_count_masked_scalar_and_singleton(self):
        _cmp("count_masked", [(3, 3)], None)
        _cmp("count_masked", [(XMA.masked, np.ma.masked)], None)

    def test_count_masked_empty(self, dev):
        x, n = make(np.zeros((0, 3)), None, dev)
        for ax in (None, 0, 1):
            with _note(ax):
                _cmp("count_masked", [(x, n)], dev, axis=ax)

    def test_count_masked_all_masked(self, dev):
        x, n = make(_A2, np.ones((3, 3), bool), dev)
        r = _cmp("count_masked", [(x, n)], dev)
        assert r == 9


class TestMaskedAll:
    def test_default_dtype(self):
        for shape in [3, (3,), (2, 3), (2, 3, 4), (0,), (2, 0), (), (1, 1), 0]:
            with _note(shape):
                r = _cmp("masked_all", [(shape, shape)], None)
                assert r is None or r.dtype == np.float64

    def test_dtype(self):
        for dtype in [np.float32, np.float64, np.int8, np.int64, np.uint8, np.bool_,
                      np.complex128, "f4", int, float, complex]:
            for shape in [(4,), (2, 3)]:
                with _note(dtype, shape):
                    _cmp("masked_all", [(shape, shape)], None, dtype=dtype)

    def test_dtype_positional(self):
        r = XMA.masked_all((2, 2), np.int16)
        assert_same(r, np.ma.masked_all((2, 2), np.int16), strict_nomask=True)

    def test_all_masked_and_data_zero_not_required(self):
        r = XMA.masked_all((3, 2))
        assert bool(np.all(host(XMA.getmaskarray(r))))
        assert r.mask is not XMA.nomask

    def test_negative_shape(self):
        _cmp("masked_all", [((-1,), (-1,))], None)

    def test_invalid_dtype(self):
        _cmp("masked_all", [((2,), (2,))], None, dtype="notadtype")

    @pytest.mark.parametrize("kind", ["masked", "nomask", "plain", "list", "npma"])
    def test_masked_all_like(self, dev, kind):
        for dt in [np.float64, np.int32, np.bool_, np.complex64]:
            x, n, d = _inp(kind, _A23.astype(dt), _M23, dev)
            # host input (plain numpy, numpy.ma) goes to the active backend
            d = dev if kind == "npma" else d
            with _note(kind, dt), xupy.backend(dev):
                _cmp("masked_all_like", [(x, n)], d)

    def test_masked_all_like_3d_and_empty(self, dev):
        x, n = make(_A3, _M3, dev)
        _cmp("masked_all_like", [(x, n)], dev)
        x, n = make(np.zeros((0, 2)), None, dev)
        _cmp("masked_all_like", [(x, n)], dev)

    def test_masked_all_like_0d(self, dev):
        x, n = make(np.float64(2.0), None, dev)
        _cmp("masked_all_like", [(x, n)], dev)


def _arr_2d_args(kind, dev, data=_A2, mask=_M2):
    x, n, d = _inp(kind, data, mask, dev)
    return [(x, n)], d


_COMPRESS = ["compress_rows", "compress_cols", "compress_rowcols"]
_MASKROWCOLS = ["mask_rows", "mask_cols", "mask_rowcols"]


class TestCompress:
    @pytest.mark.parametrize("kind", _KINDS)
    def test_2d_simple(self, dev, kind):
        args, d = _arr_2d_args(kind, dev)
        for name in _COMPRESS:
            with _note(kind, name):
                _cmp(name, args, d)

    @pytest.mark.parametrize("kind", ["masked", "nomask", "plain"])
    def test_compress_rowcols_axis(self, dev, kind):
        args, d = _arr_2d_args(kind, dev)
        for axis in [None, 0, 1, -1, -2, 2, 3]:
            with _note(kind, axis):
                _cmp("compress_rowcols", args, d, axis=axis)

    def test_all_masked(self, dev):
        x, n = make(_A2, np.ones((3, 3), bool), dev)
        for name in _COMPRESS:
            with _note(name):
                _cmp(name, [(x, n)], dev)

    def test_wrong_ndim(self, dev):
        for data in (_A1, _A3):
            x, n = make(data, np.zeros(data.shape, bool), dev)
            for name in _COMPRESS:
                with _note(name, data.ndim):
                    _cmp(name, [(x, n)], dev)

    def test_rectangular(self, dev):
        x, n = make(_A23, _M23, dev)
        for nm in _COMPRESS:
            with _note(nm):
                _cmp(nm, [(x, n)], dev)

    @pytest.mark.parametrize("kind", _KINDS)
    def test_compress_nd_3d(self, dev, kind):
        x, n, d = _inp(kind, _A3, _M3, dev)
        for axis in [None, 0, 1, 2, -1, (0, 1), (0, 2), (1, 2), (0, 1, 2), (), -3]:
            with _note(kind, axis):
                _cmp("compress_nd", [(x, n)], d, axis=axis)

    def test_compress_nd_2d(self, dev):
        x, n = make(_A2, _M2, dev)
        for axis in [None, 0, 1, (0, 1)]:
            with _note(axis):
                _cmp("compress_nd", [(x, n)], dev, axis=axis)

    def test_compress_nd_1d(self, dev):
        x, n = make(_A1, _M1, dev)
        _cmp("compress_nd", [(x, n)], dev)
        _cmp("compress_nd", [(x, n)], dev, axis=0)

    def test_compress_nd_bad_axis(self, dev):
        x, n = make(_A3, _M3, dev)
        for axis in [3, -4, (0, 3), (0, 0), (5,), 10]:
            with _note(axis):
                _cmp("compress_nd", [(x, n)], dev, axis=axis)

    def test_compress_nd_bad_axis_is_axis_error(self, dev):
        x, _ = make(_A3, _M3, dev)
        with pytest.raises(np.exceptions.AxisError):
            XMA.compress_nd(x, axis=3)
        x2, _ = make(_A2, _M2, dev)
        with pytest.raises(np.exceptions.AxisError):
            XMA.compress_rowcols(x2, axis=2)

    def test_compress_nd_repeated_axis_value_error(self, dev):
        x, _ = make(_A3, _M3, dev)
        with pytest.raises(ValueError):
            XMA.compress_nd(x, axis=(0, 0))

    def test_compress_nd_result_is_plain_array(self, dev):
        x, _ = make(_A3, _M3, dev)
        r = XMA.compress_nd(x)
        assert not isinstance(r, XMA.MaskedArray) and on_dev(r, dev)

    def test_input_not_modified(self, dev):
        x, n = make(_A2, _M2, dev)
        before = host(XMA.getmaskarray(x)).copy()
        XMA.compress_rowcols(x)
        XMA.compress_nd(x)
        np.testing.assert_array_equal(host(XMA.getmaskarray(x)), before)


class TestMaskRowcols:
    @pytest.mark.parametrize("kind", _KINDS)
    def test_2d(self, dev, kind):
        args, d = _arr_2d_args(kind, dev)
        for name in _MASKROWCOLS:
            with _note(kind, name):
                _cmp(name, args, d)

    @pytest.mark.parametrize("kind", ["masked", "nomask", "plain"])
    def test_mask_rowcols_axis(self, dev, kind):
        args, d = _arr_2d_args(kind, dev)
        for axis in [None, 0, 1, -1, -2, 2, -3, 3]:
            with _note(kind, axis):
                _cmp("mask_rowcols", args, d, axis=axis)

    def test_bad_axis_matches_numpy_behaviour(self, dev):
        # numpy's mask_rowcols does not validate the axis eagerly; mirror it
        x, n = make(_A2, _M2, dev)
        _cmp("mask_rowcols", [(x, n)], dev, axis=3)
        _cmp("mask_rowcols", [(x, n)], dev, axis=-4)

    def test_wrong_ndim(self, dev):
        for data in (_A1, _A3, np.asarray(np.float64(2.0))):
            x, n = make(data, np.zeros(data.shape, bool), dev)
            for name in _MASKROWCOLS:
                with _note(name, data.ndim):
                    _cmp(name, [(x, n)], dev)

    def test_rectangular_and_empty(self, dev):
        for name in _MASKROWCOLS:
            x, n = make(_A23, _M23, dev)
            with _note(name):
                _cmp(name, [(x, n)], dev)
                x, n = make(np.zeros((0, 3)), None, dev)
                _cmp(name, [(x, n)], dev)
                x, n = make(np.zeros((2, 0)), np.zeros((2, 0), bool), dev)
                _cmp(name, [(x, n)], dev)

    def test_preserves_fill_value_and_dtype(self, dev):
        x, n = make(_A2.astype(np.int16), _M2, dev, fill_value=33)
        for name in _MASKROWCOLS:
            with _note(name):
                _cmp(name, [(x, n)], dev)

    def test_input_not_modified(self, dev):
        x, n = make(_A2, _M2, dev)
        before = host(XMA.getmaskarray(x)).copy()
        for name in _MASKROWCOLS:
            r = getattr(XMA, name)(x)
            np.testing.assert_array_equal(host(XMA.getmaskarray(x)), before)
            assert r is not x

    def test_hardmask_input(self, dev):
        x, n = make(_A2, _M2, dev, hard_mask=True)
        for name in _MASKROWCOLS:
            with _note(name):
                _cmp(name, [(x, n)], dev)


class TestFlattenInplace:
    def test_list(self):
        import copy
        for seq in ([1, [2, [3, 4]], 5], [], [[]], [1, 2], [[1], [2], [[3]]],
                    [(1, 2), [3]], [[[[1]]]]):
            a, b = copy.deepcopy(seq), copy.deepcopy(seq)
            with _note(seq):
                try:
                    rn = np.ma.extras.flatten_inplace(b)
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        XMA.flatten_inplace(a)
                    continue
                rx = XMA.flatten_inplace(a)
                assert rx == rn and a == b
                assert rx is a  # in place, returns the same object

    def test_non_sequence(self):
        for obj in (3, None):
            try:
                np.ma.extras.flatten_inplace(obj)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    XMA.flatten_inplace(obj)


class TestLikeFunctions:
    @pytest.mark.parametrize("kind", _KINDS)
    def test_kinds(self, dev, kind):
        x, n, d = _inp(kind, _A23, _M23, dev)
        d = dev if kind == "npma" else d  # host input goes to the active backend
        for name in ("zeros_like", "ones_like"):
            with _note(kind, name), xupy.backend(dev):
                assert_same(getattr(XMA, name)(x), getattr(np.ma, name)(n), dev=d)

    def test_dtype(self, dev):
        x, n = make(_A23, _M23, dev)
        for name in ("zeros_like", "ones_like"):
            for dtype in [None, np.int32, np.float32, np.bool_, np.complex128]:
                with _note(name, dtype):
                    _cmp(name, [(x, n)], dev, dtype=dtype)

    def test_shape_and_order(self, dev):
        x, n = make(_A23, _M23, dev)
        for name in ("zeros_like", "ones_like"):
            with _note(name):
                rx, rn = both(getattr(XMA, name), getattr(np.ma, name), (x, n), shape=(3, 2))
                if rn is not None:
                    assert_same(rx, rn, dev=dev)
                _cmp(name, [(x, n)], dev, order="F")

    def test_int_dtype_keeps_fill_value(self, dev):
        x, n = make(_A2.astype(np.int8), _M2, dev, fill_value=9)
        for name in ("zeros_like", "ones_like"):
            with _note(name):
                _cmp(name, [(x, n)], dev)

    def test_lists_and_scalars(self):
        for name in ("zeros_like", "ones_like"):
            with _note(name):
                _cmp(name, [([1, 2, 3], [1, 2, 3])], None)
                _cmp(name, [(2.5, 2.5)], None)

    def test_empty_like_shape_dtype_mask(self, dev):
        for kind in _KINDS:
            x, n, d = _inp(kind, _A23, _M23, dev)
            d = dev if kind == "npma" else d  # host input goes to the active backend
            with _note(kind), xupy.backend(dev):
                r = XMA.empty_like(x)
                rn = np.ma.empty_like(n)
                assert isinstance(r, XMA.MaskedArray)
                assert r.shape == rn.shape and r.dtype == rn.dtype
                np.testing.assert_array_equal(host(XMA.getmaskarray(r)),
                                              np.ma.getmaskarray(rn))
                assert (r.mask is XMA.nomask) == (rn.mask is NOMASK)
                if d is not None:
                    assert on_dev(r.data, d)

    def test_empty_like_dtype(self, dev):
        x, n = make(_A23, _M23, dev)
        r = XMA.empty_like(x, dtype=np.int16)
        assert r.dtype == np.int16 == np.ma.empty_like(n, dtype=np.int16).dtype

    def test_zeros_ones_empty_input_untouched(self, dev):
        x, n = make(_A23, _M23, dev)
        XMA.zeros_like(x)
        XMA.ones_like(x)
        assert_same(x, n, dev=dev)


_ATLEAST_DATA = [(np.float64(3.0), np.True_), (np.float64(3.0), np.False_),
                 (_A1, _M1), (_A23, _M23), (_A3, _M3),
                 (np.zeros((0,)), np.zeros((0,), bool)),
                 (np.zeros((0, 2)), np.zeros((0, 2), bool)),
                 (np.zeros((4, 1, 0, 2)), np.zeros((4, 1, 0, 2), bool))]
_ATLEAST = ["atleast_1d", "atleast_2d", "atleast_3d"]


class TestAtleast:
    @pytest.mark.parametrize("kind", ["masked", "nomask", "plain"])
    def test_single(self, dev, kind):
        for name in _ATLEAST:
            for i, (data, mask) in enumerate(_ATLEAST_DATA):
                x, n, d = _inp(kind, data, mask, dev)
                with _note(kind, name, i):
                    _cmp(name, [(x, n)], d)

    def test_multiple(self, dev):
        x1, n1 = make(_A1, _M1, dev)
        x2, n2 = make(_A23, None, dev)
        for name in _ATLEAST:
            with _note(name):
                _cmp(name, [(x1, n1), (x2, n2)], dev)
                _cmp(name, [(x1, n1), (to_dev(_A3, dev), _A3)], dev)

    def test_lists_and_scalars(self):
        for name in _ATLEAST:
            with _note(name):
                _cmp(name, [(3.0, 3.0)], None)
                _cmp(name, [([1, 2], [1, 2])], None)
                _cmp(name, [([[1, 2]], [[1, 2]])], None)

    def test_no_args(self):
        for name in _ATLEAST:
            with _note(name):
                rx, rn = both(getattr(XMA, name), getattr(np.ma, name))
                if rn is not None:
                    if NP_LT_25 and rn == []:
                        rn = ()   # numpy < 2.5 (checked 2.0.2, 2.2.6) returns a list for no arguments; 2.5 a tuple
                    assert rx == rn == ()

    def test_returns_new_mask_semantics(self, dev):
        x, n = make(_A1, _M1, dev)
        for name in _ATLEAST:
            with _note(name):
                r = getattr(XMA, name)(x)
                rn = getattr(np.ma, name)(n)
                assert_same(r, rn, dev=dev)
                assert r.shape == rn.shape


_STACK_NAMES = ["vstack", "hstack", "column_stack", "dstack", "row_stack"]


class TestStacking:
    @pytest.mark.parametrize("mix", ["masked", "nomask", "mixed", "plain", "plain_masked"])
    def test_basic(self, dev, mix):
        for shape in [(3,), (2, 3), (2, 3, 2)]:
            d1 = _arange(shape)
            d2 = _arange(shape) + 100
            m1, m2 = _rmask(shape, 0.4), _rmask(shape, 0.4)
            if mix == "masked":
                (x1, n1), (x2, n2) = make(d1, m1, dev), make(d2, m2, dev)
            elif mix == "nomask":
                (x1, n1), (x2, n2) = make(d1, None, dev), make(d2, None, dev)
            elif mix == "mixed":
                (x1, n1), (x2, n2) = make(d1, m1, dev), make(d2, None, dev)
            elif mix == "plain":
                x1, n1, x2, n2 = to_dev(d1, dev), d1, to_dev(d2, dev), d2
            else:
                (x1, n1) = make(d1, m1, dev)
                x2, n2 = to_dev(d2, dev), d2
            for name in _STACK_NAMES:
                with _note(mix, shape, name):
                    _cmp(name, [([x1, x2], [n1, n2])], dev)
                    _cmp(name, [((x1, x2, x1), (n1, n2, n1))], dev)

    def test_single_element_and_lists(self, dev):
        x, n = make(_A23, _M23, dev)
        for name in _STACK_NAMES:
            with _note(name):
                _cmp(name, [([x], [n])], dev)
                _cmp(name, [([[1, 2], [3, 4]], [[1, 2], [3, 4]])], None)
                _cmp(name, [([x, [[1., 2., 3.]]], [n, [[1., 2., 3.]]])], None)

    def test_mixed_ndim(self, dev):
        x1, n1 = make(_A1[:3], _M1[:3], dev)
        x2, n2 = make(_A23, _M23, dev)
        for name in _STACK_NAMES:
            with _note(name):
                _cmp(name, [([x1, x2], [n1, n2])], dev)
                _cmp(name, [([x2, x1], [n2, n1])], dev)

    def test_shape_mismatch(self, dev):
        x1, n1 = make(_A23, _M23, dev)
        x2, n2 = make(_A2, _M2, dev)
        for name in _STACK_NAMES:
            with _note(name):
                _cmp(name, [([x1, x2], [n1, n2])], dev)

    def test_empty_sequence(self):
        for name in _STACK_NAMES:
            with _note(name):
                _cmp(name, [([], [])], None)

    def test_dtype_promotion_and_empty_arrays(self, dev):
        x1, n1 = make(_A23.astype(np.int8), _M23, dev)
        x2, n2 = make(_A23.astype(np.float32), None, dev)
        x3, n3 = make(np.zeros((0, 3)), None, dev)
        for name in _STACK_NAMES:
            with _note(name):
                _cmp(name, [([x1, x2], [n1, n2])], dev)
                _cmp(name, [([x3, x3], [n3, n3])], dev)

    def test_npma_inputs(self):
        a = np.ma.masked_array(_A23, mask=_M23)
        for name in _STACK_NAMES:
            with _note(name):
                _cmp(name, [([a, a], [a, a])], "cpu")

    def test_fill_value_hardmask_of_result(self, dev):
        x, n = make(_A23, _M23, dev, fill_value=-5.0, hard_mask=True)
        for name in ("vstack", "hstack", "dstack", "column_stack"):
            with _note(name):
                _cmp(name, [([x, x], [n, n])], dev)

    @pytest.mark.parametrize("mix", ["masked", "nomask", "mixed"])
    def test_stack_axis(self, dev, mix):
        m1 = None if mix == "nomask" else _M23
        m2 = _M23 if mix == "masked" else None
        (x1, n1) = make(_A23, m1, dev)
        (x2, n2) = make(_A23 * 2, m2, dev)
        for axis in [0, 1, 2, -1, -2, -3, 3, -4]:
            with _note(mix, axis):
                _cmp("stack", [([x1, x2], [n1, n2])], dev, axis=axis)

    def test_stack_signature(self, dev):
        x1, n1 = make(_A23, _M23, dev)
        x2, n2 = make(_A23 + 1, None, dev)
        _cmp("stack", [([x1, x2], [n1, n2])], dev, axis=1, out=None)
        _cmp("stack", [([x1, x2], [n1, n2])], dev, axis=0, dtype=np.float32)
        _cmp("stack", [([x1, x2], [n1, n2])], dev, dtype=np.float32, casting="same_kind")
        _cmp("stack", [([x1, x2], [n1, n2])], dev, dtype=np.int8, casting="same_kind")
        _cmp("stack", [([x1, x2], [n1, n2])], dev, dtype=np.int8, casting="unsafe")
        _cmp("stack", [([x1, x2], [n1, n2])], dev, casting="no")

    def test_stack_positional_axis(self, dev):
        x1, n1 = make(_A23, _M23, dev)
        r = XMA.stack([x1, x1], 1)
        assert_same(r, np.ma.stack([n1, n1], 1), dev=dev)
        r = XMA.stack([x1, x1], axis=2)
        assert_same(r, np.ma.stack([n1, n1], axis=2), dev=dev)

    def test_stack_scalars_lists(self):
        _cmp("stack", [([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])], None)
        _cmp("stack", [([[1, 2], [3, 4]], [[1, 2], [3, 4]])], None, axis=1)

    def test_stack_shape_mismatch(self, dev):
        x1, n1 = make(_A23, _M23, dev)
        x2, n2 = make(_A2, _M2, dev)
        _cmp("stack", [([x1, x2], [n1, n2])], dev)

    def test_row_stack_equals_vstack(self, dev):
        x, n = make(_A23, _M23, dev)
        assert_same(XMA.row_stack([x, x]), np.ma.row_stack([n, n]), dev=dev)
        assert_same(XMA.row_stack([x, x]), np.ma.vstack([n, n]), dev=dev)

    def test_stack_mask_shared_with_input_not_aliased(self, dev):
        x, _ = make(_A23, _M23, dev)
        r = XMA.vstack([x, x])
        before = host(XMA.getmaskarray(x)).copy()
        r[0, 0] = XMA.masked
        r[0, 1] = 99.0
        np.testing.assert_array_equal(host(XMA.getmaskarray(x)), before)


class TestHsplitDiagflatEdiff:
    @pytest.mark.parametrize("kind", ["masked", "plain", "npma"])
    def test_hsplit_2d(self, dev, kind):
        data = np.arange(12.).reshape(2, 6)
        mask = _rmask((2, 6), 0.3)
        x, n, d = _inp(kind, data, mask, dev)
        for ind in [1, 2, 3, 6, [1, 4], [2], [], [0, 6], [4, 1], [10]]:
            with _note(kind, ind):
                rx, rn = both(XMA.hsplit, _ref("hsplit"), (x, n), (ind, ind))
                if rn is not None:
                    assert isinstance(rx, list) == isinstance(rn, list)
                    assert_same(rx, rn, dev=d)

    def test_hsplit_1d(self, dev):
        x, n = make(_A1, _M1, dev)
        for ind in [1, 2, 4, 3, [2]]:
            with _note(ind):
                rx, rn = both(XMA.hsplit, _ref("hsplit"), (x, n), (ind, ind))
                if rn is not None:
                    assert_same(rx, rn, dev=dev)

    def test_hsplit_3d_and_0d(self, dev):
        x, n = make(_A3, _M3, dev)
        rx, rn = both(XMA.hsplit, _ref("hsplit"), (x, n), (3, 3))
        if rn is not None:
            assert_same(rx, rn, dev=dev)
        x, n = make(np.float64(1.0), None, dev)
        both(XMA.hsplit, _ref("hsplit"), (x, n), (1, 1))

    def test_hsplit_pieces_independent(self, dev):
        x, n = make(np.arange(12.).reshape(2, 6), _rmask((2, 6), 0.3), dev)
        parts = XMA.hsplit(x, 3)
        assert len(parts) == 3
        assert all(isinstance(p, XMA.MaskedArray) for p in parts)

    @pytest.mark.parametrize("kind", ["masked", "nomask", "list", "npma"])
    def test_diagflat_1d(self, dev, kind):
        x, n, d = _inp(kind, _A1, _M1, dev)
        for k in [0, 1, -1, 3, -4, 10]:
            with _note(kind, k):
                _cmp("diagflat", [(x, n)], d, k=k)

    def test_diagflat_2d_and_0d(self, dev):
        for k in [0, 1, -2]:
            x, n = make(_A23, _M23, dev)
            with _note(k):
                _cmp("diagflat", [(x, n)], dev, k=k)
                x, n = make(np.float64(2.0), np.True_, dev)
                _cmp("diagflat", [(x, n)], dev, k=k)

    def test_diagflat_empty_and_hardmask(self, dev):
        x, n = make(np.zeros(0), None, dev)
        _cmp("diagflat", [(x, n)], dev)
        x, n = make(_A1, _M1, dev, hard_mask=True, fill_value=4.0)
        _cmp("diagflat", [(x, n)], dev)

    @pytest.mark.parametrize("kind", ["masked", "nomask", "list", "npma"])
    def test_ediff1d(self, dev, kind):
        x, n, d = _inp(kind, np.array([1., 4., 9., 16., 25.]),
                       np.array([0, 0, 1, 0, 0], bool), dev)
        for to_end, to_begin in [(None, None), ([9.0], None), (None, [7.0, 8.0]),
                                 ([1.0, 2.0], [3.0]), (5.0, 6.0)]:
            with _note(kind, to_end, to_begin):
                _cmp("ediff1d", [(x, n)], d, to_end=to_end, to_begin=to_begin)

    def test_ediff1d_edge_shapes(self, dev):
        for data in (np.zeros(0), np.array([3.0]), np.arange(6.).reshape(2, 3)):
            x, n = make(data, np.zeros(data.shape, bool), dev)
            with _note(data.shape):
                _cmp("ediff1d", [(x, n)], dev)
                _cmp("ediff1d", [(x, n)], dev, to_end=[1.0], to_begin=[0.0])

    def test_ediff1d_masked_ends_and_ints(self, dev):
        x, n = make(np.arange(6), np.array([1, 0, 0, 0, 1, 0], bool), dev)
        _cmp("ediff1d", [(x, n)], dev)
        _cmp("ediff1d", [(x, n)], dev, to_end=[1, 2])
        xe, ne = make(np.array([5, 6]), np.array([0, 1], bool), dev)
        # masked to_begin / to_end
        rx, rn = both(lambda a, b: XMA.ediff1d(a, to_end=b), lambda a, b: np.ma.ediff1d(a, to_end=b),
                      (x, n), (xe, ne))
        if rn is not None:
            assert_same(rx, rn, dev=dev)

    def test_ediff1d_int_to_float_ends(self, dev):
        x, n = make(np.arange(4), None, dev)
        _cmp("ediff1d", [(x, n)], dev, to_end=[1.5])
        _cmp("ediff1d", [(x, n)], dev, to_begin=[1.5, 2.5])


def _mr_cases():
    a = np.array([1., 2., 3., 4.])
    ma_ = np.array([0, 1, 0, 0], bool)
    a2 = np.arange(6.).reshape(2, 3)
    m2 = np.array([[0, 1, 0], [0, 0, 0]], bool)
    return {
        "two_masked_1d": ([("m", a, ma_), ("m", a + 10, ~ma_)], True),
        "masked_nomask": ([("m", a, ma_), ("m", a + 10, None)], True),
        "three": ([("m", a, ma_), ("m", a, ma_), ("m", a[:2], None)], True),
        "plain_masked": ([("p", a, None), ("m", a, ma_)], True),
        "plain_plain": ([("p", a, None), ("p", a * 2, None)], True),
        "int_float": ([("m", np.arange(3), np.array([1, 0, 0], bool)), ("m", a, ma_)], True),
        "single_masked": ([("m", a, ma_)], True),
        "single_plain": ([("p", a, None)], True),
        "single_2d": ([("m", a2, m2)], True),
        "two_2d": ([("m", a2, m2), ("m", a2 + 1, None)], True),
        "empty_1d": ([("m", np.zeros(0), None), ("m", a, ma_)], True),
        "slice": ([("s", slice(1, 4), None), ("m", a, ma_)], False),
        "slice_only": ([("s", slice(0, 5), None)], False),
        "slice_scalar": ([("s", slice(1, 4), None), ("c", 7, None)], False),
        "scalar_masked": ([("m", a, ma_), ("c", 5, None)], False),
        "scalars": ([("c", 1, None), ("c", 2.5, None)], False),
        "scalar_slice_masked": ([("m", a, ma_), ("c", 5, None), ("s", slice(2, 4), None)], False),
        "complex_step": ([("s", slice(0, 1, 3j), None), ("c", 5, None)], False),
        "list_item": ([("l", [1, 2], None), ("m", a, ma_)], False),
        "neg_step": ([("s", slice(5, 0, -2), None)], False),
        "bool_masked": ([("m", np.array([True, False]), np.array([0, 1], bool)),
                         ("m", np.array([False]), None)], True),
        "mask_all": ([("m", a, np.ones(4, bool)), ("m", a, np.zeros(4, bool))], True),
        "empty_tuple": ([], False),
    }


def _mr_build(spec, dev):
    xs, ns = [], []
    for kind, v, m in spec:
        if kind == "m":
            x, n = make(v, m, dev)
        elif kind == "p":
            x, n = to_dev(v, dev), v.copy()
        else:
            x, n = v, v
        xs.append(x)
        ns.append(n)
    return xs, ns




class TestMr:
    @pytest.mark.parametrize("arrays_only", [True, False])
    def test_mr_tuple(self, dev, arrays_only):
        for case, (spec, ao) in _mr_cases().items():
            if ao != arrays_only:
                continue
            xs, ns = _mr_build(spec, dev)
            key_x, key_n = tuple(xs), tuple(ns)
            with _note(case):
                try:
                    rn = np.ma.mr_[key_n]
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        XMA.mr_[key_x]
                    continue
                rx = XMA.mr_[key_x]
                assert_same(rx, rn, dev=dev if arrays_only else None, ctx=f"mr_[{case}]")

    def test_mr_single_nonarray_key(self):
        for key in [slice(0, 5), slice(5), slice(1, 10, 3), slice(0, 1, 5j),
                    slice(2, -3, -1), slice(0, 1, 0.25), 5, 2.5, [1, 2, 3], (3,)]:
            with _note(key):
                try:
                    rn = np.ma.mr_[key]
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        XMA.mr_[key]
                    continue
                assert_same(XMA.mr_[key], rn, ctx=f"mr_[{key!r}]")

    def test_mr_1d_inputs_concat_to_1d(self, dev):
        for n in [1, 2, 5]:
            parts = [make(_A1 + i, _RNG.random(4) < 0.4, dev) for i in range(n)]
            r = XMA.mr_[tuple(p[0] for p in parts)]
            rn = np.ma.mr_[tuple(p[1] for p in parts)]
            with _note(n):
                assert r.ndim == 1 and r.shape == (4 * n,)
                assert_same(r, rn, dev=dev)

    def test_mr_nonmasked_ndarray(self, dev):
        a = to_dev(_A1, dev)
        assert_same(XMA.mr_[a], np.ma.mr_[_A1], dev=dev)
        assert_same(XMA.mr_[a, a], np.ma.mr_[_A1, _A1], dev=dev)

    def test_mr_mask_independent_of_inputs(self, dev):
        x, _ = make(_A1, _M1, dev)
        r = XMA.mr_[x, x]
        r[0] = XMA.masked
        assert not bool(host(XMA.getmaskarray(x))[0])

    def test_mr_is_indexable_instance_not_callable(self):
        with pytest.raises(Exception):
            XMA.mr_()  # numpy's mr_ is not callable either
        with pytest.raises(Exception):
            np.ma.mr_()


# ---------------------------------------------------------------------------
# C. core function forms
# ---------------------------------------------------------------------------
def _core_inputs(dev):
    xm, nm = make(_A23, _M23, dev)
    xn, nn = make(_A23, None, dev)
    xf, nf = make(_A23, np.zeros((2, 3), bool), dev)
    return {
        "masked": (xm, nm), "nomask": (xn, nn), "mask_false": (xf, nf),
        "plain": (to_dev(_A23, dev), _A23.copy()),
        "list": ([1, 2, 3], [1, 2, 3]), "scalar": (3.0, 3.0), "int": (3, 3),
        "none": (None, None), "str": ("a", "a"),
        "bool_arr": (to_dev(_M23, dev), _M23.copy()),
        "0d_masked": make(np.float64(1.0), np.True_, dev),
    }


_IN_KEYS = ["masked", "nomask", "mask_false", "plain", "list", "scalar", "int", "none",
            "str", "bool_arr", "0d_masked"]


def _fn(name):
    fx = getattr(XMA, name, None)
    assert fx is not None, f"xupy.ma.{name} missing"
    return fx


def _mask_result_same(rx, rn, dev):
    if rn is NOMASK:
        assert rx is XMA.nomask, f"expected nomask, got {rx!r}"
    else:
        assert rx is not XMA.nomask
        assert_same(rx, np.asarray(rn), dev=dev)



def _where_operand(k, off, dev):
    data = _A23 + off
    if k == "masked":
        return make(data, _M23, dev)
    if k == "nomask":
        return make(data, None, dev)
    if k == "plain":
        return to_dev(data, dev), data
    if k == "scalar":
        return 1.5 + off, 1.5 + off
    if k == "int_scalar":
        return 7, 7
    return XMA.masked, np.ma.masked


class TestGetters:
    def test_getmask(self, dev):
        ins, fx = _core_inputs(dev), _fn("getmask")
        for key in _IN_KEYS:
            x, n = ins[key]
            with _note(key):
                rx, rn = both(fx, np.ma.getmask, (x, n))
                if rn is None and rx is None:
                    continue
                _mask_result_same(rx, rn, dev if key in ("masked", "mask_false") else None)

    def test_getmaskarray(self, dev):
        ins = _core_inputs(dev)
        for key in _IN_KEYS:
            x, n = ins[key]
            with _note(key):
                rx, rn = both(_fn("getmaskarray"), np.ma.getmaskarray, (x, n))
                if rn is None and rx is None:
                    continue
                assert rn.dtype == np.bool_
                assert host(rx).dtype == np.bool_
                np.testing.assert_array_equal(host(rx), rn)
                if key in ("masked", "nomask", "mask_false", "plain"):
                    assert on_dev(rx, dev)
                    assert rx is not XMA.nomask

    def test_getdata(self, dev):
        ins = _core_inputs(dev)
        for key in _IN_KEYS:
            x, n = ins[key]
            with _note(key):
                rx, rn = both(_fn("getdata"), np.ma.getdata, (x, n))
                if rn is None and rx is None:
                    continue
                assert_same(rx, rn, dev=dev if key in ("masked", "nomask", "plain",
                                                       "mask_false") else None)

    def test_getdata_subok_false(self, dev):
        x, n = make(_A23, _M23, dev)
        rx, rn = both(XMA.getdata, np.ma.getdata, (x, n), subok=False)
        if rn is not None:
            assert_same(rx, rn, dev=dev)

    def test_getdata_getmask_masked_singleton(self):
        assert_same(XMA.getdata(XMA.masked), np.ma.getdata(np.ma.masked))
        assert_same(XMA.getmask(XMA.masked), np.ma.getmask(np.ma.masked))
        assert_same(XMA.getmaskarray(XMA.masked), np.ma.getmaskarray(np.ma.masked))

    def test_getmask_nomask_returns_nomask_singleton(self, dev):
        x, n = make(_A23, None, dev)
        assert XMA.getmask(x) is XMA.nomask and np.ma.getmask(n) is NOMASK

    def test_getdata_returns_same_data_view(self, dev):
        x, _ = make(_A23, _M23, dev)
        d = XMA.getdata(x)
        assert on_dev(d, dev) and d.shape == (2, 3)

    def test_predicates(self, dev):
        ins = _core_inputs(dev)
        for name in ["is_masked", "isMaskedArray", "isMA", "is_mask"]:
            for key in _IN_KEYS:
                x, n = ins[key]
                with _note(name, key):
                    rx, rn = both(_fn(name), getattr(np.ma, name), (x, n))
                    if rn is None and rx is None:
                        continue
                    assert type(rx) is bool and type(rn) is bool
                    assert rx == rn

    def test_predicates_singletons(self):
        for name in ["is_masked", "isMaskedArray", "isMA", "is_mask"]:
            for x, n in ((XMA.masked, np.ma.masked), (XMA.nomask, NOMASK)):
                with _note(name):
                    rx, rn = both(_fn(name), getattr(np.ma, name), (x, n))
                    if rn is not None:
                        assert type(rx) is bool and rx == rn

    def test_is_masked_partial_vs_all_false(self, dev):
        x, n = make(_A23, _M23, dev)
        assert XMA.is_masked(x) is True
        x, n = make(_A23, np.zeros((2, 3), bool), dev)
        assert XMA.is_masked(x) is False is np.ma.is_masked(n)
        x, n = make(_A23, np.ones((2, 3), bool), dev)
        assert XMA.is_masked(x) is True
        x, n = make(np.zeros((0,)), None, dev)
        assert XMA.is_masked(x) is False is np.ma.is_masked(n)

    def test_is_mask_arrays(self, dev):
        for m in [np.array([True, False]), np.array([1, 0]), np.array([0.0]),
                  np.zeros((2, 2), bool), np.zeros(0, bool), np.array(True)]:
            with _note(m):
                rx, rn = both(_fn("is_mask"), np.ma.is_mask, (to_dev(m, dev), m))
                if rn is not None:
                    assert rx == rn

    def test_isMA_alias(self, dev):
        x, n = make(_A23, _M23, dev)
        assert XMA.isMA is XMA.isMaskedArray or XMA.isMA(x) is True
        assert XMA.isMaskedArray(x) is True
        assert XMA.isMaskedArray(to_dev(_A23, dev)) is False


class TestMakeMaskNone:
    def test_shapes(self):
        fx = _fn("make_mask_none")
        for shape in [(), (0,), (3,), (2, 3), (2, 0, 3), 4, [2, 3]]:
            with _note(shape):
                rx, rn = both(fx, np.ma.make_mask_none, (shape, shape))
                if rn is None:
                    continue
                assert host(rx).dtype == rn.dtype == np.bool_
                assert host(rx).shape == rn.shape
                assert not host(rx).any()
                assert rx is not XMA.nomask

    def test_dtype_arg(self):
        fx = _fn("make_mask_none")
        for dt in [None, np.int8, "?"]:
            with _note(dt):
                rx, rn = both(fx, np.ma.make_mask_none, ((2,), (2,)), dtype=dt)
                if rn is not None:
                    assert host(rx).dtype == rn.dtype
                    assert host(rx).shape == rn.shape
        # structured mask dtypes are unsupported by design (DESIGN.md: numeric and bool data only)
        for dt in [[("a", int), ("b", float)], np.dtype([("a", "?")])]:
            with pytest.raises(NotImplementedError):
                fx((2,), dtype=dt)

    def test_negative_shape(self):
        both(_fn("make_mask_none"), np.ma.make_mask_none, ((-1,), (-1,)))

    def test_fresh_arrays(self):
        fx = _fn("make_mask_none")
        a, b = fx((3,)), fx((3,))
        assert a is not b
        a[0] = True
        assert not b[0]


class TestMaskOr:
    @pytest.mark.parametrize("m1k", ["nomask", "arr", "zeros", "ones"])
    def test_combinations(self, dev, m1k):
        base = {"arr": np.array([1, 0, 0, 1], bool), "zeros": np.zeros(4, bool),
                "ones": np.ones(4, bool), "nomask": None}
        other = {"arr": np.array([0, 0, 1, 1], bool), "zeros": np.zeros(4, bool),
                 "ones": np.ones(4, bool), "nomask": None}

        def pick(k, table):
            if k == "nomask":
                return XMA.nomask, NOMASK
            return to_dev(table[k], dev), table[k].copy()

        fx = _fn("mask_or")
        for m2k in ["nomask", "arr", "zeros", "ones"]:
            for shrink in (True, False):
                for copy in (True, False):
                    a, an = pick(m1k, base)
                    b, bn = pick(m2k, other)
                    with _note(m1k, m2k, shrink, copy):
                        rx, rn = both(fx, np.ma.mask_or, (a, an), (b, bn),
                                      shrink=shrink, copy=copy)
                        if rn is None and rx is None:
                            continue
                        _mask_result_same(rx, rn, dev)

    def test_does_not_alias_when_copy(self, dev):
        m = np.array([1, 0, 1], bool)
        a = to_dev(m, dev)
        r = XMA.mask_or(a, XMA.nomask, copy=True)
        assert r is not XMA.nomask
        r[0] = False
        assert bool(host(a)[0])

    def test_shape_mismatch(self, dev):
        a = to_dev(np.array([1, 0], bool), dev)
        b = to_dev(np.array([1, 0, 1], bool), dev)
        both(_fn("mask_or"), np.ma.mask_or, (a, np.array([1, 0], bool)), (b, np.array([1, 0, 1], bool)))

    def test_broadcast_scalarlike(self, dev):
        a = to_dev(np.array([[1, 0], [0, 0]], bool), dev)
        b = to_dev(np.array([0, 1], bool), dev)
        rx, rn = both(_fn("mask_or"), np.ma.mask_or,
                      (a, np.array([[1, 0], [0, 0]], bool)), (b, np.array([0, 1], bool)))
        if rn is not None:
            _mask_result_same(rx, rn, dev)

    def test_structured_masks(self):
        dt = [("a", "?"), ("b", "?")]
        m1 = np.array([(True, False), (False, False)], dtype=dt)
        m2 = np.array([(False, True), (False, False)], dtype=dt)
        # structured masks are unsupported by design (DESIGN.md: numeric and bool data only)
        with pytest.raises(NotImplementedError):
            _fn("mask_or")(m1, m2)


class TestConcatenateWhere:
    @pytest.mark.parametrize("mix", ["masked", "nomask", "mixed", "plain", "npma"])
    def test_concatenate(self, dev, mix):
        (x1, n1) = make(_A23, _M23, dev)
        if mix == "masked":
            x2, n2 = make(_A23 + 10, np.array([[1, 0, 0], [0, 0, 1]], bool), dev)
        elif mix == "nomask":
            x1, n1 = make(_A23, None, dev)
            x2, n2 = make(_A23 + 10, None, dev)
        elif mix == "mixed":
            x2, n2 = make(_A23 + 10, None, dev)
        elif mix == "plain":
            x2, n2 = to_dev(_A23 + 10, dev), _A23 + 10
        else:
            a = np.ma.masked_array(_A23 + 10, mask=_M23)
            x2, n2 = a, a.copy()
        # mixed devices: cupy wins (DESIGN.md decision 4), so a gpu XuPy operand with a numpy.ma one gives gpu data
        d = dev
        fx = _fn("concatenate")
        for axis in [0, 1, -1, None]:
            with _note(mix, axis):
                rx, rn = both(fx, np.ma.concatenate, ([x1, x2], [n1, n2]), axis=axis)
                if rn is not None:
                    assert_same(rx, rn, dev=d)

    def test_concatenate_default_axis(self, dev):
        x, n = make(_A23, _M23, dev)
        assert_same(XMA.concatenate([x, x]), np.ma.concatenate([n, n]), dev=dev)
        assert_same(XMA.concatenate((x, x, x)), np.ma.concatenate((n, n, n)), dev=dev)

    def test_concatenate_errors_and_edges(self, dev):
        fx = _fn("concatenate")
        x, n = make(_A23, _M23, dev)
        x1, n1 = make(_A1, _M1, dev)
        for args in ((([], []),), (([x, x1], [n, n1]),), (([x], [n]),)):
            rx, rn = both(fx, np.ma.concatenate, *args)
            if rn is not None:
                assert_same(rx, rn, dev=dev)
        rx, rn = both(fx, np.ma.concatenate, ([x, x], [n, n]), axis=5)
        e0, e0n = make(np.zeros((0, 3)), None, dev)
        rx, rn = both(fx, np.ma.concatenate, ([e0, x], [e0n, n]))
        if rn is not None:
            assert_same(rx, rn, dev=dev)

    def test_concatenate_dtype_promotion_and_fill(self, dev):
        x, n = make(_A23.astype(np.int8), _M23, dev, fill_value=3)
        y, m = make(_A23.astype(np.float32), None, dev)
        rx, rn = both(_fn("concatenate"), np.ma.concatenate, ([x, y], [n, m]))
        assert_same(rx, rn, dev=dev)
        rx, rn = both(_fn("concatenate"), np.ma.concatenate, ([x, x], [n, n]))
        assert_same(rx, rn, dev=dev)

    def test_concatenate_hard_mask_and_masked_const(self, dev):
        x, n = make(_A1, _M1, dev, hard_mask=True)
        rx, rn = both(_fn("concatenate"), np.ma.concatenate, ([x, x], [n, n]))
        assert_same(rx, rn, dev=dev)
        both(_fn("concatenate"), np.ma.concatenate, ([x, XMA.masked], [n, np.ma.masked]))

    def test_where_one_arg(self, dev):
        fx = _fn("where")
        for data, mask in ((_A23 > 2, _M23), (_A23 > 2, None), (_A23 > 100, None),
                           (_A1 > 0, _M1)):
            x, n = make(data, mask, dev)
            rx, rn = both(fx, np.ma.where, (x, n))
            assert_same(rx, rn, dev=dev)

    @pytest.mark.parametrize("cmask", [None, "partial", "all"])
    def test_where_three_args(self, dev, cmask):
        fx = _fn("where")
        cm = {None: None, "partial": np.array([[1, 0, 0], [0, 0, 1]], bool),
              "all": np.ones((2, 3), bool)}[cmask]
        c, cn = make(_A23 > 2, cm, dev)
        for xk in ["masked", "plain", "scalar", "nomask", "masked_const"]:
            for yk in ["masked", "plain", "scalar", "int_scalar", "masked_const"]:
                (xx, xn), (yy, yn) = _where_operand(xk, 0, dev), _where_operand(yk, 100, dev)
                with _note(cmask, xk, yk):
                    rx, rn = both(fx, np.ma.where, (c, cn), (xx, xn), (yy, yn))
                    if rn is not None:
                        # numpy.ma shrinks an all-False result mask to nomask (data-dependent, host sync);
                        # XuPy does not (DESIGN.md decision 6).  With a masked condition the result mask
                        # always has a True, so nomask-ness is still compared there.
                        assert_same(rx, rn, dev=dev, strict_nomask=cmask is not None)

    def test_where_plain_condition_and_broadcast(self, dev):
        fx = _fn("where")
        cond = np.array([True, False, True])
        x, n = make(_A23, _M23, dev)
        # nomask-ness relaxed: numpy.ma shrinks an all-False result mask (DESIGN.md decision 6: no host sync)
        rx, rn = both(fx, np.ma.where, (to_dev(cond, dev), cond), (x, n), (0.0, 0.0))
        assert_same(rx, rn, dev=dev, strict_nomask=False)  # see NO-SYNC note
        rx, rn = both(fx, np.ma.where, (to_dev(cond, dev), cond), (1, 1), (2.0, 2.0))
        if rn is not None:
            assert_same(rx, rn, dev=dev, strict_nomask=False)  # see NO-SYNC note

    def test_where_wrong_arg_count_and_shapes(self, dev):
        fx = _fn("where")
        x, n = make(_A23, _M23, dev)
        both(fx, np.ma.where, (x > 1, n > 1), (x, n))
        c, cn = make(np.array([True, False, True, True]), None, dev)
        both(fx, np.ma.where, (c, cn), (x, n), (x, n))
