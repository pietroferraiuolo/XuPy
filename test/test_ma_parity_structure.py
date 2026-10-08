"""
Differential tests of the *structure* of ``xupy.ma`` against ``numpy.ma``.

Scope: singletons (``nomask``/``masked``), the constructor and its mask /
dtype / fill_value handling, ``__getitem__``/``__setitem__``, metadata and
shape methods, Python protocols and copy/pickle.  numpy.ma (2.5) is the
reference: where numpy.ma has a quirk the tests mirror it.
"""
# NO-SYNC: numpy.ma shrinks the mask to nomask via a host-side .any(); xupy.ma never syncs
# (DESIGN decision 6), so a result mask may be a full all-False array where numpy.ma has
# nomask.  Calls that pass `strict_nomask=False` accept exactly that difference (see NO-SYNC note).

import contextlib
import copy
import pickle
import warnings

import numpy as np
import pytest

import xupy

from ._ma_parity_helpers import (
    GPU_OK, XMA, assert_same, both, cp, host, make, mka, on_dev, to_dev, xp_of,
)

NM = XMA.nomask
NNM = np.ma.nomask
MSK = XMA.masked
NMSK = np.ma.masked

needs_gpu = pytest.mark.skipif(not GPU_OK, reason="no usable GPU")


# --------------------------------------------------------------------------
# local helpers
# --------------------------------------------------------------------------
@contextlib.contextmanager
def case(*info):
    """Annotate failures of an in-test loop with the loop variables."""
    try:
        yield
    except Exception as e:  # noqa: BLE001
        if hasattr(e, "add_note"):
            e.add_note(f"case: {info!r}")
        raise


def rdata(shape, dtype=float, seed=0, positive=True):
    rng = np.random.default_rng(seed)
    a = rng.random(shape) + (0.5 if positive else -0.5)
    return a.astype(dtype)


def pattern(shape, mod=3, off=1):
    """Deterministic boolean pattern with both True and False entries."""
    n = int(np.prod(shape, dtype=int))
    return ((np.arange(n) % mod) == off).reshape(shape)


def run_pair(f, x, n, dev, **kw):
    """Apply ``f(module, array)`` to the (xupy, numpy.ma) pair and compare.

    If numpy raises, XuPy must raise the same exception type.
    """
    try:
        rn = f(np.ma, n)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            f(XMA, x)
        return
    rx = f(XMA, x)
    assert_same(rx, rn, dev=dev, **kw)


def dev_index(idx, dev):
    """Convert the array / list parts of an index to the device (gpu only)."""
    if dev != "gpu":
        return idx

    def conv(i):
        if isinstance(i, (np.ndarray, list)):
            return cp.asarray(np.asarray(i))
        return i

    if isinstance(idx, tuple):
        return tuple(conv(i) for i in idx)
    return conv(idx)


def same_mask(xm, nm, dev=None):
    """Compare mask objects returned by getmask-like functions."""
    if nm is NNM:
        assert xm is NM, f"expected xupy nomask, got {xm!r}"
    else:
        assert xm is not NM
        assert host(xm).dtype == np.bool_
        if dev is not None:
            assert on_dev(xm, dev)
        np.testing.assert_array_equal(host(xm), nm)


# ==========================================================================
# 1. nomask
# ==========================================================================
BOOL_OPERANDS = [
    pytest.param(True, id="py_True"),
    pytest.param(False, id="py_False"),
    pytest.param(np.True_, id="np_True"),
    pytest.param(np.array([True, False, True]), id="arr1d"),
    pytest.param(np.array([[True], [False]]), id="arr2d"),
    pytest.param(np.array(True), id="arr0d"),
    pytest.param(np.array([], dtype=bool), id="empty"),
]
BIN_OPS = {
    "or": lambda a, b: a | b,
    "and": lambda a, b: a & b,
    "xor": lambda a, b: a ^ b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


def _same_bool_result(rx, rn):
    if isinstance(rn, np.generic):
        assert isinstance(rx, np.generic) and type(rx) is type(rn), (type(rx), rx)
        assert bool(rx) == bool(rn)
    else:
        assert rx.shape == rn.shape
        assert host(rx).dtype == rn.dtype
        np.testing.assert_array_equal(host(rx), rn)


class TestNomaskSingleton:
    def test_invert(self):
        rx, rn = ~NM, ~NNM
        _same_bool_result(rx, rn)
        assert bool(rx) is True

    @pytest.mark.parametrize("other", BOOL_OPERANDS)
    def test_binary_numpy_semantics(self, other):
        for op, f in BIN_OPS.items():
            for reflected in (False, True):
                with case(op, reflected):
                    if reflected:
                        rx, rn = f(other, NM), f(other, NNM)
                    else:
                        rx, rn = f(NM, other), f(NNM, other)
                    _same_bool_result(rx, rn)

    @needs_gpu
    @pytest.mark.parametrize("reflected", [False, True])
    @pytest.mark.parametrize("shape", [(3,), (2, 1), ()])
    def test_binary_with_cupy_operand(self, reflected, shape):
        h = pattern(shape, 2, 0)
        other = cp.asarray(h)
        for op, f in BIN_OPS.items():
            with case(op):
                exp = f(h, np.False_) if reflected else f(np.False_, h)
                got = f(other, NM) if reflected else f(NM, other)
                assert isinstance(got, cp.ndarray)
                assert got.dtype == np.bool_
                np.testing.assert_array_equal(host(got), exp)

    def test_with_other_nomask_objects(self):
        for op, f in BIN_OPS.items():
            for a, b in ((NM, NM), (NM, NNM), (NNM, NM)):
                with case(op):
                    rx = f(a, b)
                    assert type(rx) is np.bool_
                    _same_bool_result(rx, f(NNM, NNM))

    def test_scalar_protocol(self):
        assert bool(NM) is False
        assert NM == False  # noqa: E712
        assert not (NM != False)  # noqa: E712
        assert NM == NNM
        assert NM.shape == () == NNM.shape
        assert NM.ndim == 0
        assert NM.size == 1
        assert NM.dtype == np.dtype(bool)
        assert isinstance(NM.dtype, np.dtype)
        _same_bool_result(NM.any(), NNM.any())
        _same_bool_result(NM.all(), NNM.all())
        rx, rn = NM.sum(), NNM.sum()
        assert type(rx) is type(rn) and rx == rn == 0

    def test_shape_methods_return_nomask(self):
        """Settled decision: shape-preserving / reshaping methods return the
        singleton itself (numpy's own ``ravel`` would return an array)."""
        fs = {
            "copy": lambda m: m.copy(),
            "ravel": lambda m: m.ravel(),
            "flatten": lambda m: m.flatten(),
            "astype": lambda m: m.astype(bool),
            "astype_nocopy": lambda m: m.astype(bool, copy=False),
            "T": lambda m: m.T,
            "transpose": lambda m: m.transpose(),
            "squeeze": lambda m: m.squeeze(),
            "reshape0": lambda m: m.reshape(()),
            "reshape1": lambda m: m.reshape(1),
            "reshape_m1": lambda m: m.reshape(-1),
            "reshape11": lambda m: m.reshape(1, 1),
            "reshape_tuple": lambda m: m.reshape((1, 1, 1)),
            "__copy__": lambda m: m.__copy__(),
            "__deepcopy__": lambda m: m.__deepcopy__({}),
        }
        for name, f in fs.items():
            assert f(NM) is NM, name

    def test_view_like_numpy(self):
        rn = NNM.view()
        rx = NM.view()
        if rn is NNM:
            assert rx is NM
        else:
            _same_bool_result(rx, rn)

    def test_asarray(self):
        a, b = np.asarray(NM), np.asarray(NNM)
        assert a.shape == b.shape == ()
        assert a.dtype == b.dtype == np.dtype(bool)
        assert not a
        assert NM.__array__().dtype == np.bool_

    def test_repr_str(self):
        # numpy 2 prints ``np.False_``; the design brief says ``False``.
        assert repr(NM) in (repr(NNM), "False")
        assert str(NM) == str(NNM) == "False"

    def test_identity_preserved(self):
        assert copy.copy(NM) is NM
        assert copy.deepcopy(NM) is NM
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.loads(pickle.dumps(NM, protocol=proto)) is NM

    def test_own_singleton_distinct_from_numpys(self):
        assert NM is XMA.nomask
        assert NM is XMA.MaskedArray([1.0, 2.0]).mask

    def test_hashable_like_numpy(self):
        try:
            hn = hash(NNM)
        except TypeError:
            with pytest.raises(TypeError):
                hash(NM)
        else:
            assert hash(NM) == hn


@pytest.mark.parametrize("builder", [
    lambda: XMA.masked_array([1.0, 2.0, 3.0]),
    lambda: XMA.masked_array(np.arange(6.0).reshape(2, 3)),
    lambda: XMA.masked_array(np.arange(3), mask=NM),
    lambda: XMA.masked_array(5.0),
    lambda: XMA.masked_array(np.array([], dtype=float)),
], ids=["list", "2d", "mask_nomask", "scalar", "empty"])
def test_mask_is_nomask_when_not_given(builder):
    with xupy.backend("cpu"):
        assert builder().mask is NM


@pytest.mark.parametrize("dev_", ["cpu", "gpu"])
def test_mask_is_nomask_when_not_given_on_device(dev_):
    if dev_ == "gpu" and not GPU_OK:
        pytest.skip("no usable GPU")
    x = mka(dev_, to_dev(np.arange(4.0), dev_))
    assert x.mask is NM
    assert on_dev(x.data, dev_)


NOMASK_CALLS = {
    "reshape": lambda M, a: a.reshape(3, 2),
    "reshape_m1": lambda M, a: a.reshape(-1),
    "ravel": lambda M, a: a.ravel(),
    "transpose": lambda M, a: a.transpose(),
    "T": lambda M, a: a.T,
    "mT": lambda M, a: a.mT,
    "swapaxes": lambda M, a: a.swapaxes(0, 1),
    "expand_dims": lambda M, a: M.expand_dims(a, 0),
    "repeat": lambda M, a: a.repeat(2),
    "copy": lambda M, a: a.copy(),
    "astype_f4": lambda M, a: a.astype("f4"),
    "std": lambda M, a: a.std(),
    "var_axis": lambda M, a: a.var(axis=0),
    "any": lambda M, a: a.any(),
    "count": lambda M, a: a.count(),
    "sum": lambda M, a: a.sum(),
    "sum_axis": lambda M, a: a.sum(axis=1),
    "M.log": lambda M, a: M.log(a),
    "neg": lambda M, a: -a,
    "abs": lambda M, a: abs(a),
    "add": lambda M, a: a + 1,
    "rtruediv": lambda M, a: 2 / a,
    "pow": lambda M, a: a ** 2,
    "eq_self": lambda M, a: a == a,
    "iadd": lambda M, a: _inplace(a, "__iadd__", 1.0),
    "take": lambda M, a: a.take([0, 2, 5]),
    "sort": lambda M, a: _sorted(a),
    "compressed": lambda M, a: a.compressed(),
    "filled": lambda M, a: a.filled(),
    "tolist": lambda M, a: a.tolist(),
    "item": lambda M, a: a.item(4),
    "getitem_int": lambda M, a: a[1],
    "getitem_slice": lambda M, a: a[:, 1:],
    "clip": lambda M, a: a.clip(0.7, 1.2),
    "diagonal": lambda M, a: a.diagonal(),
    "squeeze": lambda M, a: a.squeeze(),
    "nonzero": lambda M, a: a.nonzero(),
    "real": lambda M, a: a.real,
    "getmaskarray": lambda M, a: M.getmaskarray(a),
    "is_masked": lambda M, a: M.is_masked(a),
    "concatenate": lambda M, a: M.concatenate([a, a], axis=0),
    "where": lambda M, a: M.where(a > 1.0, a, 0.0),
}


def _inplace(a, name, v):
    a = a.copy()
    r = getattr(a, name)(v)
    assert r is a
    return a


def _sorted(a):
    a = a.copy()
    a.sort(axis=1)
    return a


# where: numpy.ma ends with `_shrink_mask` (`not mask.any()`, a host sync)
NOMASK_RELAXED = {"pow", "where"}


class TestNomaskArraysWorkLikeNumpy:
    """Every method on an array that has ``nomask`` (no mask fixtures)."""

    def test_method(self, dev):
        d = rdata((2, 3))
        for name, f in NOMASK_CALLS.items():
            with case(name):
                x, n = make(d, None, dev)
                assert x.mask is NM and n.mask is NNM
                # power: numpy.ma re-shrinks to nomask via `invalid.any()` (a host sync)
                kw = {"strict_nomask": False} if name in NOMASK_RELAXED else {}
                run_pair(f, x, n, dev, **kw)

    def test_methods_squeezable_shape(self, dev):
        x, n = make(rdata((1, 3, 1)), None, dev)
        fs = {
            "squeeze": lambda M, a: a.squeeze(),
            "reshape": lambda M, a: a.reshape(3),
            "T": lambda M, a: a.T,
            "sum": lambda M, a: a.sum(axis=(0, 2)),
            "std": lambda M, a: a.std(axis=0),
        }
        for name, f in fs.items():
            with case(name):
                run_pair(f, x, n, dev)

    def test_zero_d_and_empty(self, dev):
        for d in (np.array(2.5), np.zeros((0,)), np.zeros((0, 3))):
            x, n = make(d, None, dev)
            fs = {
                "sum": lambda M, a: a.sum(),
                "std": lambda M, a: a.std(),
                "ravel": lambda M, a: a.ravel(),
                "reshape": lambda M, a: a.reshape(-1) if a.size else a.reshape(0),
                "T": lambda M, a: a.T,
                "sort": lambda M, a: _sorted_any(a),
                "copy": lambda M, a: a.copy(),
            }
            for name, f in fs.items():
                with case(name, d.shape), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    run_pair(f, x, n, dev)

    def test_same_with_mask_for_contrast(self, dev):
        d = rdata((2, 3))
        for name in ["sum_axis", "var_axis", "take", "neg", "getitem_int", "filled"]:
            with case(name):
                x, n = make(d, pattern((2, 3)), dev)
                run_pair(NOMASK_CALLS[name], x, n, dev)

    def test_unary_results_keep_nomask(self, dev):
        x, n = make(rdata((2, 3)), None, dev)
        for f in (lambda a: -a, lambda a: abs(a), lambda a: a + 1, lambda a: a.copy(),
                  lambda a: a.reshape(3, 2), lambda a: a[:, 0]):
            # numpy.ma allocates a mask for some of these (e.g. unary minus)
            assert (f(x).mask is NM) == (f(n).mask is NNM)

    def test_domain_op_allocates_mask_like_numpy(self, dev):
        x, n = make(np.array([1.0, 0.0, 4.0]), None, dev)
        run_pair(lambda M, a: 1 / a, x, n, dev)
        run_pair(lambda M, a: M.sqrt(a - 2), x, n, dev)
        run_pair(lambda M, a: M.log(a), x, n, dev)


def _sorted_any(a):
    a = a.copy()
    a.sort()
    return a


# ==========================================================================
# 2. masked singleton and module-level mask helpers
# ==========================================================================
ARITH = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "truediv": lambda a, b: a / b,
    "floordiv": lambda a, b: a // b,
    "pow": lambda a, b: a ** b,
    "mod": lambda a, b: a % b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


class TestMaskedSingleton:
    @pytest.mark.parametrize("operand", [1, 2.5, np.int32(2), True])
    @pytest.mark.parametrize("reflected", [False, True])
    def test_with_scalars(self, operand, reflected):
        for op, f in ARITH.items():
            with case(op), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rn = f(operand, NMSK) if reflected else f(NMSK, operand)
                rx = f(operand, MSK) if reflected else f(MSK, operand)
                assert_same(rx, rn)

    @pytest.mark.parametrize("mask", [None, "pat"])
    def test_with_arrays(self, mask, dev):
        d = rdata((2, 3))
        x, n = make(d, None if mask is None else pattern((2, 3)), dev)
        for op, f in ARITH.items():
            for reflected in (False, True):
                with case(op, reflected), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rn = f(n, NMSK) if reflected else f(NMSK, n)
                    rx = f(x, MSK) if reflected else f(MSK, x)
                    assert_same(rx, rn, dev=dev)
                    assert bool(np.all(host(XMA.getmaskarray(rx))))

    def test_with_plain_arrays_and_masked(self, dev):
        a = to_dev(rdata((3,)), dev)
        for op in ("add", "mul", "eq", "lt"):
            f = ARITH[op]
            with case(op):
                assert_same(f(MSK, a), f(NMSK, host(a)), dev=dev)

    def test_masked_with_masked(self):
        for op in ARITH:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                assert_same(ARITH[op](MSK, MSK), ARITH[op](NMSK, NMSK))

    def test_unary(self):
        assert_same(-MSK, -NMSK)
        assert_same(+MSK, +NMSK)
        assert_same(abs(MSK), abs(NMSK))

    def test_repr_str_format(self):
        assert repr(MSK) == repr(NMSK) == "masked"
        assert str(MSK) == str(NMSK) == "--"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert format(MSK, "") == format(NMSK, "") == "--"
            assert format(MSK, ">5") == format(NMSK, ">5")

    def test_scalar_conversions(self):
        with pytest.warns(UserWarning):
            fx = float(MSK)
        with pytest.warns(UserWarning):
            fn = float(NMSK)
        assert np.isnan(fx) and np.isnan(fn)
        with pytest.raises(np.ma.MaskError):
            int(NMSK)
        with pytest.raises(np.ma.MaskError):
            int(MSK)
        assert bool(MSK) is bool(NMSK) is False
        assert complex(MSK) == complex(NMSK) == 0j

    def test_attributes(self):
        assert MSK.shape == NMSK.shape == ()
        assert MSK.ndim == 0
        assert MSK.dtype == NMSK.dtype
        assert bool(MSK.mask) is True
        assert float(MSK.data) == float(NMSK.data) == 0.0
        assert MSK.size == 1
        assert MSK is XMA.masked

    def test_singleton_identity_through_copy_pickle(self):
        assert MSK.copy() is MSK
        assert copy.copy(MSK) is MSK
        assert copy.deepcopy(MSK) is MSK
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            assert pickle.loads(pickle.dumps(MSK, protocol=proto)) is MSK

    def test_asmarray_and_asnumpy(self):
        assert MSK.asmarray() is NMSK
        assert xupy.asnumpy(MSK) is NMSK

    def test_getitem_returns_masked(self, dev):
        x, n = make(np.array([1.0, 2.0, 3.0]), [0, 1, 0], dev)
        assert x[1] is MSK and n[1] is NMSK
        for i in (0, 2, -1):
            rx, rn = x[i], n[i]
            assert isinstance(rx, np.generic) and type(rx) is type(rn)
            assert rx == rn
        x2, n2 = make(np.arange(6.0).reshape(2, 3), pattern((2, 3)), dev)
        assert_same(x2[0, 1], n2[0, 1])
        assert_same(x2[1, 0], n2[1, 0])

    def test_setitem_masked(self, dev):
        x, n = make(np.array([1.0, 2.0, 3.0]), None, dev)
        x[1] = MSK
        n[1] = NMSK
        assert_same(x, n, dev=dev)
        x[:] = MSK
        n[:] = NMSK
        assert_same(x, n, dev=dev)

    def test_equality_with_masked_array(self, dev):
        x, n = make(np.array([1.0, 2.0, 3.0]), [0, 1, 0], dev)
        assert_same(x == MSK, n == NMSK, dev=dev)
        assert_same(x != MSK, n != NMSK, dev=dev)
        assert_same(MSK == x, NMSK == n, dev=dev)


class TestMaskFunctions:
    @pytest.mark.parametrize("mask", [None, "none", "all", "pat"])
    def test_basic(self, mask, dev):
        shape = (2, 3)
        d = rdata(shape)
        m = {None: None, "none": np.zeros(shape, bool), "all": np.ones(shape, bool),
             "pat": pattern(shape)}[mask]
        x, n = make(d, m, dev)
        assert XMA.is_masked(x) == np.ma.is_masked(n)
        assert type(XMA.is_masked(x)) is type(np.ma.is_masked(n))
        assert XMA.isMaskedArray(x) is np.ma.isMaskedArray(n) is True
        assert XMA.isMA(x) is np.ma.isMA(n) is True
        same_mask(XMA.getmask(x), np.ma.getmask(n), dev)
        gma = XMA.getmaskarray(x)
        assert on_dev(gma, dev) and gma.dtype == np.bool_
        np.testing.assert_array_equal(host(gma), np.ma.getmaskarray(n))
        gd = XMA.getdata(x)
        assert on_dev(gd, dev) and not isinstance(gd, XMA.MaskedArray)
        np.testing.assert_array_equal(host(gd), np.ma.getdata(n))

    def test_nonmasked_inputs(self, dev):
        a = to_dev(rdata((3,)), dev)
        h = host(a)
        assert XMA.isMaskedArray(a) is False
        assert XMA.isMA(a) is False
        assert XMA.isMaskedArray(3) is False
        assert XMA.is_masked(a) == np.ma.is_masked(h) is False
        assert XMA.is_masked(3.0) is False
        assert XMA.getmask(a) is NM
        assert XMA.getmask([1, 2]) is NM
        gma = XMA.getmaskarray(a)
        assert on_dev(gma, dev) and gma.dtype == np.bool_ and not gma.any()
        assert gma.shape == (3,)
        assert on_dev(XMA.getdata(a), dev)
        np.testing.assert_array_equal(host(XMA.getdata([1, 2])), [1, 2])

    def test_numpy_ma_input_is_recognised_as_ma(self):
        n = np.ma.array([1.0, 2.0], mask=[0, 1])
        assert XMA.is_masked(n) is True
        assert XMA.isMaskedArray(n) is True
        np.testing.assert_array_equal(host(XMA.getmaskarray(n)), [False, True])

    def test_masked_singleton_helpers(self):
        assert XMA.is_masked(MSK) == np.ma.is_masked(NMSK)
        assert XMA.isMaskedArray(MSK) == np.ma.isMaskedArray(NMSK)

    def test_is_mask(self, dev):
        vals = [
        pytest.param(lambda d: NM, id="nomask"),
        pytest.param(lambda d: to_dev(np.array([True, False]), d), id="bool_arr"),
        pytest.param(lambda d: to_dev(np.array([1, 0]), d), id="int_arr"),
        pytest.param(lambda d: to_dev(np.array([1.0, 0.0]), d), id="float_arr"),
        pytest.param(lambda d: to_dev(np.array(True), d), id="bool0d"),
        pytest.param(lambda d: [True, False], id="list"),
        pytest.param(lambda d: True, id="pybool"),
        pytest.param(lambda d: np.True_, id="npbool"),
        pytest.param(lambda d: None, id="none"),
        ]
        for val in vals:
            with case(val.id):
                v = val.values[0](dev)
                vn = np.ma.nomask if v is NM else (host(v) if hasattr(v, "dtype") else v)
                try:
                    rn = np.ma.is_mask(vn)
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        XMA.is_mask(v)
                    continue
                assert XMA.is_mask(v) is rn

    def test_make_mask_none(self, dev):
        for shape in [(), (3,), (2, 3), (0,), (2, 0)]:
            with case(shape):
                rx = XMA.make_mask_none(shape)
                rn = np.ma.make_mask_none(shape)
                assert rx.shape == rn.shape and host(rx).dtype == np.bool_
                assert not host(rx).any()
                with xupy.backend("cpu"):
                    assert isinstance(XMA.make_mask_none(shape), np.ndarray)

    @needs_gpu
    def test_make_mask_none_gpu_backend(self):
        with xupy.backend("gpu"):
            r = XMA.make_mask_none((2, 3))
        assert isinstance(r, cp.ndarray) and r.dtype == np.bool_

    def test_mask_or(self, dev):
        for case_ in ["nm_nm", "nm_arr", "arr_nm", "arr_arr", "arr_arr_all_false"]:
            for shrink in (True, False):
                for copy_ in (False, True):
                    with case(case_, shrink, copy_):
                        self._mask_or_one(case_, shrink, copy_, dev)

    @staticmethod
    def _mask_or_one(case, shrink, copy_, dev):
        a = pattern((2, 3), 3, 0)
        b = pattern((2, 3), 3, 1)
        z = np.zeros((2, 3), bool)

        def args(arrs, mod):
            if arrs == "nm":
                return mod_nm[mod]
            h = {"a": a, "b": b, "z": z}[arrs]
            return to_dev(h, dev) if mod == "x" else h

        mod_nm = {"x": NM, "n": NNM}
        spec = {"nm_nm": ("nm", "nm"), "nm_arr": ("nm", "a"), "arr_nm": ("a", "nm"),
                "arr_arr": ("a", "b"), "arr_arr_all_false": ("z", "z")}[case]
        rn = np.ma.mask_or(args(spec[0], "n"), args(spec[1], "n"), copy=copy_, shrink=shrink)
        rx = XMA.mask_or(args(spec[0], "x"), args(spec[1], "x"), copy=copy_, shrink=shrink)
        same_mask(rx, rn, dev if rn is not NNM else None)

    def test_mask_or_does_not_alias_inputs(self, dev):
        a = to_dev(pattern((4,), 2, 0), dev)
        r = XMA.mask_or(a, NM, copy=True)
        r[0] = not bool(host(r)[0])
        assert host(a)[0] == True  # noqa: E712

    def test_getmaskarray_of_nomask_array_is_fresh(self, dev):
        x, _ = make(rdata((3,)), None, dev)
        g = XMA.getmaskarray(x)
        g[0] = True
        assert x.mask is NM


# ==========================================================================
# 3. constructor parity
# ==========================================================================
def try_make(data, mask, dev, **kw):
    """Like ``make`` but exceptions must agree; returns None if both raised."""
    try:
        n = np.ma.masked_array(np.array(data).copy(),
                               **({} if mask is None else {"mask": np.array(mask)}), **kw)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            mka(dev, to_dev(np.array(data).copy(), dev),
                             **({} if mask is None else {"mask": to_dev(np.array(mask), dev)}), **kw)
        return None
    x = mka(dev, to_dev(np.array(data).copy(), dev),
                         **({} if mask is None else {"mask": to_dev(np.array(mask), dev)}), **kw)
    return x, n


class TestConstructorMasks:
    @pytest.mark.parametrize("mask", [
        None, True, False, [0, 1, 0, 0, 1, 0], [[0, 1, 0], [0, 0, 1]], [0, 1, 0], [1, 0],
        [[1], [0]], np.zeros(6, bool), np.ones(6, bool), [0, 1, 0, 0],
    ], ids=lambda m: str(m).replace(" ", ""))
    def test_mask_forms(self, mask, dev):
        for shape in [(2, 3), (6,)]:
            with case(shape):
                d = np.arange(6.0).reshape(shape)
                if isinstance(mask, bool):
                    n = np.ma.masked_array(d.copy(), mask=mask)
                    x = mka(dev, to_dev(d.copy(), dev), mask=mask)
                    assert_same(x, n, dev=dev)
                    continue
                r = try_make(d, mask, dev)
                if r is not None:
                    assert_same(r[0], r[1], dev=dev)

    def test_wrong_size_mask_raises_maskerror(self, dev):
        with pytest.raises(np.ma.MaskError):
            mka(dev, to_dev(np.arange(3.0), dev), mask=to_dev(np.array([True, False]), dev))
        with pytest.raises(np.ma.MaskError):
            mka(dev, to_dev(np.arange(4.0), dev), mask=to_dev(np.zeros(3, bool), dev))

    def test_mask_nomask(self, dev):
        x = mka(dev, to_dev(np.arange(3.0), dev), mask=NM)
        n = np.ma.masked_array(np.arange(3.0), mask=NNM)
        assert_same(x, n, dev=dev)

    def test_shrink(self, dev):
        d = np.arange(3.0)
        for shrink in (True, False):
            for mask in (False, [0, 0, 0], [0, 1, 0], True):
                with case(shrink, mask):
                    x = mka(
                        dev,
                        to_dev(d, dev),
                        mask=mask if isinstance(mask, bool) else to_dev(np.array(mask), dev),
                        shrink=shrink)
                    n = np.ma.masked_array(d, mask=mask, shrink=shrink)
                    assert_same(x, n, dev=dev)

    def test_ndmin(self, dev):
        for ndmin in (0, 1, 2, 3):
            for mask in (None, [0, 1, 0]):
                with case(ndmin, mask):
                    r = try_make(np.arange(3.0), mask, dev, ndmin=ndmin)
                    assert_same(r[0], r[1], dev=dev)

    def test_dtype(self, dev):
        for dt in (float, "f4", np.int64, "c16", bool, np.dtype("i2"), "u1"):
            for mask in (None, [0, 1, 0]):
                with case(dt, mask):
                    x, n = try_make(np.array([1, 2, 3]), mask, dev, dtype=dt)
                    assert isinstance(x.dtype, np.dtype) and x.dtype == n.dtype
                    assert_same(x, n, dev=dev)

    @pytest.mark.parametrize("dt", ["i4", "f4", "?", "u1"])
    def test_fill_value_argument(self, dt, dev):
        d = np.array([1, 0, 3]).astype(dt)
        for fv in (None, 7, 1.5, -1, 0, True, np.int64(3)):
            with case(dt, fv):
                r = try_make(d, [0, 1, 0], dev, fill_value=fv)
                if r:
                    assert_same(r[0], r[1], dev=dev)

    def test_hard_mask(self, dev):
        for hard in (True, False):
            for mask in (None, [0, 1, 0]):
                with case(hard, mask):
                    r = try_make(np.arange(3.0), mask, dev, hard_mask=hard)
                    assert_same(r[0], r[1], dev=dev)
                    assert bool(r[0].hardmask) is hard

    @pytest.mark.parametrize("copy_", [True, False])
    def test_copy_semantics_numpy_input(self, copy_, dev):
        d = np.arange(4.0)
        dd = to_dev(d.copy(), dev)
        x = mka(dev, dd, copy=copy_)
        x[0] = 99.0
        if copy_:
            assert host(dd)[0] == 0.0
        else:
            assert host(dd)[0] == 99.0
        n_in = d.copy()
        n = np.ma.masked_array(n_in, copy=copy_)
        n[0] = 99.0
        assert (n_in[0] == 99.0) == (not copy_)

    @pytest.mark.parametrize("copy_", [True, False])
    def test_copy_semantics_mask_input(self, copy_, dev):
        m = pattern((4,), 2, 0)
        mm = to_dev(m.copy(), dev)
        x = mka(dev, to_dev(np.arange(4.0), dev), mask=mm, copy=copy_)
        x[1] = XMA.masked
        n_m = m.copy()
        n = np.ma.masked_array(np.arange(4.0), mask=n_m, copy=copy_)
        n[1] = np.ma.masked
        assert (host(mm)[1]) == n_m[1]

    @pytest.mark.parametrize("src", ["xupy", "npma"])
    def test_keep_mask(self, src, dev):
        for keep_mask in (True, False):
            for new_mask in (None, [1, 0, 0], [0, 0, 1]):
                with case(keep_mask, new_mask):
                    self._keep_mask_one(keep_mask, new_mask, src, dev)

    @staticmethod
    def _keep_mask_one(keep_mask, new_mask, src, dev):
        d = np.arange(3.0)
        n0 = np.ma.masked_array(d.copy(), mask=[0, 1, 0])
        x0 = mka(dev, to_dev(d.copy(), dev), mask=to_dev(np.array([0, 1, 0], bool), dev))
        kw = {} if new_mask is None else {"mask": np.array(new_mask)}
        kx = {} if new_mask is None else {"mask": to_dev(np.array(new_mask), dev)}
        n = np.ma.masked_array(n0, keep_mask=keep_mask, **kw)
        # a numpy.ma source (and its host mask override) goes to the active backend
        x = mka(dev, x0 if src == "xupy" else n0, keep_mask=keep_mask,
                **({} if src == "npma" or new_mask is None else kx),
                **({"mask": np.array(new_mask)} if src == "npma" and new_mask is not None else {}))
        # ``_mask`` must always be set (old keep_mask=False bug) and the mask readable
        assert x.shape == (3,)
        _ = x.mask
        assert_same(x, n, dev=dev)

    def test_keep_mask_false_with_nothing_else(self, dev):
        x0 = mka(dev, to_dev(np.arange(3.0), dev), mask=to_dev(np.array([0, 1, 0], bool), dev))
        n0 = np.ma.masked_array(np.arange(3.0), mask=[0, 1, 0])
        x = mka(dev, x0, keep_mask=False)
        n = np.ma.masked_array(n0, keep_mask=False)
        assert_same(x, n, dev=dev)
        # and the new array is fully usable
        assert_same(x + 1, n + 1, dev=dev)
        assert_same(x.sum(), n.sum())

    def test_hard_mask_from_existing(self, dev):
        n0 = np.ma.masked_array(np.arange(3.0), mask=[0, 1, 0], hard_mask=True, fill_value=5.0)
        x0 = mka(dev, to_dev(np.arange(3.0), dev), mask=to_dev(np.array([0, 1, 0], bool), dev),
                              hard_mask=True, fill_value=5.0)
        assert_same(XMA.masked_array(x0), np.ma.masked_array(n0), dev=dev)
        assert_same(XMA.masked_array(x0, hard_mask=False), np.ma.masked_array(n0, hard_mask=False), dev=dev)
        assert_same(XMA.masked_array(x0, fill_value=2.0), np.ma.masked_array(n0, fill_value=2.0), dev=dev)

    def test_data_aliasing_when_copy_false(self, dev):
        x0, n0 = make(np.arange(4.0), [0, 1, 0, 0], dev)
        x1 = XMA.masked_array(x0, copy=False)
        n1 = np.ma.masked_array(n0, copy=False)
        x1[0] = 77.0
        n1[0] = 77.0
        assert_same(x0, n0, dev=dev)


class TestConstructorSources:
    def test_from_numpy_array(self, dev):
        d = np.arange(4.0)
        x = mka(dev, d)
        assert on_dev(x.data, dev)
        # default backend (GPU when usable): host input is transferred
        default = XMA.masked_array(d)
        assert isinstance(default.data, cp.ndarray if GPU_OK else np.ndarray)

    def test_from_nested_lists_cpu_backend(self):
        with xupy.backend("cpu"):
            x = XMA.masked_array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]])
            assert isinstance(x.data, np.ndarray) and isinstance(x.mask, np.ndarray)
        n = np.ma.masked_array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]])
        assert_same(x, n, dev="cpu")

    @needs_gpu
    def test_from_nested_lists_gpu_backend(self):
        with xupy.backend("gpu"):
            x = XMA.masked_array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]])
        n = np.ma.masked_array([[1, 2], [3, 4]], mask=[[0, 1], [0, 0]])
        assert isinstance(x.data, cp.ndarray) and isinstance(x.mask, cp.ndarray)
        assert_same(x, n, dev="gpu")

    @pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=needs_gpu)])
    def test_from_scalar_follows_active_backend(self, backend):
        for value in (3, 2.5, True, 1 + 2j, np.float32(1.5), np.int8(3), np.array(4.0)):
            with case(value), xupy.backend(backend):
                x = XMA.masked_array(value)
                n = np.ma.masked_array(value)
                assert x.shape == ()
                # host scalars, including a numpy 0-d array, go to the active backend
                edev = "gpu" if backend == "gpu" else "cpu"
                assert on_dev(x.data, edev)
                assert_same(x, n, dev=edev)

    @pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=needs_gpu)])
    def test_from_list_with_mixed_types(self, backend):
        with xupy.backend(backend):
            x = XMA.masked_array([1, 2.5, 3], mask=[False, True, False])
        n = np.ma.masked_array([1, 2.5, 3], mask=[False, True, False])
        assert_same(x, n, dev="gpu" if backend == "gpu" else "cpu")

    @pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=needs_gpu)])
    def test_empty_list(self, backend):
        with xupy.backend(backend):
            x = XMA.masked_array([])
        assert_same(x, np.ma.masked_array([]), dev="gpu" if backend == "gpu" else "cpu")

    @pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=needs_gpu)])
    def test_numpy_input_goes_to_active_backend(self, backend):
        with xupy.backend(backend):
            x = XMA.masked_array(np.arange(4.0), mask=np.array([0, 1, 0, 0], bool))
        edev = "gpu" if backend == "gpu" else "cpu"
        assert on_dev(x.data, edev) and on_dev(x.mask, edev)
        np.testing.assert_array_equal(host(x.mask), [0, 1, 0, 0])

    @pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=needs_gpu)])
    def test_np_ma_input_goes_to_active_backend(self, backend):
        n0 = np.ma.masked_array(np.arange(4.0), mask=[0, 1, 0, 0])
        with xupy.backend(backend):
            x = XMA.masked_array(n0)
        edev = "gpu" if backend == "gpu" else "cpu"
        assert on_dev(x.data, edev) and on_dev(x.mask, edev)  # the numpy.ma mask moves too
        assert_same(x, n0, dev=edev)

    @needs_gpu
    @pytest.mark.parametrize("backend", ["cpu", "gpu"])
    def test_cupy_input_stays_cupy(self, backend):
        with xupy.backend(backend):
            x = XMA.masked_array(cp.arange(4.0), mask=cp.asarray([0, 1, 0, 0], bool))
        assert isinstance(x.data, cp.ndarray) and isinstance(x.mask, cp.ndarray)

    @needs_gpu
    @pytest.mark.parametrize("backend", ["cpu", "gpu"])
    def test_mask_follows_data_device(self, backend):
        with xupy.backend(backend):
            # cupy data stays on the GPU and drags a host mask along, whatever the backend
            x = XMA.masked_array(cp.arange(4.0), mask=np.array([0, 1, 0, 0], bool))
            assert isinstance(x.data, cp.ndarray) and isinstance(x.mask, cp.ndarray)
            # host data and host mask go to the active backend together
            y = XMA.masked_array(np.arange(4.0), mask=[0, 1, 0, 0])
            xp = cp if backend == "gpu" else np
            assert isinstance(y.data, xp.ndarray) and isinstance(y.mask, xp.ndarray)

    @needs_gpu
    def test_existing_numpy_xupy_array_stays_numpy_under_gpu_backend(self):
        with xupy.backend("cpu"):
            x0 = XMA.masked_array(np.arange(4.0), mask=[0, 1, 0, 0])
        with xupy.backend("gpu"):
            x1 = XMA.masked_array(x0)
            x2 = XMA.array(x0)
            x3 = XMA.zeros_like(x0)
            x4 = XMA.masked_all_like(x0)
        for x in (x1, x2, x3, x4):
            assert isinstance(x.data, np.ndarray) and isinstance(x.mask, np.ndarray)

    def test_from_xupy_array_keeps_device(self, dev):
        x0, n0 = make(np.arange(4.0), [0, 1, 0, 0], dev)
        assert_same(XMA.masked_array(x0), np.ma.masked_array(n0), dev=dev)
        assert_same(XMA.array(x0), np.ma.array(n0), dev=dev)

    def test_from_np_ma_with_mask_override(self, dev):
        n0 = np.ma.masked_array(np.arange(4.0), mask=[0, 1, 0, 0])
        x = mka(dev, n0, mask=[0, 0, 1, 0])
        n = np.ma.masked_array(n0, mask=[0, 0, 1, 0])
        assert_same(x, n, dev=dev)

    def test_from_ma_masked_constant(self):
        n = np.ma.masked_array(NMSK)
        for dev in ("cpu", "gpu") if GPU_OK else ("cpu",):
            x = mka(dev, MSK)
            assert x.shape == n.shape
            assert_same(XMA.getmaskarray(x), np.ma.getmaskarray(n), dev=dev)

    def test_dtype_property_is_np_dtype(self, dev):
        d = to_dev(np.arange(3), dev)
        for dtype in (float, "f4", int, "i2", np.float32, np.dtype("f8"), complex, bool):
            with case(dtype):
                x = XMA.masked_array(d, dtype=dtype)
                assert type(x.dtype) is type(np.dtype(dtype))
                assert x.dtype == np.dtype(dtype)
                assert type(XMA.masked_array(d).dtype) is type(d.dtype)
                y = x.astype(dtype)
                assert isinstance(y.dtype, np.dtype) and y.dtype == np.dtype(dtype)

    def test_dtype_property_is_np_dtype_lists(self):
        for dt in (float, "f4", int):
            with xupy.backend("cpu"):
                x = XMA.masked_array([1, 2, 3], dtype=dt)
            assert isinstance(x.dtype, np.dtype)


class TestAstypeAndCopy:
    @pytest.mark.parametrize("mask", [None, [0, 1, 0]])
    def test_astype_never_aliases(self, mask, dev):
        for target in ("f4", "f8", "i4", "c16", "?", float):
            with case(target):
                x, n = make(np.array([1.0, 2.0, 3.0]), mask, dev)
                y = x.astype(target, copy=True)
                m = n.astype(target, copy=True)
                assert_same(y, m, dev=dev)
                y[0] = 99
                y[1] = XMA.masked
                m[0] = 99
                m[1] = np.ma.masked
                assert_same(x, n, dev=dev)
                assert_same(y, m, dev=dev)

    @pytest.mark.parametrize("target", ["f8", float])
    def test_astype_default_copy_independent(self, target, dev):
        x, n = make(np.array([1.0, 2.0, 3.0]), [0, 1, 0], dev)
        y, m = x.astype(target), n.astype(target)
        y[0] = 55.0
        m[0] = 55.0
        assert_same(x, n, dev=dev)
        assert_same(y, m, dev=dev)
        # mask independent
        y[2] = XMA.masked
        m[2] = np.ma.masked
        assert_same(x, n, dev=dev)

    def test_astype_copy_false_same_dtype_aliases(self, dev):
        x, n = make(np.array([1.0, 2.0, 3.0]), [0, 1, 0], dev)
        y, m = x.astype("f8", copy=False), n.astype("f8", copy=False)
        y[0] = 42.0
        m[0] = 42.0
        assert_same(x, n, dev=dev)
        assert (y is x) == (m is n)

    def test_astype_copy_false_other_dtype_copies(self, dev):
        x, n = make(np.array([1.0, 2.0, 3.0]), [0, 1, 0], dev)
        y, m = x.astype("f4", copy=False), n.astype("f4", copy=False)
        y[0] = 42.0
        m[0] = 42.0
        assert_same(x, n, dev=dev)
        assert_same(y, m, dev=dev)

    def test_astype_keeps_fill_and_hardmask(self, dev):
        x, n = make(np.array([1.0, 2.0, 3.0]), [0, 1, 0], dev, fill_value=7.0, hard_mask=True)
        assert_same(x.astype("f4"), n.astype("f4"), dev=dev)
        assert_same(x.astype("i4"), n.astype("i4"), dev=dev)

    def test_astype_invalid(self, dev):
        x, n = make(np.array([1.0, 2.0]), None, dev)
        for bad in ("not_a_dtype", "f4x"):
            with pytest.raises(TypeError):
                n.astype(bad)
            with pytest.raises(TypeError):
                x.astype(bad)

    def test_copy_is_independent(self, dev):
        x, n = make(np.arange(4.0), [0, 1, 0, 0], dev)
        y, m = x.copy(), n.copy()
        y[0] = 9.0
        y[3] = XMA.masked
        m[0] = 9.0
        m[3] = np.ma.masked
        assert_same(x, n, dev=dev)
        assert_same(y, m, dev=dev)

    @pytest.mark.parametrize("fv,hard", [(7.0, True), (None, False)])
    def test_fill_and_hardmask_propagation(self, fv, hard, dev):
        kw = {"hard_mask": hard}
        if fv is not None:
            kw["fill_value"] = fv
        x, n = make(rdata((2, 3)), pattern((2, 3)), dev, **kw)
        fs = {
            "copy": lambda M, a: a.copy(),
            "view": lambda M, a: a.view(),
            "neg": lambda M, a: -a,
            "add": lambda M, a: a + 1,
            "slice": lambda M, a: a[1:],
            "reshape": lambda M, a: a.reshape(3, 2),
            "T": lambda M, a: a.T,
            "ravel": lambda M, a: a.ravel(),
            "astype": lambda M, a: a.astype("f4"),
            "take": lambda M, a: a.take([0, 1]),
            "filled_arr": lambda M, a: M.getdata(a),
            "mul_arr": lambda M, a: a * a,
            "np_sqrt": lambda M, a: np.sqrt(a),
            "getitem_list": lambda M, a: a[[0, 1]],
            "squeeze": lambda M, a: a.squeeze(),
            "swapaxes": lambda M, a: a.swapaxes(0, 1),
            "repeat": lambda M, a: a.repeat(2),
            "M_array": lambda M, a: M.array(a),
            "M_copy": lambda M, a: M.array(a, copy=True),
            "sort": lambda M, a: _sorted(a),
            "cumsum": lambda M, a: a.cumsum(axis=1),
            "abs": lambda M, a: abs(a),
            "clip": lambda M, a: a.clip(0.7, 1.2),
            "round": lambda M, a: a.round(1),
        }
        for name, f in fs.items():
            with case(name):
                run_pair(f, x, n, dev)

    def test_view_with_dtype_and_type(self, dev):
        x, n = make(np.arange(4, dtype="i4"), [0, 1, 0, 0], dev)
        run_pair(lambda M, a: a.view("u4"), x, n, dev)
        run_pair(lambda M, a: a.view(dtype="f4"), x, n, dev)


class TestFillValueCheck:
    def test_default_fill_value(self, dev):
        for dt in ["i1", "i4", "i8", "u1", "u8", "f2", "f4", "f8", "c8", "c16", "?",
                   "U3", "S3", "M8[s]", "m8[s]"]:
            if np.dtype(dt).kind in "USMmO":
                # XuPy ma supports numeric and bool dtypes only (settled decision),
                # whichever backend the host array is built on.
                with pytest.raises(NotImplementedError):
                    mka(dev, np.zeros(2, dt))
                continue
            with case(dt):
                x = mka(dev, to_dev(np.zeros(2, dt), dev))
                n = np.ma.masked_array(np.zeros(2, dt))
                fx, fn = x.fill_value, n.fill_value
                assert type(fx) is type(fn)
                np.testing.assert_array_equal(np.asarray(fx), np.asarray(fn))

    def test_fill_value_validation(self, dev):
        cases = [
            (1.5, "i4"), (300, "i1"), (-1, "u1"), (256, "u1"), ("abc", "f8"), ("1.5", "f8"),
            (1e40, "f4"), (np.inf, "f4"), (np.nan, "f8"), (np.nan, "i4"), (None, "i4"),
            ([1, 2], "i4"), ([1, 2], "f8"), (np.array(3.0), "i4"), (np.array([1.0, 2.0]), "f8"),
            (2 ** 70, "i8"), (1 + 2j, "f8"), (1 + 2j, "c16"), (True, "i4"), (3, "?"), ("x", "U3"),
            ("toolong", "U3"), (5, "U3"), (1.0, "c8"), (-0.0, "f8"), (1e400, "f8"),
        ]
        for fv, dt in cases:
            if np.dtype(dt).kind in "USMmO":
                # XuPy ma supports numeric and bool dtypes only (settled decision),
                # whichever backend the host array is built on.
                d = np.array(["a", "b", "c"])
                with pytest.raises(NotImplementedError):
                    mka(dev, d, fill_value=fv)
                continue
            with case(fv, dt):
                d = np.arange(3).astype(dt)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        n = np.ma.masked_array(d.copy(), fill_value=fv)
                        fn = n.fill_value
                    except Exception as e:  # noqa: BLE001
                        with pytest.raises(type(e)):
                            mka(dev, to_dev(d.copy(), dev), fill_value=fv)
                        continue
                    x = mka(dev, to_dev(d.copy(), dev), fill_value=fv)
                    fx = x.fill_value
                assert type(fx) is type(fn)
                np.testing.assert_array_equal(np.asarray(fx), np.asarray(fn))

    def test_fill_value_setter(self, dev):
        for fv, dt in [(1.5, "i4"), (300, "i1"), ("a", "f8"), (7, "f4"), (None, "f8"),
                       (2.5, "i4"), ([1, 2], "f8")]:
            with case(fv, dt):
                d = np.arange(3).astype(dt)
                x, n = make(d, [0, 1, 0], dev)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        n.fill_value = fv
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        x.fill_value = fv
                    assert_same(x, n, dev=dev)
                    continue
                x.fill_value = fv
                assert_same(x, n, dev=dev)

    def test_fill_value_is_numpy_scalar(self, dev):
        x, n = make(np.arange(3.0), None, dev, fill_value=2.0)
        assert isinstance(x.fill_value, np.generic)
        assert type(x.fill_value) is type(n.fill_value)

    def test_filled_uses_fill_value_cast(self, dev):
        x, n = make(np.arange(3), [0, 1, 0], dev, fill_value=1.5)
        assert_same(x.filled(), n.filled(), dev=dev)
        assert_same(x.filled(8), n.filled(8), dev=dev)

    def test_filled_invalid_fill(self, dev):
        x, n = make(np.arange(3), [0, 1, 0], dev)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rn = n.filled("abc")
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x.filled("abc")
        else:
            assert_same(x.filled("abc"), rn, dev=dev)

    def test_default_fill_value_function(self):
        for obj in (1, 1.0, 1j, True, "a", np.float32(1), np.arange(3), np.zeros(2, "i1"), [1.0]):
            assert XMA.default_fill_value(obj) == np.ma.default_fill_value(obj)
            assert type(XMA.default_fill_value(obj)) is type(np.ma.default_fill_value(obj))


class TestMaskSetter:
    @pytest.mark.parametrize("hard", [False, True])
    def test_assign(self, hard, dev):
        values = [
            True, False, [0, 0, 0, 0], [True, False, False, True], np.array([1, 1, 0, 0], bool),
            np.array(True), [True], [True, False], np.ones(4, bool), NM, np.zeros(4, bool),
            [[1, 0], [0, 1]], 1, 0,
        ]
        for start in (None, [0, 1, 0, 0]):
            for value in values:
                with case(start, value if value is not NM else "nomask"):
                    x, n = make(np.arange(4.0), start, dev, hard_mask=hard)
                    vn = NNM if value is NM else value
                    vx = NM if value is NM else (to_dev(value, dev) if isinstance(value, np.ndarray) else value)
                    try:
                        n.mask = vn
                    except Exception as e:  # noqa: BLE001
                        with pytest.raises(type(e)):
                            x.mask = vx
                        continue
                    x.mask = vx
                    assert_same(x, n, dev=dev)
                    if value is not NM:
                        assert on_dev(x.mask, dev) or x.mask is NM

    def test_assign_2d_broadcast(self, dev):
        for v in ([True, False, True], [[True], [False]], True, np.array([[1, 0, 1], [0, 0, 1]], bool)):
            x, n = make(np.zeros((2, 3)), None, dev)
            n.mask = v
            x.mask = to_dev(v, dev) if isinstance(v, np.ndarray) else v
            assert_same(x, n, dev=dev)

    def test_assign_wrong_shape(self, dev):
        x, n = make(np.zeros((2, 3)), None, dev)
        v = np.zeros(4, bool)
        try:
            n.mask = v
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x.mask = to_dev(v, dev)
        else:
            x.mask = to_dev(v, dev)
            assert_same(x, n, dev=dev)

    def test_mask_setter_does_not_alias_value(self, dev):
        x, n = make(np.arange(4.0), None, dev)
        v = to_dev(np.array([1, 0, 0, 0], bool), dev)
        x.mask = v
        x[1] = XMA.masked
        assert not host(v)[1]

    def test_mask_assign_nomask_on_masked_array(self, dev):
        x, n = make(np.arange(4.0), [1, 0, 0, 0], dev)
        x.mask = NM
        n.mask = NNM
        assert_same(x, n, dev=dev)

    def test_mask_property_is_live_view_like_numpy(self, dev):
        x, n = make(np.arange(4.0), [1, 0, 0, 0], dev)
        mx, mn = x.mask, n.mask
        mx[1] = True
        mn[1] = True
        assert_same(x, n, dev=dev)

    def test_set_mask_nomask_hardmask(self, dev):
        x, n = make(np.arange(4.0), [1, 0, 0, 0], dev, hard_mask=True)
        x.mask = np.zeros(4, bool) if dev == "cpu" else cp.zeros(4, bool)
        n.mask = np.zeros(4, bool)
        assert_same(x, n, dev=dev)


class TestToDevice:
    def test_to_cpu(self, dev):
        x, n = make(np.arange(4.0), [0, 1, 0, 0], dev, fill_value=3.0, hard_mask=True)
        y = x.to_device("cpu")
        assert_same(y, n, dev="cpu")

    def test_to_cpu_nomask(self, dev):
        x, n = make(np.arange(4.0), None, dev)
        y = x.to_device("cpu")
        assert y.mask is NM
        assert_same(y, n, dev="cpu")

    @needs_gpu
    @pytest.mark.parametrize("name", ["gpu", "cuda"])
    def test_to_gpu(self, name):
        x, n = make(np.arange(4.0), [0, 1, 0, 0], "cpu", fill_value=3.0, hard_mask=True)
        y = x.to_device(name)
        assert_same(y, n, dev="gpu")
        z = y.to_device("cpu")
        assert_same(z, n, dev="cpu")

    @needs_gpu
    def test_to_gpu_device_object_and_int(self):
        x, n = make(np.arange(4.0), [0, 1, 0, 0], "cpu")
        assert_same(x.to_device(cp.cuda.Device(0)), n, dev="gpu")
        assert_same(x.to_device(0), n, dev="gpu")

    @needs_gpu
    def test_to_device_copies_when_moving_and_is_independent(self):
        dev = "gpu"
        x, n = make(np.arange(4.0), [0, 1, 0, 0], dev)
        y = x.to_device("cpu")
        y[0] = 100.0
        assert_same(x, n, dev=dev)

    def test_invalid_device(self, dev):
        x, _ = make(np.arange(4.0), None, dev)
        with pytest.raises((ValueError, TypeError)):
            x.to_device("tpu")


# ==========================================================================
# 4. indexing
# ==========================================================================
IDX_1D = {
    "int": 2, "neg": -1, "zero": 0, "slice": slice(1, 4), "slice_step": slice(None, None, 2),
    "slice_neg_step": slice(None, None, -1), "empty_slice": slice(3, 3), "ellipsis": Ellipsis,
    "newaxis": None, "tuple_ell": (Ellipsis, slice(1, 3)), "tuple_new": (None, slice(None)),
    "bool": np.array([1, 0, 1, 0, 0, 1], bool), "bool_all_false": np.zeros(6, bool),
    "fancy": np.array([0, 2, 5]), "fancy_neg": np.array([-1, 0]), "fancy_2d": np.array([[0, 1], [2, 3]]),
    "fancy_repeat": np.array([1, 1, 1]), "fancy_empty": np.array([], dtype=int),
    "list": [0, 2, 5], "list_bool": [True, False, True, False, False, True],
    "oob": 6, "oob_neg": -7, "oob_fancy": np.array([0, 9]), "too_many": (1, 2),
    "bool_wrong_len": np.array([True, False]), "str": "a", "float": 1.0, "tuple_empty": (),
    "int0d": np.array(2), "np_int": np.int64(3),
}
IDX_2D = {
    "int": 1, "row_neg": -1, "int_int": (1, 2), "neg_neg": (-1, -1), "slice_int": (slice(None), 1),
    "int_slice": (0, slice(1, None)), "slice_slice": (slice(0, 2), slice(1, 3)),
    "ellipsis": (Ellipsis, 0), "ell_only": Ellipsis, "newaxis": (None, 0, slice(None)),
    "newaxis_mid": (slice(None), None, 1), "bool2d": pattern((3, 4), 2, 0), "bool_row": np.array([1, 0, 1], bool),
    "fancy_rows": np.array([0, 2]), "fancy_pair": (np.array([0, 1, 2]), np.array([3, 2, 1])),
    "fancy_slice": (np.array([0, 2]), slice(1, 3)), "slice_fancy": (slice(None), np.array([0, 3])),
    "fancy_bcast": (np.array([[0], [1]]), np.array([0, 3])), "oob_row": 3, "oob_col": (0, 4),
    "too_many": (0, 0, 0), "list_rows": [0, 1], "int_fancy": (1, np.array([0, 2])),
    "two_ellipsis": (Ellipsis, Ellipsis),
}


def _mk_idx(idx, dev):
    return dev_index(idx, dev)


class TestGetitem:
    @pytest.mark.parametrize("mask", [None, "pat", "all", "none"])
    def test_1d(self, mask, dev):
        d = rdata((6,))
        m = {None: None, "pat": pattern((6,)), "all": np.ones(6, bool), "none": np.zeros(6, bool)}[mask]
        x, n = make(d, m, dev)
        for key, idx in IDX_1D.items():
            if dev == "gpu" and key == "oob_fancy":
                # Known cupy limitation: out-of-bounds integer-array indices do not
                # raise on cupy (checking would need a host sync); still checked on cpu.
                continue
            with case(key):
                run_pair(lambda M, a: a[_mk_idx(idx, dev) if a is x else idx], x, n, dev)

    @pytest.mark.parametrize("mask", [None, "pat"])
    def test_2d(self, mask, dev):
        d = rdata((3, 4))
        x, n = make(d, None if mask is None else pattern((3, 4)), dev)
        for key, idx in IDX_2D.items():
            with case(key):
                run_pair(lambda M, a: a[_mk_idx(idx, dev) if a is x else idx], x, n, dev)

    def test_scalar_types(self, dev):
        for dt in ("f8", "f4", "i4", "u1", "?", "c16"):
            with case(dt):
                x, n = make(np.arange(5).astype(dt), [0, 1, 0, 0, 0], dev)
                for i in (0, 1, -1):
                    assert_same(x[i], n[i])
                assert x[1] is MSK

    def test_scalar_0d_array(self, dev):
        x, n = make(np.array(3.5), None, dev)
        run_pair(lambda M, a: a[()], x, n, dev)
        run_pair(lambda M, a: a[...], x, n, dev)
        run_pair(lambda M, a: a[None], x, n, dev)
        run_pair(lambda M, a: a[0], x, n, dev)
        xm, nm_ = make(np.array(3.5), np.array(True), dev)
        run_pair(lambda M, a: a[()], xm, nm_, dev)
        run_pair(lambda M, a: a[...], xm, nm_, dev)

    def test_3d_mixed(self, dev):
        d = rdata((2, 3, 4))
        x, n = make(d, pattern((2, 3, 4)), dev)
        for idx in [(1, Ellipsis, 2), (slice(None), 1), (Ellipsis, None), (0, slice(None), slice(None, None, 2)),
                    (1, 2, 3), (np.array([0, 1]), slice(None), np.array([1, 2]))]:
            run_pair(lambda M, a, idx=idx: a[_mk_idx(idx, dev) if a is x else idx], x, n, dev)

    def test_masked_array_as_index(self, dev):
        d = rdata((6,))
        x, n = make(d, pattern((6,)), dev)
        bx, bn = make(np.arange(6) % 2 == 0, None, dev)
        run_pair(lambda M, a: a[bx if a is x else bn], x, n, dev)

    def test_xupy_bool_comparison_index(self, dev):
        x, n = make(rdata((6,)), pattern((6,)), dev)
        gx = x[x > 1.0]
        gn = n[n > 1.0]
        assert_same(gx, gn, dev=dev)

    def test_getitem_result_mask_nomask_ness(self, dev):
        x, n = make(rdata((3, 4)), None, dev)
        assert x[0].mask is NM and n[0].mask is NNM
        assert x[:, 1:].mask is NM
        xm, nm_ = make(rdata((3, 4)), pattern((3, 4)), dev)
        assert xm[0].mask is not NM and nm_[0].mask is not NNM
        xz, nz = make(rdata((3, 4)), np.zeros((3, 4), bool), dev, shrink=False)
        assert_same(xz[1], nz[1], dev=dev)

    def test_basic_slices_share_memory(self, dev):
        for key in ["slice", "slice_step", "ellipsis", "tuple_ell", "newaxis"]:
            with case(key):
                x, n = make(np.arange(6.0), [0, 1, 0, 0, 0, 0], dev)
                idx = IDX_1D[key]
                vx, vn = x[idx], n[idx]
                vx.data[...] = 77.0
                vn.data[...] = 77.0
                assert_same(x, n, dev=dev)

    def test_basic_slice_setitem_writes_base(self, dev):
        for key in ["slice", "slice_step", "ellipsis", "tuple_ell", "newaxis"]:
            with case(key):
                x, n = make(np.arange(6.0), [0, 1, 0, 0, 0, 0], dev)
                idx = IDX_1D[key]
                vx, vn = x[idx], n[idx]
                vx[..., 0] = 55.0
                vn[..., 0] = 55.0
                assert_same(x, n, dev=dev)
                assert_same(vx, vn, dev=dev)

    def test_2d_slice_write_through(self, dev):
        x, n = make(np.arange(12.0).reshape(3, 4), pattern((3, 4)), dev)
        vx, vn = x[1:, ::2], n[1:, ::2]
        vx[0, 0] = -5.0
        vn[0, 0] = -5.0
        vx[1, 1] = XMA.masked
        vn[1, 1] = np.ma.masked
        assert_same(x, n, dev=dev)
        assert_same(vx, vn, dev=dev)

    def test_fancy_index_copies(self, dev):
        x, n = make(np.arange(6.0), [0, 1, 0, 0, 0, 0], dev)
        vx, vn = x[[0, 1, 2] if dev == "cpu" else cp.asarray([0, 1, 2])], n[[0, 1, 2]]
        vx[0] = 100.0
        vn[0] = 100.0
        assert_same(x, n, dev=dev)
        assert_same(vx, vn, dev=dev)

    def test_bool_index_copies(self, dev):
        x, n = make(np.arange(6.0), [0, 1, 0, 0, 0, 0], dev)
        b = np.array([1, 1, 0, 0, 0, 1], bool)
        vx, vn = x[to_dev(b, dev)], n[b]
        vx[0] = 100.0
        vn[0] = 100.0
        assert_same(x, n, dev=dev)

    def test_getitem_keeps_dtype_and_fill(self, dev):
        x, n = make(np.arange(6, dtype="i2"), [0, 1, 0, 0, 0, 0], dev, fill_value=5, hard_mask=True)
        assert_same(x[1:4], n[1:4], dev=dev)
        assert_same(x[[0, 2] if dev == "cpu" else cp.asarray([0, 2])], n[[0, 2]], dev=dev)

    def test_structured_field_access_unsupported_on_plain(self, dev):
        x, n = make(np.arange(3.0), None, dev)
        with pytest.raises(IndexError):
            n["a"]
        with pytest.raises(IndexError):
            x["a"]


def _val_scalar(shape, dev):
    return 9.0, 9.0


def _val_int(shape, dev):
    return 4, 4


def _val_masked(shape, dev):
    return MSK, NMSK


def _val_np0d(shape, dev):
    return to_dev(np.array(6.5), dev), np.array(6.5)


def _val_array_exact(shape, dev):
    v = rdata(shape, seed=5) * 10
    return to_dev(v, dev), v.copy()


def _val_array_host(shape, dev):
    v = rdata(shape, seed=6) * 10
    return v.copy(), v.copy()


def _val_array_row(shape, dev):
    sh = shape[-1:] if len(shape) else ()
    v = rdata(sh, seed=7) * 10
    return to_dev(v, dev), v.copy()


def _val_list(shape, dev):
    v = (rdata(shape, seed=8) * 10).tolist()
    return v, copy.deepcopy(v)


def _val_ma_exact(shape, dev):
    v = rdata(shape, seed=9) * 10
    m = pattern(shape, 2, 0) if len(shape) else np.array(True)
    n = np.ma.masked_array(v.copy(), mask=m.copy())
    x = mka(dev, to_dev(v.copy(), dev), mask=to_dev(m.copy(), dev))
    return x, n


def _val_ma_nomask(shape, dev):
    v = rdata(shape, seed=10) * 10
    n = np.ma.masked_array(v.copy())
    x = mka(dev, to_dev(v.copy(), dev))
    return x, n


def _val_npma(shape, dev):
    v = rdata(shape, seed=11) * 10
    m = pattern(shape, 2, 1) if len(shape) else np.array(False)
    n = np.ma.masked_array(v.copy(), mask=m.copy())
    return n, n.copy()


def _val_ma_row(shape, dev):
    sh = shape[-1:] if len(shape) else ()
    v = rdata(sh, seed=12) * 10
    m = pattern(sh, 2, 0) if len(sh) else np.array(True)
    n = np.ma.masked_array(v.copy(), mask=m.copy())
    x = mka(dev, to_dev(v.copy(), dev), mask=to_dev(m.copy(), dev))
    return x, n


def _val_nan(shape, dev):
    return np.nan, np.nan


VALUES = {
    "scalar": _val_scalar, "int": _val_int, "masked": _val_masked, "np0d": _val_np0d,
    "array": _val_array_exact, "array_host": _val_array_host, "array_row": _val_array_row,
    "list": _val_list, "ma": _val_ma_exact, "ma_nomask": _val_ma_nomask, "np_ma": _val_npma,
    "ma_row": _val_ma_row, "nan": _val_nan,
}

SET_KEYS_1D = ["int", "neg", "slice", "slice_step", "ellipsis", "bool", "fancy", "fancy_repeat",
               "list", "oob", "empty_slice", "bool_all_false", "slice_neg_step", "fancy_neg"]
SET_KEYS_2D = ["int", "int_int", "slice_int", "slice_slice", "ellipsis", "ell_only", "bool2d", "fancy_rows",
               "fancy_pair", "fancy_slice", "oob_row", "int_slice"]


def _do_setitem(shape, key, vkey, mask, hard, dev, idxmap):
    d = rdata(shape, seed=3)
    m = None if mask is None else pattern(shape)
    x, n = make(d, m, dev, hard_mask=hard)
    idx = idxmap[key]
    try:
        tshape = np.empty(shape)[idx].shape
    except Exception:  # noqa: BLE001 - invalid index: values irrelevant
        tshape = ()
    xv, nv = VALUES[vkey](tshape, dev)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n[idx] = nv
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            x[dev_index(idx, dev)] = xv
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x[dev_index(idx, dev)] = xv
    assert_same(x, n, dev=dev, ctx=f"{key}<-{vkey}")


class TestSetitem:
    @pytest.mark.parametrize("hard", [False, True], ids=["soft", "hard"])
    def test_1d(self, hard, dev):
        for vkey in VALUES:
            for key in SET_KEYS_1D:
                if dev == "gpu" and key == "fancy_repeat" and vkey not in ("scalar", "int", "masked", "np0d", "nan"):
                    # Known cupy limitation: with repeated indices and array values
                    # the surviving write is order-undefined on cupy (numpy: last wins).
                    continue
                for mask in (None, "pat"):
                    with case(key, vkey, mask, hard):
                        _do_setitem((6,), key, vkey, mask, hard, dev, IDX_1D)

    @pytest.mark.parametrize("hard", [False, True], ids=["soft", "hard"])
    def test_2d(self, hard, dev):
        for vkey in VALUES:
            for key in SET_KEYS_2D:
                for mask in (None, "pat"):
                    with case(key, vkey, mask, hard):
                        _do_setitem((3, 4), key, vkey, mask, hard, dev, IDX_2D)

    def test_hard_mask_leaves_masked_slots_unchanged(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 1, 0], dev, hard_mask=True)
        x[:] = 100.0
        n[:] = 100.0
        assert_same(x, n, dev=dev)
        x[1] = 5.0
        n[1] = 5.0
        assert_same(x, n, dev=dev)
        assert bool(host(XMA.getmaskarray(x))[1])

    def test_soft_mask_unmasks_on_assignment(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 1, 0], dev)
        x[1] = 5.0
        n[1] = 5.0
        assert_same(x, n, dev=dev)
        assert not bool(host(XMA.getmaskarray(x))[1])
        x[3] = XMA.masked
        n[3] = np.ma.masked
        assert_same(x, n, dev=dev)

    def test_harden_then_soften(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 1, 0], dev)
        x.harden_mask()
        n.harden_mask()
        x[:] = 3.0
        n[:] = 3.0
        assert_same(x, n, dev=dev)
        x.soften_mask()
        n.soften_mask()
        x[:] = 4.0
        n[:] = 4.0
        assert_same(x, n, dev=dev)

    def test_nomask_mask_allocated_only_when_needed(self, dev):
        x, n = make(np.arange(5.0), None, dev)
        x[0] = 7.0
        n[0] = 7.0
        assert x.mask is NM and n.mask is NNM
        x[1:3] = np.array([1.0, 2.0]) if dev == "cpu" else cp.asarray([1.0, 2.0])
        n[1:3] = np.array([1.0, 2.0])
        assert x.mask is NM
        x[2] = MSK
        n[2] = NMSK
        assert x.mask is not NM and n.mask is not NNM
        assert_same(x, n, dev=dev)

    def test_nomask_assign_masked_array_with_nomask_stays_nomask(self, dev):
        x, n = make(np.arange(4.0), None, dev)
        vx, vn = make(np.array([9.0, 8.0]), None, dev)
        x[:2] = vx
        n[:2] = vn
        assert_same(x, n, dev=dev)

    def test_nomask_assign_masked_array_with_mask(self, dev):
        x, n = make(np.arange(4.0), None, dev)
        vx, vn = make(np.array([9.0, 8.0]), [1, 0], dev)
        x[1:3] = vx
        n[1:3] = vn
        assert_same(x, n, dev=dev)

    @pytest.mark.parametrize("hard", [False, True])
    def test_setitem_full_row_masked_in_2d(self, hard, dev):
        x, n = make(np.arange(12.0).reshape(3, 4), pattern((3, 4)), dev, hard_mask=hard)
        x[1] = XMA.masked
        n[1] = np.ma.masked
        assert_same(x, n, dev=dev)
        x[:, 2] = XMA.masked
        n[:, 2] = np.ma.masked
        assert_same(x, n, dev=dev)

    def test_setitem_returns_none_and_value_not_aliased(self, dev):
        x, n = make(np.arange(4.0), [0, 1, 0, 0], dev)
        v = to_dev(np.array([5.0, 6.0]), dev)
        x[0:2] = v
        v[0] = -1.0
        assert float(host(x.data)[0]) == 5.0

    def test_setitem_dtype_casting(self, dev):
        for dt, val in [("i4", 2.9), ("i4", 7), ("u1", 3), ("f4", 1e10), ("?", 5), ("i2", np.int64(3))]:
            x, n = make(np.arange(4).astype(dt), [0, 1, 0, 0], dev)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    n[0] = val
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    x[0] = val
                continue
            x[0] = val
            assert_same(x, n, dev=dev)

    def test_setitem_dtype_overflow(self, dev):
        x, n = make(np.arange(4).astype("u1"), None, dev)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n[0] = 300
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x[0] = 300
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x[0] = 300
            assert_same(x, n, dev=dev)

    def test_setitem_bad_value(self, dev):
        x, n = make(np.arange(4.0), None, dev)
        with pytest.raises(Exception) as en:
            n[0] = "abc"
        with pytest.raises(en.type):
            x[0] = "abc"

    def test_setitem_shape_mismatch(self, dev):
        x, n = make(np.arange(4.0), None, dev)
        v = np.arange(3.0)
        with pytest.raises(Exception) as en:
            n[0:2] = v
        with pytest.raises(en.type):
            x[0:2] = to_dev(v, dev)

    def test_setitem_bad_index_same_error(self, dev):
        for idx in (4, -5, (0, 0), np.array([0, 7]), "a", 1.5):
            if dev == "gpu" and isinstance(idx, np.ndarray):
                # Known cupy limitation: out-of-bounds integer-array indices do not
                # raise on cupy (would need a host sync); still checked on cpu.
                continue
            with case(idx):
                x, n = make(np.arange(4.0), [0, 1, 0, 0], dev)
                with pytest.raises(Exception) as en:
                    n[idx] = 1.0
                with pytest.raises(en.type):
                    x[dev_index(idx, dev)] = 1.0

    def test_setitem_masked_on_int_array(self, dev):
        x, n = make(np.arange(4), None, dev)
        x[2] = XMA.masked
        n[2] = np.ma.masked
        assert_same(x, n, dev=dev)

    def test_setitem_on_empty(self, dev):
        x, n = make(np.zeros((0,)), None, dev)
        x[:] = 1.0
        n[:] = 1.0
        assert_same(x, n, dev=dev)
        with pytest.raises(IndexError):
            n[0] = 1.0
        with pytest.raises(IndexError):
            x[0] = 1.0

    def test_setitem_0d(self, dev):
        x, n = make(np.array(2.0), None, dev)
        x[()] = 5.0
        n[()] = 5.0
        assert_same(x, n, dev=dev)
        x[...] = XMA.masked
        n[...] = np.ma.masked
        assert_same(x, n, dev=dev)

    def test_setitem_slice_with_scalar_mask_value(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 0, 0], dev)
        v = mka(dev, to_dev(np.array(3.0), dev), mask=True)
        vn = np.ma.masked_array(np.array(3.0), mask=True)
        x[1:4] = v
        n[1:4] = vn
        assert_same(x, n, dev=dev)


class TestSharedMask:
    def test_view_shares_mask_until_setitem(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 0, 0], dev)
        vx, vn = x.view(), n.view()
        assert bool(vx._sharedmask) == bool(vn._sharedmask)
        vx[2] = XMA.masked
        vn[2] = np.ma.masked
        assert_same(x, n, dev=dev)
        assert_same(vx, vn, dev=dev)

    def test_slice_setitem_masked_affects_base_like_numpy(self, dev):
        x, n = make(np.arange(6.0), [0, 1, 0, 0, 0, 0], dev)
        vx, vn = x[1:5], n[1:5]
        vx[0] = 1.0
        vn[0] = 1.0
        vx[1] = XMA.masked
        vn[1] = np.ma.masked
        assert_same(x, n, dev=dev)
        assert_same(vx, vn, dev=dev)

    def test_unshare_mask(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 0, 0], dev)
        vx, vn = x.view(), n.view()
        rx, rn = vx.unshare_mask(), vn.unshare_mask()
        assert rx is vx and rn is vn
        vx[0] = XMA.masked
        vn[0] = np.ma.masked
        assert_same(x, n, dev=dev)
        assert_same(vx, vn, dev=dev)
        assert bool(vx.sharedmask) == bool(vn.sharedmask)

    def test_sharedmask_flag_of_new_arrays(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 0, 0], dev)
        assert bool(x.sharedmask) == bool(n.sharedmask)
        assert bool(x[1:].sharedmask) == bool(n[1:].sharedmask)
        assert bool(x.copy().sharedmask) == bool(n.copy().sharedmask)
        assert bool((x + 1).sharedmask) == bool((n + 1).sharedmask)

    def test_two_views_modify_independent_after_unshare(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 0, 0], dev)
        v1x, v2x = x.view(), x.view()
        v1n, v2n = n.view(), n.view()
        v1x.unshare_mask()
        v1n.unshare_mask()
        v1x[3] = XMA.masked
        v1n[3] = np.ma.masked
        assert_same(v2x, v2n, dev=dev)
        assert_same(x, n, dev=dev)

    def test_mask_setter_on_view_does_not_touch_base(self, dev):
        x, n = make(np.arange(5.0), [0, 1, 0, 0, 0], dev)
        vx, vn = x.view(), n.view()
        vx.mask = np.ones(5, bool) if dev == "cpu" else cp.ones(5, bool)
        vn.mask = np.ones(5, bool)
        assert_same(x, n, dev=dev)
        assert_same(vx, vn, dev=dev)


# ==========================================================================
# 5. metadata, shape methods and Python protocols
# ==========================================================================
SHAPES = [(6,), (2, 3), (1, 6), (2, 1, 3), (0,), ()]


def _mk(shape, dev, masked=True, **kw):
    d = rdata(shape) if shape else np.array(2.5)
    m = None
    if masked and d.size:
        m = pattern(shape) if shape else np.array(False)
    return make(d, m, dev, **kw)


class TestShapeMethods:
    def test_simple(self, dev):
        fs = {
            "T": lambda M, a: a.T, "mT": lambda M, a: a.mT, "ravel": lambda M, a: a.ravel(),
            "flatten": lambda M, a: a.flatten(), "compressed": lambda M, a: a.compressed(),
            "tolist": lambda M, a: a.tolist(), "squeeze": lambda M, a: a.squeeze(),
            "copy": lambda M, a: a.copy(), "filled": lambda M, a: a.filled(),
            "real": lambda M, a: a.real, "imag": lambda M, a: a.imag,
            "transpose": lambda M, a: a.transpose(), "view": lambda M, a: a.view(),
            "nonzero": lambda M, a: a.nonzero(),
        }
        for name, f in fs.items():
            for shape in SHAPES:
                for masked in (True, False):
                    with case(name, shape, masked):
                        x, n = _mk(shape, dev, masked)
                        run_pair(f, x, n, dev)

    @pytest.mark.parametrize("name", ["ravel", "flatten"])
    def test_order(self, name, dev):
        x, n = _mk((2, 3), dev)
        xt, nt = x.T, n.T
        for order in ("C", "F", "A", "K"):
            with case(order):
                run_pair(lambda M, a: getattr(a, name)(order), x, n, dev)
                run_pair(lambda M, a: getattr(a, name)(order), xt, nt, dev)

    def test_reshape(self, dev):
        x, n = _mk((2, 3), dev)
        for args in [(3, 2), ((3, 2),), (-1,), (1, -1), (6,), (7,), (2, 2), (-1, -1), ()]:
            with case(args):
                run_pair(lambda M, a: a.reshape(*args), x, n, dev)

    def test_reshape_order(self, dev):
        x, n = _mk((2, 3), dev)
        run_pair(lambda M, a: a.reshape(3, 2, order="F"), x, n, dev)

    def test_reshape_shares_data(self, dev):
        x, n = _mk((2, 3), dev)
        rx, rn = x.reshape(3, 2), n.reshape(3, 2)
        rx[0, 0] = -3.0
        rn[0, 0] = -3.0
        assert_same(x, n, dev=dev)

    def test_ravel_view_shares_data(self, dev):
        x, n = _mk((2, 3), dev)
        rx, rn = x.ravel(), n.ravel()
        rx[0] = -3.0
        rn[0] = -3.0
        assert_same(x, n, dev=dev)

    def test_flatten_is_copy(self, dev):
        x, n = _mk((2, 3), dev)
        rx, rn = x.flatten(), n.flatten()
        rx[0] = -3.0
        rn[0] = -3.0
        assert_same(x, n, dev=dev)

    def test_T_is_view(self, dev):
        x, n = _mk((2, 3), dev)
        x.T[0, 1] = -3.0
        n.T[0, 1] = -3.0
        assert_same(x, n, dev=dev)

    def test_mT_3d(self, dev):
        x, n = _mk((2, 3, 4), dev)
        run_pair(lambda M, a: a.mT, x, n, dev)
        run_pair(lambda M, a: a.mT.shape, x, n, dev)

    def test_mT_1d_raises_like_numpy(self, dev):
        x, n = _mk((6,), dev)
        with pytest.raises(Exception) as en:
            n.mT
        with pytest.raises(en.type):
            x.mT

    def test_transpose_axes(self, dev):
        x, n = _mk((2, 3, 4), dev)
        for axes in [None, (1, 0), (0, 1), (2, 0, 1), (0, 0), (5,)]:
            with case(axes):
                run_pair(lambda M, a: a.transpose(axes) if axes is not None else a.transpose(), x, n, dev)

    @pytest.mark.parametrize("masked", [True, False])
    def test_take(self, masked, dev):
        x, n = _mk((2, 3), dev, masked)
        for args, kw in [
            ([0], {}), ([[0, 2, 5]], {}), ([[0, 5]], {"axis": None}), ([[1, 0]], {"axis": 1}),
            ([2], {}), ([[-1]], {}), ([[0, 9]], {}), ([[0, 9]], {"mode": "clip"}), ([[0, 9]], {"mode": "wrap"}),
            ([[0, 9]], {"mode": "raise"}), ([[[0, 1], [2, 3]]], {}), ([[]], {}), ([[0]], {"axis": 5}),
            ([[1]], {"axis": -1}),
        ]:
            def f(M, a):
                return a.take(*[dev_index(np.asarray(v, dtype=np.intp), dev)
                                if (a is x and isinstance(v, list)) else v for v in args], **kw)
            with case(args, kw):
                run_pair(f, x, n, dev)

    def test_take_does_not_share(self, dev):
        x, n = _mk((6,), dev)
        tx, tn = x.take([0, 1]), n.take([0, 1])
        tx[0] = -9.0
        tn[0] = -9.0
        assert_same(x, n, dev=dev)

    @pytest.mark.parametrize("masked", ["pat", "all", "none", "nomask"])
    def test_compressed(self, masked, dev):
        for shape in [(6,), (2, 3)]:
            with case(shape):
                m = {"pat": pattern(shape), "all": np.ones(shape, bool), "none": np.zeros(shape, bool),
                     "nomask": None}[masked]
                x, n = make(rdata(shape), m, dev)
                run_pair(lambda M, a: a.compressed(), x, n, dev)

    def test_tolist(self, dev):
        for shape in SHAPES:
            with case(shape):
                x, n = _mk(shape, dev)
                rx, rn = x.tolist(), n.tolist()
                assert rx == rn
                assert type(rx) is type(rn)
                if shape == (6,):
                    assert rx.count(None) == rn.count(None) > 0

    def test_tolist_dtypes(self, dev):
        for dt in ("i4", "?", "f4", "c16", "u2"):
            x, n = make(np.arange(4).astype(dt), [0, 1, 0, 0], dev)
            rx, rn = x.tolist(), n.tolist()
            assert rx == rn and [type(v) for v in rx] == [type(v) for v in rn]

    @pytest.mark.parametrize("masked", [True, False])
    def test_fill_sets_all_elements(self, masked, dev):
        for v in (5.0, 0, -1.5, np.float32(2), True, 1e300, np.nan, None):
            with case(v):
                x, n = _mk((2, 3), dev, masked)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        n.fill(v)
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        x.fill(v)
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x.fill(v)
                assert_same(x, n, dev=dev)
                if v is not None:
                    np.testing.assert_array_equal(host(x.data), n.data)

    def test_fill_int_array_with_float(self, dev):
        x, n = make(np.arange(4), [0, 1, 0, 0], dev)
        x.fill(2.7)
        n.fill(2.7)
        assert_same(x, n, dev=dev)

    @pytest.mark.parametrize("masked", [True, False])
    def test_filled(self, masked, dev):
        x, n = _mk((2, 3), dev, masked)
        for fv in (None, 0.0, -1.0, 99, np.nan, "x"):
            with case(fv):
                run_pair(lambda M, a: a.filled(fv), x, n, dev)

    def test_filled_returns_independent_array_when_masked(self, dev):
        x, n = _mk((6,), dev, True)
        fx, fn = x.filled(0.0), n.filled(0.0)
        fx[0] = 12345.0
        fn[0] = 12345.0
        assert_same(x, n, dev=dev)

    def test_filled_nomask_aliases_like_numpy(self, dev):
        x, n = _mk((6,), dev, False)
        fx, fn = x.filled(), n.filled()
        fx[0] = 12345.0
        fn[0] = 12345.0
        assert_same(x, n, dev=dev)

    @pytest.mark.parametrize("masked", [True, False])
    def test_item(self, masked, dev):
        x, n = _mk((2, 3), dev, masked)
        for args in [(), (0,), (4,), (-1,), (6,), ((1, 2),), (1, 2), ((0, 1),)]:
            with case(args):
                run_pair(lambda M, a: a.item(*args), x, n, dev)

    def test_item_size_one(self, dev):
        x, n = make(np.array([[3.5]]), None, dev)
        assert_same(x.item(), n.item())
        xm, nm_ = make(np.array([[3.5]]), [[True]], dev)
        run_pair(lambda M, a: a.item(), xm, nm_, dev)

    def test_real_imag_complex(self, dev):
        for dt in ("c16", "c8"):
            d = (rdata((4,)) + 1j * rdata((4,), seed=2)).astype(dt)
            x, n = make(d, [0, 1, 0, 0], dev)
            for name in ("real", "imag"):
                with case(dt, name):
                    run_pair(lambda M, a, name=name: getattr(a, name), x, n, dev)

    def test_real_imag_assign(self, dev):
        d = (rdata((4,)) + 1j * rdata((4,), seed=2))
        x, n = make(d, [0, 1, 0, 0], dev)
        x.real[0] = 9.0
        n.real[0] = 9.0
        x.imag[1] = 3.0
        n.imag[1] = 3.0
        assert_same(x, n, dev=dev)

    def test_real_imag_of_real_array(self, dev):
        x, n = make(rdata((4,)), [0, 1, 0, 0], dev)
        run_pair(lambda M, a: a.imag, x, n, dev)
        run_pair(lambda M, a: a.real, x, n, dev)

    def test_harden_soften_return_self_and_flag(self, dev):
        x, n = _mk((6,), dev)
        assert x.harden_mask() is x and n.harden_mask() is n
        assert bool(x.hardmask) is bool(n.hardmask) is True
        assert x.soften_mask() is x and n.soften_mask() is n
        assert bool(x.hardmask) is bool(n.hardmask) is False

    def test_hardmask_property_assignment(self, dev):
        x, n = _mk((6,), dev)
        try:
            n.hardmask = True
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x.hardmask = True
        else:
            x.hardmask = True
            assert bool(x.hardmask)

    def test_shrink_mask(self, dev):
        shape = (2, 3)
        for mask in (None, "none", "pat", "all"):
            with case(mask):
                m = {None: None, "none": np.zeros(shape, bool), "pat": pattern(shape),
                     "all": np.ones(shape, bool)}[mask]
                x, n = make(rdata(shape), m, dev, shrink=False)
                rx, rn = x.shrink_mask(), n.shrink_mask()
                assert rx is x and rn is n
                assert_same(x, n, dev=dev)

    def test_hardmask_on_views_and_results(self, dev):
        x, n = _mk((6,), dev, hard_mask=True)
        for f in (lambda a: a.view(), lambda a: a[1:], lambda a: a + 1, lambda a: a.copy(), lambda a: a.reshape(2, 3),
                  lambda a: a.T, lambda a: a.ravel(), lambda a: a.astype("f4")):
            assert bool(f(x).hardmask) == bool(f(n).hardmask)


class TestFlat:
    def test_iter(self, dev):
        for shape in [(6,), (2, 3), (2, 1, 3), ()]:
            with case(shape):
                x, n = _mk(shape, dev)
                lx, ln = list(x.flat), list(n.flat)
                assert len(lx) == len(ln)
                for a, b in zip(lx, ln):
                    assert_same(a, b)

    def test_iter_yields_numpy_scalars_or_masked(self, dev):
        x, n = _mk((6,), dev)
        for a, b in zip(x.flat, n.flat):
            if b is NMSK:
                assert a is MSK
            else:
                assert isinstance(a, np.generic) and type(a) is type(b)

    def test_getitem_int(self, dev):
        x, n = _mk((2, 3), dev)
        for i in (0, 1, 4, -1, -6, 6, 100):
            with case(i):
                run_pair(lambda M, a: a.flat[i], x, n, dev)

    def test_getitem_other(self, dev):
        x, n = _mk((2, 3), dev)
        for key in [slice(None), slice(1, 4), slice(None, None, 2), slice(None, None, -1),
                    [0, 2, 5], np.array([1, 3]), ..., np.array([1, 0, 1, 0, 0, 1], bool),
                    (0,), slice(5, 2)]:
            with case(key):
                run_pair(lambda M, a: a.flat[dev_index(key, dev) if a is x and isinstance(key, np.ndarray) else key],
                         x, n, dev)

    @pytest.mark.parametrize("hard", [False, True])
    def test_setitem(self, hard, dev):
        for v in (9.0, MSK, [1.0, 2.0]):
            for i in (0, 4, -1, slice(1, 3), slice(None)):
                with case(v, i):
                    x, n = _mk((2, 3), dev, hard_mask=hard)
                    vn = NMSK if v is MSK else v
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            n.flat[i] = vn
                    except Exception as e:  # noqa: BLE001
                        with pytest.raises(type(e)):
                            x.flat[i] = v
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        x.flat[i] = v
                    assert_same(x, n, dev=dev)

    def test_flat_assign_whole(self, dev):
        x, n = _mk((2, 3), dev)
        x.flat = 5.0
        n.flat = 5.0
        assert_same(x, n, dev=dev)

    def test_flat_attrs(self, dev):
        x, n = _mk((2, 3), dev)
        assert len(list(x.flat)) == len(list(n.flat)) == 6
        it_x, it_n = iter(x.flat), iter(n.flat)
        next(it_x)
        next(it_n)
        assert_same(next(it_x), next(it_n))

    def test_flat_is_not_a_copy(self, dev):
        x, n = _mk((2, 3), dev)
        x.flat[1] = 42.0
        n.flat[1] = 42.0
        assert_same(x, n, dev=dev)

    def test_flat_copy(self, dev):
        x, n = _mk((2, 3), dev)
        run_pair(lambda M, a: a.flat.copy(), x, n, dev)


class TestPythonProtocols:
    def test_len(self, dev):
        for shape in [(6,), (2, 3), (0,), (1,), (3, 0)]:
            with case(shape):
                x, n = _mk(shape, dev)
                assert len(x) == len(n)

    def test_len_0d_raises(self, dev):
        x, n = make(np.array(2.0), None, dev)
        with pytest.raises(TypeError):
            len(n)
        with pytest.raises(TypeError):
            len(x)

    def test_iter(self, dev):
        for shape in [(6,), (2, 3), (1, 3), (3, 1), (0,), (1,)]:
            with case(shape):
                x, n = _mk(shape, dev)
                lx, ln = list(x), list(n)
                assert len(lx) == len(ln)
                for a, b in zip(lx, ln):
                    assert_same(a, b, dev=dev)

    def test_iter_0d_raises(self, dev):
        x, n = make(np.array(2.0), None, dev)
        with pytest.raises(TypeError):
            iter(n)
        with pytest.raises(TypeError):
            iter(x)

    def test_iter_rows_are_views(self, dev):
        x, n = _mk((2, 3), dev)
        r0x, r0n = next(iter(x)), next(iter(n))
        r0x[0] = -4.0
        r0n[0] = -4.0
        assert_same(x, n, dev=dev)

    def test_iter_1d_masked_is_singleton(self, dev):
        x, n = make(np.arange(3.0), [0, 1, 0], dev)
        lx = list(x)
        assert lx[1] is MSK
        assert all(isinstance(lx[i], np.generic) for i in (0, 2))

    @pytest.mark.parametrize("mask", [None, [0, 1, 0, 0, 0, 0]])
    def test_contains(self, mask, dev):
        x, n = make(np.arange(6.0), mask, dev)
        for needle in (0.0, 1.0, 2.0, 5.0, 3, "a", None, True, [1.0], np.float64(2.0)):
            with case(needle):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rn = needle in n
                except Exception as e:  # noqa: BLE001
                    with pytest.raises(type(e)):
                        needle in x
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    assert (needle in x) == rn

    def test_contains_masked(self, dev):
        x, n = make(np.arange(4.0), [0, 1, 0, 0], dev)
        assert (MSK in x) == (NMSK in n)
        x2, n2 = make(np.arange(4.0), None, dev)
        assert (MSK in x2) == (NMSK in n2)

    CASES = {
        "size1": (np.array([3.0]), None),
        "size1_2d": (np.array([[3.0]]), None),
        "size1_zero": (np.array([0.0]), None),
        "size1_masked": (np.array([3.0]), [True]),
        "size1_unmasked_mask": (np.array([3.0]), [False]),
        "0d": (np.array(3.0), None),
        "0d_masked": (np.array(3.0), np.array(True)),
        "size2": (np.array([1.0, 2.0]), None),
        "size2_masked": (np.array([1.0, 2.0]), [True, True]),
        "empty": (np.zeros((0,)), None),
        "int1": (np.array([7]), None),
        "bool1": (np.array([True]), None),
        "complex1": (np.array([1 + 2j]), None),
        "nan1": (np.array([np.nan]), None),
    }

    def test_scalar_conversions(self, dev):
        for conv in (bool, float, int, complex, "index", "round", "abs_int"):
            for name, (d, m) in self.CASES.items():
                with case(conv, name):
                    self._conv_one(d, m, conv, dev)

    @staticmethod
    def _conv_one(d, m, conv, dev):
        x, n = make(d, m, dev)

        def call(a):
            if conv == "index":
                return a.__index__()
            if conv == "round":
                return round(a)
            if conv == "abs_int":
                return int(abs(a))
            return conv(a)

        with warnings.catch_warnings(record=True) as wn:
            warnings.simplefilter("always")
            try:
                rn = call(n)
                err = None
            except Exception as e:  # noqa: BLE001
                rn, err = None, e
        if err is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(type(err)):
                    call(x)
            return
        wcats = sorted({w.category.__name__ for w in wn})
        with warnings.catch_warnings(record=True) as wx:
            warnings.simplefilter("always")
            rx = call(x)
        assert sorted({w.category.__name__ for w in wx}) == wcats
        assert type(rx) is type(rn)
        if isinstance(rn, float) and np.isnan(rn):
            assert np.isnan(rx)
        else:
            assert rx == rn

    def test_bool_ambiguous_raises(self, dev):
        x, n = make(np.array([1.0, 2.0]), None, dev)
        with pytest.raises(ValueError):
            bool(n)
        with pytest.raises(ValueError):
            bool(x)

    def test_comparison_in_if_like_numpy(self, dev):
        x, n = make(np.array([1.0]), None, dev)
        assert bool(x == 1.0) == bool(n == 1.0)

    def test_whitelisted_attrs(self, dev):
        for shape in [(6,), (2, 3), (2, 1, 3)]:
            for dt in ("f8", "f4", "i2", "c16"):
                x, n = make(rdata(shape).astype(dt), pattern(shape), dev)
                for name in ("nbytes", "itemsize", "strides", "size", "ndim", "shape"):
                    with case(shape, dt, name):
                        assert getattr(x, name) == getattr(n, name)

    def test_strides_of_transposed(self, dev):
        x, n = _mk((2, 3), dev)
        assert x.T.strides == n.T.strides

    def test_flags_and_base_device(self, dev):
        x, n = _mk((2, 3), dev)
        assert hasattr(x, "flags")
        assert x.flags.c_contiguous == n.flags.c_contiguous
        assert x.T.flags.f_contiguous == n.T.flags.f_contiguous
        assert hasattr(x, "base")
        assert hasattr(x, "device")
        assert x.device is not None
        if dev == "cpu":
            assert x.device == n.device

    def test_base_of_view_and_owner(self, dev):
        x, n = _mk((6,), dev)
        v = x[1:4]
        assert v.base is not None

    def test_removed_and_bogus_attrs(self, dev):
        x, n = _mk((6,), dev)
        for name in ["itemset", "newbyteorder", "tostring", "__cuda_array_interface__",
                     "bogus", "get", "cuda", "_data_typo"]:
            with case(name):
                assert not hasattr(n, name)
                assert not hasattr(x, name)
                with pytest.raises(AttributeError) as ex:
                    getattr(x, name)
                assert name in str(ex.value)

    def test_attribute_error_message_names_class(self, dev):
        x, n = _mk((6,), dev)
        with pytest.raises(AttributeError, match="bogus"):
            x.bogus

    def test_no_forwarding_of_raw_array_methods(self, dev):
        x, n = _mk((6,), dev)
        for name in ("tobytes", "dump", "dumps", "tofile", "resize", "searchsorted", "partition",
                     "put", "compress", "diagonal"):
            assert hasattr(x, name) == hasattr(n, name), name
        # structured/record/ctypes ndarray internals: explicit NotImplementedError
        for name in ("setflags", "setfield", "getfield", "toflex", "torecords"):
            assert hasattr(n, name)
            with pytest.raises(NotImplementedError):
                getattr(x, name)(*([0] if name == "setfield" else []))
        for name in ("byteswap", "choose"):
            assert hasattr(x, name) and hasattr(n, name)

    def test_setting_new_attribute_like_numpy(self, dev):
        x, n = _mk((6,), dev)
        n.custom = 1
        x.custom = 1
        assert x.custom == n.custom == 1

    def test_data_property(self, dev):
        x, n = _mk((2, 3), dev)
        assert on_dev(x.data, dev)
        assert not isinstance(x.data, XMA.MaskedArray)
        np.testing.assert_array_equal(host(x.data), n.data)
        x.data[0, 0] = 123.0
        n.data[0, 0] = 123.0
        assert_same(x, n, dev=dev)

    def test_data_not_settable_like_numpy(self, dev):
        x, n = _mk((6,), dev)
        with pytest.raises(AttributeError):
            n.data = np.zeros(6)
        with pytest.raises(AttributeError):
            x.data = np.zeros(6)

    def test_np_asarray_returns_host_data(self, dev):
        for shape in [(6,), (2, 3), (), (0,)]:
            for masked in (True, False):
                with case(shape, masked):
                    x, n = _mk(shape, dev, masked)
                    a, b = np.asarray(x), np.asarray(n)
                    assert type(a) is np.ndarray
                    assert a.dtype == b.dtype and a.shape == b.shape
                    np.testing.assert_array_equal(a, b)

    def test_asarray_returns_unfilled_data(self, dev):
        x, n = make(np.arange(4.0), [0, 1, 0, 0], dev, fill_value=-1.0)
        np.testing.assert_array_equal(np.asarray(x), np.asarray(n))
        assert np.asarray(x)[1] == 1.0

    @pytest.mark.parametrize("copy_", [None, True, False])
    def test_dunder_array(self, copy_, dev):
        for dtype in (None, "f4", "i8", "c16", "?"):
            with case(dtype):
                self._dunder_array_one(dtype, copy_, dev)

    @staticmethod
    def _dunder_array_one(dtype, copy_, dev):
        x, n = _mk((2, 3), dev)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rn = n.__array__(dtype, copy_) if dtype is not None or copy_ is not None else n.__array__()
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x.__array__(dtype, copy_) if (dtype is not None or copy_ is not None) else x.__array__()
            return
        if dev == "gpu" and copy_ is False:
            with pytest.raises(ValueError):
                x.__array__(dtype, copy_)
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rx = x.__array__(dtype, copy_) if (dtype is not None or copy_ is not None) else x.__array__()
        assert isinstance(rx, np.ndarray) and not isinstance(rx, XMA.MaskedArray)
        assert rx.dtype == rn.dtype
        np.testing.assert_array_equal(rx, rn)

    @needs_gpu
    @pytest.mark.parametrize("kw", [dict(copy=False), dict(dtype="f4", copy=False)])
    def test_gpu_copy_false_raises(self, kw):
        x, _ = _mk((2, 3), "gpu")
        with pytest.raises(ValueError):
            x.__array__(**kw)
        with pytest.raises(ValueError):
            np.asarray(x, **kw)

    def test_asarray_copy_true_is_copy(self, dev):
        x, _ = _mk((2, 3), dev)
        a = np.asarray(x, copy=True)
        a[0, 0] = 1234.0
        assert host(x.data)[0, 0] != 1234.0

    def test_cpu_asarray_shares_memory(self):
        x, n = _mk((2, 3), "cpu")
        assert np.shares_memory(np.asarray(x), x.data) == np.shares_memory(np.asarray(n), n.data)

    def test_np_array_of_xupy(self, dev):
        x, n = _mk((2, 3), dev)
        np.testing.assert_array_equal(np.array(x), np.array(n))
        np.testing.assert_array_equal(np.array(x, dtype="f4"), np.array(n, dtype="f4"))

    def test_asmarray(self, dev):
        x, n = make(rdata((2, 3)), pattern((2, 3)), dev, fill_value=3.0, hard_mask=True)
        r = x.asmarray()
        assert type(r) is np.ma.MaskedArray
        assert r.dtype == n.dtype and r.shape == n.shape
        np.testing.assert_array_equal(np.ma.getmaskarray(r), np.ma.getmaskarray(n))
        np.testing.assert_array_equal(r.filled(0), n.filled(0))
        assert r.fill_value == n.fill_value
        assert bool(r.hardmask) is True

    def test_asmarray_nomask(self, dev):
        x, n = make(rdata((2, 3)), None, dev)
        r = x.asmarray()
        assert type(r) is np.ma.MaskedArray
        np.testing.assert_array_equal(r.data, n.data)
        assert r.mask is np.ma.nomask

    def test_asmarray_is_a_copy(self, dev):
        x, n = make(rdata((4,)), [0, 1, 0, 0], dev)
        r = x.asmarray()
        r[0] = 555.0
        assert host(x.data)[0] != 555.0

    def test_xupy_asnumpy(self, dev):
        x, n = make(rdata((4,)), [0, 1, 0, 0], dev)
        r = xupy.asnumpy(x)
        assert isinstance(r, (np.ma.MaskedArray, np.ndarray))
        if isinstance(r, np.ma.MaskedArray):
            np.testing.assert_array_equal(np.ma.getmaskarray(r), np.ma.getmaskarray(n))
        np.testing.assert_array_equal(np.ma.getdata(r), n.data)

    def test_asnumpy_nomask_array(self, dev):
        x, n = make(rdata((4,)), None, dev)
        r = xupy.asnumpy(x)
        assert isinstance(r, np.ndarray)
        np.testing.assert_array_equal(np.ma.getdata(r), n.data)

    def test_isinstance_relationships(self, dev):
        x, n = _mk((3,), dev)
        assert isinstance(x, XMA.MaskedArray)
        assert XMA.masked_array is XMA.MaskedArray
        assert not isinstance(n, XMA.MaskedArray) or XMA is np.ma

    def test_hash_unhashable_like_numpy(self, dev):
        x, n = _mk((3,), dev)
        with pytest.raises(TypeError):
            hash(n)
        with pytest.raises(TypeError):
            hash(x)

    def test_array_priority(self):
        assert XMA.MaskedArray.__array_priority__ == 15

    def test_repr_does_not_raise_on_nomask(self, dev):
        x, _ = _mk((3,), dev, False)
        assert isinstance(repr(x), str) and isinstance(str(x), str)

    def test_np_ma_functions_on_xupy_dispatch(self, dev):
        x, n = _mk((2, 3), dev)
        assert XMA.isMaskedArray(x)
        assert_same(XMA.getmaskarray(x), np.ma.getmaskarray(n), dev=dev)

    def test_bool_of_masked_array_nomask_1elem(self, dev):
        x, n = make(np.array([0.0]), None, dev)
        assert bool(x) is bool(n) is False

    def test_size_1_float_int_cast_of_masked_element_warns_only_float(self, dev):
        x, n = make(np.array([5.0]), [True], dev)
        with pytest.warns(UserWarning):
            float(n)
        with pytest.warns(UserWarning):
            float(x)
        with pytest.raises(np.ma.MaskError):
            int(n)
        with pytest.raises(np.ma.MaskError):
            int(x)


# ==========================================================================
# 6. copy / deepcopy / pickle
# ==========================================================================
RT_CASES = {
    "f8_masked": (rdata((4,)), [0, 1, 0, 0], {}),
    "f8_nomask": (rdata((4,)), None, {}),
    "f4_2d": (rdata((2, 3), "f4"), pattern((2, 3)), {}),
    "i4": (np.arange(5, dtype="i4"), [1, 0, 0, 0, 1], {}),
    "bool": (np.array([True, False, True]), [0, 1, 0], {}),
    "c16": (rdata((3,)) + 1j * rdata((3,), seed=2), [0, 0, 1], {}),
    "0d_masked": (np.array(2.5), np.array(True), {}),
    "empty": (np.zeros((0,)), None, {}),
    "fill_hard_int": (np.arange(4), [0, 1, 0, 0], {"fill_value": -3, "hard_mask": True}),
    "all_false_mask": (rdata((4,)), np.zeros(4, bool), {"shrink": False}),
}


@pytest.fixture(params=list(RT_CASES))
def rt(request, dev):
    d, m, kw = RT_CASES[request.param]
    x, n = make(d, m, dev, **kw)
    return x, n, dev


class TestCopyPickle:
    def test_copy_pickle_independence(self, rt):
        self._copies(*rt)
        self._pickle_roundtrip(*rt)
        self._independence(*rt)

    @staticmethod
    def _copies(x, n, dev):
        assert_same(copy.copy(x), n, dev=dev)
        y = copy.deepcopy(x)
        assert y is not x
        assert_same(y, n, dev=dev)
        assert_same(x.copy(), n.copy(), dev=dev)

    @staticmethod
    def _pickle_roundtrip(x, n, dev):
        for proto in [None] + list(range(pickle.HIGHEST_PROTOCOL + 1)):
            with case(proto):
                payload = pickle.dumps(x) if proto is None else pickle.dumps(x, protocol=proto)
                y = pickle.loads(payload)
                assert y is not x
                # numpy.ma pickling drops nomask-ness and hard_mask; XuPy must keep them
                assert_same(y, n, dev=dev, strict_nomask=False)  # see NO-SYNC note
                assert (y.mask is NM) == (x.mask is NM)
        if x.size and x.ndim:
            y = pickle.loads(pickle.dumps(x))
            y[0] = 1.0 if y.dtype.kind != "b" else True
            n2 = n.copy()
            n2[0] = 1.0 if n.dtype.kind != "b" else True
            assert_same(y, n2, dev=dev, strict_nomask=False)  # see NO-SYNC note

    @staticmethod
    def _independence(x, n, dev):
        if x.size == 0:
            return
        for cf in (copy.copy, copy.deepcopy):
            y = cf(x)
            y.data[...] = 0
            np.testing.assert_array_equal(host(x.data), np.asarray(n.data))
        if x.ndim == 0:
            return
        for cf in (copy.copy, copy.deepcopy, lambda a: pickle.loads(pickle.dumps(a))):
            y, m = cf(x), cf(n)
            if n.hardmask and not m.hardmask:
                m.harden_mask()  # numpy.ma pickling drops hard_mask; XuPy keeps it (see above)
            y[0] = 5 if y.dtype.kind in "iub" else 5.0
            m[0] = 5 if m.dtype.kind in "iub" else 5.0
            y[-1] = XMA.masked
            m[-1] = np.ma.masked
            assert_same(y, m, dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
            assert_same(x, n, dev=dev)

    def test_deepcopy_mask_is_independent(self, dev):
        x, n = make(rdata((4,)), [0, 1, 0, 0], dev)
        y, m = copy.deepcopy(x), copy.deepcopy(n)
        y.mask[0] = True
        m.mask[0] = True
        assert_same(x, n, dev=dev)
        assert_same(y, m, dev=dev)

    def test_shallow_copy_mask_like_numpy(self, dev):
        x, n = make(rdata((4,)), [0, 1, 0, 0], dev)
        y, m = copy.copy(x), copy.copy(n)
        y[0] = XMA.masked
        m[0] = np.ma.masked
        assert_same(x, n, dev=dev)
        assert_same(y, m, dev=dev)

    def test_pickle_memoizes_shared_objects(self, dev):
        x, n = make(rdata((4,)), [0, 1, 0, 0], dev)
        a, b = pickle.loads(pickle.dumps([x, x]))
        assert a is b
        assert_same(a, n, dev=dev)

    def test_pickle_container_with_singletons(self, dev):
        x, n = make(rdata((4,)), [0, 1, 0, 0], dev)
        out = pickle.loads(pickle.dumps({"a": x, "nm": NM, "m": MSK, "v": [NM, MSK]}))
        assert out["nm"] is NM and out["m"] is MSK
        assert out["v"][0] is NM and out["v"][1] is MSK
        assert_same(out["a"], n, dev=dev)

    def test_deepcopy_container_with_singletons(self):
        out = copy.deepcopy([NM, MSK, {"k": MSK}])
        assert out[0] is NM and out[1] is MSK and out[2]["k"] is MSK

    @needs_gpu
    def test_gpu_array_roundtrip_returns_on_gpu(self):
        x, n = make(rdata((2, 3)), pattern((2, 3)), "gpu", fill_value=2.5, hard_mask=True)
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with case(proto):
                y = pickle.loads(pickle.dumps(x, protocol=proto))
                assert isinstance(y.data, cp.ndarray) and isinstance(y.mask, cp.ndarray)
                assert_same(y, n, dev="gpu")

    @needs_gpu
    def test_gpu_pickle_is_host_payload(self):
        x, _ = make(rdata((4,)), [0, 1, 0, 0], "gpu")
        payload = pickle.dumps(x)
        with xupy.backend("cpu"):
            y = pickle.loads(payload)
        assert y.shape == (4,)

    def test_cpu_pickle_stays_cpu_even_under_gpu_backend(self):
        x, n = make(rdata((4,)), [0, 1, 0, 0], "cpu")
        payload = pickle.dumps(x)
        y = pickle.loads(payload)
        assert isinstance(y.data, np.ndarray)
        assert_same(y, n, dev="cpu")

    def test_pickle_numpy_ma_array_loads_into_xupy_equivalent(self, dev):
        n = np.ma.masked_array(np.arange(4.0), mask=[0, 1, 0, 0])
        y = mka(dev, pickle.loads(pickle.dumps(n)))
        assert_same(y, n, dev=dev)

    def test_getstate_setstate_roundtrip(self, dev):
        x, n = make(rdata((4,)), [0, 1, 0, 0], dev, fill_value=3.0, hard_mask=True)
        if hasattr(n, "__reduce__"):
            y = pickle.loads(pickle.dumps(x))
            assert bool(y.hardmask) is True
            assert y.fill_value == n.fill_value

    def test_reduce_ex_callable(self, dev):
        x, _ = make(rdata((4,)), [0, 1, 0, 0], dev)
        r = x.__reduce__()
        assert isinstance(r, tuple) and callable(r[0])


# --------------------------------------------------------------------------
# singletons: numpy protocols / numeric surface (review W1, W2)
# --------------------------------------------------------------------------
def _outcome(f, arg):
    """('ok', description) or ('exc', ExceptionType) of ``f(arg)`` (warnings silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = f(arg)
        except Exception as e:  # noqa: BLE001
            return ("exc", type(e).__name__)
    if r is MSK or r is NMSK:
        return ("ok", "masked")
    if isinstance(r, tuple):
        return ("ok", tuple("masked" if (v is MSK or v is NMSK) else type(v).__name__ for v in r))
    if hasattr(r, "mask") and hasattr(r, "dtype"):
        return ("ok", ("array", tuple(r.shape), str(r.dtype), np.asarray(r.mask).tolist()))
    return ("ok", (type(r).__name__.replace("64", "").replace("32", ""), str(r)))


_ARR3 = np.arange(3.0)
_MASKED_SURFACE = {
    "np.sqrt": np.sqrt, "np.exp": np.exp, "np.isnan": np.isnan, "np.isfinite": np.isfinite, "np.abs": np.abs,
    "np.negative": np.negative, "np.floor": np.floor, "np.add_1": lambda m: np.add(m, 1),
    "np.add_arr": lambda m: np.add(_ARR3, m), "np.greater_arr": lambda m: np.greater(m, _ARR3),
    "np.maximum": lambda m: np.maximum(m, 1), "np.add.outer": lambda m: np.add.outer(m, _ARR3),
    "np.sum": np.sum, "arr+m": lambda m: _ARR3 + m, "m+arr": lambda m: m + _ARR3,
    "np.left_shift": lambda m: np.left_shift(m, 1), "np.divmod": lambda m: np.divmod(m, 2),
    "np.modf": np.modf, "np.matmul": lambda m: np.matmul(m, m),
    "sum": lambda m: m.sum(), "prod": lambda m: m.prod(), "any": lambda m: m.any(), "all": lambda m: m.all(),
    "max": lambda m: m.max(), "min": lambda m: m.min(), "ptp": lambda m: m.ptp(),
    "argmax": lambda m: m.argmax(), "argmin": lambda m: m.argmin(),
    "cumsum": lambda m: m.cumsum(), "cumprod": lambda m: m.cumprod(),
    "astype_int": lambda m: m.astype(int), "astype_f32": lambda m: m.astype(np.float32),
    "reshape": lambda m: m.reshape(1), "reshape2": lambda m: m.reshape((1, 1)), "ravel": lambda m: m.ravel(),
    "flatten": lambda m: m.flatten(), "squeeze": lambda m: m.squeeze(), "transpose": lambda m: m.transpose(),
    "T": lambda m: m.T, "count": lambda m: m.count(), "round": lambda m: m.round(), "conj": lambda m: m.conj(),
    "clip": lambda m: m.clip(0, 1), "item": lambda m: m.item(), "tolist": lambda m: m.tolist(),
    "[()]": lambda m: m[()], "[...]": lambda m: m[...], "[0]": lambda m: m[0], "[None]": lambda m: m[None],
    "repeat": lambda m: m.repeat(2), "take": lambda m: m.take(0), "filled": lambda m: m.filled(3),
}


class TestMaskedConstantNumpySurface:
    @pytest.mark.parametrize("name", list(_MASKED_SURFACE))
    def test_matches_numpy_ma_masked(self, name):
        f = _MASKED_SURFACE[name]
        got, exp = _outcome(f, MSK), _outcome(f, NMSK)
        if got[0] == "ok" and exp[0] == "ok" and isinstance(exp[1], tuple) and exp[1][:1] == ("int",):
            exp = ("ok", ("int", exp[1][1]))
            got = ("ok", ("int", got[1][1]))
        assert got == exp

    def test_ndarray_op_masked_still_gives_masked_array(self, dev):
        a = to_dev(_ARR3, dev)
        for f in (lambda v: a + v, lambda v: v + a, lambda v: a * v, lambda v: np.add(a, v)):
            rx = f(MSK)
            assert isinstance(rx, XMA.MaskedArray)
            assert bool(np.all(host(XMA.getmaskarray(rx))))

    def test_masked_op_xma(self, dev):
        x, n = make(_ARR3, [0, 1, 0], dev)
        assert_same(MSK + x, NMSK + n, dev=dev)
        assert_same(np.add(MSK, x), np.add(NMSK, n), dev=dev)


_NOMASK_SURFACE = {}
for _name in ("add sub mul truediv floordiv mod pow lshift rshift and_ or_ xor lt le gt ge eq ne").split():
    _f = getattr(__import__("operator"), _name)
    for _tag, _o in (("1", 1), ("2.5", 2.5), ("arr", np.array([1, 0, 2])), ("nm", "nm"), ("True", True)):
        _NOMASK_SURFACE[f"{_name}({_tag})"] = (lambda m, _f=_f, _o=_o: _f(m, m if _o == "nm" else _o))
        _NOMASK_SURFACE[f"r{_name}({_tag})"] = (lambda m, _f=_f, _o=_o: _f(m if _o == "nm" else _o, m))
_NOMASK_SURFACE.update({
    "neg": lambda m: -m, "pos": lambda m: +m, "abs": abs, "int": int, "float": float, "complex": complex,
    "invert": lambda m: ~m, "sum": lambda m: m.sum(), "any": lambda m: m.any(), "all": lambda m: m.all(),
    "divmod": lambda m: divmod(m, 2), "rdivmod": lambda m: divmod(2, m), "np.add": lambda m: np.add(m, 1),
    "np.sum": np.sum, "round": round, "index": lambda m: [1, 2][m], "hash": lambda m: hash(m) == hash(np.False_),
})


class TestNomaskNumericSurface:
    @pytest.mark.parametrize("name", list(_NOMASK_SURFACE))
    def test_matches_numpy_nomask(self, name):
        f = _NOMASK_SURFACE[name]
        assert _outcome(f, NM) == _outcome(f, NNM)


class TestMaskNoneAndMisc:
    @pytest.mark.parametrize("src", ["list", "array", "xma", "npma"])
    def test_mask_none_like_numpy(self, src, dev):
        base = np.array([1.0, 2.0, 3.0])
        m = np.array([True, False, False])
        if src == "list":
            rx, rn = XMA.array([1.0, 2.0], mask=None), np.ma.array([1.0, 2.0], mask=None)
        elif src == "array":
            with xupy.backend(dev):
                rx, rn = XMA.array(to_dev(base, dev), mask=None), np.ma.array(base, mask=None)
        elif src == "xma":
            x, n = make(base, m, dev)
            rx, rn = XMA.array(x, mask=None), np.ma.array(n, mask=None)
        else:
            _, n = make(base, m, dev)
            with xupy.backend(dev):  # a numpy.ma source goes to the active backend
                rx, rn = XMA.array(n, mask=None), np.ma.array(n, mask=None)
        assert_same(rx, rn, dev=None if src == "list" else dev)

    @pytest.mark.parametrize("data", [[True, False], [1, 2], [1.5, 2.5], [1 + 2j, 3j]])
    def test_conj_dtype(self, data, dev):
        x, n = make(np.array(data), [0, 1], dev)
        assert_same(x.conj(), n.conj(), dev=dev)
        assert_same(x.conjugate(), n.conjugate(), dev=dev)
        assert x.conj().dtype == n.conj().dtype

    @pytest.mark.parametrize("mk", [True, False])
    def test_clip_0d(self, mk, dev):
        x, n = make(np.array(1.0), np.array(mk), dev)
        rx, rn = x.clip(0, 1), n.clip(0, 1)
        if rn is NMSK:
            assert rx is MSK
        else:
            assert_same(rx, rn, dev=dev, strict_nomask=False)  # see NO-SYNC note

    def test_round_out_plain_ndarray(self, dev):
        x, n = make(np.array([1.26, 2.54, 3.0]), [0, 1, 0], dev)
        ox, on = to_dev(np.zeros(3), dev), np.zeros(3)
        rx, rn = x.round(1, out=ox), n.round(1, out=on)
        assert rx is ox and rn is on
        np.testing.assert_array_equal(host(ox), on)
        with pytest.raises(TypeError):
            x.round(1, out=3)

    @pytest.mark.parametrize("name", ["ones_like", "zeros_like", "empty_like"])
    @pytest.mark.parametrize("kw", [{}, {"dtype": np.int32}, {"shape": (3,)}])
    def test_numpy_like_functions(self, name, kw, dev):
        x, n = make(np.array([1.5, 2.5, 3.5]), [0, 1, 0], dev)
        rx, rn = getattr(np, name)(x, **kw), getattr(np, name)(n, **kw)
        assert isinstance(rx, XMA.MaskedArray)
        assert rx.dtype == rn.dtype and rx.shape == rn.shape
        np.testing.assert_array_equal(host(XMA.getmaskarray(rx)), np.ma.getmaskarray(rn))
        if name != "empty_like":
            np.testing.assert_array_equal(host(rx.data), np.asarray(rn.data))

    @pytest.mark.parametrize("kw", [{}, {"dtype": np.int32}])
    def test_numpy_full_like(self, kw, dev):
        x, n = make(np.array([1.5, 2.5, 3.5]), [0, 1, 0], dev)
        rx, rn = np.full_like(x, 7, **kw), np.full_like(n, 7, **kw)
        assert isinstance(rx, XMA.MaskedArray)
        assert_same(rx, rn, dev=dev, check_fill=False)
