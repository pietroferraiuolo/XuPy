"""Differential tests of operators / ufunc interop of ``xupy.ma`` against ``numpy.ma``.

Every test builds the same operands as XuPy masked arrays (numpy or cupy data,
fixture ``dev``) and as ``numpy.ma`` arrays, applies the same callable to both
and compares the results (``assert_same``).  If numpy raises, XuPy must raise
the same exception type.  numpy.ma is the source of truth, quirks included.

Size policy: small arrays, hand-picked representative dtypes / mask patterns,
and loops over small tables INSIDE a test (collected through ``Cases`` so that
one failing combination does not hide the others) instead of huge
parametrize cross products.
"""
# NO-SYNC: numpy.ma shrinks the mask to nomask via a host-side .any(); xupy.ma never syncs
# (DESIGN decision 6), so a result mask may be a full all-False array where numpy.ma has
# nomask.  Calls that pass `strict_nomask=False` accept exactly that difference (see NO-SYNC note).

import operator
import warnings

import numpy as np
import pytest

import xupy

from ._ma_parity_helpers import (
    GPU_OK, XMA, assert_same, cp, host, make, on_dev, to_dev,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
class Cases:
    """Run many small cases in one test; report all failing labels at the end."""

    def __init__(self):
        self.n = 0
        self.fails = []

    def run(self, label, fn, *args, **kw):
        self.n += 1
        try:
            fn(*args, **kw)
        except (Exception, pytest.fail.Exception) as e:  # noqa: BLE001
            self.fails.append(f"[{label}] {type(e).__name__}: {str(e)[:160]}")

    def done(self):
        assert not self.fails, f"{len(self.fails)}/{self.n} cases failed:\n" + "\n".join(self.fails[:8])


def sample(dtype, shape, seed=0):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        a = rng.integers(0, 2, shape).astype(bool)
    elif dtype.kind in "iu":
        a = rng.integers(-4, 5, shape).astype(dtype)
    elif dtype.kind == "f":
        a = np.round(rng.normal(0, 2, shape), 1).astype(dtype)
    else:
        a = (np.round(rng.normal(0, 2, shape), 1) + 1j * np.round(rng.normal(0, 2, shape), 1)).astype(dtype)
    if a.size > 2 and dtype.kind != "b":
        a.flat[0] = 0
    return a


def mask_for(kind, shape, seed=1):
    if kind is None:
        return None
    if kind == "full":
        return np.ones(shape, bool)
    if kind == "false":
        return np.zeros(shape, bool)
    rng = np.random.default_rng(seed)
    m = rng.random(shape) < 0.4
    if m.size >= 2:
        m.flat[0], m.flat[-1] = True, False
    return m


def pair(dt, shape, mk=None, dev="cpu", seed=0, **kw):
    return make(sample(dt, shape, seed), mask_for(mk, shape, seed + 7), dev, **kw)


def nd_pair(dt, shape, dev, seed=5):
    a = sample(dt, shape, seed)
    return to_dev(a.copy(), dev), a.copy()


def host_pair(dt, shape, seed=5):
    a = sample(dt, shape, seed)
    return a.copy(), a.copy()


def check(fn, dev, *pairs, **akw):
    """Apply ``fn`` to the XuPy and numpy.ma operands and compare."""
    xs = [p[0] for p in pairs]
    ns = [p[1] for p in pairs]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            rn = fn(*ns)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                fn(*xs)
            return None
        rx = fn(*xs)
    assert_same(rx, rn, dev=dev, **akw)
    return rx, rn


def assert_data_full(x, n, ctx="data"):
    """Compare ALL data (also under masked positions) incl. NaN positions."""
    hx, hn = host(x.data), np.asarray(n.data)
    assert hx.shape == hn.shape
    if hn.dtype.kind in "fc":
        np.testing.assert_allclose(hx, hn, rtol=float(np.finfo(hn.dtype).eps) * 64, atol=0,
                                   equal_nan=True, err_msg=ctx)
    else:
        np.testing.assert_array_equal(hx, hn, err_msg=ctx)


def swap(op):
    return lambda a, b: op(b, a)


def fwd(op, rev):
    return swap(op) if rev else op


def dn(d):
    return np.dtype(d).name


def gpu_intpow_unchecked(dev, op, base_dt, exp_dt):
    """True for the documented cupy limitation: integer ``**`` negative integer does not raise when the
    exponent is a cupy array (it would need a host sync); numpy raises ValueError.  Still checked on cpu."""
    return (dev == "gpu" and op is operator.pow and np.dtype(base_dt).kind in "biu"
            and np.dtype(exp_dt).kind in "iu")


# representative dtypes
DT4 = [np.int32, np.float64, np.bool_, np.complex128]
DT5 = [np.int32, np.float32, np.float64, np.bool_, np.complex128]
# (left dtype, right dtype)
DT_PAIRS = [(np.int32, np.int32), (np.int32, np.float64), (np.float32, np.float64), (np.float64, np.float32),
            (np.int64, np.float32), (np.bool_, np.bool_), (np.bool_, np.int32), (np.complex128, np.float64),
            (np.int64, np.int32), (np.float32, np.complex128), (np.float64, np.float64),
            (np.complex128, np.complex128), (np.int32, np.bool_)]

ARITH = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul,
    "truediv": operator.truediv, "floordiv": operator.floordiv, "mod": operator.mod,
    "pow": operator.pow,
}
BITS = {"and": operator.and_, "or": operator.or_, "xor": operator.xor,
        "lshift": operator.lshift, "rshift": operator.rshift}
CMP = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
       "gt": operator.gt, "ge": operator.ge}
BINOPS = {**ARITH, **BITS, "divmod": divmod, **CMP}
IOPS = {**ARITH, **BITS}   # inplace variants (operator.i<name>)
op_param = pytest.mark.parametrize("op", list(BINOPS.values()), ids=list(BINOPS))

IOP_FUNCS = {k: getattr(operator, "i" + k) for k in IOPS}
iop_param = pytest.mark.parametrize("iop", list(IOP_FUNCS.values()), ids=list(IOP_FUNCS))


@pytest.fixture
def gpu():
    if not GPU_OK:
        pytest.skip("no usable GPU")
    return "gpu"


@pytest.fixture
def restore_backend():
    orig = xupy.on_gpu
    yield
    (xupy.use_gpu if orig else xupy.use_cpu)()


# ---------------------------------------------------------------------------
# 1. binary operators
# ---------------------------------------------------------------------------
class TestBinaryMaMa:
    @op_param
    def test_dtype_pairs(self, dev, op):
        c = Cases()
        for da, db in DT_PAIRS:
            if gpu_intpow_unchecked(dev, op, da, db):
                continue
            c.run(f"{dn(da)}-{dn(db)}", check, op, dev, pair(da, (3, 4), "partial", dev, 0),
                  pair(db, (3, 4), "partial", dev, 3))
        c.done()

    def test_pow_nonneg_exponent(self, dev):
        c = Cases()
        for name, op in list(ARITH.items()) + [("divmod", divmod)]:
            for dt in (np.int32, np.int64, np.float32, np.float64):
                a = np.abs(sample(dt, (3, 4), 0))
                b = np.abs(sample(dt, (3, 4), 1))
                c.run(f"{name}-{dn(dt)}", check, op, dev, make(a, mask_for("partial", (3, 4)), dev),
                      make(b, mask_for("partial", (3, 4), 9), dev))
        c.done()

    @pytest.mark.parametrize("sa,sb", [
        ((), ()), ((5,), (5,)), ((3, 4), (3, 4)), ((2, 3, 4), (2, 3, 4)),
        ((3, 4), (4,)), ((3, 1), (1, 4)), ((2, 1, 4), (3, 1)), ((), (3, 4)), ((3, 4), ()),
        ((0,), (0,)), ((0, 3), (3,)), ((1,), (5,)), ((3,), (4,)), ((2, 3), (4, 3)),
    ], ids=str)
    def test_shapes_and_broadcast(self, dev, sa, sb):
        c = Cases()
        for name in ("add", "sub", "mul", "truediv", "floordiv", "mod", "eq", "lt"):
            for dt in (np.int32, np.float64):
                c.run(f"{name}-{dn(dt)}", check, {**ARITH, **CMP}[name], dev,
                      pair(dt, sa, "partial", dev, 0), pair(dt, sb, "partial", dev, 3))
        c.done()

    @pytest.mark.parametrize("op", [operator.add, operator.truediv, operator.floordiv, operator.eq],
                             ids=["add", "truediv", "floordiv", "eq"])
    def test_mask_kinds(self, dev, op):
        c = Cases()
        for dt in (np.int64, np.float64):
            for ma in (None, "partial", "full", "false"):
                for mb in (None, "partial", "full", "false"):
                    # comparisons: no host sync -> the mask is not shrunk to nomask (strict_nomask=False)
                    c.run(f"{dn(dt)} {ma}/{mb}", check, op, dev, pair(dt, (3, 4), ma, dev, 0), pair(dt, (3, 4), mb, dev, 3),
                          strict_nomask=op is not operator.eq)
        c.done()

    def test_hardmask_propagation(self, dev):
        c = Cases()
        for op in (operator.add, operator.truediv, operator.lt):
            for ha, hb in ((True, False), (False, True), (True, True)):
                c.run(f"{op.__name__} {ha}/{hb}", check, op, dev,
                      pair(np.float64, (3, 4), "partial", dev, 0, hard_mask=ha),
                      pair(np.float64, (3, 4), "partial", dev, 3, hard_mask=hb))
        c.done()

    def test_fill_value_propagation(self, dev):
        c = Cases()
        for op in (operator.add, operator.mul, operator.truediv, operator.eq, operator.floordiv):
            for fa, fb in ((7, None), (None, 7), (7, 9)):
                ka = {} if fa is None else {"fill_value": fa}
                kb = {} if fb is None else {"fill_value": fb}
                c.run(f"{op.__name__} {fa}/{fb}", check, op, dev,
                      pair(np.int64, (3, 4), "partial", dev, 0, **ka), pair(np.int64, (3, 4), "partial", dev, 3, **kb))
        for op in (operator.add, operator.truediv, operator.mul):
            c.run(f"float fill {op.__name__}", check, op, dev,
                  pair(np.float64, (4,), "partial", dev, 0, fill_value=-1.5), pair(np.float64, (4,), "partial", dev, 3))
        c.done()

    def test_operands_not_modified(self, dev):
        (xa, na), (xb, nb) = pair(np.float64, (3, 4), "partial", dev, 0), pair(np.float64, (3, 4), None, dev, 1)
        r = xa + xb
        r[0, 0] = XMA.masked
        r.data[...] = 99
        assert_same(xa, na, dev=dev)
        assert_same(xb, nb, dev=dev)

    def test_result_mask_independent(self, dev):
        (xa, _), (xb, _) = pair(np.float64, (3, 4), "partial", dev, 0), pair(np.float64, (3, 4), "partial", dev, 3)
        r = xa + xb
        before = host(XMA.getmaskarray(xa)).copy()
        r[1, 1] = XMA.masked
        r.mask[...] = True
        np.testing.assert_array_equal(host(XMA.getmaskarray(xa)), before)


class TestSpecialValues:
    GRID = np.array([-np.inf, -2.0, -0.5, -0.0, 0.0, 1e-320, 0.5, 1.0, 3.0, 1e308, np.nan, np.inf])

    @op_param
    def test_grid(self, dev, op):
        c = Cases()
        for dt in (np.float64, np.float32):
            for mk in (None, "partial"):
                g = self.GRID.astype(dt)
                a = make(g[:, None], mask_for(mk, (g.size, 1)), dev)
                b = make(g[None, :], mask_for(mk, (1, g.size), 4), dev)
                c.run(f"{dn(dt)} {mk}", check, op, dev, a, b)
        c.done()

    @pytest.mark.parametrize("name", ["add", "sub", "mul", "pow"])
    def test_nan_inf_pass_through(self, dev, name):
        op = ARITH[name]
        a = make(np.array([np.nan, 1.0, np.inf, -np.inf, 2.0]), None, dev)
        b = make(np.array([1.0, np.nan, 2.0, 2.0, np.inf]), None, dev)
        r = check(op, dev, a, b)
        assert r is not None
        assert int(host(XMA.getmaskarray(r[0])).sum()) == int(np.ma.getmaskarray(r[1]).sum())

    def test_nan_not_masked_on_construction_ops(self, dev):
        x, n = make(np.array([np.nan, 1.0, np.inf]), None, dev)
        for f in (lambda a: a + 1, lambda a: a * 0, lambda a: a - a, lambda a: a ** 2, lambda a: -a, lambda a: abs(a)):
            check(f, dev, (x, n))
            if f(n).mask is not np.ma.nomask and np.ma.getmaskarray(f(n)).any():
                continue  # numpy.ma itself masks the non-finite results of `a ** 2` (domained pow)
            assert not host(XMA.getmaskarray(f(x))).any()

    def test_int_division_by_zero_masked(self, dev):
        c = Cases()
        for dt in (np.int32, np.int64):
            for op in (operator.floordiv, operator.mod, divmod):
                a = make(np.array([5, -5, 0, 7, 3], dt), None, dev)
                b = make(np.array([0, 0, 0, 2, -2], dt), None, dev)
                r = check(op, dev, a, b)
                assert r is not None
                if op is divmod:
                    continue  # numpy.ma does NOT mask divmod by zero (parity already checked above)
                out = r[0][0] if isinstance(r[0], tuple) else r[0]
                c.run(f"{dn(dt)} {op.__name__} masked", lambda o=out: host(XMA.getmaskarray(o))[:3].all() or pytest.fail("not masked"))
        c.done()

    def test_float_division_by_zero(self, dev):
        c = Cases()
        for dt in (np.float32, np.float64):
            for op in (operator.truediv, operator.floordiv, operator.mod, divmod):
                a = make(np.array([1.0, -1.0, 0.0, np.nan, np.inf, 5.0], dt), None, dev)
                b = make(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.0], dt), None, dev)
                c.run(f"{dn(dt)} {op.__name__}", check, op, dev, a, b)
        c.done()

    def test_truediv_tiny_denominator(self, dev):
        c = Cases()
        for dt in (np.float32, np.float64):
            if dev == "gpu" and dt is np.float32:
                # GPU limitation: cupy flushes float32 denormals to zero (the denormal denominators are
                # then masked as zero); float64 and cpu are checked.
                continue
            tiny = np.finfo(dt).tiny
            a = make(np.array([1.0, 1.0, 1.0, 1.0, 1.0], dt), None, dev)
            b = make(np.array([tiny, tiny / 2, tiny * 10, 1e-45, np.finfo(dt).max], dt), None, dev)
            c.run(dn(dt), check, operator.truediv, dev, a, b)
        c.done()

    def test_pow_domain(self, dev):
        a = make(np.array([-8.0, -1.0, 0.0, 0.0, 2.0, 4.0, np.nan, np.inf]), None, dev)
        b = make(np.array([1 / 3, 0.5, -1.0, 0.0, 0.5, -0.5, 2.0, 0.5]), None, dev)
        check(operator.pow, dev, a, b)

    def test_int_negative_power_raises_like_numpy(self, dev):
        if dev == "gpu":
            # cupy limitation: a cupy exponent array is not checked (needs a host sync); host exponents are.
            e = (np.array([1, -1, 2]),) * 2
            check(operator.pow, dev, make(np.array([1, 2, 3]), None, dev), e)
            check(operator.pow, dev, make(np.array([1, 2, 3]), [0, 1, 0], dev), e)
            return
        check(operator.pow, dev, make(np.array([1, 2, 3]), None, dev), make(np.array([1, -1, 2]), None, dev))
        check(operator.pow, dev, make(np.array([1, 2, 3]), [0, 1, 0], dev), make(np.array([1, -1, 2]), None, dev))

    def test_zero_pow_zero(self, dev):
        # strict_nomask=False: no host sync, so float pow does not shrink the mask to nomask
        check(operator.pow, dev, make(np.zeros(3), None, dev), make(np.zeros(3), None, dev),
              strict_nomask=False)  # see NO-SYNC note


class TestBinaryScalars:
    SCALARS = [2, 0, -3, 2.5, 0.0, True, 1 + 2j, 1e300, 2 ** 40, 2 ** 70, float("nan"), float("inf"), -(2 ** 63)]
    NP_SCALARS = [np.float64(0.5), np.float32(0.5), np.int64(3), np.int8(3), np.bool_(True),
                  np.complex64(1 + 1j), np.uint8(200), np.float16(2.0)]

    @op_param
    def test_xma_op_scalar_and_reflected(self, dev, op):
        c = Cases()
        for dt in DT4:
            for s in self.SCALARS:
                if dev == "gpu" and s == 2 ** 70 and type(s) is int:
                    # cupy limitation: numpy.ma turns 2**70 into an object array, which the GPU cannot hold
                    # (XuPy raises TypeError on the GPU); still checked on cpu.
                    continue
                if dev == "gpu" and (
                        (op is operator.truediv and np.dtype(dt).kind == "c" and s == float("inf"))
                        or (op in (operator.lshift, operator.rshift) and np.dtype(dt).kind == "b"
                            and s in (2 ** 40, -(2 ** 63)))):
                    # cupy limitations (cpu still checks them): complex division by inf gives nan+nanj
                    # (numpy: 0j); shift counts >= the bit width wrap around (C undefined behaviour; numpy: 0 / -1).
                    continue
                if op is operator.pow and s == 2 ** 70 and np.dtype(dt).kind in "iu":
                    # numpy.ma itself turns 2**70 into an object array and then computes
                    # int ** 2**70 elementwise in Python (effectively never terminates).
                    continue
                c.run(f"{dn(dt)} x op {s!r}", check, op, dev, pair(dt, (3, 4), "partial", dev), (s, s))
            for s in (2, 2.5, True, 1 + 2j):
                if dev == "gpu" and op is operator.pow and np.dtype(dt).kind in "iu" and type(s) in (int, bool):
                    # cupy limitation: `int ** int cupy array` does not raise for negative exponents
                    # (it would need a host sync); still checked on cpu.
                    continue
                c.run(f"{dn(dt)} {s!r} op x", check, swap(op), dev, pair(dt, (3, 4), "partial", dev), (s, s))
        c.done()

    @pytest.mark.parametrize("rev", [False, True], ids=["lr", "rl"])
    def test_numpy_scalars(self, dev, rev):
        c = Cases()
        for name, op in {**ARITH, **CMP}.items():
            for dt in (np.int32, np.float32, np.bool_, np.complex128):
                for s in self.NP_SCALARS:
                    if dev == "gpu" and name == "pow" and (
                            (rev and np.dtype(dt).kind in "iu" and s.dtype.kind in "iub")
                            or (not rev and dt is np.complex128 and s.dtype == np.uint8)):
                        # cupy limitations (cpu still checks them): `int ** int cupy array` does not raise for
                        # negative exponents (needs a host sync); cupy complex ** 200 is less accurate than numpy.
                        continue
                    c.run(f"{name} {dn(dt)} {type(s).__name__}", check, fwd(op, rev), dev,
                          pair(dt, (6,), "partial", dev), (s, s))
        c.done()

    def test_overflow_and_nep50(self, dev):
        c = Cases()
        scal = [1e300, 1e-300, 1e39, 3.5, 16777217.0, 2 ** 31, 2 ** 63, -(2 ** 31) - 1, 255, 256, 128]
        for op in (operator.add, operator.mul, operator.truediv, operator.eq, operator.lt):
            for dt in (np.float32, np.int32, np.int64, np.float64):
                for s in scal:
                    c.run(f"{op.__name__} {dn(dt)} {s!r}", check, op, dev, pair(dt, (5,), "partial", dev), (s, s))
        c.done()

    def test_small_dtypes(self, dev):
        c = Cases()
        for op in (operator.add, operator.sub, operator.mul, operator.eq):
            for dt in (np.int8, np.uint8, np.int16, np.float16):
                for s in (100, 200, 300, -1, 2.0):
                    c.run(f"{op.__name__} {dn(dt)} {s!r}", check, op, dev, pair(dt, (5,), "partial", dev), (s, s))
        c.done()

    def test_scalar_mask_kinds(self, dev):
        c = Cases()
        for op in (operator.add, operator.mul, operator.lt):
            for s in (2, 2.5):
                for mk in (None, "full", "false"):
                    # comparison: no host sync -> mask not shrunk to nomask (strict_nomask=False)
                    c.run(f"{op.__name__} {s} {mk}", check, op, dev, pair(np.float64, (3, 4), mk, dev), (s, s),
                          strict_nomask=op is not operator.lt)
        c.done()

    def test_zero_d(self, dev):
        c = Cases()
        for s in (2, 0):
            for op in (operator.add, operator.truediv, operator.floordiv, operator.mod, operator.lt):
                c.run(f"{op.__name__} {s} nomask", check, op, dev, pair(np.float64, (), None, dev), (s, s))
                c.run(f"{op.__name__} {s} full", check, op, dev, pair(np.int64, (), "full", dev), (s, s))
                c.run(f"{op.__name__} {s} false", check, op, dev, pair(np.int64, (), "false", dev), (s, s))
        c.done()


class TestBinaryArrays:
    @op_param
    def test_xma_op_ndarray_and_reflected(self, dev, op):
        c = Cases()
        for da, db in DT_PAIRS:
            for rev in (False, True):
                if gpu_intpow_unchecked(dev, op, *((db, da) if rev else (da, db))):
                    continue
                c.run(f"{dn(da)}-{dn(db)} rev={rev}", check, fwd(op, rev), dev,
                      pair(da, (3, 4), "partial", dev), nd_pair(db, (3, 4), dev))
        c.done()

    def test_ndarray_broadcast(self, dev):
        c = Cases()
        for op in (operator.add, operator.mul, operator.truediv, operator.lt, operator.eq):
            for rev in (False, True):
                for sb in ((4,), (3, 1), (), (2, 3, 4)):
                    c.run(f"{op.__name__} rev={rev} {sb}", check, fwd(op, rev), dev,
                          pair(np.float64, (3, 4), "partial", dev), nd_pair(np.float64, sb, dev))
        c.done()

    def test_ndarray_mask_kinds(self, dev):
        c = Cases()
        for op in (operator.add, operator.truediv, operator.lt, operator.floordiv):
            for rev in (False, True):
                for mk in (None, "full", "false"):
                    # comparison: no host sync -> mask not shrunk to nomask (strict_nomask=False)
                    c.run(f"{op.__name__} rev={rev} {mk}", check, fwd(op, rev), dev,
                          pair(np.float64, (3, 4), mk, dev), nd_pair(np.float64, (3, 4), dev),
                          strict_nomask=op is not operator.lt)
        c.done()

    def test_host_ndarray_operand(self, dev):
        """A numpy array operand on any device: result device follows cupy-wins."""
        c = Cases()
        for op in (operator.add, operator.sub, operator.mul, operator.truediv, operator.lt, operator.eq):
            for rev in (False, True):
                for dt in (np.float64, np.int32):
                    c.run(f"{op.__name__} rev={rev} {dn(dt)}", check, fwd(op, rev), dev,
                          pair(dt, (3, 4), "partial", dev), host_pair(dt, (3, 4)))
        c.done()

    def test_list_and_tuple_operands(self, dev):
        c = Cases()
        lst = [[1, 2, 3, 4], [0, -1, 2, 5], [3, 3, 0, 1]]
        for op in (operator.add, operator.mul, operator.truediv, operator.lt, operator.floordiv, operator.pow):
            for rev in (False, True):
                for dt in (np.int32, np.float64, np.complex128):
                    f = fwd(op, rev)
                    if rev and gpu_intpow_unchecked(dev, op, np.int64, dt):
                        continue  # list ** cupy int array with negative exponents (see gpu_intpow_unchecked)
                    c.run(f"{op.__name__} rev={rev} {dn(dt)} 2d", check, f, dev, pair(dt, (3, 4), "partial", dev), (lst, lst))
                    c.run(f"{op.__name__} rev={rev} {dn(dt)} 1d", check, f, dev,
                          pair(dt, (3, 4), "partial", dev), ([1.5, 2, 0, 1], [1.5, 2, 0, 1]))
        for op in (operator.add, operator.truediv, operator.eq, operator.floordiv):
            for rev in (False, True):
                f = fwd(op, rev)
                c.run(f"{op.__name__} rev={rev} tuple", check, f, dev, pair(np.int64, (4,), "partial", dev), ((1, 2, 0, 3),) * 2)
                c.run(f"{op.__name__} rev={rev} boollist", check, f, dev,
                      pair(np.int64, (4,), "partial", dev), ([True, False, True, True],) * 2)
        c.done()

    def test_masked_constant(self, dev):
        mm = (XMA.masked, np.ma.masked)
        c = Cases()
        for name, op in BINOPS.items():
            for dt in DT4:
                for rev in (False, True):
                    f = fwd(op, rev)
                    c.run(f"{name} {dn(dt)} rev={rev} partial", check, f, dev, pair(dt, (3, 4), "partial", dev), mm)
                    c.run(f"{name} {dn(dt)} rev={rev} nomask", check, f, dev, pair(dt, (3, 4), None, dev), mm)
                    c.run(f"{name} {dn(dt)} rev={rev} 0d", check, f, dev, pair(dt, (), None, dev), mm)
        c.done()

    def test_masked_op_masked(self, dev):
        for op in (operator.add, operator.eq, operator.truediv):
            assert op(XMA.masked, XMA.masked) is XMA.masked and op(np.ma.masked, np.ma.masked) is np.ma.masked
            assert op(XMA.masked, 3) is XMA.masked
            assert op(3, XMA.masked) is XMA.masked


class TestBinaryDtypeAndKind:
    def test_result_dtype_promotion(self, dev):
        c = Cases()
        prs = [(np.int32, np.float32), (np.int64, np.float32), (np.int8, np.int8), (np.uint8, np.int8),
               (np.uint64, np.int64), (np.float16, np.float32), (np.complex64, np.float64)]
        for op in (operator.add, operator.mul, operator.sub):
            for da, db in prs:
                c.run(f"{op.__name__} {dn(da)}-{dn(db)}", check, op, dev,
                      pair(da, (6,), "partial", dev), pair(db, (6,), "partial", dev, 2))
        c.done()

    def test_truediv_result_dtype(self, dev):
        for dt in (np.float32, np.float64, np.int32):
            r = pair(dt, (4,), None, dev)[0] / pair(dt, (4,), None, dev, 1)[0]
            assert r.dtype == (np.ma.array(np.ones(2, dt)) / np.ma.array(np.ones(2, dt))).dtype

    def test_comparison_returns_masked_bool(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev)
        r = x > 0
        assert isinstance(r, XMA.MaskedArray) and r.dtype == np.bool_
        assert_same(r, n > 0, dev=dev)

    def test_unmasked_in_nomask_out(self, dev):
        x, _ = pair(np.float64, (3, 4), None, dev)
        assert (x + 1).mask is XMA.nomask
        assert (x * x).mask is XMA.nomask
        assert (x == x).mask is XMA.nomask

    def test_result_has_input_device(self, dev):
        x, _ = pair(np.float64, (3, 4), "partial", dev)
        for r in (x + 1, 1 + x, x * x, x < 2, -x, x ** 2):
            assert on_dev(r.data, dev) and (r.mask is XMA.nomask or on_dev(r.mask, dev))


# ---------------------------------------------------------------------------
# 2. unary operators and unary ufunc table
# ---------------------------------------------------------------------------
UNARY_DUNDER = {"neg": operator.neg, "pos": operator.pos, "abs": abs, "invert": operator.invert}


class TestUnaryDunders:
    @pytest.mark.parametrize("f", list(UNARY_DUNDER.values()), ids=list(UNARY_DUNDER))
    def test_unary(self, dev, f):
        c = Cases()
        for dt in (np.int32, np.int64, np.float32, np.float64, np.bool_, np.complex128):
            for mk in (None, "partial", "full", "false"):
                for shape in ((), (5,), (3, 4), (0,)):
                    c.run(f"{dn(dt)} {mk} {shape}", check, f, dev, pair(dt, shape, mk, dev))
        c.done()

    def test_unary_data_under_mask_and_attrs(self, dev):
        for f in UNARY_DUNDER.values():
            x, n = pair(np.int64, (3, 4), "partial", dev, fill_value=5, hard_mask=True)
            r = check(f, dev, (x, n))
            assert_data_full(r[0], r[1])

    def test_unary_masked_constant(self, dev):
        for f in UNARY_DUNDER.values():
            try:
                rn = f(np.ma.masked)
            except Exception as e:  # noqa: BLE001
                with pytest.raises(type(e)):
                    f(XMA.masked)
            else:
                assert rn is np.ma.masked and f(XMA.masked) is XMA.masked

    def test_neg_of_nan_inf_unmasked(self, dev):
        x, n = make(np.array([np.nan, np.inf, -np.inf, 0.0, -0.0]), None, dev)
        check(operator.neg, dev, (x, n))
        check(abs, dev, (x, n))

    def test_abs_int_min(self, dev):
        x, n = make(np.array([np.iinfo(np.int32).min, -5, 5], np.int32), None, dev)
        check(abs, dev, (x, n))
        check(operator.neg, dev, (x, n))


FLOAT_GRID = np.array([-np.inf, -2.0, -1.0, -0.5, -0.0, 0.0, 1e-300, 0.5, 1.0, 1.0000001, 2.0, 100.0, np.nan, np.inf])
UNARY_FUNCS = [n for n in ("sqrt log log2 log10 exp sin cos tan arcsin arccos arctan sinh cosh tanh arcsinh arccosh "
                           "arctanh floor ceil conjugate negative absolute fabs").split()
               if hasattr(np.ma, n)]


def _grid(dt):
    if np.dtype(dt).kind in "ib":
        return np.nan_to_num(FLOAT_GRID, posinf=1e30, neginf=-1e30).astype(dt)
    return FLOAT_GRID.astype(dt)


class TestUnaryUfuncTable:
    @pytest.mark.parametrize("name", UNARY_FUNCS)
    def test_function_table(self, dev, name):
        """Module function, np ufunc dispatch, data under mask, shapes."""
        c = Cases()
        modf = lambda a: getattr(XMA if isinstance(a, XMA.MaskedArray) else np.ma, name)(a)  # noqa: E731
        for dt in (np.float64, np.float32, np.int32, np.bool_, np.complex128):
            if dev == "gpu" and name == "log2" and dt is np.complex128:
                # cupy limitation: complex log2(0j) gives -inf+nanj (numpy: -inf+0j); cpu is checked.
                continue
            for mk in (None, "partial"):
                g = _grid(dt)
                c.run(f"module {dn(dt)} {mk}", check, modf, dev, make(g, mask_for(mk, g.shape), dev))
                c.run(f"np.{name} {dn(dt)} {mk}", check, getattr(np, name), dev, make(g, mask_for(mk, g.shape), dev))
        x, n = make(FLOAT_GRID, mask_for("partial", FLOAT_GRID.shape), dev)
        r = check(getattr(np, name), dev, (x, n))
        if r is not None and isinstance(r[1], np.ma.MaskedArray):
            c.run("data under mask", assert_data_full, r[0], r[1], f"np.{name}")
        for shape in ((), (4,), (2, 3, 2)):
            a = np.linspace(-1.5, 1.5, int(np.prod(shape, dtype=int))).reshape(shape)
            c.run(f"shape {shape}", check, getattr(np, name), dev, make(a, mask_for("partial", shape), dev))
        c.done()

    def test_domain_masks_new_positions(self, dev):
        a = np.array([-1.0, 0.0, 0.5, 1.5, np.pi / 2, 1e-320, np.nan, np.inf])
        for name in ("sqrt", "log", "arccos", "arccosh", "tan"):
            x, n = make(a, None, dev)
            assert check(getattr(np, name), dev, (x, n)) is not None

    def test_masked_constant_and_python_scalars(self, dev):
        for name in ("sqrt", "log", "arcsin", "arctanh"):
            assert getattr(XMA, name)(XMA.masked) is XMA.masked
            for v in (0.5, -0.5, 2.0, 0.0):
                with np.errstate(all="ignore"):
                    assert_same(getattr(XMA, name)(v), getattr(np.ma, name)(v), ctx=f"{name}({v})")

    def test_round_method_and_function(self, dev):
        c = Cases()
        for dt in (np.float64, np.float32, np.int32, np.complex128):
            a = (sample(dt, (3, 4), 2) * 3.14159).astype(dt) if np.dtype(dt).kind in "fc" else sample(dt, (3, 4), 2) * 17
            for mk in (None, "partial"):
                x, n = make(a, mask_for(mk, (3, 4)), dev)
                for dec in (0, 1, -1, 3):
                    lab = f"{dn(dt)} {mk} dec={dec}"
                    c.run(lab + " method", check, lambda m: m.round(dec), dev, (x, n))
                    c.run(lab + " kw", check, lambda m: m.round(decimals=dec), dev, (x, n))
                    c.run(lab + " ma.round", check, lambda m: (XMA if isinstance(m, XMA.MaskedArray) else np.ma).round(m, dec), dev, (x, n))
                    c.run(lab + " np.round", check, lambda m: np.round(m, dec), dev, (x, n))
                    c.run(lab + " np.around", check, lambda m: np.around(m, dec), dev, (x, n))
        c.done()

    @pytest.mark.parametrize("dt", [np.float64, np.complex128, np.int32], ids=["float64", "complex128", "int32"])
    def test_conjugate_method(self, dev, dt):
        x, n = pair(dt, (3, 4), "partial", dev)
        check(lambda m: m.conjugate(), dev, (x, n))
        check(lambda m: m.conj(), dev, (x, n))
        check(lambda m: np.conj(m), dev, (x, n))

    def test_ufunc_out(self, dev):
        c = Cases()
        for name in ("sqrt", "exp", "negative"):
            x, n = pair(np.float64, (3, 4), "partial", dev)
            ox, on = pair(np.float64, (3, 4), None, dev, 9)
            c.run(name, check, lambda a, o: (getattr(np, name)(a, out=o), o), dev, (x, n), (ox, on))
        c.done()

    def test_predicates(self, dev):
        c = Cases()
        for name in ("isnan", "isfinite", "isinf", "signbit", "logical_not"):
            x, n = make(np.array([np.nan, np.inf, -1.0, 0.0, 2.0]), [0, 0, 1, 0, 1], dev)
            c.run(name, check, getattr(np, name), dev, (x, n))
        c.done()


# ---------------------------------------------------------------------------
# 3. in-place operators
# ---------------------------------------------------------------------------
def run_inplace(iop, dev, xa_na, other, check_data=True, strict_nomask=True):
    """Apply ``a iop= b`` on both sides; return (x, n) if numpy succeeded, None otherwise."""
    xa, na = xa_na
    xb, nb = other
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            rn = iop(na, nb)
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                iop(xa, xb)
            return None
        rx = iop(xa, xb)
    assert rx is xa, "in-place operator must return the same object"
    assert rn is na
    assert_same(xa, na, dev=dev, strict_nomask=strict_nomask)
    if check_data:
        assert_data_full(xa, na)
    assert bool(xa._sharedmask) == bool(na._sharedmask)
    return xa, na


class TestInplace:
    @iop_param
    def test_operand_kinds(self, dev, iop):
        c = Cases()
        for da, db in DT_PAIRS:
            if not (dev == "gpu" and iop is operator.ipow and da != np.bool_
                    and gpu_intpow_unchecked(dev, operator.pow, da, db)):
                c.run(f"ma {dn(da)}-{dn(db)}", run_inplace, iop, dev, pair(da, (3, 4), "partial", dev, 0),
                      pair(db, (3, 4), "partial", dev, 3))
            for host_other in (False, True):
                if dev == "gpu" and iop is operator.ipow and not host_other and da != np.bool_ \
                        and gpu_intpow_unchecked(dev, operator.pow, da, db):
                    continue  # cupy limitation: negative integer exponents in a cupy array are not detected
                other = host_pair(db, (3, 4)) if host_other else nd_pair(db, (3, 4), dev)
                c.run(f"ndarray host={host_other} {dn(da)}-{dn(db)}", run_inplace, iop, dev,
                      pair(da, (3, 4), "partial", dev), other)
        for dt in DT4:
            for s in (2, 0, 2.5, True, 1 + 2j, 2 ** 40, 1e300):
                c.run(f"scalar {dn(dt)} {s!r}", run_inplace, iop, dev, pair(dt, (3, 4), "partial", dev), (s, s))
        for dt in (np.int64, np.float64, np.bool_):
            c.run(f"masked const {dn(dt)} partial", run_inplace, iop, dev, pair(dt, (3, 4), "partial", dev),
                  (XMA.masked, np.ma.masked))
            c.run(f"masked const {dn(dt)} nomask", run_inplace, iop, dev, pair(dt, (3, 4), None, dev),
                  (XMA.masked, np.ma.masked))
        c.done()

    def test_mask_kinds_and_hardmask(self, dev):
        c = Cases()
        for iop in (operator.iadd, operator.itruediv, operator.ifloordiv, operator.imod, operator.imul):
            for dt in (np.int64, np.float64):
                for hard in (False, True):
                    for ma in (None, "partial", "full", "false"):
                        for mb in (None, "partial", "full", "false"):
                            # no host sync -> in-place ops never shrink the mask to nomask (strict_nomask=False)
                            c.run(f"{iop.__name__} {dn(dt)} hard={hard} {ma}/{mb}", run_inplace, iop, dev,
                                  pair(dt, (3, 4), ma, dev, 0, hard_mask=hard), pair(dt, (3, 4), mb, dev, 3),
                                  strict_nomask=False)  # see NO-SYNC note
        c.done()

    def test_broadcast_other(self, dev):
        c = Cases()
        for iop in (operator.iadd, operator.itruediv, operator.isub):
            for sb in ((4,), (3, 1), (), (1, 4), (5, 3, 4), (4, 3)):
                c.run(f"{iop.__name__} {sb}", run_inplace, iop, dev, pair(np.float64, (3, 4), "partial", dev),
                      pair(np.float64, sb, "partial", dev, 3))
        c.done()

    def test_shapes(self, dev):
        c = Cases()
        for sa in ((), (4,), (2, 3, 4), (0,)):
            for iop in (operator.iadd, operator.itruediv):
                c.run(f"{iop.__name__} {sa}", run_inplace, iop, dev, pair(np.float64, sa, "partial", dev),
                      pair(np.float64, sa, "partial", dev, 3))
        c.done()

    def test_data_under_masked_positions_unchanged(self, dev):
        for iop in (operator.iadd, operator.imul, operator.isub, operator.itruediv):
            x, n = pair(np.float64, (3, 4), "partial", dev)
            before = host(x.data).copy()
            m = host(XMA.getmaskarray(x)).copy()
            iop(x, 3.0)
            np.testing.assert_array_equal(host(x.data)[m], before[m])
            np.testing.assert_array_equal(host(XMA.getmaskarray(x)), m)

    def test_other_masked_positions_dont_alter_data(self, dev):
        for iop in (operator.iadd, operator.imul, operator.itruediv, operator.ifloordiv):
            x, n = pair(np.float64, (3, 4), None, dev)
            y, ny = pair(np.float64, (3, 4), "partial", dev, 3)
            before = host(x.data).copy()
            iop(x, y)
            iop(n, ny)
            assert_same(x, n, dev=dev)
            assert_data_full(x, n)
            if iop is operator.ifloordiv:
                continue  # numpy.ma `//=` DOES change the data under the other's mask (checked by assert_data_full)
            my = host(XMA.getmaskarray(y))
            np.testing.assert_array_equal(host(x.data)[my], before[my])

    def test_return_is_same_object(self, dev):
        x, _ = pair(np.float64, (3, 4), "partial", dev)
        r = x
        r += 1
        assert r is x
        r *= x
        assert r is x
        r -= XMA.masked
        assert r is x

    def test_float32_inplace_float64_keeps_float32(self, dev):
        x, n = pair(np.float32, (3, 4), "partial", dev)
        run_inplace(operator.iadd, dev, (x, n), nd_pair(np.float64, (3, 4), dev))
        assert x.dtype == np.float32 and x.data.dtype == np.float32
        x2, n2 = pair(np.float32, (3, 4), None, dev)
        run_inplace(operator.imul, dev, (x2, n2), (np.float64(1.5), np.float64(1.5)))
        assert x2.dtype == np.float32
        x3, n3 = pair(np.float32, (3, 4), None, dev)
        run_inplace(operator.imul, dev, (x3, n3), (1.5, 1.5))
        assert x3.dtype == np.float32

    def test_int_inplace_float_raises_like_numpy(self, dev):
        c = Cases()
        for iop in (operator.iadd, operator.imul, operator.itruediv, operator.ipow, operator.isub):
            for dt in (np.int32, np.int64, np.bool_):
                c.run(f"{iop.__name__} {dn(dt)} pyfloat", run_inplace, iop, dev, pair(dt, (3, 4), "partial", dev), (2.5, 2.5))
                for odt in (np.float64, np.float32, np.complex128):
                    c.run(f"{iop.__name__} {dn(dt)} {dn(odt)}arr", run_inplace, iop, dev,
                          pair(dt, (3, 4), "partial", dev), nd_pair(odt, (3, 4), dev))
        for dt in (np.int32, np.int64):
            c.run(f"int {dn(dt)} += float64 masked array", run_inplace, operator.iadd, dev,
                  pair(dt, (3, 4), "partial", dev), pair(np.float64, (3, 4), "partial", dev, 3))
        c.done()

    def test_division_by_zero_inplace(self, dev):
        c = Cases()
        for iop in (operator.ifloordiv, operator.imod, operator.itruediv):
            for dt in (np.int32, np.float64):
                a = make(np.array([5, -5, 0, 7, 3], dt), [0, 0, 0, 1, 0], dev)
                b = make(np.array([0, 0, 0, 2, 0], dt), None, dev)
                # no host sync -> in-place ops never shrink the mask to nomask (strict_nomask=False)
                c.run(f"{iop.__name__} {dn(dt)} masked target", run_inplace, iop, dev, a, b, strict_nomask=False)  # see NO-SYNC note
                a = make(np.array([5, -5, 0, 7, 3], dt), None, dev)
                c.run(f"{iop.__name__} {dn(dt)} array", run_inplace, iop, dev, a,
                      make(np.array([0, 1, 2, 0, 0], dt), None, dev), strict_nomask=False)  # see NO-SYNC note
                a = make(np.array([5, -5, 0, 7, 3], dt), None, dev)
                c.run(f"{iop.__name__} {dn(dt)} scalar 0", run_inplace, iop, dev, a, (0, 0), strict_nomask=False)  # see NO-SYNC note
        c.done()

    def test_pow_inplace_domain(self, dev):
        for dt in (np.float64, np.int64):
            a = make(np.array([4, 0, 2, 9], dt), None, dev)
            run_inplace(operator.ipow, dev, a, make(np.array([2, 0, 3, 2], dt), [0, 1, 0, 0], dev))
        a = make(np.array([4.0, -8.0, 0.0]), None, dev)
        run_inplace(operator.ipow, dev, a, make(np.array([0.5, 1 / 3, -1.0]), None, dev))

    def test_inplace_nan_inf_not_masked(self, dev):
        a = make(np.array([np.nan, 1.0, np.inf]), None, dev)
        run_inplace(operator.iadd, dev, a, (1.0, 1.0))
        assert not host(XMA.getmaskarray(a[0])).any()
        assert a[0].mask is XMA.nomask

    def test_ndarray_inplace_stays_on_device(self, dev):
        for dt in (np.int64, np.float64):
            x, n = pair(dt, (3, 4), "partial", dev)
            x += host_pair(dt, (3, 4))[0]
            assert on_dev(x.data, dev)


class TestInplaceAliasing:
    def test_nomask_target_masked_other_no_alias(self, dev):
        for iop in (operator.iadd, operator.imul, operator.isub, operator.itruediv, operator.ifloordiv):
            (c, nc), (d, nd) = pair(np.float64, (3, 4), None, dev, 0), pair(np.float64, (3, 4), "partial", dev, 3)
            iop(c, d)
            iop(nc, nd)
            assert_same(c, nc, dev=dev)
            dmask = host(XMA.getmaskarray(d)).copy()
            c[1, 1] = XMA.masked
            c.mask[...] = True
            nc[1, 1] = np.ma.masked
            np.testing.assert_array_equal(host(XMA.getmaskarray(d)), dmask)
            np.testing.assert_array_equal(np.ma.getmaskarray(nd), dmask)
            assert_same(d, nd, dev=dev)

    def test_both_nomask_then_mask_target(self, dev):
        (c, nc), (d, nd) = pair(np.float64, (3, 4), None, dev, 0), pair(np.float64, (3, 4), None, dev, 3)
        c += d
        nc += nd
        c[0, 0] = XMA.masked
        nc[0, 0] = np.ma.masked
        assert d.mask is XMA.nomask and nd.mask is np.ma.nomask
        assert_same(c, nc, dev=dev)
        assert_same(d, nd, dev=dev)

    def test_masked_target_masked_other_other_untouched(self, dev):
        (c, nc), (d, nd) = pair(np.float64, (3, 4), "partial", dev, 0), pair(np.float64, (3, 4), "partial", dev, 3)
        c += d
        nc += nd
        assert_same(c, nc, dev=dev)
        assert_same(d, nd, dev=dev)

    def test_self_inplace(self, dev):
        for op in (operator.iadd, operator.imul):
            x, n = pair(np.float64, (3, 4), "partial", dev)
            op(x, x)
            op(n, n)
            assert_same(x, n, dev=dev)

    def test_constructor_mask_input_not_changed_by_ops(self, dev):
        m = mask_for("partial", (3, 4))
        xm = to_dev(m.copy(), dev)
        x = XMA.masked_array(to_dev(sample(np.float64, (3, 4)), dev), mask=xm)
        n = np.ma.masked_array(sample(np.float64, (3, 4)), mask=m)
        ox, on = make(sample(np.float64, (3, 4)), mask_for("partial", (3, 4), 11), dev)
        x += ox
        n += on
        # numpy.ma (2.5.3) shares the constructor-supplied mask with the array and `+=` writes
        # through it: the input mask array is mutated identically on both sides.
        np.testing.assert_array_equal(host(xm), m)

    def test_sharedmask_view(self, dev):
        for iop in (operator.iadd, operator.itruediv):
            x, n = pair(np.float64, (3, 4), "partial", dev, 0)
            vx, vn = x.view(), n.view()
            assert bool(vx._sharedmask) == bool(vn._sharedmask)
            o = pair(np.float64, (3, 4), "partial", dev, 3)
            iop(vx, o[0])
            iop(vn, o[1])
            assert_same(vx, vn, dev=dev)
            assert_same(x, n, dev=dev)          # numpy semantics for the original
            assert_data_full(x, n)
            assert bool(vx._sharedmask) == bool(vn._sharedmask)
            assert bool(x._sharedmask) == bool(n._sharedmask)

    def test_sharedmask_view_nomask_target(self, dev):
        x, n = pair(np.float64, (3, 4), None, dev, 0)
        vx, vn = x.view(), n.view()
        o = pair(np.float64, (3, 4), "partial", dev, 3)
        vx += o[0]
        vn += o[1]
        assert_same(vx, vn, dev=dev)
        assert_same(x, n, dev=dev)

    def test_sharedmask_slice_inplace(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev, 0)
        sx, sn = x[:2], n[:2]
        sx *= 2
        sn *= 2
        assert_same(x, n, dev=dev)
        assert_same(sx, sn, dev=dev)
        assert_data_full(x, n)
        sx[0, 0] = XMA.masked
        sn[0, 0] = np.ma.masked
        assert_same(x, n, dev=dev)

    def test_unshare_mask_then_inplace(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev, 0)
        vx, vn = x.view(), n.view()
        vx.unshare_mask()
        vn.unshare_mask()
        assert bool(vx._sharedmask) == bool(vn._sharedmask)
        o = pair(np.float64, (3, 4), "partial", dev, 3)
        vx += o[0]
        vn += o[1]
        assert_same(x, n, dev=dev)
        assert_same(vx, vn, dev=dev)

    def test_hardmask_inplace(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev, 0, hard_mask=True)
        run_inplace(operator.iadd, dev, (x, n), pair(np.float64, (3, 4), "partial", dev, 3))
        x2, n2 = pair(np.float64, (3, 4), "partial", dev, 0, hard_mask=True)
        run_inplace(operator.iadd, dev, (x2, n2), (5.0, 5.0))


# ---------------------------------------------------------------------------
# 4. matmul / dot
# ---------------------------------------------------------------------------
def ref_matmul(an, bn):
    """numpy.ma ``dot(strict=False)`` semantics generalised to batched shapes.

    Masked entries count as 0; a result element is masked iff no pair of
    unmasked operands contributes to it.  0-d results follow the XuPy
    scalar-return decision (numpy scalar or ``masked``).
    """
    am, bm = ~np.ma.getmaskarray(an), ~np.ma.getmaskarray(bn)
    d = np.matmul(np.ma.filled(an, 0), np.ma.filled(bn, 0))
    m = ~np.matmul(am, bm)
    if np.ndim(d) == 0:
        return np.ma.masked if m else d[()]
    return np.ma.masked_array(d, mask=m)


def check_matmul(dev, xa_na, xb_nb):
    (xa, na), (xb, nb) = xa_na, xb_nb
    ref = ref_matmul(na, nb)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        rx = operator.matmul(xa, xb)
    assert_same(rx, ref, dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
    return rx, ref


MM_SHAPES = [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)), ((2, 3), (3, 4)), ((4, 2), (2, 5)), ((1, 1), (1, 1)),
             ((2, 3, 4), (4, 5)), ((2, 3, 4), (2, 4, 5)), ((2, 1, 3, 4), (3, 4, 2)), ((5, 4), (3, 4, 2)),
             ((0, 3), (3, 2)), ((2, 0), (0, 3)), ((3, 3), (3, 3))]
MM_IDS = [f"{a}@{b}" for a, b in MM_SHAPES]


def test_matmul_oracle_matches_ma_dot():
    """Numpy-only: the oracle used below agrees with numpy.ma.dot (2-D or less)."""
    for sa, sb in [s for s in MM_SHAPES if len(s[0]) <= 2 and len(s[1]) <= 2]:
        an = np.ma.masked_array(sample(np.float64, sa), mask=mask_for("partial", sa))
        bn = np.ma.masked_array(sample(np.float64, sb, 1), mask=mask_for("partial", sb, 5))
        ref = ref_matmul(an, bn)
        d = np.ma.dot(an, bn, strict=False)
        if np.ndim(d) == 0:
            assert (ref is np.ma.masked) == bool(d.mask), (sa, sb)
        else:
            np.testing.assert_array_equal(np.ma.getmaskarray(ref), np.ma.getmaskarray(d))
            np.testing.assert_allclose(ref.filled(0), d.filled(0))


class TestMatmul:
    @pytest.mark.parametrize("sa,sb", MM_SHAPES, ids=MM_IDS)
    def test_shapes_and_masks(self, dev, sa, sb):
        c = Cases()
        for mk in ((None, None), ("partial", None), (None, "partial"), ("partial", "partial"), ("full", "partial"),
                   ("false", "false")):
            c.run(str(mk), check_matmul, dev, pair(np.float64, sa, mk[0], dev, 0), pair(np.float64, sb, mk[1], dev, 3))
        c.done()

    def test_dtype_promotion(self, dev):
        c = Cases()
        for da, db in DT_PAIRS + [(np.int8, np.int8), (np.float16, np.float16)]:
            def one(da=da, db=db):
                rx, _ = check_matmul(dev, pair(da, (2, 3), "partial", dev, 0), pair(db, (3, 4), "partial", dev, 3))
                assert rx.dtype == np.result_type(da, db)
            c.run(f"{dn(da)}@{dn(db)}", one)
        c.done()

    def test_int_at_float_not_zeros(self, dev):
        for dt in (np.int32, np.int64):
            a = make(np.arange(1, 7, dtype=dt).reshape(2, 3), None, dev)
            b = make(np.full((3, 2), 0.5), None, dev)
            rx, _ = check_matmul(dev, a, b)
            assert rx.dtype == np.float64
            np.testing.assert_allclose(host(rx.data), np.matmul(np.arange(1, 7).reshape(2, 3), np.full((3, 2), 0.5)))
            a = make(np.arange(1, 7, dtype=dt).reshape(2, 3), [[0, 1, 0], [0, 0, 0]], dev)
            nd = nd_pair(np.float64, (3, 2), dev)
            rx, _ = check_matmul(dev, a, (nd[0] * 0 + 0.5, nd[1] * 0 + 0.5))
            assert rx.dtype == np.float64

    def test_xma_at_ndarray_and_reflected(self, dev):
        c = Cases()
        for sa, sb in MM_SHAPES[:9]:
            for host_arr in (False, True):
                o = host_pair(np.float64, sb) if host_arr else nd_pair(np.float64, sb, dev)
                c.run(f"xma@nd {sa}@{sb} host={host_arr}", check_matmul, dev, pair(np.float64, sa, "partial", dev), o)
                o = host_pair(np.float64, sa) if host_arr else nd_pair(np.float64, sa, dev)
                xb, nb = pair(np.float64, sb, "partial", dev, 3)
                c.run(f"nd@xma {sa}@{sb} host={host_arr}", assert_same, o[0] @ xb, ref_matmul(o[1], nb),
                      dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
        c.done()

    def test_list_at_xma_and_xma_at_list(self, dev):
        for sa, sb in MM_SHAPES[:6]:
            la = sample(np.float64, sa).tolist()
            lb = sample(np.float64, sb, 1).tolist()
            xb, nb = pair(np.float64, sb, "partial", dev, 3)
            assert_same(la @ xb, ref_matmul(np.asarray(la), nb), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
            xa, na = pair(np.float64, sa, "partial", dev, 0)
            assert_same(xa @ lb, ref_matmul(na, np.asarray(lb)), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note

    def test_matmul_masked_values_are_zero_not_nan(self, dev):
        a = make(np.array([[1.0, np.nan], [3.0, 4.0]]), [[0, 1], [0, 0]], dev)
        b = make(np.array([[1.0, 2.0], [np.inf, 4.0]]), [[0, 0], [1, 0]], dev)
        rx, _ = check_matmul(dev, a, b)
        assert np.isfinite(host(rx.data)[~host(XMA.getmaskarray(rx))]).all()

    def test_fully_masked_operand(self, dev):
        rx, _ = check_matmul(dev, pair(np.float64, (2, 3), "full", dev), pair(np.float64, (3, 4), None, dev, 3))
        assert host(XMA.getmaskarray(rx)).all()

    def test_matmul_scalar_operand_raises(self, dev):
        x, n = pair(np.float64, (3, 3), None, dev)
        check(operator.matmul, dev, (x, n), (2.0, 2.0))
        check(operator.matmul, dev, (2.0, 2.0), (x, n))

    def test_matmul_shape_mismatch_raises(self, dev):
        with pytest.raises(ValueError):
            pair(np.float64, (2, 3), None, dev)[0] @ pair(np.float64, (4, 2), None, dev)[0]

    def test_dot_method_matches_ma_dot(self, dev):
        (xa, na), (xb, nb) = pair(np.float64, (2, 3), "partial", dev), pair(np.float64, (3, 4), "partial", dev, 3)
        assert_same(xa.dot(xb), na.dot(nb), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
        assert_same(XMA.dot(xa, xb), np.ma.dot(na, nb), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
        assert_same(XMA.dot(xa, xb, strict=True), np.ma.dot(na, nb, strict=True), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note

    def test_imatmul_nomask_square(self, dev):
        a = make(np.arange(9.0).reshape(3, 3), None, dev)[0]
        b = make(np.eye(3) * 2, None, dev)[0]
        orig = a
        a @= b
        assert a is orig
        np.testing.assert_allclose(host(a.data), np.arange(9.0).reshape(3, 3) @ (np.eye(3) * 2))
        assert on_dev(a.data, dev)


# ---------------------------------------------------------------------------
# 5. numpy interop
# ---------------------------------------------------------------------------
BIN_UFUNCS = ("add subtract multiply divide true_divide floor_divide remainder mod power maximum minimum fmax fmin "
              "arctan2 hypot copysign equal not_equal less less_equal greater greater_equal logical_and logical_or "
              "logical_xor").split()


def _power_rev_as_operator(a, b):
    """``np.power(b, a)`` on XuPy arrays vs. ``b ** a`` on numpy.ma arrays (see test_ufunc_xma_ndarray)."""
    if isinstance(a, np.ma.MaskedArray):
        return operator.pow(b, a)
    return np.power(b, a)


class TestUfuncDispatch:
    def test_ufunc_xma_xma(self, dev):
        c = Cases()
        for name in BIN_UFUNCS:
            for dt in (np.float64, np.int32):
                if name == "power" and gpu_intpow_unchecked(dev, operator.pow, dt, dt):
                    continue  # cupy limitation: np.power(int, negative int cupy array) does not raise
                c.run(f"{name} {dn(dt)}", check, getattr(np, name), dev, pair(dt, (3, 4), "partial", dev, 0),
                      pair(dt, (3, 4), "partial", dev, 3))
        c.done()

    def test_ufunc_xma_ndarray(self, dev):
        c = Cases()
        for name in BIN_UFUNCS:
            for rev in (False, True):
                f = fwd(getattr(np, name), rev)
                if name == "power" and rev:
                    # Documented divergence (Option A): np.power(ndarray, xma) behaves like the operator
                    # ``ndarray ** xma`` (non-finite results masked, see _ops.py header) whereas numpy.ma's
                    # wrap path does not mask them.  Compare against numpy.ma's *operator* instead.
                    f = _power_rev_as_operator
                c.run(f"{name} rev={rev}", check, f, dev,
                      pair(np.float64, (3, 4), "partial", dev), nd_pair(np.float64, (4,), dev))
        c.done()

    def test_ufunc_xma_scalar(self, dev):
        c = Cases()
        for name in BIN_UFUNCS[:8]:
            for rev in (False, True):
                c.run(f"{name} rev={rev}", check, fwd(getattr(np, name), rev), dev,
                      pair(np.float64, (3, 4), "partial", dev), (2.5, 2.5))
        c.done()

    def test_ones_op_xma(self, dev):
        c = Cases()
        for op in (operator.add, operator.mul, operator.sub, operator.truediv, operator.lt):
            for rev in (False, True):
                f = fwd(op, rev)
                c.run(f"{op.__name__} rev={rev} nd", check, f, dev, pair(np.float64, (4,), "partial", dev), nd_pair(np.float64, (4,), dev))
                c.run(f"{op.__name__} rev={rev} np.ones", check, f, dev, pair(np.float64, (4,), "partial", dev),
                      (np.ones(4), np.ones(4)))
        c.done()

    def test_np_add_nd_xma(self, dev):
        check(np.add, dev, nd_pair(np.float64, (3, 4), dev), pair(np.float64, (3, 4), "partial", dev))
        check(np.add, dev, pair(np.float64, (3, 4), "partial", dev), nd_pair(np.float64, (3, 4), dev))

    def test_ufunc_out_masked_array(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev, 0)
        y, ny = pair(np.float64, (3, 4), "partial", dev, 3)
        ox, on = pair(np.float64, (3, 4), None, dev, 9)
        r = check(lambda a, b, o: (np.add(a, b, out=o), o), dev, (x, n), (y, ny), (ox, on))
        assert r is not None

    def test_ufunc_out_plain_ndarray_never_drops_mask(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev)
        out = to_dev(np.zeros((3, 4)), dev)
        # DELIBERATE deviation: numpy.ma writes into the plain ndarray and silently drops the mask
        # (returns the ndarray); XuPy refuses instead of losing mask information.
        with pytest.raises(TypeError):
            np.add(x, 1.0, out=out)

    def test_ufunc_dtype_kwarg(self, dev):
        x, n = pair(np.int32, (3, 4), "partial", dev)
        y, ny = pair(np.int32, (3, 4), "partial", dev, 3)
        check(lambda a, b: np.add(a, b, dtype=np.float32), dev, (x, n), (y, ny), check_fill=False)
        check(lambda a, b: np.multiply(a, b, dtype=np.float64), dev, (x, n), (y, ny), check_fill=False)

    def test_ufunc_where_all_true(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev)
        w = (to_dev(np.ones((3, 4), bool), dev), np.ones((3, 4), bool))
        rx = np.add(x, 1.0, where=w[0])
        assert_same(rx, np.add(n, 1.0, where=w[1]), dev=dev)

    def test_ufunc_where_never_drops_mask(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev)
        w = np.ones((3, 4), bool)
        w[1, 1] = False
        rx = np.add(x, 1.0, where=to_dev(w, dev), out=XMA.masked_array(to_dev(np.zeros((3, 4)), dev)))
        m = np.ma.getmaskarray(n)
        got = host(XMA.getmaskarray(rx))
        assert got[m].all()
        np.testing.assert_allclose(host(rx.data)[~m & w], (n.data + 1.0)[~m & w])

    def test_ufunc_reduce_matches_numpy_ma(self, dev):
        c = Cases()
        for uf, meth in (("add", "sum"), ("multiply", "prod"), ("maximum", "max"), ("minimum", "min")):
            for axis in (None, 0, 1, -1):
                x, n = pair(np.float64, (3, 4), "partial", dev)

                def one(uf=uf, meth=meth, axis=axis, x=x, n=n):
                    rx = getattr(np, uf).reduce(x, axis=axis)
                    assert_same(rx, getattr(n, meth)(axis=axis), dev=dev, check_fill=False, strict_nomask=False)  # see NO-SYNC note
                c.run(f"{uf} axis={axis}", one)
        c.done()

    def test_ufunc_accumulate_matches_numpy_ma(self, dev):
        c = Cases()
        for uf, meth in (("add", "cumsum"), ("multiply", "cumprod")):
            for axis in (0, 1):
                x, n = pair(np.float64, (3, 4), "partial", dev)

                def one(uf=uf, meth=meth, axis=axis, x=x, n=n):
                    rx = getattr(np, uf).accumulate(x, axis=axis)
                    assert_same(rx, getattr(n, meth)(axis=axis), dev=dev, check_fill=False, strict_nomask=False)  # see NO-SYNC note
                c.run(f"{uf} axis={axis}", one)
        c.done()

    def test_multiply_outer_matches_numpy_ma(self, dev):
        (x, n), (y, ny) = pair(np.float64, (4,), "partial", dev, 0), pair(np.float64, (3,), "partial", dev, 3)
        rx = np.multiply.outer(x, y)
        assert_same(rx, np.ma.outer(n, ny), dev=dev, check_fill=False, strict_nomask=False)  # see NO-SYNC note

    def test_reduce_never_returns_unmasked_garbage(self, dev):
        x, n = pair(np.float64, (4,), "partial", dev)
        r = np.add.reduce(x)
        assert float(r) == pytest.approx(float(n.sum()))


_AF = {
    "sum": lambda a: np.sum(a), "sum0": lambda a: np.sum(a, axis=0), "sum1": lambda a: np.sum(a, axis=1),
    "mean": lambda a: np.mean(a), "mean0": lambda a: np.mean(a, axis=0), "std": lambda a: np.std(a),
    "std1": lambda a: np.std(a, axis=1), "var": lambda a: np.var(a), "sort": lambda a: np.sort(a),
    "sort0": lambda a: np.sort(a, axis=0), "cumsum": lambda a: np.cumsum(a), "cumsum1": lambda a: np.cumsum(a, axis=1),
    "cumprod0": lambda a: np.cumprod(a, axis=0), "prod": lambda a: np.prod(a), "amax": lambda a: np.amax(a),
    "amin0": lambda a: np.amin(a, axis=0), "max": lambda a: np.max(a), "argmax": lambda a: np.argmax(a),
    "argmin1": lambda a: np.argmin(a, axis=1), "any": lambda a: np.any(a), "all0": lambda a: np.all(a, axis=0),
    "ravel": lambda a: np.ravel(a), "transpose": lambda a: np.transpose(a), "reshape": lambda a: np.reshape(a, (4, 3)),
    "squeeze": lambda a: np.squeeze(a[:1]), "shape": lambda a: np.shape(a), "ndim": lambda a: np.ndim(a),
    "size": lambda a: np.size(a), "argsort": lambda a: np.argsort(a),
    "clip": lambda a: np.clip(a, -1, 1), "round": lambda a: np.round(a, 1), "expand_dims": lambda a: np.expand_dims(a, 0),
    "swapaxes": lambda a: np.swapaxes(a, 0, 1), "count_nonzero": lambda a: np.count_nonzero(a),
    "nonzero": lambda a: np.nonzero(a), "diagonal": lambda a: np.diagonal(a), "trace": lambda a: np.trace(a),
}


class TestArrayFunction:
    def test_numpy_functions(self, dev):
        c = Cases()
        for name, f in _AF.items():
            for mk in (None, "partial", "full"):
                c.run(f"{name} {mk}", check, f, dev, pair(np.float64, (3, 4), mk, dev))
        c.done()

    def test_np_ptp_is_mask_aware(self, dev):
        # numpy.ma: `np.ptp(masked)` silently ignores the mask (it returns the raw 0-d max-min, which
        # differs from `MaskedArray.ptp()`); XuPy's `np.ptp` is mask-aware and equals `MaskedArray.ptp()`.
        c = Cases()
        for mk in (None, "partial", "full"):
            for axis in (None, 0, 1):
                x, n = pair(np.float64, (3, 4), mk, dev)
                c.run(f"ptp {mk} axis={axis}",
                      lambda x=x, n=n, axis=axis: assert_same(np.ptp(x, axis=axis), n.ptp(axis=axis), dev=dev,
                                                              strict_nomask=False))  # see NO-SYNC note
        c.done()

    def test_dtypes(self, dev):
        c = Cases()
        for name in ("sum", "mean", "cumsum", "sort"):
            for dt in (np.int32, np.float32, np.bool_, np.complex128):
                if dev == "gpu" and name == "mean" and dt is np.float32:
                    # GPU accumulation order: the float32 sum differs from numpy's pairwise sum by a few
                    # float32 ulps, far above the float64 tolerance of the float64 mean; cpu is checked.
                    continue
                c.run(f"{name} {dn(dt)}", check, _AF[name], dev, pair(dt, (3, 4), "partial", dev))
        c.done()

    def test_stacking_keeps_mask(self, dev):
        c = Cases()
        for fname in ("concatenate", "hstack", "vstack", "stack", "column_stack", "dstack"):
            for mks in ((None, None), ("partial", None), (None, "partial"), ("partial", "partial"), ("full", "false")):
                def one(fname=fname, mks=mks):
                    (x1, n1), (x2, n2) = pair(np.float64, (3, 4), mks[0], dev, 0), pair(np.float64, (3, 4), mks[1], dev, 3)
                    rx = getattr(np, fname)([x1, x2])
                    rn = getattr(np.ma, fname)([n1, n2])
                    assert_same(rx, rn, dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
                    assert isinstance(rx, XMA.MaskedArray)
                c.run(f"{fname} {mks}", one)
        c.done()

    def test_concatenate_axis_and_dtype(self, dev):
        c = Cases()
        for axis in (0, 1, -1):
            for dtb in (np.float64, np.int32, np.complex128):
                (x1, n1), (x2, n2) = pair(np.float64, (3, 4), "partial", dev, 0), pair(dtb, (3, 4), "partial", dev, 3)
                c.run(f"axis={axis} {dn(dtb)}", assert_same, np.concatenate([x1, x2], axis=axis),
                      np.ma.concatenate([n1, n2], axis=axis), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
        c.done()

    def test_concatenate_with_plain_ndarray(self, dev):
        x1, n1 = pair(np.float64, (4,), "partial", dev)
        nd = nd_pair(np.float64, (4,), dev)
        assert_same(np.concatenate([x1, nd[0]]), np.ma.concatenate([n1, nd[1]]), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
        assert_same(np.concatenate([nd[0], x1]), np.ma.concatenate([nd[1], n1]), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note

    def test_concatenate_result_independent_of_inputs(self, dev):
        (x1, n1), (x2, n2) = pair(np.float64, (4,), "partial", dev, 0), pair(np.float64, (4,), "partial", dev, 3)
        r = np.concatenate([x1, x2])
        r[0] = 123.0
        r.mask[:] = True
        assert_same(x1, n1, dev=dev)
        assert_same(x2, n2, dev=dev)

    @pytest.mark.parametrize("mk", [None, "partial"])
    def test_np_where_matches_numpy_ma(self, dev, mk):
        (x1, n1), (x2, n2) = pair(np.float64, (3, 4), mk, dev, 0), pair(np.float64, (3, 4), mk, dev, 3)
        cnd = sample(np.float64, (3, 4), 9) > 0
        rx = np.where(to_dev(cnd, dev), x1, x2)
        assert_same(rx, np.ma.where(cnd, n1, n2), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note

    def test_np_where_masked_condition_matches_numpy_ma(self, dev):
        (x1, n1), (x2, n2) = pair(np.float64, (3, 4), "partial", dev, 0), pair(np.float64, (3, 4), "partial", dev, 3)
        rx = np.where(x1 > 0, x1, x2)
        assert_same(rx, np.ma.where(n1 > 0, n1, n2), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note

    def test_xma_where_function(self, dev):
        (x1, n1), (x2, n2) = pair(np.float64, (3, 4), "partial", dev, 0), pair(np.float64, (3, 4), "partial", dev, 3)
        assert_same(XMA.where(x1 > 0, x1, x2), np.ma.where(n1 > 0, n1, n2), dev=dev, strict_nomask=False, check_fill=False)  # see NO-SYNC note
        assert_same(XMA.where(x1 > 0), np.ma.where(n1 > 0), dev=dev)

    def test_unsupported_numpy_functions_raise_typeerror(self, dev):
        c = Cases()
        fns = {
            "median": (lambda a: np.median(a), (3, 4)), "median0": (lambda a: np.median(a, axis=0), (3, 4)),
            "inv": (lambda a: np.linalg.inv(a), (3, 3)), "det": (lambda a: np.linalg.det(a), (3, 3)),
            "fft": (lambda a: np.fft.fft(a), (3, 4)), "percentile": (lambda a: np.percentile(a, 50), (3, 4)),
            "histogram": (lambda a: np.histogram(a), (3, 4)), "unique": (lambda a: np.unique(a), (3, 4)),
            "norm": (lambda a: np.linalg.norm(a), (3, 4)), "convolve": (lambda a: np.convolve(a[0], a[1]), (3, 4)),
            "cross": (lambda a: np.cross(a[:, :3], a[:, 1:]), (3, 4)),
        }
        for name, (fn, shape) in fns.items():
            for mk in (None, "partial"):
                x = pair(np.float64, shape, mk, dev)[0]

                def one(fn=fn, x=x):
                    with pytest.raises(TypeError):
                        fn(x)
                c.run(f"{name} {mk}", one)
        c.done()

    def test_np_asarray_on_xma_returns_data(self, dev):
        x, n = pair(np.float64, (3, 4), "partial", dev)
        np.testing.assert_array_equal(np.asarray(n), np.asarray(n.data))
        if dev == "cpu":
            np.testing.assert_array_equal(np.asarray(x), np.asarray(n))


class TestMixedDevices:
    def test_binary_mixed(self, gpu):
        c = Cases()
        for op in (operator.add, operator.mul, operator.truediv, operator.lt, operator.floordiv):
            for f in (op, swap(op)):
                c.run(f"{op.__name__} cpu-xma/gpu-xma", check, f, "gpu",
                      pair(np.float64, (3, 4), "partial", "cpu", 0), pair(np.float64, (3, 4), "partial", "gpu", 3))
                c.run(f"{op.__name__} cpu-xma/cupy", check, f, "gpu",
                      pair(np.float64, (3, 4), "partial", "cpu"), nd_pair(np.float64, (3, 4), "gpu"))
                c.run(f"{op.__name__} gpu-xma/numpy", check, f, "gpu",
                      pair(np.float64, (3, 4), "partial", "gpu"), host_pair(np.float64, (3, 4)))
                c.run(f"{op.__name__} gpu-xma/cupy", check, f, "gpu",
                      pair(np.float64, (3, 4), "partial", "gpu"), nd_pair(np.float64, (3, 4), "gpu"))
        c.done()

    def test_cupy_array_op_cpu_xma(self, gpu):
        cu = cp.asarray(sample(np.float64, (3, 4), 5))
        x, n = pair(np.float64, (3, 4), "partial", "cpu")
        r = cu + x
        assert on_dev(r.data, "gpu")
        assert_same(r, sample(np.float64, (3, 4), 5) + n, dev="gpu")

    def test_np_add_mixed(self, gpu):
        a, b = pair(np.float64, (3, 4), "partial", "cpu", 0), pair(np.float64, (3, 4), "partial", "gpu", 3)
        check(np.add, "gpu", a, b)
        check(np.add, "gpu", b, a)

    def test_concatenate_mixed(self, gpu):
        (x1, n1), (x2, n2) = pair(np.float64, (4,), "partial", "cpu", 0), pair(np.float64, (4,), "partial", "gpu", 3)
        r = np.concatenate([x1, x2])
        assert_same(r, np.ma.concatenate([n1, n2]), dev="gpu", strict_nomask=False, check_fill=False)  # see NO-SYNC note

    def test_inplace_mixed(self, gpu):
        x, n = pair(np.float64, (3, 4), "partial", "gpu", 0)
        y, ny = pair(np.float64, (3, 4), "partial", "cpu", 3)
        x += y
        n += ny
        assert_same(x, n, dev="gpu")

    def test_matmul_mixed(self, gpu):
        check_matmul("gpu", pair(np.float64, (2, 3), "partial", "cpu", 0), pair(np.float64, (3, 4), "partial", "gpu", 3))
        check_matmul("gpu", pair(np.float64, (3, 3), "partial", "gpu"), pair(np.float64, (3, 3), "partial", "cpu", 3))


# ---------------------------------------------------------------------------
# 6. backend follows the data
# ---------------------------------------------------------------------------
class TestBackendFromData:
    def test_gpu_array_under_cpu_backend_stays_gpu(self, gpu, restore_backend):
        x, n = pair(np.float64, (3, 4), "partial", "gpu")
        with xupy.backend("cpu"):
            assert_same(x + 1, n + 1, dev="gpu")
            assert_same(x * x, n * n, dev="gpu")
            assert_same(-x, -n, dev="gpu")
            assert_same(np.sqrt(abs(x)), np.sqrt(abs(n)), dev="gpu")
            assert_same(x.T @ x, ref_matmul(n.T, n), dev="gpu", strict_nomask=False, check_fill=False)  # see NO-SYNC note
            y, ny = pair(np.float64, (3, 4), None, "gpu", 4)
            y += x
            ny += n
            assert_same(y, ny, dev="gpu")
            assert on_dev(x.data, "gpu")

    def test_numpy_array_under_gpu_backend_stays_numpy(self, gpu, restore_backend):
        x, n = pair(np.float64, (3, 4), "partial", "cpu")
        with xupy.backend("gpu"):
            assert_same(x + 1, n + 1, dev="cpu")
            assert_same(x * x, n * n, dev="cpu")
            assert_same(np.sqrt(abs(x)), np.sqrt(abs(n)), dev="cpu")
            assert_same(x < 0.5, n < 0.5, dev="cpu")
            assert_same(x @ x.T, ref_matmul(n, n.T), dev="cpu", strict_nomask=False, check_fill=False)  # see NO-SYNC note
            y, ny = pair(np.float64, (3, 4), None, "cpu", 4)
            y += x
            ny += n
            assert_same(y, ny, dev="cpu")
            assert isinstance(x.data, np.ndarray)

    def test_numpy_ma_input_under_gpu_backend_is_numpy(self, gpu, restore_backend):
        n = np.ma.masked_array(sample(np.float64, (3,)), mask=[0, 1, 0])
        with xupy.backend("gpu"):
            r = XMA.masked_array(n)
            assert isinstance(r.data, np.ndarray)
            r2 = XMA.masked_array(n.data.copy(), mask=n.mask.copy())
            assert isinstance(r2.data, np.ndarray)

    def test_list_follows_cpu_backend(self, restore_backend):
        with xupy.backend("cpu"):
            r = XMA.masked_array([1, 2, 3])
            assert isinstance(r.data, np.ndarray)
            r = XMA.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0])
            assert isinstance(r.data, np.ndarray) and isinstance(r.mask, np.ndarray)
            assert_same(r + [1, 2, 3], np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0]) + [1, 2, 3], dev="cpu")

    def test_list_follows_gpu_backend(self, gpu, restore_backend):
        with xupy.backend("gpu"):
            r = XMA.masked_array([1, 2, 3])
            assert isinstance(r.data, cp.ndarray)
            r = XMA.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0])
            assert isinstance(r.data, cp.ndarray) and isinstance(r.mask, cp.ndarray)
            assert_same(r + [1, 2, 3], np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0]) + [1, 2, 3], dev="gpu")

    def test_scalar_ops_follow_data_not_backend(self, restore_backend, dev):
        x, n = pair(np.float64, (3,), "partial", dev)
        for target in (["cpu", "gpu"] if GPU_OK else ["cpu"]):
            with xupy.backend(target):
                assert_same(x + 1, n + 1, dev=dev)
                assert_same(2 * x, 2 * n, dev=dev)
                assert_same(x == 1.0, n == 1.0, dev=dev)

    def test_creation_in_cpu_context_does_not_move_gpu_result(self, gpu, restore_backend):
        x, n = pair(np.float64, (3, 4), "partial", "gpu")
        with xupy.backend("cpu"):
            r = XMA.masked_array([[1.0, 2.0]])
            assert isinstance(r.data, np.ndarray)
            out = x + 1
        assert on_dev(out.data, "gpu")


# ---------------------------------------------------------------------------
# 7. python protocols touching operators
# ---------------------------------------------------------------------------
class TestProtocols:
    def test_if_on_element_comparison(self, dev):
        c = Cases()
        x, n = make(np.array([1.0, 7.0, 3.0, 9.0]), [0, 1, 0, 0], dev)
        for i in (0, 1, 2, -1):
            for thr in (0, 2, 5):
                def one(i=i, thr=thr):
                    rx, rn = x[i] > thr, n[i] > thr
                    assert bool(rx) == bool(rn)
                    assert (rx is XMA.masked) == (rn is np.ma.masked)
                    if rn is not np.ma.masked:
                        assert type(rx) is type(rn)
                    assert (True if rx else False) == (True if rn else False)
                c.run(f"i={i} thr={thr}", one)
        c.done()

    def test_bool_of_array_comparison(self, dev):
        for thr in (-100, 0, 1.5, 3, 100):
            for mk in (None, "partial", "full"):
                x, n = pair(np.float64, (4,), mk, dev)
                with pytest.raises(ValueError):
                    bool(n > thr)
                with pytest.raises(ValueError):
                    bool(x > thr)

    def test_bool_matches_numpy(self, dev):
        c = Cases()
        for shape in ((), (1,), (1, 1), (0,), (2,), (2, 2)):
            for mk in (None, "full", "false", "partial"):
                x, n = pair(np.float64, shape, mk, dev, 2)

                def one(x=x, n=n):
                    try:
                        exp = bool(n)
                    except Exception as e:  # noqa: BLE001
                        with pytest.raises(type(e)):
                            bool(x)
                    else:
                        assert bool(x) == exp
                c.run(f"{shape} {mk}", one)
        c.done()

    @pytest.mark.parametrize("meth", ["any", "all"])
    def test_any_all_in_conditions(self, dev, meth):
        x, n = pair(np.float64, (4,), "partial", dev)
        assert bool(getattr(x > 0, meth)()) == bool(getattr(n > 0, meth)())

    def test_eq_returns_masked_array(self, dev):
        x, n = pair(np.float64, (4,), "partial", dev)
        y, ny = pair(np.float64, (4,), "partial", dev, 3)
        for f in (operator.eq, operator.ne):
            check(f, dev, (x, n), (y, ny))
            assert isinstance(f(x, y), XMA.MaskedArray)
        check(operator.eq, dev, (x, n), (x, n))
        check(lambda a, b: a.__eq__(b), dev, (x, n), (y, ny))
        check(lambda a, b: a.__ne__(b), dev, (x, n), (y, ny))

    def test_comparison_with_odd_objects(self, dev):
        c = Cases()
        for other in (None, "abc", object(), {}, (1, 2, 3, 4), [1, 2]):
            for op in (operator.eq, operator.ne, operator.lt):
                if dev == "gpu" and ((type(other) is str and op is operator.lt)
                                     or (type(other) in (object, dict) and op in (operator.eq, operator.ne))):
                    # cupy limitation (documented in _ops.py): text ordering comparisons and object operands
                    # raise TypeError on the GPU (numpy: UFuncTypeError / elementwise object result).
                    continue
                def one(other=other, op=op):
                    x, n = pair(np.float64, (4,), "partial", dev)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            rn = op(n, other)
                        except Exception as e:  # noqa: BLE001
                            with pytest.raises(type(e)):
                                op(x, other)
                            return
                        rx = op(x, other)
                    if isinstance(rn, np.ma.MaskedArray):
                        assert_same(rx, rn, dev=dev)
                    else:
                        assert (type(rx) is type(rn) or rx is NotImplemented or isinstance(rx, XMA.MaskedArray)
                                or (np.ndim(rx) == 0 and rx == rn)), f"{type(rx)!r} vs {type(rn)!r}"
                c.run(f"{type(other).__name__} {op.__name__}", one)
        c.done()

    def test_hash_unhashable(self, dev):
        x, n = pair(np.float64, (4,), "partial", dev)
        with pytest.raises(TypeError):
            hash(n)
        with pytest.raises(TypeError):
            hash(x)
        assert XMA.MaskedArray.__hash__ is None
        with pytest.raises(TypeError):
            {x: 1}
        with pytest.raises(TypeError):
            {x}

    def test_zero_d_hash_like_numpy(self, dev):
        x, n = pair(np.float64, (), None, dev)
        try:
            hash(n)
        except TypeError:
            with pytest.raises(TypeError):
                hash(x)

    def test_masked_constant_comparison(self, dev):
        x, n = pair(np.float64, (4,), "partial", dev)
        check(operator.eq, dev, (x, n), (XMA.masked, np.ma.masked))
        check(operator.ne, dev, (x, n), (XMA.masked, np.ma.masked))
        for i in range(4):
            assert_same(x[i] == XMA.masked, n[i] == np.ma.masked, ctx=f"elem {i}")

    def test_comparison_with_python_conditions_in_loops(self, dev):
        c = Cases()
        x, n = pair(np.int64, (6,), "partial", dev)
        for name, op in CMP.items():
            c.run(name, lambda op=op: [bool(op(x[i], 1)) for i in range(6)] == [bool(op(n[i], 1)) for i in range(6)]
                  or pytest.fail("loop results differ"))
        c.done()

    def test_xma_in_list_membership_raises_like_numpy(self, dev):
        x, n = pair(np.float64, (4,), "partial", dev)
        try:
            r = n in [n]
        except Exception as e:  # noqa: BLE001
            with pytest.raises(type(e)):
                x in [x]
        else:
            assert (x in [x]) == r

    def test_unsupported_operand_types(self, dev):
        x, _ = pair(np.float64, (4,), "partial", dev)
        with pytest.raises(TypeError):
            x + "a"
        with pytest.raises(TypeError):
            x + object()
        with pytest.raises(TypeError):
            "a" + x
        with pytest.raises(TypeError):
            x @ object()

    def test_none_operand(self, dev):
        c = Cases()
        for op in (operator.add, operator.mul, operator.truediv, operator.floordiv, operator.pow, operator.and_):
            c.run(op.__name__, check, op, dev, pair(np.float64, (4,), "partial", dev), (None, None))
            c.run(op.__name__ + " rev", check, swap(op), dev, pair(np.float64, (4,), "partial", dev), (None, None))
        c.done()
