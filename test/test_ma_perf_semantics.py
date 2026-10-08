"""
GPU-only: semantics of the fused (single-launch) fast paths of ``xupy.ma``.

The fast paths (binary ``+ - *``, domained/plain unary ufuncs, sum/mean/var/std of
masked float64 data, ``where``) must agree with ``numpy.ma``: data under the mask
(binary: left operand's data), mask values, dtype, mask ownership.
"""
import sys

import numpy as np
import pytest

import xupy
from xupy import _core

cp = _core._cupy
if cp is None:
    pytest.skip("no usable GPU", allow_module_level=True)

M = sys.modules["xupy.ma"]
SHAPE = (37, 53)


def _host(rng, dtype, neg=True):
    d = rng.standard_normal(SHAPE) * 3
    return (d if neg else np.abs(d) + 0.1).astype(dtype), rng.random(SHAPE) < 0.3


def _same(x, ref, rtol=0.0):
    xd, xm = cp.asnumpy(x._data), np.ma.getmaskarray(ref)
    assert x.dtype == ref.dtype
    assert np.array_equal(cp.asnumpy(M.getmaskarray(x)), xm)
    keep = ~xm
    np.testing.assert_allclose(xd[keep], np.ma.getdata(ref)[keep], rtol=rtol, atol=0)


@pytest.fixture(autouse=True)
def _gpu():
    with xupy.backend("gpu"):
        yield


@pytest.mark.parametrize("dt", ["f8", "f4", "i4", "i8", "u2"])
@pytest.mark.parametrize("ma,mb", [(1, 1), (1, 0), (0, 1)])
@pytest.mark.parametrize("op", ["__add__", "__sub__", "__mul__"])
def test_binary_matches_numpy_ma(dt, ma, mb, op):
    rng = np.random.default_rng(0)
    (da, ka), (db, kb) = _host(rng, dt), _host(rng, dt)
    ha = np.ma.masked_array(da, mask=ka if ma else False)
    hb = np.ma.masked_array(db, mask=kb if mb else False)
    xa = M.masked_array(da, mask=ka if ma else M.nomask)
    xb = M.masked_array(db, mask=kb if mb else M.nomask)
    res, ref = getattr(xa, op)(xb), getattr(ha, op)(hb)
    _same(res, ref)
    # data under the mask is the left operand's, as numpy.ma
    mk = cp.asnumpy(res._mask)
    assert np.array_equal(cp.asnumpy(res._data)[mk], da[mk])


def test_binary_mixed_dtypes_and_scalars_fall_back():
    rng = np.random.default_rng(1)
    (da, ka) = _host(rng, "f4")
    xa = M.masked_array(da, mask=ka)
    ha = np.ma.masked_array(da, mask=ka)
    for other in (2, 2.5, np.float64(2.0), np.int8(3)):
        _same(xa + other, ha + other)
        _same(other * xa, other * ha)
    _same(xa + M.masked_array(da.astype("f8"), mask=ka), ha + np.ma.masked_array(da.astype("f8"), mask=ka))


def test_binary_broadcast_and_new_mask():
    rng = np.random.default_rng(2)
    (da, ka), (db, kb) = _host(rng, "f8"), _host(rng, "f8")
    xa = M.masked_array(da, mask=ka)
    xb = M.masked_array(db[:1], mask=kb[:1])
    _same(xa + xb, np.ma.masked_array(da, mask=ka) + np.ma.masked_array(db[:1], mask=kb[:1]))
    r = xa + M.masked_array(db)
    assert r._mask is not xa._mask
    r._mask[...] = True  # must not alias the operand's mask
    assert not bool(xa._mask.all())


def test_binary_nomask_stays_nomask():
    xa, xb = M.masked_array(np.ones(SHAPE)), M.masked_array(np.ones(SHAPE))
    assert (xa + xb)._mask is M.nomask


@pytest.mark.parametrize("name", ["sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tanh", "arctan"])
@pytest.mark.parametrize("dt", ["f8", "f4"])
@pytest.mark.parametrize("masked_in", [True, False])
def test_unary_matches_numpy_ma(name, dt, masked_in):
    rng = np.random.default_rng(3)
    d, k = _host(rng, dt)
    d[0, :3] = [0.0, np.inf, np.nan]
    h = np.ma.masked_array(d, mask=k if masked_in else False)
    x = M.masked_array(d, mask=k if masked_in else M.nomask)
    with np.errstate(all="ignore"):
        ref = getattr(np.ma, name)(h)
    res = getattr(M, name)(x)
    rtol = 1e-5 if dt == "f4" else 1e-13
    _same(res, ref, rtol)
    if name in ("exp", "sin", "cos", "tanh", "arctan") and not masked_in:
        assert res._mask is M.nomask  # non-domained, unmasked: nomask is kept


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0,), (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_reductions_match_numpy_ma(axis, keepdims):
    rng = np.random.default_rng(4)
    d, k = _host(rng, "f8")
    k[:, 5] = True  # one fully masked column
    h, x = np.ma.masked_array(d, mask=k), M.masked_array(d, mask=k)
    for name, kw in [("sum", {}), ("mean", {}), ("var", {}), ("std", {}), ("std", {"ddof": 2})]:
        ref = getattr(h, name)(axis=axis, keepdims=keepdims, **kw)
        res = getattr(x, name)(axis=axis, keepdims=keepdims, **kw)
        if isinstance(res, M.MaskedArray):
            _same(res, ref, 1e-12)
        else:
            np.testing.assert_allclose(res, ref, rtol=1e-12)


def test_reductions_all_masked_and_scalar():
    d = np.arange(12.0).reshape(3, 4)
    x = M.masked_array(d, mask=np.ones((3, 4), bool))
    for name in ("sum", "mean", "var", "std"):
        assert getattr(x, name)() is M.masked
    x = M.masked_array(d, mask=np.eye(3, 4, dtype=bool))
    assert float(x.mean()) == pytest.approx(np.ma.masked_array(d, mask=np.eye(3, 4, dtype=bool)).mean())


def test_std_ddof_exceeds_count_is_masked():
    d = np.arange(8.0).reshape(2, 4)
    k = np.array([[1, 1, 1, 0], [0, 0, 0, 0]], bool)
    res = M.masked_array(d, mask=k).std(axis=1, ddof=1)
    ref = np.ma.masked_array(d, mask=k).std(axis=1, ddof=1)
    _same(res, ref, 1e-12)


def test_reductions_float32_and_int_unchanged():
    rng = np.random.default_rng(5)
    for dt in ("f4", "i4"):
        d, k = _host(rng, dt)
        h, x = np.ma.masked_array(d, mask=k), M.masked_array(d, mask=k)
        for name in ("sum", "mean"):
            np.testing.assert_allclose(getattr(x, name)(axis=0)._data.get(), getattr(h, name)(axis=0).data, rtol=1e-5, atol=1e-4)


def test_where_matches_numpy_ma():
    rng = np.random.default_rng(6)
    (da, ka), (db, kb) = _host(rng, "f8"), _host(rng, "f8")
    cond = rng.random(SHAPE) < 0.5
    ha, hb = np.ma.masked_array(da, mask=ka), np.ma.masked_array(db, mask=kb)
    xa, xb = M.masked_array(da, mask=ka), M.masked_array(db, mask=kb)
    _same(M.where(cp.asarray(cond), xa, xb), np.ma.where(cond, ha, hb))
    cm = rng.random(SHAPE) < 0.2
    _same(M.where(M.masked_array(cond, mask=cm), xa, xb), np.ma.where(np.ma.masked_array(cond, mask=cm), ha, hb))
    _same(M.where(cp.asarray(cond), xa, 3.0), np.ma.where(cond, ha, 3.0))
