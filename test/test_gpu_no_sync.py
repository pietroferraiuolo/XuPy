"""
GPU-only: ``xupy.ma`` operations must not synchronize with the host.

Each op runs once as warm-up (kernel compilation may sync), then again under
``cupyx.allow_synchronize(False)``.  Ops in ``NO_SYNC`` must not raise
``cupyx.DeviceSynchronized``; ops in ``KNOWN_SYNC`` (scalar results, repr,
data-dependent output size, ...) are asserted to DO raise it, so the test
notices when that changes.
"""
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import xupy
from xupy import _core

cp = _core._cupy
if cp is None:
    pytest.skip("no usable GPU", allow_module_level=True)

import cupyx  # noqa: E402

M = sys.modules["xupy.ma"]
E = sys.modules["xupy.ma.extras"]
N = 512


@pytest.fixture(scope="module")
def S():
    with xupy.backend("gpu"):
        rng = np.random.default_rng(0)
        a = M.masked_array(rng.random((N, N)), mask=rng.random((N, N)) < 0.1)
        b = M.masked_array(rng.random((N, N)) + 1, mask=rng.random((N, N)) < 0.1)
        u = M.masked_array(rng.random((N, N)))
        v = M.masked_array(rng.random(N), mask=rng.random(N) < 0.1)
        w = M.masked_array(rng.random(8), mask=rng.random(8) < 0.1)
        idx = cp.asarray(rng.integers(0, N * N, 50))
        idx1 = cp.asarray(rng.integers(0, N, 50))
        bad = M.masked_array(cp.where(cp.asarray(rng.random((N, N)) < 0.01),
                                      cp.nan, cp.asarray(rng.random((N, N)))))
        s = SimpleNamespace(
            a=a, b=b, u=u, v=v, w=w, idx=idx, idx1=idx1, bad=bad,
            cond=a.data > 0.5, c1=M.masked_array(rng.random((N, 4)), mask=rng.random((N, 4)) < 0.1),
            ints=M.masked_array(cp.asarray(rng.integers(0, 5, (N, N))),
                                mask=cp.asarray(rng.random((N, N)) < 0.1)),
        )
        cp.cuda.Device().synchronize()
        yield s


# (id, fn(S)) -- must not synchronize
NO_SYNC = {
    # elementwise
    "a+b": lambda S: S.a + S.b,
    "a-3": lambda S: S.a - 3,
    "2*a": lambda S: 2 * S.a,
    "a/b": lambda S: S.a / S.b,
    "a//b": lambda S: S.a // S.b,
    "a%b": lambda S: S.a % S.b,
    "a**2": lambda S: S.a ** 2,
    "a>b": lambda S: S.a > S.b,
    "a<=b": lambda S: S.a <= S.b,
    "a==b": lambda S: S.a == S.b,
    "a!=b": lambda S: S.a != S.b,
    "(a>.2)&(b>1.5)": lambda S: (S.a > 0.2) & (S.b > 1.5),
    "-a": lambda S: -S.a,
    "abs(a)": lambda S: abs(S.a),
    "u+1 nomask": lambda S: S.u + 1,
    # ufuncs
    "np.add": lambda S: np.add(S.a, S.b),
    "np.sqrt": lambda S: np.sqrt(S.a),
    "np.log": lambda S: np.log(S.a),
    "np.exp": lambda S: np.exp(S.a),
    "ma.sqrt": lambda S: M.sqrt(S.a),
    "ma.log": lambda S: M.log(S.a),
    "ma.exp": lambda S: M.exp(S.a),
    "ma.sin": lambda S: M.sin(S.a),
    "ma.power": lambda S: M.power(S.a, 2.5),
    "ma.divide": lambda S: M.divide(S.a, S.b),
    "ma.arctan2": lambda S: M.arctan2(S.a, S.b),
    # in-place
    "a+=b": lambda S: S.a.copy().__iadd__(S.b),
    "a*=2": lambda S: S.a.copy().__imul__(2),
    "a/=b": lambda S: S.a.copy().__itruediv__(S.b),
    "a-=u": lambda S: S.a.copy().__isub__(S.u),
    "a@b": lambda S: S.a @ S.b,
    # axis reductions
    **{f"{n}(axis={ax})": (lambda n, ax: lambda S: getattr(S.a, n)(axis=ax))(n, ax)
       for n in ("sum", "mean", "std", "var", "min", "max", "prod", "cumsum", "cumprod",
                 "argmax", "argmin", "any", "all", "count", "ptp")
       for ax in (0, 1)},
    "var(ddof=1)": lambda S: S.a.var(axis=0, ddof=1),
    "ma.sum(axis)": lambda S: M.sum(S.a, axis=0),
    # sort & co
    "sort": lambda S: S.a.copy().sort(axis=0),
    "argsort": lambda S: S.a.argsort(axis=0),
    "ma.sort": lambda S: M.sort(S.a, axis=1),
    "clip": lambda S: S.a.clip(0.2, 0.8),
    "filled": lambda S: S.a.filled(0),
    "astype": lambda S: S.a.astype(np.float32),
    "slice": lambda S: S.a[::2, 1:],
    "a[0]=masked": lambda S: S.a.copy().__setitem__(0, M.masked),
    "a[1:3]=0": lambda S: S.a.copy().__setitem__(slice(1, 3), 0),
    "where": lambda S: M.where(S.cond, S.a, S.b),
    "concatenate": lambda S: M.concatenate([S.a, S.b]),
    "average(axis)": lambda S: M.average(S.a, axis=0),
    "ma.average(axis,w)": lambda S: M.average(S.a, axis=1, weights=S.v),
    "copy": lambda S: S.a.copy(),
    "zeros_like": lambda S: M.zeros_like(S.a),
    "ones_like": lambda S: M.ones_like(S.a),
    "empty_like": lambda S: M.empty_like(S.a),
    "reshape": lambda S: S.a.reshape(-1),
    "transpose": lambda S: S.a.T,
    "ravel": lambda S: S.a.ravel(),
    "swapaxes": lambda S: S.a.swapaxes(0, 1),
    "squeeze": lambda S: S.a[None].squeeze(),
    "trace": lambda S: S.a.reshape(8, 8, -1).trace(axis1=0, axis2=1),
    # phase 4
    "masked_where": lambda S: M.masked_where(S.cond, S.a, copy=False),
    "masked_greater": lambda S: M.masked_greater(S.a, 0.5),
    "masked_less": lambda S: M.masked_less(S.a, 0.5),
    "masked_equal": lambda S: M.masked_equal(S.ints, 2),
    "masked_inside": lambda S: M.masked_inside(S.a, 0.3, 0.6),
    "masked_outside": lambda S: M.masked_outside(S.a, 0.3, 0.6),
    "masked_invalid": lambda S: M.masked_invalid(S.bad),
    "fix_invalid": lambda S: M.fix_invalid(S.bad),
    "masked_values noshrink": lambda S: M.masked_values(S.ints, 2, shrink=False),
    "diff": lambda S: M.diff(S.a, axis=0),
    "convolve": lambda S: M.convolve(S.v, S.w),
    "correlate": lambda S: M.correlate(S.v, S.w),
    "putmask": lambda S: M.putmask(S.a.copy(), S.cond, 0.5),
    "maximum": lambda S: M.maximum(S.a, S.b),
    "minimum": lambda S: M.minimum(S.a, S.b),
    "maximum.reduce": lambda S: M.maximum.reduce(S.a, axis=0),
    "minimum.reduce": lambda S: M.minimum.reduce(S.a, axis=0),
    "outer": lambda S: M.outer(S.v, S.v),
    "inner": lambda S: M.inner(S.a, S.b),
    "outerproduct": lambda S: M.outerproduct(S.v, S.v),
    "arange": lambda S: M.arange(N),
    "ones": lambda S: M.ones((N, N)),
    "zeros": lambda S: M.zeros((N, N)),
    "empty": lambda S: M.empty((N, N)),
    "identity": lambda S: M.identity(N),
    "indices": lambda S: M.indices((N, N)),
    "fromfunction": lambda S: M.fromfunction(lambda i, j: i + j, (N, N)),
    "take wrap": lambda S: M.take(S.a, S.idx, mode="wrap"),
    "take clip": lambda S: M.take(S.a, S.idx1, axis=0, mode="clip"),
    "put wrap": lambda S: M.put(S.a.copy(), S.idx, 0.5, mode="wrap"),
    "put clip": lambda S: M.put(S.a.copy(), S.idx, 0.5, mode="clip"),
    "choose wrap": lambda S: M.choose(S.ints, [S.a, S.b], mode="wrap"),
    "choose clip": lambda S: M.choose(S.ints, [S.a, S.b, S.u, S.a, S.b], mode="clip"),
    "cov": lambda S: E.cov(S.c1, rowvar=False),
    "corrcoef": lambda S: E.corrcoef(S.c1, rowvar=False),
    "vander": lambda S: E.vander(S.v, 4),
    "extras.std": lambda S: E.std(S.a, axis=0) if hasattr(E, "std") else M.std(S.a, axis=0),
    "extras.var": lambda S: E.var(S.a, axis=0) if hasattr(E, "var") else M.var(S.a, axis=0),
    "extras.anom": lambda S: S.a.anom(axis=0),
    "apply_over_axes": lambda S: E.apply_over_axes(M.sum, S.a, [0]),
    "in1d assume_unique": lambda S: E.in1d(S.v, S.w, assume_unique=True) if hasattr(E, "in1d") else E.isin(S.v, S.w, assume_unique=True),
    "isin assume_unique": lambda S: E.isin(S.v, S.w, assume_unique=True),
    "median(axis)": lambda S: E.median(S.a, axis=0),
    "byteswap": lambda S: S.a.byteswap(),
    "ids": lambda S: S.a.ids(),
    "iscontiguous": lambda S: S.a.iscontiguous(),
    # module-level function forms
    "diag": lambda S: M.diag(S.a),
    "diag 1d": lambda S: M.diag(S.v),
    "resize": lambda S: M.resize(S.a, (N, 2 * N)),
    "append": lambda S: M.append(S.a, S.b, axis=0),
    "alltrue(axis)": lambda S: M.alltrue(S.a, axis=0),
    "sometrue(axis)": lambda S: M.sometrue(S.a, axis=0),
    "asarray": lambda S: M.asarray(S.a),
    "left_shift": lambda S: M.left_shift(S.ints, 2),
    "right_shift": lambda S: M.right_shift(S.ints, 1),
    "ma.ptp(axis)": lambda S: M.ptp(S.a, axis=0),
    "ma.count(axis)": lambda S: M.count(S.a, axis=0),
    "masked_where cupy cond, host data": lambda S: M.masked_where(S.cond, np.zeros((N, N))),
}

# ops that unavoidably synchronize (scalar result, repr, data-dependent size, ...)
KNOWN_SYNC = {
    "trace scalar": lambda S: S.a.trace(),
    "a.sum()": lambda S: S.a.sum(),
    "a.mean()": lambda S: S.a.mean(),
    "a.std()": lambda S: S.a.std(),
    "a.min()": lambda S: S.a.min(),
    "a.max()": lambda S: S.a.max(),
    "a.count()": lambda S: S.a.count(),
    "float(a[0,0])": lambda S: float(S.a[0, 0]),
    "a[0,0]": lambda S: S.a[0, 0],
    "repr": lambda S: repr(S.a),
    "str": lambda S: str(S.a),
    "a[boolmask]=0": lambda S: S.a.copy().__setitem__(S.cond, 0),
    "compressed": lambda S: S.a.compressed(),
    "nonzero": lambda S: S.a.nonzero(),
    "unique": lambda S: E.unique(S.ints),
    "intersect1d": lambda S: E.intersect1d(S.ints, S.ints),
    "union1d": lambda S: E.union1d(S.v, S.w),
    "setdiff1d": lambda S: E.setdiff1d(S.v, S.w),
    "in1d": lambda S: E.in1d(S.v, S.w),
    "clump_masked": lambda S: E.clump_masked(S.v),
    "clump_unmasked": lambda S: E.clump_unmasked(S.v),
    "notmasked_edges": lambda S: E.notmasked_edges(S.a, axis=0),
    "notmasked_contiguous": lambda S: E.notmasked_contiguous(S.v),
    "median()": lambda S: E.median(S.a),
    "take raise": lambda S: M.take(S.a, S.idx, mode="raise"),
    "put raise": lambda S: M.put(S.a.copy(), S.idx, 0.5, mode="raise"),
    "allclose": lambda S: M.allclose(S.a, S.a),
    "allequal": lambda S: M.allequal(S.a, S.a),
    "masked_values shrink": lambda S: M.masked_values(S.ints, 2, shrink=True),
}


def _run(fn, S):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn(S)  # warm-up outside the no-sync region
        cp.cuda.Device().synchronize()
        with cupyx.allow_synchronize(False):
            fn(S)


@pytest.mark.parametrize("name", list(NO_SYNC))
def test_no_sync(S, name):
    _run(NO_SYNC[name], S)


@pytest.mark.parametrize("name", list(KNOWN_SYNC))
def test_known_sync(S, name):
    with pytest.raises(cupyx.DeviceSynchronized):
        _run(KNOWN_SYNC[name], S)
