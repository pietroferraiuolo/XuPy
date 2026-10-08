"""Tests for the GPU-mode NumPy 2 shims (xupy._shims). GPU only."""
import inspect
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import xupy as xp
from xupy import _core

pytestmark = pytest.mark.skipif(not _core._GPU_AVAILABLE, reason="CuPy is not usable")

HAS_VECMAT = hasattr(np, "matvec") and hasattr(np, "vecmat")
_sort_params = inspect.signature(np.sort).parameters
NP_STABLE = "stable" in _sort_params
NP_DESC = "descending" in _sort_params


@pytest.fixture(autouse=True)
def _gpu():
    orig = _core._global_gpu
    with xp.backend("gpu"):
        yield
    if orig:
        xp.use_gpu()
    else:
        xp.use_cpu()


def _h(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _g(a):
    return xp.asarray(a)


rng = np.random.default_rng(1234)


def _rand(*shape, complex_=False):
    a = rng.standard_normal(shape)
    if complex_:
        a = a + 1j * rng.standard_normal(shape)
    return a


class TestVecdot:
    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float32, np.bool_])
    def test_result_dtype_matches_numpy(self, dtype):
        a = (np.arange(12).reshape(3, 4) % 3).astype(dtype)
        b = (np.arange(12).reshape(3, 4) % 2).astype(dtype)
        got = _h(xp.vecdot(_g(a), _g(b)))
        want = np.vecdot(a, b)
        assert got.dtype == want.dtype
        assert_array_equal(got, want)

    def test_real(self):
        a, b = _rand(5, 7), _rand(5, 7)
        assert_allclose(_h(xp.vecdot(_g(a), _g(b))), np.vecdot(a, b))

    def test_complex_conjugates_first_arg(self):
        a, b = _rand(4, 6, complex_=True), _rand(4, 6, complex_=True)
        got = _h(xp.vecdot(_g(a), _g(b)))
        assert_allclose(got, np.vecdot(a, b))
        assert_allclose(got, np.sum(np.conj(a) * b, axis=-1))
        # swapping the arguments gives the conjugate, not the same value
        assert_allclose(_h(xp.vecdot(_g(b), _g(a))), np.conj(got))
        assert not np.allclose(got, np.sum(a * b, axis=-1))

    def test_broadcast(self):
        a, b = _rand(3, 1, 5), _rand(4, 5)
        got = _h(xp.vecdot(_g(a), _g(b)))
        assert got.shape == (3, 4)
        assert_allclose(got, np.vecdot(a, b))

    def test_axis(self):
        a, b = _rand(4, 5, 6), _rand(4, 5, 6)
        for ax in (0, 1, 2, -2):
            assert_allclose(_h(xp.vecdot(_g(a), _g(b), axis=ax)), np.vecdot(a, b, axis=ax))

    def test_1d_scalar_result(self):
        a, b = _rand(9), _rand(9)
        assert_allclose(_h(xp.vecdot(_g(a), _g(b))), np.vecdot(a, b))

    def test_int_inputs(self):
        a = np.arange(6).reshape(2, 3)
        assert_array_equal(_h(xp.vecdot(_g(a), _g(a))), np.vecdot(a, a))

    def test_accepts_lists(self):
        assert_allclose(_h(xp.vecdot([1.0, 2.0], [3.0, 4.0])), 11.0)

    def test_bad_axis(self):
        with pytest.raises(Exception):
            xp.vecdot(_g(_rand(3)), _g(_rand(3)), axis=3)

    def test_non_broadcastable(self):
        with pytest.raises(ValueError):
            xp.vecdot(_g(_rand(3, 4)), _g(_rand(5, 4)))

    def test_empty(self):
        a = np.zeros((3, 0))
        assert_array_equal(_h(xp.vecdot(_g(a), _g(a))), np.vecdot(a, a))


@pytest.mark.skipif(not HAS_VECMAT, reason="numpy has no matvec/vecmat")
class TestMatvecVecmat:
    def test_matvec(self):
        m, v = _rand(3, 4), _rand(4)
        assert_allclose(_h(xp.matvec(_g(m), _g(v))), np.matvec(m, v))

    def test_matvec_stacked(self):
        m, v = _rand(2, 5, 3, 4), _rand(5, 4)
        got = _h(xp.matvec(_g(m), _g(v)))
        assert got.shape == (2, 5, 3)
        assert_allclose(got, np.matvec(m, v))

    def test_vecmat(self):
        v, m = _rand(3), _rand(3, 4)
        assert_allclose(_h(xp.vecmat(_g(v), _g(m))), np.vecmat(v, m))

    def test_vecmat_complex_conj(self):
        v, m = _rand(3, complex_=True), _rand(2, 3, 4, complex_=True)
        assert_allclose(_h(xp.vecmat(_g(v), _g(m))), np.vecmat(v, m))

    def test_matvec_complex(self):
        m, v = _rand(3, 4, complex_=True), _rand(4, complex_=True)
        assert_allclose(_h(xp.matvec(_g(m), _g(v))), np.matvec(m, v))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            xp.matvec(_g(_rand(3, 4)), _g(_rand(5)))
        with pytest.raises(ValueError):
            xp.vecmat(_g(_rand(5)), _g(_rand(3, 4)))


class TestUnstack:
    def test_axis0(self):
        a = np.arange(24).reshape(2, 3, 4)
        got = xp.unstack(_g(a))
        assert isinstance(got, tuple) and len(got) == 2
        for g, e in zip(got, a):
            assert_array_equal(_h(g), e)

    def test_axis_last(self):
        a = np.arange(24).reshape(2, 3, 4)
        got = xp.unstack(_g(a), axis=-1)
        assert len(got) == 4
        for i, g in enumerate(got):
            assert_array_equal(_h(g), a[..., i])

    def test_middle_axis(self):
        a = np.arange(24).reshape(2, 3, 4)
        got = xp.unstack(_g(a), axis=1)
        assert len(got) == 3
        assert_array_equal(_h(got[2]), a[:, 2, :])

    def test_1d(self):
        got = xp.unstack(_g(np.arange(3)))
        assert len(got) == 3 and all(g.ndim == 0 for g in got)

    def test_empty_axis(self):
        assert xp.unstack(_g(np.zeros((0, 3)))) == ()

    def test_bad_axis(self):
        with pytest.raises(Exception):
            xp.unstack(_g(np.zeros((2, 2))), axis=2)

    def test_0d_raises(self):
        with pytest.raises(Exception):
            xp.unstack(_g(np.float64(1.0)))


def _np_sort(a, axis=-1, descending=False):
    """Stable reference sort with NaNs last, ties in original order."""
    if axis is None:
        a, axis = a.ravel(), -1
    if not descending:
        return np.sort(a, axis=axis, kind="stable")
    key = -a if a.dtype.kind == "f" else ~a
    idx = np.argsort(key, axis=axis, kind="stable")
    return np.take_along_axis(a, idx, axis=axis), idx


DTYPES = [np.int8, np.int32, np.int64, np.uint16, np.float32, np.float64]


class TestSort:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("axis", [None, 0, 1, -1])
    def test_stable(self, dtype, axis):
        a = rng.integers(0, 5, size=(6, 7)).astype(dtype)  # lots of ties
        assert_array_equal(_h(xp.sort(_g(a), axis=axis, stable=True)),
                           np.sort(a, axis=axis, kind="stable"))
        assert_array_equal(_h(xp.argsort(_g(a), axis=axis, stable=True)),
                           np.argsort(a, axis=axis, kind="stable"))

    def test_default_matches_numpy(self):
        a = _rand(5, 6)
        assert_array_equal(_h(xp.sort(_g(a))), np.sort(a))
        assert_array_equal(_h(xp.argsort(_g(a))), np.argsort(a))

    def test_nan_last(self):
        a = np.array([3.0, np.nan, 1.0, np.nan, 2.0])
        assert_array_equal(_h(xp.sort(_g(a), stable=True)), np.sort(a))
        assert_array_equal(_h(xp.argsort(_g(a), stable=True)), np.argsort(a, kind="stable"))

    def test_kind_still_accepted(self):
        a = _rand(10)
        assert_array_equal(_h(xp.sort(_g(a), kind="stable")), np.sort(a))

    def test_order_not_supported(self):
        with pytest.raises(NotImplementedError):
            xp.sort(_g(_rand(3)), order="x")
        with pytest.raises(NotImplementedError):
            xp.argsort(_g(_rand(3)), order="x")

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_descending(self, dtype, axis):
        a = rng.integers(0, 5, size=(6, 7)).astype(dtype)
        vals, idx = _np_sort(a, axis, descending=True)
        assert_array_equal(_h(xp.sort(_g(a), axis=axis, descending=True)), vals)
        assert_array_equal(_h(xp.argsort(_g(a), axis=axis, descending=True)), idx)

    def test_descending_values_are_descending(self):
        a = _rand(50)
        got = _h(xp.sort(_g(a), descending=True))
        assert_array_equal(got, np.sort(a)[::-1])

    def test_descending_nan(self):
        a = np.array([3.0, np.nan, 1.0, np.nan, 2.0])
        vals, idx = _np_sort(a, -1, descending=True)
        got = _h(xp.sort(_g(a), descending=True))
        assert_array_equal(got, vals)  # NaN stays last, equal_nan compare
        assert_array_equal(_h(xp.argsort(_g(a), descending=True)), idx)

    def test_descending_unsigned_and_extremes(self):
        a = np.array([0, 255, 7, 255, 0], dtype=np.uint8)
        vals, idx = _np_sort(a, -1, descending=True)
        assert_array_equal(_h(xp.sort(_g(a), descending=True)), vals)
        assert_array_equal(_h(xp.argsort(_g(a), descending=True)), idx)
        b = np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype=np.int64)
        assert_array_equal(_h(xp.sort(_g(b), descending=True)), np.sort(b)[::-1])

    def test_descending_float_inf_and_negzero(self):
        a = np.array([np.inf, -np.inf, 0.0, -0.0, 1.0])
        got = _h(xp.sort(_g(a), descending=True))
        assert_array_equal(got, np.array([np.inf, 1.0, 0.0, -0.0, -np.inf]))

    def test_descending_bool(self):
        a = np.array([True, False, True, False])
        got = _h(xp.sort(_g(a), descending=True))
        assert_array_equal(got, np.array([True, True, False, False]))

    def test_descending_complex(self):
        a = np.array([1 + 1j, 2 + 0j, 1 + 0j, 0 + 5j])
        got = _h(xp.sort(_g(a), descending=True))
        assert_array_equal(got, np.sort(a)[::-1])

    @pytest.mark.skipif(not NP_DESC, reason="numpy has no descending=")
    def test_matches_numpy_descending(self):
        a = rng.integers(0, 4, size=(5, 6)).astype(np.float64)
        assert_array_equal(_h(xp.sort(_g(a), descending=True, axis=1)),
                           np.sort(a, descending=True, axis=1))
        assert_array_equal(_h(xp.argsort(_g(a), descending=True, axis=1)),
                           np.argsort(a, descending=True, axis=1))

    def test_bad_axis(self):
        with pytest.raises(Exception):
            xp.sort(_g(_rand(3, 3)), axis=2)
        with pytest.raises(Exception):
            xp.sort(_g(_rand(3, 3)), axis=2, descending=True)

    def test_empty(self):
        assert xp.sort(_g(np.array([])), descending=True).shape == (0,)

    def test_input_not_modified(self):
        a = _g(np.array([3, 1, 2]))
        xp.sort(a, descending=True)
        assert_array_equal(_h(a), [3, 1, 2])


class TestUnique:
    def test_default(self):
        a = np.array([3, 1, 2, 3, 1])
        assert_array_equal(_h(xp.unique(_g(a))), np.unique(a))

    def test_sorted_false_as_set(self):
        a = np.array([3, 1, 2, 3, 1, 9])
        got = _h(xp.unique(_g(a), sorted=False))
        assert set(got.tolist()) == set(a.tolist())
        assert len(got) == len(set(a.tolist()))

    def test_sorted_false_with_returns(self):
        a = np.array([5, 5, 2, 7, 2])
        u, idx, inv, cnt = xp.unique(_g(a), True, True, True, sorted=False)
        u, idx, inv, cnt = map(_h, (u, idx, inv, cnt))
        assert_array_equal(u[inv], a)
        assert_array_equal(a[idx], u)
        assert dict(zip(u.tolist(), cnt.tolist())) == {5: 2, 2: 2, 7: 1}

    def test_equal_nan_true(self):
        a = np.array([1.0, np.nan, np.nan, 2.0])
        got = _h(xp.unique(_g(a), equal_nan=True))
        assert_array_equal(got, np.unique(a, equal_nan=True))
        assert len(got) == 3

    def test_equal_nan_false(self):
        a = np.array([1.0, np.nan, np.nan, 2.0])
        got = _h(xp.unique(_g(a), equal_nan=False))
        assert_array_equal(got, np.unique(a, equal_nan=False))
        assert len(got) == 4

    def test_axis(self):
        a = np.array([[1, 2], [1, 2], [3, 4]])
        assert_array_equal(_h(xp.unique(_g(a), axis=0)), np.unique(a, axis=0))

    def test_empty(self):
        assert _h(xp.unique(_g(np.array([], dtype=np.int64)))).shape == (0,)


class TestErrstate:
    def test_ignore_context(self):
        with xp.errstate(divide="ignore"):
            r = xp.ones(2) / xp.zeros(2)
        assert np.all(np.isinf(_h(r)))

    def test_warn_all(self):
        with xp.errstate(all="warn"):
            xp.sum(xp.ones(2))

    def test_all_ignore_and_none(self):
        with xp.errstate(all="ignore"):
            pass
        with xp.errstate():
            pass

    @pytest.mark.parametrize("mode", ["raise", "call", "print", "log"])
    def test_unsupported_modes(self, mode):
        with pytest.raises(NotImplementedError):
            with xp.errstate(divide=mode):
                pass
        with pytest.raises(NotImplementedError):
            with xp.errstate(all=mode):
                pass
        with pytest.raises(NotImplementedError):
            xp.seterr(over=mode)

    def test_call_kw_unsupported(self):
        with pytest.raises(NotImplementedError):
            with xp.errstate(call=lambda *a: None):
                pass

    def test_unsupported_leaves_state_intact(self):
        before = xp.geterr()
        with pytest.raises(NotImplementedError):
            xp.seterr(divide="raise", invalid="ignore")
        assert xp.geterr() == before

    def test_geterr_dict(self):
        e = xp.geterr()
        assert isinstance(e, dict)
        assert all(v is not None for v in e.values())

    def test_seterr_returns_old(self):
        old = xp.seterr(all="ignore")
        assert isinstance(old, dict)

    def test_exception_in_block_propagates(self):
        before = xp.geterr()
        with pytest.raises(KeyError):
            with xp.errstate(divide="ignore"):
                raise KeyError("x")
        assert xp.geterr() == before

    def test_positional_args_rejected(self):
        with pytest.raises(TypeError):
            xp.errstate("ignore")

    def test_no_numpy_warnings_leak(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with xp.errstate(divide="ignore", invalid="ignore"):
                xp.zeros(2) / xp.zeros(2)


def _spd_batch(*lead, n=4):
    return _rand(*lead, n, n)


class TestLinalgProxy:
    def test_matrix_norm_fro(self):
        a = _rand(4, 5)
        assert_allclose(_h(xp.linalg.matrix_norm(_g(a))), np.linalg.matrix_norm(a))

    @pytest.mark.parametrize("ord_", ["fro", "nuc", 1, -1, 2, -2, np.inf, -np.inf])
    def test_matrix_norm_ord(self, ord_):
        a = _rand(2, 3, 4, 5)
        assert_allclose(_h(xp.linalg.matrix_norm(_g(a), ord=ord_)),
                        np.linalg.matrix_norm(a, ord=ord_), rtol=1e-10)

    def test_matrix_norm_keepdims(self):
        a = _rand(2, 4, 5)
        got = _h(xp.linalg.matrix_norm(_g(a), keepdims=True))
        assert got.shape == (2, 1, 1)
        assert_allclose(got, np.linalg.matrix_norm(a, keepdims=True))

    def test_matrix_norm_1d_raises(self):
        with pytest.raises(Exception):
            xp.linalg.matrix_norm(_g(_rand(5)))

    @pytest.mark.parametrize("ord_", [1, 2, np.inf, -np.inf, 0, 3, 0.5, None])
    def test_vector_norm_axis_none(self, ord_):
        a = _rand(3, 4, 5)
        assert_allclose(_h(xp.linalg.vector_norm(_g(a), ord=ord_)),
                        np.linalg.vector_norm(a, ord=ord_ if ord_ is not None else 2),
                        rtol=1e-10)

    @pytest.mark.parametrize("ord_", [1, 2, np.inf])
    @pytest.mark.parametrize("axis", [0, -1, (0, 1)])
    def test_vector_norm_axis(self, ord_, axis):
        a = _rand(3, 4, 5)
        for kd in (False, True):
            got = _h(xp.linalg.vector_norm(_g(a), axis=axis, ord=ord_, keepdims=kd))
            exp = np.linalg.vector_norm(a, axis=axis, ord=ord_, keepdims=kd)
            assert got.shape == exp.shape
            assert_allclose(got, exp, rtol=1e-10)

    def test_vector_norm_complex(self):
        a = _rand(6, complex_=True)
        for o in (1, 2, np.inf):
            assert_allclose(_h(xp.linalg.vector_norm(_g(a), ord=o)),
                            np.linalg.vector_norm(a, ord=o))

    def test_vector_norm_int_input(self):
        a = np.array([3, 4])
        assert_allclose(_h(xp.linalg.vector_norm(_g(a))), 5.0)

    def test_vector_norm_zero_vector(self):
        assert _h(xp.linalg.vector_norm(_g(np.zeros(4)))) == 0.0

    def test_vector_norm_float32_dtype(self):
        a = _rand(5).astype(np.float32)
        assert _h(xp.linalg.vector_norm(_g(a))).dtype == np.float32

    def test_vecdot(self):
        a, b = _rand(3, 4, complex_=True), _rand(3, 4, complex_=True)
        assert_allclose(_h(xp.linalg.vecdot(_g(a), _g(b))), np.linalg.vecdot(a, b))

    def test_vecdot_axis(self):
        a, b = _rand(3, 4), _rand(3, 4)
        assert_allclose(_h(xp.linalg.vecdot(_g(a), _g(b), axis=0)),
                        np.linalg.vecdot(a, b, axis=0))

    @pytest.mark.parametrize("offset", [0, 1, -2])
    def test_diagonal(self, offset):
        a = _rand(2, 3, 5, 5)
        assert_array_equal(_h(xp.linalg.diagonal(_g(a), offset=offset)),
                           np.linalg.diagonal(a, offset=offset))

    def test_diagonal_rectangular(self):
        a = _rand(3, 5)
        assert_array_equal(_h(xp.linalg.diagonal(_g(a))), np.linalg.diagonal(a))

    @pytest.mark.parametrize("offset", [0, 1, -1])
    def test_trace(self, offset):
        a = _rand(2, 4, 4)
        assert_allclose(_h(xp.linalg.trace(_g(a), offset=offset)),
                        np.linalg.trace(a, offset=offset))

    def test_trace_dtype(self):
        a = np.arange(9).reshape(3, 3)
        got = _h(xp.linalg.trace(_g(a), dtype=np.float32))
        assert got.dtype == np.float32
        assert got == np.linalg.trace(a)

    def test_outer(self):
        a, b = _rand(4), _rand(3)
        assert_allclose(_h(xp.linalg.outer(_g(a), _g(b))), np.linalg.outer(a, b))

    def test_outer_flattens_like_cupy(self):
        a = _rand(2, 2)
        b = _rand(3)
        assert_allclose(_h(xp.linalg.outer(_g(a), _g(b))), np.outer(a, b))

    def test_svdvals(self):
        a = _rand(5, 3)
        assert_allclose(_h(xp.linalg.svdvals(_g(a))), np.linalg.svdvals(a), rtol=1e-8)

    def test_svdvals_stacked(self):
        a = _rand(2, 3, 4, 3)
        got = _h(xp.linalg.svdvals(_g(a)))
        assert got.shape == (2, 3, 3)
        assert_allclose(got, np.linalg.svdvals(a), rtol=1e-8)

    def test_svdvals_sorted_descending(self):
        s = _h(xp.linalg.svdvals(_g(_rand(6, 6))))
        assert np.all(np.diff(s) <= 0)

    @pytest.mark.parametrize("axes", [2, 1, 0, ([1, 0], [0, 1])])
    def test_tensordot(self, axes):
        a, b = _rand(3, 4), _rand(4, 3)
        if axes == 2:
            b = _rand(3, 4)
        if axes == 1:
            b = _rand(4, 5)
        if axes == 0:
            b = _rand(2, 5)
        assert_allclose(_h(xp.linalg.tensordot(_g(a), _g(b), axes=axes)),
                        np.linalg.tensordot(a, b, axes=axes))

    def test_tensordot_mismatch(self):
        with pytest.raises(Exception):
            xp.linalg.tensordot(_g(_rand(3, 4)), _g(_rand(5, 3)), axes=1)

    def test_matrix_transpose(self):
        a = _rand(2, 3, 4)
        got = _h(xp.linalg.matrix_transpose(_g(a)))
        assert got.shape == (2, 4, 3)
        assert_array_equal(got, np.linalg.matrix_transpose(a))

    def test_matrix_transpose_1d_raises(self):
        with pytest.raises(Exception):
            xp.linalg.matrix_transpose(_g(_rand(3)))

    def test_cross(self):
        a, b = _rand(4, 3), _rand(4, 3)
        assert_allclose(_h(xp.linalg.cross(_g(a), _g(b))), np.linalg.cross(a, b))

    def test_cross_wrong_dim(self):
        with pytest.raises(Exception):
            xp.linalg.cross(_g(_rand(4)), _g(_rand(4)))

    def test_multi_dot(self):
        mats = [_rand(3, 4), _rand(4, 5), _rand(5, 2)]
        assert_allclose(_h(xp.linalg.multi_dot([_g(m) for m in mats])),
                        np.linalg.multi_dot(mats))

    def test_multi_dot_two_and_out(self):
        a, b = _rand(3, 4), _rand(4, 2)
        out = xp.empty((3, 2))
        r = xp.linalg.multi_dot([_g(a), _g(b)], out=out)
        assert r is out
        assert_allclose(_h(out), a @ b)

    def test_multi_dot_single_array(self):
        # numpy requires >= 2 arrays
        with pytest.raises(ValueError):
            np.linalg.multi_dot([_rand(3, 3)])
        with pytest.raises(Exception):
            xp.linalg.multi_dot([_g(_rand(3, 3))])

    def test_native_functions_forwarded(self):
        a = _spd_batch(2, n=4) + 4 * np.eye(4)
        assert_allclose(_h(xp.linalg.inv(_g(a))), np.linalg.inv(a), rtol=1e-8)
        assert_allclose(_h(xp.linalg.det(_g(a))), np.linalg.det(a), rtol=1e-8)
        assert_allclose(_h(xp.linalg.norm(_g(a))), np.linalg.norm(a))

    def test_proxy_dir_and_attrs(self):
        d = dir(xp.linalg)
        for n in ("vector_norm", "matrix_norm", "svdvals", "multi_dot", "inv", "cross"):
            assert n in d, n
        with pytest.raises(AttributeError):
            xp.linalg.not_a_function
        assert xp.linalg is not np.linalg


class TestForwardedDtypes:
    def test_identity(self):
        assert xp.datetime64 is np.datetime64
        assert xp.dtypes is np.dtypes
        assert xp.finfo is np.finfo
        assert xp.iinfo is np.iinfo
        assert xp.dtype is np.dtype
        assert xp.timedelta64 is np.timedelta64

    def test_usable_with_gpu_arrays(self):
        assert xp.finfo(xp.float32).eps == np.finfo(np.float32).eps
        assert xp.zeros(2, dtype=xp.dtype("float32")).dtype == np.float32
        assert xp.isdtype(xp.zeros(2).dtype, "real floating")

    def test_gpu_arrays_are_cupy(self):
        import cupy

        assert isinstance(xp.zeros(3), cupy.ndarray)
        assert xp.zeros is cupy.zeros


class TestArrayApiVersion:
    def test_gpu(self):
        assert isinstance(xp.__array_api_version__, str)

    def test_cpu(self):
        with xp.backend("cpu"):
            v = xp.__array_api_version__
        assert isinstance(v, str)
