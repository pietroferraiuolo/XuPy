"""NumPy 2 compatibility tests for xupy.ma (average axis errors, __array__ protocol)."""
import warnings

import numpy as np
import pytest
from numpy.exceptions import AxisError

import xupy as xp
from xupy import ma
from xupy.ma.extras import average

on_gpu = xp.on_gpu
requires_gpu = pytest.mark.skipif(not on_gpu, reason="XuPy is not running on GPU")
requires_cpu = pytest.mark.skipif(on_gpu, reason="XuPy is running on GPU")


def _make(dtype=None):
    data = xp.arange(12, dtype=dtype or xp.float64).reshape(3, 4)
    mask = xp.zeros((3, 4), dtype=bool)
    mask[0, 0] = True
    return ma.masked_array(data, mask=mask)


class TestAverageAxisError:
    def test_exported_in_ma(self):
        assert ma.average is average

    @pytest.mark.parametrize("axis", [5, -3, 2, -10])
    def test_out_of_range_axis(self, axis):
        with pytest.raises(AxisError):
            average(_make(), axis=axis)

    def test_axis_error_is_value_and_index_error(self):
        with pytest.raises((ValueError, IndexError)):
            average(_make(), axis=5)

    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    def test_valid_axes_still_work(self, axis):
        res = average(_make(), axis=axis)
        assert np.asarray(res).shape == ((4,) if axis in (0, -2) else (3,))

    def test_axis_none(self):
        res = average(_make())
        assert float(res) == pytest.approx(np.arange(1, 12).mean())

    def test_axis_error_with_weights(self):
        with pytest.raises(AxisError):
            average(_make(), axis=5, weights=xp.ones((3, 4)))


class TestArrayProtocol:
    def test_asarray_works(self):
        out = np.asarray(_make())
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, 4)
        np.testing.assert_array_equal(out, np.arange(12, dtype=float).reshape(3, 4))

    def test_asarray_no_deprecation_warning(self):
        xma = _make()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            np.asarray(xma)
            np.array(xma)

    def test_asarray_honours_dtype(self):
        out = np.asarray(_make(), dtype=np.float32)
        assert out.dtype == np.float32

    def test_asarray_dtype_int(self):
        out = np.asarray(_make(), dtype=np.int64)
        assert out.dtype == np.int64
        assert out[2, 3] == 11

    def test_dunder_array_direct(self):
        out = _make().__array__()
        assert isinstance(out, np.ndarray)

    @requires_gpu
    def test_copy_false_raises_on_gpu(self):
        with pytest.raises(ValueError):
            np.asarray(_make(), copy=False)

    @requires_gpu
    def test_dunder_array_copy_false_raises_on_gpu(self):
        with pytest.raises(ValueError):
            _make().__array__(copy=False)

    @requires_gpu
    def test_copy_true_on_gpu_ok(self):
        out = np.asarray(_make(), copy=True)
        assert isinstance(out, np.ndarray)

    @requires_cpu
    def test_copy_true_does_not_share_memory_on_cpu(self):
        xma = _make()
        out = np.asarray(xma, copy=True)
        assert not np.shares_memory(out, xma.data)
        out[0, 1] = -99
        assert xma.data[0, 1] == 1

    @requires_cpu
    def test_copy_false_on_cpu_does_not_raise(self):
        out = np.asarray(_make(), copy=False)
        assert isinstance(out, np.ndarray)
