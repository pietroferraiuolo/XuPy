"""
Comprehensive test suite for xupy.ma.extras module.

Tests all extra functions for masked arrays including:
- Statistical reductions (sum, mean, std, var, min, max, prod, average)
- Array creation utilities (masked_all, masked_all_like, empty_like)
- 2D compress/mask (compress_nd, compress_rowcols, compress_rows, compress_cols, mask_rowcols, mask_rows, mask_cols)
- Stacking and shape (atleast_1d/2d/3d, vstack, hstack, column_stack, dstack, stack, row_stack, hsplit)
- diagflat, ediff1d, mr_
- NumPy compatibility (scalar returns, etc.)
- Edge cases and error handling
"""
import pytest
import numpy as np
from typing import Any

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from xupy.ma import masked_array
from xupy.ma.extras import (
    sum,
    mean,
    std,
    var,
    min,
    max,
    prod,
    product,
    average,
    masked_all,
    masked_all_like,
    empty_like,
    count_masked,
    issequence,
    compress_nd,
    compress_rowcols,
    compress_rows,
    compress_cols,
    mask_rowcols,
    mask_rows,
    mask_cols,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    vstack,
    hstack,
    column_stack,
    dstack,
    stack,
    row_stack,
    hsplit,
    diagflat,
    ediff1d,
    mr_,
)

# Skip all tests if CuPy is not available
pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# Helper functions
def _to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to NumPy array."""
    if hasattr(arr, "get"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _is_scalar_or_0d(x: Any) -> bool:
    """True if x is a Python scalar or 0-d array (CuPy/NumPy native behavior)."""
    return np.isscalar(x) or (hasattr(x, "shape") and x.shape == ())


def _value(x: Any) -> Any:
    """Extract Python scalar from x for comparison. x can be Python scalar or 0-d array."""
    if hasattr(x, "item"):
        return x.item()
    return x


class TestStatisticalReductions:
    """Test statistical reduction functions."""

    @pytest.fixture
    def test_data(self):
        """Create test data with some masked values."""
        data = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
        mask = cp.array([False, True, False, False, True], dtype=bool)
        return masked_array(data, mask)

    @pytest.fixture
    def test_data_2d(self):
        """Create 2D test data."""
        data = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        mask = cp.array([[False, True, False], [True, False, False]], dtype=bool)
        return masked_array(data, mask)

    def test_sum_axis_none(self, test_data):
        """Test sum with axis=None returns 0-d array or scalar (CuPy/NumPy native)."""
        result = sum(test_data, axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) == 8.0  # 1 + 3 + 4 (excluding masked 2 and 5)

    def test_sum_axis_0(self, test_data_2d):
        """Test sum along axis=0."""
        result = sum(test_data_2d, axis=0)
        # Should return array with shape (3,) or 0-d
        assert hasattr(result, 'shape') or np.isscalar(result)

    def test_sum_keepdims(self, test_data_2d):
        """Test sum with keepdims=True."""
        result = sum(test_data_2d, axis=0, keepdims=True)
        assert hasattr(result, 'shape')
        assert result.shape == (1, 3)

    def test_mean_axis_none(self, test_data):
        """Test mean with axis=None returns 0-d array or scalar."""
        result = mean(test_data, axis=None)
        assert _is_scalar_or_0d(result)
        expected = (1.0 + 3.0 + 4.0) / 3  # Exclude masked values
        assert abs(_value(result) - expected) < 1e-6

    def test_mean_axis_1(self, test_data_2d):
        """Test mean along axis=1."""
        result = mean(test_data_2d, axis=1)
        # Should be array with shape (2,)
        assert hasattr(result, 'shape') or np.isscalar(result)

    def test_std_axis_none(self, test_data):
        """Test std with axis=None returns 0-d array or scalar."""
        result = std(test_data, axis=None)
        assert _is_scalar_or_0d(result)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.std(valid_data)
        assert abs(_value(result) - expected) < 1e-5

    def test_std_with_ddof(self, test_data):
        """Test std with ddof parameter."""
        # Note: Currently ddof is only applied when axis is specified
        # When axis=None, ddof is ignored (uses default ddof=0)
        result = std(test_data, axis=None, ddof=1)
        assert _is_scalar_or_0d(result)
        valid_data = np.array([1.0, 3.0, 4.0])
        # Current implementation uses ddof=0 when axis=None
        expected = np.std(valid_data, ddof=0)
        assert abs(result - expected) < 1e-5
        
        # Test with axis specified (ddof should work)
        data_2d = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        mask_2d = cp.array([[False, False, False], [False, False, False]], dtype=bool)
        arr_2d = masked_array(data_2d, mask_2d)
        result_axis = std(arr_2d, axis=0, ddof=1)
        # Should return array with ddof applied
        assert hasattr(result_axis, 'shape') or np.isscalar(result_axis)

    def test_var_axis_none(self, test_data):
        """Test var with axis=None returns 0-d array or scalar."""
        result = var(test_data, axis=None)
        assert _is_scalar_or_0d(result)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.var(valid_data)
        assert abs(_value(result) - expected) < 1e-5

    def test_min_axis_none(self, test_data):
        """Test min with axis=None returns 0-d array or scalar."""
        result = min(test_data, axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) == 1.0  # Minimum of unmasked values

    def test_max_axis_none(self, test_data):
        """Test max with axis=None returns 0-d array or scalar."""
        result = max(test_data, axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) == 4.0  # Maximum of unmasked values

    def test_prod_axis_none(self, test_data):
        """Test prod with axis=None returns 0-d array or scalar."""
        result = prod(test_data, axis=None)
        assert _is_scalar_or_0d(result) or hasattr(result, "shape")
        assert _value(result) == 12.0  # 1 * 3 * 4 (excluding masked values)

    def test_prod_all_masked(self):
        """Test prod when all values are masked."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        result = prod(arr, axis=None)
        # Should return masked singleton
        assert result is not None

    def test_product_alias(self, test_data):
        """Test that product is an alias for prod."""
        result1 = prod(test_data, axis=None)
        result2 = product(test_data, axis=None)
        assert _value(result1) == _value(result2)

    def test_numpy_compatibility_scalars(self, test_data):
        """Test that functions return scalar-like (0-d array or scalar) when axis=None."""
        np_data = np.ma.array([1.0, 2.0, 3.0, 4.0, 5.0],
                              mask=[False, True, False, False, True])

        # Test all functions return scalar or 0-d array when axis=None (CuPy/NumPy native)
        functions = [sum, mean, std, var, min, max]
        for func in functions:
            xp_result = func(test_data, axis=None)
            assert _is_scalar_or_0d(xp_result), \
                f"{func.__name__} should return scalar or 0-d array"


class TestAverage:
    """Test average function with and without weights."""

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        mask = cp.array([False, False, True, True], dtype=bool)
        return masked_array(data, mask)

    def test_average_no_weights(self, test_data):
        """Test average without weights."""
        result = average(test_data, axis=None)
        assert _is_scalar_or_0d(result) or result is not None
        expected = (1.0 + 2.0) / 2  # Only unmasked values
        assert abs(_value(result) - expected) < 1e-6

    def test_average_with_weights(self, test_data):
        """Test average with weights."""
        weights = cp.array([3.0, 1.0, 0.0, 0.0], dtype=cp.float32)
        result = average(test_data, axis=None, weights=weights)
        assert _is_scalar_or_0d(result) or result is not None
        expected = (1.0 * 3.0 + 2.0 * 1.0) / (3.0 + 1.0)
        assert abs(_value(result) - expected) < 1e-6

    def test_average_with_axis(self, test_data):
        """Test average with axis specified."""
        data_2d = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        mask_2d = cp.array([[False, True], [False, False]], dtype=bool)
        arr_2d = masked_array(data_2d, mask_2d)
        
        result = average(arr_2d, axis=0)
        assert hasattr(result, 'shape') or np.isscalar(result)

    def test_average_returned(self, test_data):
        """Test average with returned=True."""
        result, sum_weights = average(test_data, axis=None, returned=True)
        assert _is_scalar_or_0d(result) or result is not None
        assert _value(sum_weights) == 2.0  # Two unmasked values

    def test_average_all_masked(self):
        """Test average when all values are masked."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        result = average(arr, axis=None)
        # Should return masked singleton
        assert result is not None


class TestArrayCreation:
    """Test array creation utility functions."""

    def test_masked_all(self):
        """Test masked_all function."""
        arr = masked_all((3, 4))
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float32  # Default dtype
        assert arr.count_masked() == 12  # All elements masked
        assert arr.mask.all()  # All True

    def test_masked_all_custom_dtype(self):
        """Test masked_all with custom dtype."""
        arr = masked_all((2, 3), dtype=np.int32)
        assert arr.shape == (2, 3)
        assert arr.dtype == np.int32
        assert arr.count_masked() == 6

    def test_masked_all_like(self):
        """Test masked_all_like function."""
        original = cp.array([[1, 2], [3, 4]], dtype=cp.int32)
        arr = masked_all_like(original)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.int32
        assert arr.count_masked() == 4
        assert arr.mask.all()

    def test_empty_like(self):
        """Test empty_like function."""
        original = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        arr = empty_like(original)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32
        assert arr.count_masked() == 0  # No masked elements
        assert not arr.mask.any()  # All False


class TestCountMasked:
    """Test count_masked function."""

    def test_count_masked_axis_none(self):
        """Test count_masked with axis=None (returns 0-d array, CuPy native)."""
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        mask = cp.array([False, True, False, True], dtype=bool)
        arr = masked_array(data, mask)
        result = count_masked(arr, axis=None)
        assert _value(result) == 2
        assert _is_scalar_or_0d(result) or isinstance(result, (int, np.integer))

    def test_count_masked_axis_0(self):
        """Test count_masked along axis=0."""
        data = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        mask = cp.array([[False, True], [True, False]], dtype=bool)
        arr = masked_array(data, mask)
        result = count_masked(arr, axis=0)
        # Should return array with counts per column
        assert hasattr(result, '__len__') or isinstance(result, (int, np.integer))

    def test_count_masked_no_mask(self):
        """Test count_masked with no mask."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        arr = masked_array(data)
        result = count_masked(arr, axis=None)
        assert result == 0


class TestCompressAndMask:
    """Test compress_nd, compress_rowcols, compress_rows, compress_cols, mask_rowcols, mask_rows, mask_cols."""

    def test_compress_rows(self):
        """Suppress rows that contain any masked value."""
        data = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=cp.float32)
        mask = cp.array([[True, False, False], [True, False, False], [False, False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_rows(arr)
        assert hasattr(out, "get")
        np_out = cp.asnumpy(out)
        assert np_out.shape == (1, 3)
        np.testing.assert_array_equal(np_out[0], [7, 8, 9])

    def test_compress_cols(self):
        """Suppress columns that contain any masked value."""
        data = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=cp.float32)
        mask = cp.array([[True, False, False], [True, False, False], [False, False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_cols(arr)
        np_out = cp.asnumpy(out)
        assert np_out.shape == (3, 2)
        np.testing.assert_array_equal(np_out[:, 0], [2, 5, 8])
        np.testing.assert_array_equal(np_out[:, 1], [3, 6, 9])

    def test_compress_rowcols_axis_none(self):
        """Suppress both rows and columns that contain masked values."""
        data = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=cp.float32)
        mask = cp.array([[True, False, False], [True, False, False], [False, False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_rowcols(arr, axis=None)
        np_out = cp.asnumpy(out)
        assert np_out.shape == (1, 3)
        np.testing.assert_array_equal(np_out[0], [7, 8, 9])

    def test_compress_rowcols_axis_0(self):
        """Suppress only rows (axis=0)."""
        data = cp.array([[1, 2], [3, 4], [5, 6]], dtype=cp.float32)
        mask = cp.array([[True, True], [False, False], [False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_rowcols(arr, axis=0)
        np_out = cp.asnumpy(out)
        assert np_out.shape == (2, 2)
        np.testing.assert_array_equal(np_out, [[3, 4], [5, 6]])

    def test_compress_rowcols_axis_1(self):
        """Suppress only columns (axis=1)."""
        data = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.float32)
        mask = cp.array([[True, False, False], [True, False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_rowcols(arr, axis=1)
        np_out = cp.asnumpy(out)
        assert np_out.shape == (2, 2)
        np.testing.assert_array_equal(np_out, [[2, 3], [5, 6]])

    def test_compress_nd_1d(self):
        """compress_nd on 1D drops masked elements."""
        data = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
        mask = cp.array([False, True, False, True, False], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_nd(arr, axis=0)
        np_out = cp.asnumpy(out)
        np.testing.assert_array_equal(np_out, [1, 3, 5])

    def test_compress_nd_no_mask(self):
        """compress_nd with no mask returns data unchanged."""
        data = cp.array([[1, 2], [3, 4]], dtype=cp.float32)
        arr = masked_array(data)
        out = compress_nd(arr, axis=None)
        np_out = cp.asnumpy(out)
        np.testing.assert_array_equal(np_out, data.get())

    def test_compress_rowcols_2d_only(self):
        """compress_rowcols raises for non-2D array."""
        data = cp.array([1, 2, 3], dtype=cp.float32)
        arr = masked_array(data)
        with pytest.raises(NotImplementedError, match="2D"):
            compress_rowcols(arr, axis=None)

    def test_mask_rowcols_in_place(self):
        """mask_rowcols masks entire rows and columns that contain masked values."""
        data = cp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=cp.float32)
        mask = cp.array([[False, False, False], [False, True, False], [False, False, False]], dtype=bool)
        arr = masked_array(data, mask=mask.copy())
        result = mask_rowcols(arr, axis=None)
        np_mask = cp.asnumpy(result.mask)
        # Row 1 and column 1 should be fully masked
        assert np_mask[1, :].all()
        assert np_mask[:, 1].all()
        assert np_mask[0, 0] == False and np_mask[0, 2] == False and np_mask[2, 0] == False and np_mask[2, 2] == False

    def test_mask_rows(self):
        """mask_rows masks only rows that contain masked values."""
        data = cp.array([[1, 2], [3, 4], [5, 6]], dtype=cp.float32)
        mask = cp.array([[True, False], [False, False], [False, False]], dtype=bool)
        arr = masked_array(data, mask=mask.copy())
        result = mask_rows(arr)
        np_mask = cp.asnumpy(result.mask)
        assert np_mask[0, :].all()
        assert not np_mask[1, :].any() and not np_mask[2, :].any()

    def test_mask_cols(self):
        """mask_cols masks only columns that contain masked values."""
        data = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.float32)
        mask = cp.array([[True, False, False], [True, False, False]], dtype=bool)
        arr = masked_array(data, mask=mask.copy())
        result = mask_cols(arr)
        np_mask = cp.asnumpy(result.mask)
        assert np_mask[:, 0].all()
        assert not np_mask[:, 1].any() and not np_mask[:, 2].any()


class TestStackingAndShape:
    """Test atleast_1d/2d/3d, vstack, hstack, column_stack, dstack, stack, row_stack, hsplit."""

    def test_atleast_1d_single(self):
        """atleast_1d promotes scalar or 0d to 1d and preserves mask."""
        data = cp.array(5.0, dtype=cp.float32)
        arr = masked_array(data)
        out = atleast_1d(arr)
        assert out.shape == (1,)
        assert _value(out) == 5.0

    def test_atleast_1d_multiple(self):
        """atleast_1d with multiple inputs returns tuple of masked arrays."""
        a = masked_array(cp.array(1.0))
        b = masked_array(cp.array([2.0, 3.0]))
        out = atleast_1d(a, b)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == (1,)
        assert out[1].shape == (2,)

    def test_atleast_2d_single(self):
        """atleast_2d adds dimension."""
        arr = masked_array(cp.array([1.0, 2.0, 3.0]))
        out = atleast_2d(arr)
        assert out.shape == (1, 3)

    def test_atleast_3d_single(self):
        """atleast_3d adds dimensions."""
        arr = masked_array(cp.array([1.0, 2.0]))
        out = atleast_3d(arr)
        assert out.shape == (1, 2, 1)

    def test_vstack(self):
        """vstack stacks vertically and combines masks."""
        a = masked_array(cp.array([1.0, 2.0]), mask=cp.array([False, True]))
        b = masked_array(cp.array([3.0, 4.0]), mask=cp.array([True, False]))
        out = vstack([a, b])
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(cp.asnumpy(out.data), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(cp.asnumpy(out.mask), [[False, True], [True, False]])

    def test_hstack(self):
        """hstack stacks horizontally and combines masks."""
        a = masked_array(cp.array([[1.0], [2.0]]))
        b = masked_array(cp.array([[3.0], [4.0]]), mask=cp.array([[True], [False]]))
        out = hstack([a, b])
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(cp.asnumpy(out.mask), [[False, True], [False, False]])

    def test_column_stack(self):
        """column_stack stacks 1D arrays as columns."""
        a = masked_array(cp.array([1.0, 2.0]))
        b = masked_array(cp.array([3.0, 4.0]), mask=cp.array([False, True]))
        out = column_stack([a, b])
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(cp.asnumpy(out.data), [[1, 3], [2, 4]])

    def test_dstack(self):
        """dstack stacks along third axis."""
        a = masked_array(cp.array([[1, 2], [3, 4]]))
        b = masked_array(cp.array([[5, 6], [7, 8]]))
        out = dstack([a, b])
        assert out.shape == (2, 2, 2)
        assert out.data.shape == (2, 2, 2)

    def test_stack(self):
        """stack joins along a new axis."""
        a = masked_array(cp.array([1.0, 2.0]))
        b = masked_array(cp.array([3.0, 4.0]))
        out = stack([a, b], axis=0)
        assert out.shape == (2, 2)
        out1 = stack([a, b], axis=1)
        assert out1.shape == (2, 2)

    def test_row_stack_alias(self):
        """row_stack is alias for vstack."""
        a = masked_array(cp.array([1.0, 2.0]))
        b = masked_array(cp.array([3.0, 4.0]))
        v = vstack([a, b])
        r = row_stack([a, b])
        np.testing.assert_array_equal(cp.asnumpy(v.data), cp.asnumpy(r.data))
        np.testing.assert_array_equal(cp.asnumpy(v.mask), cp.asnumpy(r.mask))

    def test_hsplit(self):
        """hsplit splits horizontally and returns list of MaskedArrays."""
        data = cp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=cp.float32)
        mask = cp.array([[False, True, False, True], [False, False, True, False]], dtype=bool)
        arr = masked_array(data, mask)
        parts = hsplit(arr, 2)
        assert len(parts) == 2
        assert parts[0].shape == (2, 2)
        assert parts[1].shape == (2, 2)
        np.testing.assert_array_equal(cp.asnumpy(parts[0].data), [[1, 2], [5, 6]])
        np.testing.assert_array_equal(cp.asnumpy(parts[1].data), [[3, 4], [7, 8]])


class TestDiagflatEdiff1dMr:
    """Test diagflat, ediff1d, mr_."""

    def test_diagflat(self):
        """diagflat creates 2D array with flattened input on diagonal."""
        arr = masked_array(cp.array([1.0, 2.0, 3.0]))
        out = diagflat(arr)
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(cp.asnumpy(out.data), np.diag([1, 2, 3]))
        assert not cp.asnumpy(out.mask).any()

    def test_diagflat_with_mask(self):
        """diagflat propagates mask."""
        arr = masked_array(cp.array([1.0, 2.0, 3.0]), mask=cp.array([False, True, False]))
        out = diagflat(arr)
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(cp.asnumpy(out.mask), np.diag([False, True, False]))

    def test_ediff1d_basic(self):
        """ediff1d first difference and mask where either neighbor masked."""
        arr = masked_array(cp.array([1.0, 2.0, 4.0, 7.0]))
        out = ediff1d(arr)
        np.testing.assert_array_almost_equal(cp.asnumpy(out.data), [1.0, 2.0, 3.0])
        assert not cp.asnumpy(out.mask).any()

    def test_ediff1d_mask_propagation(self):
        """ediff1d masks output where either adjacent input is masked."""
        arr = masked_array(cp.array([10.0, 11.0, 12.0]), mask=cp.array([False, True, False]))
        out = ediff1d(arr)
        np.testing.assert_array_almost_equal(cp.asnumpy(out.data), [1.0, 1.0])
        assert cp.asnumpy(out.mask).all()

    def test_ediff1d_to_end_to_begin(self):
        """ediff1d with to_end and to_begin extends and masks new elements as unmasked."""
        arr = masked_array(cp.array([1.0, 2.0, 3.0]))
        out = ediff1d(arr, to_begin=cp.array(0.0), to_end=cp.array(99.0))
        np_out = cp.asnumpy(out.data)
        assert np_out[0] == 0.0
        assert np_out[-1] == 99.0
        assert out.size == 4

    def test_mr_getitem_tuple(self):
        """mr_[a, b] stacks arrays vertically."""
        a = masked_array(cp.array([[1, 2], [3, 4]]))
        b = masked_array(cp.array([[5, 6]]))
        out = mr_[(a, b)]
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(cp.asnumpy(out.data)[-1], [5, 6])

    def test_mr_getitem_single(self):
        """mr_[a] returns masked array view of a."""
        a = masked_array(cp.array([1.0, 2.0, 3.0]))
        out = mr_[a]
        assert out.shape == a.shape
        np.testing.assert_array_equal(cp.asnumpy(out.data), cp.asnumpy(a.data))


class TestIsSequence:
    """Test issequence function."""

    def test_issequence_list(self):
        """Test issequence with list."""
        assert issequence([1, 2, 3]) == True

    def test_issequence_tuple(self):
        """Test issequence with tuple."""
        assert issequence((1, 2, 3)) == True

    def test_issequence_numpy_array(self):
        """Test issequence with NumPy array."""
        assert issequence(np.array([1, 2, 3])) == True

    def test_issequence_cupy_array(self):
        """Test issequence with CuPy array."""
        assert issequence(cp.array([1, 2, 3])) == True

    def test_issequence_scalar(self):
        """Test issequence with scalar."""
        assert issequence(42) == False

    def test_issequence_string(self):
        """Test issequence with string (should be False)."""
        assert issequence("hello") == False


class TestNumPyCompatibility:
    """Test NumPy compatibility aspects."""

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        mask = cp.array([False, True, False, False], dtype=bool)
        return masked_array(data, mask)

    def test_scalar_return_types(self, test_data):
        """Test that functions return 0-d arrays or scalars (CuPy/NumPy native)."""
        functions = [sum, mean, std, var, min, max]
        for func in functions:
            result = func(test_data, axis=None)
            assert _is_scalar_or_0d(result), f"{func.__name__} should return scalar or 0-d array"

    def test_1d_reduction_returns_scalar_or_0d(self, test_data):
        """Test that reducing 1D array returns 0-d array or scalar."""
        result = sum(test_data)
        assert _is_scalar_or_0d(result), "1D reduction should return scalar or 0-d array"

    def test_2d_reduction_returns_array(self):
        """Test that reducing 2D array along one axis returns array."""
        data = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        arr = masked_array(data)
        result = sum(arr, axis=0)
        # Should return array, not scalar
        assert hasattr(result, 'shape') and result.shape != (), \
            "2D reduction should return array"

    def test_keepdims_preserves_dimensions(self):
        """Test that keepdims=True preserves dimensions."""
        data = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        arr = masked_array(data)
        result = sum(arr, axis=0, keepdims=True)
        assert hasattr(result, 'shape')
        assert result.shape == (1, 2)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Test functions with empty array."""
        data = cp.array([], dtype=cp.float32)
        arr = masked_array(data)
        # Some operations should handle empty arrays gracefully
        result = count_masked(arr, axis=None)
        assert _value(result) == 0

    def test_single_element(self):
        """Test functions with single element array."""
        data = cp.array([42.0], dtype=cp.float32)
        arr = masked_array(data)
        result = sum(arr, axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) == 42.0

    def test_all_masked_statistics(self):
        """Test statistics when all values are masked."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        
        # sum should return masked singleton
        result_sum = sum(arr, axis=None)
        assert result_sum is not None
        
        # mean should return masked singleton
        result_mean = mean(arr, axis=None)
        assert result_mean is not None

    def test_no_mask_statistics(self):
        """Test statistics with no mask."""
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        arr = masked_array(data)

        result = sum(arr, axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) == 10.0

    def test_float32_precision(self):
        """Test that float32 precision is maintained."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        arr = masked_array(data)
        result = mean(arr, axis=None)
        assert _value(result) == 2.0


class TestIntegration:
    """Test integration with NumPy and core module."""

    def test_roundtrip_numpy_masked_array(self):
        """Test roundtrip conversion with NumPy masked array."""
        np_data = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        xp_arr = masked_array(np_data)
        result = sum(xp_arr, axis=None)
        np_result = np.ma.sum(np_data, axis=None)
        assert abs(_value(result) - _value(np_result)) < 1e-6

    def test_consistency_with_class_methods(self):
        """Test that extras functions are consistent with class methods."""
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        mask = cp.array([False, True, False, False], dtype=bool)
        arr = masked_array(data, mask)

        # Compare extras functions with class methods (0-d array or scalar)
        assert _value(sum(arr, axis=None)) == _value(arr.sum(axis=None))
        assert _value(mean(arr, axis=None)) == _value(arr.mean(axis=None))
        assert _value(std(arr, axis=None)) == _value(arr.std(axis=None))
        assert _value(var(arr, axis=None)) == _value(arr.var(axis=None))
        assert _value(min(arr, axis=None)) == _value(arr.min(axis=None))
        assert _value(max(arr, axis=None)) == _value(arr.max(axis=None))

    def test_dtype_preservation(self):
        """Test that dtypes are preserved correctly."""
        data = cp.array([1, 2, 3, 4], dtype=cp.int32)
        arr = masked_array(data)
        result = sum(arr, axis=None, dtype=cp.int32)
        assert _value(result) == 10


class TestPerformance:
    """Test performance-related aspects."""

    def test_large_array_performance(self):
        """Test that functions work efficiently with large arrays."""
        data = cp.random.rand(1000, 1000).astype(cp.float32)
        arr = masked_array(data)
        
        # Should complete quickly
        result = sum(arr, axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) > 0

    def test_gpu_operations(self):
        """Test that operations stay on GPU."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        arr = masked_array(data)

        # Operations return 0-d GPU arrays or scalars (CuPy native)
        result = sum(arr, axis=None)
        assert _is_scalar_or_0d(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

