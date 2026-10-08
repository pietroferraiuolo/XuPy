"""
Comprehensive test suite for xupy.ma.extras module.

Tests all extra functions for masked arrays including:
- Statistical reductions (sum, mean, std, var, min, max, prod, average)
- Array creation utilities (masked_all, masked_all_like, empty_like, zeros_like, ones_like)
- 2D compress/mask (compress_nd, compress_rowcols, compress_rows, compress_cols, mask_rowcols, mask_rows, mask_cols)
- Stacking and shape (atleast_1d/2d/3d, vstack, hstack, column_stack, dstack, stack, row_stack, hsplit)
- diagflat, ediff1d, mr_
- NumPy compatibility (scalar returns, etc.)
- Edge cases and error handling
"""
import pytest
import numpy as np
from typing import Any

from xupy import _core

# Array module of the XuPy data: cupy when a usable GPU is available (the
# default backend then is the GPU), numpy otherwise.  The tests run on both.
cp = _core._cupy
xpm = cp if cp is not None else np

from xupy.ma import masked_array, MaskedArray, masked, nomask
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
    zeros_like,
    ones_like,
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

# Helper functions
def _to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to NumPy array."""
    if cp is not None and isinstance(arr, xpm.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


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
        data = xpm.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=xpm.float32)
        mask = xpm.array([False, True, False, False, True], dtype=bool)
        return masked_array(data, mask)

    @pytest.fixture
    def test_data_2d(self):
        """Create 2D test data."""
        data = xpm.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=xpm.float32)
        mask = xpm.array([[False, True, False], [True, False, False]], dtype=bool)
        return masked_array(data, mask)

    def test_sum_axis_none(self, test_data):
        """Test sum with axis=None returns a numpy scalar."""
        result = sum(test_data, axis=None)
        assert isinstance(result, np.generic)
        assert result == 8.0  # 1 + 3 + 4 (excluding masked 2 and 5)

    def test_sum_axis_0(self, test_data_2d):
        """Test sum along axis=0."""
        result = sum(test_data_2d, axis=0)
        assert result.shape == (3,)
        np.testing.assert_array_equal(_to_numpy(result.data), [1.0, 5.0, 9.0])

    def test_sum_keepdims(self, test_data_2d):
        """Test sum with keepdims=True."""
        result = sum(test_data_2d, axis=0, keepdims=True)
        assert hasattr(result, 'shape')
        assert result.shape == (1, 3)

    def test_mean_axis_none(self, test_data):
        """Test mean with axis=None returns a numpy scalar."""
        result = mean(test_data, axis=None)
        assert isinstance(result, np.generic)
        expected = (1.0 + 3.0 + 4.0) / 3  # Exclude masked values
        assert abs(_value(result) - expected) < 1e-6

    def test_mean_axis_1(self, test_data_2d):
        """Test mean along axis=1."""
        result = mean(test_data_2d, axis=1)
        # Should be array with shape (2,)
        assert hasattr(result, 'shape') or np.isscalar(result)

    def test_std_axis_none(self, test_data):
        """Test std with axis=None returns a numpy scalar."""
        result = std(test_data, axis=None)
        assert isinstance(result, np.generic)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.std(valid_data)
        assert abs(_value(result) - expected) < 1e-5

    def test_std_with_ddof(self, test_data):
        """Test std with ddof parameter."""
        # ddof is honoured also with axis=None (numpy.ma semantics)
        result = std(test_data, axis=None, ddof=1)
        assert isinstance(result, np.generic)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.std(valid_data, ddof=1)
        assert abs(result - expected) < 1e-5
        
        # Test with axis specified (ddof should work)
        data_2d = xpm.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=xpm.float32)
        mask_2d = xpm.array([[False, False, False], [False, False, False]], dtype=bool)
        arr_2d = masked_array(data_2d, mask_2d)
        result_axis = std(arr_2d, axis=0, ddof=1)
        # Should return an array with ddof applied
        assert result_axis.shape == (3,)
        np.testing.assert_allclose(_to_numpy(result_axis.data),
                                   np.std(_to_numpy(data_2d), axis=0, ddof=1), rtol=1e-5)

    def test_var_axis_none(self, test_data):
        """Test var with axis=None returns a numpy scalar."""
        result = var(test_data, axis=None)
        assert isinstance(result, np.generic)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.var(valid_data)
        assert abs(_value(result) - expected) < 1e-5

    def test_min_axis_none(self, test_data):
        """Test min with axis=None returns a numpy scalar."""
        result = min(test_data, axis=None)
        assert isinstance(result, np.generic)
        assert _value(result) == 1.0  # Minimum of unmasked values

    def test_max_axis_none(self, test_data):
        """Test max with axis=None returns a numpy scalar."""
        result = max(test_data, axis=None)
        assert isinstance(result, np.generic)
        assert _value(result) == 4.0  # Maximum of unmasked values

    def test_prod_axis_none(self, test_data):
        """Test prod with axis=None returns a numpy scalar."""
        result = prod(test_data, axis=None)
        assert isinstance(result, np.generic)
        assert result == 12.0  # 1 * 3 * 4 (excluding masked values)

    def test_prod_all_masked(self):
        """Test prod when all values are masked."""
        data = xpm.array([1.0, 2.0, 3.0], dtype=xpm.float32)
        mask = xpm.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        result = prod(arr, axis=None)
        # Should return masked singleton
        assert result is masked

    def test_product_alias(self, test_data):
        """Test that product is an alias for prod."""
        result1 = prod(test_data, axis=None)
        result2 = product(test_data, axis=None)
        assert result1 == result2

    def test_numpy_compatibility_scalars(self, test_data):
        """Test that functions return numpy scalars when axis=None."""
        np_data = np.ma.array([1.0, 2.0, 3.0, 4.0, 5.0],
                              mask=[False, True, False, False, True])

        # Test all functions return numpy scalars when axis=None
        functions = [sum, mean, std, var, min, max]
        for func in functions:
            xp_result = func(test_data, axis=None)
            assert isinstance(xp_result, np.generic), \
                f"{func.__name__} should return a numpy scalar"


class TestAverage:
    """Test average function with and without weights."""

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        data = xpm.array([1.0, 2.0, 3.0, 4.0], dtype=xpm.float32)
        mask = xpm.array([False, False, True, True], dtype=bool)
        return masked_array(data, mask)

    def test_average_no_weights(self, test_data):
        """Test average without weights."""
        result = average(test_data, axis=None)
        assert isinstance(result, np.generic)
        expected = (1.0 + 2.0) / 2  # Only unmasked values
        assert abs(result - expected) < 1e-6

    def test_average_with_weights(self, test_data):
        """Test average with weights."""
        weights = xpm.array([3.0, 1.0, 0.0, 0.0], dtype=xpm.float32)
        result = average(test_data, axis=None, weights=weights)
        assert isinstance(result, np.generic)
        expected = (1.0 * 3.0 + 2.0 * 1.0) / (3.0 + 1.0)
        assert abs(result - expected) < 1e-6

    def test_average_with_axis(self, test_data):
        """Test average with axis specified."""
        data_2d = xpm.array([[1.0, 2.0], [3.0, 4.0]], dtype=xpm.float32)
        mask_2d = xpm.array([[False, True], [False, False]], dtype=bool)
        arr_2d = masked_array(data_2d, mask_2d)
        
        result = average(arr_2d, axis=0)
        assert result.shape == (2,)
        np.testing.assert_allclose(_to_numpy(result.data), [2.0, 4.0])

    def test_average_returned(self, test_data):
        """Test average with returned=True."""
        result, sum_weights = average(test_data, axis=None, returned=True)
        assert isinstance(result, np.generic)
        assert isinstance(sum_weights, np.generic)
        assert sum_weights == 2.0  # Two unmasked values

    def test_average_all_masked(self):
        """Test average when all values are masked."""
        data = xpm.array([1.0, 2.0, 3.0], dtype=xpm.float32)
        mask = xpm.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        result = average(arr, axis=None)
        # Should return masked singleton
        assert result is masked


class TestArrayCreation:
    """Test array creation utility functions."""

    def test_masked_all(self):
        """Test masked_all function."""
        arr = masked_all((3, 4))
        assert arr.shape == (3, 4)
        assert arr.dtype == np.float64  # Default dtype (numpy.ma parity)
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
        original = xpm.array([[1, 2], [3, 4]], dtype=xpm.int32)
        arr = masked_all_like(original)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.int32
        assert arr.count_masked() == 4
        assert arr.mask.all()

    def test_empty_like(self):
        """Test empty_like function."""
        original = xpm.array([[1.0, 2.0], [3.0, 4.0]], dtype=xpm.float32)
        arr = empty_like(original)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32
        assert arr.count_masked() == 0  # No masked elements
        assert not arr.mask.any()  # All False

    def test_zeros_like(self):
        """Test zeros_like function."""
        original = xpm.array([[1, 2], [3, 4]], dtype=xpm.int32)
        arr = zeros_like(original)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.int32
        assert arr.count_masked() == 0
        assert not arr.mask.any()
        assert float(_to_numpy(arr.data).sum()) == 0.0

    def test_ones_like(self):
        """Test ones_like function."""
        original = xpm.array([[1.0, 2.0], [3.0, 4.0]], dtype=xpm.float32)
        arr = ones_like(original)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32
        assert arr.count_masked() == 0
        assert not arr.mask.any()
        assert float(_to_numpy(arr.data).sum()) == 4.0


class TestCountMasked:
    """Test count_masked function."""

    def test_count_masked_axis_none(self):
        """Test count_masked with axis=None (returns a numpy integer)."""
        data = xpm.array([1.0, 2.0, 3.0, 4.0], dtype=xpm.float32)
        mask = xpm.array([False, True, False, True], dtype=bool)
        arr = masked_array(data, mask)
        result = count_masked(arr, axis=None)
        assert isinstance(result, np.integer)
        assert result == 2

    def test_count_masked_axis_0(self):
        """Test count_masked along axis=0."""
        data = xpm.array([[1.0, 2.0], [3.0, 4.0]], dtype=xpm.float32)
        mask = xpm.array([[False, True], [True, False]], dtype=bool)
        arr = masked_array(data, mask)
        result = count_masked(arr, axis=0)
        # Should return an array with counts per column
        assert result.shape == (2,)
        np.testing.assert_array_equal(_to_numpy(result), [1, 1])

    def test_count_masked_no_mask(self):
        """Test count_masked with no mask."""
        data = xpm.array([1.0, 2.0, 3.0], dtype=xpm.float32)
        arr = masked_array(data)
        result = count_masked(arr, axis=None)
        assert result == 0


class TestCompressAndMask:
    """Test compress_nd, compress_rowcols, compress_rows, compress_cols, mask_rowcols, mask_rows, mask_cols."""

    def test_compress_rows(self):
        """Suppress rows that contain any masked value."""
        data = xpm.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=xpm.float32)
        mask = xpm.array([[True, False, False], [True, False, False], [False, False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_rows(arr)
        assert isinstance(out, xpm.ndarray)  # plain array on the data's device
        np_out = _to_numpy(out)
        assert np_out.shape == (1, 3)
        np.testing.assert_array_equal(np_out[0], [7, 8, 9])

    def test_compress_cols(self):
        """Suppress columns that contain any masked value."""
        data = xpm.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=xpm.float32)
        mask = xpm.array([[True, False, False], [True, False, False], [False, False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_cols(arr)
        np_out = _to_numpy(out)
        assert np_out.shape == (3, 2)
        np.testing.assert_array_equal(np_out[:, 0], [2, 5, 8])
        np.testing.assert_array_equal(np_out[:, 1], [3, 6, 9])

    def test_compress_rowcols_axis_none(self):
        """Suppress both rows and columns that contain masked values."""
        data = xpm.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=xpm.float32)
        mask = xpm.array([[True, False, False], [True, False, False], [False, False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_rowcols(arr, axis=None)
        np_out = _to_numpy(out)
        # Rows 0, 1 and column 0 contain masked values: only [[8, 9]] survives
        # (numpy.ma.compress_rowcols semantics).
        np_ref = np.ma.compress_rowcols(np.ma.array(_to_numpy(data), mask=_to_numpy(mask)), axis=None)
        assert np_out.shape == (1, 2)
        np.testing.assert_array_equal(np_out, np_ref)
        np.testing.assert_array_equal(np_out[0], [8, 9])

    def test_compress_rowcols_axis_0(self):
        """Suppress only rows (axis=0)."""
        data = xpm.array([[1, 2], [3, 4], [5, 6]], dtype=xpm.float32)
        mask = xpm.array([[True, True], [False, False], [False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_rowcols(arr, axis=0)
        np_out = _to_numpy(out)
        assert np_out.shape == (2, 2)
        np.testing.assert_array_equal(np_out, [[3, 4], [5, 6]])

    def test_compress_rowcols_axis_1(self):
        """Suppress only columns (axis=1)."""
        data = xpm.array([[1, 2, 3], [4, 5, 6]], dtype=xpm.float32)
        mask = xpm.array([[True, False, False], [True, False, False]], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_rowcols(arr, axis=1)
        np_out = _to_numpy(out)
        assert np_out.shape == (2, 2)
        np.testing.assert_array_equal(np_out, [[2, 3], [5, 6]])

    def test_compress_nd_1d(self):
        """compress_nd on 1D drops masked elements."""
        data = xpm.array([1, 2, 3, 4, 5], dtype=xpm.float32)
        mask = xpm.array([False, True, False, True, False], dtype=bool)
        arr = masked_array(data, mask)
        out = compress_nd(arr, axis=0)
        np_out = _to_numpy(out)
        np.testing.assert_array_equal(np_out, [1, 3, 5])

    def test_compress_nd_no_mask(self):
        """compress_nd with no mask returns data unchanged."""
        data = xpm.array([[1, 2], [3, 4]], dtype=xpm.float32)
        arr = masked_array(data)
        out = compress_nd(arr, axis=None)
        np_out = _to_numpy(out)
        np.testing.assert_array_equal(np_out, _to_numpy(data))

    def test_compress_rowcols_2d_only(self):
        """compress_rowcols raises for non-2D array."""
        data = xpm.array([1, 2, 3], dtype=xpm.float32)
        arr = masked_array(data)
        with pytest.raises(NotImplementedError, match="2D"):
            compress_rowcols(arr, axis=None)

    def test_mask_rowcols_in_place(self):
        """mask_rowcols masks entire rows and columns that contain masked values."""
        data = xpm.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=xpm.float32)
        mask = xpm.array([[False, False, False], [False, True, False], [False, False, False]], dtype=bool)
        arr = masked_array(data, mask=mask.copy())
        result = mask_rowcols(arr, axis=None)
        np_mask = _to_numpy(result.mask)
        # Row 1 and column 1 should be fully masked
        assert np_mask[1, :].all()
        assert np_mask[:, 1].all()
        assert np_mask[0, 0] == False and np_mask[0, 2] == False and np_mask[2, 0] == False and np_mask[2, 2] == False

    def test_mask_rows(self):
        """mask_rows masks only rows that contain masked values."""
        data = xpm.array([[1, 2], [3, 4], [5, 6]], dtype=xpm.float32)
        mask = xpm.array([[True, False], [False, False], [False, False]], dtype=bool)
        arr = masked_array(data, mask=mask.copy())
        result = mask_rows(arr)
        np_mask = _to_numpy(result.mask)
        assert np_mask[0, :].all()
        assert not np_mask[1, :].any() and not np_mask[2, :].any()

    def test_mask_cols(self):
        """mask_cols masks only columns that contain masked values."""
        data = xpm.array([[1, 2, 3], [4, 5, 6]], dtype=xpm.float32)
        mask = xpm.array([[True, False, False], [True, False, False]], dtype=bool)
        arr = masked_array(data, mask=mask.copy())
        result = mask_cols(arr)
        np_mask = _to_numpy(result.mask)
        assert np_mask[:, 0].all()
        assert not np_mask[:, 1].any() and not np_mask[:, 2].any()


class TestStackingAndShape:
    """Test atleast_1d/2d/3d, vstack, hstack, column_stack, dstack, stack, row_stack, hsplit."""

    def test_atleast_1d_single(self):
        """atleast_1d promotes scalar or 0d to 1d and preserves mask."""
        data = xpm.array(5.0, dtype=xpm.float32)
        arr = masked_array(data)
        out = atleast_1d(arr)
        assert out.shape == (1,)
        assert _value(out) == 5.0

    def test_atleast_1d_multiple(self):
        """atleast_1d with multiple inputs returns tuple of masked arrays."""
        a = masked_array(xpm.array(1.0))
        b = masked_array(xpm.array([2.0, 3.0]))
        out = atleast_1d(a, b)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == (1,)
        assert out[1].shape == (2,)

    def test_atleast_2d_single(self):
        """atleast_2d adds dimension."""
        arr = masked_array(xpm.array([1.0, 2.0, 3.0]))
        out = atleast_2d(arr)
        assert out.shape == (1, 3)

    def test_atleast_3d_single(self):
        """atleast_3d adds dimensions."""
        arr = masked_array(xpm.array([1.0, 2.0]))
        out = atleast_3d(arr)
        assert out.shape == (1, 2, 1)

    def test_vstack(self):
        """vstack stacks vertically and combines masks."""
        a = masked_array(xpm.array([1.0, 2.0]), mask=xpm.array([False, True]))
        b = masked_array(xpm.array([3.0, 4.0]), mask=xpm.array([True, False]))
        out = vstack([a, b])
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(_to_numpy(out.data), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(_to_numpy(out.mask), [[False, True], [True, False]])

    def test_hstack(self):
        """hstack stacks horizontally and combines masks."""
        a = masked_array(xpm.array([[1.0], [2.0]]))
        b = masked_array(xpm.array([[3.0], [4.0]]), mask=xpm.array([[True], [False]]))
        out = hstack([a, b])
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(_to_numpy(out.mask), [[False, True], [False, False]])

    def test_column_stack(self):
        """column_stack stacks 1D arrays as columns."""
        a = masked_array(xpm.array([1.0, 2.0]))
        b = masked_array(xpm.array([3.0, 4.0]), mask=xpm.array([False, True]))
        out = column_stack([a, b])
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(_to_numpy(out.data), [[1, 3], [2, 4]])

    def test_dstack(self):
        """dstack stacks along third axis."""
        a = masked_array(xpm.array([[1, 2], [3, 4]]))
        b = masked_array(xpm.array([[5, 6], [7, 8]]))
        out = dstack([a, b])
        assert out.shape == (2, 2, 2)
        assert out.data.shape == (2, 2, 2)

    def test_stack(self):
        """stack joins along a new axis."""
        a = masked_array(xpm.array([1.0, 2.0]))
        b = masked_array(xpm.array([3.0, 4.0]))
        out = stack([a, b], axis=0)
        assert out.shape == (2, 2)
        out1 = stack([a, b], axis=1)
        assert out1.shape == (2, 2)

    def test_row_stack_alias(self):
        """row_stack is alias for vstack."""
        a = masked_array(xpm.array([1.0, 2.0]))
        b = masked_array(xpm.array([3.0, 4.0]))
        v = vstack([a, b])
        r = row_stack([a, b])
        np.testing.assert_array_equal(_to_numpy(v.data), _to_numpy(r.data))
        np.testing.assert_array_equal(_to_numpy(v.mask), _to_numpy(r.mask))

    def test_hsplit(self):
        """hsplit splits horizontally and returns list of MaskedArrays."""
        data = xpm.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=xpm.float32)
        mask = xpm.array([[False, True, False, True], [False, False, True, False]], dtype=bool)
        arr = masked_array(data, mask)
        parts = hsplit(arr, 2)
        assert len(parts) == 2
        assert parts[0].shape == (2, 2)
        assert parts[1].shape == (2, 2)
        np.testing.assert_array_equal(_to_numpy(parts[0].data), [[1, 2], [5, 6]])
        np.testing.assert_array_equal(_to_numpy(parts[1].data), [[3, 4], [7, 8]])


class TestDiagflatEdiff1dMr:
    """Test diagflat, ediff1d, mr_."""

    def test_diagflat(self):
        """diagflat creates 2D array with flattened input on diagonal."""
        arr = masked_array(xpm.array([1.0, 2.0, 3.0]))
        out = diagflat(arr)
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(_to_numpy(out.data), np.diag([1, 2, 3]))
        assert not _to_numpy(out.mask).any()

    def test_diagflat_with_mask(self):
        """diagflat propagates mask."""
        arr = masked_array(xpm.array([1.0, 2.0, 3.0]), mask=xpm.array([False, True, False]))
        out = diagflat(arr)
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(_to_numpy(out.mask), np.diag([False, True, False]))

    def test_ediff1d_basic(self):
        """ediff1d first difference and mask where either neighbor masked."""
        arr = masked_array(xpm.array([1.0, 2.0, 4.0, 7.0]))
        out = ediff1d(arr)
        np.testing.assert_array_almost_equal(_to_numpy(out.data), [1.0, 2.0, 3.0])
        assert not _to_numpy(out.mask).any()

    def test_ediff1d_mask_propagation(self):
        """ediff1d masks output where either adjacent input is masked."""
        arr = masked_array(xpm.array([10.0, 11.0, 12.0]), mask=xpm.array([False, True, False]))
        out = ediff1d(arr)
        assert _to_numpy(out.mask).all()
        # Masked slots keep the data of the minuend like numpy.ma (11 and 12), not the difference.
        ref = np.ma.ediff1d(np.ma.array([10.0, 11.0, 12.0], mask=[False, True, False]))
        np.testing.assert_array_equal(_to_numpy(out.data), ref.data)
        np.testing.assert_array_equal(_to_numpy(out.mask), ref.mask)

    def test_ediff1d_to_end_to_begin(self):
        """ediff1d with to_end and to_begin extends and masks new elements as unmasked."""
        arr = masked_array(xpm.array([1.0, 2.0, 3.0]))
        out = ediff1d(arr, to_begin=xpm.array(0.0), to_end=xpm.array(99.0))
        np_out = _to_numpy(out.data)
        assert np_out[0] == 0.0
        assert np_out[-1] == 99.0
        assert out.size == 4

    def test_mr_getitem_tuple(self):
        """mr_[a, b] stacks arrays vertically."""
        a = masked_array(xpm.array([[1, 2], [3, 4]]))
        b = masked_array(xpm.array([[5, 6]]))
        out = mr_[(a, b)]
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(_to_numpy(out.data)[-1], [5, 6])

    def test_mr_getitem_single(self):
        """mr_[a] returns masked array view of a."""
        a = masked_array(xpm.array([1.0, 2.0, 3.0]))
        out = mr_[a]
        assert out.shape == a.shape
        np.testing.assert_array_equal(_to_numpy(out.data), _to_numpy(a.data))


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
        assert issequence(xpm.array([1, 2, 3])) == True

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
        data = xpm.array([1.0, 2.0, 3.0, 4.0], dtype=xpm.float32)
        mask = xpm.array([False, True, False, False], dtype=bool)
        return masked_array(data, mask)

    def test_scalar_return_types(self, test_data):
        """Test that functions return numpy scalars."""
        functions = [sum, mean, std, var, min, max]
        for func in functions:
            result = func(test_data, axis=None)
            assert isinstance(result, np.generic), f"{func.__name__} should return a numpy scalar"

    def test_1d_reduction_returns_numpy_scalar(self, test_data):
        """Test that reducing a 1D array returns a numpy scalar."""
        result = sum(test_data)
        assert isinstance(result, np.generic), "1D reduction should return a numpy scalar"

    def test_2d_reduction_returns_array(self):
        """Test that reducing 2D array along one axis returns array."""
        data = xpm.array([[1.0, 2.0], [3.0, 4.0]], dtype=xpm.float32)
        arr = masked_array(data)
        result = sum(arr, axis=0)
        # Should return array, not scalar
        assert isinstance(result, MaskedArray) and result.shape == (2,), \
            "2D reduction should return a masked array"

    def test_keepdims_preserves_dimensions(self):
        """Test that keepdims=True preserves dimensions."""
        data = xpm.array([[1.0, 2.0], [3.0, 4.0]], dtype=xpm.float32)
        arr = masked_array(data)
        result = sum(arr, axis=0, keepdims=True)
        assert isinstance(result, MaskedArray)
        assert result.shape == (1, 2)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Test functions with empty array."""
        data = xpm.array([], dtype=xpm.float32)
        arr = masked_array(data)
        # Some operations should handle empty arrays gracefully
        result = count_masked(arr, axis=None)
        assert _value(result) == 0

    def test_single_element(self):
        """Test functions with single element array."""
        data = xpm.array([42.0], dtype=xpm.float32)
        arr = masked_array(data)
        result = sum(arr, axis=None)
        assert isinstance(result, np.generic)
        assert result == 42.0

    def test_all_masked_statistics(self):
        """Test statistics when all values are masked."""
        data = xpm.array([1.0, 2.0, 3.0], dtype=xpm.float32)
        mask = xpm.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        
        # sum should return masked singleton
        result_sum = sum(arr, axis=None)
        assert result_sum is masked

        # mean should return masked singleton
        result_mean = mean(arr, axis=None)
        assert result_mean is masked

    def test_no_mask_statistics(self):
        """Test statistics with no mask."""
        data = xpm.array([1.0, 2.0, 3.0, 4.0], dtype=xpm.float32)
        arr = masked_array(data)

        result = sum(arr, axis=None)
        assert isinstance(result, np.generic)
        assert result == 10.0

    def test_float32_precision(self):
        """Test that float32 precision is maintained."""
        data = xpm.array([1.0, 2.0, 3.0], dtype=xpm.float32)
        arr = masked_array(data)
        result = mean(arr, axis=None)
        assert isinstance(result, np.float32)
        assert result == 2.0


class TestIntegration:
    """Test integration with NumPy and core module."""

    def test_roundtrip_numpy_masked_array(self):
        """Test roundtrip conversion with NumPy masked array."""
        np_data = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        xp_arr = masked_array(np_data)
        result = sum(xp_arr, axis=None)
        np_result = np.ma.sum(np_data, axis=None)
        assert abs(result - np_result) < 1e-6

    def test_consistency_with_class_methods(self):
        """Test that extras functions are consistent with class methods."""
        data = xpm.array([1.0, 2.0, 3.0, 4.0], dtype=xpm.float32)
        mask = xpm.array([False, True, False, False], dtype=bool)
        arr = masked_array(data, mask)

        # Compare extras functions with class methods (numpy scalars)
        assert sum(arr, axis=None) == arr.sum(axis=None)
        assert mean(arr, axis=None) == arr.mean(axis=None)
        assert std(arr, axis=None) == arr.std(axis=None)
        assert var(arr, axis=None) == arr.var(axis=None)
        assert min(arr, axis=None) == arr.min(axis=None)
        assert max(arr, axis=None) == arr.max(axis=None)

    def test_dtype_preservation(self):
        """Test that dtypes are preserved correctly."""
        data = xpm.array([1, 2, 3, 4], dtype=xpm.int32)
        arr = masked_array(data)
        result = sum(arr, axis=None, dtype=xpm.int32)
        assert result == 10


class TestPerformance:
    """Test performance-related aspects."""

    def test_large_array_performance(self):
        """Test that functions work efficiently with large arrays."""
        data = xpm.random.rand(1000, 1000).astype(xpm.float32)
        arr = masked_array(data)
        
        # Should complete quickly
        result = sum(arr, axis=None)
        assert isinstance(result, np.generic)
        assert result > 0

    def test_data_stays_on_device(self):
        """Axis reductions stay on the device of the data (cupy on GPU, numpy on CPU)."""
        data = xpm.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=xpm.float32)
        arr = masked_array(data)

        result = sum(arr, axis=0)
        assert isinstance(result.data, xpm.ndarray)
        assert isinstance(result.mask, xpm.ndarray) or result.mask is nomask

        # Reductions to a scalar are the only point where data reaches the host.
        assert isinstance(sum(arr, axis=None), np.generic)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

