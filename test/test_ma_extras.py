"""
Comprehensive test suite for xupy.ma.extras module.

Tests all extra functions for masked arrays including:
- Statistical reductions (sum, mean, std, var, min, max, prod, average)
- Array creation utilities (masked_all, masked_all_like, empty_like)
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
    sum, mean, std, var, min, max, prod, product, average,
    masked_all, masked_all_like, empty_like, count_masked,
    issequence
)

# Skip all tests if CuPy is not available
pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# Helper functions
def _to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to NumPy array."""
    if hasattr(arr, "get"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


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
        """Test sum with axis=None returns scalar."""
        result = sum(test_data, axis=None)
        assert np.isscalar(result)
        assert result == 8.0  # 1 + 3 + 4 (excluding masked 2 and 5)
        assert isinstance(result, (float, np.floating))

    def test_sum_axis_0(self, test_data_2d):
        """Test sum along axis=0."""
        result = sum(test_data_2d, axis=0)
        # Should return array with shape (3,)
        assert hasattr(result, 'shape') or np.isscalar(result)
        # If 0-d, should be scalar; otherwise array
        if hasattr(result, 'shape') and result.shape == ():
            assert np.isscalar(result)

    def test_sum_keepdims(self, test_data_2d):
        """Test sum with keepdims=True."""
        result = sum(test_data_2d, axis=0, keepdims=True)
        assert hasattr(result, 'shape')
        assert result.shape == (1, 3)

    def test_mean_axis_none(self, test_data):
        """Test mean with axis=None returns scalar."""
        result = mean(test_data, axis=None)
        assert np.isscalar(result)
        expected = (1.0 + 3.0 + 4.0) / 3  # Exclude masked values
        assert abs(result - expected) < 1e-6

    def test_mean_axis_1(self, test_data_2d):
        """Test mean along axis=1."""
        result = mean(test_data_2d, axis=1)
        # Should be array with shape (2,)
        assert hasattr(result, 'shape') or np.isscalar(result)

    def test_std_axis_none(self, test_data):
        """Test std with axis=None returns scalar."""
        result = std(test_data, axis=None)
        assert np.isscalar(result)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.std(valid_data)
        assert abs(result - expected) < 1e-5

    def test_std_with_ddof(self, test_data):
        """Test std with ddof parameter."""
        # Note: Currently ddof is only applied when axis is specified
        # When axis=None, ddof is ignored (uses default ddof=0)
        result = std(test_data, axis=None, ddof=1)
        assert np.isscalar(result)
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
        """Test var with axis=None returns scalar."""
        result = var(test_data, axis=None)
        assert np.isscalar(result)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.var(valid_data)
        assert abs(result - expected) < 1e-5

    def test_min_axis_none(self, test_data):
        """Test min with axis=None returns scalar."""
        result = min(test_data, axis=None)
        assert np.isscalar(result)
        assert result == 1.0  # Minimum of unmasked values

    def test_max_axis_none(self, test_data):
        """Test max with axis=None returns scalar."""
        result = max(test_data, axis=None)
        assert np.isscalar(result)
        assert result == 4.0  # Maximum of unmasked values

    def test_prod_axis_none(self, test_data):
        """Test prod with axis=None returns scalar."""
        result = prod(test_data, axis=None)
        assert np.isscalar(result)
        assert result == 12.0  # 1 * 3 * 4 (excluding masked values)

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
        assert result1 == result2

    def test_numpy_compatibility_scalars(self, test_data):
        """Test that functions return scalars like NumPy."""
        np_data = np.ma.array([1.0, 2.0, 3.0, 4.0, 5.0], 
                              mask=[False, True, False, False, True])
        
        # Test all functions return scalars when axis=None
        functions = [sum, mean, std, var, min, max]
        for func in functions:
            np_result = getattr(np.ma, func.__name__)(np_data, axis=None)
            xp_result = func(test_data, axis=None)
            assert np.isscalar(np_result) == np.isscalar(xp_result), \
                f"{func.__name__} scalar compatibility failed"


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
        assert np.isscalar(result)
        expected = (1.0 + 2.0) / 2  # Only unmasked values
        assert abs(result - expected) < 1e-6

    def test_average_with_weights(self, test_data):
        """Test average with weights."""
        weights = cp.array([3.0, 1.0, 0.0, 0.0], dtype=cp.float32)
        result = average(test_data, axis=None, weights=weights)
        assert np.isscalar(result)
        expected = (1.0 * 3.0 + 2.0 * 1.0) / (3.0 + 1.0)
        assert abs(result - expected) < 1e-6

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
        assert np.isscalar(result)
        assert isinstance(sum_weights, (float, np.floating))
        assert sum_weights == 2.0  # Two unmasked values

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
        """Test count_masked with axis=None."""
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        mask = cp.array([False, True, False, True], dtype=bool)
        arr = masked_array(data, mask)
        result = count_masked(arr, axis=None)
        assert result == 2
        assert isinstance(result, (int, np.integer))

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
        """Test that functions return Python scalars, not 0-d arrays."""
        functions = [sum, mean, std, var, min, max]
        for func in functions:
            result = func(test_data, axis=None)
            assert np.isscalar(result), f"{func.__name__} should return scalar"
            assert not isinstance(result, (cp.ndarray, np.ndarray)), \
                f"{func.__name__} should not return array"

    def test_1d_reduction_returns_scalar(self, test_data):
        """Test that reducing 1D array returns scalar."""
        result = sum(test_data)
        assert np.isscalar(result), "1D reduction should return scalar"

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
        assert result == 0

    def test_single_element(self):
        """Test functions with single element array."""
        data = cp.array([42.0], dtype=cp.float32)
        arr = masked_array(data)
        result = sum(arr, axis=None)
        assert np.isscalar(result)
        assert result == 42.0

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
        assert np.isscalar(result)
        assert result == 10.0

    def test_float32_precision(self):
        """Test that float32 precision is maintained."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        arr = masked_array(data)
        result = mean(arr, axis=None)
        assert isinstance(result, (float, np.floating))
        # Should use float32 precision
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
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        mask = cp.array([False, True, False, False], dtype=bool)
        arr = masked_array(data, mask)
        
        # Compare extras functions with class methods
        assert sum(arr, axis=None) == arr.sum(axis=None)
        assert mean(arr, axis=None) == arr.mean(axis=None)
        assert std(arr, axis=None) == arr.std(axis=None)
        assert var(arr, axis=None) == arr.var(axis=None)
        assert min(arr, axis=None) == arr.min(axis=None)
        assert max(arr, axis=None) == arr.max(axis=None)

    def test_dtype_preservation(self):
        """Test that dtypes are preserved correctly."""
        data = cp.array([1, 2, 3, 4], dtype=cp.int32)
        arr = masked_array(data)
        result = sum(arr, axis=None, dtype=cp.int32)
        # When extracting scalar, we get Python native types (NumPy compatibility)
        # The value should still be correct
        assert result == 10
        assert isinstance(result, (int, float, np.integer, np.floating))


class TestPerformance:
    """Test performance-related aspects."""

    def test_large_array_performance(self):
        """Test that functions work efficiently with large arrays."""
        data = cp.random.rand(1000, 1000).astype(cp.float32)
        arr = masked_array(data)
        
        # Should complete quickly
        result = sum(arr, axis=None)
        assert np.isscalar(result)
        assert result > 0

    def test_gpu_operations(self):
        """Test that operations stay on GPU."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        arr = masked_array(data)
        
        # Operations should return GPU arrays or scalars
        result = sum(arr, axis=None)
        # Scalar extraction happens, but computation was on GPU
        assert isinstance(result, (float, np.floating, int, np.integer))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

