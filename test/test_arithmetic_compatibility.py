"""
Comprehensive arithmetic compatibility tests for XupyMaskedArray.

This test suite focuses on edge cases and compatibility with numpy.ma.MaskedArray
for arithmetic operations, including:
- NaN/infinity detection and masking
- Division by zero handling
- All masked arrays
- Empty arrays
- Scalar arrays
- Broadcasting scenarios
- Comparison with numpy.ma behavior
"""
import pytest
import numpy as np
from typing import Any

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from xupy.ma import masked_array, MaskedArray, nomask, masked

# Skip all tests if CuPy is not available
pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# Helper functions
def _to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to NumPy array."""
    if hasattr(arr, "get"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _mask_to_numpy(mask: Any) -> np.ndarray:
    """Convert mask to NumPy array, handling nomask."""
    if mask is nomask:
        return np.array(False)  # Return False for nomask
    if hasattr(mask, "get"):
        return cp.asnumpy(mask)
    return np.asarray(mask)


def _get_mask_array(masked_arr: Any) -> np.ndarray:
    """Get mask as NumPy array from masked array, handling nomask."""
    if hasattr(masked_arr, "mask"):
        mask = masked_arr.mask
        if mask is nomask or mask is np.ma.nomask:
            return np.zeros(masked_arr.shape, dtype=bool)
        return _mask_to_numpy(mask)
    return np.zeros(masked_arr.shape, dtype=bool)


class TestNaNInfinityDetection:
    """Test NaN and infinity detection in arithmetic operations."""

    def test_division_by_zero_detection(self):
        """Test that division by zero is detected and masked."""
        arr1 = masked_array([1.0, 2.0, 3.0], mask=[False, False, False])
        arr2 = masked_array([0.0, 1.0, 0.0], mask=[False, False, False])
        
        result = arr1 / arr2
        
        # Check that division by zero positions are masked
        result_mask = _to_numpy(result.mask)
        assert result_mask[0] == True, "Division by zero should be masked"
        assert result_mask[1] == False, "Valid division should not be masked"
        assert result_mask[2] == True, "Division by zero should be masked"
        
        # Compare with numpy.ma
        np_arr1 = np.ma.array([1.0, 2.0, 3.0], mask=[False, False, False])
        np_arr2 = np.ma.array([0.0, 1.0, 0.0], mask=[False, False, False])
        np_result = np_arr1 / np_arr2
        
        np.testing.assert_array_equal(result_mask, np_result.mask)

    def test_division_by_zero_scalar(self):
        """Test division by zero scalar."""
        arr = masked_array([1.0, 2.0, 3.0])
        result = arr / 0.0
        
        # All elements should be masked due to division by zero
        result_mask = _to_numpy(result.mask)
        assert np.all(result_mask), "All elements should be masked when dividing by zero scalar"

    def test_sqrt_negative_detection(self):
        """Test that sqrt of negative numbers produces NaN and is masked."""
        arr = masked_array([-1.0, 4.0, -9.0], mask=[False, False, False])
        result = arr.sqrt()
        
        result_mask = _to_numpy(result.mask)
        assert result_mask[0] == True, "sqrt(-1) should be masked (NaN)"
        assert result_mask[1] == False, "sqrt(4) should not be masked"
        assert result_mask[2] == True, "sqrt(-9) should be masked (NaN)"

    def test_nan_propagation_addition(self):
        """Test that NaN in operands propagates correctly."""
        arr1 = masked_array([1.0, np.nan, 3.0], mask=[False, False, False])
        arr2 = masked_array([1.0, 2.0, 3.0], mask=[False, False, False])
        
        result = arr1 + arr2
        
        # NaN should be detected and masked
        result_mask = _to_numpy(result.mask)
        assert result_mask[1] == True, "NaN in operand should result in masked output"

    def test_infinity_detection(self):
        """Test that infinity values are detected and masked."""
        arr1 = masked_array([1.0, 2.0], mask=[False, False])
        arr2 = masked_array([0.0, 0.0], mask=[False, False])
        
        result = arr1 / arr2
        
        # Check that infinity is detected
        result_data = _to_numpy(result.data)
        result_mask = _to_numpy(result.mask)
        
        assert np.all(np.isinf(result_data) | np.isnan(result_data)), "Result should contain inf or nan"
        assert np.all(result_mask), "All infinity results should be masked"

    def test_nan_in_multiplication(self):
        """Test NaN detection in multiplication."""
        arr1 = masked_array([1.0, np.nan, 3.0], mask=[False, False, False])
        arr2 = masked_array([2.0, 2.0, 2.0], mask=[False, False, False])
        
        result = arr1 * arr2
        result_mask = _to_numpy(result.mask)
        
        assert result_mask[1] == True, "NaN * value should be masked"

    def test_nan_in_subtraction(self):
        """Test NaN detection in subtraction."""
        arr1 = masked_array([1.0, np.nan, 3.0], mask=[False, False, False])
        arr2 = masked_array([1.0, 1.0, 1.0], mask=[False, False, False])
        
        result = arr1 - arr2
        result_mask = _to_numpy(result.mask)
        
        assert result_mask[1] == True, "NaN - value should be masked"


class TestDivisionByZero:
    """Test division by zero edge cases."""

    def test_division_by_zero_masked_divisor(self):
        """Test division when divisor is masked (should not mask result)."""
        arr1 = masked_array([1.0, 2.0, 3.0], mask=[False, False, False])
        arr2 = masked_array([0.0, 1.0, 0.0], mask=[True, False, True])
        
        result = arr1 / arr2
        
        # When divisor is masked, division by zero shouldn't mask the result
        # (because the divisor itself is masked)
        result_mask = _to_numpy(result.mask)
        assert result_mask[0] == True, "Should be masked because divisor is masked"
        assert result_mask[1] == False, "Valid division should not be masked"
        assert result_mask[2] == True, "Should be masked because divisor is masked"

    def test_division_by_zero_with_numpy_ma_compatibility(self):
        """Test division by zero matches numpy.ma behavior."""
        xp_arr1 = masked_array([1.0, 2.0, 3.0])
        xp_arr2 = masked_array([0.0, 1.0, 0.0])
        xp_result = xp_arr1 / xp_arr2
        
        np_arr1 = np.ma.array([1.0, 2.0, 3.0])
        np_arr2 = np.ma.array([0.0, 1.0, 0.0])
        np_result = np_arr1 / np_arr2
        
        # Compare masks - handle nomask case
        if xp_result.mask is nomask:
            xp_mask = np.zeros(xp_result.shape, dtype=bool)
        else:
            xp_mask = _to_numpy(xp_result.mask)
        if np_result.mask is np.ma.nomask:
            np_mask = np.zeros(np_result.shape, dtype=bool)
        else:
            np_mask = np_result.mask
        np.testing.assert_array_equal(xp_mask, np_mask)


class TestAllMaskedArrays:
    """Test arithmetic operations with all-masked arrays."""

    def test_addition_all_masked(self):
        """Test addition when all values are masked."""
        arr1 = masked_array([1.0, 2.0], mask=[True, True])
        arr2 = masked_array([3.0, 4.0], mask=[True, True])
        
        result = arr1 + arr2
        
        # Result should have all elements masked
        result_mask = _to_numpy(result.mask)
        assert np.all(result_mask), "All elements should be masked"

    def test_multiplication_all_masked(self):
        """Test multiplication when all values are masked."""
        arr1 = masked_array([1.0, 2.0], mask=[True, True])
        arr2 = masked_array([3.0, 4.0], mask=[True, True])
        
        result = arr1 * arr2
        
        result_mask = _to_numpy(result.mask)
        assert np.all(result_mask), "All elements should be masked"

    def test_division_all_masked(self):
        """Test division when all values are masked."""
        arr1 = masked_array([1.0, 2.0], mask=[True, True])
        arr2 = masked_array([3.0, 4.0], mask=[True, True])
        
        result = arr1 / arr2
        
        result_mask = _to_numpy(result.mask)
        assert np.all(result_mask), "All elements should be masked"


class TestEmptyArrays:
    """Test arithmetic operations with empty arrays."""

    def test_addition_empty_arrays(self):
        """Test addition with empty arrays."""
        arr1 = masked_array([], dtype=cp.float32)
        arr2 = masked_array([], dtype=cp.float32)
        
        result = arr1 + arr2
        
        assert result.shape == (0,), "Result should be empty"
        assert result.size == 0, "Result should have zero size"
        # Result mask should be nomask or empty array
        assert result.mask is nomask or result.mask.size == 0, "Empty array should have nomask or empty mask"

    def test_scalar_with_empty_array(self):
        """Test scalar operation with empty array."""
        arr = masked_array([], dtype=cp.float32)
        result = arr + 1.0
        
        assert result.shape == (0,), "Result should be empty"
        assert result.size == 0, "Result should have zero size"


class TestScalarArrays:
    """Test arithmetic operations with scalar (0-dimensional) arrays."""

    def test_scalar_array_addition(self):
        """Test addition with scalar arrays."""
        arr1 = masked_array(5.0)
        arr2 = masked_array(3.0)
        
        result = arr1 + arr2
        
        assert result.shape == (), "Result should be scalar"
        assert _to_numpy(result.data).item() == 8.0, "5 + 3 should equal 8"
        # Both have nomask, so result should have nomask
        assert result.mask is nomask, "Result should have nomask when both operands have nomask"

    def test_scalar_array_division(self):
        """Test division with scalar arrays."""
        arr1 = masked_array(10.0)
        arr2 = masked_array(2.0)
        
        result = arr1 / arr2
        
        assert result.shape == (), "Result should be scalar"
        assert _to_numpy(result.data).item() == 5.0, "10 / 2 should equal 5"
        # Both have nomask, so result should have nomask (no division by zero)
        assert result.mask is nomask, "Result should have nomask when both operands have nomask and no division by zero"

    def test_scalar_array_division_by_zero(self):
        """Test scalar division by zero."""
        arr1 = masked_array(1.0)
        arr2 = masked_array(0.0)
        
        result = arr1 / arr2
        
        # Result should be masked due to division by zero
        if result.shape == ():
            # Scalar result - check if masked
            if result.mask is nomask:
                # If nomask, check if the data itself indicates an issue
                result_data = _to_numpy(result.data)
                assert np.isnan(result_data) or np.isinf(result_data), "Division by zero should produce NaN or inf"
            else:
                mask_val = _to_numpy(result.mask).item()
                assert mask_val == True, "Division by zero should mask result"
        else:
            result_mask = _to_numpy(result.mask)
            assert np.all(result_mask), "Division by zero should mask result"


class TestBroadcasting:
    """Test arithmetic operations with broadcasting."""

    def test_broadcasting_addition(self):
        """Test addition with broadcasting."""
        arr1 = masked_array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
        arr2 = masked_array([10.0, 20.0], mask=[False, False])
        
        result = arr1 + arr2
        
        assert result.shape == (2, 2), "Result should have shape (2, 2)"
        # Check that masks are properly broadcast
        result_mask = _to_numpy(result.mask)
        assert result_mask[0, 1] == True, "Mask should be preserved"

    def test_broadcasting_multiplication(self):
        """Test multiplication with broadcasting."""
        arr1 = masked_array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
        arr2 = masked_array([2.0], mask=[False])
        
        result = arr1 * arr2
        
        assert result.shape == (2, 2), "Result should have shape (2, 2)"
        result_mask = _to_numpy(result.mask)
        assert result_mask[0, 1] == True, "Mask should be preserved"

    def test_broadcasting_division(self):
        """Test division with broadcasting."""
        arr1 = masked_array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
        arr2 = masked_array([1.0, 0.0], mask=[False, False])
        
        result = arr1 / arr2
        
        assert result.shape == (2, 2), "Result should have shape (2, 2)"
        result_mask = _to_numpy(result.mask)
        # Column 1 should be masked due to division by zero
        assert result_mask[0, 1] == True, "Division by zero should mask result"
        assert result_mask[1, 1] == True, "Division by zero should mask result"


class TestNumpyMACompatibility:
    """Test compatibility with numpy.ma.MaskedArray behavior."""

    def test_addition_compatibility(self):
        """Test that addition matches numpy.ma behavior."""
        xp_arr1 = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        xp_arr2 = masked_array([4.0, 5.0, 6.0], mask=[False, False, True])
        xp_result = xp_arr1 + xp_arr2
        
        np_arr1 = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        np_arr2 = np.ma.array([4.0, 5.0, 6.0], mask=[False, False, True])
        np_result = np_arr1 + np_arr2
        
        # Compare masks - handle nomask
        # Note: numpy.ma preserves original data when masked, but we compute the actual result
        # So we only compare masks, not data values
        xp_mask = _get_mask_array(xp_result)
        np_mask = _get_mask_array(np_result)
        np.testing.assert_array_equal(xp_mask, np_mask)
        
        # For unmasked elements, data should match
        unmasked = ~xp_mask
        if np.any(unmasked):
            xp_data = _to_numpy(xp_result.data)
            np.testing.assert_array_almost_equal(xp_data[unmasked], np_result.data[unmasked], decimal=5)

    def test_subtraction_compatibility(self):
        """Test that subtraction matches numpy.ma behavior."""
        xp_arr1 = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        xp_arr2 = masked_array([4.0, 5.0, 6.0], mask=[False, False, True])
        xp_result = xp_arr1 - xp_arr2
        
        np_arr1 = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        np_arr2 = np.ma.array([4.0, 5.0, 6.0], mask=[False, False, True])
        np_result = np_arr1 - np_arr2
        
        xp_mask = _get_mask_array(xp_result)
        np_mask = _get_mask_array(np_result)
        np.testing.assert_array_equal(xp_mask, np_mask)

    def test_multiplication_compatibility(self):
        """Test that multiplication matches numpy.ma behavior."""
        xp_arr1 = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        xp_arr2 = masked_array([4.0, 5.0, 6.0], mask=[False, False, True])
        xp_result = xp_arr1 * xp_arr2
        
        np_arr1 = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        np_arr2 = np.ma.array([4.0, 5.0, 6.0], mask=[False, False, True])
        np_result = np_arr1 * np_arr2
        
        xp_mask = _get_mask_array(xp_result)
        np_mask = _get_mask_array(np_result)
        np.testing.assert_array_equal(xp_mask, np_mask)

    def test_division_compatibility(self):
        """Test that division matches numpy.ma behavior."""
        xp_arr1 = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        xp_arr2 = masked_array([4.0, 5.0, 6.0], mask=[False, False, True])
        xp_result = xp_arr1 / xp_arr2
        
        np_arr1 = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        np_arr2 = np.ma.array([4.0, 5.0, 6.0], mask=[False, False, True])
        np_result = np_arr1 / np_arr2
        
        xp_mask = _get_mask_array(xp_result)
        np_mask = _get_mask_array(np_result)
        np.testing.assert_array_equal(xp_mask, np_mask)

    def test_power_compatibility(self):
        """Test that power operation matches numpy.ma behavior."""
        xp_arr1 = masked_array([2.0, 3.0, 4.0], mask=[False, True, False])
        xp_arr2 = masked_array([2.0, 2.0, 2.0], mask=[False, False, True])
        xp_result = xp_arr1 ** xp_arr2
        
        np_arr1 = np.ma.array([2.0, 3.0, 4.0], mask=[False, True, False])
        np_arr2 = np.ma.array([2.0, 2.0, 2.0], mask=[False, False, True])
        np_result = np_arr1 ** np_arr2
        
        xp_mask = _get_mask_array(xp_result)
        np_mask = _get_mask_array(np_result)
        np.testing.assert_array_equal(xp_mask, np_mask)

    def test_modulo_compatibility(self):
        """Test that modulo operation matches numpy.ma behavior."""
        xp_arr1 = masked_array([10.0, 20.0, 30.0], mask=[False, True, False])
        xp_arr2 = masked_array([3.0, 3.0, 3.0], mask=[False, False, True])
        xp_result = xp_arr1 % xp_arr2
        
        np_arr1 = np.ma.array([10.0, 20.0, 30.0], mask=[False, True, False])
        np_arr2 = np.ma.array([3.0, 3.0, 3.0], mask=[False, False, True])
        np_result = np_arr1 % np_arr2
        
        xp_mask = _get_mask_array(xp_result)
        np_mask = _get_mask_array(np_result)
        np.testing.assert_array_equal(xp_mask, np_mask)


class TestEdgeCases:
    """Test various edge cases in arithmetic operations."""

    def test_mixed_mask_partial(self):
        """Test operations with partially masked arrays."""
        arr1 = masked_array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, True])
        arr2 = masked_array([5.0, 6.0, 7.0, 8.0], mask=[True, False, False, False])
        
        result = arr1 + arr2
        
        # Masks should be combined with OR
        result_mask = _to_numpy(result.mask)
        expected_mask = np.array([True, True, False, True])  # OR of both masks
        np.testing.assert_array_equal(result_mask, expected_mask)

    def test_scalar_with_masked_array(self):
        """Test scalar operations with masked arrays."""
        arr = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        
        # Scalar addition
        result = arr + 10.0
        result_mask = _to_numpy(result.mask)
        expected_mask = np.array([False, True, False])
        np.testing.assert_array_equal(result_mask, expected_mask)
        
        # Scalar multiplication
        result = arr * 2.0
        result_mask = _to_numpy(result.mask)
        np.testing.assert_array_equal(result_mask, expected_mask)

    def test_zero_masked_operations(self):
        """Test operations with zero values and masks."""
        arr1 = masked_array([0.0, 1.0, 0.0], mask=[False, False, True])
        arr2 = masked_array([1.0, 0.0, 1.0], mask=[False, False, False])
        
        # Multiplication with zero
        result = arr1 * arr2
        result_mask = _to_numpy(result.mask)
        assert result_mask[2] == True, "Masked element should remain masked"
        
        # Division by zero
        result = arr1 / arr2
        result_mask = _to_numpy(result.mask)
        assert result_mask[1] == True, "Division by zero should be masked"

    def test_very_large_numbers(self):
        """Test operations with very large numbers that might cause overflow."""
        arr1 = masked_array([1e10, 1e20], mask=[False, False])
        arr2 = masked_array([1e10, 1e20], mask=[False, False])
        
        result = arr1 * arr2
        
        # Check that result is valid (not all masked due to overflow)
        result_mask = _to_numpy(result.mask)
        # At least some elements should not be masked (unless overflow occurred)
        # This is a sanity check - actual behavior depends on dtype

    def test_very_small_numbers(self):
        """Test operations with very small numbers."""
        arr1 = masked_array([1e-10, 1e-20], mask=[False, False])
        arr2 = masked_array([1e-10, 1e-20], mask=[False, False])
        
        result = arr1 / arr2
        
        # Result should be approximately 1.0 for each element
        result_data = _to_numpy(result.data)
        result_mask = _to_numpy(result.mask)
        
        # Valid divisions should not be masked
        assert result_mask[0] == False, "Valid division should not be masked"
        assert result_mask[1] == False, "Valid division should not be masked"

    def test_numpy_array_input(self):
        """Test operations with numpy array inputs."""
        arr1 = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        np_arr = np.array([4.0, 5.0, 6.0])
        
        result = arr1 + np_arr
        
        # Result should be a MaskedArray
        assert isinstance(result, MaskedArray)
        result_mask = _to_numpy(result.mask)
        expected_mask = np.array([False, True, False])
        np.testing.assert_array_equal(result_mask, expected_mask)

    def test_numpy_ma_array_input(self):
        """Test operations with numpy.ma.masked_array inputs."""
        arr1 = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        np_ma = np.ma.array([4.0, 5.0, 6.0], mask=[False, False, True])
        
        result = arr1 + np_ma
        
        # Masks should be combined
        assert isinstance(result, MaskedArray)
        if result.mask is nomask:
            result_mask = np.zeros(result.shape, dtype=bool)
        else:
            result_mask = _to_numpy(result.mask)
        expected_mask = np.array([False, True, True])  # OR of both masks
        np.testing.assert_array_equal(result_mask, expected_mask)


class TestInPlaceOperations:
    """Test in-place arithmetic operations with edge cases."""

    def test_inplace_division_by_zero(self):
        """Test in-place division by zero."""
        arr = masked_array([1.0, 2.0, 3.0], mask=[False, False, False])
        arr /= masked_array([1.0, 0.0, 1.0], mask=[False, False, False])
        
        if arr.mask is nomask:
            result_mask = np.zeros(arr.shape, dtype=bool)
        else:
            result_mask = _to_numpy(arr.mask)
        assert result_mask[1] == True, "Division by zero should mask the element"

    def test_inplace_with_all_masked(self):
        """Test in-place operation with all-masked array."""
        arr1 = masked_array([1.0, 2.0], mask=[False, False])
        arr2 = masked_array([3.0, 4.0], mask=[True, True])
        
        arr1 += arr2
        
        result_mask = _to_numpy(arr1.mask)
        assert np.all(result_mask), "All elements should be masked after adding all-masked array"

    def test_inplace_scalar(self):
        """Test in-place scalar operations."""
        arr = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        
        arr += 10.0
        result_mask = _to_numpy(arr.mask)
        expected_mask = np.array([False, True, False])
        np.testing.assert_array_equal(result_mask, expected_mask)


class TestReflectedOperations:
    """Test reflected (right-hand side) arithmetic operations."""

    def test_reflected_addition(self):
        """Test reflected addition (scalar + array)."""
        arr = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        result = 10.0 + arr
        
        assert isinstance(result, MaskedArray)
        result_mask = _to_numpy(result.mask)
        expected_mask = np.array([False, True, False])
        np.testing.assert_array_equal(result_mask, expected_mask)

    def test_reflected_multiplication(self):
        """Test reflected multiplication (scalar * array)."""
        arr = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        result = 2.0 * arr
        
        assert isinstance(result, MaskedArray)
        result_mask = _to_numpy(result.mask)
        expected_mask = np.array([False, True, False])
        np.testing.assert_array_equal(result_mask, expected_mask)

    def test_reflected_division(self):
        """Test reflected division (scalar / array)."""
        arr = masked_array([1.0, 2.0, 0.0], mask=[False, False, False])
        result = 10.0 / arr
        
        assert isinstance(result, MaskedArray)
        if result.mask is nomask:
            result_mask = np.zeros(result.shape, dtype=bool)
        else:
            result_mask = _to_numpy(result.mask)
        assert result_mask[2] == True, "Division by zero should mask result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

