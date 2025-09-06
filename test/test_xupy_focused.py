import pytest
import numpy as np
from typing import Any
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from xupy._core import masked_array, MaskedArray

# Skip all tests if CuPy is not available
pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# Helper functions
def _to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to NumPy array."""
    if hasattr(arr, "get"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _asmarray(cupy_arr: Any) -> "MaskedArray":
    """Convert a CuPy array to NumPy masked array."""
    return cupy_arr.asmarray()


class TestXupyMaskedArrayFocused:
    """Focused test suite for _XupyMaskedArray class with practical scenarios."""

    # Fixtures
    @pytest.fixture
    def simple_data(self) -> np.ndarray:
        """Return simple test data."""
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    @pytest.fixture
    def simple_mask(self) -> np.ndarray:
        """Return simple test mask."""
        return np.array([[False, False, True], [True, False, False]], dtype=bool)

    @pytest.fixture
    def masked_arr(self, simple_data: np.ndarray, simple_mask: np.ndarray):
        """Return a masked array with simple data and mask."""
        return masked_array(simple_data, simple_mask)

    @pytest.fixture
    def numpy_masked_arr(self, simple_data: np.ndarray, simple_mask: np.ndarray) -> np.ma.MaskedArray:
        """Return a NumPy masked array for comparison."""
        return np.ma.masked_array(simple_data, simple_mask)

    # Core Functionality Tests
    class TestCoreFunctionality:
        """Test core functionality that should work reliably."""

        def test_creation_and_basic_properties(self, masked_arr):
            """Test basic creation and properties."""
            assert masked_arr.shape == (2, 3)
            assert masked_arr.size == 6
            assert masked_arr.ndim == 2
            assert isinstance(masked_arr.data, cp.ndarray)
            assert isinstance(masked_arr._mask, cp.ndarray)

        def test_mask_access(self, masked_arr, simple_mask):
            """Test mask access."""
            np.testing.assert_array_equal(_to_numpy(masked_arr._mask), simple_mask)

        def test_data_access(self, masked_arr, simple_data):
            """Test data access."""
            np.testing.assert_array_equal(_to_numpy(masked_arr.data), simple_data)

        def test_count_masked_unmasked(self, masked_arr):
            """Test counting masked and unmasked elements."""
            assert masked_arr.count_masked() == 2  # Two True values in mask
            assert masked_arr.count_unmasked() == 4  # Six total - two masked

        def test_is_masked(self, masked_arr):
            """Test is_masked method."""
            assert masked_arr.is_masked() == True

            # Test with no masked values
            data = np.array([1, 2, 3])
            ma = masked_array(data)
            assert ma.is_masked() == False

        def test_copy(self, masked_arr):
            """Test copy method."""
            copied = masked_arr.copy()
            assert copied is not masked_arr
            np.testing.assert_array_equal(_to_numpy(copied.data), _to_numpy(masked_arr.data))
            np.testing.assert_array_equal(_to_numpy(copied._mask), _to_numpy(masked_arr._mask))

        def test_astype(self, masked_arr):
            """Test astype method."""
            # CuPy doesn't support casting parameter, so test basic conversion
            data = masked_arr
            casted = data.astype(cp.float64)
            assert casted.dtype == cp.float64

    # Array Manipulation Tests
    class TestArrayManipulation:
        """Test array manipulation methods."""

        def test_reshape(self, masked_arr):
            """Test reshape method."""
            reshaped = masked_arr.reshape(6)
            assert reshaped.shape == (6,)
            assert reshaped.count_masked() == 2

        def test_flatten(self, masked_arr):
            """Test flatten method."""
            flattened = masked_arr.flatten()
            assert flattened.shape == (6,)
            assert flattened.count_masked() == 2

        def test_transpose(self, masked_arr):
            """Test transpose method."""
            transposed = masked_arr.transpose()
            assert transposed.shape == (3, 2)
            assert transposed.count_masked() == 2

        def test_squeeze(self):
            """Test squeeze method."""
            data = np.array([[[1], [2]]])
            mask = np.array([[[True], [False]]])
            ma = masked_array(data, mask)
            squeezed = ma.squeeze()
            assert squeezed.shape == (2,)
            assert squeezed.count_masked() == 1

    # Statistical Operations Tests
    class TestStatisticalOperations:
        """Test statistical operations."""

        def test_mean(self):
            """Test mean calculation."""
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mask = np.array([False, False, True, False, False])
            ma = masked_array(data, mask)
            mean_val = ma.mean()
            expected = (1.0 + 2.0 + 4.0 + 5.0) / 4  # Exclude masked value
            assert abs(mean_val - expected) < 1e-6

        def test_sum(self):
            """Test sum calculation."""
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mask = np.array([False, False, True, False, False])
            ma = masked_array(data, mask)
            sum_val = ma.sum()
            expected = 1.0 + 2.0 + 4.0 + 5.0  # Exclude masked value
            assert abs(sum_val - expected) < 1e-6

        def test_std(self):
            """Test standard deviation."""
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mask = np.array([False, False, True, False, False])
            ma = masked_array(data, mask)
            std_val = ma.std()
            valid_data = np.array([1.0, 2.0, 4.0, 5.0])
            expected = np.std(valid_data)
            assert abs(std_val - expected) < 1e-6

        def test_min_max(self):
            """Test min and max."""
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mask = np.array([False, False, True, False, False])
            ma = masked_array(data, mask)
            assert ma.min() == 1.0
            assert ma.max() == 5.0

    # Universal Functions Tests
    class TestUniversalFunctions:
        """Test universal functions."""

        def test_sqrt(self):
            """Test square root function."""
            data = np.array([1.0, 4.0, 9.0, 16.0])
            ma = masked_array(data)
            result = ma.sqrt()
            expected = np.sqrt(data)
            np.testing.assert_array_almost_equal(_to_numpy(result.data), expected)

        def test_exp(self):
            """Test exponential function."""
            data = np.array([0.0, 1.0, 2.0])
            ma = masked_array(data)
            result = ma.exp()
            expected = np.exp(data)
            np.testing.assert_array_almost_equal(_to_numpy(result.data), expected)

        def test_log(self):
            """Test logarithm function."""
            data = np.array([1.0, 2.0, 3.0])
            ma = masked_array(data)
            result = ma.log()
            expected = np.log(data)
            np.testing.assert_array_almost_equal(_to_numpy(result.data), expected)

        def test_trigonometric_functions(self):
            """Test trigonometric functions."""
            data = np.array([0.0, np.pi/4, np.pi/2])
            ma = masked_array(data)
            sin_result = ma.sin()
            cos_result = ma.cos()

            np.testing.assert_array_almost_equal(_to_numpy(sin_result.data), np.sin(data))
            np.testing.assert_array_almost_equal(_to_numpy(cos_result.data), np.cos(data))

    # Logical Operations Tests
    class TestLogicalOperations:
        """Test logical operations."""

        def test_any(self):
            """Test any() method."""
            data = np.array([True, False, True])
            mask = np.array([False, False, True])
            ma = masked_array(data, mask, dtype=bool)
            result = ma.any()
            assert result == True  # True from first element

        def test_all(self):
            """Test all() method."""
            data = np.array([True, True, False])
            mask = np.array([False, False, True])
            ma = masked_array(data, mask, dtype=bool)
            result = ma.all()
            assert result == True  # False is masked, so all unmasked are True

    # Mask Information Tests
    class TestMaskInformation:
        """Test mask information methods."""

        def test_compressed(self):
            """Test compressed method."""
            data = np.array([1, 2, 3, 4, 5])
            mask = np.array([True, False, True, False, True])
            ma = masked_array(data, mask)
            compressed = ma.compressed()
            np.testing.assert_array_equal(_to_numpy(compressed), np.array([2, 4]))

        def test_fill_value_method(self):
            """Test fill_value method."""
            data = np.array([1, 2, 3])
            mask = np.array([True, False, False])
            ma = masked_array(data, mask)
            ma.fill_value(99)
            assert _to_numpy(ma.data)[0] == 99

    # Conversion Tests
    class TestConversions:
        """Test conversion methods."""

        def test_tolist(self, masked_arr):
            """Test tolist method."""
            result = masked_arr.tolist()
            expected = _to_numpy(masked_arr.data).tolist()
            assert result == expected

        def test_item_method(self, masked_arr):
            """Test item method."""
            assert masked_arr.item(0, 0) == _to_numpy(masked_arr.data)[0, 0]

        def test_asmarray_conversion(self, masked_arr, numpy_masked_arr):
            """Test conversion to numpy masked array."""
            np_ma = masked_arr.asmarray()
            assert isinstance(np_ma, np.ma.MaskedArray)
            np.testing.assert_array_equal(np_ma.data, numpy_masked_arr.data)
            np.testing.assert_array_equal(np_ma.mask, numpy_masked_arr.mask)

    # Indexing and Slicing Tests
    class TestIndexing:
        """Test indexing and slicing."""

        def test_basic_indexing(self, masked_arr):
            """Test basic indexing."""
            # Access unmasked element
            assert masked_arr[0, 0] == 1.0
            # Access masked element should return masked
            assert masked_arr[0, 2] is np.ma.masked or str(masked_arr[0, 2]) == '--'

        def test_slicing(self, masked_arr, simple_mask):
            """Test slicing."""
            sliced = masked_arr[0, :]
            np.testing.assert_array_equal(_to_numpy(sliced.data), _to_numpy(masked_arr.data)[0, :])
            np.testing.assert_array_equal(_to_numpy(sliced._mask), simple_mask[0, :])

    # Arithmetic Operations Tests
    class TestArithmeticOperations:
        """Test arithmetic operations."""

        def test_addition_with_same_type(self):
            """Test addition with another masked array."""
            a = masked_array(np.array([1, 2, 3]), np.array([True, False, False]))
            b = masked_array(np.array([4, 5, 6]), np.array([False, True, False]))
            result = a + b
            assert isinstance(result, MaskedArray)

        def test_multiplication_with_scalar(self, masked_arr):
            """Test multiplication with scalar."""
            result = masked_arr * 2
            expected = _asmarray(masked_arr) * 2
            # The result should preserve the original data where not masked
            # and the mask should remain the same
            np.testing.assert_array_equal(result.asmarray().data, expected.data)
            np.testing.assert_array_equal(result.asmarray().mask, expected.mask)

        def test_inplace_operations(self, masked_arr):
            """Test in-place operations."""
            original_data = _to_numpy(masked_arr.data).copy()
            masked_arr += 1
            expected = original_data + 1
            np.testing.assert_array_equal(_to_numpy(masked_arr.data), expected)

    # Matrix Operations Tests
    class TestMatrixOperations:
        """Test matrix operations."""

        def test_matmul(self):
            """Test matrix multiplication."""
            a = masked_array(np.array([[1, 2], [3, 4]]))
            b = masked_array(np.array([[5, 6], [7, 8]]))
            result = a @ b
            expected = np.array([[1, 2], [3, 4]]) @ np.array([[5, 6], [7, 8]])
            np.testing.assert_array_equal(_to_numpy(result.data), expected)

    # Comparison Operations Tests
    class TestComparisonOperations:
        """Test comparison operations."""

        def test_equality(self, masked_arr):
            """Test equality comparison."""
            other = masked_array(np.array([[1, 3, 3], [5, 5, 6]]))
            result = masked_arr == other
            # Result should be a boolean array
            assert result is not None

    # Edge Cases Tests
    class TestEdgeCases:
        """Test edge cases."""

        def test_empty_array(self):
            """Test empty array."""
            empty_data = np.array([], dtype=np.float32)
            ma = masked_array(empty_data)
            assert ma.size == 0
            assert ma.shape == (0,)

        def test_single_element(self):
            """Test single element array."""
            data = np.array([42.0])
            ma = masked_array(data)
            assert ma.shape == (1,)
            assert ma.item() == 42.0

        def test_scalar_input(self):
            """Test scalar input."""
            ma = masked_array(42.0)
            assert ma.item() == 42.0

        def test_all_masked(self):
            """Test array with all elements masked."""
            data = np.array([1, 2, 3])
            mask = np.array([True, True, True])
            ma = masked_array(data, mask)
            assert ma.count_masked() == 3
            assert ma.count_unmasked() == 0

        def test_no_mask(self):
            """Test array with no mask."""
            data = np.array([1, 2, 3])
            ma = masked_array(data)
            assert ma.count_masked() == 0
            assert ma.count_unmasked() == 3

    # GPU-Specific Tests
    class TestGPUFeatures:
        """Test GPU-specific features."""

        @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
        def test_gpu_backend(self):
            """Test that arrays are on GPU."""
            data = np.array([1, 2, 3])
            ma = masked_array(data)
            assert isinstance(ma.data, cp.ndarray)
            assert ma.data.device.id == cp.cuda.runtime.getDevice()

        @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
        def test_memory_efficiency(self):
            """Test basic memory operations."""
            data = cp.random.rand(1000, 1000, dtype=cp.float32)
            ma = masked_array(data)
            assert ma.shape == (1000, 1000)
            # Clean up
            del data, ma
            cp.get_default_memory_pool().free_all_blocks()

    # Integration Tests
    class TestIntegration:
        """Test integration with NumPy."""

        def test_numpy_compatibility(self, masked_arr):
            """Test compatibility with NumPy operations via asmarray."""
            np_ma = masked_arr.asmarray()
            assert isinstance(np_ma, np.ma.MaskedArray)

            # Test that NumPy operations work
            mean_val = np_ma.mean()
            assert isinstance(mean_val, (float, np.ma.MaskedArray))

        def test_roundtrip_conversion(self, masked_arr):
            """Test roundtrip conversion."""
            np_ma = masked_arr.asmarray()
            ma2 = masked_array(np_ma)
            np.testing.assert_array_equal(_to_numpy(ma2.data), _to_numpy(masked_arr.data))
            np.testing.assert_array_equal(_to_numpy(ma2._mask), _to_numpy(masked_arr._mask))

    # Performance Tests
    class TestPerformance:
        """Test performance aspects."""

        def test_lazy_operations(self):
            """Test that operations don't immediately transfer to CPU."""
            data = np.random.rand(100, 100)
            ma = masked_array(data)

            # Operations should not immediately transfer to CPU
            result = ma + 1
            assert isinstance(result, MaskedArray)

        def test_reasonable_performance(self):
            """Test that operations complete in reasonable time."""
            data = np.random.rand(1000, 1000)
            ma = masked_array(data)

            start_time = time.time()
            result = ma.sum()
            end_time = time.time()

            # Should complete in less than 1 second
            assert end_time - start_time < 1.0

    # Error Handling Tests
    class TestErrorHandling:
        """Test error handling."""

        def test_invalid_reshape(self, masked_arr):
            """Test invalid reshape."""
            with pytest.raises(ValueError):
                masked_arr.reshape(10)  # Invalid size

        def test_invalid_indexing(self, masked_arr):
            """Test invalid indexing."""
            with pytest.raises(IndexError):
                _ = masked_arr[10, 10]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
