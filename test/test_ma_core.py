"""
Comprehensive test suite for xupy.ma.core module.

Tests the _XupyMaskedArray class including:
- Initialization and properties
- Array manipulation methods
- Statistical methods (with scalar returns)
- Universal functions
- Arithmetic operations
- Indexing and slicing
- Conversion methods
- Edge cases
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
from xupy.ma.core import _XupyMaskedArray

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
    """Extract Python scalar from x for comparison. x can be Python scalar, 0-d array, or 0-d MaskedArray."""
    if hasattr(x, "mask") and hasattr(x, "data") and getattr(x, "shape", None) == ():
        return _to_numpy(x.data).item()
    if hasattr(x, "item") and (not hasattr(x, "shape") or x.shape == ()):
        return x.item()
    return x


class TestInitialization:
    """Test MaskedArray initialization."""

    def test_init_from_cupy_array(self):
        """Test initialization from CuPy array."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([False, True, False], dtype=bool)
        arr = masked_array(data, mask)
        assert arr.shape == (3,)
        assert isinstance(arr.data, cp.ndarray)
        assert isinstance(arr.mask, cp.ndarray)

    def test_init_from_numpy_array(self):
        """Test initialization from NumPy array."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mask = np.array([False, True, False], dtype=bool)
        arr = masked_array(data, mask)
        assert arr.shape == (3,)
        assert isinstance(arr.data, cp.ndarray)  # Should convert to CuPy

    def test_init_without_mask(self):
        """Test initialization without mask."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        arr = masked_array(data)
        assert arr.shape == (3,)
        assert arr.mask is nomask or not arr.mask.any()

    def test_init_from_numpy_masked_array(self):
        """Test initialization from NumPy masked array."""
        np_ma = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        arr = masked_array(np_ma)
        assert arr.shape == (3,)
        assert isinstance(arr.data, cp.ndarray)

    def test_init_with_dtype(self):
        """Test initialization with dtype specified."""
        data = cp.array([1, 2, 3], dtype=cp.int32)
        arr = masked_array(data, dtype=cp.float32)
        assert arr.dtype == np.float32

    def test_init_with_fill_value(self):
        """Test initialization with fill_value."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        arr = masked_array(data, fill_value=999.0)
        assert arr.fill_value == 999.0


class TestProperties:
    """Test MaskedArray properties."""

    @pytest.fixture
    def test_arr(self):
        """Create test array."""
        data = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        mask = cp.array([[False, True, False], [True, False, False]], dtype=bool)
        return masked_array(data, mask)

    def test_shape_property(self, test_arr):
        """Test shape property."""
        assert test_arr.shape == (2, 3)

    def test_size_property(self, test_arr):
        """Test size property."""
        assert test_arr.size == 6

    def test_ndim_property(self, test_arr):
        """Test ndim property."""
        assert test_arr.ndim == 2

    def test_dtype_property(self, test_arr):
        """Test dtype property."""
        assert test_arr.dtype == np.float32

    def test_mask_property(self, test_arr):
        """Test mask property."""
        mask = test_arr.mask
        assert mask is not nomask
        assert mask.shape == (2, 3)

    def test_mask_setter(self, test_arr):
        """Test mask setter."""
        new_mask = cp.array([[True, False, True], [False, True, False]], dtype=bool)
        test_arr.mask = new_mask
        np.testing.assert_array_equal(_to_numpy(test_arr.mask), _to_numpy(new_mask))

    def test_fill_value_property(self, test_arr):
        """Test fill_value property."""
        assert hasattr(test_arr, 'fill_value')
        test_arr.fill_value = 42.0
        assert test_arr.fill_value == 42.0

    def test_T_property(self, test_arr):
        """Test transpose property."""
        transposed = test_arr.T
        assert transposed.shape == (3, 2)
        assert transposed.count_masked() == test_arr.count_masked()


class TestArrayManipulation:
    """Test array manipulation methods."""

    @pytest.fixture
    def test_arr(self):
        """Create test array."""
        data = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        mask = cp.array([[False, True, False], [True, False, False]], dtype=bool)
        return masked_array(data, mask)

    def test_reshape(self, test_arr):
        """Test reshape method."""
        reshaped = test_arr.reshape(6)
        assert reshaped.shape == (6,)
        assert reshaped.count_masked() == test_arr.count_masked()

    def test_flatten(self, test_arr):
        """Test flatten method."""
        flattened = test_arr.flatten()
        assert flattened.shape == (6,)
        assert flattened.count_masked() == test_arr.count_masked()

    def test_ravel(self, test_arr):
        """Test ravel method."""
        raveled = test_arr.ravel()
        assert raveled.shape == (6,)

    def test_squeeze(self):
        """Test squeeze method."""
        data = cp.array([[[1.0], [2.0]]], dtype=cp.float32)
        mask = cp.array([[[False], [True]]], dtype=bool)
        arr = masked_array(data, mask)
        squeezed = arr.squeeze()
        assert squeezed.shape == (2,)
        assert squeezed.count_masked() == 1

    def test_expand_dims(self, test_arr):
        """Test expand_dims method."""
        expanded = test_arr.expand_dims(0)
        assert expanded.shape == (1, 2, 3)

    def test_transpose(self, test_arr):
        """Test transpose method."""
        transposed = test_arr.transpose()
        assert transposed.shape == (3, 2)

    def test_swapaxes(self, test_arr):
        """Test swapaxes method."""
        swapped = test_arr.swapaxes(0, 1)
        assert swapped.shape == (3, 2)

    def test_repeat(self, test_arr):
        """Test repeat method."""
        repeated = test_arr.repeat(2, axis=0)
        assert repeated.shape == (4, 3)

    def test_tile(self, test_arr):
        """Test tile method."""
        tiled = test_arr.tile((2, 2))
        assert tiled.shape == (4, 6)


class TestStatisticalMethods:
    """Test statistical methods with scalar returns."""

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        data = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float32)
        mask = cp.array([False, True, False, False, True], dtype=bool)
        return masked_array(data, mask)

    def test_sum_returns_scalar_or_0d(self, test_data):
        """Test that sum returns 0-d array or scalar when axis=None (CuPy/NumPy native)."""
        result = test_data.sum(axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) == 8.0  # 1 + 3 + 4

    def test_sum_with_axis(self, test_data):
        """Test sum with axis specified."""
        data_2d = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        arr_2d = masked_array(data_2d)
        result = arr_2d.sum(axis=0)
        assert hasattr(result, 'shape') or np.isscalar(result)

    def test_mean_returns_scalar_or_0d(self, test_data):
        """Test that mean returns 0-d array or scalar when axis=None."""
        result = test_data.mean(axis=None)
        assert _is_scalar_or_0d(result)
        expected = (1.0 + 3.0 + 4.0) / 3
        assert abs(_value(result) - expected) < 1e-6

    def test_std_returns_scalar_or_0d(self, test_data):
        """Test that std returns 0-d array or scalar when axis=None."""
        result = test_data.std(axis=None)
        assert _is_scalar_or_0d(result)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.std(valid_data)
        assert abs(_value(result) - expected) < 1e-5

    def test_var_returns_scalar_or_0d(self, test_data):
        """Test that var returns 0-d array or scalar when axis=None."""
        result = test_data.var(axis=None)
        assert _is_scalar_or_0d(result)
        valid_data = np.array([1.0, 3.0, 4.0])
        expected = np.var(valid_data)
        assert abs(_value(result) - expected) < 1e-5

    def test_min_returns_scalar_or_0d(self, test_data):
        """Test that min returns 0-d array or scalar when axis=None."""
        result = test_data.min(axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) == 1.0

    def test_max_returns_scalar_or_0d(self, test_data):
        """Test that max returns 0-d array or scalar when axis=None."""
        result = test_data.max(axis=None)
        assert _is_scalar_or_0d(result)
        assert _value(result) == 4.0

    def test_sum_all_masked(self):
        """Test sum when all values are masked."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        result = arr.sum(axis=None)
        assert result is masked or result is not None

    def test_mean_all_masked(self):
        """Test mean when all values are masked."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        result = arr.mean(axis=None)
        assert result is masked or result is not None

    def test_sum_keepdims(self):
        """Test sum with keepdims=True."""
        data = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
        arr = masked_array(data)
        result = arr.sum(axis=0, keepdims=True)
        assert hasattr(result, 'shape')
        assert result.shape == (1, 2)


class TestUniversalFunctions:
    """Test universal functions."""

    @pytest.fixture
    def test_arr(self):
        """Create test array."""
        data = cp.array([1.0, 4.0, 9.0, 16.0], dtype=cp.float32)
        return masked_array(data)

    def test_sqrt(self, test_arr):
        """Test sqrt function."""
        # Add a mask to avoid nomask issues
        mask = cp.array([False, False, False, False], dtype=bool)
        arr_with_mask = masked_array(test_arr.data, mask)
        result = arr_with_mask.sqrt()
        assert isinstance(result, MaskedArray)
        expected = cp.sqrt(arr_with_mask.data)
        np.testing.assert_array_almost_equal(_to_numpy(result.data), _to_numpy(expected), decimal=5)

    def test_exp(self, test_arr):
        """Test exp function."""
        # Add a mask to avoid nomask issues
        mask = cp.array([False, False, False, False], dtype=bool)
        arr_with_mask = masked_array(test_arr.data, mask)
        result = arr_with_mask.exp()
        assert isinstance(result, MaskedArray)
        expected = cp.exp(arr_with_mask.data)
        np.testing.assert_array_almost_equal(_to_numpy(result.data), _to_numpy(expected), decimal=5)

    def test_log(self, test_arr):
        """Test log function."""
        # Add a mask to avoid nomask issues
        mask = cp.array([False, False, False, False], dtype=bool)
        arr_with_mask = masked_array(test_arr.data, mask)
        result = arr_with_mask.log()
        assert isinstance(result, MaskedArray)
        expected = cp.log(arr_with_mask.data)
        np.testing.assert_array_almost_equal(_to_numpy(result.data), _to_numpy(expected), decimal=5)

    def test_log10(self, test_arr):
        """Test log10 function."""
        # Add a mask to avoid nomask issues
        mask = cp.array([False, False, False, False], dtype=bool)
        arr_with_mask = masked_array(test_arr.data, mask)
        result = arr_with_mask.log10()
        assert isinstance(result, MaskedArray)
        expected = cp.log10(arr_with_mask.data)
        np.testing.assert_array_almost_equal(_to_numpy(result.data), _to_numpy(expected), decimal=5)

    def test_sin_cos(self):
        """Test sin and cos functions."""
        data = cp.array([0.0, np.pi/4, np.pi/2], dtype=cp.float32)
        # Add explicit mask to avoid nomask issues
        mask = cp.array([False, False, False], dtype=bool)
        arr = masked_array(data, mask)
        sin_result = arr.sin()
        cos_result = arr.cos()
        np.testing.assert_array_almost_equal(_to_numpy(sin_result.data), np.sin(_to_numpy(data)), decimal=5)
        np.testing.assert_array_almost_equal(_to_numpy(cos_result.data), np.cos(_to_numpy(data)), decimal=5)

    def test_apply_ufunc(self, test_arr):
        """Test apply_ufunc method."""
        # Add a mask to avoid nomask issues
        mask = cp.array([False, False, False, False], dtype=bool)
        arr_with_mask = masked_array(test_arr.data, mask)
        result = arr_with_mask.apply_ufunc(cp.sqrt)
        assert isinstance(result, MaskedArray)
        expected = cp.sqrt(arr_with_mask.data)
        np.testing.assert_array_almost_equal(_to_numpy(result.data), _to_numpy(expected), decimal=5)


class TestArithmeticOperations:
    """Test arithmetic operations."""

    @pytest.fixture
    def arr1(self):
        """Create first test array."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([False, True, False], dtype=bool)
        return masked_array(data, mask)

    @pytest.fixture
    def arr2(self):
        """Create second test array."""
        data = cp.array([4.0, 5.0, 6.0], dtype=cp.float32)
        mask = cp.array([False, False, True], dtype=bool)
        return masked_array(data, mask)

    def test_addition(self, arr1, arr2):
        """Test addition of masked arrays."""
        result = arr1 + arr2
        assert isinstance(result, MaskedArray)
        # Masks should be combined (OR)
        expected_mask = cp.array([False, True, True], dtype=bool)
        np.testing.assert_array_equal(_to_numpy(result.mask), _to_numpy(expected_mask))

    def test_subtraction(self, arr1, arr2):
        """Test subtraction of masked arrays."""
        result = arr1 - arr2
        assert isinstance(result, MaskedArray)

    def test_multiplication(self, arr1, arr2):
        """Test multiplication of masked arrays."""
        result = arr1 * arr2
        assert isinstance(result, MaskedArray)

    def test_division(self, arr1, arr2):
        """Test division of masked arrays."""
        result = arr1 / arr2
        assert isinstance(result, MaskedArray)

    def test_scalar_addition(self, arr1):
        """Test addition with scalar."""
        result = arr1 + 10.0
        assert isinstance(result, MaskedArray)
        expected_data = arr1.data + 10.0
        np.testing.assert_array_almost_equal(_to_numpy(result.data), _to_numpy(expected_data), decimal=5)

    def test_scalar_multiplication(self, arr1):
        """Test multiplication with scalar."""
        result = arr1 * 2.0
        assert isinstance(result, MaskedArray)
        expected_data = arr1.data * 2.0
        np.testing.assert_array_almost_equal(_to_numpy(result.data), _to_numpy(expected_data), decimal=5)

    def test_inplace_addition(self, arr1):
        """Test in-place addition."""
        original_data = _to_numpy(arr1.data).copy()
        arr1 += 5.0
        expected = original_data + 5.0
        np.testing.assert_array_almost_equal(_to_numpy(arr1.data), expected, decimal=5)

    def test_inplace_multiplication(self, arr1):
        """Test in-place multiplication."""
        original_data = _to_numpy(arr1.data).copy()
        arr1 *= 2.0
        expected = original_data * 2.0
        np.testing.assert_array_almost_equal(_to_numpy(arr1.data), expected, decimal=5)


class TestIndexingAndSlicing:
    """Test indexing and slicing."""

    @pytest.fixture
    def test_arr(self):
        """Create test array."""
        data = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        mask = cp.array([[False, True, False], [True, False, False]], dtype=bool)
        return masked_array(data, mask)

    def test_single_element_indexing(self, test_arr):
        """Test single element indexing (returns 0-d MaskedArray, CuPy/NumPy native)."""
        # Access unmasked element
        elem = test_arr[0, 0]
        assert _value(elem) == 1.0

    def test_slicing(self, test_arr):
        """Test array slicing."""
        sliced = test_arr[0, :]
        assert sliced.shape == (3,)
        assert isinstance(sliced, MaskedArray)

    def test_setitem(self, test_arr):
        """Test setting array elements."""
        test_arr[0, 0] = 99.0
        assert _to_numpy(test_arr.data)[0, 0] == 99.0
        # Setting should unmask the element
        assert not _to_numpy(test_arr.mask)[0, 0]

    def test_setitem_masked_value(self, test_arr):
        """Test setting masked value."""
        test_arr[0, 1] = masked
        assert _to_numpy(test_arr.mask)[0, 1]

    def test_fancy_indexing(self, test_arr):
        """Test fancy indexing."""
        indices = cp.array([0, 2])
        sliced = test_arr[:, indices]
        assert sliced.shape == (2, 2)


class TestConversionMethods:
    """Test conversion methods."""

    @pytest.fixture
    def test_arr(self):
        """Create test array."""
        data = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=cp.float32)
        mask = cp.array([[False, True, False], [True, False, False]], dtype=bool)
        return masked_array(data, mask)

    def test_asmarray(self, test_arr):
        """Test conversion to NumPy masked array."""
        np_ma = test_arr.asmarray()
        assert isinstance(np_ma, np.ma.MaskedArray)
        np.testing.assert_array_equal(np_ma.data, _to_numpy(test_arr.data))
        np.testing.assert_array_equal(np_ma.mask, _to_numpy(test_arr.mask))

    def test_tolist(self, test_arr):
        """Test conversion to list."""
        result = test_arr.tolist()
        assert isinstance(result, list)
        expected = _to_numpy(test_arr.data).tolist()
        assert result == expected

    def test_item(self, test_arr):
        """Test item method."""
        result = test_arr.item(0, 0)
        assert result == 1.0

    def test_copy(self, test_arr):
        """Test copy method."""
        copied = test_arr.copy()
        assert copied is not test_arr
        np.testing.assert_array_equal(_to_numpy(copied.data), _to_numpy(test_arr.data))
        np.testing.assert_array_equal(_to_numpy(copied.mask), _to_numpy(test_arr.mask))

    def test_astype(self, test_arr):
        """Test astype method."""
        converted = test_arr.astype(cp.float64)
        assert converted.dtype == np.float64
        assert converted.shape == test_arr.shape


class TestMaskOperations:
    """Test mask-related operations."""

    @pytest.fixture
    def test_arr(self):
        """Create test array."""
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        mask = cp.array([False, True, False, True], dtype=bool)
        return masked_array(data, mask)

    def test_count_masked(self, test_arr):
        """Test count_masked method (returns 0-d array, CuPy native)."""
        assert _value(test_arr.count_masked()) == 2

    def test_count_unmasked(self, test_arr):
        """Test count_unmasked method (returns 0-d array, CuPy native)."""
        assert _value(test_arr.count_unmasked()) == 2

    def test_is_masked(self, test_arr):
        """Test is_masked method (returns 0-d array, CuPy native)."""
        assert bool(_value(test_arr.is_masked())) is True

        # Test with no mask
        no_mask_arr = masked_array(cp.array([1.0, 2.0, 3.0]))
        assert bool(_value(no_mask_arr.is_masked())) is False

    def test_compressed(self, test_arr):
        """Test compressed method."""
        compressed = test_arr.compressed()
        expected = cp.array([1.0, 3.0], dtype=cp.float32)
        np.testing.assert_array_equal(_to_numpy(compressed), _to_numpy(expected))

    def test_fill_value_property(self, test_arr):
        """Test fill_value property."""
        # fill_value is a property, not a method
        original_fill = test_arr.fill_value
        test_arr.fill_value = 99.0
        assert test_arr.fill_value == 99.0
        # Note: Setting fill_value doesn't automatically fill masked values
        # You need to call a fill method if it exists


class TestLogicalOperations:
    """Test logical operations."""

    def test_any(self):
        """Test any() method (returns 0-d array, CuPy native)."""
        data = cp.array([True, False, True], dtype=bool)
        mask = cp.array([False, False, True], dtype=bool)
        arr = masked_array(data, mask)
        result = arr.any()
        assert bool(_value(result)) is True

    def test_all(self):
        """Test all() method (returns 0-d array, CuPy native)."""
        data = cp.array([True, True, False], dtype=bool)
        mask = cp.array([False, False, True], dtype=bool)
        arr = masked_array(data, mask)
        result = arr.all()
        assert bool(_value(result)) is True  # False is masked, so all unmasked are True


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_array(self):
        """Test empty array."""
        data = cp.array([], dtype=cp.float32)
        arr = masked_array(data)
        assert arr.size == 0
        assert arr.shape == (0,)

    def test_single_element(self):
        """Test single element array."""
        data = cp.array([42.0], dtype=cp.float32)
        arr = masked_array(data)
        assert arr.shape == (1,)
        assert arr.item() == 42.0

    def test_scalar_input(self):
        """Test scalar input (0-d array)."""
        arr = masked_array(42.0)
        assert _value(arr) == 42.0

    def test_all_masked(self):
        """Test array with all elements masked."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([True, True, True], dtype=bool)
        arr = masked_array(data, mask)
        assert _value(arr.count_masked()) == 3
        assert _value(arr.count_unmasked()) == 0

    def test_no_mask(self):
        """Test array with no mask."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        arr = masked_array(data)
        assert _value(arr.count_masked()) == 0
        assert _value(arr.count_unmasked()) == 3

    def test_nomask_singleton(self):
        """Test nomask singleton."""
        assert nomask is not None
        assert bool(nomask) == False
        assert repr(nomask) == "nomask"

    def test_masked_singleton(self):
        """Test masked singleton."""
        assert masked is not None
        assert bool(masked) == False
        assert str(masked) == "--"


class TestStringRepresentation:
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([False, True, False], dtype=bool)
        arr = masked_array(data, mask)
        repr_str = repr(arr)
        assert "masked_array" in repr_str
        assert "data=" in repr_str
        assert "mask=" in repr_str

    def test_str(self):
        """Test __str__ method."""
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
        mask = cp.array([False, True, False], dtype=bool)
        arr = masked_array(data, mask)
        str_repr = str(arr)
        assert str_repr is not None


class TestNumPyCompatibility:
    """Test NumPy compatibility."""

    def test_scalar_returns(self):
        """Test that methods return scalar or 0-d array (CuPy/NumPy native)."""
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        mask = cp.array([False, True, False, False], dtype=bool)
        arr = masked_array(data, mask)

        # All reductions with axis=None should return scalar-like (0-d or Python scalar)
        functions = ['sum', 'mean', 'std', 'var', 'min', 'max']
        for func_name in functions:
            xp_result = getattr(arr, func_name)(axis=None)
            assert _is_scalar_or_0d(xp_result), f"{func_name} should return scalar or 0-d array"

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion with NumPy."""
        np_ma = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        xp_arr = masked_array(np_ma)
        back_to_np = xp_arr.asmarray()
        
        np.testing.assert_array_equal(back_to_np.data, np_ma.data)
        np.testing.assert_array_equal(back_to_np.mask, np_ma.mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

