import pytest
import numpy as np
from typing import Any, Tuple, List, Dict, Optional

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from xupy._core import masked_array, MaskedArray


# Helper functions
def _to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to NumPy array."""
    if HAS_CUPY and hasattr(arr, "get"):
        return cp.asnumpy(arr)
    return np.asarray(arr)

def _asmarray(cupy_arr: Any) -> np.ndarray:
    """Convert a CuPy array to NumPy masked array."""
    return cupy_arr.asmarray() if HAS_CUPY else cupy_arr


# Fixtures for test data
@pytest.fixture
def simple_data() -> np.ndarray:
    """Return simple test data."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)


@pytest.fixture
def simple_mask() -> np.ndarray:
    """Return simple test mask."""
    return np.array([[False, False, True], [True, False, False]], dtype=bool)


@pytest.fixture
def masked_arr(simple_data: np.ndarray, simple_mask: np.ndarray) -> MaskedArray:
    """Return a masked array with simple data and mask."""
    return masked_array(simple_data, simple_mask)

@pytest.fixture
def simple_masked_arr(simple_data: np.ndarray, simple_mask: np.ndarray) -> MaskedArray:
    """Return a masked array with simple data and mask."""
    return np.ma.masked_array(simple_data, simple_mask)


@pytest.fixture
def scalar_data() -> float:
    """Return a scalar value."""
    return 10.0


# Test initialization and basic properties
class TestInitialization:
    def test_init_with_data_and_mask(self, simple_data: np.ndarray, simple_mask: np.ndarray) -> None:
        """Test initializing with data and mask."""
        ma = masked_array(simple_data, simple_mask)
        np.testing.assert_array_equal(_to_numpy(ma.data), simple_data)
        np.testing.assert_array_equal(_to_numpy(ma.mask), simple_mask)

    def test_init_with_data_only(self, simple_data: np.ndarray) -> None:
        """Test initializing with data only."""
        ma = masked_array(simple_data)
        np.testing.assert_array_equal(_to_numpy(ma.data), simple_data)
        np.testing.assert_array_equal(
            _to_numpy(ma.mask), np.zeros_like(simple_data, dtype=bool)
        )

    def test_init_with_dtype(self, simple_data: np.ndarray) -> None:
        """Test initializing with specific dtype."""
        ma = masked_array(simple_data, dtype=np.float64)
        assert ma.data.dtype == np.float64

    def test_init_with_mismatched_shapes(self) -> None:
        """Test that mismatched shapes raise an error."""
        with pytest.raises(ValueError):
            masked_array(np.array([1, 2, 3]), np.array([True, False]))


# Test representation
class TestRepresentation:
    def test_repr(self, masked_arr: MaskedArray) -> None:
        """Test string representation shows masked values as '--'."""
        repr_str = repr(masked_arr)
        assert "--" in repr_str
        assert "XupyMaskedArray" in repr_str
        assert "data" in repr_str
        assert "mask" in repr_str


# Test arithmetic operations
class TestArithmeticOperations:
    # Binary operations
    def test_add(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test addition with array and scalar."""
        # Array + Array
        result = masked_arr + masked_arr
        expected_data = simple_masked_arr + simple_masked_arr
        np.testing.assert_array_equal(_asmarray(result).data, expected_data.data)
        
        # Array + Scalar
        result = masked_arr + 2.0
        expected_data = simple_masked_arr + 2.0
        np.testing.assert_array_equal(_asmarray(result).data, expected_data.data)

    def test_subtract(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test subtraction with array and scalar."""
        # Array - Array
        result = masked_arr - masked_arr
        expected_data = simple_masked_arr - simple_masked_arr
        np.testing.assert_array_equal(_asmarray(result).data, expected_data)
        
        # Array - Scalar
        result = masked_arr - 2.0
        expected_data = simple_masked_arr - 2.0
        np.testing.assert_array_equal(_asmarray(result).data, expected_data)

    def test_multiply(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test multiplication with array and scalar."""
        # Array * Array
        result = masked_arr * masked_arr
        expected_data = simple_masked_arr * simple_masked_arr
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)
        
        # Array * Scalar
        result = masked_arr * 2.0
        expected_data = simple_masked_arr * 2.0
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    def test_truediv(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test division with array and scalar."""
        # Array / Array
        result = masked_arr / masked_arr
        # Avoid division by zero for masked values
        valid_mask = (_to_numpy(masked_arr.data) != 0)
        expected_data = np.zeros_like(simple_masked_arr)
        expected_data[valid_mask] = simple_masked_arr[valid_mask] / simple_masked_arr[valid_mask]
        np.testing.assert_allclose(_to_numpy(result.data)[valid_mask], expected_data[valid_mask])
        
        # Array / Scalar
        result = masked_arr / 2.0
        expected_data = simple_masked_arr / 2.0
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    def test_floordiv(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test floor division with array and scalar."""
        # Array // Array
        result = masked_arr // masked_arr
        # Avoid division by zero for masked values
        valid_mask = (_to_numpy(masked_arr.data) != 0)
        expected_data = np.zeros_like(simple_masked_arr)
        expected_data[valid_mask] = simple_masked_arr[valid_mask] // simple_masked_arr[valid_mask]
        np.testing.assert_allclose(_to_numpy(result.data)[valid_mask], expected_data[valid_mask])
        
        # Array // Scalar
        result = masked_arr // 2.0
        expected_data = simple_masked_arr // 2.0
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    def test_modulo(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test modulo with array and scalar."""
        # Array % Array
        result = masked_arr % masked_arr
        # Avoid division by zero for masked values
        valid_mask = (_to_numpy(masked_arr.data) != 0)
        expected_data = np.zeros_like(simple_masked_arr)
        expected_data[valid_mask] = simple_masked_arr[valid_mask] % simple_masked_arr[valid_mask]
        np.testing.assert_allclose(_to_numpy(result.data)[valid_mask], expected_data[valid_mask])
        
        # Array % Scalar
        result = masked_arr % 2.0
        expected_data = simple_masked_arr % 2.0
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    def test_power(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test power with array and scalar."""
        # Array ** Array
        result = masked_arr ** 2
        expected_data = simple_masked_arr ** 2
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)
        
        # Array ** Scalar
        result = masked_arr ** 2.0
        expected_data = simple_masked_arr ** 2.0
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    # Reflected operations
    def test_radd(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test reflected addition."""
        result = 2.0 + masked_arr
        expected_data = 2.0 + simple_masked_arr
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    def test_rsub(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test reflected subtraction."""
        result = 10.0 - masked_arr
        expected_data = 10.0 - simple_masked_arr
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    # In-place operations
    def test_iadd(self, masked_arr: MaskedArray) -> None:
        """Test in-place addition."""
        original_data = _asmarray(masked_arr).data.copy()
        masked_arr += 2.0
        expected_data = original_data + 2.0
        np.testing.assert_array_equal(_asmarray(masked_arr).data, expected_data)

    def test_isub(self, masked_arr: MaskedArray) -> None:
        """Test in-place subtraction."""
        original_data = _asmarray(masked_arr).data.copy()
        masked_arr -= 2.0
        expected_data = original_data - 2.0
        np.testing.assert_array_equal(_asmarray(masked_arr).data, expected_data)


# Test matrix multiplication
class TestMatrixMultiplication:
    def test_matmul(self) -> None:
        """Test matrix multiplication."""
        a = masked_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = masked_array(np.array([[5.0, 6.0], [7.0, 8.0]]))
        
        result = a @ b
        expected = _asmarray(a) @ _asmarray(b)
        np.testing.assert_array_equal(_asmarray(result), expected)


# Test unary operations
class TestUnaryOperations:
    def test_neg(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test negation."""
        result = -masked_arr
        expected_data = -simple_masked_arr
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    def test_pos(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test unary plus."""
        result = +masked_arr
        expected_data = +simple_masked_arr
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)

    def test_abs(self) -> None:
        """Test absolute value."""
        a = masked_array(np.array([-1.0, 2.0, -3.0]))
        result = abs(a)
        expected_data = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(_to_numpy(result.data), expected_data)


# Test comparison operations
class TestComparisonOperations:
    def test_eq(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test equality comparison."""
        result = masked_arr == masked_arr
        expected = np.ones_like(simple_masked_arr, dtype=bool)
        np.testing.assert_array_equal(_to_numpy(result), expected)
        
        result = masked_arr == 3.0
        expected = simple_masked_arr == 3.0
        np.testing.assert_array_equal(_to_numpy(result), expected)

    def test_ne(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test inequality comparison."""
        result = masked_arr != masked_arr
        expected = np.zeros_like(simple_masked_arr, dtype=bool)
        np.testing.assert_array_equal(_to_numpy(result), expected)
        
        result = masked_arr != 3.0
        expected = simple_masked_arr != 3.0
        np.testing.assert_array_equal(_to_numpy(result), expected)

    def test_lt(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test less than comparison."""
        result = masked_arr < 3.0
        expected = simple_masked_arr < 3.0
        np.testing.assert_array_equal(_to_numpy(result), expected)

    def test_le(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test less than or equal comparison."""
        result = masked_arr <= 3.0
        expected = simple_masked_arr <= 3.0
        np.testing.assert_array_equal(_to_numpy(result), expected)

    def test_gt(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test greater than comparison."""
        result = masked_arr > 3.0
        expected = simple_masked_arr > 3.0
        np.testing.assert_array_equal(_to_numpy(result), expected)

    def test_ge(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray) -> None:
        """Test greater than or equal comparison."""
        result = masked_arr >= 3.0
        expected = simple_masked_arr >= 3.0
        np.testing.assert_array_equal(_to_numpy(result), expected)


# Test indexing and slicing
class TestIndexingAndSlicing:
    def test_basic_indexing(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray, simple_mask: np.ndarray) -> None:
        """Test basic indexing."""
        # Single element
        if simple_mask[0, 0]:  # If masked
            assert masked_arr[0, 0] is np.ma.masked
        else:
            assert masked_arr[0, 0] == simple_masked_arr[0, 0]
        
        # Slicing
        result = masked_arr[0, :]
        np.testing.assert_array_equal(_to_numpy(result.data), simple_masked_arr[0, :])
        np.testing.assert_array_equal(_to_numpy(result.mask), simple_mask[0, :])

    def test_advanced_indexing(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray, simple_mask: np.ndarray) -> None:
        """Test advanced indexing."""
        idx = np.array([0, 1])
        result = masked_arr[idx, :]
        np.testing.assert_array_equal(_to_numpy(result.data), simple_masked_arr[idx, :])
        np.testing.assert_array_equal(_to_numpy(result.mask), simple_mask[idx, :])


# Test attribute access
class TestAttributeAccess:
    def test_getattr(self, masked_arr: MaskedArray) -> None:
        """Test attribute access delegates to data."""
        assert masked_arr.shape == (2, 3)
        assert masked_arr.dtype == np.float32
        assert hasattr(masked_arr, 'T')  # Transpose property


# Test conversion
class TestConversion:
    def test_asmarray(self, masked_arr: MaskedArray, simple_masked_arr: np.ndarray, simple_mask: np.ndarray) -> None:
        """Test conversion to NumPy masked array."""
        marray = masked_arr.asmarray()
        assert isinstance(marray, np.ma.MaskedArray)
        np.testing.assert_array_equal(marray.data, simple_masked_arr)
        np.testing.assert_array_equal(marray.mask, simple_mask)


# Test mask propagation
class TestMaskPropagation:
    def test_binary_op_mask_propagation(self) -> None:
        """Test mask propagation in binary operations."""
        a = masked_array(np.array([1.0, 2.0, 3.0]), np.array([True, False, False]))
        b = masked_array(np.array([4.0, 5.0, 6.0]), np.array([False, True, False]))
        
        # Result should have both masks combined (logical OR)
        result = a + b
        expected_mask = np.array([True, True, False])
        np.testing.assert_array_equal(_to_numpy(result.mask), expected_mask)

    def test_unary_op_mask_preservation(self) -> None:
        """Test mask preservation in unary operations."""
        a = masked_array(np.array([1.0, 2.0, 3.0]), np.array([True, False, False]))
        
        result = -a
        np.testing.assert_array_equal(_to_numpy(result.mask), _to_numpy(a.mask))