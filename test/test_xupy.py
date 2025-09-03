import pytest
import numpy as np
from typing import Any

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from xupy._core import masked_array, MaskedArray, zeros, ones, eye, linspace, arange, random, normal, uniform, concatenate, stack, vstack, hstack, split


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


# Test new array manipulation methods
class TestArrayManipulation:
    def test_reshape(self) -> None:
        """Test array reshaping."""
        a = masked_array(np.array([[1, 2, 3], [4, 5, 6]]))
        reshaped = a.reshape(3, 2)
        assert reshaped.shape == (3, 2)
        np.testing.assert_array_equal(_to_numpy(reshaped.data), np.array([[1, 2], [3, 4], [5, 6]]))
    
    def test_transpose(self) -> None:
        """Test array transposition."""
        a = masked_array(np.array([[1, 2], [3, 4]]))
        transposed = a.transpose()
        assert transposed.shape == (2, 2)
        np.testing.assert_array_equal(_to_numpy(transposed.data), np.array([[1, 3], [2, 4]]))
    
    def test_flatten(self) -> None:
        """Test array flattening."""
        a = masked_array(np.array([[1, 2], [3, 4]]))
        flattened = a.flatten()
        assert flattened.shape == (4,)
        np.testing.assert_array_equal(_to_numpy(flattened.data), np.array([1, 2, 3, 4]))
    
    def test_squeeze(self) -> None:
        """Test array squeezing."""
        a = masked_array(np.array([[[1], [2]]]))
        squeezed = a.squeeze()
        assert squeezed.shape == (2,)
        np.testing.assert_array_equal(_to_numpy(squeezed.data), np.array([1, 2]))
    
    def test_expand_dims(self) -> None:
        """Test dimension expansion."""
        a = masked_array(np.array([1, 2, 3]))
        expanded = a.expand_dims(axis=1)
        assert expanded.shape == (3, 1)
        np.testing.assert_array_equal(_to_numpy(expanded.data), np.array([[1], [2], [3]]))


# Test new statistical methods
class TestStatisticalMethods:
    def test_mean_global(self) -> None:
        """Test global mean calculation."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([False, False, True, False, False]))
        mean_val = a.mean()
        expected = (1 + 2 + 4 + 5) / 4  # Excluding masked value 3
        assert abs(mean_val - expected) < 1e-6
    
    def test_sum_global(self) -> None:
        """Test global sum calculation."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([False, False, True, False, False]))
        sum_val = a.sum()
        expected = 1 + 2 + 4 + 5  # Excluding masked value 3
        assert sum_val == expected
    
    def test_std_global(self) -> None:
        """Test global standard deviation calculation."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([False, False, True, False, False]))
        std_val = a.std()
        valid_data = np.array([1, 2, 4, 5])
        expected = np.std(valid_data)
        assert abs(std_val - expected) < 1e-6
    
    def test_var_global(self) -> None:
        """Test global variance calculation."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([False, False, True, False, False]))
        var_val = a.var()
        valid_data = np.array([1, 2, 4, 5])
        expected = np.var(valid_data)
        assert abs(var_val - expected) < 1e-6
    
    def test_min_global(self) -> None:
        """Test global minimum calculation."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([False, False, True, False, False]))
        min_val = a.min()
        expected = 1  # Minimum of unmasked values
        assert min_val == expected
    
    def test_max_global(self) -> None:
        """Test global maximum calculation."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([False, False, True, False, False]))
        max_val = a.max()
        expected = 5  # Maximum of unmasked values
        assert max_val == expected


# Test new utility functions
class TestUtilityFunctions:
    def test_zeros(self) -> None:
        """Test zeros array creation."""
        a = zeros((3, 3))
        assert a.shape == (3, 3)
        assert a.dtype == np.float32
        np.testing.assert_array_equal(_to_numpy(a.data), np.zeros((3, 3)))
        np.testing.assert_array_equal(_to_numpy(a.mask), np.zeros((3, 3), dtype=bool))
    
    def test_ones(self) -> None:
        """Test ones array creation."""
        a = ones((3, 3))
        assert a.shape == (3, 3)
        assert a.dtype == np.float32
        np.testing.assert_array_equal(_to_numpy(a.data), np.ones((3, 3)))
        np.testing.assert_array_equal(_to_numpy(a.mask), np.zeros((3, 3), dtype=bool))
    
    def test_eye(self) -> None:
        """Test identity matrix creation."""
        a = eye(3)
        assert a.shape == (3, 3)
        np.testing.assert_array_equal(_to_numpy(a.data), np.eye(3))
        np.testing.assert_array_equal(_to_numpy(a.mask), np.zeros((3, 3), dtype=bool))
    
    def test_linspace(self) -> None:
        """Test linspace array creation."""
        a = linspace(0, 10, 5)
        assert a.shape == (5,)
        np.testing.assert_array_equal(_to_numpy(a.data), np.linspace(0, 10, 5))
        np.testing.assert_array_equal(_to_numpy(a.mask), np.zeros(5, dtype=bool))
    
    def test_arange(self) -> None:
        """Test arange array creation."""
        a = arange(0, 10, 2)
        assert a.shape == (5,)
        np.testing.assert_array_equal(_to_numpy(a.data), np.arange(0, 10, 2))
        np.testing.assert_array_equal(_to_numpy(a.mask), np.zeros(5, dtype=bool))
    
    def test_random(self) -> None:
        """Test random array creation."""
        a = random((3, 3))
        assert a.shape == (3, 3)
        assert a.dtype == np.float32
        # Check that values are between 0 and 1
        data = _to_numpy(a.data)
        assert np.all(data >= 0) and np.all(data < 1)
        np.testing.assert_array_equal(_to_numpy(a.mask), np.zeros((3, 3), dtype=bool))
    
    def test_normal(self) -> None:
        """Test normal distribution array creation."""
        a = normal(0, 1, (3, 3))
        assert a.shape == (3, 3)
        assert a.dtype == np.float32
        np.testing.assert_array_equal(_to_numpy(a.mask), np.zeros((3, 3), dtype=bool))
    
    def test_uniform(self) -> None:
        """Test uniform distribution array creation."""
        a = uniform(-1, 1, (3, 3))
        assert a.shape == (3, 3)
        assert a.dtype == np.float32
        # Check that values are between -1 and 1
        data = _to_numpy(a.data)
        assert np.all(data >= -1) and np.all(data <= 1)
        np.testing.assert_array_equal(_to_numpy(a.mask), np.zeros((3, 3), dtype=bool))


# Test new array information methods
class TestArrayInformationMethods:
    def test_count_masked(self) -> None:
        """Test counting masked elements."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([True, False, True, False, True]))
        assert a.count_masked() == 3
    
    def test_count_unmasked(self) -> None:
        """Test counting unmasked elements."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([True, False, True, False, True]))
        assert a.count_unmasked() == 2
    
    def test_is_masked(self) -> None:
        """Test checking if array has masked values."""
        a = masked_array(np.array([1, 2, 3]), np.array([False, False, False]))
        assert not a.is_masked()
        
        b = masked_array(np.array([1, 2, 3]), np.array([False, True, False]))
        assert b.is_masked()
    
    def test_compressed(self) -> None:
        """Test getting compressed (unmasked) data."""
        a = masked_array(np.array([1, 2, 3, 4, 5]), np.array([True, False, True, False, True]))
        compressed = a.compressed()
        np.testing.assert_array_equal(_to_numpy(compressed), np.array([2, 4]))


# Test new copy and conversion methods
class TestCopyAndConversionMethods:
    def test_copy(self) -> None:
        """Test array copying."""
        a = masked_array(np.array([1, 2, 3]), np.array([False, True, False]))
        b = a.copy()
        assert b is not a
        np.testing.assert_array_equal(_to_numpy(b.data), _to_numpy(a.data))
        np.testing.assert_array_equal(_to_numpy(b.mask), _to_numpy(a.mask))
    
    def test_astype(self) -> None:
        """Test array type conversion."""
        a = masked_array(np.array([1, 2, 3]), np.array([False, True, False]))
        b = a.astype(np.float64)
        assert b.dtype == np.float64
        np.testing.assert_array_equal(_to_numpy(b.data), np.array([1., 2., 3.]))
        np.testing.assert_array_equal(_to_numpy(b.mask), _to_numpy(a.mask))
    
    def test_tolist(self) -> None:
        """Test conversion to list."""
        a = masked_array(np.array([[1, 2], [3, 4]]))
        result = a.tolist()
        assert result == [[1, 2], [3, 4]]
    
    def test_item(self) -> None:
        """Test item extraction."""
        a = masked_array(np.array([5]))
        assert a.item() == 5
        
        b = masked_array(np.array([[1, 2], [3, 4]]))
        assert b.item(0, 1) == 2


# Test new universal functions
class TestUniversalFunctions:
    def test_sqrt(self) -> None:
        """Test square root function."""
        a = masked_array(np.array([1, 4, 9, 16]))
        result = a.sqrt()
        np.testing.assert_array_equal(_to_numpy(result.data), np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(_to_numpy(result.mask), _to_numpy(a.mask))
    
    def test_exp(self) -> None:
        """Test exponential function."""
        a = masked_array(np.array([0, 1, 2]))
        result = a.exp()
        np.testing.assert_array_equal(_to_numpy(result.data), np.exp([0, 1, 2]))
        np.testing.assert_array_equal(_to_numpy(result.mask), _to_numpy(a.mask))
    
    def test_sin(self) -> None:
        """Test sine function."""
        a = masked_array(np.array([0, np.pi/2, np.pi]))
        result = a.sin()
        np.testing.assert_array_almost_equal(_to_numpy(result.data), np.sin([0, np.pi/2, np.pi]))
        np.testing.assert_array_equal(_to_numpy(result.mask), _to_numpy(a.mask))
    
    def test_floor(self) -> None:
        """Test floor function."""
        a = masked_array(np.array([1.1, 2.9, -1.1, -2.9]))
        result = a.floor()
        np.testing.assert_array_equal(_to_numpy(result.data), np.array([1, 2, -2, -3]))
        np.testing.assert_array_equal(_to_numpy(result.mask), _to_numpy(a.mask))
    
    def test_ceil(self) -> None:
        """Test ceiling function."""
        a = masked_array(np.array([1.1, 2.9, -1.1, -2.9]))
        result = a.ceil()
        np.testing.assert_array_equal(_to_numpy(result.data), np.array([2, 3, -1, -2]))
        np.testing.assert_array_equal(_to_numpy(result.mask), _to_numpy(a.mask))
    
    def test_round(self) -> None:
        """Test round function."""
        a = masked_array(np.array([1.1, 2.9, -1.1, -2.9]))
        result = a.round()
        np.testing.assert_array_equal(_to_numpy(result.data), np.array([1, 3, -1, -3]))
        np.testing.assert_array_equal(_to_numpy(result.mask), _to_numpy(a.mask))
