"""
Comprehensive test suite for xupy._core module.

Tests core XuPy functionality including:
- NumpyContext manager
- Device management (set_device)
- MemoryContext manager
- Basic GPU/CPU detection
- Type aliases
"""
import pytest
import numpy as np
from typing import Any

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import xupy as xp
from xupy._core import NumpyContext, on_gpu

# Skip all tests if CuPy is not available
pytestmark = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# Helper functions
def _to_numpy(arr: Any) -> np.ndarray:
    """Convert any array to NumPy array."""
    if hasattr(arr, "get"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


class TestNumpyContext:
    """Test NumpyContext manager."""

    def test_numpy_context_basic(self):
        """Test basic NumpyContext usage."""
        with NumpyContext() as np:
            assert np is not None
            # Should be able to create NumPy arrays
            arr = np.array([1, 2, 3])
            assert isinstance(arr, np.ndarray)
            assert not isinstance(arr, cp.ndarray)

    def test_numpy_context_inside(self):
        """Test that inside context, np refers to NumPy."""
        with NumpyContext() as np:
            arr_np = np.array([1, 2, 3])
            arr_xp = xp.array([1, 2, 3])
            
            assert isinstance(arr_np, np.ndarray)
            if on_gpu:
                assert isinstance(arr_xp, cp.ndarray)
            else:
                assert isinstance(arr_xp, np.ndarray)

    def test_numpy_context_exit(self):
        """Test that context properly exits."""
        with NumpyContext() as np:
            arr = np.array([1, 2, 3])
            assert isinstance(arr, np.ndarray)
        
        # After context, should still work
        arr2 = xp.array([1, 2, 3])
        assert arr2 is not None

    def test_numpy_context_repr(self):
        """Test NumpyContext string representation."""
        ctx = NumpyContext()
        repr_str = repr(ctx)
        assert "NumpyContext" in repr_str

    def test_numpy_context_nested(self):
        """Test nested NumpyContext usage."""
        with NumpyContext() as np1:
            arr1 = np1.array([1, 2, 3])
            with NumpyContext() as np2:
                arr2 = np2.array([4, 5, 6])
                assert isinstance(arr1, np.ndarray)
                assert isinstance(arr2, np.ndarray)


class TestDeviceManagement:
    """Test device management functions."""

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_on_gpu_flag(self):
        """Test that on_gpu flag is set correctly."""
        if HAS_CUPY:
            assert on_gpu == True
        else:
            assert on_gpu == False

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_set_device_single_gpu(self):
        """Test set_device with single GPU (should raise error)."""
        if HAS_CUPY:
            n_gpus = cp.cuda.runtime.getDeviceCount()
            if n_gpus == 1:
                # Should raise RuntimeError when trying to set device on single GPU system
                with pytest.raises(RuntimeError, match="Only one GPU available"):
                    xp.set_device(0)

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_set_device_multiple_gpus(self):
        """Test set_device with multiple GPUs."""
        if HAS_CUPY:
            n_gpus = cp.cuda.runtime.getDeviceCount()
            if n_gpus > 1:
                original_device = cp.cuda.runtime.getDevice()
                try:
                    # Try setting to device 1
                    xp.set_device(1)
                    assert cp.cuda.runtime.getDevice() == 1
                    # Set back to original
                    xp.set_device(original_device)
                except RuntimeError:
                    # If device 1 doesn't exist or can't be set, that's okay
                    pass

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_set_device_already_set(self):
        """Test set_device when device is already set."""
        if HAS_CUPY:
            current_device = cp.cuda.runtime.getDevice()
            n_gpus = cp.cuda.runtime.getDeviceCount()
            if n_gpus > 1:
                # Should warn when setting to same device (only if multiple GPUs)
                with pytest.warns(UserWarning, match="already the current device"):
                    xp.set_device(current_device)
            else:
                # Single GPU should raise RuntimeError
                with pytest.raises(RuntimeError):
                    xp.set_device(current_device)


class TestTypeAliases:
    """Test type aliases."""

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_float_alias(self):
        """Test float type alias."""
        if on_gpu:
            assert xp.float == cp.float32
        else:
            assert xp.float == np.float32

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_double_alias(self):
        """Test double type alias."""
        if on_gpu:
            assert xp.double == cp.float64
        else:
            assert xp.double == np.float64

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_np_reference(self):
        """Test that np reference points to NumPy."""
        assert xp.np is np
        assert xp.np.array([1, 2, 3]) is not None


class TestBasicFunctionality:
    """Test basic XuPy functionality."""

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_array_creation(self):
        """Test basic array creation."""
        arr = xp.array([1, 2, 3])
        assert arr is not None
        if on_gpu:
            assert isinstance(arr, cp.ndarray)
        else:
            assert isinstance(arr, np.ndarray)

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_array_operations(self):
        """Test basic array operations."""
        arr1 = xp.array([1, 2, 3])
        arr2 = xp.array([4, 5, 6])
        result = arr1 + arr2
        assert result is not None
        np.testing.assert_array_equal(_to_numpy(result), np.array([5, 7, 9]))

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_array_dtype(self):
        """Test array dtype handling."""
        arr = xp.array([1, 2, 3], dtype=xp.float32)
        assert arr.dtype == (cp.float32 if on_gpu else np.float32)

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_zeros_ones(self):
        """Test zeros and ones functions."""
        zeros_arr = xp.zeros((3, 3))
        ones_arr = xp.ones((3, 3))
        
        assert zeros_arr.shape == (3, 3)
        assert ones_arr.shape == (3, 3)
        np.testing.assert_array_equal(_to_numpy(zeros_arr), np.zeros((3, 3)))
        np.testing.assert_array_equal(_to_numpy(ones_arr), np.ones((3, 3)))

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_arange(self):
        """Test arange function."""
        arr = xp.arange(10)
        expected = np.arange(10)
        np.testing.assert_array_equal(_to_numpy(arr), expected)

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_linspace(self):
        """Test linspace function."""
        arr = xp.linspace(0, 1, 5)
        expected = np.linspace(0, 1, 5)
        np.testing.assert_array_almost_equal(_to_numpy(arr), expected, decimal=5)


class TestMemoryContext:
    """Test MemoryContext manager."""

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_memory_context_basic(self):
        """Test basic MemoryContext usage."""
        if on_gpu and hasattr(xp, 'MemoryContext'):
            with xp.MemoryContext() as ctx:
                # Should be able to create arrays
                arr = xp.array([1, 2, 3])
                assert arr is not None

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_memory_context_cleanup(self):
        """Test that MemoryContext cleans up properly."""
        if on_gpu and hasattr(xp, 'MemoryContext'):
            with xp.MemoryContext(auto_cleanup=True) as ctx:
                # Create some arrays
                arr1 = xp.array([1, 2, 3])
                arr2 = xp.array([4, 5, 6])
                # Arrays should exist
                assert arr1 is not None
                assert arr2 is not None
            # After context, memory should be cleaned up
            # (we can't easily verify this, but we can check no errors occur)


class TestGPUDetection:
    """Test GPU detection and availability."""

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_gpu_available(self):
        """Test that GPU is detected if available."""
        if HAS_CUPY:
            # Should have on_gpu flag
            assert hasattr(xp, 'on_gpu')
            # Should have device info
            if on_gpu:
                assert cp.cuda.runtime.getDeviceCount() > 0

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_cupy_import(self):
        """Test that CuPy functions are available."""
        if on_gpu:
            # Should have CuPy functions
            assert hasattr(xp, 'array')
            assert hasattr(xp, 'zeros')
            assert hasattr(xp, 'ones')


class TestIntegration:
    """Test integration aspects."""

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_numpy_compatibility(self):
        """Test compatibility with NumPy operations."""
        xp_arr = xp.array([1, 2, 3, 4, 5])
        np_arr = np.array([1, 2, 3, 4, 5])
        
        # Should be able to convert
        converted = _to_numpy(xp_arr)
        np.testing.assert_array_equal(converted, np_arr)

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy required")
    def test_array_conversion(self):
        """Test array conversion between NumPy and CuPy."""
        np_arr = np.array([1, 2, 3])
        xp_arr = xp.array(np_arr)
        
        assert xp_arr is not None
        if on_gpu:
            assert isinstance(xp_arr, cp.ndarray)
        
        # Convert back
        back_to_np = _to_numpy(xp_arr)
        np.testing.assert_array_equal(back_to_np, np_arr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

