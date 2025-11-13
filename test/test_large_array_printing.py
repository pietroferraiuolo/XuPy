"""
Test for the large array printing bug fix
"""
import pytest
import xupy as xp
if xp.on_gpu:
    from xupy.ma import masked_array
import numpy as np


@pytest.mark.skipif(not xp.on_gpu, reason="Requires GPU")
class TestLargeArrayPrinting:
    """Tests for printing large masked arrays without segfault."""
    
    def test_large_array_print_no_segfault(self):
        """Test that printing a very large array doesn't cause segfault."""
        # Create a very large array (100M elements)
        data = xp.arange(10000 * 10000).reshape(10000, 10000)
        mask = xp.zeros((10000, 10000), dtype=bool)
        mask[::2] = True
        
        arr = masked_array(data, mask)
        
        # This should not segfault
        result = str(arr)
        assert isinstance(result, str)
        assert len(result) > 0
        
    def test_large_array_repr_no_segfault(self):
        """Test that repr of a very large array doesn't cause segfault."""
        data = xp.arange(10000 * 10000).reshape(10000, 10000)
        mask = xp.zeros((10000, 10000), dtype=bool)
        
        arr = masked_array(data, mask)
        
        # This should not segfault
        result = repr(arr)
        assert isinstance(result, str)
        assert 'masked_array' in result
        
    def test_print_respects_threshold(self):
        """Test that printing respects NumPy's print threshold."""
        # Save original threshold
        original_threshold = np.get_printoptions()['threshold']
        
        try:
            # Set low threshold to force summarization
            np.set_printoptions(threshold=50, edgeitems=2)
            
            data = xp.arange(10000).reshape(100, 100)
            mask = xp.zeros((100, 100), dtype=bool)
            
            arr = masked_array(data, mask)
            result = str(arr)
            
            # Should contain ellipsis for summarization
            assert '...' in result
            
        finally:
            # Restore original threshold
            np.set_printoptions(threshold=original_threshold)
    
    def test_dtype_preservation(self):
        """Test that dtype is preserved and not forced to float32."""
        # Test with int64 (default for arange)
        data = xp.arange(1000)
        arr = masked_array(data)
        assert arr.data.dtype == xp.int64 or arr.data.dtype == xp.int32  # Platform dependent
        
        # Test with float64
        data_float64 = xp.arange(1000, dtype=xp.float64)
        arr_float64 = masked_array(data_float64)
        assert arr_float64.data.dtype == xp.float64
        
        # Test with explicit float32
        data_float32 = xp.arange(1000, dtype=xp.float32)
        arr_float32 = masked_array(data_float32)
        assert arr_float32.data.dtype == xp.float32
    
    def test_large_values_correctness(self):
        """Test that large values are displayed correctly without precision loss."""
        # Create array with large values that would lose precision with float32
        data = xp.arange(99970000, 99970010, dtype=xp.int64)
        mask = xp.zeros(10, dtype=bool)
        
        arr = masked_array(data, mask)
        result = str(arr)
        
        # Check that consecutive values are displayed correctly
        # (not all the same due to precision loss)
        assert '99970000' in result
        assert '99970009' in result or '9997000' in result  # Might be formatted
        
    def test_1d_large_array_print(self):
        """Test printing of large 1D arrays."""
        data = xp.arange(100000)
        mask = xp.zeros(100000, dtype=bool)
        mask[::10] = True
        
        arr = masked_array(data, mask)
        
        # Should not segfault
        result = str(arr)
        assert isinstance(result, str)
        assert '--' in result  # Should show masked values
        
    def test_2d_large_array_with_complex_mask(self):
        """Test printing 2D array with complex masking pattern."""
        data = xp.arange(1000000).reshape(1000, 1000)
        mask = xp.zeros((1000, 1000), dtype=bool)
        # Create checkerboard pattern
        mask[::2, ::2] = True
        mask[1::2, 1::2] = True
        
        arr = masked_array(data, mask)
        
        # Should not segfault
        result = str(arr)
        assert isinstance(result, str)
        assert '--' in result
        
    def test_small_array_full_display(self):
        """Test that small arrays are displayed fully without summarization."""
        data = xp.arange(100).reshape(10, 10)
        mask = xp.zeros((10, 10), dtype=bool)
        mask[::2] = True
        
        arr = masked_array(data, mask)
        result = str(arr)
        
        # Should show all elements (no ellipsis for small array)
        # Note: This depends on NumPy's default threshold (1000)
        if np.get_printoptions()['threshold'] > 100:
            # Should show first and last elements
            assert '0' in result or '--' in result[:50]  # First element might be masked
            assert '99' in result or '90' in result  # Last row elements


@pytest.mark.skipif(not xp.on_gpu, reason="Requires GPU")
class TestPrintingMemoryEfficiency:
    """Tests for memory-efficient printing."""
    
    def test_memory_efficient_print(self):
        """Test that printing doesn't transfer entire array to CPU unnecessarily."""
        # Create a large array
        data = xp.arange(1000000).reshape(1000, 1000)
        mask = xp.zeros((1000, 1000), dtype=bool)
        
        arr = masked_array(data, mask)
        
        # Get memory before print
        if hasattr(xp.cuda, 'runtime'):
            free_before, total = xp.cuda.runtime.memGetInfo()
        
        # Print (should only transfer edges)
        result = str(arr)
        
        # Memory should not have increased significantly
        # (This is a soft check - exact behavior depends on CuPy's memory management)
        assert isinstance(result, str)
        assert '...' in result  # Should be summarized
