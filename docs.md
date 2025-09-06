# XuPy Documentation

## Overview

XuPy is a comprehensive Python package that provides GPU-accelerated masked arrays with automatic CPU fallback. It seamlessly integrates CuPy's high-performance GPU computing with NumPy's familiar masked array interface, making it easy to leverage GPU acceleration for scientific computing tasks involving masked data.

## Key Features

- **GPU Acceleration**: Automatic GPU detection with CuPy fallback to NumPy
- **Masked Arrays**: Full support for masked arrays with GPU acceleration
- **NumPy Compatibility**: Drop-in replacement for numpy.ma.MaskedArray
- **Memory Management**: Efficient GPU memory management with context managers
- **Performance**: Optimized for large-scale data processing on GPU
- **Automatic Fallback**: Graceful degradation to CPU when GPU is unavailable

## Installation

### Basic Installation

```bash
pip install xupy
```

### GPU Requirements

For GPU acceleration, install CuPy with CUDA support:

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 10.2
pip install cupy-cuda102
```

### System Requirements

- **Python**: >= 3.10
- **CUDA-compatible GPU**: Compute capability 3.0+ (optional, CPU fallback available)
- **Dependencies**: NumPy, CuPy (optional)

## Quick Start

```python
import xupy as xp
import numpy as np

# Create arrays with automatic GPU detection
a = xp.random.normal(0, 1, (1000, 1000))
b = xp.random.normal(0, 1, (1000, 1000))

# Create masks
mask = xp.random.random((1000, 1000)) > 0.1

# Create masked arrays
am = xp.masked_array(a, mask)
bm = xp.masked_array(b, mask)

# Perform operations (masks are automatically handled)
result = am + bm
mean_val = am.mean()
std_val = am.std()
```

## Core Concepts

### GPU Acceleration and Fallback

XuPy automatically detects GPU availability and uses CuPy for GPU acceleration. If no GPU is available or CuPy is not installed, it gracefully falls back to NumPy:

```python
import xupy as xp

# XuPy automatically detects GPU and prints status
a = xp.zeros((100, 100))  # Uses CuPy if available, NumPy otherwise
```

### Masked Arrays

Masked arrays allow you to work with data that contains invalid or missing values. The mask is a boolean array where `True` indicates masked (invalid) elements:

```python
import xupy as xp
import numpy as np

# Create data with some invalid values
data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
mask = np.isnan(data)  # [False, False, True, False, False]

# Create masked array
ma = xp.masked_array(data, mask)
print(ma)  # Shows: [1.0 2.0 -- 4.0 5.0]
```

## Detailed API Reference

### MaskedArray Class

The core of XuPy is the `MaskedArray` class, which provides GPU-accelerated masked array functionality.

#### Constructor

```python
xp.masked_array(data, mask=None, dtype=None, fill_value=None, keep_mask=True, hard_mask=False, order=None)
```

**Parameters:**

- `data`: array-like - Input data (NumPy array, CuPy array, or Python sequence)
- `mask`: array-like, optional - Boolean mask (True = masked)
- `dtype`: data-type, optional - Desired data type (defaults to float32 for GPU)
- `fill_value`: scalar, optional - Value for filling masked elements
- `keep_mask`: bool, optional - Whether to combine with existing mask
- `hard_mask`: bool, optional - Immutable mask flag
- `order`: str, optional - Memory layout ('C', 'F', 'A')

#### Core Properties

- `data`: Underlying array (CuPy or NumPy)
- `mask`: Boolean mask array
- `shape`: Tuple of array dimensions
- `dtype`: Data type of the array
- `size`: Total number of elements
- `ndim`: Number of dimensions
- `T`: Transposed array

#### Array Manipulation Methods

```python
# Reshaping
ma.reshape(*shape)
ma.flatten(order='C')
ma.ravel(order='C')
ma.squeeze(axis=None)
ma.expand_dims(axis)

# Transposition
ma.transpose(*axes)
ma.swapaxes(axis1, axis2)

# Repetition
ma.repeat(repeats, axis=None)
ma.tile(reps)
```

#### Statistical Methods

```python
ma.mean(axis=None, **kwargs)    # Arithmetic mean
ma.sum(axis=None, **kwargs)     # Sum of elements
ma.std(axis=None, **kwargs)     # Standard deviation
ma.var(axis=None, **kwargs)     # Variance
ma.min(axis=None, **kwargs)     # Minimum value
ma.max(axis=None, **kwargs)     # Maximum value
```

#### Mathematical Functions

```python
# Trigonometric
ma.sin()    # Sine
ma.cos()    # Cosine
ma.tan()    # Tangent
ma.arcsin() # Inverse sine
ma.arccos() # Inverse cosine
ma.arctan() # Inverse tangent

# Hyperbolic
ma.sinh()   # Hyperbolic sine
ma.cosh()   # Hyperbolic cosine
ma.tanh()   # Hyperbolic tangent

# Exponential and Logarithmic
ma.exp()    # Exponential
ma.log()    # Natural logarithm
ma.log10()  # Base-10 logarithm

# Rounding
ma.floor()  # Floor
ma.ceil()   # Ceiling
ma.round(decimals=0)  # Round to decimals
```

#### Array Information Methods

```python
ma.any(axis=None, **kwargs)     # Test if any element is True
ma.all(axis=None, **kwargs)     # Test if all elements are True
ma.count_masked()               # Number of masked elements
ma.count_unmasked()             # Number of unmasked elements
ma.is_masked()                  # True if any element is masked
ma.compressed()                 # 1D array of unmasked elements
```

#### Conversion Methods

```python
ma.copy(order='C')              # Copy array
ma.astype(dtype)                # Cast to new type
ma.tolist()                     # Convert to nested list
ma.item(*args)                  # Extract scalar
ma.asmarray(**kwargs)           # Convert to NumPy masked array
```

### Array Creation Functions

XuPy provides convenient functions for creating arrays:

```python
# Basic arrays
xp.zeros(shape, dtype=None)           # Array of zeros
xp.ones(shape, dtype=None)            # Array of ones
xp.eye(N, dtype=None)                 # Identity matrix
xp.identity(n, dtype=None)            # Identity matrix

# Sequences
xp.linspace(start, stop, num)         # Linearly spaced values
xp.logspace(start, stop, num)         # Logarithmically spaced values
xp.arange(start, stop, step)          # Arithmetic progression

# Random arrays
xp.random(shape)                      # Uniform random [0, 1)
xp.normal(loc, scale, shape)          # Normal distribution
xp.uniform(low, high, shape)          # Uniform distribution
```

## GPU Masked Array Features

### GPU Acceleration Details

XuPy's GPU masked arrays provide several key advantages:

1. **Automatic GPU Detection**: Detects CUDA-compatible GPUs and uses CuPy
2. **Memory Efficiency**: Optimized memory usage on GPU
3. **Performance**: Significant speedup for large arrays
4. **Compatibility**: NumPy-like interface with GPU acceleration

### GPU Memory Management

XuPy includes a `MemoryContext` class for efficient GPU memory management:

```python
import xupy as xp

# Create memory context
with xp.MemoryContext(device_id=0) as ctx:
    # GPU operations within context
    a = xp.random.normal(0, 1, (10000, 10000))
    b = xp.random.normal(0, 1, (10000, 10000))
    result = a + b
    mem_info_0 = ctx.get_memory_info()
    print(f"GPU Memory Used: {mem_info_0['used']/(1024*1000)} Mb")
    # Clear cache when done
    ctx.clear_cache()

    # Get memory info
    mem_info_1 = ctx.get_memory_info()
    print(f"GPU Memory Used: {mem_info_1['used']/(1024*1000)} Mb")
    print(f"Freed GPU Memory: {(mem_info_0['used']-mem_info_1['used'])/(1024*1000)} Mb")

```

### Performance Characteristics

- **Small arrays (< 1000 elements)**: CPU may be faster due to GPU overhead
- **Medium arrays (1000-10000 elements)**: GPU provides 2-5x speedup
- **Large arrays (> 10000 elements)**: GPU provides 5-20x speedup

### GPU System Requirements

- **CUDA Version**: 10.2, 11.x, or 12.x
- **GPU Compute Capability**: 3.0 or higher
- **VRAM**: Depends on array size (minimum 2GB recommended)

### Advanced GPU Memory Management

XuPy includes an advanced `MemoryContext` class for efficient GPU memory management:

```python
import xupy as xp

# Basic usage with automatic cleanup
with xp.MemoryContext() as ctx:
    # GPU operations
    data = xp.random.normal(0, 1, (10000, 10000))
    result = data.mean()
# Memory automatically cleaned up on exit

# Advanced features
with xp.MemoryContext(memory_threshold=0.8, auto_cleanup=True) as ctx:
    # Monitor memory usage
    mem_info = ctx.get_memory_info()
    print(f"GPU Memory: {mem_info['used'] / (1024**3):.2f} GB")
    
    # Aggressive cleanup when needed
    if ctx.check_memory_pressure():
        ctx.aggressive_cleanup()
    
    # Emergency cleanup for critical situations
    ctx.emergency_cleanup()
```

#### MemoryContext Features

- **Automatic Cleanup**: Memory freed automatically when exiting context
- **Memory Monitoring**: Real-time tracking of GPU memory usage
- **Pressure Detection**: Automatic cleanup when memory usage is high
- **Aggressive Cleanup**: Force garbage collection and cache clearing
- **Emergency Cleanup**: Nuclear option for out-of-memory situations
- **Object Tracking**: Track GPU objects for proper cleanup
- **Memory History**: Keep history of memory usage over time

#### MemoryContext Methods

```python
# Context management
with xp.MemoryContext(device_id=0, auto_cleanup=True) as ctx:
    pass

# Memory information
mem_info = ctx.get_memory_info()  # Returns dict with memory stats
print(repr(ctx))  # String representation with current memory usage

# Cleanup methods
ctx.clear_cache()              # Basic memory pool cleanup
ctx.aggressive_cleanup()       # Force GC and cache clearing
ctx.emergency_cleanup()        # Nuclear cleanup option

# Memory monitoring
ctx.check_memory_pressure()    # Check if above threshold
ctx.auto_cleanup_if_needed()   # Auto cleanup if pressure high
ctx.monitor_memory(10.0)       # Monitor for specified duration

# Object tracking
ctx.track_object(gpu_array)    # Track GPU object for cleanup
```

#### Memory Information Structure

The `get_memory_info()` method returns:

```python
{
    "device": 0,           # GPU device ID
    "total": 8589934592,   # Total GPU memory (bytes)
    "free": 7516192768,    # Free GPU memory (bytes)
    "used": 1073741824,    # Used GPU memory (bytes)
    "memory_percent": 0.125,  # Memory usage percentage
    "pool_used": 524288,     # CuPy pool used (bytes)
    "pool_capacity": 1048576, # CuPy pool capacity (bytes)
    "pool_free": 524288       # CuPy pool free (bytes)
}
```

## Advanced Usage Examples

### Scientific Computing with Masks

```python
import xupy as xp
import numpy as np
from skimage.draw import disk

# Create astronomical image data
image = xp.random.normal(100, 10, (1024, 1024))

# Create circular mask for region of interest
mask = xp.ones((1024, 1024), dtype=bool)
center = (512, 512)
radius = 400
rr, cc = disk(center, radius)
mask[rr, cc] = False

# Create masked array
masked_image = xp.masked_array(image, mask)

# Calculate statistics excluding masked region
mean_signal = masked_image.mean()
std_signal = masked_image.std()
max_signal = masked_image.max()
```

### Large-Scale Data Processing

```python
import xupy as xp

# Process large datasets
with xp.MemoryContext() as ctx:
    # Create large masked arrays
    data = xp.random.normal(0, 1, (50000, 50000))
    mask = xp.random.random((50000, 50000)) > 0.05  # 5% masked
    ma = xp.masked_array(data, mask)
    
    # Perform computations
    result = ma.mean(axis=0)  # Mean along rows
    variance = ma.var(axis=1)  # Variance along columns
    
    # Apply mathematical transformations
    transformed = ma.exp().log()  # Complex operations
    
    ctx.clear_cache()  # Free GPU memory
```

### Interoperability with NumPy

```python
import xupy as xp
import numpy as np

# Create XuPy masked array
xupy_ma = xp.masked_array([1, 2, 3, 4, 5], [False, True, False, True, False])

# Convert to NumPy for CPU-only operations
numpy_ma = xupy_ma.asmarray()

# Use NumPy functions
result = np.ma.median(numpy_ma)

# Convert back to XuPy for GPU operations
gpu_result = xp.masked_array(numpy_ma.data, numpy_ma.mask)
```

## Best Practices

### Memory Management

1. **Use MemoryContext**: Always use context managers for large operations
2. **Clear Cache**: Call `clear_cache()` after large computations
3. **Monitor Memory**: Use `get_memory_info()` to track GPU memory usage

### Performance Optimization

1. **Batch Operations**: Process large arrays in batches if memory constrained
2. **Avoid CPU Transfers**: Keep data on GPU when possible
3. **Use GPU Ufuncs**: Prefer `apply_ufunc` for custom operations

### Error Handling

```python
import xupy as xp

try:
    # GPU operations
    with xp.MemoryContext() as ctx:
        data = xp.random.normal(0, 1, (10000, 10000))
        result = data.mean()
        
except RuntimeError as e:
    print(f"GPU operation failed: {e}")
    # Fallback to CPU operations
    data_cpu = data.asmarray()
    result = data_cpu.mean()
```

## Troubleshooting

### Common Issues

1. **CUDA Not Found**: Ensure CUDA is installed and CuPy is correctly installed
2. **Out of Memory**: Use smaller batches or clear GPU cache
3. **Slow Performance**: Check if GPU is being used (`xp.cuda.runtime.getDevice()`)

### GPU Detection

```python
import xupy as xp

# Check GPU availability
try:
    device_count = xp.cuda.runtime.getDeviceCount()
    print(f"Found {device_count} GPU(s)")
    
    for i in range(device_count):
        props = xp.cuda.runtime.getDeviceProperties(i)
        print(f"Device {i}: {props['name'].decode()}")
        
except Exception as e:
    print(f"No GPU detected: {e}")
```

## API Compatibility

XuPy maintains high compatibility with NumPy's masked array interface:

- All standard properties (`shape`, `dtype`, `size`, `ndim`, `T`)
- Comprehensive arithmetic operations with mask propagation
- Array manipulation methods (`reshape`, `transpose`, `squeeze`)
- Statistical methods (`mean`, `std`, `var`, `min`, `max`)
- Conversion to NumPy masked arrays via `asmarray()`

## Contributing

Contributions are welcome! Please see the main README for contribution guidelines.

## License

MIT License - see LICENSE file for details.

## Citation

If you use XuPy in your research, please cite:

```bibtex
@software{xupy2024,
  title={XuPy: GPU-Accelerated Masked Arrays for Scientific Computing},
  author={Ferraiuolo, Pietro},
  year={2024},
  url={https://github.com/pietroferraiuolo/XuPy}
}
```
