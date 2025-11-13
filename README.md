# XuPy

![logo](docs/logo.png)

XuPy is a comprehensive Python package that provides GPU-accelerated masked arrays and NumPy-compatible functionality using CuPy. It automatically handles GPU/CPU fallback and offers an intuitive interface for scientific computing with masked data.

## Features

- **GPU Acceleration**: Automatic GPU detection with CuPy fallback to NumPy
- **Masked Arrays**: Full support for masked arrays with GPU acceleration
- **Statistical Functions**: Comprehensive statistical operations (mean, std, var, min, max, etc.)
- **Array Manipulation**: Reshape, transpose, squeeze, expand_dims, and more
- **Mathematical Functions**: Trigonometric, exponential, logarithmic, and rounding functions
- **Random Generation**: Various random number generators (normal, uniform, etc.)
- **Universal Functions**: Support for applying any CuPy/NumPy ufunc with mask preservation
- **Performance**: Optimized for large-scale data processing on GPU

## Installation

```bash
pip install xupy
```

## Quick Start

```python
import xupy as xp

# Create arrays with automatic GPU detection
a = xp.random.normal(0, 1, (1000, 1000))
b = xp.random.normal(0, 1, (1000, 1000))

# Create masks
mask = xp.random.random((1000, 1000)) > 0.1

# Create masked arrays
am = xp.ma.masked_array(a, mask)
bm = xp.ma.masked_array(b, mask)

# Perform operations (masks are automatically handled)
result = am + bm
mean_val = am.mean()
std_val = am.std()
```

## Performance Benefits

XuPy automatically detects GPU availability and provides significant speedup for large arrays:

- **Small arrays (< 1000 elements)**: CPU (NumPy) may be faster due to GPU overhead
- **Medium arrays (1000-10000 elements)**: GPU provides 2-5x speedup
- **Large arrays (> 10000 elements)**: GPU provides 5-20x speedup depending on operation complexity

## GPU Requirements

- **CUDA-compatible GPU** with compute capability 3.0+
- **CuPy** package installed (`pip install cupy-cuda12x` for CUDA 12.x)
- **Automatic fallback** to NumPy if GPU is unavailable

## API Compatibility

XuPy maintains high compatibility with NumPy's masked array interface while leveraging CuPy's optimized operations:

- All standard properties (`shape`, `dtype`, `size`, `ndim`, `T`)
- Comprehensive arithmetic operations with mask propagation
- **Memory-optimized statistical methods** (`mean`, `std`, `var`, `min`, `max`) using CuPy's native operations
- Array manipulation methods (`reshape`, `transpose`, `squeeze`)
- Universal function support through `apply_ufunc`
- Conversion to NumPy masked arrays via `asmarray()`
- **GPU memory management** through `MemoryContext`

## GPU Memory Management

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

### MemoryContext Features

- **Automatic Cleanup**: Memory freed automatically when exiting context
- **Memory Monitoring**: Real-time tracking of GPU memory usage
- **Pressure Detection**: Automatic cleanup when memory usage is high
- **Aggressive Cleanup**: Force garbage collection and cache clearing
- **Emergency Cleanup**: Nuclear option for out-of-memory situations
- **Object Tracking**: Track GPU objects for proper cleanup
- **Memory History**: Keep history of memory usage over time

## Documentation

For detailed documentation, including comprehensive API reference and advanced usage examples, see [docs.md](docs.md).

## License

See [LICENSE](LICENSE).
