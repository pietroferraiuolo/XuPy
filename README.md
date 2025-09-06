# XuPy

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
am = xp.masked_array(a, mask)
bm = xp.masked_array(b, mask)

# Perform operations (masks are automatically handled)
result = am + bm
mean_val = am.mean()
std_val = am.std()
```

## Comprehensive Examples

### Array Creation

```python
import xupy as xp

# Basic array creation
zeros = xp.zeros((3, 3))
ones = xp.ones((3, 3))
eye = xp.eye(3)
identity = xp.identity(3)

# Sequences
linspace = xp.linspace(0, 10, 100)
logspace = xp.logspace(0, 3, 100)
arange = xp.arange(0, 10, 0.5)

# Random arrays
random = xp.random((100, 100))
normal = xp.normal(0, 1, (100, 100))
uniform = xp.uniform(-1, 1, (100, 100))
```

### Masked Array Operations

```python
import xupy as xp
from skimage.draw import disk

# Create data and mask
data = xp.random.normal(0, 1, (500, 500))
mask = xp.ones((500, 500), dtype=bool)

# Create circular mask
circle_coords = disk((250, 250), 200)
mask[circle_coords] = False

# Create masked array
masked_data = xp.masked_array(data, mask)

# Statistical operations
global_mean = masked_data.mean()
global_std = masked_data.std()
global_var = masked_data.var()
global_min = masked_data.min()
global_max = masked_data.max()

# Axis-wise operations
row_means = masked_data.mean(axis=0)
col_sums = masked_data.sum(axis=1)
```

### Mathematical Functions

```python
import xupy as xp

# Create masked array
data = xp.random.normal(0, 1, (100, 100))
mask = xp.random.random((100, 100)) > 0.8
ma = xp.masked_array(data, mask)

# Trigonometric functions
sin_result = ma.sin()
cos_result = ma.cos()
tan_result = ma.tan()

# Inverse trigonometric functions
arcsin_result = ma.arcsin()
arccos_result = ma.arccos()
arctan_result = ma.arctan()

# Hyperbolic functions
sinh_result = ma.sinh()
cosh_result = ma.cosh()
tanh_result = ma.tanh()

# Exponential and logarithmic functions
exp_result = ma.exp()
log_result = ma.log()
log10_result = ma.log10()

# Rounding functions
floor_result = ma.floor()
ceil_result = ma.ceil()
round_result = ma.round(decimals=2)

# Square root
sqrt_result = ma.sqrt()
```

### Array Manipulation

```python
import xupy as xp

# Create masked array
data = xp.random.normal(0, 1, (4, 4))
mask = xp.random.random((4, 4)) > 0.5
ma = xp.masked_array(data, mask)

# Reshape
reshaped = ma.reshape(2, 8)
flattened = ma.flatten()
raveled = ma.ravel()

# Transpose and axes
transposed = ma.T
swapped = ma.swapaxes(0, 1)

# Expand and squeeze dimensions
expanded = ma.expand_dims(axis=1)
squeezed = expanded.squeeze()

# Repeat and tile
repeated = ma.repeat(2, axis=0)
tiled = ma.tile((2, 2))
```

### Advanced Operations

```python
import xupy as xp

# Create complex masked array
data = xp.random.normal(0, 1, (100, 100, 3))
mask = xp.random.random((100, 100, 3)) > 0.9
ma = xp.masked_array(data, mask)

# Multi-axis operations
result = ma.mean(axis=(0, 1))
variance = ma.var(axis=1, ddof=1)

# Boolean operations
any_true = ma.any(axis=0)
all_true = ma.all(axis=1)

# Mask information
masked_count = ma.count_masked()
unmasked_count = ma.count_unmasked()
is_masked = ma.is_masked()

# Compressed data
valid_data = ma.compressed()

# Fill masked values
ma.fill_value(0.0)
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

## Key Improvements

- **Eliminated redundant functions** - Uses CuPy/NumPy directly for basic operations
- **Memory-efficient statistical operations** - Leverages CuPy's optimized reduction operations
- **Proper mask propagation** - Maintains mask integrity across all operations
- **GPU memory management** - Context manager for efficient memory usage

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

Run the memory management demo:

```bash
python memory_demo.py
```

## Examples

Run the comprehensive examples script to see XuPy in action:

```bash
python examples.py
```

This script demonstrates:

- GPU detection and information
- Basic masked array operations
- GPU-accelerated computations
- Mathematical functions
- Memory management
- Scientific computing use cases
- Performance comparisons

## Documentation

For detailed documentation, including comprehensive API reference and advanced usage examples, see [docs.md](docs.md).

## License

See [LICENSE](LICENSE).

## Citation

If you use XuPy in your research, please cite:

```bibtex
@software{xupy2025,
  title={XuPy: GPU-Accelerated Masked Arrays for Scientific Computing},
  author={Ferraiuolo, Pietro},
  year={2024},
  url={https://github.com/pietroferraiuolo/XuPy}
}
```
