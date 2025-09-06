# Introduction

XuPy is a Python library for GPU-accelerated masked arrays. It automatically
selects CuPy when a CUDA GPU is available and falls back to NumPy otherwise,
exposing a NumPy-like interface for array operations with rich mask semantics.

## Installation

Install from source in editable mode:

```bash
pip install xupy
```

If you have CUDA installed, XuPy can guide CuPy installation at runtime. You may
also install a specific CuPy wheel explicitly (example for CUDA 12.x):

```bash
pip install cupy-cuda12x
```

## Quick start

```python
import xupy as xp

data = xp.random.normal(0, 1, (1000, 1000))
mask = xp.random.random((1000, 1000)) > 0.1
ma = xp.masked_array(data, mask)

result = (ma + 1).mean()
```

For more examples and background, see the project README on GitHub.

