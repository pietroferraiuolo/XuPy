# XuPy

XuPy is a Python package intended to make easier the use numpy and cupy on systems with gpu acceleration.
It also offers an easy interface for masked arrays on gpu.

## Installation

```bash
pip install .
```

## Usage

```python
import xupy as xp
from skimage.draw import disk

a = xp.random.normal(1, 3, (100,100))
b = xp.random.normal(0, 1, a.shape)

mask = xp.ones(a.shape)
masked = disk((500,500), 256)
mask[masked] = 0

am = xp.masked_array(data = a, mask=mask)
bm = xp.masked_array(data = b, mask=mask)
```

## License

See [LICENSE](LICENSE).
