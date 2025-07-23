import numpy as _np
from xupy import _typings as _t
from builtins import any as _any

try:
    from cupy import *
    import cupy as _xp

    gpu = _xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    _GPU = True
    print(
        f"""
[XuPy] Device {_xp.cuda.runtime.getDevice()} available - GPU : `{gpu}`
[XuPy] Memory = {_xp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024 * 1000):.2f} MB | Compute Capability = {_xp.cuda.runtime.getDeviceProperties(0)['major']}.{_xp.cuda.runtime.getDeviceProperties(0)['minor']}
[XuPy] Using CuPy {_xp.__version__} for acceleration."""
    )
except Exception as err:
    if isinstance(err, ImportError):
        print("[XuPy] No GPU accelerators found. Fallback to NumPy instead.")
        _GPU = False
        from numpy import *


class XupyMaskedArray:
    """
    A simple masked array wrapper for CuPy arrays.

    Parameters
    ----------
    data : array-like
        The data array (will be converted to CuPy array).
    mask : array-like
        Boolean mask array (True means masked).
    dtype : data-type, optional
        Desired data type for the data array.
    """
    def __init__(self, data, mask=None, dtype=None):
        self.data = _xp.asarray(data, dtype=dtype if dtype else _xp.float32)
        if mask is None:
            self.mask = _xp.zeros(self.data.shape, dtype=bool)
        else:
            self.mask = _xp.asarray(mask, dtype=bool)

    def __repr__(self) -> str:
        data = _xp.asnumpy(self.data)
        mask = _xp.asnumpy(self.mask)
        display = data.astype(object)
        display[mask] = "--"
        return f"XupyMaskedArray(\ndata=\n{display},\nmask=\n{mask}\n)"

    def __mul__(self, other):
        if isinstance(other, XupyMaskedArray):
            result_data = self.data * other.data
            result_mask = self.mask | other.mask
        else:
            result_data = self.data * other
            result_mask = self.mask
        return XupyMaskedArray(result_data, result_mask)

    def __truediv__(self, other):
        if isinstance(other, XupyMaskedArray):
            result_data = self.data / other.data
            result_mask = self.mask | other.mask
        else:
            result_data = self.data / other
            result_mask = self.mask
        return XupyMaskedArray(result_data, result_mask)
    
    def __getattr__(self, key):
        """Get attribute from the underlying CuPy array."""
        if hasattr(self.data, key):
            return getattr(self.data, key)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __getitem__(self, item):
        """Get item from the underlying CuPy array."""
        data_item = self.data[item]
        mask_item = self.mask[item]
        return XupyMaskedArray(data_item, mask_item)

    def asmarray(self, **kwargs):
        """Return a NumPy masked array on CPU."""
        return _np.ma.masked_array(_xp.asnumpy(self.data), mask=_xp.asnumpy(self.mask), **kwargs)
    

class _XuPyMaskedArray(_xp.ndarray):
    """A masked array that supports GPU acceleration with CuPy."""
    
    def __new__(
        cls,
        data: _np.ndarray[_t.Any, _t.Any],
        mask: _np.ndarray[bool | int, _t.Any] = None,
        **kwargs: dict[_t.Any, _t.Any],
    ) -> "_XuPyMaskedArray":
        """The constructor"""
        if isinstance(data, _xp.ndarray):
            obj = data.view(cls)
        else:
            obj = _xp.asarray(data, **kwargs).view(cls)
        if mask is None:
            mask = _xp.zeros(obj.shape, dtype=_xp.bool_)
        elif isinstance(mask, _np.ndarray):
            mask = _xp.asarray(mask, dtype=_xp.bool_)
        if mask.dtype not in (_xp.bool_, _xp.int_, _np.bool_, _np.int_):
            mask = mask.astype(_xp.bool_)
        obj._mask = mask
        return obj


    @property
    def _mask(self):
        """Get the mask of the array."""
        return self.__dict__.get("_mask", _xp.zeros(self.shape, dtype=_xp.bool_))
    
    mask = _mask

    @_mask.setter
    def _mask(self, value: _np.ndarray[bool | int, _t.Any]) -> None:
        """Set the mask of the array."""
        if isinstance(value, _np.ndarray):
            value = _xp.asarray(value, dtype=_xp.bool_)
        if value.shape != self.shape:
            raise ValueError("Mask shape must match data shape.")
        self.__dict__["_mask"] = value
    
    def __array_finalize__(self, obj):
        """Finalize the array."""
        if obj is None:
            return
        self._mask = getattr(obj, "_mask", _xp.zeros(self.shape, dtype=_xp.bool_))
        super().__array_finalize__(obj)

    def __repr__(self) -> str:
        """
        Return the official string representation of the masked array.

        Returns
        -------
        str
            The string representation of the masked array, showing masked values as '--'.
        """
        data = _xp.asnumpy(self)
        mask = _xp.asnumpy(self._mask)
        display = data.astype(object)
        display[mask] = "--"
        return (
            f"masked_array(\n"
            f"data=\n{display},\n"
            f"mask=\n{mask}\n"
            f")"
        )

    def __str__(self) -> str:
        """
        Return the informal string representation of the masked array.

        Returns
        -------
        str
            The string representation of the masked array, showing masked values as '--'.
        """
        data = _xp.asnumpy(self)
        mask = _xp.asnumpy(self._mask)
        display = data.astype(object)
        display[mask] = "--"
        return str(display)
    
    def __mul__(self, other: _t.Any) -> "_XuPyMaskedArray":
        """
        Element-wise multiplication of the masked array with another array or scalar.

        Parameters
        ----------
        other : Any
            The value to multiply with the masked array.

        Returns
        -------
        _XuPyMaskedArray
            A new masked array resulting from the element-wise multiplication.
        """
        result = super().__mul__(other)
        if isinstance(other, _XuPyMaskedArray):
            mask = self._mask | other._mask
        else:
            mask = self._mask
        return _XuPyMaskedArray(result, mask)

    def __truediv__(self, other: _t.Any) -> "_XuPyMaskedArray":
        """
        Element-wise division of the masked array with another array or scalar.

        Parameters
        ----------
        other : Any
            The value to divide the masked array by.

        Returns
        -------
        _XuPyMaskedArray
            A new masked array resulting from the element-wise division.
        """
        result = super().__truediv__(other)
        if isinstance(other, _XuPyMaskedArray):
            mask = self._mask | other._mask
        else:
            mask = self._mask
        return _XuPyMaskedArray(result, mask)

    def asmarray(self, **kwargs) -> _t.masked_array[_t.Any,_t.Any]:
        """
        Return a NumPy masked array from the GPU-backed masked array.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to `numpy.ma.masked_array`.

        Returns
        -------
        numpy.ma.MaskedArray
            The masked array on the CPU as a NumPy masked array.
        """
        data = _xp.asnumpy(self)
        mask = _xp.asnumpy(self._mask)
        return _np.ma.masked_array(data, mask=mask, **kwargs)

if _GPU:

    def masked_array(
        data: _t.NDArray[_t.Any],
        mask: _np.ndarray[bool | int, _t.Any] = None,
        **kwargs: dict[_t.Any, _t.Any],
    ) -> _t.XupyMaskedArray:
        """
        Create an N-dimensional masked array with GPU support.
        
        The class `XupyMaskedArray` is a child of `cupy.ndarray` and provides
        additional functionality for handling masked arrays on the GPU.
        It defines the additional property `mask`, which can be an array of booleans or integers,
        where `True` indicates a masked value.
        
        Parameters
        ----------
        data : NDArray[Any]
            The data to be stored in the masked array.
        mask : NDArray[bool | int, Any], optional
            The mask for the array, where `True` indicates a masked value.
            If not provided, a mask of all `False` values is created.
        **kwargs : Any    
            Additional keyword arguments to pass to the masked array constructor.
            
        Returns
        -------
        XupyMaskedArray
            A masked array with GPU support.
        """
        if isinstance(data, _np.ndarray):
            data = _xp.asarray(data, dtype=_xp.float32)
        return _XuPyMaskedArray(data, mask, **kwargs)
