import numpy as _np
from xupy import _typings as _t

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
    if isinstance(e, ImportError):
        print("[XuPy] No GPU accelerators found. Fallback to NumPy instead.")
        _GPU = False
        from numpy import *


class _XuPyMaskedArray(_xp.ndarray):
    """A masked array that supports GPU acceleration with CuPy."""

    def __init__(
        self,
        data: _np.ndarray[_t.Any,_t.Any],
        mask: _np.ndarray[bool | int, _t.Any] = None,
        **kwargs: dict[_t.Any, _t.Any],
    ) -> None:
        """The constructor"""
        if isinstance(data, _xp.ndarray):
            data = _xp.asnumpy(data)
        super().__init__(data, **kwargs)
        if mask is None:
            mask = _xp.zeros(data.shape, dtype=_xp.bool_)
        elif isinstance(mask, _np.ndarray):
            mask = _xp.asarray(mask, dtype=_xp.bool_)
        self._mask = mask

    @property
    def _mask(self):
        """Get the mask of the array."""
        return self.__dict__.get("_mask", _xp.zeros(self.shape, dtype=_xp.bool_))

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
