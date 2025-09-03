import numpy as _np
from xupy import _typings as _t
from builtins import any as _any
from typing import Optional, Union

_GPU = False

try:
    from cupy import *              # type: ignore
    import cupy as _xp

    gpu = _xp.cuda.runtime.getDeviceProperties(0)
    gpu_name = gpu['name'].decode()
    _GPU = True
    print(
        f"""
[XuPy] Device {_xp.cuda.runtime.getDevice()} available - GPU : `{gpu_name}`
[XuPy] Memory = {_xp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024 * 1000):.2f} MB | Compute Capability = {_xp.cuda.runtime.getDeviceProperties(0)['major']}.{_xp.cuda.runtime.getDeviceProperties(0)['minor']}
[XuPy] Using CuPy {_xp.__version__} for acceleration."""
    )
    a = _xp.array([1, 2, 3]) # test array
    del a # cleanup
except Exception as err:
    print(err)
    print("[XuPy] No GPU accelerators found. Fallback to NumPy instead.")
    _GPU = False # just to be sure ...
    from numpy import *         # type: ignore


if _GPU:

    class _XupyMaskedArray:
        """
        A comprehensive masked array wrapper for CuPy arrays with NumPy-like interface.

        Parameters
        ----------
        data : array-like
            The data array (will be converted to CuPy array).
        mask : array-like
            Boolean mask array (True means masked).
        dtype : data-type, optional
            Desired data type for the data array.
        """
        def __init__(self, data:_t.ArrayLike, mask:_t.ArrayLike = None, dtype: _t.DTypeLike = None):
            self.data = _xp.asarray(data, dtype=dtype if dtype else _xp.float32)
            if hasattr(data, 'mask') and data.mask is not None and mask is None:
                mask = _xp.asarray(data.mask, dtype=bool)
            else:
                if mask is None:
                    self.mask = _xp.zeros(self.data.shape, dtype=bool)
                else:
                    self.mask = _xp.asarray(mask, dtype=bool)
                if self.mask.shape != self.data.shape:
                    raise ValueError("Mask shape must match data shape.") 

        # --- Core Properties ---
        @property
        def shape(self) -> tuple[int, ...]:
            """Return the shape of the array."""
            return self.data.shape
        
        @property
        def dtype(self):
            """Return the data type of the array."""
            return self.data.dtype
        
        @property
        def size(self) -> int:
            """Return the total number of elements."""
            return self.data.size
        
        @property
        def ndim(self) -> int:
            """Return the number of dimensions."""
            return self.data.ndim
        
        @property
        def T(self):
            """Return the transpose of the array."""
            return _XupyMaskedArray(self.data.T, self.mask.T)
        
        @property
        def flat(self):
            """Return a flat iterator over the array."""
            return self.data.flat

        def __repr__(self) -> str:
            data = _xp.asnumpy(self.data)
            mask = _xp.asnumpy(self.mask)
            display = data.astype(object)
            display[mask] = "--"
            return f"XupyMaskedArray(\ndata=\n{display},\nmask=\n{mask}\n)"
        
        def __str__(self) -> str:
            data = _xp.asnumpy(self.data)
            mask = _xp.asnumpy(self.mask)
            display = data.astype(object)
            display[mask == True] = "--"
            return f"{display}"
        
        # --- Array Manipulation Methods ---
        def reshape(self, *shape: int) -> "_XupyMaskedArray":
            """Return a new array with the same data but a new shape."""
            new_data = self.data.reshape(*shape)
            new_mask = self.mask.reshape(*shape)
            return _XupyMaskedArray(new_data, new_mask)
        
        def flatten(self, order: str = 'C') -> "_XupyMaskedArray":
            """Return a copy of the array collapsed into one dimension."""
            new_data = self.data.flatten(order=order)
            new_mask = self.mask.flatten(order=order)
            return _XupyMaskedArray(new_data, new_mask)
        
        def ravel(self, order: str = 'C') -> "_XupyMaskedArray":
            """Return a flattened array."""
            return self.flatten(order=order)
        
        def squeeze(self, axis: Optional[tuple[int, ...]] = None) -> "_XupyMaskedArray":
            """Remove single-dimensional entries from the shape of an array."""
            new_data = self.data.squeeze(axis=axis)
            new_mask = self.mask.squeeze(axis=axis)
            return _XupyMaskedArray(new_data, new_mask)
        
        def expand_dims(self, axis: int) -> "_XupyMaskedArray":
            """Expand the shape of an array by inserting a new axis."""
            new_data = _xp.expand_dims(self.data, axis=axis)
            new_mask = _xp.expand_dims(self.mask, axis=axis)
            return _XupyMaskedArray(new_data, new_mask)
        
        def transpose(self, *axes: int) -> "_XupyMaskedArray":
            """Return an array with axes transposed."""
            new_data = self.data.transpose(*axes)
            new_mask = self.mask.transpose(*axes)
            return _XupyMaskedArray(new_data, new_mask)
        
        def swapaxes(self, axis1: int, axis2: int) -> "_XupyMaskedArray":
            """Return an array with axis1 and axis2 interchanged."""
            new_data = self.data.swapaxes(axis1, axis2)
            new_mask = self.mask.swapaxes(axis1, axis2)
            return _XupyMaskedArray(new_data, new_mask)
        
        def repeat(self, repeats: Union[int, _t.ArrayLike], axis: Optional[int] = None) -> "_XupyMaskedArray":
            """Repeat elements of an array."""
            new_data = _xp.repeat(self.data, repeats, axis=axis)
            new_mask = _xp.repeat(self.mask, repeats, axis=axis)
            return _XupyMaskedArray(new_data, new_mask)
        
        def tile(self, reps: Union[int, tuple[int, ...]]) -> "_XupyMaskedArray":
            """Construct an array by repeating A the number of times given by reps."""
            new_data = _xp.tile(self.data, reps)
            new_mask = _xp.tile(self.mask, reps)
            return _XupyMaskedArray(new_data, new_mask)


        # --- Statistical Methods (Memory-Optimized) ---
        def mean(self, **kwargs: dict[str,_t.Any]) -> _t.Scalar:
            """Compute the arithmetic mean along the specified axis."""
            own = self.asmarray()
            result = own.mean(**kwargs)
            return result
        
        def sum(self, **kwargs: dict[str,_t.Any]) -> _t.Scalar:
            """Sum of array elements over a given axis."""
            own = self.asmarray()
            result = own.sum(**kwargs)
            return result
        
        def std(self, **kwargs: dict[str,_t.Any]) -> _t.Scalar:
            """Compute the standard deviation along the specified axis."""
            own = self.asmarray()
            result = own.std(**kwargs)
            return result
        
        def var(self, **kwargs: dict[str,_t.Any]) -> _t.Scalar:
            """Compute the variance along the specified axis."""
            own = self.asmarray()
            result = own.var(**kwargs)
            return result
        
        def min(self, **kwargs: dict[str,_t.Any]) -> _t.Scalar:
            """Return the minimum along a given axis."""
            own = self.asmarray()
            result = own.min(**kwargs)
            return result
        
        def max(self, **kwargs: dict[str,_t.Any]) -> _t.Scalar:
            """Return the maximum along a given axis."""
            own = self.asmarray()
            result = own.max(**kwargs)
            return result

        # --- Universal Functions Support ---
        def apply_ufunc(self, ufunc, *args, **kwargs) -> "_XupyMaskedArray":
            """Apply a universal function to the array, respecting masks."""
            # Apply ufunc to data
            result_data = ufunc(self.data, *args, **kwargs)
            result_mask = _np.where(_np.isnan(result_data), True, self.mask)
            # Preserve mask
            return _XupyMaskedArray(result_data, result_mask)
        
        def sqrt(self) -> "_XupyMaskedArray":
            """Return the positive square-root of an array, element-wise."""
            return self.apply_ufunc(_xp.sqrt)
        
        def exp(self) -> "_XupyMaskedArray":
            """Calculate the exponential of all elements in the input array."""
            return self.apply_ufunc(_xp.exp)
        
        def log(self) -> "_XupyMaskedArray":
            """Natural logarithm, element-wise."""
            return self.apply_ufunc(_xp.log)
        
        def log10(self) -> "_XupyMaskedArray":
            """Return the base 10 logarithm of the input array, element-wise."""
            return self.apply_ufunc(_xp.log10)
        
        def sin(self) -> "_XupyMaskedArray":
            """Trigonometric sine, element-wise."""
            return self.apply_ufunc(_xp.sin)
        
        def cos(self) -> "_XupyMaskedArray":
            """Cosine element-wise."""
            return self.apply_ufunc(_xp.cos)
        
        def tan(self) -> "_XupyMaskedArray":
            """Compute tangent element-wise."""
            return self.apply_ufunc(_xp.tan)
        
        def arcsin(self) -> "_XupyMaskedArray":
            """Inverse sine, element-wise."""
            return self.apply_ufunc(_xp.arcsin)
        
        def arccos(self) -> "_XupyMaskedArray":
            """Inverse cosine, element-wise."""
            return self.apply_ufunc(_xp.arccos)

        def arctan(self) -> "_XupyMaskedArray":
            """Inverse tangent, element-wise."""
            return self.apply_ufunc(_xp.arctan)
        
        def sinh(self) -> "_XupyMaskedArray":
            """Hyperbolic sine, element-wise."""
            return self.apply_ufunc(_xp.sinh)
        
        def cosh(self) -> "_XupyMaskedArray":
            """Hyperbolic cosine, element-wise."""
            return self.apply_ufunc(_xp.cosh)
        
        def tanh(self) -> "_XupyMaskedArray":
            """Compute hyperbolic tangent element-wise."""
            return self.apply_ufunc(_xp.tanh)
        
        def floor(self) -> "_XupyMaskedArray":
            """Return the floor of the input, element-wise."""
            return self.apply_ufunc(_xp.floor)
        
        def ceil(self) -> "_XupyMaskedArray":
            """Return the ceiling of the input, element-wise."""
            return self.apply_ufunc(_xp.ceil)
        
        def round(self, decimals: int = 0) -> "_XupyMaskedArray":
            """Evenly round to the given number of decimals."""
            return self.apply_ufunc(_xp.round, decimals=decimals)

        # --- Array Information Methods ---
        def any(self, **kwargs: dict[str,_t.Any]) -> bool:
            """Test whether any array element along a given axis evaluates to True."""
            own = self.asmarray()
            result = own.any(**kwargs)
            return result
        
        def all(self, **kwargs: dict[str,_t.Any]) -> bool:
            """Test whether all array elements along a given axis evaluate to True."""
            own = self.asmarray()
            result = own.all(**kwargs)
            return result
        
        def count_masked(self) -> int:
            """Return the number of masked elements."""
            return int(_xp.sum(self.mask))
        
        def count_unmasked(self) -> int:
            """Return the number of unmasked elements."""
            return int(_xp.sum(~self.mask))
        
        def is_masked(self) -> bool:
            """Return True if the array has any masked values."""
            return bool(_xp.any(self.mask))
        
        def compressed(self) -> _xp.ndarray:
            """Return all the non-masked data as a 1-D array."""
            return self.data[~self.mask]
        
        def fill_value(self, value: _t.Scalar) -> None:
            """Set the fill value for masked elements."""
            self.data[self.mask] = value


        # --- Copy and Conversion Methods ---
        def copy(self, order: str = 'C') -> "_XupyMaskedArray":
            """Return a copy of the array."""
            return _XupyMaskedArray(self.data.copy(order=order), self.mask.copy(order=order))
        
        def astype(self, dtype: _t.DTypeLike, order: str = 'K', casting: str = 'unsafe', 
                  subok: bool = True, copy: bool = True) -> "_XupyMaskedArray":
            """Copy of the array, cast to a specified type."""
            new_data = self.data.astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
            new_mask = self.mask.copy() if copy else self.mask
            return _XupyMaskedArray(new_data, new_mask, dtype=dtype)
        
        def tolist(self) -> list:
            """Return the array as a nested list."""
            return self.data.tolist()
        
        def item(self, *args: int) -> _t.Scalar:
            """Copy an element of an array to a standard Python scalar and return it."""
            own = self.asmarray()
            result = own.item(*args)
            return result


        # --- Arithmetic Operators ---
        def __radd__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected element-wise addition with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own.__radd__(other)
            return _XupyMaskedArray(result.data, result.mask)

        def __iadd__(self, other: object) -> "_XupyMaskedArray":
            """
            In-place element-wise addition with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to add.

            Returns
            -------
            _XupyMaskedArray
                The updated masked array.
            """
            if isinstance(other, _XupyMaskedArray):
                self.data += other.data
                self.mask |= other.mask
            else:
                self.data += other
            return self

        def __rsub__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected element-wise subtraction with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own.__rsub__(other)
            return _XupyMaskedArray(result.data, result.mask)

        def __isub__(self, other: object) -> "_XupyMaskedArray":
            """
            In-place element-wise subtraction with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to subtract.

            Returns
            -------
            _XupyMaskedArray
                The updated masked array.
            """
            if isinstance(other, _XupyMaskedArray):
                self.data -= other.data
                self.mask |= other.mask
            else:
                self.data -= other
            return self

        def __rmul__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected element-wise multiplication with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own.__rmul__(other)
            return _XupyMaskedArray(result.data, result.mask)

        def __imul__(self, other: object) -> "_XupyMaskedArray":
            """
            In-place element-wise multiplication with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to multiply.

            Returns
            -------
            _XupyMaskedArray
                The updated masked array.
            """
            if isinstance(other, _XupyMaskedArray):
                self.data *= other.data
                self.mask |= other.mask
            else:
                self.data *= other
            return self

        def __rtruediv__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected element-wise true division with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own.__rtruediv__(other)
            return _XupyMaskedArray(result.data, result.mask)

        def __itruediv__(self, other: object) -> "_XupyMaskedArray":
            """
            In-place element-wise true division with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to divide by.

            Returns
            -------
            _XupyMaskedArray
                The updated masked array.
            """
            if isinstance(other, _XupyMaskedArray):
                self.data /= other.data
                self.mask |= other.mask
            else:
                self.data /= other
            return self

        def __rfloordiv__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected element-wise floor division with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own.__rfloordiv__(other)
            return _XupyMaskedArray(result.data, result.mask)

        def __ifloordiv__(self, other: object) -> "_XupyMaskedArray":
            """
            In-place element-wise floor division with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to divide by.

            Returns
            -------
            _XupyMaskedArray
                The updated masked array.
            """
            if isinstance(other, _XupyMaskedArray):
                self.data //= other.data
                self.mask |= other.mask
            else:
                self.data //= other
            return self

        def __rmod__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected element-wise modulo operation with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own.__rmod__(other)
            return _XupyMaskedArray(result.data, result.mask)

        def __imod__(self, other: object) -> "_XupyMaskedArray":
            """
            In-place element-wise modulo operation with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to modulo by.

            Returns
            -------
            _XupyMaskedArray
                The updated masked array.
            """
            if isinstance(other, _XupyMaskedArray):
                self.data %= other.data
                self.mask |= other.mask
            else:
                self.data %= other
            return self

        def __rpow__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected element-wise exponentiation with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own.__rpow__(other)
            return _XupyMaskedArray(result.data, result.mask)

        def __ipow__(self, other: object) -> "_XupyMaskedArray":
            """
            In-place element-wise exponentiation with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to exponentiate by.

            Returns
            -------
            _XupyMaskedArray
                The updated masked array.
            """
            if isinstance(other, _XupyMaskedArray):
                self.data **= other.data
                self.mask |= other.mask
            else:
                self.data **= other
            return self

        # --- Matrix Multiplication ---
        def __matmul__(self, other: object) -> "_XupyMaskedArray":
            """
            Matrix multiplication with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to matrix-multiply with.

            Returns
            -------
            _XupyMaskedArray
                The result of the matrix multiplication with combined mask.
            """
            if isinstance(other, (_XupyMaskedArray, _np.ma.masked_array)):
                result_data = self.data @ other.data
                result_mask = self.mask | other.mask
            else:
                result_data = self.data @ other
                result_mask = self.mask
            return _XupyMaskedArray(result_data, result_mask)

        def __rmatmul__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected matrix multiplication with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = other @ own
            return _XupyMaskedArray(result.data, result.mask)

        def __imatmul__(self, other: object) -> "_XupyMaskedArray":
            """
            In-place matrix multiplication with mask propagation.

            Parameters
            ----------
            other : object
                The value or array to matrix-multiply with.

            Returns
            -------
            _XupyMaskedArray
                The updated masked array.
            """
            if isinstance(other, _XupyMaskedArray):
                self.data = self.data @ other.data
                self.mask = self.mask | other.mask
            else:
                self.data = self.data @ other
                # mask unchanged
            return self

        # --- Unary Operators ---
        def __neg__(self) -> "_XupyMaskedArray":
            """
            Element-wise negation with mask propagation.
            """
            result = -self.data
            return _XupyMaskedArray(result, self.mask)

        def __pos__(self) -> "_XupyMaskedArray":
            """
            Element-wise unary plus with mask propagation.
            """
            result = +self.data
            return _XupyMaskedArray(result, self.mask)

        def __abs__(self) -> "_XupyMaskedArray":
            """
            Element-wise absolute value with mask propagation.
            """
            result = _xp.abs(self.data)
            return _XupyMaskedArray(result, self.mask)

        # --- Comparison Operators (optional for mask logic) ---
        def __eq__(self, other: object) -> _xp.ndarray:
            """
            Element-wise equality comparison.

            Returns
            -------
            xp.ndarray
                Boolean array with the result of the comparison.
            """
            if isinstance(other, _XupyMaskedArray):
                return self.data == other.data
            return self.data == other

        def __ne__(self, other: object) -> _xp.ndarray:
            """
            Element-wise inequality comparison.

            Returns
            -------
            xp.ndarray
                Boolean array with the result of the comparison.
            """
            if isinstance(other, _XupyMaskedArray):
                return self.data != other.data
            return self.data != other

        def __lt__(self, other: object) -> _xp.ndarray:
            """
            Element-wise less-than comparison.

            Returns
            -------
            xp.ndarray
                Boolean array with the result of the comparison.
            """
            if isinstance(other, _XupyMaskedArray):
                return self.data < other.data
            return self.data < other

        def __le__(self, other: object) -> _xp.ndarray:
            """
            Element-wise less-than-or-equal comparison.

            Returns
            -------
            xp.ndarray
                Boolean array with the result of the comparison.
            """
            if isinstance(other, _XupyMaskedArray):
                return self.data <= other.data
            return self.data <= other

        def __gt__(self, other: object) -> _xp.ndarray:
            """
            Element-wise greater-than comparison.

            Returns
            -------
            xp.ndarray
                Boolean array with the result of the comparison.
            """
            if isinstance(other, _XupyMaskedArray):
                return self.data > other.data
            return self.data > other

        def __ge__(self, other: object) -> _xp.ndarray:
            """
            Element-wise greater-than-or-equal comparison.

            Returns
            -------
            xp.ndarray
                Boolean array with the result of the comparison.
            """
            if isinstance(other, _XupyMaskedArray):
                return self.data >= other.data
            return self.data >= other

        def __mul__(self, other: object):
            """
            Element-wise matrix multiplicationwithmask propagation
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own * other
            return _XupyMaskedArray(result.data, result.mask)

        def __truediv__(self, other: object):
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own / other
            return _XupyMaskedArray(result.data, result.mask)
        
        def __add__(self, other: object) -> "_XupyMaskedArray":
            """
            Element-wise addition with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own + other
            return _XupyMaskedArray(result.data, result.mask)

        def __sub__(self, other: object) -> "_XupyMaskedArray":
            """
            Element-wise subtraction with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own - other
            return _XupyMaskedArray(result.data, result.mask)

        def __pow__(self, other: object) -> "_XupyMaskedArray":
            """
            Element-wise exponentiation with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own ** other
            return _XupyMaskedArray(result.data, result.mask)

        def __floordiv__(self, other: object) -> "_XupyMaskedArray":
            """
            Element-wise floor division with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own // other
            return _XupyMaskedArray(result.data, result.mask)

        def __mod__(self, other: object) -> "_XupyMaskedArray":
            """
            Element-wise modulo operation with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = own % other
            return _XupyMaskedArray(result.data, result.mask)
        
        def __getattr__(self, key: str):
            """Get attribute from the underlying CuPy array."""
            if hasattr(self.data, key):
                return getattr(self.data, key)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        
        def __getitem__(self, item: slice) -> "_XupyMaskedArray":
            """
            Get item(s) from the masked array, preserving the mask.

            Parameters
            ----------
            item : int, slice, or array-like
                The index or slice to retrieve.

            Returns
            -------
            _XupyMaskedArray or scalar
                The indexed masked array or scalar value if the result is 0-dimensional.
            """
            data_item = self.data[item]
            mask_item = self.mask[item]
            # If the result is a scalar, return a masked value
            if data_item.shape == ():
                if mask_item:
                    return _np.ma.masked
                return data_item.item()
            return _XupyMaskedArray(data_item, mask_item)

        def asmarray(self, **kwargs: dict[str,_t.Any]) -> _np.ma.MaskedArray[_t.Any,_t.Any]:
            """Return a NumPy masked array on CPU."""
            return _np.ma.masked_array(_xp.asnumpy(self.data), mask=_xp.asnumpy(self.mask), **kwargs)

    MaskedArray = _XupyMaskedArray

    # --- GPU Memory Management Context Manager ---
    class MemoryContext:
        """Context manager for efficient GPU memory management."""
        
        def __init__(self, device_id: Optional[int] = None):
            self.device_id = device_id
            self.original_device = None
        
        def __enter__(self):
            if _GPU and self.device_id is not None:
                self.original_device = _xp.cuda.runtime.getDevice()
                _xp.cuda.runtime.setDevice(self.device_id)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if _GPU and self.original_device is not None:
                _xp.cuda.runtime.setDevice(self.original_device)
        
        def clear_cache(self):
            """Clear GPU memory cache to free up memory."""
            if _GPU:
                _xp.get_default_memory_pool().free_all_blocks()
                _xp.get_default_pinned_memory_pool().free_all_blocks()
        
        def get_memory_info(self) -> dict:
            """Get current GPU memory usage information."""
            if not _GPU:
                return {"error": "No GPU available"}
            
            try:
                mempool = _xp.get_default_memory_pool()
                
                return {
                    "total": _xp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"],
                    "used": mempool.used_bytes(),
                    "free": mempool.total_bytes() - mempool.used_bytes(),
                }
            except Exception as e:
                return {"error": str(e)}

    def masked_array(
        data: _t.NDArray[_t.Any],
        mask: _np.ndarray[_t.ArrayLike,_t.Any] = None,
        **kwargs: dict[_t.Any, _t.Any],
    ) -> _t.XupyMaskedArray:
        """
        Create an N-dimensional masked array with GPU support.
        
        The class `XupyMaskedArray` is a wrapper of `cupy.ndarray` with
        additional functionality for handling masked arrays on the GPU.
        It defines the additional property `mask`, which can be an array of booleans or integers,
        where `True` indicates a masked value.
        
        Parameters
        ----------
        data : NDArray[Any]
            The data to be stored in the masked array.
        mask : ArrayLike[bool|int], optional
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
        return _XupyMaskedArray(data, mask, **kwargs)
