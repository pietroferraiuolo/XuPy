import numpy as _np
from xupy import _typings as _t
from builtins import any as _any

_GPU = False

try:
    from cupy import *              # type: ignore
    import cupy as _xp

    gpu = _xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    _GPU = True
    print(
        f"""
[XuPy] Device {_xp.cuda.runtime.getDevice()} available - GPU : `{gpu}`
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
        def __init__(self, data:_t.ArrayLike, mask:_t.ArrayLike = None, dtype: _t.DTypeLike = None):
            self.data = _xp.asarray(data, dtype=dtype if dtype else _xp.float32)
            if mask is None:
                self.mask = _xp.zeros(self.data.shape, dtype=bool)
            else:
                self.mask = _xp.asarray(mask, dtype=bool)
            if self.mask.shape != self.data.shape:
                raise ValueError("Mask shape must match data shape.") 

        def __repr__(self) -> str:
            data = _xp.asnumpy(self.data)
            mask = _xp.asnumpy(self.mask)
            display = data.astype(object)
            display[mask] = "--"
            return f"XupyMaskedArray(\ndata=\n{display},\nmask=\n{mask}\n)"

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
