import builtins
import numpy as _np
from xupy import _typings as _t
from typing import Optional, Union
import time
from ._cupy_install import __check_availability__ as __check__

_GPU = False

__check__.xupy_init()

del __check__

try:
    from cupy import *              # type: ignore
    import cupy as _xp

    gpu = _xp.cuda.runtime.getDeviceProperties(0)
    gpu_name = gpu['name'].decode()
    _GPU = True
    print(
        f"""
[XuPy] Device {_xp.cuda.runtime.getDevice()} available - GPU : `{gpu_name}`
       Memory = {_xp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024 * 1000):.2f} MB | Compute Capability = {_xp.cuda.runtime.getDeviceProperties(0)['major']}.{_xp.cuda.runtime.getDeviceProperties(0)['minor']}
       Using CuPy {_xp.__version__} for acceleration."""
    )
    
    # Test cupy is working on the system
    import gc
    a = _xp.array([1, 2, 3]) # test array
    del a # cleanup
    gc.collect()
except Exception as err:
    print("""
[XuPy] GPU Acceleration not available. 
       Fallback to NumPy instead.""")
    _GPU = False  # just to be sure ...
    from numpy import *         # type: ignore


if _GPU:
    
    float = _xp.float32
    double = _xp.float64
    np = _np

    class _XupyMaskedArray:
        """
        Description
        ===========
        A masked-array wrapper around GPU-backed arrays (CuPy) that provides a
        NumPy-like / numpy.ma-compatible interface while preserving mask semantics
        and offering convenience methods for common array operations. This class is
        designed to let you work with large arrays on CUDA-enabled devices using
        CuPy for numerical computation while retaining the expressive masked-array
        API familiar from numpy.ma.
        
        Key features
        ------------
        - Wraps a CuPy ndarray ("data") together with a boolean mask of the same
            shape ("mask") where True indicates invalid/masked elements.
        - Lazy/convenient conversion to NumPy masked arrays for CPU-side operations
            (asmarray()) while performing heavy computation on GPU when possible.
        - Implements many common array methods and arithmetic/ufunc support with
            mask propagation semantics compatible with numpy.ma.
        - Several convenience methods for reshaping, copying, slicing and converting
            to Python lists / scalars.
        - Designed for memory-optimized operations: many reductions and logical
            tests convert to NumPy masked arrays only when necessary.
        
        Parameters
        ----------
        data : array-like
            Input array data. Accepted inputs include CuPy arrays, NumPy arrays,
            Python sequences and numpy.ma.masked_array objects. The data will be
            converted to the configured GPU array module (CuPy) on construction.
        mask : array-like, optional
            Boolean mask indicating invalid elements (True == masked). If omitted
            and `keep_mask` is True, an existing mask on an input masked_array
            (if present) will be used; otherwise the mask defaults to all False.
        dtype : dtype, optional
            Desired data-type for the stored data. If omitted, a default dtype
            (commonly float32 for GPU performance) will be used when converting
            the input data to a GPU array.
            Value to be used when filling masked elements. If None and the input
            was a numpy.ma.masked_array, the fill_value of that array will be used.
            Otherwise a dtype-dependent default is chosen (consistent with numpy.ma).
            If True (default) and the input `data` is a masked array, combine the
            input mask with the provided `mask`. If False, the provided `mask` (or
            default) is used alone.
        fill_value : scalar, optional
            Value used to fill in the masked values when necessary.
            If None, if the input `data` is a masked_array then the fill_value
            will be taken from the masked_array's fill_value attribute,
            otherwise a default based on the data-type is used.
        keep_mask : bool, optional
            Whether to combine `mask` with the mask of the input data, if any
            (True), or to use only `mask` for the output (False). Default is True.
        hard_mask : bool, optional (Not Implemented Yet)
            If True, indicates that the mask should be treated as an immutable
            "hard" mask. This influence is primarily semantic in this wrapper but
            can be used by higher-level logic to avoid accidental unmasking.
        order : {'C', 'F', 'A', None}, optional
            Memory order for array conversion if a copy is required. Behaves like
            numpy.asarray / cupy.asarray ordering.

        Attributes
        ----------
        data : cupy.ndarray
                Underlying GPU array (CuPy). Contains numeric values for both masked
                and unmasked elements. Access directly to run GPU computations.
        mask : cupy.ndarray (boolean)
                Boolean mask array with the same shape as `data`. True means the
                corresponding element is masked/invalid.
        dtype : dtype
                User-specified or inferred dtype used for conversions and some repr
                logic.
        fill_value : scalar
                Default value used when explicitly filling masked entries.
        _is_hard_mask : bool
                Internal flag indicating whether the mask is "hard" (semantically
                immutable).

                
        Mask semantics and behavior
        ---------------------------
        - The mask is always a boolean array aligned with `data`. Users can access
            and manipulate it directly (e.g. arr.mask |= other.mask) to combine masks.
        - Mask propagation follows numpy.ma semantics: arithmetic and ufuncs
            produce masks that reflect invalid operations (e.g. NaNs) and combine
            masks where appropriate.
        - Many in-place mutation operations (+=, -=, *=, /=, etc.) will update
            `data` in place and combine masks when the rhs is another masked array.
        - Some operations convert to a NumPy masked_array for convenience or to
            reuse numpy.ma utilities; this conversion copies data from GPU to CPU.
            Use asmarray() explicitly to force conversion when needed.
            
        Common methods (overview)
        -------------------------
        - reshape, flatten, ravel, squeeze, expand_dims, transpose, swapaxes,
            repeat, tile: shape-manipulation methods that preserve masks.
        - mean, sum, std, var, min, max: reductions implemented by converting to
            numpy.ma.MaskedArray via asmarray() for accuracy and mask-awareness.
        - apply_ufunc: apply a (u)func to the data while updating the mask when
            the result contains NaNs; intended for GPU-backed CuPy ufuncs.
        - sqrt, exp, log, log10, sin, cos, tan, arcsin, arccos, arctan, sinh,
            cosh, tanh, floor, ceil, round: convenience wrappers around apply_ufunc.
        - any, all: logical reductions via asmarray() to respect masked semantics.
        - count_masked, count_unmasked, is_masked, compressed: mask inspection
            and extraction utilities.
        - fill_value(value): write `value` into `data` at masked positions.
        - copy, astype: copy and cast operations preserving mask.
        - tolist, item: conversion to Python data structures / scalars.
        - __getitem__/__setitem__: indexing and slicing preserve mask shape and
            return MaskedArray views or scalars consistent with numpy.ma rules.
        - asmarray: convert to numpy.ma.MaskedArray on CPU (copies data and mask
            from GPU to host memory). Use as the bridge to CPU-only utilities.
            
        Arithmetic, ufuncs and operator behavior
        ----------------------------------------
        - Binary operations and ufuncs between _XupyMaskedArray instances will
            generally:
                - convert operands to GPU arrays when possible,
                - perform the operation on their `data`, and
                - combine masks using logical OR (|) to mark any element masked if it
                    was masked in either operand or if the operation produced NaN.
        - In-place operators (+=, -=, *=, etc.) modify `data` in place and
            perform mask combination when the RHS is a masked array.
        - Reflected operators (radd, rsub, ...) are supported; when either side
            is a masked array, mask propagation rules are applied.
        - Some operators are implemented by delegating to asmarray() which can
            cause a GPU -> CPU transfer. This is a trade-off to retain correct
            mask-aware behavior; performance-critical code should prefer explicit
            GPU-safe ufuncs when possible.
            
        Performance and memory considerations
        -------------------------------------
        - The object is optimized for GPU computation by using CuPy arrays for
            numerical work. However, some convenience operations (e.g., many
            reductions and string formatting in __repr__) convert to NumPy masked
            arrays on the host, which involves a device->host copy.
        - Avoid calling asmarray() or methods that rely on it (mean, sum, std,
            min, max, any, all, etc.) in tight GPU-bound loops unless you intend
            to move data to CPU.
        - Use apply_ufunc and the provided GPU ufunc wrappers (sqrt, exp, sin,
            etc.) to keep computation on the device and minimize data transfer.
        - Copying and type casting can allocate additional GPU memory; use views
            or in-place methods when memory is constrained.
            
        Representation and printing
        ---------------------------
        - __repr__ attempts to follow numpy.ma formatting conventions while
            displaying masked elements as a placeholder (e.g., "--") by converting
            the minimal necessary data to the host for a readable representation.
        - __str__ delegates to a masked-display conversion that replaces masked
            entries with a human-readable token. These operations involve a
            transfer from GPU to CPU.
            
        Interoperability with numpy.ma and CuPy
        --------------------------------------
        - asmarray() returns a numpy.ma.MaskedArray with the data and mask copied
            to host memory; this is useful for interoperability with NumPy APIs
            that expect masked arrays.
        - When interacting with NumPy or numpy.ma masked arrays passed as inputs,
            _XupyMaskedArray will honor existing masks (subject to keep_mask) and
            attempt to preserve semantics on the GPU.
        - When mixing with plain NumPy ndarrays or scalars, values are promoted
            to CuPy arrays for computation, and mask behavior follows numpy.ma rules
            (masked elements propagate).
            
        Examples
        --------
        Create from a NumPy array with a mask:
        >>> data = np.array([1.0, 2.0, np.nan, 4.0])
        >>> mask = np.isnan(data)
        >>> m = _XupyMaskedArray(data, mask)
        >>> m.count_masked()
        1
        >>> m + 1  # arithmetic preserves mask
        Use GPU ufuncs without moving data to CPU:
        >>> m_gpu = _XupyMaskedArray(cupy.array([0.0, 1.0, -1.0]))
        >>> m_gpu.sqrt()  # computes on GPU via apply_ufunc
        Convert to NumPy masked array for CPU-only operations:
        >>> ma = m_gpu.asmarray()
        >>> ma.mean()
        
        Notes and caveats
        -----------------
        - The wrapper is not a drop-in replacement for numpy.ma in every edge
            case; it attempts to mirror numpy.ma semantics where feasible while
            leveraging GPU acceleration.
        - Some methods intentionally convert to numpy.ma.MaskedArray for semantic
            fidelity; these are clearly documented and an explicit asmarray() call
            is recommended when you want to guarantee a CPU-side masked array.
        - Users should be mindful of device-host memory transfers when mixing
            GPU operations and mask-aware CPU computations.
            
        Extensibility
        -------------
        - The class is intended to be extended with additional ufunc wrappers,
            GPU-optimized masked reductions, and richer I/O/serialization support.
        - Because mask handling is explicit and mask arrays are plain boolean
            arrays, users can implement custom mask logic (e.g., hierarchical masks,
            multi-state masks) on top of this wrapper.
        See also
        --------
        numpy.ma.MaskedArray : Reference implementation and semantics for masked arrays.
        cupy.ndarray : GPU-backed numerical arrays used as the data store.
        
        ----
        
        
        A comprehensive masked array wrapper for CuPy arrays with NumPy-like interface.

        Parameters
        ----------
        data : array-like
            The input data array (will be converted to CuPy array).
        mask : array-like
            Mask. Must be convertible to an array of booleans with the same
            shape as `data`. True indicates a masked (i.e. invalid) data.
        dtype : data-type, optional
            Desired data type for the output array. Defaults to `float32` for optimized
            GPU performances on computations.
        fill_value : scalar, optional
            Value used to fill in the masked values when necessary.
            If None, if the input `data` is a masked_array then the fill_value
            will be taken from the masked_array's fill_value attribute,
            otherwise a default based on the data-type is used.
        keep_mask : bool, optional
            Whether to combine `mask` with the mask of the input data, if any
            (True), or to use only `mask` for the output (False). Default is True.
        order : {'C', 'F', 'A'}, optional
            Specify the order of the array.  If order is 'C', then the array
            will be in C-contiguous order (last-index varies the fastest).
            If order is 'F', then the returned array will be in
            Fortran-contiguous order (first-index varies the fastest).
            If order is 'A' (default), then the returned array may be
            in any order (either C-, Fortran-contiguous, or even discontiguous),
            unless a copy is required, in which case it will be C-contiguous.
        """

        _print_width = 100
        _print_width_1d = 1500

        def __init__(
            self, 
            data:_t.ArrayLike, 
            mask:_t.ArrayLike = None,
            dtype: _t.DTypeLike = None,
            fill_value: _t.Scalar = None,
            keep_mask: bool = True,
            hard_mask: bool = False,
            order: _t.Optional[str] = None
        ):
            """The constructor"""

            self._dtype = dtype
            self.data = _xp.asarray(data, dtype=dtype if dtype else _xp.float32, order=order)

            if mask is None:
                if keep_mask is True:
                    if hasattr(data, 'mask'):
                        try:
                            self._mask = _xp.asarray(data.mask, dtype=bool)
                        except Exception as e:
                            print(f"Failed to retrieve mask from data: {e}")
                            self._mask = _xp.zeros(self.data.shape, dtype=bool)
                    else:
                        self._mask = _xp.zeros(self.data.shape, dtype=bool)
            else:
                self._mask = _xp.asarray(mask, dtype=bool)

            self._is_hard_mask = hard_mask

            if fill_value is None:
                if hasattr(data, 'fill_value'):
                    self._fill_value = data.fill_value
                else:
                    self._fill_value = _np.ma.default_fill_value(self.data)
            else:
                self._fill_value = fill_value

        # --- Core Properties ---
        @property
        def mask(self) -> _xp.ndarray:
            """Return the mask array."""
            return self._mask

        @mask.setter
        def mask(self, value: _xp.ndarray) -> None:
            """Set the mask array."""
            self._mask = value

        @property
        def shape(self) -> tuple[int, ...]:
            """Return the shape of the array."""
            return self.data.shape

        @property
        def dtype(self):
            """Return the data type of the array."""
            return self._dtype

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
            return _XupyMaskedArray(self.data.T, self._mask.T)

        @property
        def flat(self):
            """Return a flat iterator over the array."""
            return self.data.flat

        def __repr__(self) -> str:
            """string representation
            
            Code adapted from NumPy official API
            https://github.com/numpy/numpy/blob/main/numpy/ma/core.py
            """
            import builtins 
            
            prefix = f"xupy_masked_array("

            dtype_needed = (
                not _np.core.arrayprint.dtype_is_implied(self.dtype) or
                _np.all(self._mask) or
                self.size == 0
            )

            # determine which keyword args need to be shown
            keys = ['data', 'mask']
            if dtype_needed:
                keys.append('dtype')

            # array has only one row (non-column)
            is_one_row = builtins.all(dim == 1 for dim in self.shape[:-1])

            # choose what to indent each keyword with
            min_indent = 4
            if is_one_row:
                # first key on the same line as the type, remaining keys
                # aligned by equals
                indents = {}
                indents[keys[0]] = prefix
                for k in keys[1:]:
                    n = builtins.max(min_indent, len(prefix + keys[0]) - len(k))
                    indents[k] = ' ' * n
                prefix = ''  # absorbed into the first indent
            else:
                # each key on its own line, indented by two spaces
                indents = {k: ' ' * min_indent for k in keys}
                prefix = prefix + '\n'  # first key on the next line

            # format the field values
            reprs = {}
            reprs['data'] = _np.array2string(
                self._insert_masked_print(),
                separator=", ",
                prefix=indents['data'] + 'data=',
                suffix=',')
            reprs['mask'] = _np.array2string(
                _xp.asnumpy(self._mask),
                separator=", ",
                prefix=indents['mask'] + 'mask=',
                suffix=',')
            if dtype_needed:
                reprs['dtype'] = _np.core.arrayprint.dtype_short_repr(self.dtype)

            # join keys with values and indentations
            result = ',\n'.join(
                '{}{}={}'.format(indents[k], k, reprs[k])
                for k in keys
            )
            return prefix + result + ')'
            
        def __str__(self) -> str:
            # data = _xp.asnumpy(self.data)
            # mask = _xp.asnumpy(self._mask)
            # display = data.astype(object)
            # display[mask == True] = "--"
            return self._insert_masked_print().__str__()
        
        def _insert_masked_print(self):
            """
            Replace masked values with masked_print_option, casting all innermost
            dtypes to object.
            """
            data = _xp.asnumpy(self.data)
            mask = _xp.asnumpy(self._mask)
            display = data.astype(object)
            display[mask] = "--"
            return display
        
        # --- Array Manipulation Methods ---
        def reshape(self, *shape: int) -> "_XupyMaskedArray":
            """Return a new array with the same data but a new shape."""
            new_data = self.data.reshape(*shape)
            new_mask = self._mask.reshape(*shape)
            return _XupyMaskedArray(new_data, new_mask)
        
        def flatten(self, order: str = 'C') -> "_XupyMaskedArray":
            """Return a copy of the array collapsed into one dimension."""
            new_data = self.data.flatten(order=order)
            new_mask = self._mask.flatten(order=order)
            return _XupyMaskedArray(new_data, new_mask)
        
        def ravel(self, order: str = 'C') -> "_XupyMaskedArray":
            """Return a flattened array."""
            return self.flatten(order=order)
        
        def squeeze(self, axis: Optional[tuple[int, ...]] = None) -> "_XupyMaskedArray":
            """Remove single-dimensional entries from the shape of an array."""
            new_data = self.data.squeeze(axis=axis)
            new_mask = self._mask.squeeze(axis=axis)
            return _XupyMaskedArray(new_data, new_mask)
        
        def expand_dims(self, axis: int) -> "_XupyMaskedArray":
            """Expand the shape of an array by inserting a new axis."""
            new_data = _xp.expand_dims(self.data, axis=axis)
            new_mask = _xp.expand_dims(self._mask, axis=axis)
            return _XupyMaskedArray(new_data, new_mask)
        
        def transpose(self, *axes: int) -> "_XupyMaskedArray":
            """Return an array with axes transposed."""
            new_data = self.data.transpose(*axes)
            new_mask = self._mask.transpose(*axes)
            return _XupyMaskedArray(new_data, new_mask)
        
        def swapaxes(self, axis1: int, axis2: int) -> "_XupyMaskedArray":
            """Return an array with axis1 and axis2 interchanged."""
            new_data = self.data.swapaxes(axis1, axis2)
            new_mask = self._mask.swapaxes(axis1, axis2)
            return _XupyMaskedArray(new_data, new_mask)
        
        def repeat(self, repeats: Union[int, _t.ArrayLike], axis: Optional[int] = None) -> "_XupyMaskedArray":
            """Repeat elements of an array."""
            new_data = _xp.repeat(self.data, repeats, axis=axis)
            new_mask = _xp.repeat(self._mask, repeats, axis=axis)
            return _XupyMaskedArray(new_data, new_mask)
        
        def tile(self, reps: Union[int, tuple[int, ...]]) -> "_XupyMaskedArray":
            """Construct an array by repeating A the number of times given by reps."""
            new_data = _xp.tile(self.data, reps)
            new_mask = _xp.tile(self._mask, reps)
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
        def apply_ufunc(self, ufunc: object, *args: _t.Any, **kwargs: dict[str, _t.Any]) -> "_XupyMaskedArray":
            """Apply a universal function to the array, respecting masks."""
            # Apply ufunc to data
            result_data = ufunc(self.data, *args, **kwargs)
            result_mask = _np.where(_np.isnan(result_data), True, self._mask)
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
            return int(_xp.sum(self._mask))
        
        def count_unmasked(self) -> int:
            """Return the number of unmasked elements."""
            return int(_xp.sum(~self._mask))
        
        def is_masked(self) -> bool:
            """Return True if the array has any masked values."""
            return bool(_xp.any(self._mask))
        
        def compressed(self) -> _xp.ndarray:
            """Return all the non-masked data as a 1-D array."""
            return self.data[~self._mask]
        
        def fill_value(self, value: _t.Scalar) -> None:
            """Set the fill value for masked elements."""
            self.data[self._mask] = value


        # --- Copy and Conversion Methods ---
        def copy(self, order: str = 'C') -> "_XupyMaskedArray":
            """Return a copy of the array."""
            return _XupyMaskedArray(self.data.copy(order=order), self._mask.copy(order=order))
        
        def astype(self, dtype: _t.DTypeLike, order: str = 'K') -> "_XupyMaskedArray":
            """
            Copy of the array, cast to a specified type.
            
            As natively cupy does not yet support casting, this method
            will simply return a copy of the array with the new dtype.
            """
            new_data = _xp.asarray(self.data, dtype=dtype, order=order)
            new_mask = self._mask.copy()
            return _XupyMaskedArray(new_data, new_mask, dtype=dtype)
        
        def tolist(self) -> list[_t.Scalar]:
            """Return the array as a nested list."""
            return self.data.tolist()
        
        def item(self, *args: int) -> _t.Scalar:
            """Copy an element of an array to a standard Python scalar and return it."""
            own = self.asmarray()
            result = own.item(*args)
            return result


        # --- Arithmetic Operators ---
        # TODO: Add to all the methods the ability to handle 
        # other as `numpy.ndarray`. Convert it to a cupy array and
        # then perform the operation.
        
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
                self._mask |= other._mask
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
                self._mask |= other._mask
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
                self._mask |= other._mask
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
                self._mask |= other._mask
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
                self._mask |= other._mask
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
                self._mask |= other._mask
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
                self._mask |= other._mask
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
            if isinstance(other, _XupyMaskedArray):
                result_data = self.data @ other.data
                result_mask = self._mask | other._mask
            elif isinstance(other, _np.ma.masked_array):
                other_data = _xp.asarray(other.data, dtype=self.dtype)
                other_mask = _xp.asarray(other.mask, dtype=bool)
                result_data = self.data @ other_data
                result_mask = self._mask | other_mask
            else:
                result_data = self.data @ other
                result_mask = self._mask
            return _XupyMaskedArray(result_data, mask=result_mask)

        def __rmatmul__(self, other: object) -> "_XupyMaskedArray":
            """
            Reflected matrix multiplication with mask propagation.
            """
            if isinstance(other, _XupyMaskedArray):
                other = other.asmarray()
            own = self.asmarray()
            result = other @ own
            return _XupyMaskedArray(result.data, mask=result.mask)

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
                self._mask = self._mask | other._mask
            elif isinstance(other, _np.ma.masked_array):
                other_data = _xp.asarray(other.data, dtype=self.dtype)
                other_mask = _xp.asarray(other.mask, dtype=bool)
                self.data = self.data @ other_data
                self._mask = self._mask | other_mask
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
            return _XupyMaskedArray(result, self._mask)

        def __pos__(self) -> "_XupyMaskedArray":
            """
            Element-wise unary plus with mask propagation.
            """
            result = +self.data
            return _XupyMaskedArray(result, self._mask)

        def __abs__(self) -> "_XupyMaskedArray":
            """
            Element-wise absolute value with mask propagation.
            """
            result = _xp.abs(self.data)
            return _XupyMaskedArray(result, self._mask)

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
            mask_item = self._mask[item]
            # If the result is a scalar, return a masked value
            if data_item.shape == ():
                if mask_item:
                    return _np.ma.masked
                return data_item.item()
            return _XupyMaskedArray(data_item, mask_item)

        def asmarray(self, **kwargs: dict[str,_t.Any]) -> _np.ma.MaskedArray[_t.Any,_t.Any]:
            """Return a NumPy masked array on CPU."""
            dtype = kwargs.get('dtype', self.dtype)
            if "dtype" in kwargs:
                kwargs.pop("dtype")
            return _np.ma.masked_array(_xp.asnumpy(self.data), mask=_xp.asnumpy(self._mask), dtype=dtype, **kwargs)

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
        return _XupyMaskedArray(data, mask, **kwargs)

    # --- GPU Memory Management Context Manager ---
    class MemoryContext:
        """Advanced GPU memory management context manager with automatic cleanup.

        Features:
        - Automatic memory cleanup on context exit
        - Memory pressure monitoring and automatic cleanup
        - Aggressive memory freeing with garbage collection
        - Memory usage tracking and reporting
        - Device context management with proper restoration
        - Memory pool management with multiple strategies
        - Emergency cleanup for out-of-memory situations
        """

        def __init__(self, device_id: Optional[int] = None, auto_cleanup: bool = True,
                     memory_threshold: float = 0.9, monitor_interval: float = 1.0):
            """
            Initialize the memory context manager.

            Parameters
            ----------
            device_id : int, optional
                GPU device ID to manage. If None, uses current device.
            auto_cleanup : bool, optional
                Whether to automatically cleanup memory on exit (default: True).
            memory_threshold : float, optional
                Memory usage threshold (0-1) for automatic cleanup (default: 0.9).
            monitor_interval : float, optional
                Interval in seconds for memory monitoring (default: 1.0).
            """
            self.device_id = device_id
            self.auto_cleanup = auto_cleanup
            self.memory_threshold = memory_threshold
            self.monitor_interval = monitor_interval

            self._device_ctx = None
            self._original_device = None
            self._gpu_objects = []  # Track GPU objects for cleanup
            self._memory_history = []
            self._start_time = None
            self._initial_memory = 0
            self._peak_memory = 0
            self._cleanup_count = 0

        def __enter__(self):
            """Enter the memory context."""
            self._start_time = time.time()

            if _GPU:
                # Store original device
                try:
                    self._original_device = _xp.cuda.runtime.getDevice()
                except Exception:
                    self._original_device = 0

                # Set target device if specified
                if self.device_id is not None:
                    try:
                        self._device_ctx = _xp.cuda.Device(self.device_id)
                        self._device_ctx.__enter__()
                    except Exception as e:
                        print(f"Warning: Could not set device {self.device_id}: {e}")

                # Record initial memory state
                initial_mem = self.get_memory_info()
                if "used" in initial_mem:
                    self._initial_memory = initial_mem["used"]
                    self._peak_memory = initial_mem["used"]

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Exit the memory context with cleanup."""
            try:
                if self.auto_cleanup:
                    self.aggressive_cleanup()

                # Cleanup tracked GPU objects
                self._cleanup_gpu_objects()

                # Restore original device
                if _GPU and self._device_ctx is not None:
                    try:
                        self._device_ctx.__exit__(exc_type, exc_val, exc_tb)
                    except Exception as e:
                        print(f"Warning: Error restoring device context: {e}")

                # Final memory report
                if self._start_time:
                    duration = time.time() - self._start_time
                    final_mem = self.get_memory_info()
                    if "used" in final_mem:
                        memory_delta = final_mem["used"] - self._initial_memory
                        print(f"[MemoryContext] Session completed in {duration:.2f}s")
                        print(f"[MemoryContext] Initial memory: {self._initial_memory / (1024**3):.2f} GB")
                        print(f"[MemoryContext] Peak memory: {self._peak_memory / (1024**3):.2f} GB")
                        print(f"[MemoryContext] Final memory: {final_mem['used'] / (1024**3):.2f} GB")
                        print(f"[MemoryContext] Memory delta: {memory_delta / (1024**3):.2f} GB")
                        if self._cleanup_count > 0:
                            print(f"[MemoryContext] Cleanup operations: {self._cleanup_count}")

            except Exception as e:
                print(f"Warning: Error during memory context cleanup: {e}")

        def track_object(self, obj):
            """Track a GPU object for cleanup."""
            if hasattr(obj, 'data') and hasattr(obj.data, 'device'):
                self._gpu_objects.append(obj)

        def _cleanup_gpu_objects(self):
            """Clean up tracked GPU objects."""
            for obj in self._gpu_objects:
                try:
                    # Clear references to GPU data
                    if hasattr(obj, 'data'):
                        obj.data = None
                    if hasattr(obj, 'mask'):
                        obj.mask = None
                except Exception:
                    pass
            self._gpu_objects.clear()

        def clear_cache(self):
            """Clear GPU memory pools (safely)."""
            if not _GPU:
                return

            try:
                # Ensure all kernels are finished
                _xp.cuda.runtime.deviceSynchronize()
            except Exception:
                pass

            try:
                # Free default memory pool
                mempool = _xp.get_default_memory_pool()
                mempool.free_all_blocks()
            except Exception as e:
                print(f"Warning: Could not free default memory pool: {e}")

            try:
                # Free pinned memory pool
                pinned_pool = _xp.get_default_pinned_memory_pool()
                pinned_pool.free_all_blocks()
            except Exception as e:
                print(f"Warning: Could not free pinned memory pool: {e}")

            try:
                # Synchronize again
                _xp.cuda.runtime.deviceSynchronize()
            except Exception:
                pass

        def aggressive_cleanup(self):
            """Perform aggressive memory cleanup."""
            if not _GPU:
                return

            print("[MemoryContext] Performing aggressive memory cleanup...")
            self._cleanup_count += 1

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CuPy caches
            try:
                _xp.clear_memo_cache()
            except Exception:
                pass

            # Clear memory pools multiple times with forced deallocation
            for _ in range(3):
                self.clear_cache()
                time.sleep(0.01)

            # Try to free unused memory more aggressively
            try:
                _xp.cuda.runtime.deviceSynchronize()
                # Force deallocation of unused memory
                _xp.cuda.runtime.free(0)
            except Exception:
                pass

            # Force another garbage collection
            gc.collect()

            # Additional aggressive measures
            try:
                # Try to force memory pool deallocation
                mempool = _xp.get_default_memory_pool()
                # Force garbage collection on the memory pool
                mempool.free_all_blocks()
                # Try to shrink the pool
                if hasattr(mempool, 'shrink'):
                    mempool.shrink()
            except Exception as e:
                print(f"Warning: Could not shrink memory pool: {e}")

            # Try to clear any cached arrays
            try:
                # Clear any cached computations
                _xp.clear_memo_cache()
                # Force synchronization
                _xp.cuda.runtime.deviceSynchronize()
            except Exception:
                pass

            # Try direct CUDA memory management
            try:
                # Force CUDA to free unused memory
                _xp.cuda.runtime.deviceSynchronize()
                # Try to trigger memory defragmentation
                free, total = _xp.cuda.runtime.memGetInfo()
                print(f"[MemoryContext] CUDA memory after cleanup: {free/(1024**3):.2f}/{total/(1024**3):.2f} GB")
            except Exception as e:
                print(f"Warning: Could not get CUDA memory info: {e}")

            # As a last resort, try memory pool reset
            try:
                self.force_memory_pool_reset()
            except Exception as e:
                print(f"Warning: Memory pool reset failed: {e}")

            # Final attempt: force memory deallocation
            try:
                self.force_memory_deallocation()
            except Exception as e:
                print(f"Warning: Forced memory deallocation failed: {e}")

        def emergency_cleanup(self):
            """Emergency cleanup for out-of-memory situations."""
            if not _GPU:
                return

            print("[MemoryContext] EMERGENCY MEMORY CLEANUP")
            self._cleanup_count += 1

            # Most aggressive cleanup possible
            import gc
            gc.collect()

            # Clear all caches multiple times
            for _ in range(5):
                try:
                    _xp.clear_memo_cache()
                except Exception:
                    pass
                self.clear_cache()
                time.sleep(0.05)

            # Try to reset the device (nuclear option)
            try:
                # Note: deviceReset may not be available in all CuPy versions
                # This is a more aggressive cleanup approach
                _xp.cuda.runtime.deviceSynchronize()
                print("[MemoryContext] Emergency synchronization performed")
            except Exception as e:
                print(f"Warning: Could not perform emergency cleanup: {e}")

            # Final garbage collection
            gc.collect()

            # Additional emergency measures
            try:
                # Try to force complete memory pool reset
                mempool = _xp.get_default_memory_pool()
                mempool.free_all_blocks()
                if hasattr(mempool, 'shrink'):
                    mempool.shrink()
                # Try to free pinned memory pool too
                pinned_pool = _xp.get_default_pinned_memory_pool()
                pinned_pool.free_all_blocks()
            except Exception as e:
                print(f"Warning: Could not reset memory pools: {e}")

            # Force final synchronization
            try:
                _xp.cuda.runtime.deviceSynchronize()
            except Exception:
                pass

        def get_memory_info(self) -> dict[str, _t.Any]:
            """Get comprehensive memory information."""
            if not _GPU:
                return {"error": "No GPU available"}

            try:
                # Get current device
                device_to_query = self.device_id if self.device_id is not None else _xp.cuda.runtime.getDevice()

                # Ensure we're on the correct device
                current = _xp.cuda.runtime.getDevice()
                if device_to_query != current:
                    _xp.cuda.runtime.setDevice(device_to_query)

                # Device-level memory info
                free, total = _xp.cuda.runtime.memGetInfo()
                used = int(total - free)

                # Memory pool info
                pool_used = 0
                pool_capacity = 0
                pool_free = 0

                try:
                    mempool = _xp.get_default_memory_pool()
                    pool_used = int(mempool.used_bytes())
                    pool_capacity = int(mempool.total_bytes())
                    pool_free = int(pool_capacity - pool_used)
                except Exception:
                    pass

                # Calculate percentages
                memory_percent = used / total if total > 0 else 0
                pool_percent = pool_used / pool_capacity if pool_capacity > 0 else 0

                # Restore original device
                if device_to_query != current:
                    _xp.cuda.runtime.setDevice(current)

                info = {
                    "device": int(device_to_query),
                    "total": int(total),
                    "free": int(free),
                    "used": int(used),
                    "memory_percent": memory_percent,
                    "pool_used": pool_used,
                    "pool_capacity": pool_capacity,
                    "pool_free": pool_free,
                    "pool_percent": pool_percent,
                }

                # Update peak memory tracking
                if used > self._peak_memory:
                    self._peak_memory = used

                # Store in history
                self._memory_history.append({
                    "timestamp": time.time(),
                    "used": used,
                    "free": free
                })

                # Keep only recent history
                if len(self._memory_history) > 100:
                    self._memory_history = self._memory_history[-100:]

                return info

            except Exception as e:
                return {"error": str(e)}

        def check_memory_pressure(self) -> bool:
            """Check if memory usage is above threshold."""
            mem_info = self.get_memory_info()
            if "memory_percent" in mem_info:
                pressure = mem_info["memory_percent"] > self.memory_threshold
                if pressure:
                    print(f"[MemoryContext] Memory pressure detected: {mem_info['memory_percent']*100:.1f}% > {self.memory_threshold*100:.1f}%")
                return pressure
            return False

        def auto_cleanup_if_needed(self):
            """Automatically cleanup if memory pressure is high."""
            if self.check_memory_pressure():
                print(f"[MemoryContext] Memory usage above {self.memory_threshold*100:.1f}%, triggering cleanup")
                self.aggressive_cleanup()

        def monitor_memory(self, duration: float = 10.0):
            """Monitor memory usage for a period of time."""
            import time

            print(f"[MemoryContext] Monitoring memory for {duration} seconds...")
            start_time = time.time()
            measurements = []

            while time.time() - start_time < duration:
                mem_info = self.get_memory_info()
                measurements.append(mem_info)
                time.sleep(self.monitor_interval)

            # Print summary
            if measurements:
                used_values = [m.get("used", 0) for m in measurements if "used" in m]
                if used_values:
                    min_used = builtins.min(used_values)
                    max_used = builtins.max(used_values)
                    avg_used = builtins.sum(used_values) / len(used_values)

                    print(f"[MemoryContext] Memory monitoring summary:")
                    print(f"  Min: {min_used / (1024**3):.2f} GB")
                    print(f"  Max: {max_used / (1024**3):.2f} GB")
                    print(f"  Avg: {avg_used / (1024**3):.2f} GB")

        def force_memory_deallocation(self):
            """Force memory deallocation by creating pressure on the memory pool."""
            if not _GPU:
                return

            print("[MemoryContext] Forcing memory deallocation...")
            try:
                # Get current memory info
                free_before, total = _xp.cuda.runtime.memGetInfo()
                print(f"[MemoryContext] Memory before forced deallocation: {free_before/(1024**3):.2f}/{total/(1024**3):.2f} GB")

                # Try to allocate a large chunk to force pool cleanup
                # This will fail if there's not enough memory, but that's okay
                try:
                    # Allocate 90% of available memory temporarily
                    alloc_size = int(free_before * 0.9)
                    if alloc_size > 100 * (1024**3):  # Only if we have more than 100MB to work with
                        temp_array = _xp.empty((alloc_size // 4,), dtype=_xp.float32)  # 4 bytes per float32
                        # Immediately delete it
                        del temp_array
                        # Force garbage collection
                        import gc
                        gc.collect()
                        # Clear memory pool
                        mempool = _xp.get_default_memory_pool()
                        mempool.free_all_blocks()
                except Exception:
                    # If allocation fails, just do normal cleanup
                    self.clear_cache()

                # Synchronize
                _xp.cuda.runtime.deviceSynchronize()

                # Check memory after
                free_after, _ = _xp.cuda.runtime.memGetInfo()
                freed = free_after - free_before
                print(f"[MemoryContext] Memory after forced deallocation: {free_after/(1024**3):.2f}/{total/(1024**3):.2f} GB")
                print(f"[MemoryContext] Memory freed: {freed/(1024**3):.2f} GB")

            except Exception as e:
                print(f"Warning: Could not force memory deallocation: {e}")

        def force_memory_pool_reset(self):
            """Force a complete memory pool reset by creating a new pool."""
            if not _GPU:
                return

            print("[MemoryContext] Performing memory pool reset...")
            try:
                # Get current pool
                old_pool = _xp.get_default_memory_pool()

                # Create a new memory pool
                new_pool = _xp.cuda.MemoryPool()

                # Set the new pool as default
                _xp.cuda.set_allocator(new_pool.malloc)

                # Force garbage collection to clean up old pool
                import gc
                gc.collect()

                # Free all blocks in old pool
                old_pool.free_all_blocks()

                # Synchronize to ensure operations are complete
                _xp.cuda.runtime.deviceSynchronize()

                print("[MemoryContext] Memory pool reset completed")

            except Exception as e:
                print(f"Warning: Could not reset memory pool: {e}")
                # Fallback to aggressive cleanup
                self.aggressive_cleanup()

        def __repr__(self) -> str:
            """String representation with memory info."""
            mem_info = self.get_memory_info()
            if "error" in mem_info:
                return f"MemoryContext(device={self.device_id}, error={mem_info['error']})"

            used_gb = mem_info.get("used", 0) / (1024**3)
            total_gb = mem_info.get("total", 0) / (1024**3)
            percent = mem_info.get("memory_percent", 0) * 100

            return f"MemoryContext(device={mem_info.get('device')}, memory={used_gb:.2f}/{total_gb:.2f} GB ({percent:.1f}%))"
    
else:
    
    float = double = _np.float64

    masked_array = _np.ma.masked_array
    MaskedArray = _np.ma.MaskedArray

    def asnumpy(array: _t.NDArray[_t.Any]) -> _np.ndarray:
        """
        Placeholder function for asnumpy when GPU is not available.
        """
        if isinstance(array, _np.ma.MaskedArray):
            return array.data
        return array