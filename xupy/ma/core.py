"""
XUPY MASKED ARRAY
=================

This module provides a comprehensive masked array wrapper for CuPy arrays with NumPy-like interface.
"""

from .. import _core as _xp
import numpy as _np
from .. import typings as _t

MaskType = _xp.bool_
nomask = MaskType(0)


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
        data: _t.ArrayLike,
        mask: _t.Optional[_t.ArrayLike] = None,
        dtype: _t.Optional[_t.DTypeLike] = None,
        fill_value: _t.Optional[_t.Scalar] = None,
        keep_mask: bool = True,
        hard_mask: bool = False,
        order: _t.Optional[str] = None,
    ) -> None:
        """The constructor"""

        self._dtype = dtype
        self.data = _xp.asarray(
            data, dtype=dtype if dtype else _xp.float32, order=order
        )

        if mask is None:
            if keep_mask is True:
                if hasattr(data, "mask"):
                    try:
                        self._mask = _xp.asarray(data.mask, dtype=bool)
                    except Exception as e:
                        print(f"Failed to retrieve mask from data: {e}")
                        self._mask = nomask
                else:
                    self._mask = nomask
        else:
            self._mask = _xp.asarray(mask, dtype=bool)
            self._has_no_mask = False

        self._is_hard_mask = hard_mask

        if fill_value is None:
            if hasattr(data, "fill_value"):
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
        return self._dtype if self._dtype is not None else self.data.dtype

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

        prefix = f"masked_array("

        dtype_needed = (
            not _np.core.arrayprint.dtype_is_implied(self.dtype)
            or _np.all(self._mask)
            or self.size == 0
        )

        # determine which keyword args need to be shown
        keys = ["data", "mask"]
        if dtype_needed:
            keys.append("dtype")

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
                indents[k] = " " * n
            prefix = ""  # absorbed into the first indent
        else:
            # each key on its own line, indented by two spaces
            indents = {k: " " * min_indent for k in keys}
            prefix = prefix + "\n"  # first key on the next line

        # format the field values
        reprs = {}

        # Determine precision based on dtype
        the_type = _np.dtype(self.dtype)
        if the_type.kind == "f":  # Floating-point
            precision = 6 if the_type.itemsize == 4 else 15  # float32 vs float64
        else:
            precision = None  # Default for integers, etc.

        reprs["data"] = _np.array2string(
            self._insert_masked_print(),
            separator=", ",
            prefix=indents["data"] + "data=",
            suffix=",",
            precision=precision,
        )
        reprs["mask"] = _np.array2string(
            _xp.asnumpy(self._mask),
            separator=", ",
            prefix=indents["mask"] + "mask=",
            suffix=",",
        )
        if dtype_needed:
            reprs["dtype"] = _np.core.arrayprint.dtype_short_repr(self.dtype)

        # join keys with values and indentations
        result = ",\n".join("{}{}={}".format(indents[k], k, reprs[k]) for k in keys)
        return prefix + result + ")"

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

    def flatten(self, order: str = "C") -> "_XupyMaskedArray":
        """Return a copy of the array collapsed into one dimension."""
        new_data = self.data.flatten(order=order)
        new_mask = self._mask.flatten(order=order)
        return _XupyMaskedArray(new_data, new_mask)

    def ravel(self, order: str = "C") -> "_XupyMaskedArray":
        """Return a flattened array."""
        return self.flatten(order=order)

    def squeeze(self, axis: _t.Optional[tuple[int, ...]] = None) -> "_XupyMaskedArray":
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

    def repeat(
        self, repeats: _t.Union[int, _t.ArrayLike], axis: _t.Optional[int] = None
    ) -> "_XupyMaskedArray":
        """Repeat elements of an array."""
        new_data = _xp.repeat(self.data, repeats, axis=axis)
        new_mask = _xp.repeat(self._mask, repeats, axis=axis)
        return _XupyMaskedArray(new_data, new_mask)

    def tile(self, reps: _t.Union[int, tuple[int, ...]]) -> "_XupyMaskedArray":
        """Construct an array by repeating A the number of times given by reps."""
        new_data = _xp.tile(self.data, reps)
        new_mask = _xp.tile(self._mask, reps)
        return _XupyMaskedArray(new_data, new_mask)

    # --- Statistical Methods (Memory-Optimized) ---
    def mean(self, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Compute the arithmetic mean along the specified axis."""
        own = self.asmarray()
        result = own.mean(**kwargs)
        return result

    def sum(self, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Sum of array elements over a given axis."""
        own = self.asmarray()
        result = own.sum(**kwargs)
        return result

    def std(self, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Compute the standard deviation along the specified axis."""
        own = self.asmarray()
        result = own.std(**kwargs)
        return result

    def var(self, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Compute the variance along the specified axis."""
        own = self.asmarray()
        result = own.var(**kwargs)
        return result

    def min(self, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Return the minimum along a given axis."""
        own = self.asmarray()
        result = own.min(**kwargs)
        return result

    def max(self, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Return the maximum along a given axis."""
        own = self.asmarray()
        result = own.max(**kwargs)
        return result

    # --- Universal Functions Support ---
    def apply_ufunc(
        self, ufunc: object, *args: _t.Any, **kwargs: dict[str, _t.Any]
    ) -> "_XupyMaskedArray":
        """Apply a universal function to the array, respecting masks."""
        # Apply ufunc to data
        result_data = ufunc(self.data, *args, **kwargs)
        # Use the appropriate array module for mask operations
        result_mask = _xp.where(_xp.isnan(result_data), True, self._mask)
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
    def any(self, **kwargs: dict[str, _t.Any]) -> bool:
        """Test whether any array element along a given axis evaluates to True."""
        own = self.asmarray()
        result = own.any(**kwargs)
        return result

    def all(self, **kwargs: dict[str, _t.Any]) -> bool:
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
    def copy(self, order: str = "C") -> "_XupyMaskedArray":
        """Return a copy of the array."""
        return _XupyMaskedArray(
            self.data.copy(order=order), self._mask.copy(order=order)
        )

    def astype(self, dtype: _t.DTypeLike, order: str = "K") -> "_XupyMaskedArray":
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
        result = own**other
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
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

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

    def asmarray(
        self, **kwargs: dict[str, _t.Any]
    ) -> _np.ma.MaskedArray[_t.Any, _t.Any]:
        """Return a NumPy masked array on CPU."""
        dtype = kwargs.pop("dtype", self.dtype)
        return _np.ma.masked_array(
            _xp.asnumpy(self.data),
            mask=_xp.asnumpy(self._mask),
            dtype=dtype,
            **kwargs,
        )

MaskedArray = masked_array = _XupyMaskedArray


def getmask(arr: _t.ArrayLike) -> MaskType:
    """
    Return the mask of a masked array, or nomask.

    Return the mask of `a` as an ndarray if `a` is a `MaskedArray` and the
    mask is not `nomask`, else return `nomask`. To guarantee a full array
    of booleans of the same shape as a, use `getmaskarray`.

    Parameters
    ----------
    a : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getdata : Return the data of a masked array as an ndarray.
    getmaskarray : Return the mask of a masked array, or full array of False.

    Examples
    --------
    >>> import xupy.ma as ma
    >>> x = ma.masked_array([[1,2],[3,4]], mask=[[False, True], [False, False]])
    >>> x
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getmask(x)
    array([[False,  True],
           [False, False]])
    """
    return getattr(arr, "_mask", nomask)


def getmaskarray(arr: _t.ArrayLike) -> MaskType:
    """
    Return the mask of a masked array, or full array of False.
    
    Return the mask of `arr` as an ndarray if `arr` is a `MaskedArray` and
    the mask is not `nomask`, else return a full boolean array of False of
    the same shape as `arr`.

    Parameters
    ----------
    arr : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getmask : Return the mask of a masked array, or nomask.
    getdata : Return the data of a masked array as an ndarray.

    Examples
    --------
    >>> import xupy.ma as ma
    >>> x = ma.masked_array([[1,2],[3,4]], mask=[[False, True], [False, False]])
    >>> x
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getmaskarray(x)
    array([[False,  True],
           [False, False]])

    Result when mask == ``nomask``

    >>> x = ma.masked_array([[1,2],[3,4]])
    >>> x
    masked_array(
      data=[[1, 2],
            [3, 4]],
      mask=False,
      fill_value=999999)
    >>> ma.getmaskarray(x)
    array([[False, False],
           [False, False]])
    """
    mask = getmask(arr)
    if mask is nomask:
        mask = _xp.zeros(_np.shape(arr), getattr(arr, 'dtype', MaskType))
    return mask


def is_mask(m: _t.ArrayLike) -> bool:
    """
    Check if a mask is a mask.
    """
    try:
        return m.dtype.type is MaskType
    except AttributeError:
        return False

__all__ = ["MaskedArray", "masked_array"]
