"""
XUPY MASKED ARRAY
=================

This module provides a comprehensive masked array wrapper for CuPy arrays with NumPy-like interface.
"""
# TODO
# FIXME
# fix `~mask` with `not mask` when working with booleans, as it is deprecated
# and will be removed in python 3.16

from .. import _core as _xp
import numpy as _np
from .. import typings as _t

MaskType = _xp.bool_

# Create a proper nomask singleton (similar to NumPy's approach)
class _NomaskSingleton:
    """Singleton representing no mask."""
    def __bool__(self):
        return False
    def __repr__(self):
        return "nomask"
    def __eq__(self, other):
        return other is nomask
    def __ne__(self, other):
        return other is not nomask

nomask = _NomaskSingleton()

# Create a masked singleton for representing masked values
class _MaskedSingleton:
    """Singleton representing a masked value."""
    def __bool__(self):
        return False
    def __repr__(self):
        return "--"
    def __str__(self):
        return "--"
    def __eq__(self, other):
        return other is masked
    def __ne__(self, other):
        return other is not masked

masked = _MaskedSingleton()


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
    hard_mask : bool, optional
        If True, indicates that the mask should be treated as an immutable
        "hard" mask. With a hard mask, masked values cannot be unmasked by
        assignment. Use harden_mask() and soften_mask() to control this
        behavior dynamically.
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
        # Don't force dtype conversion if not specified - let CuPy/NumPy decide
        # This preserves the input dtype and avoids precision loss
        if dtype is not None:
            self.data = _xp.asarray(data, dtype=dtype, order=order)
        else:
            self.data = _xp.asarray(data, order=order)

        if mask is None:
            if keep_mask is True:
                if hasattr(data, "mask"):
                    try:
                        data_mask = data.mask
                        if data_mask is nomask or data_mask is False:
                            self._mask = nomask
                        else:
                            self._mask = _xp.asarray(data_mask, dtype=bool)
                    except Exception as e:
                        print(f"Failed to retrieve mask from data: {e}")
                        self._mask = nomask
                else:
                    self._mask = nomask
        else:
            if mask is nomask or mask is False:
                self._mask = nomask
            else:
                self._mask = _xp.asarray(mask, dtype=bool)

        # Validate and broadcast mask shape to match data shape
        if self._mask is not nomask:
            try:
                # Try to broadcast mask to data shape
                if self._mask.shape != self.data.shape:
                    self._mask = _xp.broadcast_to(self._mask, self.data.shape).copy()
            except ValueError:
                raise ValueError(
                    f"Mask shape {self._mask.shape} cannot be broadcast to data shape {self.data.shape}"
                )

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

    @property
    def fill_value(self):
        """Return the fill value."""
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: float|int|complex):
        """Set the fill value."""
        self._fill_value = value

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
        Replace masked values with '--' for printing, handling large arrays efficiently.
        
        For large arrays (exceeding NumPy's print threshold), this method only
        transfers the edge items from GPU to CPU to avoid memory issues and
        potential segmentation faults.
        """
        # Get NumPy's print options to respect threshold settings
        threshold = _np.get_printoptions()['threshold']
        
        # For small arrays or when threshold is sys.maxsize, transfer everything
        import sys
        if self.size <= threshold or threshold == sys.maxsize:
            # Small array or threshold disabled - safe to transfer entirely
            data = _xp.asnumpy(self.data)
            mask = _xp.asnumpy(self._mask)
            display = data.astype(object)
            display[mask] = "--"
            return display
        
        # Large array - only transfer edges that will be displayed
        # Get edge items setting (how many items to show at start/end of each axis)
        edgeitems = _np.get_printoptions()['edgeitems']
        
        # Strategy: Convert to numpy.ma.MaskedArray and let NumPy handle the printing
        # But only transfer the edge items to minimize GPU->CPU transfer
        
        # For 1D arrays
        if self.ndim == 1:
            if self.size > 2 * edgeitems:
                # Transfer only edges
                data_head = _xp.asnumpy(self.data[:edgeitems])
                data_tail = _xp.asnumpy(self.data[-edgeitems:])
                
                # Handle mask
                if self._mask is nomask:
                    mask_head = _np.zeros(data_head.shape, dtype=bool)
                    mask_tail = _np.zeros(data_tail.shape, dtype=bool)
                else:
                    mask_head = _xp.asnumpy(self._mask[:edgeitems])
                    mask_tail = _xp.asnumpy(self._mask[-edgeitems:])
                
                # Create compact arrays for display
                data_compact = _np.concatenate([data_head, data_tail])
                mask_compact = _np.concatenate([mask_head, mask_tail])
                
                # Convert to object array with masked values
                display = data_compact.astype(object)
                display[mask_compact] = "--"
                
                # Insert ellipsis marker (NumPy will handle it in array2string)
                display = _np.concatenate([
                    display[:edgeitems],
                    _np.array(['...'], dtype=object),
                    display[edgeitems:]
                ])
                return display
            else:
                # Transfer all
                data = _xp.asnumpy(self.data)
                
                # Handle mask
                if self._mask is nomask:
                    mask = _np.zeros(data.shape, dtype=bool)
                else:
                    mask = _xp.asnumpy(self._mask)
                
                display = data.astype(object)
                display[mask] = "--"
                return display
        
        # For multi-dimensional arrays (2D is most common)
        elif self.ndim == 2:
            nrows, ncols = self.shape
            
            # Determine which rows and columns to extract
            if nrows > 2 * edgeitems:
                row_indices = list(range(edgeitems)) + list(range(nrows - edgeitems, nrows))
                need_row_ellipsis = True
            else:
                row_indices = list(range(nrows))
                need_row_ellipsis = False
            
            if ncols > 2 * edgeitems:
                col_head = slice(0, edgeitems)
                col_tail = slice(ncols - edgeitems, ncols)
                need_col_ellipsis = True
            else:
                col_head = slice(None)
                col_tail = None
                need_col_ellipsis = False
            
            # Transfer only the required corners
            if need_col_ellipsis:
                # Extract top-left, top-right, bottom-left, bottom-right corners
                data_parts = []
                
                for idx in row_indices:
                    row_data_head = _xp.asnumpy(self.data[idx, col_head])
                    row_data_tail = _xp.asnumpy(self.data[idx, col_tail])
                    
                    # Handle mask
                    if self._mask is nomask:
                        row_mask_head = _np.zeros(row_data_head.shape, dtype=bool)
                        row_mask_tail = _np.zeros(row_data_tail.shape, dtype=bool)
                    else:
                        row_mask_head = _xp.asnumpy(self._mask[idx, col_head])
                        row_mask_tail = _xp.asnumpy(self._mask[idx, col_tail])
                    
                    # Combine head and tail for this row
                    row_data = _np.concatenate([row_data_head, row_data_tail])
                    row_mask = _np.concatenate([row_mask_head, row_mask_tail])
                    
                    # Convert to object and apply mask
                    row_display = row_data.astype(object)
                    row_display[row_mask] = "--"
                    
                    # Insert column ellipsis
                    row_display = _np.concatenate([
                        row_display[:edgeitems],
                        _np.array(['...'], dtype=object),
                        row_display[edgeitems:]
                    ])
                    data_parts.append(row_display)
                
                # Stack rows
                if need_row_ellipsis:
                    # Insert row ellipsis
                    ellipsis_row = _np.full((1, len(data_parts[0])), '...', dtype=object)
                    display = _np.vstack([
                        _np.vstack(data_parts[:edgeitems]),
                        ellipsis_row,
                        _np.vstack(data_parts[edgeitems:])
                    ])
                else:
                    display = _np.vstack(data_parts)
            else:
                # Only need row ellipsis (or no ellipsis)
                data_parts = []
                for idx in row_indices:
                    row_data = _xp.asnumpy(self.data[idx, :])
                    
                    # Handle mask
                    if self._mask is nomask:
                        row_mask = _np.zeros(row_data.shape, dtype=bool)
                    else:
                        row_mask = _xp.asnumpy(self._mask[idx, :])
                    
                    row_display = row_data.astype(object)
                    row_display[row_mask] = "--"
                    data_parts.append(row_display)
                
                if need_row_ellipsis:
                    ellipsis_row = _np.full((1, data_parts[0].shape[0]), '...', dtype=object)
                    display = _np.vstack([
                        _np.vstack(data_parts[:edgeitems]),
                        ellipsis_row,
                        _np.vstack(data_parts[edgeitems:])
                    ])
                else:
                    display = _np.vstack(data_parts)
            
            return display
        
        else:
            # For higher dimensional arrays (3D+), we use a simplified approach
            # Transfer more data but still limit based on edgeitems
            # This is a safety fallback - could be optimized further
            
            # Build slice objects for each dimension
            slices = []
            for axis_size in self.shape:
                if axis_size > 2 * edgeitems:
                    # Create indices for edges
                    indices = list(range(edgeitems)) + list(range(axis_size - edgeitems, axis_size))
                    slices.append(indices)
                else:
                    slices.append(slice(None))
            
            # Use numpy's advanced indexing to extract only edges
            # For simplicity, just transfer edges along first 2 dimensions
            if len(slices) >= 2:
                # Extract a subset
                idx0 = slices[0] if isinstance(slices[0], slice) else slices[0]
                idx1 = slices[1] if isinstance(slices[1], slice) else slices[1]
                remaining_slices = tuple(slices[2:])
                
                # This gets complex, so for now transfer more data
                # Could be optimized with more sophisticated indexing
                data = _xp.asnumpy(self.data)
                mask = _xp.asnumpy(self._mask)
                display = data.astype(object)
                display[mask] = "--"
                return display
            else:
                # Fallback
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
    def _extract_scalar_if_0d(self, result: _t.Any) -> _t.Any:
        """
        Extract Python scalar from 0-dimensional array for NumPy compatibility.
        
        If result is a 0-dimensional CuPy/NumPy array, extract the scalar value.
        Otherwise, return as-is. This ensures compatibility with NumPy's behavior
        where reductions return scalars when appropriate.
        """
        if isinstance(result, (_xp.ndarray, _np.ndarray)):
            if result.ndim == 0:
                return result.item()
        return result

    def _combine_masks(self, mask1: _t.Union[_xp.ndarray, _NomaskSingleton], mask2: _t.Union[_xp.ndarray, _NomaskSingleton]) -> _t.Union[_xp.ndarray, _NomaskSingleton]:
        """
        Combine two masks using OR operation, handling nomask properly.
        
        Parameters
        ----------
        mask1 : array-like or nomask
            First mask.
        mask2 : array-like or nomask
            Second mask.
            
        Returns
        -------
        mask : array-like or nomask
            Combined mask (OR of both masks).
        """
        if mask1 is nomask and mask2 is nomask:
            return nomask
        elif mask1 is nomask:
            return mask2
        elif mask2 is nomask:
            return mask1
        else:
            return mask1 | mask2

    def _detect_nan_inf(self, data: _xp.ndarray, existing_mask: _t.Union[_xp.ndarray, _NomaskSingleton]) -> _t.Union[_xp.ndarray, _NomaskSingleton]:
        """
        Detect NaN and infinity values in floating-point arrays and update mask.
        
        This helper function ensures that NaN and infinity values resulting from
        operations (e.g., division by zero, sqrt of negative numbers) are properly
        masked, matching numpy.ma behavior.
        
        Parameters
        ----------
        data : array-like
            The data array to check for NaNs and infinities.
        existing_mask : array-like or nomask
            The existing mask to combine with NaN/infinity detection.
            
        Returns
        -------
        mask : array-like or nomask
            Updated mask that includes NaN and infinity positions. Returns nomask
            if no NaNs/infs are found and existing_mask is nomask.
        """
        # Only check for NaNs/infinities in floating-point types
        if _xp.issubdtype(data.dtype, _xp.floating):
            nan_mask = _xp.isnan(data) | _xp.isinf(data)
            # Check if nan_mask has any True values
            if _xp.any(nan_mask):
                if existing_mask is nomask:
                    return nan_mask
                else:
                    return existing_mask | nan_mask
            else:
                # No NaNs/infs found - return existing_mask as-is
                return existing_mask
        else:
            # For non-floating types, return existing mask
            return existing_mask

    def mean(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Compute the arithmetic mean along the specified axis.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the mean. If None, compute the mean
            of the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        scalar or array
            The mean value(s). Returns a scalar if axis is None, otherwise
            an array with the mean along the specified axis.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
        >>> mask = xp.array([False, True, False, False])
        >>> arr = masked_array(data, mask)
        >>> arr.mean()
        2.6666666666666665
        """
        # Handle all-masked case
        if self._mask is not nomask and _xp.all(self._mask):
            if axis is None:
                return masked
            else:
                # Return masked array with same shape as result
                result_shape = list(self.shape)
                if axis is not None:
                    result_shape.pop(axis)
                return _XupyMaskedArray(
                    _xp.full(result_shape, _xp.nan, dtype=self.dtype),
                    _xp.ones(result_shape, dtype=bool)
                )
        
        # Use GPU
        if axis is None:
            # Global mean
            if self._mask is nomask:
                result = _xp.mean(self.data)
            else:
                unmasked_data = self.data[~self._mask]
                if unmasked_data.size == 0:
                    return masked
                result = _xp.mean(unmasked_data)
            return self._extract_scalar_if_0d(result)
        else:
            # Mean along specific axis - return a masked array
            keepdims = kwargs.get('keepdims', False)
            if self._mask is nomask:
                # No mask: wrap the result in a masked array with False mask
                result_data = _xp.mean(self.data, axis=axis, **kwargs)
                result_mask = _xp.zeros(result_data.shape, dtype=bool)
                return _XupyMaskedArray(result_data, result_mask)
            else:
                # Compute mean using sum/count of unmasked values to avoid NaNs
                masked_zero = _xp.where(self._mask, _xp.zeros_like(self.data), self.data)
                sum_unmasked = _xp.sum(masked_zero, axis=axis, **kwargs)
                count_unmasked = _xp.sum(~self._mask, axis=axis, keepdims=keepdims)
                # Mask where there are no unmasked samples
                result_mask = (count_unmasked == 0)
                # Safe division: use count 1 where masked to avoid inf; values will be overwritten
                safe_count = _xp.where(result_mask, _xp.ones_like(count_unmasked), count_unmasked)
                result_data = sum_unmasked / safe_count
                # For floats, fill masked positions with NaN (cosmetic; value is masked anyway)
                try:
                    is_float = _np.dtype(self.dtype).kind == "f"
                except Exception:
                    is_float = False
                if is_float:
                    result_data = _xp.where(result_mask, _xp.nan, result_data)
                return _XupyMaskedArray(result_data, result_mask.astype(bool))

    def sum(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Sum of array elements over a given axis.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the sum. If None, compute the sum
            of the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        scalar or array
            The sum value(s). Returns a scalar if axis is None, otherwise
            an array with the sum along the specified axis.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
        >>> mask = xp.array([False, True, False, False])
        >>> arr = masked_array(data, mask)
        >>> arr.sum()
        8.0
        """
        # Handle all-masked case
        if self._mask is not nomask and _xp.all(self._mask):
            if axis is None:
                return masked
            else:
                # Return masked array with same shape as result
                result_shape = list(self.shape)
                if axis is not None:
                    result_shape.pop(axis)
                return _XupyMaskedArray(
                    _xp.zeros(result_shape, dtype=self.dtype),
                    _xp.ones(result_shape, dtype=bool)
                )
        
        # Use GPU operations for better performance
        if axis is None:
            # Global sum
            if self._mask is nomask:
                result = _xp.sum(self.data)
            else:
                unmasked_data = self.data[~self._mask]
                if unmasked_data.size == 0:
                    return masked
                result = _xp.sum(unmasked_data)
            return self._extract_scalar_if_0d(result)
        else:
            # Sum along specific axis - return a masked array
            keepdims = kwargs.get('keepdims', False)
            if self._mask is nomask:
                result_data = _xp.sum(self.data, axis=axis, **kwargs)
                result_mask = _xp.zeros(result_data.shape, dtype=bool)
                return _XupyMaskedArray(result_data, result_mask)
            else:
                # Zero-out masked values and sum
                masked_zero = _xp.where(self._mask, _xp.zeros_like(self.data), self.data)
                result_data = _xp.sum(masked_zero, axis=axis, **kwargs)
                # Positions where all entries were masked remain masked
                result_mask = _xp.all(self._mask, axis=axis, keepdims=keepdims)
                return _XupyMaskedArray(result_data, result_mask.astype(bool))

    def std(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Compute the standard deviation along the specified axis.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the standard deviation. If None, compute
            the standard deviation of the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        scalar or array
            The standard deviation value(s). Returns a scalar if axis is None,
            otherwise an array with the standard deviation along the specified axis.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
        >>> mask = xp.array([False, True, False, False])
        >>> arr = masked_array(data, mask)
        >>> arr.std()
        1.4142135623730951
        """
        # Use GPU operations for better performance
        if axis is None:
            # Global std
            unmasked_data = self.data[~self._mask]
            if unmasked_data.size == 0:
                return _xp.nan
            result = _xp.std(unmasked_data)
            return self._extract_scalar_if_0d(result)
        else:
            # Std along specific axis - return a masked array
            keepdims = kwargs.get('keepdims', False)
            if self._mask is nomask:
                result_data = _xp.std(self.data, axis=axis, **kwargs)
                result_mask = _xp.zeros(result_data.shape, dtype=bool)
                return _XupyMaskedArray(result_data, result_mask)
            else:
                data_copy = self.data.copy()
                data_copy[self._mask] = _xp.nan
                result_data = _xp.nanstd(data_copy, axis=axis, **kwargs)
                result_mask = _xp.all(self._mask, axis=axis, keepdims=keepdims)
                # For floats, optionally set masked results to NaN
                try:
                    is_float = _np.dtype(self.dtype).kind == "f"
                except Exception:
                    is_float = False
                if is_float:
                    result_data = _xp.where(result_mask, _xp.nan, result_data)
                return _XupyMaskedArray(result_data, result_mask.astype(bool))

    def var(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Compute the variance along the specified axis.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to compute the variance. If None, compute
            the variance of the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        scalar or array
            The variance value(s). Returns a scalar if axis is None,
            otherwise an array with the variance along the specified axis.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
        >>> mask = xp.array([False, True, False, False])
        >>> arr = masked_array(data, mask)
        >>> arr.var()
        2.0
        """
        # Use GPU operations for better performance
        if axis is None:
            # Global var
            unmasked_data = self.data[~self._mask]
            if unmasked_data.size == 0:
                return _xp.nan
            result = _xp.var(unmasked_data)
            return self._extract_scalar_if_0d(result)
        else:
            # Var along specific axis - return a masked array
            keepdims = kwargs.get('keepdims', False)
            if self._mask is nomask:
                result_data = _xp.var(self.data, axis=axis, **kwargs)
                result_mask = _xp.zeros(result_data.shape, dtype=bool)
                return _XupyMaskedArray(result_data, result_mask)
            else:
                data_copy = self.data.copy()
                data_copy[self._mask] = _xp.nan
                result_data = _xp.nanvar(data_copy, axis=axis, **kwargs)
                result_mask = _xp.all(self._mask, axis=axis, keepdims=keepdims)
                try:
                    is_float = _np.dtype(self.dtype).kind == "f"
                except Exception:
                    is_float = False
                if is_float:
                    result_data = _xp.where(result_mask, _xp.nan, result_data)
                return _XupyMaskedArray(result_data, result_mask.astype(bool))

    def min(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Return the minimum along a given axis.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to find the minimum. If None, find the minimum
            of the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        scalar or array
            The minimum value(s). Returns a scalar if axis is None,
            otherwise an array with the minimum along the specified axis.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
        >>> mask = xp.array([False, True, False, False])
        >>> arr = masked_array(data, mask)
        >>> arr.min()
        1.0
        """
        # Use GPU operations for better performance
        if axis is None:
            # Global min
            if self._mask is not nomask:
                unmasked_data = self.data[~self._mask]
                if unmasked_data.size == 0:
                    # All values are masked - return masked singleton (numpy.ma behavior)
                    return masked
            else:
                unmasked_data = self.data
            result = _xp.min(unmasked_data)
            return self._extract_scalar_if_0d(result)
        else:
            # Min along specific axis - return a masked array
            keepdims = kwargs.get('keepdims', False)
            if self._mask is nomask:
                result_data = _xp.min(self.data, axis=axis, **kwargs)
                result_mask = _xp.zeros(result_data.shape, dtype=bool)
                return _XupyMaskedArray(result_data, result_mask)
            else:
                data_copy = self.data.copy()
                data_copy[self._mask] = _xp.nan
                result_data = _xp.nanmin(data_copy, axis=axis, **kwargs)
                result_mask = _xp.all(self._mask, axis=axis, keepdims=keepdims)
                return _XupyMaskedArray(result_data, result_mask.astype(bool))

    def max(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> _t.Scalar:
        """Return the maximum along a given axis.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to find the maximum. If None, find the maximum
            of the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        scalar or array
            The maximum value(s). Returns a scalar if axis is None,
            otherwise an array with the maximum along the specified axis.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
        >>> mask = xp.array([False, True, False, False])
        >>> arr = masked_array(data, mask)
        >>> arr.max()
        4.0
        """
        # Use GPU operations for better performance
        if axis is None:
            # Global max
            if self._mask is not nomask:
                unmasked_data = self.data[~self._mask]
                if unmasked_data.size == 0:
                    # All values are masked - return masked singleton (numpy.ma behavior)
                    return masked
            else:
                unmasked_data = self.data
            result = _xp.max(unmasked_data)
            return self._extract_scalar_if_0d(result)
        else:
            # Max along specific axis - return a masked array
            keepdims = kwargs.get('keepdims', False)
            if self._mask is nomask:
                result_data = _xp.max(self.data, axis=axis, **kwargs)
                result_mask = _xp.zeros(result_data.shape, dtype=bool)
                return _XupyMaskedArray(result_data, result_mask)
            else:
                data_copy = self.data.copy()
                data_copy[self._mask] = _xp.nan
                result_data = _xp.nanmax(data_copy, axis=axis, **kwargs)
                result_mask = _xp.all(self._mask, axis=axis, keepdims=keepdims)
                return _XupyMaskedArray(result_data, result_mask.astype(bool))

    # --- Universal Functions Support ---
    def apply_ufunc(
        self, ufunc: object, *args: _t.Any, **kwargs: dict[str, _t.Any]
    ) -> "_XupyMaskedArray":
        """Apply a universal function to the array, respecting masks.
        
        Parameters
        ----------
        ufunc : callable
            The universal function to apply (e.g., xp.sqrt, xp.exp, etc.).
        *args : any
            Additional arguments to pass to the ufunc.
        **kwargs : dict
            Additional keyword arguments to pass to the ufunc.
            
        Returns
        -------
        _XupyMaskedArray
            A new masked array with the ufunc applied to the data and
            mask updated to reflect any NaN results.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 4.0, 9.0, 16.0])
        >>> mask = xp.array([False, True, False, False])
        >>> arr = masked_array(data, mask)
        >>> result = arr.apply_ufunc(xp.sqrt)
        >>> result
        masked_array(data=[1.0 '--' 3.0 4.0], mask=[False True False False])
        """
        # Apply ufunc to data
        result_data = ufunc(self.data, *args, **kwargs)
        # Use the appropriate array module for mask operations
        result_mask = _xp.where(_xp.isnan(result_data), True, self._mask)
        # Preserve mask
        return _XupyMaskedArray(result_data, result_mask)

    def sqrt(self) -> "_XupyMaskedArray":
        """Return the positive square-root of an array, element-wise.
        
        Returns
        -------
        _XupyMaskedArray
            A new masked array with the square root of each element.
            Masked elements remain masked.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 4.0, 9.0, 16.0])
        >>> mask = xp.array([False, True, False, False])
        >>> arr = masked_array(data, mask)
        >>> arr.sqrt()
        masked_array(data=[1.0 '--' 3.0 4.0], mask=[False True False False])
        """
        return self.apply_ufunc(_xp.sqrt)

    def exp(self) -> "_XupyMaskedArray":
        """Calculate the exponential of all elements in the input array.
        
        Returns
        -------
        _XupyMaskedArray
            A new masked array with e raised to the power of each element.
            Masked elements remain masked.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([0.0, 1.0, 2.0])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr.exp()
        masked_array(data=[1.0 '--' 7.38905609893065], mask=[False True False])
        """
        return self.apply_ufunc(_xp.exp)

    def log(self) -> "_XupyMaskedArray":
        """Natural logarithm, element-wise.
        
        Returns
        -------
        _XupyMaskedArray
            A new masked array with the natural logarithm of each element.
            Masked elements remain masked.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.718281828459045, 7.38905609893065])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr.log()
        masked_array(data=[0.0 '--' 2.0], mask=[False True False])
        """
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
    def any(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> bool:
        """Test whether any array element along a given axis evaluates to True.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to test. If None, test the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        bool or array
            True if any element along the axis is True. Returns a scalar if
            axis is None, otherwise an array with boolean results.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([True, False, True])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr.any()
        True
        """
        # Use GPU operations for better performance
        if axis is None:
            # Global any
            unmasked_data = self.data[~self._mask]
            if unmasked_data.size == 0:
                return False
            return _xp.any(unmasked_data)
        else:
            # Any along specific axis - use CuPy operations
            # Create a copy of data and set masked values to False
            data_copy = self.data.copy()
            data_copy[self._mask] = False
            return _xp.any(data_copy, axis=axis, **kwargs)

    def all(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> bool:
        """Returns True if all array elements along a given axis evaluate to True.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to test. If None, test the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        bool or array
            True if all elements along the axis are True. Returns a scalar if
            axis is None, otherwise an array with boolean results.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([True, True, True])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr.all()
        True
        """
        # Use GPU operations for better performance
        if axis is None:
            # Global all
            unmasked_data = self.data[~self._mask]
            if unmasked_data.size == 0:
                return True  # Empty array is considered all True
            return _xp.all(unmasked_data)
        else:
            # All along specific axis - use CuPy operations
            # Create a copy of data and set masked values to True
            data_copy = self.data.copy()
            data_copy[self._mask] = True
            return _xp.all(data_copy, axis=axis, **kwargs)

    def count(self, axis: _t.Optional[int] = None, **kwargs: dict[str, _t.Any]) -> int:
        """Return the number of unmasked elements along the given axis.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to count. If None, count the flattened array.
        **kwargs : dict
            Additional keyword arguments passed to the underlying CuPy function.
            
        Returns
        -------
        int or array
            The number of unmasked elements. Returns a scalar if axis is None,
            otherwise an array with counts along the specified axis.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
        >>> mask = xp.array([False, True, False, True])
        >>> arr = masked_array(data, mask)
        >>> arr.count()
        2
        """
        # Use GPU operations for better performance
        if axis is None:
            # Global count
            return int(_xp.sum(~self._mask))
        else:
            # Count along specific axis - use CuPy operations
            return int(_xp.sum(~self._mask, axis=axis, **kwargs))

    def is_masked(self) -> bool:
        """Return True if the array has any masked values.
        
        Returns
        -------
        bool
            True if any elements are masked, False otherwise.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr.is_masked()
        True
        """
        if self._mask is nomask:
            return False
        return bool(_xp.any(self._mask))

    def compressed(self) -> _xp.ndarray:
        """Return all the non-masked data as a 1-D array.
        
        Returns
        -------
        cupy.ndarray
            A 1-D CuPy array containing only the unmasked elements.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
        >>> mask = xp.array([False, True, False, True])
        >>> arr = masked_array(data, mask)
        >>> arr.compressed()
        array([1., 3.])
        """
        if self._mask is nomask:
            return self.data.flatten()
        return self.data[~self._mask]

    def count_masked(self, axis: _t.Optional[int] = None) -> int:
        """Count the number of masked elements.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to count. If None, count the flattened array.
            
        Returns
        -------
        int or array
            Number of masked elements.
        """
        if self._mask is nomask:
            return 0
        if axis is None:
            return int(_xp.sum(self._mask))
        else:
            return int(_xp.sum(self._mask, axis=axis))

    def count_unmasked(self, axis: _t.Optional[int] = None) -> int:
        """Count the number of unmasked elements.
        
        Parameters
        ----------
        axis : int, optional
            Axis along which to count. If None, count the flattened array.
            
        Returns
        -------
        int or array
            Number of unmasked elements.
        """
        if self._mask is nomask:
            return self.size
        if axis is None:
            return int(_xp.sum(~self._mask))
        else:
            return int(_xp.sum(~self._mask, axis=axis))

    def fill(self, value: _t.Scalar) -> None:
        """Set the fill value for masked elements.
        
        Parameters
        ----------
        value : scalar
            The value to fill masked elements with.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr.fill(999.0)
        >>> arr
        masked_array(data=[1.0 999.0 3.0], mask=[False True False])
        """
        if self._mask is not nomask:
            self.data[self._mask] = value

    def filled(self, fill_value: _t.Optional[_t.Scalar] = None) -> _xp.ndarray:
        """Return a copy of the array with masked values filled.
        
        Parameters
        ----------
        fill_value : scalar, optional
            Value to use for filling. If None, uses the stored fill_value.
            
        Returns
        -------
        cupy.ndarray
            Array with masked values filled.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr.filled()
        array([1.0, 999999.0, 3.0])
        """
        if fill_value is None:
            fill_value = self._fill_value
        
        result = self.data.copy()
        if self._mask is not nomask:
            result[self._mask] = fill_value
        return result

    # --- Copy and Conversion Methods ---
    def copy(self, order: str = "C") -> "_XupyMaskedArray":
        """Return a copy of the array.
        
        Parameters
        ----------
        order : str, optional
            Memory layout order for the copy. Default is 'C'.
            
        Returns
        -------
        _XupyMaskedArray
            A new masked array with copied data and mask.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr_copy = arr.copy()
        >>> arr_copy is arr
        False
        """
        return _XupyMaskedArray(
            self.data.copy(order=order), self._mask.copy(order=order)
        )

    def astype(self, dtype: _t.DTypeLike, order: str = "K") -> "_XupyMaskedArray":
        """Copy of the array, cast to a specified type.

        As natively cupy does not yet support casting, this method
        will simply return a copy of the array with the new dtype.
        
        Parameters
        ----------
        dtype : dtype
            Target data type for the conversion.
        order : str, optional
            Memory layout order. Default is 'K'.
            
        Returns
        -------
        _XupyMaskedArray
            A new masked array with the specified dtype.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> arr_int = arr.astype(xp.int32)
        >>> arr_int.dtype
        dtype('int32')
        """
        new_data = _xp.asarray(self.data, dtype=dtype, order=order)
        new_mask = self._mask.copy()
        return _XupyMaskedArray(new_data, new_mask, dtype=dtype)

    def tolist(self) -> list[_t.Scalar]:
        """Return the array as a nested list.
        
        Returns
        -------
        list
            A nested list representation of the array data.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> mask = xp.array([[False, True], [False, False]])
        >>> arr = masked_array(data, mask)
        >>> arr.tolist()
        [[1.0, 2.0], [3.0, 4.0]]
        """
        return self.data.tolist()

    def item(self, *args: int) -> _t.Scalar:
        """Copy an element of an array to a standard Python scalar and return it.
        
        Parameters
        ----------
        *args : int
            Index or indices to access the element.
            
        Returns
        -------
        scalar
            The element as a Python scalar.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> mask = xp.array([[False, True], [False, False]])
        >>> arr = masked_array(data, mask)
        >>> arr.item(0, 0)
        1.0
        """
        own = self.asmarray()
        result = own.item(*args)
        return result

    # --- Arithmetic Operators ---
    # All arithmetic operators now handle numpy.ndarray inputs by converting
    # them to CuPy arrays and performing operations on GPU for optimal performance.

    def __radd__(self, other: object) -> "_XupyMaskedArray":
        """
        Reflected element-wise addition with mask propagation.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = other.data + self.data
            result_mask = self._combine_masks(other._mask, self._mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = other_data + self.data
            result_mask = self._combine_masks(other_mask, self._mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = other + self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)

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
            self._mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            self.data += other_data
            self._mask = self._combine_masks(self._mask, other_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            self.data += other_data
        else:
            
            self.data += other
        return self

    def __rsub__(self, other: object) -> "_XupyMaskedArray":
        """
        Reflected element-wise subtraction with mask propagation.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = other.data - self.data
            result_mask = self._combine_masks(other._mask, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = other_data - self.data
            result_mask = self._combine_masks(other_mask, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = other_data - self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = other - self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)

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
            self._mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            self.data -= other_data
            self._mask = self._combine_masks(self._mask, other_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            self.data -= other_data
        else:
            
            self.data -= other
        return self

    def __rmul__(self, other: object) -> "_XupyMaskedArray":
        """
        Reflected element-wise multiplication with mask propagation.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = other.data * self.data
            result_mask = self._combine_masks(other._mask, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = other_data * self.data
            result_mask = self._combine_masks(other_mask, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = other_data * self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = other * self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)

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
            self._mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            self.data *= other_data
            self._mask = self._combine_masks(self._mask, other_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            self.data *= other_data
        else:
            
            self.data *= other
        return self

    def __rtruediv__(self, other: object) -> "_XupyMaskedArray":
        """
        Reflected element-wise true division with mask propagation.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = other.data / self.data
            result_mask = self._combine_masks(other._mask, self._mask)
            # Also mask where divisor (self.data) is zero
            if _xp.issubdtype(self.data.dtype, _xp.floating):
                if self._mask is nomask:
                    zero_divisor = (self.data == 0)
                else:
                    zero_divisor = (self.data == 0) & (~self._mask)
                # Only update mask if there are actual zeros
                if _xp.any(zero_divisor):
                    if result_mask is nomask:
                        result_mask = zero_divisor
                    else:
                        result_mask = result_mask | zero_divisor
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = other_data / self.data
            result_mask = self._combine_masks(other_mask, self._mask)
            # Mask where divisor (self.data) is zero
            if _xp.issubdtype(self.data.dtype, _xp.floating):
                if self._mask is nomask:
                    zero_divisor = (self.data == 0)
                else:
                    zero_divisor = (self.data == 0) & (~self._mask)
                # Only update mask if there are actual zeros
                if _xp.any(zero_divisor):
                    if result_mask is nomask:
                        result_mask = zero_divisor
                    else:
                        result_mask = result_mask | zero_divisor
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = other / self.data
            result_mask = self._mask
            # Mask where divisor (self.data) is zero (unless already masked)
            if not (isinstance(other, (int, float, complex)) and other == 0):
                # Check for division by zero (only for non-zero numerators)
                if _xp.issubdtype(self.data.dtype, _xp.floating) or isinstance(other, (float, complex)):
                    zero_divisor = (self.data == 0)
                    if self._mask is not nomask:
                        # Only mask where divisor is zero AND not already masked
                        zero_divisor = zero_divisor & (~self._mask)
                    if result_mask is nomask:
                        result_mask = zero_divisor
                    else:
                        result_mask = result_mask | zero_divisor
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)

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
            self._mask = self._combine_masks(self._mask, other._mask)
            # Also mask where divisor is zero (division by zero)
            if _xp.issubdtype(other.data.dtype, _xp.floating):
                if other._mask is nomask:
                    zero_divisor = (other.data == 0)
                else:
                    zero_divisor = (other.data == 0) & (~other._mask)
                if self._mask is nomask:
                    self._mask = zero_divisor
                else:
                    self._mask = self._mask | zero_divisor
            self._mask = self._detect_nan_inf(self.data, self._mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            self.data /= other_data
            self._mask = self._combine_masks(self._mask, other_mask)
            # Mask where divisor is zero
            if _xp.issubdtype(other_data.dtype, _xp.floating):
                if other_mask is nomask:
                    zero_divisor = (other_data == 0)
                else:
                    zero_divisor = (other_data == 0) & (~other_mask)
                if self._mask is nomask:
                    self._mask = zero_divisor
                else:
                    self._mask = self._mask | zero_divisor
            self._mask = self._detect_nan_inf(self.data, self._mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            self.data /= other_data
            # Mask where divisor is zero
            if _xp.issubdtype(other_data.dtype, _xp.floating):
                zero_divisor = (other_data == 0)
                if self._mask is nomask:
                    self._mask = zero_divisor
                else:
                    self._mask = self._mask | zero_divisor
            self._mask = self._detect_nan_inf(self.data, self._mask)
        else:
            
            self.data /= other
        return self

    def __rfloordiv__(self, other: object) -> "_XupyMaskedArray":
        """
        Reflected element-wise floor division with mask propagation.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = other.data // self.data
            result_mask = other._mask | self._mask
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            result_data = other_data // self.data
            result_mask = other_mask | self._mask
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = other_data // self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = other // self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)

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
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            self.data //= other_data
            self._mask |= other_mask
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            self.data //= other_data
        else:
            
            self.data //= other
        return self

    def __rmod__(self, other: object) -> "_XupyMaskedArray":
        """
        Reflected element-wise modulo operation with mask propagation.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = other.data % self.data
            result_mask = other._mask | self._mask
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            result_data = other_data % self.data
            result_mask = other_mask | self._mask
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = other_data % self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = other % self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)

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
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            self.data %= other_data
            self._mask |= other_mask
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            self.data %= other_data
        else:
            
            self.data %= other
        return self

    def __rpow__(self, other: object) -> "_XupyMaskedArray":
        """
        Reflected element-wise exponentiation with mask propagation.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = other.data ** self.data
            result_mask = other._mask | self._mask
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            result_data = other_data ** self.data
            result_mask = other_mask | self._mask
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = other_data ** self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = other ** self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)

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
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            self.data **= other_data
            self._mask |= other_mask
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            self.data **= other_data
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
            result_mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            other_data = _xp.asarray(other.data, dtype=self.dtype)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data @ other_data
            result_mask = self._combine_masks(self._mask, other_mask)
        elif isinstance(other, _np.ndarray):
            other_data = _xp.asarray(other)
            result_data = self.data @ other_data
            result_mask = self._mask
        else:
            result_data = self.data @ other
            result_mask = self._mask
        return _XupyMaskedArray(result_data, mask=result_mask)

    def __rmatmul__(self, other: object) -> "_XupyMaskedArray":
        """
        Reflected matrix multiplication with mask propagation.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = other.data @ self.data
            result_mask = self._combine_masks(other._mask, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        
        elif isinstance(other, _np.ma.masked_array):
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = other_data @ self.data
            result_mask = self._combine_masks(other_mask, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        
        elif isinstance(other, _np.ndarray):
            other_data = _xp.asarray(other)
            result_data = other_data @ self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)
        
        else:
            
            result_data = other @ self.data
            result_mask = self._mask
            return _XupyMaskedArray(result_data, result_mask)

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
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            self.data = self.data @ other_data
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
    def __eq__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise equality comparison with mask propagation.

        Returns
        -------
        _XupyMaskedArray
            Masked array with boolean comparison results. Masked elements in either
            operand result in masked elements in the output.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data == other.data
            result_mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data == other_data
            result_mask = self._combine_masks(self._mask, other_mask)
        else:
            result_data = self.data == other
            result_mask = self._mask
        return _XupyMaskedArray(result_data, result_mask)

    def __ne__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise inequality comparison with mask propagation.

        Returns
        -------
        _XupyMaskedArray
            Masked array with boolean comparison results. Masked elements in either
            operand result in masked elements in the output.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data != other.data
            result_mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data != other_data
            result_mask = self._combine_masks(self._mask, other_mask)
        else:
            result_data = self.data != other
            result_mask = self._mask
        return _XupyMaskedArray(result_data, result_mask)

    def __lt__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise less-than comparison with mask propagation.

        Returns
        -------
        _XupyMaskedArray
            Masked array with boolean comparison results. Masked elements in either
            operand result in masked elements in the output.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data < other.data
            result_mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data < other_data
            result_mask = self._combine_masks(self._mask, other_mask)
        else:
            result_data = self.data < other
            result_mask = self._mask
        return _XupyMaskedArray(result_data, result_mask)

    def __le__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise less-than-or-equal comparison with mask propagation.

        Returns
        -------
        _XupyMaskedArray
            Masked array with boolean comparison results. Masked elements in either
            operand result in masked elements in the output.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data <= other.data
            result_mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data <= other_data
            result_mask = self._combine_masks(self._mask, other_mask)
        else:
            result_data = self.data <= other
            result_mask = self._mask
        return _XupyMaskedArray(result_data, result_mask)

    def __gt__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise greater-than comparison with mask propagation.

        Returns
        -------
        _XupyMaskedArray
            Masked array with boolean comparison results. Masked elements in either
            operand result in masked elements in the output.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data > other.data
            result_mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data > other_data
            result_mask = self._combine_masks(self._mask, other_mask)
        else:
            result_data = self.data > other
            result_mask = self._mask
        return _XupyMaskedArray(result_data, result_mask)

    def __ge__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise greater-than-or-equal comparison with mask propagation.

        Returns
        -------
        _XupyMaskedArray
            Masked array with boolean comparison results. Masked elements in either
            operand result in masked elements in the output.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data >= other.data
            result_mask = self._combine_masks(self._mask, other._mask)
        elif isinstance(other, _np.ma.masked_array):
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data >= other_data
            result_mask = self._combine_masks(self._mask, other_mask)
        else:
            result_data = self.data >= other
            result_mask = self._mask
        return _XupyMaskedArray(result_data, result_mask)

    def __mul__(self, other: object):
        """
        Element-wise multiplication with mask propagation and NaN detection.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data * other.data
            result_mask = self._combine_masks(self._mask, other._mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data * other_data
            result_mask = self._combine_masks(self._mask, other_mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = self.data * other_data
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = self.data * other
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, result_mask)

    def __truediv__(self, other: object):
        """
        Element-wise true division with mask propagation and NaN detection.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data / other.data
            result_mask = self._combine_masks(self._mask, other._mask)
            # Also mask where divisor is zero (division by zero)
            if _xp.issubdtype(other.data.dtype, _xp.floating):
                if other._mask is nomask:
                    zero_divisor = (other.data == 0)
                else:
                    zero_divisor = (other.data == 0) & (~other._mask)
                # Only update mask if there are actual zeros
                if _xp.any(zero_divisor):
                    if result_mask is nomask:
                        result_mask = zero_divisor
                    else:
                        result_mask = result_mask | zero_divisor
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data / other_data
            result_mask = self._combine_masks(self._mask, other_mask)
            # Mask where divisor is zero
            if _xp.issubdtype(other_data.dtype, _xp.floating):
                if other_mask is nomask:
                    zero_divisor = (other_data == 0)
                else:
                    zero_divisor = (other_data == 0) & (~other_mask)
                # Only update mask if there are actual zeros
                if _xp.any(zero_divisor):
                    if result_mask is nomask:
                        result_mask = zero_divisor
                    else:
                        result_mask = result_mask | zero_divisor
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = self.data / other_data
            result_mask = self._mask
            # Mask where divisor is zero
            if _xp.issubdtype(other_data.dtype, _xp.floating):
                zero_divisor = (other_data == 0)
                result_mask = result_mask | zero_divisor if result_mask is not nomask else zero_divisor
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = self.data / other
            result_mask = self._mask
            # Mask where divisor is zero (for scalar)
            if other == 0:
                if result_mask is nomask:
                    result_mask = _xp.ones(self.data.shape, dtype=bool)
                else:
                    # Mask all elements when dividing by zero
                    result_mask = _xp.ones(self.data.shape, dtype=bool) | result_mask
            # Only call _detect_nan_inf if we have floating point or if result_mask is not nomask
            # (to avoid creating unnecessary mask arrays)
            if result_mask is not nomask or _xp.issubdtype(result_data.dtype, _xp.floating):
                result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)

    def __add__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise addition with mask propagation and NaN detection.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data + other.data
            result_mask = self._combine_masks(self._mask, other._mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, mask=result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            # Check for nomask BEFORE converting to array
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            else:
                # Convert mask to CuPy array
                other_mask = _xp.asarray(other.mask, dtype=bool)
                # Ensure mask has the right shape (numpy.ma might have scalar masks)
                if other_mask.ndim == 0:
                    # Scalar mask - broadcast to data shape
                    other_mask = _xp.full(other_data.shape, bool(other_mask), dtype=bool)
            result_data = self.data + other_data
            result_mask = self._combine_masks(self._mask, other_mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, mask=result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = self.data + other_data
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, mask=result_mask)
        else:
            
            result_data = self.data + other
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, mask=result_mask)

    def __sub__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise subtraction with mask propagation and NaN detection.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data - other.data
            result_mask = self._combine_masks(self._mask, other._mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, mask=result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data - other_data
            result_mask = self._combine_masks(self._mask, other_mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, mask=result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = self.data - other_data
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, mask=result_mask)
        else:
            
            result_data = self.data - other
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, mask=result_mask)

    def __pow__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise exponentiation with mask propagation and NaN detection.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data ** other.data
            result_mask = self._combine_masks(self._mask, other._mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data ** other_data
            result_mask = self._combine_masks(self._mask, other_mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = self.data ** other_data
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = self.data ** other
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, result_mask)

    def __floordiv__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise floor division with mask propagation and NaN detection.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data // other.data
            result_mask = self._combine_masks(self._mask, other._mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data // other_data
            result_mask = self._combine_masks(self._mask, other_mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = self.data // other_data
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = self.data // other
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, result_mask)

    def __mod__(self, other: object) -> "_XupyMaskedArray":
        """
        Element-wise modulo operation with mask propagation and NaN detection.
        """
        if isinstance(other, _XupyMaskedArray):
            result_data = self.data % other.data
            result_mask = self._combine_masks(self._mask, other._mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ma.masked_array):
            
            other_data = _xp.asarray(other.data)
            other_mask = _xp.asarray(other.mask, dtype=bool)
            if other.mask is _np.ma.nomask:
                other_mask = nomask
            result_data = self.data % other_data
            result_mask = self._combine_masks(self._mask, other_mask)
            result_mask = self._detect_nan_inf(result_data, result_mask)
            return _XupyMaskedArray(result_data, result_mask)
        elif isinstance(other, _np.ndarray):
            
            other_data = _xp.asarray(other)
            result_data = self.data % other_data
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, result_mask)
        else:
            
            result_data = self.data % other
            result_mask = self._detect_nan_inf(result_data, self._mask)
            return _XupyMaskedArray(result_data, result_mask)

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return len(self.data)

    def __iter__(self):
        """Return an iterator over the array."""
        for i in range(len(self)):
            yield self[i]

    def __array__(self, dtype=None):
        """Convert to numpy array."""
        return _xp.asnumpy(self.data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle ufunc calls."""
        if method == '__call__':
            # Handle the ufunc call
            if len(inputs) == 1:
                return self.apply_ufunc(ufunc)
            else:
                # For binary operations, we need to handle this in the arithmetic operators
                return NotImplemented
        return NotImplemented

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
        if self._mask is nomask:
            mask_item = nomask
        else:
            mask_item = self._mask[item]
        
        # If the result is a scalar, return a masked value
        if data_item.shape == ():
            if mask_item is not nomask and mask_item:
                return masked
            return data_item.item()
        return _XupyMaskedArray(data_item, mask_item)

    def __setitem__(self, key, value):
        """
        Set item(s) in the masked array.

        Parameters
        ----------
        key : int, slice, or array-like
            The index or slice to set.
        value : scalar or array-like
            The value to set.
        """
        # Handle hard mask: prevent unmasking of already masked elements
        if self._is_hard_mask and self._mask is not nomask:
            # Check if we're trying to unmask any elements
            if value is not masked:
                # Get the mask slice to check
                try:
                    mask_slice = self._mask[key]
                    if isinstance(mask_slice, _xp.ndarray):
                        # Check if any of the target positions are masked
                        if _xp.any(mask_slice):
                            # Some elements are masked - check if we're trying to unmask them
                            if isinstance(value, _XupyMaskedArray):
                                # If value has unmasked elements where we have masked, prevent it
                                if value._mask is not nomask:
                                    # Check if value would unmask any currently masked positions
                                    would_unmask = mask_slice & (~value._mask)
                                else:
                                    # Value has no mask, so it would unmask all masked positions
                                    would_unmask = mask_slice
                            else:
                                # Regular value assignment would unmask all masked positions
                                would_unmask = mask_slice
                            
                            if _xp.any(would_unmask):
                                raise ValueError(
                                    "Cannot unmask elements in a hard mask. "
                                    "Use soften_mask() first or assign masked values."
                                )
                    elif mask_slice:
                        # Single masked element - check if value would unmask it
                        if isinstance(value, _XupyMaskedArray):
                            if value._mask is nomask or not _xp.any(value._mask):
                                # Value is not masked, would unmask
                                raise ValueError(
                                    "Cannot unmask elements in a hard mask. "
                                    "Use soften_mask() first or assign masked values."
                                )
                        else:
                            # Regular value would unmask
                            raise ValueError(
                                "Cannot unmask elements in a hard mask. "
                                "Use soften_mask() first or assign masked values."
                            )
                except (IndexError, KeyError):
                    # Invalid key - let it fail naturally
                    pass
        
        # Handle different types of values
        if isinstance(value, _XupyMaskedArray):
            # Set data and combine masks
            self.data[key] = value.data
            if self._mask is nomask:
                self._mask = _xp.zeros(self.data.shape, dtype=bool)
            if value._mask is not nomask:
                self._mask[key] |= value._mask
        elif isinstance(value, _np.ma.MaskedArray):
            # Convert numpy masked array
            self.data[key] = _xp.asarray(value.data)
            if self._mask is nomask:
                self._mask = _xp.zeros(self.data.shape, dtype=bool)
            if value.mask is not _np.ma.nomask:
                self._mask[key] |= _xp.asarray(value.mask, dtype=bool)
        elif value is masked:
            # Set mask to True for these positions
            if self._mask is nomask:
                self._mask = _xp.zeros(self.data.shape, dtype=bool)
            self._mask[key] = True
        else:
            # Regular assignment
            self.data[key] = value
            # Unmask the assigned positions (only if not hard mask or not already masked)
            if self._mask is not nomask:
                if self._is_hard_mask:
                    # In hard mask, only unmask if not already masked
                    mask_slice = self._mask[key]
                    if isinstance(mask_slice, _xp.ndarray):
                        self._mask[key] = self._mask[key] & mask_slice  # Keep existing masks
                    else:
                        # Scalar mask - if already masked, keep it masked
                        if not mask_slice:
                            self._mask[key] = False
                else:
                    self._mask[key] = False

    def harden_mask(self) -> None:
        """
        Force the mask to hard, preventing unmasking of elements.
        
        When a mask is hard, masked values cannot be unmasked by assignment.
        This is useful to prevent accidental unmasking of invalid data.
        
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> arr = masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
        >>> arr.harden_mask()
        >>> arr[1] = 99.0  # Raises ValueError: Cannot unmask elements in a hard mask
        """
        self._is_hard_mask = True

    def soften_mask(self) -> None:
        """
        Force the mask to soft, allowing unmasking of elements.
        
        When a mask is soft (default), masked values can be unmasked by assignment.
        
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> arr = masked_array([1.0, 2.0, 3.0], mask=[False, True, False], hard_mask=True)
        >>> arr.soften_mask()
        >>> arr[1] = 99.0  # Now works
        """
        self._is_hard_mask = False

    def asmarray(
        self, **kwargs: dict[str, _t.Any]
    ) -> _np.ma.MaskedArray[_t.Any, _t.Any]:
        """
        Return a NumPy masked array on CPU.
        
        Parameters
        ----------
        dtype : dtype, optional
            Data type of the output.
            If `dtype` is None, the type of the data argument (``data.dtype``)
            is used. If `dtype` is not None and different from ``data.dtype``,
            a copy is performed.
        copy : bool, optional
            Whether to copy the input data (True), or to use a reference instead.
            Default is False.
        subok : bool, optional
            Whether to return a subclass of `MaskedArray` if possible (True) or a
            plain `MaskedArray`. Default is True.
        ndmin : int, optional
            Minimum number of dimensions. Default is 0.
        fill_value : scalar, optional
            Value used to fill in the masked values when necessary.
            If None, a default based on the data-type is used.
        keep_mask : bool, optional
            Whether to combine `mask` with the mask of the input data, if any
            (True), or to use only `mask` for the output (False). Default is True.
        hard_mask : bool, optional
            Whether to use a hard mask or not. With a hard mask, masked values
            cannot be unmasked. Default is False.
        shrink : bool, optional
            Whether to force compression of an empty mask. Default is True.
        order : {'C', 'F', 'A'}, optional
            Specify the order of the array.  If order is 'C', then the array
            will be in C-contiguous order (last-index varies the fastest).
            If order is 'F', then the returned array will be in
            Fortran-contiguous order (first-index varies the fastest).
            If order is 'A' (default), then the returned array may be
            in any order (either C-, Fortran-contiguous, or even discontiguous),
            unless a copy is required, in which case it will be C-contiguous.
            
        Returns
        -------
        numpy.ma.MaskedArray
            A NumPy masked array with data and mask copied to CPU memory.
            
        Examples
        --------
        >>> import xupy as xp
        >>> from xupy.ma import masked_array
        >>> data = xp.array([1.0, 2.0, 3.0])
        >>> mask = xp.array([False, True, False])
        >>> arr = masked_array(data, mask)
        >>> np_arr = arr.asmarray()
        >>> type(np_arr)
        <class 'numpy.ma.core.MaskedArray'>
        """
        dtype = kwargs.pop("dtype", self.dtype)
        fill_value = kwargs.pop("fill_value", self._fill_value)
        hard_mask = kwargs.pop("hard_mask", self._is_hard_mask)
        
        return _np.ma.masked_array(
            _xp.asnumpy(self.data),
            mask=_xp.asnumpy(self._mask),
            dtype=dtype,
            fill_value=fill_value,
            hard_mask=hard_mask,
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

__all__ = ["MaskedArray", "masked_array", "nomask", "masked", "getmask", "getmaskarray", "is_mask"]
