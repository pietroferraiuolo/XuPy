"""
MASKED ARRAY EXTRAS module
==========================

This module provides additional functions for xupy masked arrays.
"""
from .core import (
    MaskType,
    nomask,
    masked,
    MaskedArray,
)
import numpy as _np
import cupy as _cp          # type: ignore
from .. import typings as _t


def _ensure_masked_array(
    a: _t.ArrayLike,
    *,
    copy: bool = False,
) -> MaskedArray:
    """Return a XuPy `MaskedArray` view of the input."""
    if isinstance(a, MaskedArray):
        return a.copy() if copy else a

    if isinstance(a, _np.ma.MaskedArray):
        data = _cp.asarray(a.data)
        mask = nomask if a.mask is _np.ma.nomask else _cp.asarray(a.mask, dtype=MaskType)
        if copy:
            data = data.copy()
        return MaskedArray(data, mask=mask, dtype=data.dtype)

    data_arr = _cp.asarray(a)
    if copy:
        data_arr = data_arr.copy()
    return MaskedArray(data_arr, mask=nomask, dtype=data_arr.dtype)


def issequence(seq: _t.ArrayLike) -> bool:
    """Check if a sequence is a sequence (ndarray, list or tuple).
    
    Parameters
    ----------
    seq : array_like
        The object to check.
        
    Returns
    -------
    bool
        True if the object is a sequence (ndarray, list, or tuple).
        
    Examples
    --------
    >>> import xupy as xp
    >>> from xupy.ma.extras import issequence
    >>> issequence([1, 2, 3])
    True
    >>> issequence(xp.array([1, 2, 3]))
    True
    >>> issequence(42)
    False
    """
    return isinstance(seq, (_np.ndarray, _cp.ndarray, tuple, list))


def count_masked(arr: _t.ArrayLike, axis: _t.Optional[int] = None) -> int:
    """Count the number of masked elements along the given axis.
    
    Parameters
    ----------
    arr : Array
        An array with (possibly) masked elements.
    axis : int, optional
        Axis along which to count. If None (default), a flattened
        version of the array is used.

    Returns
    -------
    int or array
        The total number of masked elements (axis=None) or the number
        of masked elements along each slice of the given axis.
        
    Examples
    --------
    >>> import xupy as xp
    >>> from xupy.ma import masked_array
    >>> from xupy.ma.extras import count_masked
    >>> data = xp.array([1.0, 2.0, 3.0, 4.0])
    >>> mask = xp.array([False, True, False, True])
    >>> arr = masked_array(data, mask)
    >>> count_masked(arr)
    2
    """
    ma = _ensure_masked_array(arr)
    mask = ma.mask

    if mask is nomask:
        if axis is None:
            return _cp.array(0)
        result = _cp.zeros(ma.data.shape, dtype=_cp.int8).sum(axis=axis)
    else:
        mask_int = mask.astype(_cp.int64, copy=False)
        result = mask_int.sum(axis=axis)

    return result


def masked_all(shape: tuple[int, ...], dtype: _t.DTypeLike = _np.float32) -> MaskedArray:
    """Empty masked array with all elements masked.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the required MaskedArray, e.g., ``(2, 3)`` or ``2``.
    dtype : dtype, optional
        Data type of the output. Default is float32.
        
    Returns
    -------
    MaskedArray
        A masked array with all elements masked.
        
    Examples
    --------
    >>> import xupy as xp
    >>> from xupy.ma.extras import masked_all
    >>> arr = masked_all((2, 3))
    >>> arr
    masked_array(data=[[0. 0. 0.]
     [0. 0. 0.]], mask=[[True True True]
     [True True True]])
    """
    data = _cp.zeros(shape, dtype=dtype)
    mask = _cp.ones(shape, dtype=MaskType)
    return MaskedArray(data, mask=mask, dtype=dtype)


def masked_all_like(arr: _t.ArrayLike) -> MaskedArray:
    """Empty masked array with the properties of an existing array.
    
    Parameters
    ----------
    arr : array_like
        An array describing the shape and dtype of the required MaskedArray.
    
    Returns
    -------
    MaskedArray
        A masked array with all data masked.
        
    Examples
    --------
    >>> import xupy as xp
    >>> from xupy.ma.extras import masked_all_like
    >>> original = xp.array([[1, 2], [3, 4]])
    >>> arr = masked_all_like(original)
    >>> arr
    masked_array(data=[[0 0]
     [0 0]], mask=[[True True]
     [True True]])
    """
    arr_cp = _cp.asarray(arr)
    data = _cp.empty_like(arr_cp)
    mask = _cp.ones_like(arr_cp, dtype=MaskType)
    return MaskedArray(data, mask=mask, dtype=data.dtype)


def _normalize_axis_tuple(
    axis: _t.Union[int, tuple[int, ...], None],
    ndim: int,
) -> tuple[int, ...]:
    """Normalize axis to a tuple of non-negative integers."""
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(ndim + ax if ax < 0 else ax for ax in axis)


# ---- 2D compress / mask (point 2) --------------------------------------------

def compress_nd(
    x: _t.ArrayLike,
    axis: _t.Union[int, tuple[int, ...], None] = None,
) -> _cp.ndarray:
    """Suppress slices from dimensions that contain masked values.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on. If not a MaskedArray (or no elements masked),
        interpreted as MaskedArray with mask set to nomask.
    axis : int or tuple of ints, optional
        Which dimensions to suppress. If a tuple, those axes are suppressed.
        If an int, only that axis. If None, all axes are selected.

    Returns
    -------
    ndarray
        The compressed array (underlying data only).
    """
    ma_arr = _ensure_masked_array(x)
    m = ma_arr.mask
    data = ma_arr.data

    axis = _normalize_axis_tuple(axis, ma_arr.ndim)

    if m is nomask or not _cp.any(m):
        return data
    if _cp.all(m):
        return _cp.array([])

    for ax in axis:
        axes_other = tuple(i for i in range(ma_arr.ndim) if i != ax)
        keep = ~_cp.any(m, axis=axes_other)
        # Index: (slice(None),)*ax + (keep,) + (slice(None),)*(ndim - ax - 1)
        idx = (slice(None),) * ax + (keep,) + (slice(None),) * (ma_arr.ndim - ax - 1)
        data = data[idx]
        m = m[idx]
    return data


def compress_rowcols(
    x: _t.ArrayLike,
    axis: _t.Optional[int] = None,
) -> _cp.ndarray:
    """Suppress rows and/or columns of a 2-D array that contain masked values.

    - If axis is None, both rows and columns are suppressed.
    - If axis is 0, only rows are suppressed.
    - If axis is 1 or -1, only columns are suppressed.

    Parameters
    ----------
    x : array_like, MaskedArray
        Must be 2D.
    axis : int, optional
        Axis along which to suppress. Default is None.

    Returns
    -------
    ndarray
        The compressed array.
    """
    ma_arr = _ensure_masked_array(x)
    if ma_arr.ndim != 2:
        raise NotImplementedError("compress_rowcols works for 2D arrays only.")
    if axis is None:
        axis_tuple: tuple[int, ...] = (0, 1)
    else:
        axis_tuple = _normalize_axis_tuple(axis, 2)
    return compress_nd(x, axis=axis_tuple)


def compress_rows(a: _t.ArrayLike) -> _cp.ndarray:
    """Suppress whole rows of a 2-D array that contain masked values."""
    return compress_rowcols(a, 0)


def compress_cols(a: _t.ArrayLike) -> _cp.ndarray:
    """Suppress whole columns of a 2-D array that contain masked values."""
    return compress_rowcols(a, 1)


def mask_rowcols(
    a: _t.ArrayLike,
    axis: _t.Optional[int] = None,
) -> MaskedArray:
    """Mask rows and/or columns of a 2D array that contain masked values.

    - If axis is None, rows and columns are masked.
    - If axis is 0, only rows are masked.
    - If axis is 1 or -1, only columns are masked.

    The input array's mask is modified in place; the array is returned.
    """
    ma_arr = _ensure_masked_array(a, copy=False)
    if ma_arr.ndim != 2:
        raise NotImplementedError("mask_rowcols works for 2D arrays only.")
    m = ma_arr.mask
    if m is nomask:
        ma_arr._mask = _cp.zeros(ma_arr.data.shape, dtype=MaskType)
        m = ma_arr._mask
    elif not _cp.any(m):
        return ma_arr
    else:
        m = ma_arr._mask
    maskedval = _cp.nonzero(m)
    if axis is None or axis == 0:
        rows = _cp.unique(maskedval[0])
        m[rows, :] = True
    if axis is None or axis in (1, -1):
        cols = _cp.unique(maskedval[1])
        m[:, cols] = True
    return ma_arr


def mask_rows(a: _t.ArrayLike) -> MaskedArray:
    """Mask rows of a 2D array that contain masked values."""
    return mask_rowcols(a, 0)


def mask_cols(a: _t.ArrayLike) -> MaskedArray:
    """Mask columns of a 2D array that contain masked values."""
    return mask_rowcols(a, 1)


#####--------------------------------------------------------------------------
#----
#####--------------------------------------------------------------------------

def flatten_inplace(seq: _t.ArrayLike) -> _t.ArrayLike:
    """
    Flatten a sequence in place.
    """
    k = 0
    while (k != len(seq)):
        while hasattr(seq[k], '__iter__'):
            seq[k:(k + 1)] = seq[k]
        k += 1
    return seq

def sum(a: _t.ArrayLike, axis: _t.Optional[int] = None, dtype: _t.Optional[_t.DTypeLike] = None,
        out: _t.Optional[_t.ArrayLike] = None, keepdims: bool = False) -> _t.ArrayLike:
    """
    Return the sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
        Masked entries are not taken into account in the computation.
    axis : int, optional
        Axis along which the sum is computed. If None, sum over
        the flattened array.
    dtype : dtype, optional
        The type used in the summation.
    out : array, optional
        A location into which the result is stored.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.

    Returns
    -------
    sum_along_axis : scalar or MaskedArray
        A scalar if axis is None or result is 0-dimensional, otherwise
        an array with the specified axis removed.
        If `out` is specified, a reference to it is returned.

    """
    ma = _ensure_masked_array(a)
    kwargs: dict[str, _t.Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if out is not None:
        kwargs["out"] = out
    if keepdims:
        kwargs["keepdims"] = True
    result = ma.sum(axis=axis, **kwargs)
    return result


def mean(
    a: _t.ArrayLike,
    axis: _t.Optional[int] = None,
    dtype: _t.Optional[_t.DTypeLike] = None,
    out: _t.Optional[_t.ArrayLike] = None,
    keepdims: bool = False,
) -> _t.ArrayLike:
    """
    Return the mean of array elements over a given axis.
    Masked entries are ignored.
    """
    ma = _ensure_masked_array(a)
    kwargs: dict[str, _t.Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if out is not None:
        kwargs["out"] = out
    if keepdims:
        kwargs["keepdims"] = True
    result = ma.mean(axis=axis, **kwargs)
    return result


def _counts_for_axis(
    valid: _cp.ndarray,
    axis: _t.Optional[int],
    keepdims: bool,
) -> _cp.ndarray:
    """Count valid entries along an axis."""
    return valid.sum(axis=axis, keepdims=keepdims)


def prod(
    a: _t.ArrayLike,
    axis: _t.Optional[int] = None,
    dtype: _t.Optional[_t.DTypeLike] = None,
    keepdims: bool = False,
) -> _t.ArrayLike:
    """
    Return the product of array elements over a given axis.
    Masked entries are ignored.
    """
    ma = _ensure_masked_array(a)
    data = ma.data
    mask = ma.mask

    prod_kwargs: dict[str, _t.Any] = {}
    if dtype is not None:
        prod_kwargs["dtype"] = dtype
    if keepdims:
        prod_kwargs["keepdims"] = True

    if mask is nomask:
        result = _cp.prod(data, axis=axis, **prod_kwargs)
        return result

    valid = ~mask

    if axis is None:
        valid_count = int(valid.sum())
        if valid_count == 0:
            return masked
        result = _cp.prod(data[valid], **prod_kwargs)
        return result

    data_filled = _cp.where(valid, data, 1)
    result = _cp.prod(data_filled, axis=axis, **prod_kwargs)
    counts = _counts_for_axis(valid.astype(_cp.int8), axis, keepdims)
    mask_result = counts == 0
    result_ma = MaskedArray(result, mask=_cp.asarray(mask_result, dtype=MaskType), dtype=result.dtype)
    return result_ma


def product(
    a: _t.ArrayLike,
    axis: _t.Optional[int] = None,
    dtype: _t.Optional[_t.DTypeLike] = None,
    keepdims: bool = False,
) -> _t.ArrayLike:
    """Alias for :func:`prod`."""
    return prod(a, axis=axis, dtype=dtype, keepdims=keepdims)


def std(
    a: _t.ArrayLike,
    axis: _t.Optional[int] = None,
    dtype: _t.Optional[_t.DTypeLike] = None,
    out: _t.Optional[_t.ArrayLike] = None,
    ddof: int = 0,
    keepdims: bool = False,
) -> _t.ArrayLike:
    """Standard deviation of array elements over a given axis."""
    ma = _ensure_masked_array(a)
    kwargs: dict[str, _t.Any] = {"ddof": ddof}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if out is not None:
        kwargs["out"] = out
    if keepdims:
        kwargs["keepdims"] = True
    result = ma.std(axis=axis, **kwargs)
    return result


def var(
    a: _t.ArrayLike,
    axis: _t.Optional[int] = None,
    dtype: _t.Optional[_t.DTypeLike] = None,
    out: _t.Optional[_t.ArrayLike] = None,
    ddof: int = 0,
    keepdims: bool = False,
) -> _t.ArrayLike:
    """Variance of array elements over a given axis."""
    ma = _ensure_masked_array(a)
    kwargs: dict[str, _t.Any] = {"ddof": ddof}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if out is not None:
        kwargs["out"] = out
    if keepdims:
        kwargs["keepdims"] = True
    result = ma.var(axis=axis, **kwargs)
    return result


def min(
    a: _t.ArrayLike,
    axis: _t.Optional[int] = None,
    out: _t.Optional[_t.ArrayLike] = None,
    keepdims: bool = False,
) -> _t.ArrayLike:
    """Minimum of array elements over a given axis."""
    ma = _ensure_masked_array(a)
    kwargs: dict[str, _t.Any] = {}
    if out is not None:
        kwargs["out"] = out
    if keepdims:
        kwargs["keepdims"] = True
    result = ma.min(axis=axis, **kwargs)
    return result


def max(
    a: _t.ArrayLike,
    axis: _t.Optional[int] = None,
    out: _t.Optional[_t.ArrayLike] = None,
    keepdims: bool = False,
) -> _t.ArrayLike:
    """Maximum of array elements over a given axis."""
    ma = _ensure_masked_array(a)
    kwargs: dict[str, _t.Any] = {}
    if out is not None:
        kwargs["out"] = out
    if keepdims:
        kwargs["keepdims"] = True
    result = ma.max(axis=axis, **kwargs)
    return result

def average(a: _t.ArrayLike, axis: _t.Optional[int] = None, weights: _t.Optional[_t.ArrayLike] = None, returned: bool = False, *,
            keepdims: bool = False):
    """
    Return the weighted average of array over the given axis.

    Parameters
    ----------
    a : array_like
        Data to be averaged.
        Masked entries are not taken into account in the computation.
    axis : int, optional
        Axis along which to average `a`. If None, averaging is done over
        the flattened array.
    weights : array_like, optional
        The importance that each element has in the computation of the average.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If ``weights=None``, then all data in `a` are assumed to have a
        weight equal to one.  The 1-D calculation is::

            avg = sum(a * weights) / sum(weights)

        The only constraint on `weights` is that `sum(weights)` must not be 0.
    returned : bool, optional
        Flag indicating whether a tuple ``(result, sum of weights)``
        should be returned as output (True), or just the result (False).
        Default is False.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        *Note:* `keepdims` will not work with instances of `numpy.matrix`
        or other classes whose methods do not support `keepdims`.

        .. versionadded:: 1.23.0

    Returns
    -------
    average, [sum_of_weights] : (tuple of) scalar or MaskedArray
        The average along the specified axis. When returned is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. The return type is `np.float32`
        if `a` is of integer type and floats smaller than `float32`, or the
        input data-type, otherwise. If returned, `sum_of_weights` is always
        `float32`.

    Examples
    --------
    >>> a = np.ma.array([1., 2., 3., 4.], mask=[False, False, True, True])
    >>> np.ma.average(a, weights=[3, 1, 0, 0])
    1.25

    >>> x = np.ma.arange(6.).reshape(3, 2)
    >>> x
    masked_array(
      data=[[0., 1.],
            [2., 3.],
            [4., 5.]],
      mask=False,
      fill_value=1e+20)
    >>> avg, sumweights = np.ma.average(x, axis=0, weights=[1, 2, 3],
    ...                                 returned=True)
    >>> avg
    masked_array(data=[2.6666666666666665, 3.6666666666666665],
                 mask=[False, False],
           fill_value=1e+20)

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.ma.average(x, axis=1, keepdims=True)
    masked_array(
      data=[[0.5],
            [2.5],
            [4.5]],
      mask=False,
      fill_value=1e+20)
    """
    ma = _ensure_masked_array(a)
    data = ma.data
    mask = ma.mask

    if axis is not None:
        axis_norm = axis if axis >= 0 else axis + data.ndim
        if axis_norm < 0 or axis_norm >= data.ndim:
            raise _np.AxisError(axis, data.ndim)
    else:
        axis_norm = None

    sum_kwargs: dict[str, _t.Any] = {}
    if axis_norm is not None:
        sum_kwargs["axis"] = axis_norm
    if keepdims:
        sum_kwargs["keepdims"] = True

    if mask is nomask:
        valid = None
    else:
        valid = ~mask

    if weights is None:
        if valid is None:
            weighted_sum = _cp.sum(data, **sum_kwargs)
            sum_weights = _cp.sum(_cp.ones_like(data, dtype=_cp.float32), **sum_kwargs)
        else:
            data_filled = _cp.where(valid, data, 0)
            weighted_sum = _cp.sum(data_filled, **sum_kwargs)
            sum_weights = _cp.sum(valid.astype(_cp.float32), **sum_kwargs)
    else:
        wgt = _cp.asarray(weights)

        if wgt.shape != data.shape:
            if axis_norm is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights differ."
                )
            if wgt.ndim != 1:
                raise TypeError("1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != data.shape[axis_norm]:
                raise ValueError("Length of weights not compatible with specified axis.")

            reshape = [1] * data.ndim
            reshape[axis_norm] = wgt.shape[0]
            wgt = wgt.reshape(reshape)

        if valid is None:
            data_effective = data
            weights_effective = wgt
        else:
            data_effective = _cp.where(valid, data, 0)
            weights_effective = _cp.where(valid, wgt, 0)

        weighted_sum = _cp.sum(data_effective * weights_effective, **sum_kwargs)
        sum_weights = _cp.sum(weights_effective, **sum_kwargs)

    sum_weights = _cp.asarray(sum_weights, dtype=_cp.float32)
    mask_result = sum_weights == 0
    safe_denominator = _cp.where(mask_result, 1.0, sum_weights)
    avg = _cp.asarray(weighted_sum, dtype=_cp.result_type(weighted_sum, _cp.float32)) / safe_denominator
    avg = _cp.where(mask_result, 0.0, avg)

    if axis_norm is None:
        result = masked if bool(mask_result) else avg
        if returned:
            return result, sum_weights
        return result

    mask_array = _cp.asarray(mask_result, dtype=MaskType)
    average_ma = MaskedArray(avg, mask=mask_array, dtype=avg.dtype)

    if returned:
        return average_ma, sum_weights
    return average_ma


def empty_like(
    arr: _t.ArrayLike,
    dtype: _t.Optional[_t.DTypeLike] = None,
) -> MaskedArray:
    """
    Return a new masked array with the same shape and type as a given array.
    """
    arr_cp = _cp.asarray(arr)
    data = _cp.empty_like(arr_cp, dtype=dtype)
    mask = _cp.zeros(data.shape, dtype=MaskType)
    return MaskedArray(data, mask=mask, dtype=data.dtype)


# ---- From-NumPy-style wrappers (apply to data and mask) ----------------------

def _mask_for_nx(ma_arr: MaskedArray) -> _cp.ndarray:
    """Full boolean mask array for use in fromnx wrappers (CuPy)."""
    if ma_arr.mask is nomask:
        return _cp.zeros(ma_arr.data.shape, dtype=MaskType)
    return _cp.asarray(ma_arr.mask, dtype=MaskType)


def _fromnxfunction_single(npfunc: _t.Callable[..., _cp.ndarray], a: _t.ArrayLike, /, *args: _t.Any, **kwargs: _t.Any) -> MaskedArray:
    """Apply a CuPy function to one masked array (data and mask)."""
    ma_arr = _ensure_masked_array(a)
    data_out = npfunc(ma_arr.data, *args, **kwargs)
    mask_out = npfunc(_mask_for_nx(ma_arr), *args, **kwargs)
    return MaskedArray(data_out, mask=mask_out.astype(MaskType, copy=False), dtype=data_out.dtype)


def _fromnxfunction_seq(npfunc: _t.Callable[..., _cp.ndarray], arys: _t.Sequence[_t.ArrayLike], /, *args: _t.Any, **kwargs: _t.Any) -> MaskedArray:
    """Apply a CuPy function that takes a sequence of arrays (e.g. vstack)."""
    mas = [_ensure_masked_array(ary) for ary in arys]
    data_out = npfunc(tuple(m.data for m in mas), *args, **kwargs)
    mask_out = npfunc(tuple(_mask_for_nx(m) for m in mas), *args, **kwargs)
    return MaskedArray(data_out, mask=mask_out.astype(MaskType, copy=False), dtype=data_out.dtype)


def _fromnxfunction_allargs(npfunc: _t.Callable[..., _t.Union[_cp.ndarray, tuple]], *arys: _t.ArrayLike, **kwargs: _t.Any) -> _t.Union[MaskedArray, tuple[MaskedArray, ...]]:
    """Apply a CuPy function to multiple array arguments (e.g. atleast_1d)."""
    mas = [_ensure_masked_array(a) for a in arys]
    data_out = npfunc(*(m.data for m in mas), **kwargs)
    mask_out = npfunc(*(_mask_for_nx(m) for m in mas), **kwargs)
    if len(arys) == 1:
        return MaskedArray(data_out, mask=mask_out.astype(MaskType, copy=False), dtype=data_out.dtype)
    return tuple(
        MaskedArray(d, mask=m.astype(MaskType, copy=False), dtype=d.dtype)
        for d, m in zip(data_out, mask_out)
    )


def atleast_1d(*arys: _t.ArrayLike, **kwargs: _t.Any) -> _t.Union[MaskedArray, tuple[MaskedArray, ...]]:
    """Convert inputs to masked arrays with at least one dimension."""
    return _fromnxfunction_allargs(_cp.atleast_1d, *arys, **kwargs)


def atleast_2d(*arys: _t.ArrayLike, **kwargs: _t.Any) -> _t.Union[MaskedArray, tuple[MaskedArray, ...]]:
    """View inputs as masked arrays with at least two dimensions."""
    return _fromnxfunction_allargs(_cp.atleast_2d, *arys, **kwargs)


def atleast_3d(*arys: _t.ArrayLike, **kwargs: _t.Any) -> _t.Union[MaskedArray, tuple[MaskedArray, ...]]:
    """View inputs as masked arrays with at least three dimensions."""
    return _fromnxfunction_allargs(_cp.atleast_3d, *arys, **kwargs)


def vstack(tup: _t.Sequence[_t.ArrayLike], *args: _t.Any, **kwargs: _t.Any) -> MaskedArray:
    """Stack arrays in sequence vertically (row wise)."""
    return _fromnxfunction_seq(_cp.vstack, tup, *args, **kwargs)


def hstack(tup: _t.Sequence[_t.ArrayLike], *args: _t.Any, **kwargs: _t.Any) -> MaskedArray:
    """Stack arrays in sequence horizontally (column wise)."""
    return _fromnxfunction_seq(_cp.hstack, tup, *args, **kwargs)


def column_stack(tup: _t.Sequence[_t.ArrayLike], *args: _t.Any, **kwargs: _t.Any) -> MaskedArray:
    """Stack 1-D arrays as columns into a 2-D masked array."""
    return _fromnxfunction_seq(_cp.column_stack, tup, *args, **kwargs)


def dstack(tup: _t.Sequence[_t.ArrayLike], *args: _t.Any, **kwargs: _t.Any) -> MaskedArray:
    """Stack arrays in sequence depth wise (along third axis)."""
    return _fromnxfunction_seq(_cp.dstack, tup, *args, **kwargs)


def stack(arrays: _t.Sequence[_t.ArrayLike], axis: int = 0, *args: _t.Any, **kwargs: _t.Any) -> MaskedArray:
    """Join a sequence of masked arrays along a new axis."""
    return _fromnxfunction_seq(lambda t: _cp.stack(t, axis=axis, *args, **kwargs), arrays)


def row_stack(tup: _t.Sequence[_t.ArrayLike], *args: _t.Any, **kwargs: _t.Any) -> MaskedArray:
    """Stack arrays in sequence vertically (row wise). Alias for vstack."""
    return vstack(tup, *args, **kwargs)


def hsplit(ary: _t.ArrayLike, indices_or_sections: _t.Union[int, _t.ArrayLike], *args: _t.Any, **kwargs: _t.Any) -> list[MaskedArray]:
    """Split a masked array horizontally (column-wise). Returns list of MaskedArrays."""
    ma_arr = _ensure_masked_array(ary)
    data_parts = _cp.hsplit(ma_arr.data, indices_or_sections, *args, **kwargs)
    mask_parts = _cp.hsplit(_mask_for_nx(ma_arr), indices_or_sections, *args, **kwargs)
    return [MaskedArray(d, mask=m.astype(MaskType, copy=False), dtype=d.dtype) for d, m in zip(data_parts, mask_parts)]


def diagflat(v: _t.ArrayLike, k: int = 0) -> MaskedArray:
    """Create a 2-D masked array with the flattened input as a diagonal."""
    return _fromnxfunction_single(_cp.diagflat, v, k=k)


def ediff1d(ary: _t.ArrayLike, to_end: _t.Optional[_t.ArrayLike] = None, to_begin: _t.Optional[_t.ArrayLike] = None) -> MaskedArray:
    """First difference of a masked array. Masked where either adjacent input is masked."""
    ma_arr = _ensure_masked_array(ary)
    data = ma_arr.data
    m = _mask_for_nx(ma_arr)
    out_data = _cp.ediff1d(data, to_end=to_end, to_begin=to_begin)
    n = int(data.size)
    if n <= 1:
        # no diff part; only to_begin / to_end may contribute to out_data
        out_mask = _cp.zeros(out_data.size, dtype=MaskType)
    else:
        # output[i] masked if either input[i] or input[i+1] masked
        out_mask = (m[1:] | m[:-1]).astype(MaskType, copy=False)
        if to_begin is not None:
            nb = int(_cp.size(to_begin))
            out_mask = _cp.concatenate([_cp.zeros(nb, dtype=MaskType), out_mask])
        if to_end is not None:
            ne = int(_cp.size(to_end))
            out_mask = _cp.concatenate([out_mask, _cp.zeros(ne, dtype=MaskType)])
    return MaskedArray(out_data, mask=out_mask.astype(MaskType, copy=False), dtype=out_data.dtype)


class _MRClass:
    """Minimal mr_ equivalent: concatenation along first axis (row stack). Use as mr_[a, b] or mr_[a, b, c]."""

    def __getitem__(self, key: _t.Union[MaskedArray, _cp.ndarray, tuple]) -> MaskedArray:
        if isinstance(key, tuple):
            return vstack(key)
        return _ensure_masked_array(key)


mr_ = _MRClass()


__all__ = [
    "issequence",
    "count_masked",
    "masked_all",
    "masked_all_like",
    "compress_nd",
    "compress_rowcols",
    "compress_rows",
    "compress_cols",
    "mask_rowcols",
    "mask_rows",
    "mask_cols",
    "flatten_inplace",
    "sum",
    "mean",
    "prod",
    "product",
    "average",
    "std",
    "var",
    "min",
    "max",
    "empty_like",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "vstack",
    "hstack",
    "column_stack",
    "dstack",
    "stack",
    "row_stack",
    "hsplit",
    "diagflat",
    "ediff1d",
    "mr_",
]
