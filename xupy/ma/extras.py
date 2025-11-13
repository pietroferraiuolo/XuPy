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


def _extract_scalar_if_0d(result: _t.Any) -> _t.Any:
    """
    Extract Python scalar from 0-dimensional array for NumPy compatibility.
    
    If result is a 0-dimensional CuPy/NumPy array, extract the scalar value.
    Otherwise, return as-is. This ensures compatibility with NumPy's behavior
    where reductions return scalars when appropriate.
    """
    if isinstance(result, (_cp.ndarray, _np.ndarray)):
        if result.ndim == 0:
            return result.item()
    return result


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
            return 0
        result = _cp.zeros(ma.data.shape, dtype=_cp.int8).sum(axis=axis)
    else:
        mask_int = mask.astype(_cp.int64, copy=False)
        result = mask_int.sum(axis=axis)

    if isinstance(result, _cp.ndarray):
        if result.ndim == 0:
            return int(result.item())
        return result
    return int(result)


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
    # Extract scalar for NumPy compatibility when result is 0-d
    if not keepdims:
        return _extract_scalar_if_0d(result)
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
    # Extract scalar for NumPy compatibility when result is 0-d
    if not keepdims:
        return _extract_scalar_if_0d(result)
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
        if not keepdims:
            return _extract_scalar_if_0d(result)
        return result

    valid = ~mask

    if axis is None:
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            return masked
        result = _cp.prod(data[valid], **prod_kwargs)
        return _extract_scalar_if_0d(result)

    data_filled = _cp.where(valid, data, 1)
    result = _cp.prod(data_filled, axis=axis, **prod_kwargs)
    counts = _counts_for_axis(valid.astype(_cp.int8), axis, keepdims)
    mask_result = counts == 0
    result_ma = MaskedArray(result, mask=_cp.asarray(mask_result, dtype=MaskType), dtype=result.dtype)
    # Extract scalar for NumPy compatibility when result is 0-d
    if not keepdims and result.ndim == 0:
        return result.item() if not mask_result.item() else masked
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
    # Extract scalar for NumPy compatibility when result is 0-d
    if not keepdims:
        return _extract_scalar_if_0d(result)
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
    # Extract scalar for NumPy compatibility when result is 0-d
    if not keepdims:
        return _extract_scalar_if_0d(result)
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
    # Extract scalar for NumPy compatibility when result is 0-d
    if not keepdims:
        return _extract_scalar_if_0d(result)
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
    # Extract scalar for NumPy compatibility when result is 0-d
    if not keepdims:
        return _extract_scalar_if_0d(result)
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
        masked_flag = bool(mask_result.item())
        if masked_flag:
            result = masked
        else:
            result = _extract_scalar_if_0d(avg)
        if returned:
            return result, float(sum_weights.item())
        return result

    mask_array = _cp.asarray(mask_result, dtype=MaskType)
    average_ma = MaskedArray(avg, mask=mask_array, dtype=avg.dtype)

    # Extract scalar for NumPy compatibility when result is 0-d
    if not keepdims and avg.ndim == 0:
        if mask_result.item():
            result = masked
        else:
            result = avg.item()
        if returned:
            return result, float(sum_weights.item())
        return result

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

__all__ = [
    "issequence",
    "count_masked",
    "masked_all",
    "masked_all_like",
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
]
