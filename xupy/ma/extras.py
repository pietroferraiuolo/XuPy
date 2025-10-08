"""
MASKED ARRAY EXTRAS module
==========================

This module provides additional functions for xupy masked arrays.
"""
from .core import (
    MaskType, nomask, MaskedArray, masked_array, getmask, getmaskarray,
)
import numpy as _np
import cupy as _cp
from .. import typings as _t


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
    m = getmaskarray(arr)
    # Use CuPy operations for GPU acceleration
    if axis is None:
        return int(_cp.sum(m))
    else:
        return int(_cp.sum(m, axis=axis))


def masked_all(shape: tuple[int, ...], dtype: _t.DTypeLike = _np.float64) -> MaskedArray:
    """Empty masked array with all elements masked.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the required MaskedArray, e.g., ``(2, 3)`` or ``2``.
    dtype : dtype, optional
        Data type of the output. Default is float64.
        
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
    return masked_array(data=_cp.zeros(shape, dtype), mask=_cp.ones(shape, dtype=MaskType))


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
    return masked_array(data=_cp.empty_like(arr), mask=_cp.ones_like(arr, dtype=MaskType))

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
        of the weights as the second element. The return type is `np.float64`
        if `a` is of integer type and floats smaller than `float64`, or the
        input data-type, otherwise. If returned, `sum_of_weights` is always
        `float64`.

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
    a = _cp.asarray(a)
    m = getmask(a)

    # inspired by 'average' in numpy/lib/function_base.py

    if keepdims is False:
        # Don't pass on the keepdims argument if one wasn't given.
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}

    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        scl = avg.dtype.type(a.count(axis))
    else:
        wgt = _cp.asarray(weights)

        if issubclass(a.dtype.type, (_cp.integer, _cp.bool_)):
            result_dtype = _cp.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = _cp.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = _cp.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape, subok=True)
            wgt = wgt.swapaxes(-1, axis)

        if m is not nomask:
            wgt = wgt*(~a.mask)
            wgt.mask |= a.mask

        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        avg = _cp.multiply(a, wgt,
                          dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != avg.shape:
            scl = _cp.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg

__all__ = []
