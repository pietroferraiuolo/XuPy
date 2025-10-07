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
    """
    Check if a sequence is a sequence (ndarray, list or tuple).
    """
    return isinstance(seq, (_np.ndarray, _cp.ndarray, tuple, list))


def count_masked(arr: _t.ArrayLike, axis: _t.Optional[int] = None) -> int:
    """
    Count the number of masked elements along the given axis.
    
    Parameters
    ----------
    arr : Array
        An array with (possibly) masked elements.
    axis : int, Optional
        Axis along which to count. If None (default), a flattened
        version of the array is used.

    Returns
    -------
    count : int, Array
        The total number of masked elements (axis=None) or the number
        of masked elements along each slice of the given axis.
    """
    m = getmaskarray(arr)
    return m.sum(axis)


def masked_all(shape: tuple[int, ...], dtype: _t.DTypeLike = _np.float64) -> MaskedArray:
    """
    Empty masked array with all elements masked.
    
    Parameters
    ----------
    shape : tuple of ints
        Shape of the required MaskedArray, e.g., ``(2, 3)`` or ``2``.
    dtype : dtype, optional
        Data type of the output.
    """
    return masked_array(data=_cp.zeros(shape, dtype), mask=_cp.ones(shape, dtype=MaskType))


def masked_all_like(arr: _t.ArrayLike) -> MaskedArray:
    """
    Empty masked array with the properties of an existing array.
    
    Parameters
    ----------
    arr : array_like
        An array describing the shape and dtype of the required MaskedArray.
    
    Returns
    -------
    a : MaskedArray
        A masked array with all data masked.
    """
    return masked_array(data=_cp.empty_like(arr), mask=_cp.ones_like(arr, dtype=MaskType))

__all__ = []
