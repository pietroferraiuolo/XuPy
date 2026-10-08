"""repr/str/format of XuPy masked arrays, identical to ``numpy.ma`` 2.x.

Only the items numpy would actually display are transferred from the device:
dimensions longer than what is printed are reduced on the device (head, tail
and, when summarizing, one arbitrary middle element) before the small block is
copied to the host and formatted by numpy itself.
"""
from __future__ import annotations

import builtins

import numpy as np

from . import _backend

__all__ = ["masked_repr", "masked_str"]

masked_print_option = np.ma.masked_print_option

# numpy.ma.MaskedArray._print_width / _print_width_1d: with a mask, dimensions
# longer than this are cut (head/tail) before the object conversion.
_PRINT_WIDTH = 100
_PRINT_WIDTH_1D = 1500

try:  # private numpy helpers, numpy>=2 layout
    from numpy._core.arrayprint import dtype_is_implied as _dtype_is_implied
    from numpy._core.arrayprint import dtype_short_repr as _dtype_short_repr
except ImportError:  # pragma: no cover - defensive fallback
    def _dtype_is_implied(dtype):
        return np.dtype(dtype).type in (np.float64, np.int_, np.complex128, np.bool_)

    def _dtype_short_repr(dtype):
        dtype = np.dtype(dtype)
        if dtype.names is not None or dtype.type is np.void:
            return str(dtype)
        return repr(dtype.name) if dtype.isbuiltin == 0 else repr(str(dtype))


def _is_real_mask(m):
    """True for a boolean array (numpy/cupy); False for the ``nomask`` singleton.

    Type based on purpose: it needs neither the singleton's identity nor a
    marker attribute, and also treats ``np.bool_`` as ``nomask``.
    """
    return isinstance(m, np.ndarray) or _backend.is_cupy_array(m)


def _xp_of(arr):
    return _backend.get_xp(arr)


def _reduce_axis(xp, arr, axis, index):
    return xp.take(arr, xp.asarray(index), axis=axis)


def _cut_print_width(xp, data, mask, width):
    """Mirror numpy.ma ``_insert_masked_print`` corner extraction."""
    ind = width // 2
    for axis in range(data.ndim):
        n = data.shape[axis]
        if n > width:
            idx = np.concatenate((np.arange(ind), np.arange(n - ind, n)))
            data = _reduce_axis(xp, data, axis, idx)
            mask = _reduce_axis(xp, mask, axis, idx)
    return data, mask


def _summary_block(xp, arrays, edgeitems):
    """Reduce every dim with n > 2*edgeitems to head + 1 middle + tail.

    numpy summarization shows head, '...', tail for such dims, so the reduced
    block (forced through summarization) prints exactly like the original.
    """
    first = arrays[0]
    out = list(arrays)
    for axis in range(first.ndim):
        n = first.shape[axis]
        if n > 2 * edgeitems:
            if edgeitems > 0:
                idx = np.concatenate((np.arange(edgeitems), [edgeitems],
                                      np.arange(n - edgeitems, n)))
            else:  # numpy then prints only the last item of the dimension
                idx = np.array([n - 1])
            idx = idx.astype(np.intp)
            out = [_reduce_axis(xp, o, axis, idx) for o in out]
    return out


def _summarize_needed(size):
    return size > np.get_printoptions()["threshold"]


def _replace_dtype_fields(dtype, primitive):
    try:
        from numpy.ma.core import _replace_dtype_fields as f
        return f(dtype, primitive)
    except ImportError:  # pragma: no cover
        return np.dtype(primitive)


def _recursive_printoption(result, mask):
    names = result.dtype.names
    if names is not None:
        for name in names:
            _recursive_printoption(result[name], mask[name])
    else:
        np.copyto(result, masked_print_option, where=mask)


def _display_block(a):
    """Host array equivalent of numpy's ``_insert_masked_print()`` result.

    Returns ``(block, summarized)``; when ``summarized`` is true the block
    must be printed with ``threshold=0`` (numpy would have summarized the
    full array).
    """
    xp = _xp_of(a._data)
    data, mask = a._data, a._mask
    has_mask = _is_real_mask(mask)
    if has_mask:
        width = _PRINT_WIDTH if a.ndim > 1 else _PRINT_WIDTH_1D
        data, mask = _cut_print_width(xp, data, mask, width)
    summarized = _summarize_needed(data.size)
    if summarized:
        edge = np.get_printoptions()["edgeitems"]
        arrs = _summary_block(xp, [data, mask] if has_mask else [data], edge)
        data = arrs[0]
        mask = arrs[1] if has_mask else mask
    hdata = _backend.to_host(data)
    if not has_mask:
        return hdata, summarized
    hmask = _backend.to_host(mask)
    res = hdata.astype(_replace_dtype_fields(hdata.dtype, "O"))
    _recursive_printoption(res, hmask)
    return res, summarized


def _mask_block(a):
    """Host mask (or ``nomask``-like scalar) and summarized flag."""
    m = a._mask
    if not _is_real_mask(m):
        return np.False_, False
    summarized = _summarize_needed(m.size)
    if summarized:
        edge = np.get_printoptions()["edgeitems"]
        m = _summary_block(_xp_of(m), [m], edge)[0]
    return _backend.to_host(m), summarized


def masked_str(a):
    """``str(np.ma.MaskedArray)`` equivalent."""
    if not masked_print_option.enabled():  # pragma: no cover - parity with numpy
        return str(a.filled(a.fill_value))
    block, summarized = _display_block(a)
    if summarized:
        with np.printoptions(threshold=0):
            return str(block)
    return str(block)


def _all_masked(a):
    m = a._mask
    if not _is_real_mask(m):
        return False
    return bool(_xp_of(m).all(m))


def masked_repr(a):
    """``repr(np.ma.MaskedArray)`` equivalent (numpy >= 2 layout)."""
    name = "array"
    prefix = f"masked_{name}("

    dtype_needed = (
        not _dtype_is_implied(a.dtype)
        or _all_masked(a)
        or a.size == 0
    )

    keys = ["data", "mask", "fill_value"]
    if dtype_needed:
        keys.append("dtype")

    is_one_row = builtins.all(dim == 1 for dim in a.shape[:-1])

    min_indent = 2
    if is_one_row:
        indents = {keys[0]: prefix}
        for k in keys[1:]:
            n = builtins.max(min_indent, len(prefix + keys[0]) - len(k))
            indents[k] = " " * n
        prefix = ""
    else:
        indents = dict.fromkeys(keys, " " * min_indent)
        prefix = prefix + "\n"

    reprs = {}
    block, summ = _display_block(a)
    extra = {"threshold": 0} if summ else {}
    reprs["data"] = np.array2string(
        block, separator=", ", prefix=indents["data"] + "data=", suffix=",",
        **extra)
    mblock, summ = _mask_block(a)
    extra = {"threshold": 0} if summ else {}
    reprs["mask"] = np.array2string(
        mblock, separator=", ", prefix=indents["mask"] + "mask=", suffix=",",
        **extra)

    fv = np.asarray(a.fill_value)
    if fv.dtype.kind in ("S", "U") and a.dtype.kind == fv.dtype.kind:
        fill_repr = repr(fv.item())
    elif fv.dtype == a.dtype and not a.dtype == object:
        fill_repr = str(fv[()])
    else:
        fill_repr = repr(fv[()])
    reprs["fill_value"] = fill_repr
    if dtype_needed:
        reprs["dtype"] = _dtype_short_repr(a.dtype)

    result = ",\n".join(f"{indents[k]}{k}={reprs[k]}" for k in keys)
    return prefix + result + ")"


class _PrintMixin:
    """``__repr__``/``__str__``/``__format__`` with numpy.ma semantics."""

    def __repr__(self):
        return masked_repr(self)

    def __str__(self):
        return masked_str(self)

    def __format__(self, format_spec):
        # Same as ndarray.__format__ (inherited by numpy.ma.MaskedArray):
        # n-d arrays only support the empty spec, 0-d format the data item.
        if self.ndim == 0:
            return format(_backend.host_scalar(self._data), format_spec)
        if format_spec == "":
            return str(self)
        raise TypeError(
            "unsupported format string passed to MaskedArray.__format__")
