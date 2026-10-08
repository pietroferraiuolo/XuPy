"""
XUPY MASKED ARRAY
=================

Masked arrays on numpy or cupy data with the ``numpy.ma`` interface.

This module holds the data model (constructor, mask/fill-value handling,
indexing, copies, Python protocols) and the module-level mask helpers.
Operators/ufuncs, reductions and printing live in the mixins ``_ops``,
``_reductions`` and ``_printing``.
"""
from __future__ import annotations

import builtins as _builtins
import operator as _operator
import warnings as _warnings

import numpy as _np

from . import _backend
from ._backend import cupy_module as _cupy_module
from ._backend import get_xp as _get_xp
from ._backend import host_scalar as _host_scalar
from ._backend import is_cupy_array as _is_cupy_array
from ._backend import to_host as _to_host
from ._ops import _OpsMixin
from ._printing import _PrintMixin
from ._reductions import _ReductionsMixin
from ._singletons import MaskType, MAError, MaskError, masked, nomask

masked_print_option = _np.ma.masked_print_option

_NO_COPY_MSG = (
    "Unable to avoid copy while creating a host array: the data lives on a "
    "CUDA device."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_xp(x, xp):
    """Move a numpy/cupy array to the device of ``xp``; leave anything else."""
    if xp is _np:
        return x.get() if _is_cupy_array(x) else x
    return xp.asarray(x) if isinstance(x, _np.ndarray) else x


def _mask_array(m, xp, copy=None):
    """Bool array of ``xp`` from a mask-like (bool/array/list/np.ma mask)."""
    if isinstance(m, (_XupyMaskedArray, _np.ma.MaskedArray)):
        m = getdata(m)
    if m is nomask:
        return xp.zeros((), dtype=bool)
    return _backend.asarray(m, xp, dtype=bool, copy=copy)


def _zeros_mask(shape, xp):
    return xp.zeros(shape, dtype=bool)


def _has_ellipsis(indx):
    if indx is Ellipsis:
        return True
    return isinstance(indx, tuple) and any(i is Ellipsis for i in indx)


_FIELD_INDEX_MSG = ("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) "
                    "and integer or boolean arrays are valid indices")


def _prep_index(indx, xp):
    """Index usable on ``xp`` arrays (devices matched, lists converted)."""
    if isinstance(indx, tuple):
        return tuple(_prep_index(i, xp) for i in indx)
    if isinstance(indx, str):
        raise IndexError(_FIELD_INDEX_MSG)  # no structured dtypes: same error as numpy on a plain array
    if isinstance(indx, (_XupyMaskedArray, _np.ma.MaskedArray)):
        indx = getdata(indx)
    if isinstance(indx, (float, complex, _np.floating, _np.complexfloating)):
        raise IndexError(_FIELD_INDEX_MSG)  # cupy would silently accept float scalars
    if isinstance(indx, list):
        return xp.asarray(indx) if xp is not _np else indx
    return _to_xp(indx, xp)


def _wrap(data, mask, *, like=None, fill_value=None, hardmask=None, sharedmask=False):
    """Build a masked array from ``data``/``mask`` as-is (no checks, no copies).

    ``mask`` must be ``nomask`` or a bool array of exactly ``data.shape`` on
    the device of ``data``.  ``like`` (a masked array) propagates its
    fill_value/hardmask as numpy's ``_update_from`` does.
    """
    out = object.__new__(_XupyMaskedArray)
    out._data = data
    out._mask = mask
    out._sharedmask = sharedmask
    if like is not None:
        out._update_from(like)
    if fill_value is not None:
        out._fill_value = _check_fill_value(fill_value, data.dtype)
    if hardmask is not None:
        out._hardmask = hardmask
    return out


def _unpack(x):
    """``(data, mask)`` of any operand, without moving it between devices.

    XuPy/numpy masked arrays give their data and mask (``nomask`` if none),
    ``masked`` a 0-d float64 zero with a 0-d True mask, anything else
    ``(array, nomask)``.
    """
    if isinstance(x, _XupyMaskedArray) or getattr(x, "_is_xupy_masked_constant", False):
        return x._data, x._mask
    if isinstance(x, _np.ma.MaskedArray):
        m = _np.ma.getmask(x)
        return _np.ma.getdata(x), (nomask if m is _np.ma.nomask else m)
    if isinstance(x, _np.ndarray) or _is_cupy_array(x):
        return x, nomask
    return _np.asarray(x), nomask


def _mask_or(m1, m2, xp=None):
    """Logical or of two masks: ``nomask`` if both are, else a NEW bool array."""
    if m1 is nomask and m2 is nomask:
        return nomask
    xp = xp or _get_xp(m1, m2)
    if m1 is nomask or m2 is nomask:
        m = m2 if m1 is nomask else m1
        return xp.array(_to_xp(m, xp), dtype=bool, copy=True)
    return xp.logical_or(_to_xp(m1, xp), _to_xp(m2, xp))


def _maybe_scalar(data, mask):
    """Scalar result of an operation: ``masked``, a numpy scalar, or an array."""
    if data.ndim == 0:
        if mask is not nomask and bool(mask):
            return masked
        return _host_scalar(data)
    return _wrap(data, mask)


# ---------------------------------------------------------------------------
# fill values
# ---------------------------------------------------------------------------
def default_fill_value(obj):
    """Default fill value for an array, dtype or scalar (as ``numpy.ma``)."""
    dt = getattr(obj, "dtype", None)
    return _np.ma.default_fill_value(obj if dt is None else _np.dtype(dt))


def _check_fill_value(fill_value, ndtype):
    """Validate ``fill_value`` for ``ndtype``; always return a 0-d numpy array."""
    ndtype = _np.dtype(ndtype)
    if fill_value is None:
        fill_value = default_fill_value(ndtype)
        if ndtype.kind == "u":
            fill_value = _np.uint(fill_value)
    elif isinstance(fill_value, str) and ndtype.char not in "OSTVU":
        raise TypeError(f"Cannot set fill value of string with array of dtype {ndtype}")
    else:
        if isinstance(fill_value, _XupyMaskedArray):
            fill_value = fill_value._data
        if _is_cupy_array(fill_value):
            fill_value = fill_value.get()
        try:
            fill_value = _np.asarray(fill_value, dtype=ndtype)
        except (OverflowError, ValueError) as e:
            raise TypeError(f"Cannot convert fill_value {fill_value} to dtype {ndtype}") from e
    return _np.array(fill_value)


def set_fill_value(a, fill_value):
    """Set the fill value of ``a`` if it is a masked array."""
    if isinstance(a, _XupyMaskedArray):
        a.fill_value = fill_value


# ---------------------------------------------------------------------------
# class
# ---------------------------------------------------------------------------
class _XupyMaskedArray(_OpsMixin, _ReductionsMixin, _PrintMixin):
    """
    Masked array on numpy or cupy data, with the interface of ``numpy.ma``.

    Parameters
    ----------
    data : array_like
        Input data (cupy/numpy array, ``numpy.ma`` or XuPy masked array, list
        or scalar).
    mask : sequence, bool, optional
        Mask (True = masked).  Scalars/size-1 masks are broadcast, masks of the
        same size are reshaped; otherwise `MaskError` is raised.  ``None`` and
        `nomask` mean "no mask".
    dtype : dtype, optional
        Data type; default is the data's.  Only numeric/boolean dtypes are
        supported (no structured, string or object dtypes).
    copy : bool, optional
        Copy the data (and mask) if True; otherwise only if needed.
    subok, ndmin, order : optional
        As in `numpy.ma.MaskedArray` (``subok`` is accepted and ignored).
    fill_value : scalar, optional
        Value used by `filled`; validated and cast to the dtype.  Default: the
        input's fill value, else the dtype's default (lazily).
    keep_mask : bool, optional
        Combine `mask` with the input's mask (True) or discard the latter.
    hard_mask : bool, optional
        Hard masks cannot be unmasked by assignment.  Default: the input's
        setting, else False.
    shrink : bool, optional
        With ``keep_mask=False`` and no mask, store `nomask` instead of an
        all-False array.

    Notes
    -----
    The array lives on the device of its data: a cupy array gives a cupy-backed
    masked array, a numpy array (or `numpy.ma` array) a numpy-backed one, and
    lists/scalars use the active XuPy backend.  The mask always follows the
    data.  Use `to_device` to move an array.  Methods mirror `numpy.ma`;
    scalars (``a[i]``, full reductions) come back as numpy scalars or `masked`.
    Instances hold ``_data``, ``_mask`` (`nomask` or a bool array of the data's
    shape), ``_fill_value``, ``_hardmask`` and ``_sharedmask``.
    """

    _is_xupy_masked = True
    __array_priority__ = 15
    _fill_value = None
    _hardmask = False
    _sharedmask = False
    _mask = nomask

    def __init__(
        self,
        data=None,
        mask=nomask,
        dtype=None,
        copy=False,
        subok=True,
        ndmin=0,
        fill_value=None,
        keep_mask=True,
        hard_mask=None,
        shrink=True,
        order=None,
    ):
        copy = True if copy else None
        src_mask, src_fill, src_hard = nomask, None, None
        raw = data
        is_seq = isinstance(data, (list, tuple))
        if isinstance(data, (_XupyMaskedArray, _np.ma.MaskedArray)) or getattr(
            data, "_is_xupy_masked_constant", False
        ):
            raw, src_mask = _unpack(data)
            src_fill = getattr(data, "_fill_value", None)
            src_hard = getattr(data, "_hardmask", None)

        if isinstance(raw, _np.ndarray):
            xp = _np
        elif _is_cupy_array(raw):
            xp = _cupy_module()
        elif is_seq and _get_xp(*raw) is not _np:
            xp = _get_xp(*raw)
        else:
            xp = _backend.default_xp()

        dtype = None if dtype is None else _np.dtype(dtype)
        if xp is _np:
            arr = _np.array(raw, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
        else:
            arr = xp.array(raw, dtype=dtype, order=order or "K") if copy else xp.asarray(raw, dtype=dtype, order=order)
            if arr.ndim < ndmin:
                arr = arr.reshape((1,) * (ndmin - arr.ndim) + arr.shape)
        if arr.dtype.kind not in "biufc":
            raise NotImplementedError(
                f"dtype {arr.dtype} is not supported: only numeric and boolean data."
            )
        if src_mask is not nomask and src_mask.shape != arr.shape:
            src_mask = src_mask.reshape(arr.shape)
            copy = True

        if mask is _np.ma.nomask:
            mask = nomask
        elif mask is None:
            mask = False  # as numpy.ma: np.array(None, dtype=bool) is False
        if mask is nomask:
            if not keep_mask:
                m = nomask if shrink else _zeros_mask(arr.shape, xp)
                shared = False
            elif is_seq:
                m, shared = nomask, False
            else:
                shared = not copy
                m = src_mask if (src_mask is nomask or not copy) else src_mask.copy()
        else:
            if mask is True or mask is False:
                m = (xp.ones if mask else xp.zeros)(arr.shape, dtype=bool)
                mcopy = True
            else:
                m = _mask_array(mask, xp, copy=copy)
                mcopy = copy
            if m.shape != arr.shape:
                nd, nm = arr.size, m.size
                if nm == 1:
                    m = xp.broadcast_to(m.reshape(()), arr.shape).copy()
                elif nm == nd:
                    m = m.reshape(arr.shape)
                else:
                    raise MaskError(
                        f"Mask and data not compatible: data size is {nd}, mask size is {nm}."
                    )
                mcopy = True
            if src_mask is nomask or not keep_mask:
                shared = not mcopy
            else:
                m = xp.logical_or(m, src_mask)
                shared = False
        self._data = arr
        self._mask = m
        self._sharedmask = shared

        if fill_value is None:
            fill_value = src_fill
        if fill_value is not None:
            self._fill_value = _check_fill_value(fill_value, arr.dtype)
        self._hardmask = bool(src_hard) if hard_mask is None else bool(hard_mask)

    def _update_from(self, obj):
        """Copy fill_value (re-validated if the dtype changed) and hardmask."""
        fv = getattr(obj, "_fill_value", None)
        if fv is not None and getattr(obj, "dtype", None) != self.dtype:
            try:
                fv = _check_fill_value(fv, self.dtype)
            except (TypeError, ValueError, OverflowError):
                fv = None
        self._fill_value = fv
        self._hardmask = getattr(obj, "_hardmask", False)

    # ---- data, mask, metadata --------------------------------------------
    @property
    def _xp(self):
        return _get_xp(self._data)

    @property
    def data(self):
        """The data as a plain numpy/cupy array (a view)."""
        return self._data.view()

    @property
    def mask(self):
        """Current mask (a view of the mask array, or `nomask`)."""
        m = self._mask
        return m if m is nomask else m.view()

    @mask.setter
    def mask(self, value):
        self.__setmask__(value)

    def __setmask__(self, mask, copy=False):
        """Set the mask in place (OR-ing into it for hard masks)."""
        if mask is masked or mask is _np.ma.masked:
            mask = True
        cur = self._mask
        if cur is nomask:
            if mask is nomask or mask is _np.ma.nomask:
                return
            cur = self._mask = _zeros_mask(self.shape, self._xp)
        xp = self._xp
        if self._hardmask and not (mask is True or mask is False or mask is nomask):
            # numpy does `current_mask |= mask` in place: non-bool operands fail the same way
            probe = _np.zeros((), dtype=bool)
            probe |= _np.zeros((), dtype=getattr(mask, "dtype", None) or _np.asarray(mask).dtype)
        if mask is True or mask is False:
            m = xp.asarray(mask)
        else:
            m = xp.zeros((), dtype=bool) if mask is nomask else _mask_array(mask, xp)
            if m.size == 1:
                m = m.reshape(())
            elif m.size == cur.size:
                m = m.reshape(cur.shape)
            elif self._hardmask or m.size == 0:
                raise ValueError(f"operands could not be broadcast: mask size {m.size}, data size {cur.size}")
            else:  # like numpy's `flat = mask`: repeat cyclically
                m = xp.tile(m.ravel(), -(-cur.size // m.size))[: cur.size].reshape(cur.shape)
        if self._hardmask:
            cur |= m
        else:
            cur[...] = m

    _set_mask = __setmask__

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def nbytes(self):
        return self._data.nbytes

    @property
    def itemsize(self):
        return self._data.itemsize

    @property
    def strides(self):
        return self._data.strides

    @property
    def flags(self):
        return self._data.flags

    @property
    def base(self):
        return self._data.base

    @property
    def device(self):
        """``"cpu"`` for numpy data, the cupy ``Device`` for cupy data."""
        return "cpu" if self._xp is _np else self._data.device

    def to_device(self, device, /, *, stream=None):
        """Array on ``device``: "cpu", "gpu"/"cuda"[:N], an int or a cupy Device.

        Returns ``self`` if it already lives there.
        """
        cp = _cupy_module()
        idx = None
        if isinstance(device, str):
            name, _, num = device.lower().partition(":")
            if name not in ("cpu", "gpu", "cuda") or (num and name == "cpu"):
                raise ValueError(f"Unknown device {device!r}")
            gpu = name != "cpu"
            idx = int(num) if num else None
        elif isinstance(device, int):
            gpu, idx = True, device
        elif cp is not None and isinstance(device, cp.cuda.Device):
            gpu, idx = True, device.id
        else:
            raise ValueError(f"Unknown device {device!r}")
        if not gpu:
            if self._xp is _np:
                return self
            conv = _to_host
        else:
            if cp is None:
                raise ValueError("CuPy is not available: cannot move data to a GPU.")
            if self._xp is cp and (idx is None or self._data.device.id == idx):
                return self
            dev = cp.cuda.Device(idx) if idx is not None else cp.cuda.Device()

            def conv(a):
                with dev:
                    return cp.asarray(a)

        m = self._mask
        return _wrap(conv(self._data), m if m is nomask else conv(m), like=self)

    @property
    def T(self):
        return self.transpose()

    @property
    def mT(self):
        """Matrix transpose (last two axes swapped), mask included."""
        if self.ndim < 2:
            raise ValueError("matrix transpose with ndim < 2 is undefined")
        return self.swapaxes(-1, -2)

    def _part(self, name):
        m = self._mask
        return _wrap(getattr(self._data, name), m if m is nomask else m.copy(), like=self)

    real = property(lambda self: self._part("real"), doc="Real part (a view of the data).")
    imag = property(lambda self: self._part("imag"), doc="Imaginary part.")
    get_real = real.fget
    get_imag = imag.fget

    @property
    def fill_value(self):
        """Fill value of `filled` (a numpy scalar); ``None`` resets the default."""
        if self._fill_value is None:
            self._fill_value = _check_fill_value(None, self.dtype)
        return self._fill_value[()]

    @fill_value.setter
    def fill_value(self, value=None):
        target = _check_fill_value(value, self.dtype)
        if target.ndim != 0:
            _warnings.warn(
                "Non-scalar arrays for the fill value are deprecated. Use arrays "
                "with scalar values instead. The filled function still supports "
                "any array as `fill_value`.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self._fill_value is None:
            self._fill_value = target
        else:  # fill in place so that views see the change
            self._fill_value[()] = target

    get_fill_value = fill_value.fget
    set_fill_value = fill_value.fset

    @property
    def hardmask(self):
        """Whether the mask is hard (masked values cannot be unmasked by assignment)."""
        return self._hardmask

    _is_hard_mask = property(
        lambda self: self._hardmask, lambda self, v: setattr(self, "_hardmask", bool(v))
    )

    @property
    def sharedmask(self):
        """Whether the mask is shared with another array (read-only)."""
        return self._sharedmask

    def harden_mask(self):
        """Make the mask hard; returns ``self``."""
        self._hardmask = True
        return self

    def soften_mask(self):
        """Make the mask soft (default); returns ``self``."""
        self._hardmask = False
        return self

    def unshare_mask(self):
        """Copy the mask if it is shared; returns ``self``."""
        if self._sharedmask:
            self._mask = self._mask.copy()
            self._sharedmask = False
        return self

    def shrink_mask(self):
        """Replace an all-False mask by `nomask` (one device sync); returns ``self``."""
        if self._mask is not nomask and not bool(self._mask.any()):
            self._mask = nomask
        return self

    def is_masked(self):
        """True if any element is masked (one device sync)."""
        return is_masked(self)

    # ---- shape manipulation ----------------------------------------------
    def _apply(self, fn, view=True):
        """Apply an array function to data and mask (nomask-safe)."""
        m = self._mask
        return _wrap(
            fn(self._data),
            m if m is nomask else fn(m),
            like=self,
            # numpy builds non-view results with ``view(cls)``, which flags the (fresh) mask as shared
            sharedmask=self._sharedmask if view else True,
        )

    def copy(self, order="C"):
        """Copy of the array (data, mask, fill_value, hardmask)."""
        out = self._apply(lambda a: a.copy(order), view=False)
        if out._fill_value is not None:
            out._fill_value = out._fill_value.copy()
        return out

    __copy__ = copy

    def __deepcopy__(self, memo=None):
        return self.copy()

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        """Cast to ``dtype`` (a copy unless ``copy=False`` and nothing changes)."""
        kw = {} if casting == "unsafe" else {"casting": casting}
        d = self._data.astype(dtype, order=order, copy=copy, **kw)
        if d is self._data:
            return self
        m = self._mask
        return _wrap(d, m if m is nomask else m.copy(), like=self, sharedmask=True)

    def view(self, dtype=None, type=None, fill_value=None):
        """New array sharing data and mask, optionally with another dtype."""
        if type is None and isinstance(dtype, _builtins.type) and issubclass(dtype, _np.ndarray):
            type, dtype = dtype, None
        if type is not None and type is not _np.ndarray and not issubclass(type, _XupyMaskedArray):
            raise NotImplementedError("Only ndarray and masked array views are supported.")
        if type is _np.ndarray:
            return self._data.view(dtype)
        if dtype is not None and _np.dtype(dtype).itemsize != self.dtype.itemsize:
            raise NotImplementedError("Views with a different itemsize are not supported.")
        d = self._data if dtype is None else self._data.view(dtype)
        out = _wrap(d.view(), self._mask, like=self, sharedmask=self._sharedmask)
        if fill_value is not None:
            out.fill_value = fill_value
        elif dtype is not None:
            out._fill_value = None
        return out

    def reshape(self, *s, **kwargs):
        """Reshaped view (data and mask); see `numpy.ndarray.reshape`."""
        if not s:
            raise TypeError("reshape() takes exactly 1 argument (0 given)")
        return self._apply(lambda a: a.reshape(*s, **kwargs))

    def flatten(self, order="C"):
        """1-D copy."""
        return self._apply(lambda a: a.flatten(order), view=False)

    def ravel(self, order="C"):
        """1-D view when possible ('K'/'A' use the data's memory order)."""
        if order in "kKaA":
            fl = self._data.flags
            order = "F" if fl.f_contiguous and not fl.c_contiguous else "C"
        return self._apply(lambda a: a.ravel(order))

    def squeeze(self, axis=None):
        return self._apply(lambda a: a.squeeze(axis))

    def expand_dims(self, axis):
        """Insert a length-1 axis at ``axis``."""
        xp = self._xp
        return self._apply(lambda a: xp.expand_dims(a, axis))

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """Diagonal of data and mask (a view); see `numpy.ndarray.diagonal`."""
        return self._apply(lambda a: a.diagonal(offset, axis1, axis2))

    def compress(self, condition, axis=None, out=None):
        """Select slices along ``axis`` where ``condition`` is true (data and mask)."""
        if out is not None:
            raise NotImplementedError("compress(out=...) is not supported.")
        xp = self._xp
        cond = _backend.asarray(getdata(condition), xp)
        return self._apply(lambda a: a.compress(cond, axis=axis), view=False)

    def put(self, indices, values, mode="raise"):
        """Set storage-indexed locations to ``values`` (the mask follows, as numpy.ma)."""
        xp = self._xp
        idx = _backend.asarray(indices, xp)
        vd, vm = _unpack(values)
        vd = _backend.asarray(vd, xp)
        vm = vm if vm is nomask else _backend.asarray(vm, xp)
        if self._hardmask and self._mask is not nomask:
            keep = ~self._mask[idx]
            idx = idx[keep]
            vd = xp.resize(vd, keep.shape)[keep]
            vm = vm if vm is nomask else xp.resize(vm, keep.shape)[keep]
        self._data.put(idx, vd, mode=mode)
        if self._mask is nomask and vm is nomask:
            return
        m = getmaskarray(self)
        m.put(idx, False if vm is nomask else vm, mode=mode)
        self._mask = m

    def searchsorted(self, v, side="left", sorter=None):
        """Indices where ``v`` would be inserted in the (unfilled) data."""
        xp = self._xp
        v = _backend.asarray(getdata(v), xp)
        res = xp.searchsorted(self._data, v, side=side, sorter=sorter)
        return _host_scalar(res) if res.ndim == 0 else res

    def partition(self, *args, **kwargs):
        """In-place partition of the data; the mask is ignored (as numpy.ma)."""
        _warnings.warn("Warning: 'partition' will ignore the 'mask' "
                       f"of the {self.__class__.__name__}.", stacklevel=2)
        return self._data.partition(*args, **kwargs)

    def argpartition(self, *args, **kwargs):
        """Indices that would partition the data; the mask is ignored (as numpy.ma)."""
        _warnings.warn("Warning: 'argpartition' will ignore the 'mask' "
                       f"of the {self.__class__.__name__}.", stacklevel=2)
        return self._data.argpartition(*args, **kwargs)

    def tobytes(self, fill_value=None, order="C"):
        """Bytes of the filled data (on the host)."""
        return _to_host(self.filled(fill_value)).tobytes(order=order)

    def tofile(self, fid, sep="", format="%s"):
        raise NotImplementedError("MaskedArray.tofile() not implemented yet.")

    def resize(self, newshape, refcheck=True, order=False):
        raise ValueError("A masked array does not own its data and therefore cannot be resized.\n"
                         "Use the numpy.ma.resize function instead.")

    def dumps(self):
        """Pickle of the array as bytes."""
        import pickle
        return pickle.dumps(self)

    def dump(self, file):
        """Pickle the array to ``file`` (path or binary file object)."""
        import pickle
        if hasattr(file, "write"):
            pickle.dump(self, file)
        else:
            with open(file, "wb") as fh:
                pickle.dump(self, fh)

    def transpose(self, *axes):
        return self._apply(lambda a: a.transpose(*axes))

    def swapaxes(self, axis1, axis2):
        return self._apply(lambda a: a.swapaxes(axis1, axis2))

    def repeat(self, repeats, axis=None):
        return self._apply(lambda a: a.repeat(repeats, axis), view=False)

    def tile(self, reps):
        """Tile the array (see `numpy.tile`)."""
        xp = self._xp
        return self._apply(lambda a: xp.tile(a, reps), view=False)

    def take(self, indices, axis=None, out=None, mode="raise"):
        """Take elements along an axis; masked indices give masked results.

        On cupy with ``mode="raise"`` the bounds check needs one host
        synchronisation (min/max of ``indices`` are read back, as cupy's own
        ``take`` only wraps); ``mode="clip"`` / ``"wrap"`` do not sync.
        """
        xp = self._xp
        _np.empty(1).take(0, mode=mode)  # validates ``mode`` like ndarray.take (ValueError)
        mi = getmask(indices)
        if mi is not nomask:
            indices = indices.filled(0)
        if not (isinstance(indices, _np.ndarray) or _is_cupy_array(indices) or hasattr(indices, "_data")):
            # python objects are cast like ``ndarray.take`` does (unsafe to intp): [] and 1.5 are valid
            indices = _np.asarray(indices).astype(_np.intp)
        indices = xp.asarray(_to_xp(getdata(indices), xp))
        if xp is _np:
            take = lambda a: a.take(indices, axis=axis, mode=mode)  # noqa: E731
        else:  # cupy's take only wraps: emulate 'clip' and 'raise'
            if axis is not None:
                axis = _np.lib.array_utils.normalize_axis_index(axis, self.ndim)
            n = self.size if axis is None else self.shape[axis]
            if mode == "clip":
                indices = xp.clip(indices, 0, n - 1)
            elif mode == "raise" and indices.size:
                lo, hi = (int(v) for v in xp.stack([indices.min(), indices.max()]).get())
                if lo < -n or hi >= n:
                    bad = hi if hi >= n else lo
                    raise IndexError(f"index {bad} is out of bounds for axis {axis or 0} with size {n}")
            take = lambda a: a.take(indices, axis=axis)  # noqa: E731
        d = take(self._data)
        m = nomask if self._mask is nomask else take(self._mask)
        if mi is not nomask:
            m = _mask_or(m, mi, xp)
        if out is not None:
            odata = out._data if isinstance(out, _XupyMaskedArray) else out
            if tuple(odata.shape) != tuple(d.shape):
                raise ValueError(f"output array does not match result of take: {odata.shape} vs {d.shape}")
            xp.copyto(odata, d, casting="same_kind")
            if isinstance(out, _XupyMaskedArray):
                out.__setmask__(m)
            return out
        if d.ndim == 0:
            return _maybe_scalar(d, m)
        return _wrap(d, m, sharedmask=m is not nomask)  # numpy.ma: fresh ``view(cls)``, fill_value/hardmask are not inherited

    def compressed(self):
        """Non-masked data as a 1-D plain array (device sync)."""
        d = self._data.ravel()
        if self._mask is not nomask:
            d = d[~self._mask.ravel()]
        return d

    def filled(self, fill_value=None):
        """Plain array with masked values replaced by ``fill_value``.

        Returns the data itself if there is no mask.
        """
        m = self._mask
        if m is nomask:
            return self._data
        if fill_value is None:
            fill_value = self.fill_value
        xp = self._xp
        fv = xp.asarray(_check_fill_value(fill_value, self.dtype))
        out = xp.array(self._data, copy=True)
        xp.copyto(out, fv, where=m)
        return out

    def fill(self, value):
        """Set every element of the data (the mask is unchanged)."""
        self._data.fill(value)

    def tolist(self, fill_value=None):
        """Nested Python list; masked values are ``fill_value`` (default None)."""
        m = self._mask
        if m is nomask:
            return self._data.tolist()
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        d, m = _to_host(self._data), _to_host(m)
        res = _np.array(d.ravel(), dtype=object)
        res[m.ravel()] = None
        return res.reshape(d.shape).tolist()

    def item(self, *args):
        """Python scalar of the underlying data at a flat or N-d index (the mask is ignored, as numpy.ma)."""
        d = self._data
        if not args:
            if self.size != 1:
                raise ValueError("can only convert an array of size 1 to a Python scalar")
            idx = (0,) * d.ndim
        elif len(args) == 1 and isinstance(args[0], tuple):
            idx = args[0]
        elif len(args) == 1:
            i, n = _operator.index(args[0]), self.size
            if not -n <= i < n:
                raise IndexError(f"index {i} is out of bounds for size {n}")
            idx = _np.unravel_index(i % n, d.shape)
        elif len(args) == d.ndim:
            idx = args
        else:
            raise ValueError("incorrect number of indices for array")
        return _host_scalar(d[idx]).item()

    def asmarray(self, **kwargs):
        """Host `numpy.ma.MaskedArray`, always independent of this array (a copy).

        Keyword arguments go to `numpy.ma.MaskedArray` (e.g. ``dtype``, ``copy``).
        """
        kwargs.setdefault("copy", self._xp is _np)  # device data is copied by the transfer
        kwargs.setdefault("fill_value", self._fill_value)
        kwargs.setdefault("hard_mask", self._hardmask)
        m = self._mask
        return _np.ma.MaskedArray(
            _to_host(self._data),
            mask=_np.ma.nomask if m is nomask else _to_host(m),
            **kwargs,
        )

    # ---- Python protocols ------------------------------------------------
    def __len__(self):
        return len(self._data)

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d array")
        if self.ndim > 1:
            return (self[i] for i in range(len(self)))
        d = _to_host(self._data)
        m = None if self._mask is nomask else _to_host(self._mask)
        if m is None:
            return iter(d)
        return (masked if mi else di for di, mi in zip(d, m))

    def __contains__(self, el):
        # ndarray.__contains__ reduces the *data* of the comparison (masks are not honoured)
        res = self == el
        return bool(_get_xp(res).any(getattr(res, "_data", res)))

    def _single(self):
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted to Python scalars")
        if self._mask is not nomask and self.size == 1 and bool(self._mask.reshape(-1)[0]):
            return masked
        return self.item()

    def __bool__(self):
        if self.size > 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )
        if self.size == 0:
            raise ValueError(
                "The truth value of an empty array is ambiguous. "
                "Use `array.size > 0` to check that an array is not empty."
            )
        return bool(_host_scalar(self._data.reshape(())))

    def __float__(self):
        v = self._single()
        if v is masked:
            _warnings.warn("Warning: converting a masked element to nan.", stacklevel=2)
            return _np.nan
        return float(v)

    def __int__(self):
        v = self._single()
        if v is masked:
            raise MaskError("Cannot convert masked element to a Python int.")
        return int(v)

    def __complex__(self):
        # as numpy.ma (ndarray.__complex__): 0-d only, and the data of a masked element is used
        if self.ndim != 0:
            raise TypeError("only 0-dimensional arrays can be converted to Python scalars")
        return complex(_host_scalar(self._data))

    def __index__(self):
        if self.ndim != 0 or self.dtype.kind not in "iu":
            raise TypeError("only integer scalar arrays can be converted to a scalar index")
        return int(_host_scalar(self._data))

    @property
    def flat(self):
        """Flat iterator (`next`, indexing and assignment) over the array."""
        return _MaskedIterator(self)

    @flat.setter
    def flat(self, value):
        self.ravel()[:] = value

    def __array__(self, dtype=None, *, copy=None):
        """The (unfilled) data as a host array, like ``np.asarray(np.ma_array)``."""
        d = self._data
        if self._xp is _np:
            return _np.array(d, dtype=dtype, copy=copy)
        if copy is False:
            raise ValueError(_NO_COPY_MSG)
        return _np.asarray(d.get(), dtype=dtype)

    def __getitem__(self, indx):
        xp = self._xp
        indx = _prep_index(indx, xp)
        d = self._data[indx]
        m = self._mask
        mout = m if m is nomask else m[indx]
        if getattr(d, "ndim", 0) == 0 and not _has_ellipsis(indx):
            return _maybe_scalar(xp.asarray(d), mout)
        out = _wrap(d, nomask, like=self, sharedmask=self._sharedmask)
        if mout is not nomask:
            out._mask = mout.reshape(d.shape)
            out._sharedmask = True
        return out

    def __setitem__(self, indx, value):
        xp = self._xp
        if isinstance(indx, str):
            raise IndexError(_FIELD_INDEX_MSG)  # no structured dtypes: same error as numpy on a plain array
        idx_is_ma = isinstance(indx, _XupyMaskedArray)
        indx = _prep_index(indx, xp)
        _data, _mask = self._data, self._mask

        with _np.errstate(over="ignore", invalid="ignore"):
            if value is masked or value is _np.ma.masked:
                if _mask is nomask:
                    _mask = self._mask = _zeros_mask(self.shape, xp)
                _mask[indx] = True
                return

            if isinstance(value, (_XupyMaskedArray, _np.ma.MaskedArray)):
                dval, mval = _unpack(value)
            else:
                dval, mval = value, nomask
            dval = _to_xp(dval, xp)
            if isinstance(dval, (list, tuple)) and xp is not _np:
                dval = xp.asarray(dval)
            if mval is not nomask:
                mval = _to_xp(mval, xp)

            if _mask is nomask or not self._hardmask:
                _data[indx] = dval
            if _mask is nomask:
                if mval is not nomask:
                    _mask = self._mask = _zeros_mask(self.shape, xp)
                    _mask[indx] = mval
            elif not self._hardmask:
                if not (idx_is_ma and not isinstance(value, _XupyMaskedArray)):
                    _mask[indx] = False if mval is nomask else mval
            elif getattr(indx, "dtype", None) == MaskType:
                _data[indx & ~_mask] = dval
            else:
                # hard mask: masked slots keep their data and stay masked
                mindx = xp.asarray(_mask[indx])
                if mval is not nomask:
                    mindx = xp.logical_or(mindx, mval)
                dindx = xp.array(_data[indx])
                xp.copyto(dindx, dval, casting="unsafe", where=~mindx)
                _data[indx] = dindx
                _mask[indx] = mindx

    # ---- copy / pickle -----------------------------------------------------
    def __reduce__(self):
        """Pickle through host arrays (restored on the GPU if it was there)."""
        m = self._mask
        return (
            _unpickle,
            (
                _to_host(self._data),
                None if m is nomask else _to_host(m),
                self._fill_value,
                self._hardmask,
                self._xp is not _np,
            ),
        )


def _unpickle(data, mask, fill_value, hardmask, gpu):
    """Rebuild a masked array; use cupy if the data was on a GPU and cupy works."""
    cp = _cupy_module() if gpu else None
    if cp is not None:
        try:
            data, mask = cp.asarray(data), (None if mask is None else cp.asarray(mask))
        except Exception:  # CUDA not usable here: stay on the host
            pass
    out = _wrap(data, nomask if mask is None else mask, hardmask=hardmask)
    out._fill_value = fill_value
    return out


def _flat_get(arr, indx):
    try:
        return arr.flat[indx]
    except (IndexError, TypeError):  # cupy's flatiter lacks fancy indexing
        return arr.ravel()[indx]


def _flat_set(arr, indx, value):
    try:
        arr.flat[indx] = value
    except (IndexError, TypeError):
        if not arr.flags.c_contiguous:
            raise NotImplementedError("Fancy flat assignment needs a C-contiguous array.")
        arr.reshape(-1)[indx] = value


class _MaskedIterator:
    """Flat iterator over a masked array (``a.flat``)."""

    def __init__(self, ma):
        self.ma = ma
        self._it = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self.ma.ravel())
        return next(self._it)

    def __len__(self):
        return self.ma.size

    def __getitem__(self, indx):
        ma = self.ma
        xp = ma._xp
        indx = _prep_index(indx, xp)
        d = xp.asarray(_flat_get(ma._data, indx))
        m = nomask if ma._mask is nomask else xp.asarray(_flat_get(ma._mask, indx))
        if d.ndim == 0:
            return _maybe_scalar(d, m)
        return _wrap(d, m, like=ma)

    def __setitem__(self, index, value):
        ma = self.ma
        xp = ma._xp
        index = _prep_index(index, xp)
        _flat_set(ma._data, index, _to_xp(getdata(value), xp))
        if ma._mask is not nomask:
            _flat_set(ma._mask, index, _to_xp(getmaskarray(value), xp))

    def __array__(self, dtype=None, copy=None):
        return self.ma.ravel().__array__(dtype, copy=copy)


MaskedArray = masked_array = _XupyMaskedArray


# ---------------------------------------------------------------------------
# module-level mask helpers
# ---------------------------------------------------------------------------
def getmask(a):
    """Mask of ``a`` (XuPy or numpy.ma array), or `nomask`."""
    if isinstance(a, _np.ma.MaskedArray):
        m = _np.ma.getmask(a)
        return nomask if m is _np.ma.nomask else m
    return getattr(a, "_mask", nomask)


def getmaskarray(arr):
    """Full bool mask of ``arr`` (zeros if it has no mask), on its device."""
    mask = getmask(arr)
    if mask is nomask:
        return _zeros_mask(_np.shape(arr), _get_xp(arr))
    return mask


def getdata(a, subok=True):
    """Data of ``a`` as a plain numpy/cupy array."""
    if isinstance(a, _XupyMaskedArray) or getattr(a, "_is_xupy_masked_constant", False):
        return a._data
    if isinstance(a, _np.ma.MaskedArray):
        return _np.ma.getdata(a)
    if _is_cupy_array(a):
        return a
    return _np.asarray(a)


def filled(a, fill_value=None):
    """Masked values of ``a`` replaced by ``fill_value`` (plain arrays pass through)."""
    if isinstance(a, _XupyMaskedArray):
        return a.filled(fill_value)
    if isinstance(a, _np.ma.MaskedArray):
        return a.filled(fill_value)
    return a if isinstance(a, _np.ndarray) or _is_cupy_array(a) else _np.array(a)


def is_masked(x):
    """True if ``x`` is a masked array with at least one masked value (device sync)."""
    m = getmask(x)
    return m is not nomask and bool(m.any())


def isMaskedArray(x):
    """True if ``x`` is a XuPy or numpy masked array."""
    return isinstance(x, (_XupyMaskedArray, _np.ma.MaskedArray)) or getattr(
        x, "_is_xupy_masked_constant", False
    )


isMA = isMaskedArray


def is_mask(m):
    """True if ``m`` has a boolean dtype (numpy/cupy array, numpy bool, `nomask`), as numpy.ma."""
    try:
        return m.dtype.type is MaskType
    except AttributeError:
        return False


def make_mask_none(newshape, dtype=None, xp=None):
    """All-False mask of ``newshape`` (on the active backend unless ``xp`` is given).

    As numpy's ``make_mask_descr``, a non-structured ``dtype`` still gives a bool mask; structured
    dtypes are not supported (``NotImplementedError``).
    """
    if dtype is not None and _np.dtype(dtype).names is not None:
        raise NotImplementedError("structured mask dtypes are not supported: only numeric and boolean data.")
    return _zeros_mask(newshape, xp or _backend.default_xp())


def make_mask(m, copy=False, shrink=True, dtype=MaskType):
    """Mask array from a mask-like; `nomask` if ``shrink`` and nothing is masked."""
    if m is nomask:
        return nomask
    xp = _get_xp(m)
    out = _backend.asarray(getdata(m), xp, dtype=dtype, copy=True if copy else None)
    if shrink and not bool(out.any()):
        return nomask
    return out


def mask_or(m1, m2, copy=False, shrink=True):
    """Combine two masks with logical or (`nomask` aware, as `numpy.ma.mask_or`)."""
    if m1 is nomask or m1 is False:
        return make_mask(m2, copy=copy, shrink=shrink)
    if m2 is nomask or m2 is False:
        return make_mask(m1, copy=copy, shrink=shrink)
    xp = _get_xp(m1, m2)
    if any(getdata(m).dtype.names is not None for m in (m1, m2)):
        raise NotImplementedError("structured masks are not supported: only numeric and boolean data.")
    out = xp.logical_or(_to_xp(m1, xp), _to_xp(m2, xp))
    return make_mask(out, copy=False, shrink=shrink)


def array(data, dtype=None, copy=False, order=None, mask=nomask, fill_value=None,
          keep_mask=True, hard_mask=False, shrink=True, subok=True, ndmin=0):
    """Shortcut to `MaskedArray` (same argument order as `numpy.ma.array`)."""
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy, subok=subok,
                       keep_mask=keep_mask, hard_mask=hard_mask, fill_value=fill_value,
                       ndmin=ndmin, shrink=shrink, order=order)


def expand_dims(a, axis):
    """Insert length-1 axes at ``axis`` (masked arrays stay masked, plain arrays stay plain)."""
    if isinstance(a, _XupyMaskedArray):
        return a.expand_dims(axis)
    if isinstance(a, _np.ma.MaskedArray):
        return _np.ma.expand_dims(a, axis)
    return _get_xp(a).expand_dims(a, axis)


# ---------------------------------------------------------------------------
# assemble the public namespace with the operator/reduction functions
# ---------------------------------------------------------------------------
from . import _ops, _reductions  # noqa: E402
from ._ops import *  # noqa: E402,F401,F403
from ._reductions import *  # noqa: E402,F401,F403

__all__ = [
    "MaskedArray",
    "masked_array",
    "nomask",
    "masked",
    "MaskType",
    "MAError",
    "MaskError",
    "masked_print_option",
    "getmask",
    "getmaskarray",
    "getdata",
    "filled",
    "is_masked",
    "isMaskedArray",
    "isMA",
    "is_mask",
    "make_mask",
    "make_mask_none",
    "mask_or",
    "default_fill_value",
    "set_fill_value",
    "array",
    "expand_dims",
]
__all__ += list(getattr(_ops, "__all__", [])) + list(getattr(_reductions, "__all__", []))
