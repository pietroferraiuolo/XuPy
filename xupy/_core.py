"""
XUPY Core Module
================

Backend manager of XuPy.  The public namespace (``xupy`` and ``xupy._core``)
is *not* stored in module globals: it is resolved on every attribute access
(PEP 562) against the table of the active backend, so that

* ``use_cpu()`` / ``use_gpu()`` only flip one flag (thread-safe, nothing is
  popped or re-added);
* ``with backend("cpu"):`` gives a thread- and asyncio-local override
  (``contextvars``);
* the namespace follows NumPy 2.x on both backends: it is built from
  ``numpy.__all__``, names removed in NumPy 2 never resolve, and GPU-mode
  gaps are filled by shims (see ``xupy._shims``).
"""

import numpy as _np
import os as _os
import shutil as _shutil
import time as _time
import builtins as _b
import sys as _sys
import logging as _logging
import threading as _threading
import warnings as _warnings
from contextvars import ContextVar as _ContextVar
from . import typings as _t
from contextlib import contextmanager as _contextmanager

__all__ = ["use_cpu", "use_gpu", "backend", "NumpyContext"]

_log = _logging.getLogger("xupy")

_B2mb_ = 1024 * 1000  # using MB = 1,000,000 bytes
_Btgb_ = 1024 * 1000 * 1000  # using GB = 1,000,000,000 bytes

_GPU_AVAILABLE = False
_MULTIGPU = False
_n_gpus = 0
_cuda_version = None
_xp = _np  # cupy when usable, numpy otherwise (used by the GPU MemoryContext)
_cupy = None  # cupy module when usable, else None


def _nvidia_gpu_present() -> bool:
    """Heuristic: True if an NVIDIA driver tool is on PATH (no subprocess is run)."""
    return _shutil.which("nvidia-smi") is not None


def _warn_gpu_unusable(reason: str, cupy_importable: bool = False) -> None:
    """Emit a single UserWarning when an NVIDIA GPU seems present but CuPy is unusable."""
    flag = _os.environ.get("XUPY_NO_GPU_WARNING", "").strip().lower()
    if flag not in ("", "0", "false", "no"):
        return
    if not (cupy_importable or _nvidia_gpu_present()):
        return
    # Point at the first frame outside the xupy package.
    pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
    level, frame = 1, _sys._getframe(0)
    while frame is not None:
        fname = _os.path.abspath(frame.f_code.co_filename)
        if "importlib" in fname and "_bootstrap" in fname:
            frame = frame.f_back  # the warnings module skips these frames itself
            continue
        if not fname.startswith(pkg_dir + _os.sep):
            break
        frame = frame.f_back
        level += 1
    _warnings.warn(
        f"[XuPy] NVIDIA GPU detected but CuPy is not usable ({reason}); using NumPy.\n"
        "Install with: pip install xupy[cuda12] / xupy[cuda13], or: "
        "python -m xupy.install_cupy. Silence with XUPY_NO_GPU_WARNING=1.",
        UserWarning,
        stacklevel=level,
    )




def _gpu_banner(cp, n: int) -> str:
    """Human-readable description of the detected GPU(s)."""
    if n > 1:
        lines = ["[XuPy] Multiple GPUs detected:"]
        for g in range(n):
            p = cp.cuda.runtime.getDeviceProperties(g)
            lines.append(
                f"       - gpu_id {g} : {p['name'].decode()} | Memory = {p['totalGlobalMem'] / _B2mb_:.2f} MB"
                f" | Compute Capability = {p['major']}.{p['minor']}"
            )
    else:
        p = cp.cuda.runtime.getDeviceProperties(0)
        lines = [
            f"[XuPy] Device {cp.cuda.runtime.getDevice()} available - GPU : `{p['name'].decode()}`",
            f"       Memory = {p['totalGlobalMem'] / _B2mb_:.2f} MB | Compute Capability = {p['major']}.{p['minor']}",
        ]
    lines.append(f"       Using CuPy {cp.__version__} for acceleration.")
    return "\n".join(lines)


_cupy_err = None
_cupy_importable = False
try:
    import cupy as _cp_mod  # type: ignore

    _cupy_importable = True
    # Prove that kernels compile and run, not just that memory can be allocated.
    if int((_cp_mod.arange(4) + 1).sum().item()) != 10:
        raise RuntimeError("CuPy kernel sanity check returned a wrong result")
    _cuda_version = (
        lambda v: f"{v // 1000}.{(v % 1000) // 10}"
    )(_cp_mod.cuda.runtime.runtimeGetVersion())
    _n_gpus = _cp_mod.cuda.runtime.getDeviceCount()
    _banner = _gpu_banner(_cp_mod, _n_gpus)
except Exception as err:  # any cupy failure means CPU fallback
    _cupy_err = err
    _cuda_version = None
    _n_gpus = 0

if _cupy_err is None:
    _cupy = _xp = _cp_mod
    _GPU_AVAILABLE = True
    _MULTIGPU = _n_gpus > 1
    _log.info(_banner)
    print(_banner)
    del _banner
else:
    _reason = (str(_cupy_err).strip().splitlines() or [""])[0]
    _reason = f"{type(_cupy_err).__name__}: {_reason}" if _reason else type(_cupy_err).__name__
    if len(_reason) > 120:
        _reason = _reason[:117] + "..."
    _warn_gpu_unusable(_reason, cupy_importable=_cupy_importable)

# ---------------------------------------------------------------------------
# Backend state
# ---------------------------------------------------------------------------
# `_global_gpu` is the process-wide default (mutated only under `_lock`);
# `_backend_var` is a per-context override (None / True / False).

_global_gpu: bool = _GPU_AVAILABLE
_lock = _threading.Lock()
_backend_var: _ContextVar = _ContextVar("xupy_backend", default=None)


def _active_gpu() -> bool:
    """True if the backend active in the current context is the GPU."""
    o = _backend_var.get()
    return _global_gpu if o is None else o


def _array_size(
    shape: tuple[int] | list[tuple[int]],
    dtype: _t.DTypeLike = _np.float32,
    out_unit: str = 'MB',
) -> int:
    """
    Computes the expected allocated size on GPU of an array with size `shape` 
    and data type `dtype`.

    Parameters
    ----------
    shape : tuple[int] | list[tuple[int]]
        The shape of the array. Can input multiple shapes as a list, and the
        result will be the total size of all arrays combined.
    dtype : DTypeLike, optional
        The data type of the array elements (default: float32).
    out_unit : str, optional
        The unit for the output size. Can be 'MB' or 'GB' (default: 'MB').

    Returns
    -------
    size : int
        The size of the array in the specified unit.

    Examples
    --------
    >>> import xupy as xp
    >>> arr = xp.array([1, 2, 3])
    >>> xp.array_size(arr)
    12  # 3 elements * 4 bytes per int32
    """
    norm = _B2mb_ if out_unit == 'MB' else _Btgb_
    if isinstance(shape, tuple):
        if isinstance(shape[0], int):
            shape = [shape]  # single shape case
    size = []
    for s in shape:
        itemsize = _np.dtype(dtype).itemsize
        num_elements = _np.prod(s)
        size_bytes = num_elements * itemsize
        size.append(int(size_bytes / norm))
    return int(_np.sum(size))

# --- NUMPY Context manager ---
class NumpyContext:
    """Context manager that provides direct access to NumPy functions.

    This context manager allows you to use NumPy functions directly while keeping
    XuPy functions unchanged. Inside the context, you get a `np` reference that
    points to NumPy functions.

    Example:
        import xupy as xp

        with xp.NumpyContext() as np:
            # np.array creates NumPy arrays
            numpy_arr = np.array([1, 2, 3])

            # xp.array creates CuPy arrays (when on_gpu=True) or NumPy arrays (when on_gpu=False)
            xupy_arr = xp.array([1, 2, 3])
    """

    def __init__(self):
        if _active_gpu():
            self.original_device = _xp.cuda.runtime.getDevice()
        else:
            self.original_device = None

    def __enter__(self):
        """Enter numpy context - return numpy module for direct access."""
        return _np

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit numpy context."""
        pass

    def __repr__(self) -> str:
        """String representation of the context manager."""
        if _active_gpu():

            return f"NumpyContext(original_device={self.original_device})"
        else:
            return "NumpyContext(no_gpu=True)"


# ---------------------------------------------------------------------------
# CPU Memory Context Manager (always available, no-op mock for CPU mode)
# ---------------------------------------------------------------------------

class _CPUMemoryContext:
    """CPU memory context manager — a no-op counterpart to the GPU _MemoryContext.

    Provides the same interface as the GPU ``MemoryContext`` so that code written
    against ``xp.MemoryContext`` runs transparently on CPU (NumPy) without any
    changes.  All GPU-specific operations (pool cleanup, device synchronisation,
    etc.) are silently skipped.

    Example
    -------
    >>> import xupy as xp          # running in CPU mode
    >>> with xp.MemoryContext() as ctx:
    ...     arr = xp.array([1, 2, 3])
    ...     print(ctx.get_memory_info())
    """

    def __init__(
        self,
        device_id: _t.Optional[int] = None,
        auto_cleanup: bool = True,
        force_cleanup: bool = False,
        print_report: bool = True,
        memory_threshold: float = 0.9,
        monitor_interval: float = 1.0,
    ):
        """
        Parameters
        ----------
        device_id : int, optional
            Ignored on CPU; present for API compatibility with the GPU version.
        auto_cleanup : bool, optional
            Kept for API compatibility; no cleanup is performed on CPU.
        memory_threshold : float, optional
            Kept for API compatibility; no threshold enforcement on CPU.
        monitor_interval : float, optional
            Kept for API compatibility; no monitoring is performed on CPU.
        """
        self.device_id = device_id
        self.auto_cleanup = auto_cleanup
        self.force_cleanup = force_cleanup
        self.print_report = print_report
        self.memory_threshold = memory_threshold
        self.monitor_interval = monitor_interval

        self._start_time: _t.Optional[float] = None

    def __enter__(self):
        """Enter the CPU memory context."""
        self._start_time = _time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the CPU memory context (no-op cleanup)."""
        if self._start_time is not None:
            duration = _time.time() - self._start_time
            if self.print_report:
                print(f"[MemoryContext] Session completed in {duration:.2f}s (CPU mode)")

    def track_object(self, obj):
        """No-op: object tracking is not needed on CPU."""
        pass

    def clear_cache(self):
        """No-op: no GPU memory pool to clear on CPU."""
        pass

    def aggressive_cleanup(self):
        """No-op: no GPU memory to aggressively free on CPU."""
        pass

    def emergency_cleanup(self):
        """No-op: no GPU memory emergency cleanup needed on CPU."""
        pass

    def get_memory_info(self) -> dict:
        """Return basic CPU/RAM memory information where available.

        Uses ``psutil`` when installed; otherwise returns a minimal dict.
        """
        info: dict = {"device": "cpu"}
        try:
            import psutil  # type: ignore
            vm = psutil.virtual_memory()
            info.update(
                {
                    "total": vm.total,
                    "free": vm.available,
                    "used": vm.used,
                    "memory_percent": vm.percent / 100.0,
                }
            )
        except ImportError:
            info["error"] = "psutil not installed; install it for detailed CPU memory info"
        return info

    def check_memory_pressure(self) -> bool:
        """Check if RAM usage is above the threshold (requires psutil)."""
        mem_info = self.get_memory_info()
        if "memory_percent" in mem_info:
            return mem_info["memory_percent"] > self.memory_threshold
        return False

    def auto_cleanup_if_needed(self):
        """No-op: no GPU pressure-based cleanup on CPU."""
        pass

    def monitor_memory(self, duration: float = 10.0):
        """No-op: memory monitoring is not performed in CPU mode."""
        pass

    def force_memory_deallocation(self):
        """No-op: forced GPU memory deallocation is not applicable on CPU."""
        pass

    def force_memory_pool_reset(self):
        """No-op: GPU memory pool reset is not applicable on CPU."""
        pass

    def __repr__(self) -> str:
        """String representation of the CPU memory context."""
        mem_info = self.get_memory_info()
        if "used" in mem_info:
            used_mb = mem_info["used"] / (1024 * 1000)
            total_mb = mem_info["total"] / (1024 * 1000)
            percent = mem_info["memory_percent"] * 100
            return f"MemoryContext(device=cpu, memory={used_mb:.2f}/{total_mb:.2f} MB ({percent:.1f}%))"
        return "MemoryContext(device=cpu)"


# ---------------------------------------------------------------------------
# GPU-only definitions (only created when CuPy was successfully loaded)
# ---------------------------------------------------------------------------


if _GPU_AVAILABLE:

    # --- GPU Memory Management Context Manager ---
    class _MemoryContext:
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

        def __init__(
            self,
            device_id: _t.Optional[int] = None,
            auto_cleanup: bool = True,
            force_cleanup: bool = False,
            print_report: bool = True,
            memory_threshold: float = 0.9,
            monitor_interval: float = 1.0,
        ):
            """
            Initialize the memory context manager.

            Parameters
            ----------
            device_id : int, optional
                GPU device ID to manage. If None, uses current device.
            auto_cleanup : bool, optional
                Whether to automatically cleanup memory on exit (default: True).
            force_cleanup : bool, optional
                Whether to force memory cleanup on exit (default: False).
            print_report : bool, optional
                Whether to print memory usage report on exit (default: True).
            memory_threshold : float, optional
                Memory usage threshold (0-1) for automatic cleanup (default: 0.9).
            monitor_interval : float, optional
                Interval in milliseconds for memory monitoring (default: 1.0).
            """
            self.device_id = device_id
            self.auto_cleanup = auto_cleanup
            self.force_cleanup = force_cleanup
            self.memory_threshold = memory_threshold
            self.monitor_interval = monitor_interval/1000
            self._print_report = print_report

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
            self._start_time = _time.time()

            if _GPU_AVAILABLE:
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
                if self.auto_cleanup or self.force_cleanup:
                    self.aggressive_cleanup()

                    # Cleanup tracked GPU objects
                    self._cleanup_gpu_objects()

                    # Restore original device
                    if _GPU_AVAILABLE and self._device_ctx is not None:
                        try:
                            self._device_ctx.__exit__(exc_type, exc_val, exc_tb)
                        except Exception as e:
                            print(f"Warning: Error restoring device context: {e}")

                # Final memory report
                if self._print_report:
                    if self._start_time:
                        duration = _time.time() - self._start_time
                        final_mem = self.get_memory_info()
                        if "used" in final_mem:
                            memory_delta = final_mem["used"] - self._initial_memory
                            print(f"[MemoryContext] Session completed in {duration:.2f}s")
                            print(
                                f"[MemoryContext] Memory delta: {memory_delta / (_B2mb_):.2f} MB"
                            )
                            if self._cleanup_count > 0:
                                print(
                                    f"[MemoryContext] Cleanup operations: {self._cleanup_count}"
                                )

            except Exception as e:
                print(f"Warning: Error during memory context cleanup: {e}")

        def track_object(self, obj):
            """Track a GPU object for cleanup."""
            if hasattr(obj, "data") and hasattr(obj.data, "device"):
                self._gpu_objects.append(obj)

        def _cleanup_gpu_objects(self):
            """Clean up tracked GPU objects."""
            for obj in self._gpu_objects:
                try:
                    # Clear references to GPU data
                    if hasattr(obj, "data"):
                        obj.data = None
                    if hasattr(obj, "mask"):
                        obj.mask = None
                except Exception:
                    pass
            self._gpu_objects.clear()

        def clear_cache(self):
            """Clear GPU memory pools (safely)."""
            if not _GPU_AVAILABLE:
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
            if not _GPU_AVAILABLE:
                return

            if self._print_report:
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
                _time.sleep(0.01)

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
                if hasattr(mempool, "shrink"):
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
                # free, total = _xp.cuda.runtime.memGetInfo()
                # print(
                #     f"[MemoryContext] CUDA memory after cleanup: {free/(_B2mb_):.2f}/{total/(_B2mb_):.2f} MB"
                # )
            except Exception as e:
                print(f"Warning: Could not get CUDA memory info: {e}")

            if self.force_cleanup:
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
            if not _GPU_AVAILABLE:
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
                _time.sleep(0.05)

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
                if hasattr(mempool, "shrink"):
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
            if not _GPU_AVAILABLE:
                return {"error": "No GPU available"}

            try:
                # Get current device
                device_to_query = (
                    self.device_id
                    if self.device_id is not None
                    else _xp.cuda.runtime.getDevice()
                )

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
                    "total": int(total / _B2mb_),
                    "free": int(free / _B2mb_),
                    "used": int(used / _B2mb_),
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
                self._memory_history.append(
                    {"timestamp": _time.time(), "used": used, "free": free}
                )

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
                    print(
                        f"[MemoryContext] Memory pressure detected: {mem_info['memory_percent']*100:.1f}% > {self.memory_threshold*100:.1f}%"
                    )
                return pressure
            return False

        def auto_cleanup_if_needed(self):
            """Automatically cleanup if memory pressure is high."""
            if self.check_memory_pressure():
                print(
                    f"[MemoryContext] Memory usage above {self.memory_threshold*100:.1f}%, triggering cleanup"
                )
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
                    min_used = _b.min(used_values)
                    max_used = _b.max(used_values)
                    avg_used = _b.sum(used_values) / len(used_values)

                    print(f"[MemoryContext] Monitoring summary:")
                    print(f"  Min: {min_used / (_B2mb_):.2f} MB")
                    print(f"  Max: {max_used / (_B2mb_):.2f} MB")
                    print(f"  Avg: {avg_used / (_B2mb_):.2f} MB")

        def force_memory_deallocation(self):
            """Force memory deallocation by creating pressure on the memory pool."""
            if not _GPU_AVAILABLE:
                return

            print("[MemoryContext] Forcing memory deallocation...")
            try:
                # Get current memory info
                free_before, total = _xp.cuda.runtime.memGetInfo()
                # print(
                #     f"[MemoryContext] Memory before forced deallocation: {free_before/(_B2mb_):.2f}/{total/(_B2mb_):.2f} MB"
                # )

                # Try to allocate a large chunk to force pool cleanup
                # This will fail if there's not enough memory, but that's okay
                try:
                    # Allocate 90% of available memory temporarily
                    alloc_size = int(free_before * 0.9)
                    if alloc_size > 100 * (
                        1024**3
                    ):  # Only if we have more than 100MB to work with
                        temp_array = _xp.empty(
                            (alloc_size // 4,), dtype=_xp.float32
                        )  # 4 bytes per float32
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
                if self._print_report:
                    print(f"[MemoryContext] Memory freed: {freed/(_B2mb_):.2f} MB")

            except Exception as e:
                print(f"Warning: Could not force memory deallocation: {e}")

        def force_memory_pool_reset(self):
            """Force a complete memory pool reset by creating a new pool."""
            if not _GPU_AVAILABLE:
                return

            if self._print_report:
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

                if self._print_report:
                    print("[MemoryContext] Memory pool reset completed")

            except Exception as e:
                print(f"Warning: Could not reset memory pool: {e}")
                # Fallback to aggressive cleanup
                self.aggressive_cleanup()

        def __repr__(self) -> str:
            """String representation with memory info."""
            mem_info = self.get_memory_info()
            if "error" in mem_info:
                return (
                    f"MemoryContext(device={self.device_id}, error={mem_info['error']})"
                )

            used_mb = mem_info.get("used", 0) / (_B2mb_)
            total_mb = mem_info.get("total", 0) / (_B2mb_)
            percent = mem_info.get("memory_percent", 0) * 100

            return f"MemoryContext(device={mem_info.get('device')}, memory={used_mb:.2f}/{total_mb:.2f} MB ({percent:.1f}%))"




# ---------------------------------------------------------------------------
# Public switching API
# ---------------------------------------------------------------------------

def use_cpu() -> None:
    """Make NumPy (CPU) the default backend, process-wide.

    Thread-safe and idempotent.  Existing arrays are **not** converted.  Use
    ``with backend("cpu"):`` for a scoped, thread/async-local switch instead.
    """
    global _global_gpu
    with _lock:
        changed, _global_gpu = _global_gpu, False
    if changed:
        _log.info("[XuPy] Switched to CPU (NumPy).")
        print("[XuPy] Switched to CPU (NumPy).")


def use_gpu() -> None:
    """Make CuPy (GPU) the default backend, process-wide.

    Thread-safe and idempotent.  Existing arrays are **not** converted.

    Raises
    ------
    RuntimeError
        If CuPy is not available on this system.
    """
    global _global_gpu
    if not _GPU_AVAILABLE:
        raise RuntimeError(
            "[XuPy] CuPy is not available on this system. Cannot switch to GPU."
        )
    with _lock:
        changed, _global_gpu = not _global_gpu, True
    if changed:
        _log.info("[XuPy] Switched to GPU (CuPy).")
        print("[XuPy] Switched to GPU (CuPy).")


@_contextmanager
def backend(name: str):
    """Scoped backend selection, local to the current thread / asyncio task.

    Parameters
    ----------
    name : {"cpu", "numpy", "gpu", "cupy"}
        Case-insensitive backend name.

    Yields
    ------
    module
        ``numpy`` or ``cupy``.

    Raises
    ------
    ValueError
        For an unknown name.
    RuntimeError
        For ``"gpu"``/``"cupy"`` when CuPy is not usable.

    Examples
    --------
    >>> with xp.backend("cpu"):
    ...     a = xp.zeros(3)   # numpy array, whatever the global default is
    """
    key = name.lower() if isinstance(name, str) else name
    if key in ("cpu", "numpy"):
        gpu = False
    elif key in ("gpu", "cupy"):
        gpu = True
        if not _GPU_AVAILABLE:
            raise RuntimeError("[XuPy] CuPy is not available on this system.")
    else:
        raise ValueError(
            f"Unknown backend {name!r}; expected 'cpu', 'numpy', 'gpu' or 'cupy'."
        )
    token = _backend_var.set(gpu)
    try:
        yield _cupy if gpu else _np
    finally:
        _backend_var.reset(token)


# ---------------------------------------------------------------------------
# Device helpers (same call signatures on both backends)
# ---------------------------------------------------------------------------

@_contextmanager
def _on_device(device_id: int):
    """
    Context manager to temporarily select a compute device.

    Parameters
    ----------
    device_id : int
        GPU mode: ``-1`` runs the block on the CPU (scoped backend override,
        the global backend is untouched); ``0 <= id < n_gpus`` makes that CUDA
        device current inside the block (also valid on a single-GPU machine).
        CPU mode: a no-op for any value, for code portability.

    Raises
    ------
    ValueError
        In GPU mode, if ``device_id`` is not -1 or a valid device index.

    Examples
    --------
    >>> with xp.on_device(0):
    ...     a = xp.array([1, 2, 3])   # allocated on GPU 0
    >>> with xp.on_device(-1):
    ...     b = xp.array([4, 5, 6])   # NumPy array
    """
    if not _active_gpu():
        yield
        return
    if device_id == -1:
        with backend("cpu"):
            yield
        return
    if not (_as_index(device_id) is not None and 0 <= device_id < _n_gpus):
        raise ValueError(
            f"[XuPy] Invalid device id {device_id!r}: expected -1 (CPU) or 0..{_n_gpus - 1}."
        )
    with _cupy.cuda.Device(device_id):
        yield


def _as_index(value):
    """Return ``operator.index(value)`` (accepts numpy integers), or None."""
    import operator

    try:
        return operator.index(value)
    except TypeError:
        return None


def _set_device(device_id: int) -> None:
    """
    Set the current CUDA device (GPU mode); a no-op on the CPU backend.

    Setting the device that is already current is a silent no-op.

    Raises
    ------
    ValueError
        In GPU mode, if ``device_id`` is not a valid device index.
    """
    if not _active_gpu():
        _log.debug("[XuPy] set_device(%r) ignored: CPU backend.", device_id)
        return
    if not (_as_index(device_id) is not None and 0 <= device_id < _n_gpus):
        raise ValueError(
            f"[XuPy] Invalid device id {device_id!r}: expected 0..{_n_gpus - 1}."
        )
    if int(_cupy.cuda.runtime.getDevice()) == device_id:
        return
    _cupy.cuda.runtime.setDevice(device_id)
    _log.info("[XuPy] Set device to %d", device_id)


def _asnumpy_gpu(array):
    if getattr(array, "_is_xupy_masked_constant", False):
        return _np.ma.masked
    if getattr(array, "_is_xupy_masked", False):
        array = array._data
    return _cupy.asnumpy(array)


def _asnumpy_cpu(array: _t.NDArray[_t.Any]) -> _t.Array:
    """Identity for NumPy arrays; the (host) data of a masked array."""
    if getattr(array, "_is_xupy_masked_constant", False):
        return _np.ma.masked
    if getattr(array, "_is_xupy_masked", False):
        array = array._data
        return array if isinstance(array, _np.ndarray) else array.get()
    if isinstance(array, _np.ma.MaskedArray):
        return array.data
    return array


def _asmarray_gpu(array: _t.NDArray[_t.Any]) -> _t.MaskedArray:
    """
    Converts an object to a (host) numpy masked array.

    Args:
        array: Input array-like object (e.g. an XuPy masked array).
    """
    if getattr(array, "_is_xupy_masked_constant", False):
        return _np.ma.masked
    try:
        return array.asmarray()
    except AttributeError:
        return _np.ma.masked_array(array.data, mask=array.mask)


def _asmarray_cpu(array: _t.NDArray[_t.Any]) -> _t.MaskedArray:
    """Return a numpy masked array (unchanged if it already is one)."""
    if isinstance(array, _np.ma.MaskedArray):
        return array
    if getattr(array, "_is_xupy_masked", False) or getattr(
        array, "_is_xupy_masked_constant", False
    ):
        return array.asmarray()
    return _np.ma.masked_array(array)


# ---------------------------------------------------------------------------
# Namespace tables
# ---------------------------------------------------------------------------

# Names in numpy.__all__ that XuPy does not forward: dunders (except
# __array_namespace_info__) and host/tooling submodules.  `ma` is dynamic.
# `ctypeslib` and `lib` stay on the CPU table (numpy parity) but are
# "unsupported on GPU": they are host-only helpers (ctypes, I/O, stride tricks)
# that must not silently mix with device arrays.
_EXCLUDE = {
    n for n in _np.__all__ if n.startswith("__") and n != "__array_namespace_info__"
} | {"core", "f2py", "test", "testing", "typing", "show_config", "show_runtime", "ma"}

# Removed in NumPy 2.0 (or earlier): never resolve on either backend.  Values
# are hints used in the AttributeError message.
_NUMPY2_REMOVED = {
    "NaN": "nan", "NAN": "nan", "Inf": "inf", "Infinity": "inf", "infty": "inf",
    "PINF": "inf", "NINF": "-inf", "PZERO": "0.0", "NZERO": "-0.0",
    "float_": "float64", "complex_": "complex128", "cfloat": "complex128",
    "singlecomplex": "complex64", "longfloat": "longdouble",
    "longcomplex": "clongdouble", "clongfloat": "clongdouble",
    "unicode_": "str_", "string_": "bytes_",
    "float": "float64 (or the builtin 'float')", "int": "int_ (or the builtin 'int')",
    "complex": "complex128 (or the builtin 'complex')",
    "object": "object_ (or the builtin 'object')", "str": "str_ (or the builtin 'str')",
    "unicode": "str_",
    "in1d": "isin", "trapz": "trapezoid", "row_stack": "vstack", "product": "prod",
    "cumproduct": "cumprod", "alltrue": "all", "sometrue": "any", "round_": "round",
    "msort": "sort(a, axis=0)", "asfarray": "asarray(a, dtype=float64)",
    "asscalar": "a.item()", "mat": "asmatrix", "cast": "asarray(x, dtype)",
    "find_common_type": "result_type or promote_types",
    "issubclass_": "issubclass", "issctype": "issubdtype", "issubsctype": "issubdtype",
    "obj2sctype": "dtype(x).type", "sctype2char": "dtype(x).char",
    "sctypes": "numpy.dtypes / issubdtype", "maximum_sctype": "dtype or finfo/iinfo",
    "set_string_function": "set_printoptions", "byte_bounds": "lib.array_utils.byte_bounds",
    "disp": "print", "who": "dir()", "safe_eval": "ast.literal_eval",
    "format_parser": "rec.format_parser", "lookfor": "numpy.info / help",
    "source": "inspect.getsource", "deprecate": "warnings.warn",
    "add_newdoc": "(removed)", "compat": "(removed)", "nbytes": "dtype(x).itemsize",
    "recfromcsv": "genfromtxt", "recfromtxt": "genfromtxt",
    "set_numeric_ops": "(removed)", "fastCopyAndTranspose": "a.T.copy()",
    "geterrobj": "geterr / errstate", "seterrobj": "seterr / errstate",
    "tracemalloc_domain": "lib.tracemalloc_domain",
    "AxisError": "exceptions.AxisError",
    "ComplexWarning": "exceptions.ComplexWarning",
    "VisibleDeprecationWarning": "exceptions.VisibleDeprecationWarning",
    "ModuleDeprecationWarning": "exceptions.ModuleDeprecationWarning",
    "RankWarning": "exceptions.RankWarning", "TooHardError": "exceptions.TooHardError",
    "DataSource": "lib.npyio.DataSource", "annotations": "(not a NumPy name)",
}

#: Canonical NumPy 2.x public names (what both backends expose).
_NUMPY_PUBLIC = frozenset(set(_np.__all__) - _EXCLUDE - set(_NUMPY2_REMOVED))

#: CuPy-only tools exposed in GPU mode.
_CUPY_ONLY = (
    "asnumpy", "get_array_module", "cuda", "fuse", "ElementwiseKernel", "RawKernel",
    "RawModule", "ReductionKernel", "get_default_memory_pool",
    "get_default_pinned_memory_pool", "is_available", "clear_memo", "memoize",
)

_MISSING = object()
_tab_cpu = None
_tab_gpu = None
_unsupported_gpu = frozenset()
_table_lock = _threading.Lock()


def _extras(gpu: bool) -> dict:
    """XuPy's own names, identical in both modes (implementations may differ)."""
    return {
        "asnumpy": _asnumpy_gpu if gpu else _asnumpy_cpu,
        "asmarray": _asmarray_gpu if gpu else _asmarray_cpu,
        "MemoryContext": _MemoryContext if gpu else _CPUMemoryContext,
        "NumpyContext": NumpyContext,
        "on_device": _on_device,
        "set_device": _set_device,
        "array_size": _array_size,
        "np": _np,
        "npma": _np.ma,
        "use_cpu": use_cpu,
        "use_gpu": use_gpu,
        "backend": backend,
        "has_multi_gpu": _MULTIGPU,
        "n_gpus": _n_gpus,
        "__cuda_version__": _cuda_version,
    }


def _build_cpu_table() -> dict:
    t = {}
    for n in _NUMPY_PUBLIC:
        v = getattr(_np, n, _MISSING)
        if v is not _MISSING:
            t[n] = v
    t["__array_api_version__"] = getattr(_np, "__array_api_version__", None)
    t.update(_extras(False))
    return t


def _build_gpu_table() -> dict:
    global _unsupported_gpu
    import cupyx  # type: ignore
    from . import _shims

    cp = _cupy
    fill, override = _shims.build(cp, cupyx)
    t, unsupported = {}, set()
    for n in _NUMPY_PUBLIC:
        v = getattr(cp, n, _MISSING)
        if v is _MISSING:
            v = fill.get(n, _MISSING)
        if v is _MISSING:
            unsupported.add(n)
        else:
            t[n] = v
    for n in _CUPY_ONLY:
        v = getattr(cp, n, _MISSING)
        if v is not _MISSING:
            t[n] = v
    t["__array_api_version__"] = fill["__array_api_version__"]
    t.update((n, v) for n, v in override.items() if n in _NUMPY_PUBLIC)
    t.update(_extras(True))
    _unsupported_gpu = frozenset(unsupported)
    return t


def _get_table(gpu: bool) -> dict:
    global _tab_cpu, _tab_gpu
    with _table_lock:
        if gpu:
            if _tab_gpu is None:
                try:
                    _tab_gpu = _build_gpu_table()
                except Exception as err:
                    raise RuntimeError(
                        f"[XuPy] Failed to build the GPU namespace: {err!r}"
                    ) from err
            return _tab_gpu
        if _tab_cpu is None:
            _tab_cpu = _build_cpu_table()
        return _tab_cpu


def _dynamic(name: str, gpu: bool):
    """Names computed at each access (never stored in the tables)."""
    if name == "on_gpu":
        return gpu
    if name == "ma":
        if gpu:
            import importlib

            return importlib.import_module("xupy.ma")
        return _np.ma
    return _MISSING


def __getattr__(name: str):
    gpu = _backend_var.get()
    if gpu is None:
        gpu = _global_gpu
    table = _tab_gpu if gpu else _tab_cpu
    if table is None:
        table = _get_table(gpu)
    try:
        return table[name]
    except KeyError:
        pass
    v = _dynamic(name, gpu)
    if v is not _MISSING:
        return v
    msg = f"xupy has no attribute {name!r}"
    if name in _NUMPY2_REMOVED:
        msg += f": removed in NumPy 2.0, use {_NUMPY2_REMOVED[name]!r}"
    elif gpu and name in _unsupported_gpu:
        msg += (
            f": numpy.{name} has no CuPy equivalent; "
            "use xp.backend('cpu') / xp.asnumpy()"
        )
    raise AttributeError(msg)


def _public_names() -> list:
    """Names exported by ``from xupy import *`` for the active backend."""
    table = _get_table(_active_gpu())
    return sorted({n for n in table if not n.startswith("_")} | {"ma", "on_gpu"})


def __dir__():
    names = {n for n in globals() if not n.startswith("_")}
    names.update(_get_table(_active_gpu()))
    names.update(("ma", "on_gpu"))
    return sorted(names)
