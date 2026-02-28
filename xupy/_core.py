"""
XUPY Core Module
================

This module is the core of XuPy, it contains the functions and classes that are used to create the XuPy library.

"""

import numpy as _np
import time as _time
import builtins as _b
import sys as _sys
from . import typings as _t
from ._cupy_install import __check_availability__ as __check__

_GPU = False
_GPU_AVAILABLE = False

__check__.xupy_init()
__cuda_version__ = __check__.get_cuda_version()

del __check__

try:
    import cupy as _xp # type: ignore

    _B2mb_ = 1024 * 1000  # using MB = 1,000,000 bytes
    n_gpus = _xp.cuda.runtime.getDeviceCount()
    if n_gpus > 1:
        gpus = {}
        line1 = """
[XuPy] Multiple GPUs detected:
"""
        for g in range(n_gpus):
            gpu = _xp.cuda.runtime.getDeviceProperties(g)
            gpu_name = gpu["name"].decode()
            gpus[g] = gpu_name
            line1 += f"       - gpu_id {g} : {gpu_name} | Memory = {gpu['totalGlobalMem'] / _B2mb_:.2f} MB | Compute Capability = {gpu['major']}.{gpu['minor']}\n"
    else:
        gpu = _xp.cuda.runtime.getDeviceProperties(0)
        gpu_name = gpu["name"].decode()
        line1 = f"[XuPy] Device {_xp.cuda.runtime.getDevice()} available - GPU : `{gpu_name}`\n"
        line1 += f"       Memory = {_xp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / _B2mb_:.2f} MB | Compute Capability = {_xp.cuda.runtime.getDeviceProperties(0)['major']}.{_xp.cuda.runtime.getDeviceProperties(0)['minor']}\n"
    print(
        f"""
{line1}       Using CuPy {_xp.__version__} for acceleration."""
    )

    # Test cupy is working on the system
    import gc

    a = _xp.array([1, 2, 3])  # test array
    del a  # cleanup
    gc.collect()
    _GPU = True
    _GPU_AVAILABLE = True
    from cupy import *  # type: ignore

except Exception as err:
    if not __cuda_version__ is None:
        print(
            f"""
[XuPy] GPU Acceleration unavailable.
       Using CPU (NumPy)."""
        )
    _GPU = False
    _GPU_AVAILABLE = False
    from numpy import *  # type: ignore

on_gpu = _GPU

# Capture the set of names brought in by the wildcard import
_backend_names = frozenset(
    getattr(_xp if _GPU_AVAILABLE else _np, '__all__',
            [n for n in dir(_xp if _GPU_AVAILABLE else _np) if not n.startswith('_')])
)

_mode_names: set[str] = set()


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
        if _GPU:
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
        if _GPU:
            return f"NumpyContext(original_device={self.original_device})"
        else:
            return "NumpyContext(no_gpu=True)"


# ---------------------------------------------------------------------------
# GPU-only definitions (only created when CuPy was successfully loaded)
# ---------------------------------------------------------------------------

if _GPU_AVAILABLE:

    def _set_device(device_id: int) -> None:
        """
        Sets the default CUDA device for computations (cupy).

        Parameters
        ----------
        device_id : int
            The ID of the CUDA device to set as default.

        Raises
        ------
        RuntimeError : If the device cannot be set or if the device is already the current device.

        Examples
        --------
        >>> xp.set_device(0)
        >>> xp.set_device(1)
        """
        import warnings

        if not _xp.cuda.runtime.getDevice() == device_id and n_gpus > 1:
            try:
                _xp.cuda.runtime.setDevice(device_id)
                print(f"[XuPy] Set device to {device_id} : {gpus[device_id]}")
            except Exception as e:
                raise RuntimeError(f"[XuPy] Failed to set device to {device_id} : {e}")
        elif _xp.cuda.runtime.getDevice() == device_id and n_gpus == 1:
            raise RuntimeError(f"[XuPy] Only one GPU available")
        else:
            warnings.warn(
                f"[XuPy] Device {device_id} is already the current device", UserWarning
            )

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
            self._start_time = _time.time()

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
            if not _GPU:
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
            if not _GPU:
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
                # print(
                #     f"[MemoryContext] Memory after forced deallocation: {free_after/(_B2mb_):.2f}/{total/(_B2mb_):.2f} MB"
                # )
                print(f"[MemoryContext] Memory freed: {freed/(_B2mb_):.2f} MB")

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
                return (
                    f"MemoryContext(device={self.device_id}, error={mem_info['error']})"
                )

            used_mb = mem_info.get("used", 0) / (_B2mb_)
            total_mb = mem_info.get("total", 0) / (_B2mb_)
            percent = mem_info.get("memory_percent", 0) * 100

            return f"MemoryContext(device={mem_info.get('device')}, memory={used_mb:.2f}/{total_mb:.2f} MB ({percent:.1f}%))"


# ---------------------------------------------------------------------------
# Mode-specific definition builders
# ---------------------------------------------------------------------------

def _gpu_definitions() -> dict:
    """Return dict of names exposed only in GPU mode."""
    def asmarray(array: _t.NDArray[_t.Any]) -> _t.MaskedArray:
        """
        Converts an object to a masked array on the GPU.

        Args:
            array: Input array-like object.

        Returns:
            cupy.ma.MaskedArray: A masked array on the GPU. If the input is already a masked array, it is returned as is.
        """
        try:
            return array.asmarray()
        except AttributeError:
            return _np.ma.masked_array(array.data, mask=array.mask)

    return {
        'float': _xp.float32,
        'double': _xp.float64,
        'cfloat': _xp.complex64,
        'cdouble': _xp.complex128,
        'np': _np,
        'npma': _np.ma,
        'asmarray': asmarray,
        'set_device': _set_device,
        'MemoryContext': _MemoryContext,
    }


def _cpu_definitions() -> dict:
    """Return dict of names exposed only in CPU mode."""
    def asnumpy(array: _t.NDArray[_t.Any]) -> _t.Array:
        """
        Placeholder function for asnumpy when GPU is not available.
        """
        if isinstance(array, _np.ma.MaskedArray):
            return array.data
        return array

    def asmarray(array: _t.NDArray[_t.Any]) -> _t.MaskedArray:
        """
        Placeholder function for asmarray when GPU is not available.
        """
        if isinstance(array, _np.ma.MaskedArray):
            return array
        return _np.ma.masked_array(array)

    return {
        'float': _np.float64,
        'double': _np.float64,
        'cfloat': _np.complex128,
        'cdouble': _np.complex128,
        'asnumpy': asnumpy,
        'asmarray': asmarray,
    }


# ---------------------------------------------------------------------------
# Internal helpers for namespace management
# ---------------------------------------------------------------------------

def _apply_mode_definitions(defs: dict) -> None:
    """Apply mode-specific names to _core and xupy namespaces."""
    global _mode_names
    _core_mod = _sys.modules[__name__]
    _pkg_mod = _sys.modules.get('xupy')

    for name in _mode_names:
        _core_mod.__dict__.pop(name, None)
        if _pkg_mod:
            _pkg_mod.__dict__.pop(name, None)

    for name, value in defs.items():
        _core_mod.__dict__[name] = value
        if _pkg_mod:
            _pkg_mod.__dict__[name] = value

    _mode_names = set(defs.keys())


def _repopulate_backend(to_gpu: bool) -> None:
    """Swap the array-library backend and mode-specific definitions."""
    global _GPU, on_gpu, _backend_names, _mode_names

    _core_mod = _sys.modules[__name__]
    _pkg_mod = _sys.modules.get('xupy')

    new_backend = _xp if to_gpu else _np
    new_backend_names = frozenset(
        getattr(new_backend, '__all__',
                [n for n in dir(new_backend) if not n.startswith('_')])
    )
    new_mode_defs = _gpu_definitions() if to_gpu else _cpu_definitions()

    # 1. Remove ALL old managed names (backend + mode) to start clean
    for name in (_backend_names | _mode_names):
        _core_mod.__dict__.pop(name, None)
        if _pkg_mod:
            _pkg_mod.__dict__.pop(name, None)

    # 2. Add new backend names
    for name in new_backend_names:
        obj = getattr(new_backend, name)
        _core_mod.__dict__[name] = obj
        if _pkg_mod:
            _pkg_mod.__dict__[name] = obj

    # 3. Layer mode-specific names on top (takes precedence over backend)
    for name, value in new_mode_defs.items():
        _core_mod.__dict__[name] = value
        if _pkg_mod:
            _pkg_mod.__dict__[name] = value

    _backend_names = new_backend_names
    _mode_names = set(new_mode_defs.keys())

    _GPU = to_gpu
    on_gpu = to_gpu
    _core_mod.__dict__['on_gpu'] = to_gpu
    if _pkg_mod:
        _pkg_mod.__dict__['on_gpu'] = to_gpu

        # Restore xupy subpackages (e.g. xupy.ma) that may have been
        # overwritten or removed by the backend wildcard swap.
        _pkg_prefix = 'xupy.'
        for _mod_name, _mod_obj in _sys.modules.items():
            if (_mod_name.startswith(_pkg_prefix)
                    and '.' not in _mod_name[len(_pkg_prefix):]):
                _pkg_mod.__dict__[_mod_name[len(_pkg_prefix):]] = _mod_obj

        # On CPU, replace xupy.ma with numpy.ma so masked arrays are native
        # numpy.ma.MaskedArray.  On GPU, the loop above already restored
        # XuPy's custom GPU-accelerated ma module.
        if not to_gpu:
            _pkg_mod.__dict__['ma'] = _np.ma


# Apply initial mode-specific definitions
if _GPU:
    _apply_mode_definitions(_gpu_definitions())
else:
    _apply_mode_definitions(_cpu_definitions())


# ---------------------------------------------------------------------------
# Public switching API
# ---------------------------------------------------------------------------

def use_cpu() -> None:
    """Switch XuPy to use NumPy (CPU) as the backend.

    After calling this function, all new array operations (e.g. ``xp.array``,
    ``xp.zeros``) will create NumPy arrays.  Existing arrays are **not**
    converted automatically.
    """
    if not _GPU:
        return
    _repopulate_backend(to_gpu=False)
    print("[XuPy] Switched to CPU (NumPy).")


def use_gpu() -> None:
    """Switch XuPy to use CuPy (GPU) as the backend.

    After calling this function, all new array operations (e.g. ``xp.array``,
    ``xp.zeros``) will create CuPy arrays.  Existing arrays are **not**
    converted automatically.

    Raises
    ------
    RuntimeError
        If CuPy is not available on this system.
    """
    if _GPU:
        return
    if not _GPU_AVAILABLE:
        raise RuntimeError(
            "[XuPy] CuPy is not available on this system. Cannot switch to GPU."
        )
    _repopulate_backend(to_gpu=True)
    print("[XuPy] Switched to GPU (CuPy).")
