"""
Tests for _CPUMemoryContext.

These tests do NOT require CuPy/CUDA and run purely on CPU (NumPy).
They validate that the CPU memory context manager exposes the same
interface as the GPU MemoryContext and behaves as a well-behaved
no-op context on CPU.
"""
import pytest
import numpy as np

import xupy as xp
from xupy._core import _CPUMemoryContext, on_gpu


class TestCPUMemoryContext:
    """Tests for _CPUMemoryContext (always available, no GPU required)."""

    def test_cpu_memory_context_is_importable(self):
        """_CPUMemoryContext can be imported from _core and is a class."""
        assert callable(_CPUMemoryContext)

    def test_cpu_memory_context_enter_returns_self(self):
        """__enter__ returns the context object itself."""
        ctx = _CPUMemoryContext()
        with ctx as c:
            assert c is ctx

    def test_cpu_memory_context_basic_usage(self, capsys):
        """Context manager completes without raising and prints a summary."""
        with _CPUMemoryContext() as ctx:
            arr = np.array([1, 2, 3])
            assert arr is not None
        captured = capsys.readouterr()
        assert "Session completed" in captured.out

    def test_cpu_memory_context_default_parameters(self):
        """Default parameters match the GPU MemoryContext API."""
        ctx = _CPUMemoryContext()
        assert ctx.device_id is None
        assert ctx.auto_cleanup is True
        assert ctx.memory_threshold == 0.9
        assert ctx.monitor_interval == 1.0

    def test_cpu_memory_context_custom_parameters(self):
        """Custom parameters are stored correctly."""
        ctx = _CPUMemoryContext(
            device_id=0,
            auto_cleanup=False,
            memory_threshold=0.8,
            monitor_interval=2.0,
        )
        assert ctx.device_id == 0
        assert ctx.auto_cleanup is False
        assert ctx.memory_threshold == 0.8
        assert ctx.monitor_interval == 2.0

    def test_cpu_memory_context_get_memory_info_returns_dict(self):
        """get_memory_info always returns a dict with at least 'device'."""
        ctx = _CPUMemoryContext()
        info = ctx.get_memory_info()
        assert isinstance(info, dict)
        assert "device" in info
        assert info["device"] == "cpu"

    def test_cpu_memory_context_get_memory_info_keys(self):
        """get_memory_info returns expected keys when psutil is available."""
        try:
            import psutil  # noqa: F401
            psutil_available = True
        except ImportError:
            psutil_available = False

        ctx = _CPUMemoryContext()
        info = ctx.get_memory_info()
        if psutil_available:
            for key in ("total", "free", "used", "memory_percent"):
                assert key in info, f"Missing key '{key}' in memory info"
            assert info["total"] > 0
            assert 0.0 <= info["memory_percent"] <= 1.0
        else:
            assert "error" in info

    def test_cpu_memory_context_no_op_methods(self):
        """All GPU-specific methods are no-ops (do not raise)."""
        ctx = _CPUMemoryContext()
        ctx.track_object(object())
        ctx.clear_cache()
        ctx.aggressive_cleanup()
        ctx.emergency_cleanup()
        ctx.auto_cleanup_if_needed()
        ctx.monitor_memory(duration=0.0)
        ctx.force_memory_deallocation()
        ctx.force_memory_pool_reset()

    def test_cpu_memory_context_check_memory_pressure(self):
        """check_memory_pressure returns a bool."""
        ctx = _CPUMemoryContext()
        result = ctx.check_memory_pressure()
        assert isinstance(result, bool)

    def test_cpu_memory_context_repr(self):
        """__repr__ contains 'MemoryContext' and 'cpu'."""
        ctx = _CPUMemoryContext()
        r = repr(ctx)
        assert "MemoryContext" in r
        assert "cpu" in r

    def test_cpu_memory_context_exception_propagation(self):
        """Exceptions raised inside the context propagate normally."""
        with pytest.raises(ValueError):
            with _CPUMemoryContext():
                raise ValueError("test error")

    def test_cpu_memory_context_exposed_when_on_cpu(self):
        """When running in CPU mode, xp.MemoryContext is _CPUMemoryContext."""
        if not on_gpu:
            assert hasattr(xp, "MemoryContext")
            assert xp.MemoryContext is _CPUMemoryContext


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
