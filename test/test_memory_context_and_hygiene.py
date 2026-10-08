"""
Tests for MemoryContext fixes (units, device restore, safe cleanup, no sleeps),
``array_size`` and the typing artefacts (py.typed, __init__.pyi).

GPU-only tests are skipped when no usable GPU is present and are tiny.
"""
import ast
import os
import time

import numpy as np
import pytest

import xupy as xp
from xupy import _core

GPU = _core._GPU_AVAILABLE
needs_gpu = pytest.mark.skipif(not GPU, reason="no usable GPU")
PKG = os.path.dirname(os.path.abspath(_core.__file__))


# ---------------------------------------------------------------------------
# array_size
# ---------------------------------------------------------------------------
class TestArraySize:
    def test_constants_are_binary(self):
        assert _core._B2mb_ == 1024**2
        assert _core._Btgb_ == 1024**3

    def test_no_per_dimension_truncation(self):
        # 3 * 300000 * 4 bytes; truncating each shape would give 0 for some
        shapes = [(300000,), (300000,), (300000,)]
        assert xp.array_size(shapes) == (3 * 300000 * 4) // 1024**2 == 3
        assert xp.array_size([(100000,)] * 30) == 11  # 12e6 B = 11.44 MiB

    def test_exact_bytes(self):
        assert xp.array_size((3,), out_unit="B") == 12
        assert xp.array_size((1024, 1024), dtype="float64", out_unit="B") == 8 * 1024**2

    def test_mib_gib_values(self):
        assert xp.array_size((1024, 1024, 4)) == 16
        assert xp.array_size((1024, 1024, 1024), dtype="float32", out_unit="GB") == 4
        assert xp.array_size((256, 1024, 1024), out_unit="gb") == 1

    def test_unit_aliases(self):
        assert xp.array_size((1024, 1024), out_unit="MiB") == 4
        assert xp.array_size((1024, 1024), out_unit="KB") == 4096

    @pytest.mark.parametrize("unit", ["TB", "", "mb ", "bytes", None])
    def test_unknown_unit_raises(self, unit):
        with pytest.raises(ValueError):
            xp.array_size((10,), out_unit=unit)

    def test_empty_shape_is_scalar(self):
        assert xp.array_size((), out_unit="B") == 4
        assert xp.array_size((), dtype="float64", out_unit="B") == 8

    def test_int_shape(self):
        assert xp.array_size(10, out_unit="B") == 40
        assert xp.array_size(np.int64(10), out_unit="B") == 40

    def test_numpy_int_entries(self):
        assert xp.array_size((np.int64(2), np.int32(3)), out_unit="B") == 24
        assert xp.array_size([(np.int64(2),), (3,)], out_unit="B") == 20

    def test_list_of_shapes_with_empty_shape(self):
        assert xp.array_size([(), (2,)], out_unit="B") == 12

    def test_empty_list_is_zero(self):
        assert xp.array_size([], out_unit="B") == 0

    def test_returns_python_int(self):
        assert type(xp.array_size((np.int64(10), 10))) is int

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError):
            xp.array_size((-1, 3))

    def test_float_entry_raises(self):
        with pytest.raises(TypeError):
            xp.array_size((2.5, 3))


# ---------------------------------------------------------------------------
# CPU memory context units
# ---------------------------------------------------------------------------
def test_cpu_info_is_mib():
    psutil = pytest.importorskip("psutil")
    info = _core._CPUMemoryContext().get_memory_info()
    assert info["total"] < psutil.virtual_memory().total / 1000  # not bytes


# ---------------------------------------------------------------------------
# GPU MemoryContext
# ---------------------------------------------------------------------------
@needs_gpu
class TestGPUMemoryContext:
    def test_info_in_mib(self):
        ctx = _core._MemoryContext(print_report=False)
        info = ctx.get_memory_info()
        free, total = _core._cupy.cuda.runtime.memGetInfo()
        assert info["total"] == pytest.approx(total / 1024**2, rel=1e-3)
        assert isinstance(info["used"], float)

    def test_delta_not_converted_twice(self, capsys):
        cp = _core._cupy
        cp.get_default_memory_pool().free_all_blocks()  # a pooled block would hide the allocation
        with _core._MemoryContext(auto_cleanup=False) as ctx:
            a = cp.ones(16 * 1024 * 1024, dtype=cp.float32)  # 64 MiB
            a.sum()
            cp.cuda.Device().synchronize()
        out = capsys.readouterr().out
        line = [l for l in out.splitlines() if "Memory delta" in l][0]
        assert "MiB" in line
        delta = float(line.split(":")[-1].split()[0])
        assert delta > 30.0  # reads ~64, not ~0.00

    def test_repr_units(self):
        r = repr(_core._MemoryContext())
        assert "MiB" in r and "MB)" not in r

    def test_no_sleep_on_exit(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("sleep called")

        monkeypatch.setattr(_core._time, "sleep", boom)
        with _core._MemoryContext(print_report=False):
            pass
        ctx = _core._MemoryContext(print_report=False)
        ctx.emergency_cleanup()

    def test_exit_is_fast(self):
        with _core._MemoryContext(print_report=False):
            pass  # warm-up
        t = time.perf_counter()
        for _ in range(5):
            with _core._MemoryContext(print_report=False):
                pass
        assert (time.perf_counter() - t) / 5 < 0.5  # no sleeps (gc.collect cost depends on the heap)

    @pytest.mark.skipif(_core._n_gpus < 2, reason="needs two GPUs to really switch device")
    @pytest.mark.parametrize("auto", [True, False])
    def test_device_restored(self, auto):
        cp = _core._cupy
        orig = cp.cuda.runtime.getDevice()
        target = (orig + 1) % _core._n_gpus
        with _core._MemoryContext(
            device_id=target, auto_cleanup=auto, print_report=False
        ) as ctx:
            assert cp.cuda.runtime.getDevice() == target
        assert cp.cuda.runtime.getDevice() == orig
        assert ctx._device_ctx is None

    @pytest.mark.skipif(_core._n_gpus < 2, reason="needs two GPUs to really switch device")
    def test_device_restored_on_error_without_cleanup(self):
        cp = _core._cupy
        orig = cp.cuda.runtime.getDevice()
        with pytest.raises(RuntimeError):
            with _core._MemoryContext(
                device_id=0, auto_cleanup=False, print_report=False
            ):
                cp.cuda.runtime.setDevice((orig + 1) % _core._n_gpus)
                raise RuntimeError("x")
        assert cp.cuda.runtime.getDevice() == orig

    def test_user_objects_not_mutated(self):
        class Holder:
            def __init__(self):
                self.data = _core._cupy.zeros(4)
                self.mask = _core._cupy.zeros(4, dtype=bool)

        h = Holder()
        d, m = h.data, h.mask
        with _core._MemoryContext(print_report=False) as ctx:
            ctx.track_object(h)
        assert h.data is d and h.mask is m
        assert float(h.data.sum()) == 0.0

    def test_tracking_does_not_keep_objects_alive(self):
        import weakref

        class Holder:
            data = _core._cupy.zeros(1)

        h = Holder()
        ref = weakref.ref(h)
        ctx = _core._MemoryContext(print_report=False)
        ctx.track_object(h)
        del h
        assert ref() is None

    def test_force_deallocation_threshold_is_100_mib(self):
        import inspect

        src = inspect.getsource(_core._MemoryContext.force_memory_deallocation)
        assert "100 * _B2mb_" in src and "1024**3" not in src
        assert "empty(" not in src  # no allocation pressure on the device

    def test_xp_exposes_gpu_context(self):
        if xp.on_gpu:
            assert xp.MemoryContext is _core._MemoryContext


# ---------------------------------------------------------------------------
# Typing artefacts
# ---------------------------------------------------------------------------
class TestTyping:
    def test_py_typed_exists(self):
        assert os.path.isfile(os.path.join(PKG, "py.typed"))

    def test_stub_parses_and_declares_api(self):
        with open(os.path.join(PKG, "__init__.pyi")) as f:
            tree = ast.parse(f.read())
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.AnnAssign):
                names.add(node.target.id)
            elif isinstance(node, ast.ImportFrom):
                names.update(a.asname or a.name for a in node.names)
        for n in ("use_cpu", "use_gpu", "backend", "on_gpu", "asnumpy", "asmarray",
                  "MemoryContext", "on_device", "set_device", "array_size", "ma",
                  "__version__"):
            assert n in names

    def test_stub_functions_exist_at_runtime(self):
        for n in ("use_cpu", "use_gpu", "backend", "asnumpy", "asmarray",
                  "MemoryContext", "on_device", "set_device", "array_size"):
            assert hasattr(xp, n)

    def test_array_alias_covers_ndarray(self):
        from xupy import typings

        assert typings.Array is not None

    def test_package_data_ships_typing_files(self):
        tomllib = pytest.importorskip("tomllib")
        root = os.path.dirname(PKG)
        with open(os.path.join(root, "pyproject.toml"), "rb") as f:
            cfg = tomllib.load(f)
        pd = cfg["tool"]["setuptools"]["package-data"]["xupy"]
        assert "py.typed" in pd and "*.pyi" in pd

    def test_typings_does_not_import_cupy_at_runtime(self):
        import xupy.typings as t

        assert not hasattr(t, "cupy")
