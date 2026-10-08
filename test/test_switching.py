"""Tests for backend switching: use_cpu/use_gpu, backend(), on_device, set_device, logging."""
import asyncio
import logging
import threading
import time
import warnings

import numpy as np
import pytest

import xupy as xp
from xupy import _core

GPU_OK = bool(_core._GPU_AVAILABLE)
requires_gpu = pytest.mark.skipif(not GPU_OK, reason="CuPy is not usable")
requires_nogpu = pytest.mark.skipif(GPU_OK, reason="CuPy is usable")


@pytest.fixture(autouse=True)
def _restore_backend():
    orig = _core._global_gpu
    yield
    _core._global_gpu = orig  # direct, so it works even if use_gpu() were broken
    assert xp.on_gpu is orig


def _cupy():
    import cupy

    return cupy


class TestUseCpuUseGpu:
    def test_use_cpu_idempotent(self):
        xp.use_cpu()
        xp.use_cpu()
        assert xp.on_gpu is False
        assert xp.zeros(2).__class__ is np.ndarray

    @requires_gpu
    def test_use_gpu_idempotent(self):
        xp.use_gpu()
        xp.use_gpu()
        assert xp.on_gpu is True
        assert isinstance(xp.zeros(2), _cupy().ndarray)

    @requires_gpu
    def test_round_trip_arrays(self):
        for _ in range(3):
            xp.use_cpu()
            assert type(xp.zeros(2)) is np.ndarray
            xp.use_gpu()
            assert isinstance(xp.zeros(2), _cupy().ndarray)

    @requires_nogpu
    def test_use_gpu_raises_without_cupy(self):
        with pytest.raises(RuntimeError, match="CuPy"):
            xp.use_gpu()
        assert xp.on_gpu is False

    @requires_nogpu
    def test_backend_gpu_raises_without_cupy(self):
        for n in ("gpu", "cupy", "GPU"):
            with pytest.raises(RuntimeError):
                with xp.backend(n):
                    pytest.fail("block must not run")
        assert xp.on_gpu is False

    @requires_nogpu
    def test_cpu_only_on_device_and_set_device_noop(self):
        with xp.on_device(5):
            assert type(xp.zeros(1)) is np.ndarray
        xp.set_device(99)
        xp.set_device(-3)

    def test_on_gpu_is_live_not_snapshot(self):
        if not GPU_OK:
            pytest.skip("CuPy is not usable")
        xp.use_cpu()
        assert xp.on_gpu is False
        xp.use_gpu()
        assert xp.on_gpu is True

    def test_global_switch_does_not_override_scoped(self):
        if not GPU_OK:
            pytest.skip("CuPy is not usable")
        xp.use_gpu()
        with xp.backend("cpu"):
            xp.use_gpu()
            assert xp.on_gpu is False
            xp.use_cpu()
        assert xp.on_gpu is False
        xp.use_gpu()
        assert xp.on_gpu is True


class TestBackendContext:
    def test_bad_name(self):
        for bad in ("tpu", "", "cpuu"):
            with pytest.raises(ValueError):
                with xp.backend(bad):
                    pass

    def test_non_string_name(self):
        for bad in (None, 3, ["cpu"]):
            with pytest.raises((ValueError, TypeError, AttributeError)):
                with xp.backend(bad):
                    pass

    def test_bad_name_does_not_change_state(self):
        before = xp.on_gpu
        with pytest.raises(ValueError):
            with xp.backend("nope"):
                pass
        assert xp.on_gpu is before

    @pytest.mark.parametrize("name", ["cpu", "numpy", "CPU", "NumPy"])
    def test_cpu_names_yield_numpy(self, name):
        with xp.backend(name) as m:
            assert m is np
            assert xp.on_gpu is False
            assert xp.zeros is np.zeros

    @requires_gpu
    @pytest.mark.parametrize("name", ["gpu", "cupy", "GPU", "CuPy"])
    def test_gpu_names_yield_cupy(self, name):
        with xp.backend(name) as m:
            assert m is _cupy()
            assert xp.on_gpu is True

    @requires_gpu
    def test_nesting(self):
        xp.use_gpu()
        with xp.backend("gpu"):
            assert xp.on_gpu
            with xp.backend("cpu"):
                assert not xp.on_gpu
                with xp.backend("gpu"):
                    assert xp.on_gpu
                    assert isinstance(xp.zeros(1), _cupy().ndarray)
                assert not xp.on_gpu
            assert xp.on_gpu
        assert xp.on_gpu

    @requires_gpu
    def test_nesting_from_cpu_global(self):
        xp.use_cpu()
        with xp.backend("gpu"):
            with xp.backend("cpu"):
                with xp.backend("gpu"):
                    assert xp.on_gpu
        assert not xp.on_gpu

    def test_nesting_cpu_only(self):
        with xp.backend("cpu"):
            with xp.backend("numpy"):
                assert not xp.on_gpu
            assert not xp.on_gpu

    def test_restores_after_exception(self):
        before = xp.on_gpu
        with pytest.raises(KeyError):
            with xp.backend("cpu"):
                assert not xp.on_gpu
                raise KeyError("boom")
        assert xp.on_gpu is before
        assert _core._backend_var.get() is None

    @requires_gpu
    def test_restores_after_exception_nested(self):
        xp.use_gpu()
        with xp.backend("gpu"):
            with pytest.raises(RuntimeError):
                with xp.backend("cpu"):
                    raise RuntimeError("x")
            assert xp.on_gpu
        assert _core._backend_var.get() is None

    @requires_gpu
    def test_generator_exit_restores(self):
        xp.use_gpu()

        def gen():
            with xp.backend("cpu"):
                yield xp.on_gpu

        g = gen()
        assert next(g) is False
        g.close()
        assert xp.on_gpu is True

    @requires_gpu
    def test_scoped_backend_unaffected_by_global_flip(self):
        xp.use_gpu()
        with xp.backend("gpu"):
            xp.use_cpu()
            assert xp.on_gpu is True
        assert xp.on_gpu is False

    @requires_gpu
    def test_namespace_follows_scope(self):
        xp.use_gpu()
        with xp.backend("cpu"):
            assert xp.sum is np.sum
            assert xp.ma is np.ma
        import cupy

        assert xp.sum is cupy.sum

    def test_decorator_reentrancy(self):
        cm = xp.backend("cpu")
        with cm:
            pass
        with pytest.raises(Exception):
            with cm:  # generator-based contexts are single use
                pass


class TestThreadIsolation:
    @requires_gpu
    def test_scoped_cpu_thread_vs_global_thread(self):
        xp.use_gpu()
        barrier = threading.Barrier(2, timeout=10)
        seen, errors = {}, []

        def a():
            try:
                with xp.backend("cpu"):
                    barrier.wait()  # B reads while A is inside the block
                    seen["a_in"] = (xp.on_gpu, type(xp.zeros(1)))
                    barrier.wait()
                seen["a_out"] = xp.on_gpu
            except Exception as e:  # pragma: no cover
                errors.append(e)
                barrier.abort()

        def b():
            try:
                barrier.wait()
                seen["b"] = (xp.on_gpu, type(xp.zeros(1)))
                barrier.wait()
            except Exception as e:  # pragma: no cover
                errors.append(e)
                barrier.abort()

        ts = [threading.Thread(target=a), threading.Thread(target=b)]
        [t.start() for t in ts]
        [t.join(15) for t in ts]
        assert not errors
        assert seen["a_in"] == (False, np.ndarray)
        assert seen["b"][0] is True and seen["b"][1] is _cupy().ndarray
        assert seen["a_out"] is True

    def test_new_thread_does_not_inherit_scope(self):
        out = []
        with xp.backend("cpu"):
            t = threading.Thread(target=lambda: out.append(xp.on_gpu))
            t.start()
            t.join(10)
        # a new thread starts with an empty context: it sees the global default
        assert out == [_core._global_gpu]

    @requires_gpu
    def test_two_threads_opposite_scopes(self):
        xp.use_gpu()
        barrier = threading.Barrier(2, timeout=10)
        res = {}

        def run(name, mode):
            with xp.backend(mode):
                barrier.wait()
                res[name] = xp.on_gpu
                barrier.wait()

        ts = [threading.Thread(target=run, args=("c", "cpu")),
              threading.Thread(target=run, args=("g", "gpu"))]
        [t.start() for t in ts]
        [t.join(15) for t in ts]
        assert res == {"c": False, "g": True}

    @requires_gpu
    def test_global_switch_visible_to_threads(self):
        xp.use_gpu()
        ev1, ev2 = threading.Event(), threading.Event()
        res = []

        def w():
            ev1.wait(10)
            res.append(xp.on_gpu)
            ev2.set()

        t = threading.Thread(target=w)
        t.start()
        xp.use_cpu()
        ev1.set()
        ev2.wait(10)
        t.join(10)
        assert res == [False]


class TestAsyncIsolation:
    @requires_gpu
    def test_two_tasks(self):
        xp.use_gpu()

        async def task(mode, ev_in, ev_other):
            with xp.backend(mode):
                ev_in.set()
                await ev_other.wait()
                await asyncio.sleep(0)
                return xp.on_gpu, type(xp.zeros(1))

        async def main():
            e1, e2 = asyncio.Event(), asyncio.Event()
            return await asyncio.gather(task("cpu", e1, e2), task("gpu", e2, e1))

        (c, g) = asyncio.run(main())
        assert c == (False, np.ndarray)
        assert g == (True, _cupy().ndarray)
        assert xp.on_gpu is True

    def test_task_scope_does_not_leak_to_caller(self):
        async def child():
            with xp.backend("cpu"):
                await asyncio.sleep(0)
                return xp.on_gpu

        async def main():
            before = xp.on_gpu
            r = await asyncio.create_task(child())
            return before, r, xp.on_gpu

        before, r, after = asyncio.run(main())
        assert r is False and after is before

    def test_task_inherits_scope_at_creation(self):
        async def child():
            return xp.on_gpu

        async def main():
            with xp.backend("cpu"):
                t = asyncio.create_task(child())
            return await t

        assert asyncio.run(main()) is False


@requires_gpu
class TestStress:
    def test_toggle_while_using(self):
        xp.use_gpu()
        errors = []
        stop = threading.Event()

        def worker():
            try:
                while not stop.is_set():
                    z = xp.zeros(3)
                    s = xp.sum(z)
                    assert float(s) == 0.0
                    assert z.shape == (3,)
            except BaseException as e:
                errors.append(e)

        def toggler():
            try:
                for _ in range(200):
                    xp.use_cpu()
                    xp.use_gpu()
            except BaseException as e:
                errors.append(e)

        ws = [threading.Thread(target=worker) for _ in range(4)]
        tg = threading.Thread(target=toggler)
        t0 = time.time()
        [w.start() for w in ws]
        tg.start()
        tg.join(30)
        stop.set()
        [w.join(30) for w in ws]
        assert time.time() - t0 < 10
        assert not errors, errors[:3]
        assert not any(w.is_alive() for w in ws)

    def test_first_access_race_builds_one_table(self):
        errors = []
        barrier = threading.Barrier(8, timeout=10)

        def w():
            try:
                barrier.wait()
                with xp.backend("gpu"):
                    xp.vecdot
                    xp.linalg.vector_norm
            except BaseException as e:
                errors.append(e)

        ts = [threading.Thread(target=w) for _ in range(8)]
        [t.start() for t in ts]
        [t.join(20) for t in ts]
        assert not errors


class TestOnDevice:
    @requires_gpu
    def test_numpy_integer_device_id(self):
        xp.use_gpu()
        with xp.on_device(np.int64(0)):
            assert xp.on_gpu is True
        xp.set_device(np.int32(0))

    @requires_gpu
    def test_minus_one_is_cpu_inside_gpu_after(self):
        xp.use_gpu()
        with xp.on_device(-1):
            assert xp.on_gpu is False
            assert type(xp.zeros(2)) is np.ndarray
        assert xp.on_gpu is True
        assert isinstance(xp.zeros(2), _cupy().ndarray)
        assert _core._global_gpu is True  # global untouched

    @requires_gpu
    def test_minus_one_restores_after_exception(self):
        xp.use_gpu()
        with pytest.raises(ValueError):
            with xp.on_device(-1):
                raise ValueError
        assert xp.on_gpu is True

    @requires_gpu
    def test_device_zero(self):
        xp.use_gpu()
        with xp.on_device(0):
            assert xp.on_gpu is True
            a = xp.ones(3)
            assert int(a.device.id) == 0
            assert int(_cupy().cuda.runtime.getDevice()) == 0

    @requires_gpu
    def test_device_restored(self):
        xp.use_gpu()
        before = int(_cupy().cuda.runtime.getDevice())
        with xp.on_device(0):
            pass
        assert int(_cupy().cuda.runtime.getDevice()) == before

    @requires_gpu
    @pytest.mark.parametrize("delta", [0, 1, 100])
    def test_out_of_range(self, delta):
        xp.use_gpu()
        with pytest.raises(ValueError):
            with xp.on_device(_core._n_gpus + delta):
                pytest.fail("block must not run")

    @requires_gpu
    @pytest.mark.parametrize("bad", [-2, -100, 1.0, "0", None])
    def test_invalid_values(self, bad):
        xp.use_gpu()
        with pytest.raises(ValueError):
            with xp.on_device(bad):
                pytest.fail("block must not run")

    @requires_gpu
    def test_bool_device_id(self):
        # bool is an int subclass: True == device 1, which is out of range on 1 GPU
        xp.use_gpu()
        if _core._n_gpus == 1:
            with pytest.raises(ValueError):
                with xp.on_device(True):
                    pass

    @requires_gpu
    def test_inside_scoped_cpu_is_noop(self):
        with xp.backend("cpu"):
            with xp.on_device(99):
                assert xp.on_gpu is False

    @requires_gpu
    def test_nested_minus_one_then_zero(self):
        xp.use_gpu()
        with xp.on_device(-1):
            assert not xp.on_gpu
            with xp.backend("gpu"):
                with xp.on_device(0):
                    assert xp.on_gpu
            assert not xp.on_gpu

    def test_cpu_noop(self):
        with xp.backend("cpu"):
            for i in (5, -1, 0, "x", None):
                with xp.on_device(i):
                    assert xp.on_gpu is False
                    assert type(xp.zeros(1)) is np.ndarray
            with pytest.raises(KeyError):
                with xp.on_device(5):
                    raise KeyError  # exceptions propagate through the no-op

    @requires_gpu
    def test_set_device_current_noop(self):
        xp.use_gpu()
        cur = int(_cupy().cuda.runtime.getDevice())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            xp.set_device(cur)
        assert int(_cupy().cuda.runtime.getDevice()) == cur

    @requires_gpu
    @pytest.mark.parametrize("bad", ["n", "-1", "str", "float"])
    def test_set_device_invalid(self, bad):
        xp.use_gpu()
        val = {"n": _core._n_gpus, "-1": -1, "str": "0", "float": 0.0}[bad]
        with pytest.raises(ValueError):
            xp.set_device(val)

    @requires_gpu
    def test_set_device_cpu_scope_noop(self):
        with xp.backend("cpu"):
            xp.set_device(_core._n_gpus + 5)


class TestLogging:
    @requires_gpu
    def test_switch_prints_and_logs(self, capsys, caplog):
        xp.use_gpu()
        capsys.readouterr()
        with caplog.at_level(logging.INFO, logger="xupy"):
            xp.use_cpu()
            xp.use_gpu()
        out = capsys.readouterr()
        assert out.out.splitlines() == [
            "[XuPy] Switched to CPU (NumPy).",
            "[XuPy] Switched to GPU (CuPy).",
        ]
        assert out.err == ""
        recs = [r for r in caplog.records if r.name == "xupy"]
        assert len(recs) == 2
        assert "CPU" in recs[0].getMessage()
        assert "GPU" in recs[1].getMessage()
        assert all(r.levelno == logging.INFO for r in recs)

    @requires_gpu
    def test_idempotent_calls_do_not_log(self, capsys, caplog):
        xp.use_gpu()
        capsys.readouterr()
        with caplog.at_level(logging.INFO, logger="xupy"):
            xp.use_gpu()
            xp.use_gpu()
        assert not [r for r in caplog.records if r.name == "xupy"]
        assert capsys.readouterr().out == ""

    def test_use_cpu_logs_once(self, capsys, caplog):
        xp.use_cpu()
        capsys.readouterr()  # the first call may print a real switch
        with caplog.at_level(logging.INFO, logger="xupy"):
            xp.use_cpu()
        assert not caplog.records
        assert capsys.readouterr().out == ""

    def test_backend_scope_is_quiet(self, capsys, caplog):
        with caplog.at_level(logging.INFO, logger="xupy"):
            with xp.backend("cpu"):
                pass
        assert capsys.readouterr().out == ""

    def test_logger_has_no_handlers_forced_on_user(self):
        lg = logging.getLogger("xupy")
        assert not [h for h in lg.handlers if isinstance(h, logging.StreamHandler)
                    and not isinstance(h, logging.NullHandler)]

    def test_import_prints_only_the_gpu_banner(self):
        import subprocess
        import sys

        r = subprocess.run(
            [sys.executable, "-c", "import xupy"], capture_output=True, text=True,
            env={**__import__("os").environ, "XUPY_NO_GPU_WARNING": "1"}, timeout=120,
        )
        assert r.returncode == 0
        if GPU_OK:
            assert r.stdout.startswith("[XuPy] ")
            assert "Using CuPy" in r.stdout
            assert "Switched" not in r.stdout
        else:
            assert r.stdout == ""
