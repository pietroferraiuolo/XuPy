"""Tests for the dynamic (PEP 562) namespace of xupy: names, removals, ``ma``, ``import *``."""
import sys

import numpy as np
import pytest

import xupy as xp
from xupy import _core

GPU_OK = bool(_core._GPU_AVAILABLE)
requires_gpu = pytest.mark.skipif(not GPU_OK, reason="CuPy is not usable")


@pytest.fixture(autouse=True)
def _restore_backend():
    orig = _core._global_gpu
    yield
    if orig:
        xp.use_gpu()
    else:
        xp.use_cpu()


@pytest.fixture
def cpu():
    with xp.backend("cpu"):
        yield


@pytest.fixture
def gpu():
    if not GPU_OK:
        pytest.skip("CuPy is not usable")
    with xp.backend("gpu"):
        xp.zeros(1)  # make sure the GPU table (and _unsupported_gpu) is built
        yield


def _unsupported():
    with xp.backend("gpu"):
        xp.zeros(1)
    return sorted(_core._unsupported_gpu)


class TestNames:
    def test_public_names_in_dir_cpu(self, cpu):
        d = set(dir(xp))
        missing = _core._NUMPY_PUBLIC - d
        assert not missing, sorted(missing)[:20]

    def test_public_names_in_dir_gpu(self, gpu):
        d = set(dir(xp))
        missing = (_core._NUMPY_PUBLIC - _core._unsupported_gpu) - d
        assert not missing, sorted(missing)[:20]

    def test_public_names_resolve_cpu(self, cpu):
        for n in _core._NUMPY_PUBLIC:
            if hasattr(np, n):
                getattr(xp, n)

    def test_public_names_resolve_gpu(self, gpu):
        for n in _core._NUMPY_PUBLIC - _core._unsupported_gpu:
            getattr(xp, n)

    def test_unsupported_set_is_disjoint_from_gpu_dir(self, gpu):
        assert not (_core._unsupported_gpu & set(_core._tab_gpu))

    def test_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute 'definitely_not_a_name'"):
            xp.definitely_not_a_name
        assert not hasattr(xp, "definitely_not_a_name")
        assert getattr(xp, "definitely_not_a_name", 7) == 7

    def test_removed_is_not_public(self):
        assert not (set(_core._NUMPY2_REMOVED) & _core._NUMPY_PUBLIC)

    def test_dir_has_no_leaks(self):
        for mode in ("cpu", "gpu") if GPU_OK else ("cpu",):
            with xp.backend(mode):
                d = set(dir(xp))
                for bad in ("gc", "gpu", "gpu_name", "line1", "gpus", "typings"):
                    assert bad not in d, (mode, bad)

    def test_dir_core_has_no_leaks(self):
        d = set(dir(_core))
        for bad in ("gc", "gpu_name", "line1", "gpus"):
            assert bad not in d

    def test_dir_sorted_and_unique(self, cpu):
        d = dir(xp)
        assert d == sorted(d)
        assert len(d) == len(set(d))

    def test_xupy_own_names_both_modes(self):
        for mode in ("cpu", "gpu") if GPU_OK else ("cpu",):
            with xp.backend(mode):
                for n in ("asnumpy", "asmarray", "MemoryContext", "NumpyContext",
                          "on_device", "set_device", "array_size", "np", "npma",
                          "use_cpu", "use_gpu", "backend", "on_gpu", "ma",
                          "has_multi_gpu", "n_gpus", "__cuda_version__"):
                    assert hasattr(xp, n), (mode, n)


class TestRemoved:
    @pytest.mark.parametrize("name", sorted(_core._NUMPY2_REMOVED))
    def test_removed_cpu(self, name):
        with xp.backend("cpu"):
            with pytest.raises(AttributeError) as ei:
                getattr(xp, name)
            assert _core._NUMPY2_REMOVED[name] in str(ei.value)
            assert "removed in NumPy 2" in str(ei.value)
            assert not hasattr(xp, name)

    @requires_gpu
    @pytest.mark.parametrize("name", sorted(_core._NUMPY2_REMOVED))
    def test_removed_gpu(self, name):
        with xp.backend("gpu"):
            with pytest.raises(AttributeError) as ei:
                getattr(xp, name)
            assert _core._NUMPY2_REMOVED[name] in str(ei.value)
            assert not hasattr(xp, name)

    def test_removed_in_core_too(self):
        with pytest.raises(AttributeError, match="use 'nan'"):
            _core.NaN

    def test_from_import_removed_fails(self):
        with pytest.raises(ImportError):
            exec("from xupy import NaN", {})


class TestGpuUnsupported:
    @requires_gpu
    def test_message_points_to_cpu(self):
        names = _unsupported()
        if not names:
            pytest.skip("no GPU-unsupported names")
        for n in names:
            with xp.backend("gpu"):
                with pytest.raises(AttributeError) as ei:
                    getattr(xp, n)
            msg = str(ei.value)
            assert "cpu" in msg.lower() and "asnumpy" in msg, (n, msg)

    @requires_gpu
    def test_unsupported_available_on_cpu(self):
        for n in _unsupported():
            with xp.backend("cpu"):
                if hasattr(np, n):
                    assert getattr(xp, n) is getattr(np, n)


class TestVersion:
    def test_version_is_str_and_matches(self):
        from xupy.__version__ import __version__ as v

        assert isinstance(xp.__version__, str)
        assert xp.__version__ == v

    @requires_gpu
    def test_version_after_round_trips(self):
        from xupy.__version__ import __version__ as v

        for _ in range(3):
            xp.use_cpu()
            assert xp.__version__ == v
            xp.use_gpu()
            assert xp.__version__ == v
        assert isinstance(xp.__version__, str)

    def test_version_in_each_backend(self):
        from xupy.__version__ import __version__ as v

        for mode in ("cpu", "gpu") if GPU_OK else ("cpu",):
            with xp.backend(mode):
                assert xp.__version__ == v
                assert "__version__" in dir(xp)


class TestMa:
    def test_ma_cpu(self, cpu):
        assert xp.ma is np.ma
        from xupy import ma

        assert ma is np.ma
        import xupy.ma as m

        assert m is np.ma

    def test_ma_gpu(self, gpu):
        xm = sys.modules["xupy.ma"]
        assert xp.ma is xm
        from xupy import ma

        assert ma is xm
        import xupy.ma as m

        assert m is xm

    def test_ma_follows_global_switch(self):
        if not GPU_OK:
            pytest.skip("CuPy is not usable")
        xp.use_cpu()
        assert xp.ma is np.ma
        xp.use_gpu()
        assert xp.ma is sys.modules["xupy.ma"]
        xp.use_cpu()
        assert xp.ma is np.ma

    def test_sys_modules_always_xupy(self):
        for mode in ("cpu", "gpu") if GPU_OK else ("cpu",):
            with xp.backend(mode):
                m = sys.modules["xupy.ma"]
                assert m is not np.ma
                assert m.__name__ == "xupy.ma"

    def test_from_ma_import_always_xupy(self):
        xm = sys.modules["xupy.ma"]
        for mode in ("cpu", "gpu") if GPU_OK else ("cpu",):
            with xp.backend(mode):
                ns = {}
                exec("from xupy.ma import masked_array, MaskedArray", ns)
                assert ns["masked_array"] is xm.masked_array
                assert ns["MaskedArray"] is xm.MaskedArray
                assert ns["masked_array"] is not np.ma.masked_array

    def test_np_and_npma_both_modes(self):
        for mode in ("cpu", "gpu") if GPU_OK else ("cpu",):
            with xp.backend(mode):
                assert xp.np is np
                assert xp.npma is np.ma
        assert _core.np is np
        assert _core.npma is np.ma

    def test_ma_not_in_package_dict(self):
        assert "ma" not in vars(xp)

    def test_submodule_import_does_not_clobber(self):
        import importlib

        importlib.import_module("xupy.ma.core")
        with xp.backend("cpu"):
            assert xp.ma is np.ma


class TestStarImport:
    @pytest.mark.parametrize("mode", ["cpu", "gpu"])
    def test_star_import(self, mode):
        if mode == "gpu" and not GPU_OK:
            pytest.skip("CuPy is not usable")
        ns = {}
        with xp.backend(mode):
            exec("from xupy import *", ns)
        assert "zeros" in ns and "ma" in ns and "on_gpu" in ns
        for n in _core._NUMPY2_REMOVED:
            assert n not in ns, n
        for bad in ("gc", "gpu_name", "line1", "gpus", "typings"):
            assert bad not in ns
        if mode == "gpu":
            import cupy

            assert ns["zeros"] is cupy.zeros
            assert ns["ma"] is sys.modules["xupy.ma"]
            assert ns["on_gpu"] is True
        else:
            assert ns["zeros"] is np.zeros
            assert ns["ma"] is np.ma
            assert ns["on_gpu"] is False

    def test_all_is_dynamic_and_resolvable(self):
        for mode in ("cpu", "gpu") if GPU_OK else ("cpu",):
            with xp.backend(mode):
                for n in xp.__all__:
                    assert hasattr(xp, n), (mode, n)

    def test_all_differs_when_gpu_lacks_names(self):
        if not GPU_OK:
            pytest.skip("CuPy is not usable")
        with xp.backend("cpu"):
            a = set(xp.__all__)
        with xp.backend("gpu"):
            b = set(xp.__all__)
        assert b.issubset(a | {"asnumpy"} | set(_core._CUPY_ONLY))
        assert not (set(_unsupported()) & b)


class TestNotClobbered:
    def test_attributes_survive_switching(self):
        import xupy

        install = xupy.install_cupy if hasattr(xupy, "install_cupy") else None
        core = xupy._core
        for _ in range(3):
            for mode in ("cpu", "gpu") if GPU_OK else ("cpu",):
                with xp.backend(mode):
                    assert xupy._core is core
                    assert sys.modules["xupy._core"] is core
                    if install is not None:
                        assert xupy.install_cupy is install
                    assert hasattr(xupy, "__cuda_version__")
                    xupy.__cuda_version__
        if GPU_OK:
            xp.use_cpu()
            xp.use_gpu()
            xp.use_cpu()
            assert xupy._core is core
            assert xupy.__cuda_version__ == core._cuda_version

    def test_install_cupy_submodule_importable(self):
        import importlib

        m = importlib.import_module("xupy.install_cupy")
        assert m is sys.modules["xupy.install_cupy"]
        with xp.backend("cpu"):
            assert xp.__cuda_version__ == _core._cuda_version


class TestCpuIdentity:
    def test_identities(self, cpu):
        assert xp.zeros is np.zeros
        assert xp.float64 is np.float64
        assert xp.linalg is np.linalg
        assert xp.errstate is np.errstate
        assert xp.sort is np.sort
        assert xp.unique is np.unique
        assert xp.dtype is np.dtype
        assert xp.fft is np.fft
        assert xp.random is np.random

    def test_on_gpu_false(self, cpu):
        assert xp.on_gpu is False
        assert xp.__array_api_version__ == getattr(np, "__array_api_version__", None)

    @requires_gpu
    def test_gpu_identity(self, gpu):
        import cupy

        assert xp.zeros is cupy.zeros
        assert xp.on_gpu is True
        assert xp.float64 is cupy.float64 or xp.float64 is np.float64
