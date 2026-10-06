"""
Tests for import-time behaviour, the GPU-unusable warning and xupy.install_cupy.

No test here may run a real pip install: every path that reaches the install
step mocks ``subprocess.run``.
"""
import ast
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

import xupy as xp
import xupy._core as core
from xupy import install_cupy as ic

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# import xupy in a subprocess
# --------------------------------------------------------------------------
def _import_subprocess(extra_env):
    env = {k: v for k, v in os.environ.items()}
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", "import xupy"],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _snapshot_xupy():
    return {
        p: p.stat().st_mtime_ns
        for p in (REPO_ROOT / "xupy").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


class TestImportDoesNotBlock:
    def test_import_with_warning_silenced(self):
        res = _import_subprocess({"XUPY_NO_GPU_WARNING": "1"})
        assert res.returncode == 0, res.stderr

    def test_import_forced_cpu(self):
        res = _import_subprocess({"CUDA_VISIBLE_DEVICES": "", "XUPY_NO_GPU_WARNING": "1"})
        assert res.returncode == 0, res.stderr

    def test_forced_cpu_import_reports_no_gpu(self):
        env = {k: v for k, v in os.environ.items()}
        env.update({"CUDA_VISIBLE_DEVICES": "", "XUPY_NO_GPU_WARNING": "1"})
        res = subprocess.run(
            [sys.executable, "-c", "import xupy; print(xupy.on_gpu)"],
            cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=120,
        )
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip().splitlines()[-1] == "False"

    def test_import_modifies_no_file_under_xupy(self):
        before = _snapshot_xupy()
        res = _import_subprocess({"XUPY_NO_GPU_WARNING": "1"})
        assert res.returncode == 0, res.stderr
        after = _snapshot_xupy()
        assert before == after

    def test_old_installer_package_is_gone(self):
        assert not (REPO_ROOT / "xupy" / "_cupy_install").exists()


# --------------------------------------------------------------------------
# _warn_gpu_unusable
# --------------------------------------------------------------------------
@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("XUPY_NO_GPU_WARNING", raising=False)
    return monkeypatch


def _call_warn(**kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        core._warn_gpu_unusable("boom", **kwargs)
    return [x for x in w if issubclass(x.category, UserWarning)]


class TestWarnGpuUnusable:
    def test_warns_when_gpu_present(self, clean_env):
        clean_env.setattr(core, "_nvidia_gpu_present", lambda: True)
        assert len(_call_warn(cupy_importable=False)) == 1

    def test_no_warning_when_no_gpu_and_no_cupy(self, clean_env):
        clean_env.setattr(core, "_nvidia_gpu_present", lambda: False)
        assert _call_warn(cupy_importable=False) == []

    def test_default_cupy_importable_is_false(self, clean_env):
        clean_env.setattr(core, "_nvidia_gpu_present", lambda: False)
        assert _call_warn() == []

    def test_warns_when_no_gpu_but_cupy_importable(self, clean_env):
        clean_env.setattr(core, "_nvidia_gpu_present", lambda: False)
        assert len(_call_warn(cupy_importable=True)) == 1

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", " Yes ", "on"])
    def test_silenced_by_truthy_env(self, clean_env, value):
        clean_env.setattr(core, "_nvidia_gpu_present", lambda: True)
        clean_env.setenv("XUPY_NO_GPU_WARNING", value)
        assert _call_warn(cupy_importable=True) == []

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "FALSE", " No "])
    def test_not_silenced_by_falsy_env(self, clean_env, value):
        clean_env.setattr(core, "_nvidia_gpu_present", lambda: True)
        clean_env.setenv("XUPY_NO_GPU_WARNING", value)
        assert len(_call_warn(cupy_importable=True)) == 1

    def test_message_contents(self, clean_env):
        clean_env.setattr(core, "_nvidia_gpu_present", lambda: True)
        (w,) = _call_warn(cupy_importable=True)
        msg = str(w.message)
        assert "pip install xupy[cuda12]" in msg
        assert "python -m xupy.install_cupy" in msg
        assert "boom" in msg
        assert "XUPY_NO_GPU_WARNING" in msg

    def test_warning_category_is_userwarning(self, clean_env):
        clean_env.setattr(core, "_nvidia_gpu_present", lambda: True)
        (w,) = _call_warn()
        assert w.category is UserWarning

    def test_nvidia_gpu_present_uses_which(self, monkeypatch):
        monkeypatch.setattr(core._shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        assert core._nvidia_gpu_present() is True
        monkeypatch.setattr(core._shutil, "which", lambda name: None)
        assert core._nvidia_gpu_present() is False


# --------------------------------------------------------------------------
# cuda version / array_size
# --------------------------------------------------------------------------
class TestCoreModuleState:
    def test_cuda_version_matches_mode(self):
        if core.on_gpu:
            assert re.match(r"^\d+\.\d+$", core.__cuda_version__)
        else:
            assert core.__cuda_version__ is None

    def test_byte_constants_module_level(self):
        assert core._B2mb_ == 1024 * 1000
        assert core._Btgb_ == 1024 * 1000 * 1000

    def test_array_size_single_shape(self):
        # 1000*1000 float32 = 4e6 bytes -> 3 "MB" (1024*1000 bytes)
        assert xp.array_size((1000, 1000)) == 3

    def test_array_size_dtype_and_list(self):
        one = xp.array_size((1000, 1000), dtype="float64")
        assert one == 7
        total = xp.array_size([(1000, 1000), (1000, 1000)], dtype="float32")
        assert total == 6

    def test_array_size_gb(self):
        assert xp.array_size((1000, 1000, 1000), dtype="float32", out_unit="GB") == 3

    def test_array_size_small_is_zero(self):
        assert xp.array_size((3,)) == 0

    def test_array_size_returns_int(self):
        assert isinstance(xp.array_size((10, 10)), int)


# --------------------------------------------------------------------------
# xupy.install_cupy
# --------------------------------------------------------------------------
class TestCupyPackageFor:
    @pytest.mark.parametrize(
        "ver,expected",
        [
            ("12.4", "cupy-cuda12x>=14"),
            ("12.0", "cupy-cuda12x>=14"),
            ("13.2", "cupy-cuda13x>=14"),
            ("12", "cupy-cuda12x>=14"),
            ("11.8", None),
            ("10.2", None),
            ("14.0", None),
            ("0.0", None),
            ("garbage", None),
            ("", None),
            ("x.12", None),
        ],
    )
    def test_mapping(self, ver, expected):
        assert ic.cupy_package_for(ver) == expected


def _fake_run_factory(outputs):
    """outputs: dict cmd[0] -> str | Exception."""
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        assert "pip" not in cmd, "pip must never be invoked here"
        val = outputs[cmd[0]]
        if isinstance(val, BaseException):
            raise val
        return SimpleNamespace(stdout=val, returncode=0)

    return fake_run, calls


SMI = "| NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4 |"
NVCC = "Cuda compilation tools, release 11.8, V11.8.89"


class TestGetCudaVersion:
    def test_nvidia_smi_parsed(self, monkeypatch):
        run, calls = _fake_run_factory({"nvidia-smi": SMI})
        monkeypatch.setattr(ic.subprocess, "run", run)
        assert ic.get_cuda_version() == "12.4"
        assert len(calls) == 1

    def test_falls_back_to_nvcc_when_smi_missing(self, monkeypatch):
        run, _ = _fake_run_factory({"nvidia-smi": FileNotFoundError(), "nvcc": NVCC})
        monkeypatch.setattr(ic.subprocess, "run", run)
        assert ic.get_cuda_version() == "11.8"

    def test_falls_back_to_nvcc_when_smi_output_unparseable(self, monkeypatch):
        run, _ = _fake_run_factory({"nvidia-smi": "NVIDIA-SMI has failed", "nvcc": NVCC})
        monkeypatch.setattr(ic.subprocess, "run", run)
        assert ic.get_cuda_version() == "11.8"

    def test_both_missing(self, monkeypatch):
        run, _ = _fake_run_factory(
            {"nvidia-smi": FileNotFoundError(), "nvcc": FileNotFoundError()}
        )
        monkeypatch.setattr(ic.subprocess, "run", run)
        assert ic.get_cuda_version() is None

    def test_timeout_returns_none(self, monkeypatch):
        exc = subprocess.TimeoutExpired(cmd="x", timeout=1)
        run, _ = _fake_run_factory({"nvidia-smi": exc, "nvcc": exc})
        monkeypatch.setattr(ic.subprocess, "run", run)
        assert ic.get_cuda_version() is None

    def test_oserror_returns_none(self, monkeypatch):
        run, _ = _fake_run_factory({"nvidia-smi": PermissionError(), "nvcc": OSError()})
        monkeypatch.setattr(ic.subprocess, "run", run)
        assert ic.get_cuda_version() is None

    def test_empty_output_returns_none(self, monkeypatch):
        run, _ = _fake_run_factory({"nvidia-smi": "", "nvcc": ""})
        monkeypatch.setattr(ic.subprocess, "run", run)
        assert ic.get_cuda_version() is None


class _PipRecorder:
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.pip_calls = []

    def __call__(self, cmd, *args, **kwargs):
        if "pip" in cmd:
            self.pip_calls.append(cmd)
            return SimpleNamespace(returncode=self.returncode, stdout="")
        raise AssertionError(f"unexpected subprocess call: {cmd}")


@pytest.fixture
def forbid_pip(monkeypatch):
    def run(cmd, *a, **k):
        pytest.fail(f"subprocess.run called unexpectedly: {cmd}")

    monkeypatch.setattr(ic.subprocess, "run", run)


@pytest.fixture
def pip(monkeypatch):
    rec = _PipRecorder()
    monkeypatch.setattr(ic.subprocess, "run", rec)
    return rec


def _set_tty(monkeypatch, value):
    monkeypatch.setattr(ic.sys, "stdin", SimpleNamespace(isatty=lambda: value))


class TestInstallMain:
    def test_dry_run_prints_command(self, monkeypatch, forbid_pip, capsys):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "12.4")
        assert ic.main(["--dry-run"]) == 0
        out = capsys.readouterr().out
        assert sys.executable in out
        assert "cupy-cuda12x>=14" in out
        assert "pip" in out

    def test_dry_run_cuda13(self, monkeypatch, forbid_pip, capsys):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "13.0")
        assert ic.main(["--dry-run"]) == 0
        assert "cupy-cuda13x>=14" in capsys.readouterr().out

    def test_dry_run_with_yes_still_does_not_install(self, monkeypatch, forbid_pip):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "12.4")
        assert ic.main(["--dry-run", "--yes"]) == 0

    def test_explicit_package_dry_run(self, monkeypatch, forbid_pip, capsys):
        monkeypatch.setattr(
            ic, "get_cuda_version", lambda: pytest.fail("detection must be skipped")
        )
        assert ic.main(["--package", "foo", "--dry-run"]) == 0
        assert "foo" in capsys.readouterr().out

    def test_non_interactive_without_yes_returns_2(self, monkeypatch, forbid_pip):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "12.4")
        _set_tty(monkeypatch, False)
        assert ic.main([]) == 2

    def test_non_interactive_with_yes_installs(self, monkeypatch, pip):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "12.4")
        _set_tty(monkeypatch, False)
        assert ic.main(["-y"]) == 0
        assert pip.pip_calls == [[sys.executable, "-m", "pip", "install", "cupy-cuda12x>=14"]]

    @pytest.mark.parametrize("answer", ["", "n", "N", "no", "maybe", "  "])
    def test_interactive_declined(self, monkeypatch, forbid_pip, answer):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "12.4")
        _set_tty(monkeypatch, True)
        monkeypatch.setattr("builtins.input", lambda *a: answer)
        assert ic.main([]) == 1

    def test_input_eof_is_declined(self, monkeypatch, forbid_pip):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "12.4")
        _set_tty(monkeypatch, True)

        def raise_eof(*a):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert ic.main([]) == 1

    @pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES "])
    def test_interactive_accepted(self, monkeypatch, pip, answer):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "13.2")
        _set_tty(monkeypatch, True)
        monkeypatch.setattr("builtins.input", lambda *a: answer)
        assert ic.main([]) == 0
        assert len(pip.pip_calls) == 1
        assert pip.pip_calls[0][-1] == "cupy-cuda13x>=14"

    def test_pip_failure_returncode_propagates(self, monkeypatch, capsys):
        rec = _PipRecorder(returncode=7)
        monkeypatch.setattr(ic.subprocess, "run", rec)
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "12.4")
        _set_tty(monkeypatch, True)
        monkeypatch.setattr("builtins.input", lambda *a: "y")
        assert ic.main([]) == 7
        assert "CuPy installed" not in capsys.readouterr().out

    def test_success_message(self, monkeypatch, pip, capsys):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "12.4")
        assert ic.main(["--yes"]) == 0
        assert "CuPy installed" in capsys.readouterr().out

    @pytest.mark.parametrize("ver", ["11.8", "14.0", "10.2"])
    def test_unsupported_cuda_returns_1(self, monkeypatch, forbid_pip, ver):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: ver)
        assert ic.main(["--yes"]) == 1

    def test_undetected_cuda_returns_1(self, monkeypatch, forbid_pip):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: None)
        assert ic.main(["--yes"]) == 1

    def test_unsupported_cuda_dry_run_still_fails(self, monkeypatch, forbid_pip):
        monkeypatch.setattr(ic, "get_cuda_version", lambda: "11.8")
        assert ic.main(["--dry-run"]) == 1

    def test_explicit_package_install(self, monkeypatch, pip):
        assert ic.main(["--package", "cupy-cuda12x>=14", "-y"]) == 0
        assert pip.pip_calls[0][-1] == "cupy-cuda12x>=14"

    def test_unknown_option_exits_2(self, forbid_pip):
        with pytest.raises(SystemExit) as e:
            ic.main(["--bogus"])
        assert e.value.code == 2


class TestInstallModuleIsolation:
    def test_no_cupy_or_core_imports(self):
        tree = ast.parse(Path(ic.__file__).read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(("." * node.level) + (node.module or ""))
                names += [a.name for a in node.names]
        for n in names:
            assert "cupy" not in n.split(".")[0]
            assert "_core" not in n
            assert "xupy" not in n
