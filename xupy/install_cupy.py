"""
Opt-in CuPy installer for XuPy
==============================

Detects the CUDA version available on this machine and installs the matching
CuPy wheel with pip. Importing ``xupy`` never runs this; it must be invoked
explicitly::

    python -m xupy.install_cupy              # detect, ask for confirmation, install
    python -m xupy.install_cupy --dry-run    # only show what would be done
    python -m xupy.install_cupy -y           # no confirmation prompt
    python -m xupy.install_cupy --package cupy-cuda12x>=14   # skip detection

Alternatively, install through the package extras: ``pip install xupy[cuda12]``
or ``pip install xupy[cuda13]``.

Exit codes: 0 success, 1 failure or declined, 2 confirmation required but the
session is not interactive (use ``--yes``). This module only uses the standard
library, so it works even when CuPy is broken.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys

_INSTALL_DOCS = "https://docs.cupy.dev/en/stable/install.html"


def _run_capture(cmd: list[str]) -> str | None:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout or ""


def get_cuda_version() -> str | None:
    """
    Return the CUDA version as ``"major.minor"``, or None if not found.

    Tries ``nvidia-smi`` first (maximum version supported by the driver), then
    ``nvcc --version``.
    """
    out = _run_capture(["nvidia-smi"])
    if out:
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    out = _run_capture(["nvcc", "--version"])
    if out:
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
    return None


def cupy_package_for(cuda_version: str) -> str | None:
    """Return the pip requirement for a CUDA version, or None if unsupported."""
    try:
        major = int(cuda_version.split(".")[0])
    except ValueError:
        return None
    return {12: "cupy-cuda12x>=14", 13: "cupy-cuda13x>=14"}.get(major)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m xupy.install_cupy",
        description="Install the CuPy wheel matching the local CUDA version.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="show the command without running it")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="do not ask for confirmation")
    parser.add_argument("--package", metavar="PKG",
                        help="pip requirement to install (skips autodetection)")
    args = parser.parse_args(argv)

    if args.package:
        pkg = args.package
        if pkg.startswith("-"):
            parser.error("--package must be a requirement, not a pip option")
    else:
        cuda = get_cuda_version()
        if cuda is None:
            print("Could not detect a CUDA installation (nvidia-smi / nvcc not found).")
            print(f"See {_INSTALL_DOCS} or use --package.")
            return 1
        print(f"Detected CUDA version: {cuda}")
        pkg = cupy_package_for(cuda)
        if pkg is None:
            print(f"CUDA {cuda} is not supported by the automatic installer (12.x and 13.x only).")
            print(f"See {_INSTALL_DOCS} or use --package.")
            return 1

    cmd = [sys.executable, "-m", "pip", "install", pkg]
    print(f"Command: {shlex.join(cmd)}")
    if args.dry_run:
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            print("Not an interactive session: pass --yes to proceed (or --dry-run).")
            return 2
        try:
            answer = input("Proceed? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print("CuPy installed. Restart Python to enable the GPU backend.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
