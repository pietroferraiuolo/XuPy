#!/usr/bin/env python3
"""
Script to automatically detect CUDA installation and install the appropriate CuPy version.
"""
ASKED_FOR_CUPY = False

import subprocess
import sys
import re


def run_command(command):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def get_cuda_version():
    """Get the CUDA version from nvcc or nvidia-smi."""
    # Try nvcc first
    success, stdout, stderr = run_command("nvcc --version")
    if success:
        # Parse version from nvcc output
        match = re.search(r'release (\d+)\.(\d+)', stdout)
        if match:
            major, minor = match.groups()
            return f"{major}.{minor}"
    
    # Fallback to nvidia-smi
    success, stdout, stderr = run_command("nvidia-smi")
    if success:
        # Parse CUDA version from nvidia-smi
        match = re.search(r'CUDA Version: (\d+)\.(\d+)', stdout)
        if match:
            major, minor = match.groups()
            return f"{major}.{minor}"
    
    return None


def get_cupy_package(cuda_version):
    """Determine the correct CuPy package based on CUDA version."""
    if cuda_version.startswith("10."):
        return "cupy-cuda102"
    elif cuda_version.startswith("11."):
        return "cupy-cuda11x"
    elif cuda_version.startswith("12."):
        return "cupy-cuda12x"
    else:
        print(f"       Unsupported CUDA version: {cuda_version}")
        return None


def install_package(package):
    """Install the package using pip."""
    print(f"       Installing {package}...")
    success, stdout, stderr = run_command(f"pip install {package}")
    if success:
        print(f"       Successfully installed {package}")
    else:
        print(f"       Failed to install {package}")
        print("       stdout:", stdout)
        print("       stderr:", stderr)
        return False
    return True


def main():
    print("\n       Checking for CUDA installation...")
    cuda_version = get_cuda_version()
    if cuda_version:
        print(f"       CUDA version detected: {cuda_version}")
        cupy_package = get_cupy_package(cuda_version)
        if cupy_package:
            if install_package(cupy_package):
                print("       CuPy installation completed successfully.")
            else:
                print("       CuPy installation failed.")
                sys.exit(1)
        else:
            print("       Could not determine appropriate CuPy package.")
            sys.exit(1)
    else:
        print("       CUDA not detected. Skipping CuPy installation.")
        sys.exit(0)


if __name__ == "__main__":
    main()
