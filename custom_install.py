from setuptools.command.install import install
import subprocess
import os

class CustomInstall(install):
    def run(self):
        install.run(self)
        print("Custom install running... Checking for CuPy installation.")
        try:
            import xupy._cupy_install.__install_cupy__
            print("Checking for CUDA and install CuPy...")
            xupy._cupy_install.__install_cupy__.main()
        except ImportError as e:
            print(f"Could not import xupy.__install_cupy__: {e}. Skipping CuPy installation.")
        except Exception as e:
            print(f"Error running CuPy installation: {e}")
