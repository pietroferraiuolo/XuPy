from setuptools import setup, find_packages
from setuptools.command.install import install
import tomllib

with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
    __version__ = data['project']['version']

class CustomInstall(install):
    def run(self):
        install.run(self)
        print("Custom install running... Checking for CuPy installation.")
        try:
            import xupy._cupy_install.__install_cupy__
            print("Running install_cupy.py to check for CUDA and install CuPy...")
            xupy._cupy_install.__install_cupy__.main()
        except ImportError as e:
            print(f"Could not import xupy.install_cupy: {e}. Skipping CuPy installation.")
        except Exception as e:
            print(f"Error running CuPy installation: {e}")

setup(
    name="XuPy",
    version=__version__,
    description="GPU Accelerated masked arrays with automatic handling of CPU and GPU arrays.",
    author="Pietro Ferraiuolo",
    author_email="pietro.ferraiuolo@inaf.it",
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.10",
    cmdclass={
        'install': CustomInstall,
    },
)
