from setuptools import setup
from setuptools.command.install import install

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
    version="1.1.0",
    description="Masked Arrays made simple for CuPy",
    author="Pietro Ferraiuolo",
    author_email="pietro.ferraiuolo@inaf.it",
    packages=["xupy"],
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.10",
    cmdclass={
        'install': CustomInstall,
    },
)
