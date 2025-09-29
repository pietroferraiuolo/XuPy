"""
XuPy: A library with authomatic handling of GPU and CPU arrays.
"""
from ._core import *
from .__version__ import __version__

if on_gpu:
    from . import ma
