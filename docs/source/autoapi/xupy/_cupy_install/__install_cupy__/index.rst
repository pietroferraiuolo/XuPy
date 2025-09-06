xupy._cupy_install.__install_cupy__
===================================

.. py:module:: xupy._cupy_install.__install_cupy__

.. autoapi-nested-parse::

   Script to automatically detect CUDA installation and install the appropriate CuPy version.



Attributes
----------

.. autoapisummary::

   xupy._cupy_install.__install_cupy__.ASKED_FOR_CUPY


Functions
---------

.. autoapisummary::

   xupy._cupy_install.__install_cupy__.run_command
   xupy._cupy_install.__install_cupy__.get_cuda_version
   xupy._cupy_install.__install_cupy__.get_cupy_package
   xupy._cupy_install.__install_cupy__.install_package
   xupy._cupy_install.__install_cupy__.main


Module Contents
---------------

.. py:data:: ASKED_FOR_CUPY
   :value: False


.. py:function:: run_command(command)

   Run a shell command and return the output.


.. py:function:: get_cuda_version()

   Get the CUDA version from nvcc or nvidia-smi.


.. py:function:: get_cupy_package(cuda_version)

   Determine the correct CuPy package based on CUDA version.


.. py:function:: install_package(package)

   Install the package using pip.


.. py:function:: main()

