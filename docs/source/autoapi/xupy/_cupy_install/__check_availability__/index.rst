xupy._cupy_install.__check_availability__
=========================================

.. py:module:: xupy._cupy_install.__check_availability__


Functions
---------

.. autoapisummary::

   xupy._cupy_install.__check_availability__._read_code
   xupy._cupy_install.__check_availability__._was_asked_once
   xupy._cupy_install.__check_availability__.xupy_init


Module Contents
---------------

.. py:function:: _read_code()

.. py:function:: _was_asked_once()

.. py:function:: xupy_init()

   Subroutine of the XuPy package for the initialization of the GPU support.
   It checks if CuPy is installed and working. If not, it prompts the user to install it.
   It will not prompt the user again if already asked once (it saves the state in __install_cupy__.py).

   It is called by xupy._core.py during the import of the package.


