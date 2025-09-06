xupy._typings
=============

.. py:module:: xupy._typings


Attributes
----------

.. autoapisummary::

   xupy._typings.Array
   xupy._typings.Scalar
   xupy._typings.XupyMaskedArray


Classes
-------

.. autoapisummary::

   xupy._typings.XupyMaskedArrayProtocol


Module Contents
---------------

.. py:data:: Array

.. py:data:: Scalar

.. py:class:: XupyMaskedArrayProtocol(data, mask = None, dtype = None)

   Bases: :py:obj:`Protocol`


   Protocol defining the interface for XuPy masked arrays.


   .. py:attribute:: data
      :type:  Array


   .. py:attribute:: mask
      :type:  Array


   .. py:property:: shape
      :type: tuple[int, Ellipsis]



   .. py:property:: dtype
      :type: Any



   .. py:property:: size
      :type: int



   .. py:property:: ndim
      :type: int



   .. py:method:: asmarray(**kwargs)


   .. py:method:: __repr__()


   .. py:method:: __str__()


.. py:data:: XupyMaskedArray

