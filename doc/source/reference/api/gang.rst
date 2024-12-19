=============
fairseq2.gang
=============

.. module:: fairseq2.gang

This module provides the implementation of the ``Gang`` class and its related classes for managing collective operations in a distributed environment.

Classes
-------

.. autoclass:: Gang
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AbstractGang
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FakeGang
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ProcessGroupGang
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GangError
   :members:
   :undoc-members:

.. autoclass:: ReduceOperation
   :members:
   :undoc-members:

Functions
---------

.. autofunction:: setup_default_gang
.. autofunction:: fake_gangs
.. autofunction:: setup_parallel_gangs
.. autofunction:: broadcast_flag
.. autofunction:: all_sum
.. autofunction:: get_world_size
.. autofunction:: get_rank
.. autofunction:: get_local_world_size
.. autofunction:: get_local_rank
.. autofunction:: is_torchrun
