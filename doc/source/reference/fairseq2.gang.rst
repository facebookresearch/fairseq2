=============
fairseq2.gang
=============

.. automodule:: fairseq2.gang
    :no-members:

ABCs
====

.. autoclass:: Gang
.. autoclass:: GangContext

Classes
=======

.. autoclass:: ProcessGroupGang
.. autoclass:: FakeGang
.. autoclass:: Gangs

Enums
=====

.. autoclass:: ReduceOperation

Factory Functions
=================

See the :meth:`ProcessGroupGang.create_default_process_group` method for
creating the default PyTorch ProcessGroup. The rest of the factory functions
listed below are used to create sub-gangs for different parallelism strategies.

.. autofunction:: create_parallel_gangs
.. autofunction:: create_fsdp_gangs
.. autofunction:: create_fake_gangs

Functions
=========

.. autofunction:: get_current_gangs
.. autofunction:: get_default_gangs
.. autofunction:: set_default_gangs
.. autofunction:: set_gangs
.. autofunction:: broadcast_flag
.. autofunction:: all_sum


Exceptions
==========

.. autoclass:: GangError
