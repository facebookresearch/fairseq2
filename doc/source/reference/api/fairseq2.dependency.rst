===================
fairseq2.dependency
===================

.. module:: fairseq2.dependency

    This module contains the lightweight `dependency injection`__ API of the
    library. See :doc:`/basics/design_philosophy` to learn more.

.. __: https://en.wikipedia.org/wiki/Dependency_injection

**ABCs**

:class:`DependencyContainer`, :class:`DependencyResolver`

**Classes**

:class:`StandardDependencyContainer`

**Protocols**

:class:`DependencyFactory`

**Exceptions**

:class:`DependencyError`, :class:`DependencyNotFoundError`

**Functions**

:func:`get_container`

ABCs
====

.. autoclass:: DependencyContainer

.. autoclass:: DependencyResolver

Classes
=======

.. autoclass:: StandardDependencyContainer

Protocols
=========

.. autoclass:: DependencyFactory()
    :special-members: __call__

Exceptions
==========

.. autoclass:: DependencyError

.. autoclass:: DependencyNotFoundError

Functions
=========

.. autofunction:: get_container
