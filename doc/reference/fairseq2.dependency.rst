===================
fairseq2.dependency
===================

.. module:: fairseq2.dependency

    This module contains the lightweight `dependency injection`__ API of the
    library. See :doc:`/topics/design_philosophy` to learn more.

.. __: https://en.wikipedia.org/wiki/Dependency_injection

**ABCs**

* :class:`DependencyContainer`
* :class:`DependencyResolver`

**Classes**

* :class:`StandardDependencyContainer`

**Protocols**

* :class:`DependencyFactory`

ABCs
====

.. autoclass:: DependencyContainer

.. autoclass:: DependencyResolver

Classes
=======

.. autoclass:: StandardDependencyContainer

    ::

        from abc import ABC, abstractmethod

        from fairseq2.dependency import DependencyResolver, StandardDependencyContainer

        container = StandardDependencyContainer()

        # The interface
        class Foo(ABC):
            @abstractmethod
            def foo(self) -> None:
                ...

        # The implementation
        class FooImpl(Foo):
            def foo(self) -> None:
                pass

        # The factory
        def create_foo(resolver: DependencyResolver) -> Foo:
            assert resolver is container

            return FooImpl()

        container.register(Foo, create_foo)

        foo = container.resolve(Foo)

        assert isinstance(foo, FooImpl)

Protocols
=========

.. autoclass:: DependencyFactory()
    :special-members: __call__
