===================
fairseq2.dependency
===================

.. module:: fairseq2.dependency
    :synopsis: A lightweight dependency injection API

    fairseq2 relies on the `dependency inversion principle`_ to maintain a clean,
    well-tested, and extensible code base.

    This module contains the abstract classes :class:`DependencyResolver` and
    :class:`DependencyContainer`, as well as the
    :class:`StandardDependencyContainer` class, as a lightweight `dependency
    injection`_ API.

.. _`dependency inversion principle`: https://en.wikipedia.org/wiki/Dependency_inversion_principle
.. _`dependency injection`: https://en.wikipedia.org/wiki/Dependency_injection

.. class:: DependencyResolver()

    extends :class:`~abc.ABC`

    .. method:: resolve[T](kls: type[T], key: str | None = None) -> T
        :abstractmethod:

        Return the singleton dependency of type ``T``. If ``key`` is not ``None``,
        the singleton dependency with the specified key will be returned.

        If a dependency of type ``T``, or a dependency of type ``T`` with ``key``
        cannot be found, a :class:`LookupError` is raised.

    .. method:: resolve_optional(kls: type[T], key: str | None = None) -> T | None
        :abstractmethod:

        Return the singleton dependency of type ``T`` similar to :meth:`resolve`,
        but return ``None`` instead of raising a :class:`LookupError` if the
        dependency is not found.

    .. method:: resolve_all(kls: type[T]) -> Iterable[T]:
        :abstractmethod:

        Return all singleton dependencies of type ``T`` that have no associated
        key.

        If multiple singleton dependencies of type ``T`` are registered,
        :meth:`resolve` only returns the last registered one. In contrast,
        :meth:`resolve_all` returns them all, in the order that they were
        registered.

        If no dependency can be found, an empty iterable will be returned.

    .. method:: resolve_all_keyed(kls: type[T]) -> Iterable[tuple[str, T]]
        :abstractmethod:

        Return all singleton dependencies of type ``T`` that have an associated
        key.

        This method behaves similar to :meth:`resolve_all`, but returns an
        iterable of key-dependency pairs instead.

.. class:: DependencyContainer()

    extends :class:`DependencyResolver`

    .. method:: register[T](kls: type[T], factory: DependencyFactory[T], key: str | None = None) -> None
        :abstractmethod:

        Register a singleton dependency of type ``T``.

        ``factory`` is expected to return an instance of type ``T``, or ``None``
        if the instance cannot be created. It should follow the protocol::

            class DependencyFactory(Protocol[T]):
                def __call__(self, resolver: DependencyResolver) -> T | None:
                    ...

        ``resolver`` passed to ``factory`` can be used to resolve the
        dependencies needed to construct the instance itself.

        Optionally ``key`` can be specified. In this case, the same key must
        be passed to :meth:`~DependencyResolver.resolve` to return the
        dependency.

        If multiple singleton dependencies of type ``T``, optionally with the
        same key, are registered, :meth:`~DependencyResolver.resolve` will
        return the last registered one.

    .. method:: register_instance[T](kls: type[T], obj: T, key: str | None = None) -> None
        :abstractmethod:

        Register an existing singleton dependency of type ``T``.

        Other than registering ``obj`` instead of a factory, the method behaves
        the same as :meth:`register`.

.. class:: StandardDependencyContainer()
    :final:

    implements :class:`DependencyContainer`

    This is the standard implementation of :class:`DependencyContainer` and
    transitively of :class:`DependencyResolver`.

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
