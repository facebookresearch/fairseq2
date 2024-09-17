# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Final, Protocol, TypeVar, final

from typing_extensions import override

T = TypeVar("T")

T_co = TypeVar("T_co", covariant=True)


class DependencyResolver(ABC):
    """Resolves objects contained in a dependency graph."""

    @abstractmethod
    def resolve(self, kls: type[T], key: str | None = None) -> T:
        """
        Returns the object of type ``T``.

        :param kls: The :class:`type` of ``T``.
        :param key: If not ``None``, the object with the specified key will be
            returned.

        :raises LookupError: when the object cannot be found.

        :returns: The resolved object.
        """

    @abstractmethod
    def resolve_optional(self, kls: type[T], key: str | None = None) -> T | None:
        """
        Returns the object of type ``T`` similar to :meth:`resolve`, but returns
        ``None`` instead of raising a :class:`LookupError` if the object is not
        found.

        :param kls: The :class:`type` of ``T``.
        :param key: If not ``None``, the object with the specified key will be
            returned.

        :returns: The resolved object, or ``None`` if the object is not found.
        """

    @abstractmethod
    def resolve_all(self, kls: type[T]) -> Iterable[T]:
        """
        Returns all non-keyed objects of type ``T``.

        If multiple objects of type ``T`` are registered, :meth:`resolve` returns
        only the last registered one. In contrast, ``resolve_all`` returns
        them all in the order they were registered.

        :param kls: The :class:`type` of ``T``.

        :returns: An iterable of resolved objects. If no object is found, an
            empty iterable.
        """

    @abstractmethod
    def resolve_all_keyed(self, kls: type[T]) -> Iterable[tuple[str, T]]:
        """
        Returns all keyed objects of type ``T``.

        This method behaves similar to :meth:`resolve_all`, but returns an
        iterable of key-object pairs instead.

        :param kls: The :class:`type` of ``T``.

        :returns: An iterable of resolved key-object pairs. If no object is
            found, an empty iterable.
        """


class DependencyFactory(Protocol[T_co]):
    """
    Used by :meth:`DependencyContainer.register` for its ``factory`` parameter.
    """

    def __call__(self, resolver: DependencyResolver) -> T_co | None:
        """
        Creates an object of type ``T_co``.

        :param resolver: The dependency resolver that the callable can use to
            resolve the dependencies of the newly-created object.

        :returns: An object of type ``T_co`` that will be registered in the
            calling :class:`DependencyContainer`, or ``None`` if the object
            cannot be created.
        """


class DependencyContainer(DependencyResolver):
    """Holds a dependency graph."""

    @abstractmethod
    def register(
        self, kls: type[T], factory: DependencyFactory[T], key: str | None = None
    ) -> None:
        """
        Registers a lazily-initialized object of type ``T``.

        If multiple objects of type ``T``, optionally with the same key, are
        registered, :meth:`~DependencyResolver.resolve` will return only the
        last registered one.

        :param kls: The :class:`type` of ``T``.
        :param factory: A callable to create an object of type ``T``. If the
            callable returns ``None``, the object is considered to not exist.
        :param key: If not ``None``, registers the object with the specified key.
            :meth:`~DependencyResolver.resolve` will return the object only if
            the same key is provided.
        """

    @abstractmethod
    def register_object(self, kls: type[T], obj: T, key: str | None = None) -> None:
        """
        Registers an object of type ``T``.

        Other than registering an existing object instead of a factory, the
        method behaves the same as :meth:`register`.

        :param kls: The :class:`type` of ``T``.
        :param obj: The object to register.
        :param key: If not ``None``, registers the object with the specified key.
            :meth:`~DependencyResolver.resolve` will return the object only if
            the same key is provided.
        """


@final
class StandardDependencyContainer(DependencyContainer):
    """
    This is the standard implementation of :class:`DependencyContainer` and
    transitively of :class:`DependencyResolver`.
    """

    _registrations: dict[type, list[_Registration]]
    _keyed_registrations: dict[type, dict[str, _Registration]]

    def __init__(self) -> None:
        self._registrations = {}
        self._keyed_registrations = {}

    @override
    def register(
        self, kls: type[T], factory: DependencyFactory[T], key: str | None = None
    ) -> None:
        self._register(kls, key, _Registration(factory=factory))

    @override
    def register_object(self, kls: type[T], obj: T, key: str | None = None) -> None:
        self._register(kls, key, _Registration(obj=obj))

    def _register(
        self, kls: type, key: str | None, registration: _Registration
    ) -> None:
        if key is None:
            registrations = self._registrations.get(kls)
            if registrations is None:
                registrations = []

                self._registrations[kls] = registrations

            registrations.append(registration)
        else:
            keyed_registrations = self._keyed_registrations.get(kls)
            if keyed_registrations is None:
                keyed_registrations = {}

                self._keyed_registrations[kls] = keyed_registrations

            keyed_registrations[key] = registration

    @override
    def resolve(self, kls: type[T], key: str | None = None) -> T:
        if key is None:
            try:
                registration = self._registrations[kls][-1]
            except (KeyError, IndexError):
                raise LookupError(
                    f"No registered factory or object found for `{kls}`."
                ) from None

            obj = self._get_object(kls, registration)
            if obj is None:
                raise LookupError(
                    f"The registered factory for `{kls}` returned `None`."
                )
        else:
            try:
                registration = self._keyed_registrations[kls][key]
            except KeyError:
                raise LookupError(
                    f"No registered factory or object found for `{kls}` with the key '{key}'."
                ) from None

            obj = self._get_object(kls, registration)
            if obj is None:
                raise LookupError(
                    f"The registered factory for `{kls}` with the key '{key}' returned `None`."
                )

        return obj

    @override
    def resolve_optional(self, kls: type[T], key: str | None = None) -> T | None:
        try:
            return self.resolve(kls, key)
        except LookupError:
            return None

    @override
    def resolve_all(self, kls: type[T]) -> Iterable[T]:
        registrations = self._registrations.get(kls)
        if registrations is None:
            return

        for registration in registrations:
            obj = self._get_object(kls, registration)
            if obj is not None:
                yield obj

    @override
    def resolve_all_keyed(self, kls: type[T]) -> Iterable[tuple[str, T]]:
        keyed_registrations = self._keyed_registrations.get(kls)
        if keyed_registrations is None:
            return

        for key, registration in keyed_registrations.items():
            obj = self._get_object(kls, registration)
            if obj is not None:
                yield (key, obj)

    def _get_object(self, kls: type[T], registration: _Registration) -> T | None:
        if registration.obj is _NOT_SET:
            obj = registration.factory(self)

            if not registration.transient:
                registration.obj = obj
        else:
            obj = registration.obj

        if obj is not None and not isinstance(obj, kls):
            raise TypeError(
                f"The object in the container is expected to be of type `{kls}`, but is of type `{type(obj)}` instead. Please file a bug report."
            )

        return obj


_NOT_SET: Final = object()


class _Registration:
    obj: object
    factory: DependencyFactory[object]
    transient: bool

    def __init__(
        self,
        *,
        obj: object = _NOT_SET,
        factory: DependencyFactory[object] | None = None,
        transient: bool = False,
    ) -> None:
        if obj is _NOT_SET and factory is None:
            raise RuntimeError(
                "Neither `obj` nor `factory` is specified. Please file a bug report."
            )

        if factory is None:
            factory = lambda _: obj

        self.obj = obj
        self.factory = factory
        self.transient = transient


_container: DependencyContainer | None = None


def _set_container(container: DependencyContainer) -> None:
    global _container

    _container = container


def get_container() -> DependencyContainer:
    """
    Returns the global :class:`DependencyContainer` instance.

    :raises RuntimeError: when :func:`~fairseq2.setup_fairseq2` has not been
        called first which is responsible for initializing the global container.
    """
    global _container

    if _container is None:
        raise RuntimeError(
            "fairseq2 is not initialized. Make sure to call `fairseq2.setup_fairseq2()`."
        )

    return _container


def get_resolver() -> DependencyResolver:
    return get_container()


def resolve(kls: type[T], key: str | None = None) -> T:
    return get_resolver().resolve(kls, key)
