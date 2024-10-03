# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from inspect import Parameter, signature
from typing import (
    Any,
    Final,
    Protocol,
    TypeVar,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import override

T = TypeVar("T")

T_co = TypeVar("T_co", covariant=True)


class DependencyResolver(ABC):
    """Resolves objects contained in a dependency graph."""

    @abstractmethod
    def resolve(self, kls: type[T], key: str | None = None) -> T:
        """
        Returns the object of type ``T``.

        :param kls: The value of ``T``.
        :param key: If not ``None``, the object with the specified key will be
            returned.

        :raises DependencyError: when the dependencies of the object cannot be
            inferred due to invalid or missing type annotations.
        :raises DependencyNotFoundError: when the object or one of its
            dependencies cannot be found.

        :returns: The resolved object.
        """

    @abstractmethod
    def resolve_optional(self, kls: type[T], key: str | None = None) -> T | None:
        """
        Returns the object of type ``T`` similar to :meth:`resolve`, but returns
        ``None`` instead of raising a :class:`DependencyNotFoundError` if the
        object is not found.

        :param kls: The value of ``T``.
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

        :param kls: The value of ``T``.

        :returns: An iterable of resolved objects. If no object is found, an
            empty iterable.
        """

    @abstractmethod
    def resolve_all_keyed(self, kls: type[T]) -> Iterable[tuple[str, T]]:
        """
        Returns all keyed objects of type ``T``.

        This method behaves similar to :meth:`resolve_all`, but returns an
        iterable of key-object pairs instead.

        :param kls: The value of ``T``.

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
        self, kls: type[T], sub_kls: type[T] | None = None, key: str | None = None
    ) -> None:
        """
        Registers a lazily-initialized object of type ``T``.

        The ``__init__()`` method of ``kls``, or if not ``None``, ``sub_kls``
        must have type annotations for all its parameters in order for
        dependency injection to work.

        If multiple objects of type ``T``, optionally with the same key, are
        registered, :meth:`~DependencyResolver.resolve` will return only the
        last registered one.

        :param kls: The value of ``T``.
        :param sub_kls: The real type of the object. If not ``None``, must be a
            subclass of ``kls``.
        :param key: If not ``None``, registers the object with the specified key.
            :meth:`~DependencyResolver.resolve` will return the object only if
            the same key is provided.

        :raises ValueError: when ``sub_kls`` is not a subclass of ``kls``.
        :raises RuntimeError: when called after the container is already used to
            resolve an object.
        """

    @abstractmethod
    def register_factory(
        self, kls: type[T], factory: DependencyFactory[T], key: str | None = None
    ) -> None:
        """
        Registers a lazily-initialized object of type ``T``.

        If multiple objects of type ``T``, optionally with the same key, are
        registered, :meth:`~DependencyResolver.resolve` will return only the
        last registered one.

        :param kls: The value of ``T``.
        :param factory: A callable to create an object of type ``T``. If the
            callable returns ``None``, the object is considered to not exist.
        :param key: If not ``None``, registers the object with the specified key.
            :meth:`~DependencyResolver.resolve` will return the object only if
            the same key is provided.

        :raises RuntimeError: when called after the container is already used to
            resolve an object.
        """

    @abstractmethod
    def register_instance(self, kls: type[T], obj: T, key: str | None = None) -> None:
        """
        Registers an object of type ``T``.

        Other than registering an existing object instead of a factory, the
        method behaves the same as :meth:`register`.

        :param kls: The value of ``T``.
        :param obj: The object to register.
        :param key: If not ``None``, registers the object with the specified key.
            :meth:`~DependencyResolver.resolve` will return the object only if
            the same key is provided.

        :raises RuntimeError: when called after the container is already used to
            resolve an object.
        """


class DependencyError(RuntimeError):
    """Raised when an error occurs while resolving a dependency."""


class DependencyNotFoundError(DependencyError):
    """Raised when a dependency cannot be found."""


@final
class StandardDependencyContainer(DependencyContainer):
    """
    This is the standard implementation of :class:`DependencyContainer` and
    transitively of :class:`DependencyResolver`.
    """

    _registrations: dict[type, list[_Registration]]
    _keyed_registrations: dict[type, dict[str, _Registration]]
    _frozen: bool

    def __init__(self) -> None:
        self._registrations = {}
        self._keyed_registrations = {}
        self._frozen = False

    @override
    def register(
        self, kls: type[T], sub_kls: type[T] | None = None, key: str | None = None
    ) -> None:
        if sub_kls is None:
            sub_kls = kls
        elif not issubclass(sub_kls, kls):
            raise ValueError(
                f"`sub_kls` must be a subclass of `kls`, but `{sub_kls}` is not a subclass of `{kls}`."
            )

        factory = self._create_factory(sub_kls)

        self._register(kls, key, _Registration(factory=factory))

    @override
    def register_factory(
        self, kls: type[T], factory: DependencyFactory[T], key: str | None = None
    ) -> None:
        self._register(kls, key, _Registration(factory=factory))

    @override
    def register_instance(self, kls: type[T], obj: T, key: str | None = None) -> None:
        self._register(kls, key, _Registration(obj=obj))

    def _register(
        self, kls: type, key: str | None, registration: _Registration
    ) -> None:
        if self._frozen:
            raise RuntimeError(
                "No new objects can be registered after the first `resolve()` call."
            )

        if kls is Iterable:
            raise ValueError(
                "`kls` must not be `Iterable` since it has special treatment within the dependency container."
            )

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
        self._frozen = True

        if key is None:
            try:
                registration = self._registrations[kls][-1]
            except (KeyError, IndexError):
                raise DependencyNotFoundError(
                    f"No registered factory or object found for `{kls}`."
                ) from None

            obj = self._get_object(kls, registration)
            if obj is None:
                raise DependencyNotFoundError(
                    f"The registered factory for `{kls}` returned `None`."
                )
        else:
            try:
                registration = self._keyed_registrations[kls][key]
            except KeyError:
                raise DependencyNotFoundError(
                    f"No registered factory or object found for `{kls}` with the key '{key}'."
                ) from None

            obj = self._get_object(kls, registration)
            if obj is None:
                raise DependencyNotFoundError(
                    f"The registered factory for `{kls}` with the key '{key}' returned `None`."
                )

        return obj

    @override
    def resolve_optional(self, kls: type[T], key: str | None = None) -> T | None:
        try:
            return self.resolve(kls, key)
        except DependencyNotFoundError:
            return None

    @override
    def resolve_all(self, kls: type[T]) -> Iterable[T]:
        self._frozen = True

        registrations = self._registrations.get(kls)
        if registrations is None:
            return

        for registration in registrations:
            obj = self._get_object(kls, registration)
            if obj is not None:
                yield obj

    @override
    def resolve_all_keyed(self, kls: type[T]) -> Iterable[tuple[str, T]]:
        self._frozen = True

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
            raise DependencyError(
                f"The object in the container is expected to be of type `{kls}`, but is of type `{type(obj)}` instead. Please file a bug report."
            )

        return obj

    @staticmethod
    def _create_factory(kls: type[T]) -> DependencyFactory[T]:
        def factory(resolver: DependencyResolver) -> T:
            init_method = getattr(kls, "__init__", None)
            if init_method is None:
                raise DependencyError(
                    f"`{kls} must have an `__init__()` method for dependency injection."
                )

            try:
                sig = signature(init_method)
            except (TypeError, ValueError) as ex:
                raise DependencyError(
                    f"The signature of `{init_method}` cannot be inspected. See nested exception for details."
                ) from ex

            try:
                type_hints = get_type_hints(init_method)
            except (TypeError, ValueError, NameError) as ex:
                raise DependencyError(
                    f"The type annotations of `{init_method}` cannot be inspected. See nested exception for details."
                ) from ex

            kwargs: dict[str, object] = {}

            for idx, (param_name, param) in enumerate(sig.parameters.items()):
                if idx == 0:  # i.e. self
                    continue

                if param.kind == Parameter.POSITIONAL_ONLY:
                    raise DependencyError(
                        f"`{init_method}` has one or more positional-only parameters which is not supported by `StandardDependencyContainer`."
                    )

                if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                    continue

                try:
                    param_type = type_hints[param_name]
                except KeyError:
                    raise DependencyError(
                        f"The `{param_name}` parameter of `{init_method}` has no type annotation."
                    )

                param_origin_type = get_origin(param_type)

                arg: Any

                if param_origin_type is Iterable:
                    param_type_args = get_args(param_type)
                    if len(param_type_args) != 1:
                        raise DependencyError(
                            f"The iterable `{param_name}` parameter of `{init_method}` has no element type expression."
                        )

                    element_type = param_type_args[0]
                    if not isinstance(element_type, type):
                        if param.default != Parameter.empty:
                            continue

                        raise DependencyError(
                            f"The element type of the iterable `{param_name}` parameter of `{init_method}` is not a `type`."
                        )

                    if get_origin(element_type) is tuple:
                        element_type_args = get_args(element_type)

                        if len(element_type_args) == 2 and element_type_args[0] is str:
                            kwargs[param_name] = resolver.resolve_all_keyed(
                                element_type_args[1]
                            )

                        continue

                    kwargs[param_name] = resolver.resolve_all(element_type)
                else:
                    if not isinstance(param_type, type):
                        if param.default != Parameter.empty:
                            continue

                        raise DependencyError(
                            f"The type of the `{param_name}` parameter of `{init_method}` is not a `type`."
                        )

                    if param.default != Parameter.empty:
                        arg = resolver.resolve_optional(param_type)
                        if arg is not None:
                            kwargs[param_name] = arg
                    else:
                        kwargs[param_name] = resolver.resolve(param_type)

            return kls(**kwargs)

        return factory


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

        self.obj = obj

        if factory is not None:
            self.factory = factory
        else:
            self.factory = lambda _: obj

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


def resolve_all(kls: type[T]) -> Iterable[T]:
    return get_resolver().resolve_all(kls)
