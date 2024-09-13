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
    @abstractmethod
    def resolve(self, kls: type[T], key: str | None = None) -> T:
        ...

    @abstractmethod
    def resolve_optional(self, kls: type[T], key: str | None = None) -> T | None:
        ...

    @abstractmethod
    def resolve_all(self, kls: type[T]) -> Iterable[T]:
        ...

    @abstractmethod
    def resolve_all_keyed(self, kls: type[T]) -> Iterable[tuple[str, T]]:
        ...


class DependencyFactory(Protocol[T_co]):
    def __call__(self, resolver: DependencyResolver) -> T_co | None:
        ...


class DependencyContainer(DependencyResolver):
    @abstractmethod
    def register(
        self, kls: type[T], factory: DependencyFactory[T], key: str | None = None
    ) -> None:
        ...

    @abstractmethod
    def register_instance(self, kls: type[T], obj: T, key: str | None = None) -> None:
        ...


@final
class StandardDependencyContainer(DependencyContainer):
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
    def register_instance(self, kls: type[T], obj: T, key: str | None = None) -> None:
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
                    f"No registered factory or instance found for `{kls}`."
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
                    f"No registered factory or instance found for `{kls}` with the key '{key}'."
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
        objs: list[T] = []

        registrations = self._registrations.get(kls)
        if registrations is None:
            return objs

        for registration in registrations:
            obj = self._get_object(kls, registration)
            if obj is not None:
                objs.append(obj)

        return objs

    @override
    def resolve_all_keyed(self, kls: type[T]) -> Iterable[tuple[str, T]]:
        objs: list[tuple[str, T]] = []

        keyed_registrations = self._keyed_registrations.get(kls)
        if keyed_registrations is None:
            return objs

        for key, registration in keyed_registrations.items():
            obj = self._get_object(kls, registration)
            if obj is not None:
                objs.append((key, obj))

        return objs

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


def get_default_container() -> DependencyContainer:
    global _container

    if _container is None:
        raise RuntimeError(
            "fairseq2 is not initialized. Make sure to call `fairseq2.setup_fairseq2()`."
        )

    return _container


def set_default_container(container: DependencyContainer) -> None:
    global _container

    _container = container


def get_default_resolver() -> DependencyResolver:
    return get_default_container()


def resolve(kls: type[T], key: str | None = None) -> T:
    return get_default_resolver().resolve(kls, key)
