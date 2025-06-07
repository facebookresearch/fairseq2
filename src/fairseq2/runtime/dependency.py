# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from typing import Final, Protocol, TypeVar, cast, final

from typing_extensions import override

from fairseq2.error import ContractError, InternalError, InvalidOperationError
from fairseq2.runtime.provider import Provider

T_co = TypeVar("T_co", covariant=True)


class DependencyFactory(Protocol[T_co]):
    def __call__(self, resolver: DependencyResolver) -> T_co: ...


T = TypeVar("T")


class DependencyContainer(ABC):
    @abstractmethod
    def register(
        self,
        kls: type[T],
        factory: DependencyFactory[T],
        *,
        key: Hashable | None = None,
    ) -> None: ...

    @abstractmethod
    def register_instance(
        self, kls: type[T], obj: T, *, key: Hashable | None = None
    ) -> None: ...


class DependencyResolver(ABC):
    @abstractmethod
    def resolve(self, kls: type[T], *, key: Hashable | None = None) -> T: ...

    @abstractmethod
    def resolve_all(
        self, kls: type[T], *, key: Hashable | None = None
    ) -> Iterable[T]: ...

    @abstractmethod
    def get_provider(self, kls: type[T]) -> Provider[T]: ...


class DependencyNotFoundError(LookupError):
    pass


class StandardDependencyContainer(DependencyContainer, DependencyResolver):
    _registrations: dict[Hashable, list[_Registration]]
    _frozen: bool

    def __init__(self) -> None:
        self._registrations = {}

        self._frozen = False

    @override
    def register(
        self,
        kls: type[T],
        factory: DependencyFactory[T],
        *,
        key: Hashable | None = None,
    ) -> None:
        self._register(kls, key, _Registration(factory=factory))

    @override
    def register_instance(
        self, kls: type[T], obj: T, *, key: Hashable | None = None
    ) -> None:
        self._register(kls, key, _Registration(obj))

    def _register(
        self, kls: type[object], key: Hashable | None, registration: _Registration
    ) -> None:
        if self._frozen:
            raise InvalidOperationError(
                "No new objects can be registered after the first `resolve()` call."
            )

        full_key = (kls, key)

        registrations = self._registrations.get(full_key)
        if registrations is None:
            registrations = []

            self._registrations[full_key] = registrations

        registrations.append(registration)

    @override
    def resolve(self, kls: type[T], *, key: Hashable | None = None) -> T:
        self._frozen = True

        full_key = (kls, key)

        try:
            registration = self._registrations[full_key][-1]
        except (IndexError, KeyError):
            if key is None:
                msg = f"No registered factory or object found for `{kls}`."
            else:
                msg = f"No registered factory or object found for `{kls}` with key '{key}'."

            raise DependencyNotFoundError(msg) from None

        obj = self._get_object(kls, registration)
        if obj is None:
            if key is None:
                msg = f"The registered factory for `{kls}` returned `None`."
            else:
                msg = f"The registered factory for `{kls}` with key '{key}' returned `None`."

            raise DependencyNotFoundError(msg) from None

        return obj

    @override
    def resolve_all(self, kls: type[T], *, key: Hashable | None = None) -> Iterable[T]:
        self._frozen = True

        full_key = (kls, key)

        registrations = self._registrations.get(full_key)
        if registrations is None:
            return

        for registration in registrations:
            obj = self._get_object(kls, registration)
            if obj is not None:
                yield obj

    @override
    def get_provider(self, kls: type[T]) -> Provider[T]:
        return DependencyProvider(self, kls)

    def _get_object(self, kls: type[T], registration: _Registration) -> T | None:
        if registration.obj is _NOT_SET:
            obj = registration.factory(self)

            registration.obj = obj
        else:
            obj = registration.obj

        if obj is not None and not isinstance(obj, kls):
            raise ContractError(
                f"The object in the container is expected to be of type `{kls}`, but is of type `{type(obj)}` instead."
            )

        return obj


_NOT_SET: Final = object()


class _Registration:
    obj: object
    factory: DependencyFactory[object]

    def __init__(
        self, obj: object = _NOT_SET, factory: DependencyFactory[object] | None = None
    ) -> None:
        if obj is _NOT_SET and factory is None:
            raise InternalError(
                "Neither `obj` nor `factory` is specified. Please file a bug report."
            )

        self.obj = obj

        if factory is not None:
            self.factory = factory
        else:
            self.factory = lambda _: obj


@final
class DependencyProvider(Provider[T]):
    _resolver: DependencyResolver
    _kls: type[object]

    def __init__(self, resolver: DependencyResolver, kls: type[object]) -> None:
        self._resolver = resolver
        self._kls = kls

    @override
    def get(self, key: Hashable) -> T:
        obj = self._resolver.resolve(self._kls, key=key)

        return cast(T, obj)


_resolver: DependencyResolver | None = None


def get_dependency_resolver() -> DependencyResolver:
    if _resolver is None:
        from fairseq2 import init_fairseq2

        init_fairseq2()

    if _resolver is None:
        raise InternalError("`_resolver` is `None`.")

    return _resolver
