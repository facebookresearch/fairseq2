# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Collection, Hashable, Iterable, Iterator, Sequence
from inspect import Parameter, signature
from typing import (
    Any,
    Final,
    Protocol,
    TypeVar,
    cast,
    final,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import override

from fairseq2.error import InternalError, InvalidOperationError
from fairseq2.runtime.lazy import Lazy
from fairseq2.runtime.lookup import Lookup

T = TypeVar("T")


class DependencyResolver(ABC):
    @abstractmethod
    def resolve(self, kls: type[T], *, key: Hashable | None = None) -> T: ...

    @abstractmethod
    def resolve_optional(
        self, kls: type[T], *, key: Hashable | None = None
    ) -> T | None: ...

    @abstractmethod
    def iter_keys(self, kls: type[T]) -> Iterator[Hashable]: ...

    @property
    @abstractmethod
    def collection(self) -> DependencyCollectionResolver: ...


class DependencyCollectionResolver(ABC):
    @abstractmethod
    def resolve(self, kls: type[T], *, key: Hashable | None = None) -> Iterable[T]: ...

    @abstractmethod
    def iter_keys(self, kls: type[T]) -> Iterator[Hashable]: ...


class DependencyNotFoundError(Exception):
    def __init__(self, kls: type[object], key: Hashable | None, msg: str) -> None:
        super().__init__(msg)

        self.kls = kls
        self.key = key


T_co = TypeVar("T_co", covariant=True)


class DependencyProvider(Protocol[T_co]):
    def __call__(self, resolver: DependencyResolver) -> T_co: ...


@final
class DependencyContainer(DependencyResolver):
    def __init__(self) -> None:
        self._registrations: dict[Hashable, _DependencyRegistration] = {}
        self._keys: dict[type[object], list[Hashable]] = defaultdict(list)
        self._collection_container = DependencyCollectionContainer(self)
        self._frozen = False

    def register(
        self,
        kls: type[T],
        provider: DependencyProvider[T],
        *,
        key: Hashable | None = None,
        singleton: bool = False,
    ) -> None:
        self._do_register(kls, key, _DependencyRegistration(provider, singleton))

    def register_type(
        self,
        kls: type[T],
        sub_kls: type[T] | None = None,
        *,
        key: Hashable | None = None,
        singleton: bool = False,
    ) -> None:
        if sub_kls is None:
            sub_kls = kls
        elif not issubclass(sub_kls, kls):
            raise ValueError(
                f"`sub_kls` must be a subclass of `kls`, but `{sub_kls}` is not a subclass of `{kls}`."
            )

        def create_instance(resolver: DependencyResolver) -> T:
            return wire_object(self, sub_kls)

        self._do_register(kls, key, _DependencyRegistration(create_instance, singleton))

    def register_instance(
        self, kls: type[T], obj: T, *, key: Hashable | None = None
    ) -> None:
        self._do_register(kls, key, _DependencyRegistration.for_(obj))

    def _do_register(
        self,
        kls: type[object],
        key: Hashable | None,
        registration: _DependencyRegistration,
    ) -> None:
        if self._frozen:
            raise InvalidOperationError(
                "No new objects can be registered after the first `resolve()` call."
            )

        full_key = (kls, key)

        if full_key in self._registrations:
            raise InvalidOperationError(f"`{kls}` has already a registered provider.")

        self._registrations[full_key] = registration

        self._keys[kls].append(key)

    @override
    def resolve(self, kls: type[T], *, key: Hashable | None = None) -> T:
        self._frozen = True

        full_key = (kls, key)

        registration = self._registrations.get(full_key)
        if registration is None:
            if key is None:
                msg = f"No registered provider found for `{kls}`."
            else:
                msg = f"No registered provider found for `{kls}` with key {key}. "

            raise DependencyNotFoundError(kls, key, msg)

        obj = registration.get_instance(kls, self)
        if obj is None:
            if key is None:
                msg = f"Registered provider for `{kls}` returned `None`."
            else:
                msg = f"Registered provider for `{kls}` with key {key} returned `None`."

            raise DependencyNotFoundError(kls, key, msg)

        return obj

    @override
    def resolve_optional(
        self, kls: type[T], *, key: Hashable | None = None
    ) -> T | None:
        try:
            return self.resolve(kls, key=key)
        except DependencyNotFoundError as ex:
            if ex.kls is kls and ex.key == key:
                return None

            raise

    @override
    def iter_keys(self, kls: type[T]) -> Iterator[Hashable]:
        keys = self._keys.get(kls)
        if keys is None:
            keys = []

        return iter(keys)

    @property
    @override
    def collection(self) -> DependencyCollectionContainer:
        return self._collection_container


@final
class DependencyCollectionContainer(DependencyCollectionResolver):
    def __init__(self, container: DependencyContainer) -> None:
        self._container = container
        self._registrations: dict[Hashable, list[_DependencyRegistration]] = defaultdict(list)  # fmt: skip
        self._keys: dict[type[object], list[Hashable]] = defaultdict(list)

    def register(
        self,
        kls: type[T],
        provider: DependencyProvider[T],
        *,
        key: Hashable | None = None,
        singleton: bool = False,
    ) -> None:
        self._do_register(kls, key, _DependencyRegistration(provider, singleton))

    def register_type(
        self,
        kls: type[T],
        sub_kls: type[T] | None = None,
        *,
        key: Hashable | None = None,
        singleton: bool = False,
    ) -> None:
        if sub_kls is None:
            sub_kls = kls
        elif not issubclass(sub_kls, kls):
            raise ValueError(
                f"`sub_kls` must be a subclass of `kls`, but `{sub_kls}` is not a subclass of `{kls}`."
            )

        def create_instance(resolver: DependencyResolver) -> T:
            return wire_object(self._container, sub_kls)

        self._do_register(kls, key, _DependencyRegistration(create_instance, singleton))

    def register_instance(
        self, kls: type[T], obj: T, *, key: Hashable | None = None
    ) -> None:
        self._do_register(kls, key, _DependencyRegistration.for_(obj))

    def _do_register(
        self,
        kls: type[object],
        key: Hashable | None,
        registration: _DependencyRegistration,
    ) -> None:
        if self._container._frozen:
            raise InvalidOperationError(
                "No new objects can be registered after the first `resolve()` call."
            )

        full_key = (kls, key)

        self._registrations[full_key].append(registration)

        self._keys[kls].append(key)

    @override
    def resolve(self, kls: type[T], *, key: Hashable | None = None) -> Iterable[T]:
        self._container._frozen = True

        full_key = (kls, key)

        registrations = self._registrations.get(full_key)
        if registrations is None:
            registrations = []

        return _DependencyCollection(self._container, kls, registrations)

    @override
    def iter_keys(self, kls: type[T]) -> Iterator[Hashable]:
        keys = self._keys.get(kls)
        if keys is None:
            keys = []

        return iter(keys)


class _DependencyCollection(Iterable[T]):
    def __init__(
        self,
        container: DependencyContainer,
        kls: type[T],
        registrations: list[_DependencyRegistration],
    ) -> None:
        self._container = container
        self._kls = kls
        self._registrations = registrations

    def __iter__(self) -> Iterator[T]:
        for registration in self._registrations:
            obj = registration.get_instance(self._kls, self._container)
            if obj is not None:
                yield obj


_NOT_SET: Final = object()


class _DependencyRegistration:
    @staticmethod
    def for_(obj: object) -> _DependencyRegistration:
        return _DependencyRegistration(provider=lambda resolver: obj)

    def __init__(
        self, provider: DependencyProvider[object], singleton: bool = False
    ) -> None:
        self.obj = _NOT_SET
        self.provider = provider
        self.singleton = singleton

    def get_instance(self, kls: type[T], resolver: DependencyResolver) -> T | None:
        if self.obj is _NOT_SET:
            obj = self.provider(resolver)

            if self.singleton:
                self.obj = obj
        else:
            obj = self.obj

        if obj is not None and not isinstance(obj, kls):
            raise InternalError(
                f"Object is expected to be of type `{kls}`, but is of type `{type(obj)}` instead."
            )

        return obj


@final
class DependencyLookup(Lookup[T]):
    def __init__(self, resolver: DependencyResolver, kls: type[T]) -> None:
        self._resolver = resolver
        self._kls = kls

    @override
    def maybe_get(self, key: Hashable) -> T | None:
        return self._resolver.resolve_optional(self._kls, key=key)

    @override
    def iter_keys(self) -> Iterator[Hashable]:
        return self._resolver.iter_keys(self._kls)

    @property
    @override
    def kls(self) -> type[T]:
        return self._kls


_resolver: DependencyResolver | None = None


def get_dependency_resolver() -> DependencyResolver:
    if _resolver is None:
        from fairseq2 import init_fairseq2

        init_fairseq2()

    if _resolver is None:
        raise InternalError("`_resolver` is `None`.")

    return _resolver


def wire_object(resolver: DependencyResolver, wire_kls: type[T], /, **kwargs: Any) -> T:
    obj = _create_auto_wired_instance(wire_kls, resolver, dict(kwargs))

    return cast(T, obj)


class AutoWireError(Exception):
    def __init__(self, kls: type[object], msg: str) -> None:
        super().__init__(msg)

        self.kls = kls


def _create_auto_wired_instance(
    kls: type[object], resolver: DependencyResolver, custom_kwargs: dict[str, object]
) -> object:
    init_method = getattr(kls, "__init__", None)
    if init_method is None:
        msg = f"`{kls}` must have an `__init__()` method for auto-wiring."

        raise AutoWireError(kls, msg)

    try:
        sig = signature(init_method)
    except (TypeError, ValueError) as ex:
        msg = f"Signature of `{init_method}` cannot be inspected."

        raise AutoWireError(kls, msg) from ex

    try:
        type_hints = get_type_hints(init_method)
    except (TypeError, ValueError, NameError) as ex:
        msg = f"Type annotations of `{init_method}` cannot be inspected."

        raise AutoWireError(kls, msg) from ex

    kwargs: dict[str, object] = {}

    for idx, (param_name, param) in enumerate(sig.parameters.items()):
        if idx == 0:  # i.e. self
            continue

        if param.kind == Parameter.POSITIONAL_ONLY:
            msg = f"`{init_method}` has one or more positional-only parameters."

            raise AutoWireError(kls, msg)

        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue

        try:
            param_type = type_hints[param_name]
        except KeyError:
            msg = f"`{param_name}` parameter of `{init_method}` has no type annotation."

            raise AutoWireError(kls, msg) from None

        arg: Any

        try:
            arg = custom_kwargs.pop(param_name)
        except KeyError:
            arg = _NOT_SET

        if arg is _NOT_SET:
            param_origin_type = get_origin(param_type)

            if param_origin_type in (Iterable, Collection, Sequence, Lookup, Lazy):
                param_type_args = get_args(param_type)
                if len(param_type_args) != 1:
                    msg = f"`{param_name}` parameter of `{init_method}` cannot be auto-wired. Its type annotation has no element type expression."

                    raise AutoWireError(kls, msg)

                element_type = param_type_args[0]
                if not isinstance(element_type, type):
                    if param.default != Parameter.empty:
                        continue

                    msg = f"`{param_name}` parameter of `{init_method}` cannot be auto-wired. The element type expression in its type annotation does not represent a `type`."

                    raise AutoWireError(kls, msg)

                if param_origin_type is Lookup:
                    arg = DependencyLookup(resolver, element_type)
                elif param_origin_type is Lazy:
                    arg = Lazy(factory=lambda: resolver.resolve(element_type))
                else:
                    arg = resolver.collection.resolve(element_type)

                    if param_origin_type is not Iterable:
                        arg = list(arg)
            elif param_origin_type is Callable:
                param_type_args = get_args(param_type)
                if len(param_type_args) != 2:
                    msg = f"`{param_name}` parameter of `{init_method}` cannot be auto-wired. Its type annotation has no signature expression."

                    raise AutoWireError(kls, msg)

                if len(param_type_args[0]) > 0:
                    if param.default != Parameter.empty:
                        continue

                    msg = f"`{param_name}` parameter of `{init_method}` cannot be auto-wired. Its type annotation has one or more parameter type expressions."

                    raise AutoWireError(kls, msg)

                return_type = param_type_args[1]
                if not isinstance(return_type, type):
                    if param.default != Parameter.empty:
                        continue

                    msg = f"`{param_name}` parameter of `{init_method}` cannot be auto-wired. The return type expression in its type annotation does not represent a `type`."

                    raise AutoWireError(kls, msg)

                arg = lambda: resolver.resolve(return_type)
            else:
                if not isinstance(param_type, type):
                    if param.default != Parameter.empty:
                        continue

                    msg = f"`{param_name}` parameter of `{init_method}` cannot be auto-wired. Its type annotation does not represent a `type`."

                    raise AutoWireError(kls, msg)

                if param_type is DependencyResolver:
                    arg = resolver
                elif param.default != Parameter.empty:
                    arg = resolver.resolve_optional(param_type)
                    if arg is None:
                        arg = _NOT_SET
                else:
                    arg = resolver.resolve(param_type)

        if arg is not _NOT_SET:
            kwargs[param_name] = arg

    if custom_kwargs:
        msg = f"`kwargs` has one or more extra arguments not used by `{kls}`. Extra arguments: {custom_kwargs}"

        raise AutoWireError(kls, msg)

    return kls(**kwargs)
