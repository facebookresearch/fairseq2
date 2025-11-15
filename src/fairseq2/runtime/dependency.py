# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Collection, Hashable, Iterable, Iterator, Sequence
from functools import partial
from inspect import Parameter, signature
from types import NoneType, UnionType
from typing import (
    Any,
    Final,
    Protocol,
    TypeVar,
    Union,
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
    def __init__(self, kls: type, key: Hashable | None, msg: str) -> None:
        super().__init__(msg)

        self.kls = kls
        self.key = key


class DependencyCycleError(Exception):
    def __init__(self, cycle: Sequence[tuple[type, Hashable | None]]) -> None:
        super().__init__("Dependency cycle detected.")

        self.cycle = cycle

    def __str__(self) -> str:
        s = " -> ".join(str(t) if k is None else f"{t} ({k})" for t, k in self.cycle)

        return f"Dependency cycle detected. {s}"


T_co = TypeVar("T_co", covariant=True)


class DependencyProvider(Protocol[T_co]):
    def __call__(self, resolver: DependencyResolver) -> T_co: ...


@final
class DependencyContainer(DependencyResolver):
    def __init__(self) -> None:
        self._registrations: dict[Hashable, _DependencyRegistration] = {}
        self._keys: dict[type, list[Hashable]] = defaultdict(list)
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
        self._do_register(kls, key, _DependencyRegistration(provider, key, singleton))

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

        self._do_register(
            kls, key, _DependencyRegistration(create_instance, key, singleton)
        )

    def register_instance(
        self, kls: type[T], obj: T, *, key: Hashable | None = None
    ) -> None:
        self._do_register(kls, key, _DependencyRegistration.for_(obj, key))

    def _do_register(
        self,
        kls: type,
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
        self._keys: dict[type, list[Hashable]] = defaultdict(list)

    def register(
        self,
        kls: type[T],
        provider: DependencyProvider[T],
        *,
        key: Hashable | None = None,
        singleton: bool = False,
    ) -> None:
        self._do_register(kls, key, _DependencyRegistration(provider, key, singleton))

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

        self._do_register(
            kls, key, _DependencyRegistration(create_instance, key, singleton)
        )

    def register_instance(
        self, kls: type[T], obj: T, *, key: Hashable | None = None
    ) -> None:
        self._do_register(kls, key, _DependencyRegistration.for_(obj, key))

    def _do_register(
        self,
        kls: type,
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
    def for_(obj: object, key: Hashable | None) -> _DependencyRegistration:
        return _DependencyRegistration(lambda resolver: obj, key, singleton=False)

    def __init__(
        self,
        provider: DependencyProvider[object],
        key: Hashable | None,
        singleton: bool,
    ) -> None:
        self.obj = _NOT_SET
        self.provider = provider
        self.key = key
        self.singleton = singleton
        self.in_call = False

    def get_instance(self, kls: type[T], resolver: DependencyResolver) -> T | None:
        if self.obj is _NOT_SET:
            if self.in_call:
                raise DependencyCycleError(cycle=[(kls, self.key)])

            self.in_call = True

            try:
                obj = self.provider(resolver)
            except DependencyCycleError as ex:
                cycle = [(kls, self.key)]

                cycle.extend(ex.cycle)

                raise DependencyCycleError(cycle) from None
            finally:
                self.in_call = False

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
    obj = _create_wired_instance(wire_kls, resolver, dict(kwargs))

    return cast(T, obj)


class AutoWireError(Exception):
    def __init__(self, kls: type, reason: str) -> None:
        super().__init__(f"`{kls}` cannot be auto-wired. {reason}")

        self.kls = kls
        self.reason = reason


def _create_wired_instance(
    kls: type, resolver: DependencyResolver, custom_kwargs: dict[str, object]
) -> object:
    def wire_error(reason: str) -> Exception:
        return AutoWireError(kls, reason)

    init_method = getattr(kls, "__init__", None)
    if init_method is None:
        raise wire_error("Must have an `__init__()` for auto-wiring.")

    try:
        sig = signature(init_method)
    except (TypeError, ValueError) as ex:
        raise wire_error("Signature of `__init__()` cannot be inspected.") from ex

    try:
        type_hints = get_type_hints(init_method)
    except (TypeError, ValueError, NameError) as ex:
        raise wire_error(
            "Type annotations of `__init__()` cannot be inspected."
        ) from ex

    kwargs: dict[str, object] = {}

    for idx, (param_name, param) in enumerate(sig.parameters.items()):
        if idx == 0:  # i.e. self
            continue

        if param.kind == Parameter.POSITIONAL_ONLY:
            raise wire_error("`__init__()` has one or more positional-only parameters.")

        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue

        try:
            param_type = type_hints[param_name]
        except KeyError:
            raise wire_error(
                f"`{param_name}` parameter has no type annotation."
            ) from None

        arg: Any

        try:
            arg = custom_kwargs.pop(param_name)
        except KeyError:
            arg = _NOT_SET

        if arg is _NOT_SET:

            def non_type_param_error() -> Exception:
                return wire_error(
                    f"Type annotation of the `{param_name}` parameter does not represent a `type`."
                )

            def get_element_kls() -> type | None:
                param_type_args = get_args(param_type)
                if len(param_type_args) != 1:
                    raise wire_error(
                        f"Type annotation of the `{param_name}` parameter has no valid element type expression."
                    )

                element_kls = param_type_args[0]
                if not isinstance(element_kls, type):
                    if param.default != Parameter.empty:
                        return None

                    raise wire_error(
                        f"Element type expression of the `{param_name}` parameter does not represent a `type`."
                    )

                return element_kls

            def get_return_kls() -> type | None:
                param_type_args = get_args(param_type)
                if len(param_type_args) != 2:
                    raise wire_error(
                        f"Type annotation of the `{param_name}` parameter has no valid signature expression."
                    )

                if len(param_type_args[0]) > 0:
                    if param.default != Parameter.empty:
                        return None

                    raise wire_error(
                        f"Type annotation of the `{param_name}` parameter has one or more parameter type expressions."
                    )

                return_kls = param_type_args[1]
                if not isinstance(return_kls, type):
                    if param.default != Parameter.empty:
                        return None

                    raise wire_error(
                        f"Return type expression of the `{param_name}` parameter does not represent a `type`."
                    )

                return return_kls

            def get_optional_kls() -> type | None:
                param_type_args = get_args(param_type)
                if len(param_type_args) != 2:
                    raise non_type_param_error()

                if param_type_args[0] is NoneType:
                    element_kls = param_type_args[1]
                elif param_type_args[1] is NoneType:
                    element_kls = param_type_args[0]
                else:
                    if param.default != Parameter.empty:
                        return None

                    raise non_type_param_error()

                if not isinstance(element_kls, type):
                    if param.default != Parameter.empty:
                        return None

                    raise non_type_param_error()

                return element_kls

            def get_param_kls() -> type | None:
                if not isinstance(param_type, type):
                    if param.default != Parameter.empty:
                        return None

                    raise non_type_param_error()

                return param_type

            param_origin_type = get_origin(param_type)

            if param_origin_type is Iterable:
                element_kls = get_element_kls()
                if element_kls is None:
                    continue

                arg = resolver.collection.resolve(element_kls)
            elif param_origin_type in (Collection, Sequence):
                element_kls = get_element_kls()
                if element_kls is None:
                    continue

                arg = resolver.collection.resolve(element_kls)

                arg = list(arg)
            elif param_origin_type is Lookup:
                element_kls = get_element_kls()
                if element_kls is None:
                    continue

                arg = DependencyLookup(resolver, element_kls)
            elif param_origin_type is Lazy:
                element_kls = get_element_kls()
                if element_kls is None:
                    continue

                arg = Lazy(factory=partial(lambda e: resolver.resolve(e), element_kls))
            elif param_origin_type is Callable:
                return_kls = get_return_kls()
                if return_kls is None:
                    continue

                arg = partial(lambda r: resolver.resolve(r), return_kls)
            elif param_origin_type in (Union, UnionType):
                element_kls = get_optional_kls()
                if element_kls is None:
                    continue

                arg = resolver.resolve_optional(element_kls)
            else:
                param_kls = get_param_kls()
                if param_kls is None:
                    continue

                if param_kls is DependencyResolver:
                    arg = resolver
                elif param.default != Parameter.empty:
                    arg = resolver.resolve_optional(param_kls)
                    if arg is None:
                        arg = _NOT_SET
                else:
                    arg = resolver.resolve(param_kls)

        if arg is not _NOT_SET:
            kwargs[param_name] = arg

    if custom_kwargs:
        raise wire_error(
            f"`kwargs` has one or more extra arguments not used. Extra arguments: {custom_kwargs}"
        )

    return kls(**kwargs)
