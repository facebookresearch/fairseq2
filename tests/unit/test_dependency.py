# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest
from typing_extensions import override

from fairseq2.dependency import (
    DependencyError,
    DependencyNotFoundError,
    DependencyResolver,
    StandardDependencyContainer,
)


class Foo(ABC):
    @abstractmethod
    def foo(self) -> int:
        ...


class FooImpl1(Foo):
    @override
    def foo(self) -> int:
        return 1


class FooImpl2(Foo):
    @override
    def foo(self) -> int:
        return 2


class Bar:
    pass


class TestStandardDependencyContainer:
    def test_register_raises_error_when_subkls_is_not_subclass_of_kls(self) -> None:
        container = StandardDependencyContainer()

        with pytest.raises(
            ValueError, match=rf"^`sub_kls` must be a subclass of `kls`, but `{Bar}` is not a subclass of `{Foo}`\.$"  # fmt: skip
        ):
            container.register(Foo, Bar)

    def test_register_factory_resolve_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo = FooImpl1()

        def create_foo(resolver: DependencyResolver) -> Foo:
            return expected_foo

        container.register_factory(Foo, create_foo)

        foo1 = container.resolve(Foo)
        foo2 = container.resolve_optional(Foo)

        assert foo1 is expected_foo
        assert foo2 is expected_foo

    def test_keyed_register_factory_resolve_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo = FooImpl1()

        def create_foo(resolver: DependencyResolver) -> Foo:
            return expected_foo

        container.register_factory(Foo, create_foo, key="foo")

        foo1 = container.resolve(Foo, "foo")
        foo2 = container.resolve_optional(Foo, "foo")

        assert foo1 is expected_foo
        assert foo2 is expected_foo

    def test_register_instance_resolve_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo = FooImpl1()

        container.register_instance(Foo, expected_foo)

        foo1 = container.resolve(Foo)
        foo2 = container.resolve_optional(Foo)

        assert foo1 is expected_foo
        assert foo2 is expected_foo

    def test_keyed_register_instance_resolve_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo = FooImpl1()

        container.register_instance(Foo, expected_foo, key="foo")

        foo1 = container.resolve(Foo, "foo")
        foo2 = container.resolve_optional(Foo, "foo")

        assert foo1 is expected_foo
        assert foo2 is expected_foo

    def test_repeated_register_resolve_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo = FooImpl2()

        def create_foo1(resolver: DependencyResolver) -> Foo:
            return FooImpl1()

        def create_foo2(resolver: DependencyResolver) -> Foo:
            return expected_foo

        container.register_instance(Foo, FooImpl1())

        container.register_factory(Foo, create_foo1)
        container.register_factory(Foo, create_foo2)

        foo1 = container.resolve(Foo)
        foo2 = container.resolve_optional(Foo)

        assert foo1 is expected_foo
        assert foo2 is expected_foo

    def test_repeated_keyed_register_resolve_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo = FooImpl2()

        def create_foo1(resolver: DependencyResolver) -> Foo:
            return FooImpl1()

        def create_foo2(resolver: DependencyResolver) -> Foo:
            return expected_foo

        container.register_instance(Foo, FooImpl1(), key="foo")

        container.register_factory(Foo, create_foo1, key="foo")
        container.register_factory(Foo, create_foo2, key="foo")

        foo1 = container.resolve(Foo, key="foo")
        foo2 = container.resolve_optional(Foo, key="foo")

        assert foo1 is expected_foo
        assert foo2 is expected_foo

    def test_unkeyed_keyed_register_resolve_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo1 = FooImpl1()
        expected_foo2 = FooImpl2()

        def create_foo1(resolver: DependencyResolver) -> Foo:
            return expected_foo1

        def create_foo2(resolver: DependencyResolver) -> Foo:
            return expected_foo2

        container.register_factory(Foo, create_foo1)
        container.register_factory(Foo, create_foo2, key="foo")

        foo1 = container.resolve(Foo)
        foo2 = container.resolve(Foo, key="foo")

        assert foo1 is expected_foo1
        assert foo2 is expected_foo2

    def test_resolve_raises_error_when_type_is_invalid(self) -> None:
        container = StandardDependencyContainer()

        def create_foo(resolver: DependencyResolver) -> str:
            return "foo"

        container.register_factory(Foo, create_foo)

        with pytest.raises(
            DependencyError, match=rf"^The object in the container is expected to be of type `{Foo}`, but is of type `{str}` instead\. Please file a bug report\.$"  # fmt: skip
        ):
            container.resolve(Foo)

    def test_keyed_resolve_raises_error_when_type_is_invalid(self) -> None:
        container = StandardDependencyContainer()

        def create_foo(resolver: DependencyResolver) -> str:
            return "foo"

        container.register_factory(Foo, create_foo, key="foo")

        with pytest.raises(
            DependencyError, match=rf"^The object in the container is expected to be of type `{Foo}`, but is of type `{str}` instead\. Please file a bug report\.$"  # fmt: skip
        ):
            container.resolve(Foo, "foo")

    def test_resolve_raises_error_when_dependency_is_not_registered(self) -> None:
        container = StandardDependencyContainer()

        with pytest.raises(
            DependencyNotFoundError, match=rf"^No registered factory or object found for `{Foo}`\.$"  # fmt: skip
        ):
            container.resolve(Foo)

    def test_keyed_resolve_raises_error_when_dependency_is_not_registered(self) -> None:
        container = StandardDependencyContainer()

        with pytest.raises(
            DependencyNotFoundError, match=rf"^No registered factory or object found for `{Foo}` with the key 'foo'\.$"  # fmt: skip
        ):
            container.resolve(Foo, "foo")

    def test_resolve_raises_error_when_dependency_is_none(self) -> None:
        container = StandardDependencyContainer()

        def create_foo(resolver: DependencyResolver) -> None:
            return None

        container.register_factory(Foo, create_foo)

        with pytest.raises(
            DependencyNotFoundError, match=rf"^The registered factory for `{Foo}` returned `None`\.$"  # fmt: skip
        ):
            container.resolve(Foo)

    def test_keyed_resolve_raises_error_when_dependency_is_none(self) -> None:
        container = StandardDependencyContainer()

        def create_foo(resolver: DependencyResolver) -> None:
            return None

        container.register_factory(Foo, create_foo, key="foo")

        with pytest.raises(
            DependencyNotFoundError, match=rf"^The registered factory for `{Foo}` with the key 'foo' returned `None`\.$"  # fmt: skip
        ):
            container.resolve(Foo, "foo")

    def test_resolve_optional_works_when_dependency_is_not_registered(self) -> None:
        container = StandardDependencyContainer()

        foo = container.resolve_optional(Foo)

        assert foo is None

    def test_keyed_resolve_optional_works_when_dependency_is_not_registered(
        self,
    ) -> None:
        container = StandardDependencyContainer()

        foo = container.resolve_optional(Foo, "foo")

        assert foo is None

    def test_resolve_optional_works_when_dependency_is_none(self) -> None:
        container = StandardDependencyContainer()

        def create_foo(resolver: DependencyResolver) -> None:
            return None

        container.register_factory(Foo, create_foo)

        foo = container.resolve_optional(Foo)

        assert foo is None

    def test_keyed_resolve_optional_works_when_dependency_is_none(self) -> None:
        container = StandardDependencyContainer()

        def create_foo(resolver: DependencyResolver) -> None:
            return None

        container.register_factory(Foo, create_foo)

        foo = container.resolve_optional(Foo, "foo")

        assert foo is None

    def test_dependency_factory_works(self) -> None:
        container = StandardDependencyContainer()

        def create_foo(resolver: DependencyResolver) -> Foo:
            assert resolver is container

            return FooImpl1()

        container.register_factory(Foo, create_foo)

        container.resolve(Foo)

    def test_register_resolve_all_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo1 = FooImpl1()
        expected_foo2 = FooImpl2()

        def create_foo1(resolver: DependencyResolver) -> Foo:
            return expected_foo1

        def create_foo2(resolver: DependencyResolver) -> Foo:
            return expected_foo2

        def create_foo3(resolver: DependencyResolver) -> None:
            return None

        container.register_factory(Foo, create_foo1)
        container.register_factory(Foo, create_foo3)
        container.register_factory(Foo, create_foo2)

        foos = list(container.resolve_all(Foo))

        assert len(foos) == 2

        assert foos[0] is expected_foo1
        assert foos[1] is expected_foo2

    def test_keyed_register_resolve_all_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo1 = FooImpl1()
        expected_foo2 = FooImpl2()

        def create_foo1(resolver: DependencyResolver) -> Foo:
            return expected_foo1

        def create_foo2(resolver: DependencyResolver) -> Foo:
            return expected_foo2

        def create_foo3(resolver: DependencyResolver) -> None:
            return None

        container.register_factory(Foo, create_foo1, key="foo1")
        container.register_factory(Foo, create_foo3, key="foo3")
        container.register_factory(Foo, create_foo2, key="foo2")

        foos = dict(container.resolve_all_keyed(Foo))

        assert len(foos) == 2

        assert foos["foo1"] is expected_foo1
        assert foos["foo2"] is expected_foo2

    def test_register_resolve_all_works_when_no_dependency_is_registered(self) -> None:
        container = StandardDependencyContainer()

        foos = list(container.resolve_all(Foo))

        assert len(foos) == 0

    def test_keyed_register_resolve_all_works_when_no_dependency_is_registered(
        self,
    ) -> None:
        container = StandardDependencyContainer()

        foos = dict(container.resolve_all_keyed(Foo))

        assert len(foos) == 0

    def test_unkeyed_keyed_register_resolve_all_works(self) -> None:
        container = StandardDependencyContainer()

        expected_foo1 = FooImpl1()
        expected_foo2 = FooImpl2()
        expected_foo3 = FooImpl1()
        expected_foo4 = FooImpl2()

        def create_foo1(resolver: DependencyResolver) -> Foo:
            return expected_foo1

        def create_foo2(resolver: DependencyResolver) -> Foo:
            return expected_foo2

        def create_foo3(resolver: DependencyResolver) -> Foo:
            return expected_foo3

        def create_foo4(resolver: DependencyResolver) -> Foo:
            return expected_foo4

        container.register_factory(Foo, create_foo1)
        container.register_factory(Foo, create_foo2, key="foo2")
        container.register_factory(Foo, create_foo3)
        container.register_factory(Foo, create_foo4, key="foo4")

        list_foos = list(container.resolve_all(Foo))

        assert len(list_foos) == 2

        assert list_foos[0] is expected_foo1
        assert list_foos[1] is expected_foo3

        dict_foos = dict(container.resolve_all_keyed(Foo))

        assert len(dict_foos) == 2

        assert dict_foos["foo2"] is expected_foo2
        assert dict_foos["foo4"] is expected_foo4

    def test_register_after_resolve_raises_error(self) -> None:
        container = StandardDependencyContainer()

        def create_foo1(resolver: DependencyResolver) -> Foo:
            return FooImpl1()

        def create_foo2(resolver: DependencyResolver) -> Foo:
            return FooImpl2()

        container.register_factory(Foo, create_foo1)

        container.resolve(Foo)

        with pytest.raises(
            RuntimeError, match=r"^No new objects can be registered after the first `resolve\(\)` call\.$"  # fmt: skip
        ):
            container.register_factory(Foo, create_foo2)
