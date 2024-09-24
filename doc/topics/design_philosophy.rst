=================
Design Philosophy
=================

One of the core goals of fairseq2 is to make it possible for researchers to
explore new ideas and implement novel features without having to fork fairseq2.
Instead of having a monolithic repository that can only be modified by
copy-pasting large chunks of code, in fairseq2, all major APIs follow the
interface/implementation convention along with the `dependency inversion principle`__.
This means, each API has an *interface* (i.e. an abstract :class:`~abc.ABC`
class) that defines the contract of that API, and one or more concrete
implementations of that interface. Different implementations can be integrated
with the rest of fairseq2 via its lightweight `dependency injection API`__.

.. __: https://en.wikipedia.org/wiki/Dependency_inversion_principle
.. __: https://en.wikipedia.org/wiki/Dependency_injection

Interface/Implementation Convention
===================================

.. currentmodule:: fairseq2.nn

The diagram below shows the :doc:`position encoder API </reference/fairseq2.nn/position_encoders>`
as an example. The API is defined by the abstract :class:`PositionEncoder`
PyTorch module. :class:`SinusoidalPositionEncoder`, :class:`LearnedPositionEncoder`,
and :class:`RotaryEncoder` implement :class:`PositionEncoder` for their
respective algorithms. Technically, any of these position encoders can be used
wherever a :class:`PositionEncoder` is expected (see `Dependency Inversion`_
below).

.. image:: /static/img/position_encoder.svg
    :width: 580px
    :align: center
    :alt: Position Encoder Hierarchy

.. currentmodule:: fairseq2.data.text

When several implementations of an API share common logic, a typical pattern is
to have an intermediate abstract class, prefixed with ``Abstract``,  between the
interface and the concrete implementations.  For example, the :doc:`text tokenizer
API </reference/fairseq2.data.text/text_tokenizers>` has :class:`AbstractTextTokenizer`
that holds the common logic for :class:`SentencePieceTokenizer` and
:class:`TiktokenTokenizer`.

.. image:: /static/img/text_tokenizer.svg
    :width: 580px
    :align: center
    :alt: Text Tokenizer Hierarchy

Dependency Inversion
====================

.. currentmodule:: fairseq2.nn.transformer

The dependency inversion principle is critical to have a clean, well-tested, and
extensible API. The example below shows the (abbreviated) ``__init__()`` method
of the :class:`StandardTransformerDecoderLayer`::

    class StandardTransformerDecoderLayer(TransformerDecoderLayer):
        def __init__(
            self,
            self_attn: MultiheadAttention,
            encoder_decoder_attn: MultiheadAttention | None,
            ffn: FeedForwardNetwork
        ) -> None:
            ...

Instead of constructing the multihead attention and feed-forward network layers
within its ``__init__()`` method, :class:`StandardTransformerDecoderLayer`
expects the caller to provide instances of :class:`MultiheadAttention` and
:class:`FeedForwardNetwork` interfaces. This loose-coupling between an instance
and its dependencies enables composing diverse object graphs, such as different
model architectures, with minimal redundancy (i.e. code duplication).

Dependency Injection
====================

.. currentmodule:: fairseq2.dependency

With dependency inversion, instead of constructing their dependencies, objects
rely on their callers to provide them. This effectively requires callers to
build a dependency graph to construct objects. For simple objects, this does not
pose a problem, but for objects with large dependency closures, manual graph
construction can become a tedious task.

fairseq2 offers a lightweight :mod:`dependency injection API <fairseq2.dependency>`
to reduce the complexity of building dependency graphs. The core piece of the
API is the :class:`DependencyContainer` interface. An implementation of it, such
as :class:`StandardDependencyContainer`, is expected that hold a lazily-initialized
dependency graph. The interface exposes two main methods; :meth:`~DependencyContainer.register_factory`
and :meth:`~DependencyContainer.resolve`. :meth:`~DependencyContainer.register_factory`
is used to register new objects with the container and :meth:`~DependencyResolver.resolve`
is used to *resolve* objects, meaning to create objects along with their
transitive dependencies by traversing the graph.

Basics
^^^^^^

The example below shows how an object of type ``Foo`` can be registered and
later resolved with a container::

    from fairseq2.dependency import DependencyResolver, StandardDependencyContainer

    container = StandardDependencyContainer()

    class Foo:
        pass

    def create_foo(resolver: DependencyResolver) -> Foo:
        return Foo()

    container.register_factory(Foo, create_foo)

    obj = container.resolve(Foo)

    assert isinstance(obj, Foo)

As arguments, :meth:`~DependencyContainer.register_factory` expects the type of the
object and a callable responsible for creating the object when called. The
callable is passed a :class:`DependencyResolver` instance that it can use to
resolve the dependencies of the newly-created object. The object returned by the
callable is cached. This means, after the first :meth:`~DependencyResolver.resolve`
call, the secondary calls will return the same instance making the object
effectively a singleton.

Since ``Foo`` is a lightweight class with no dependencies, :meth:`~DependencyContainer.register_instance`
can be used instead of :meth:`~DependencyContainer.register_factory`. Unlike
:meth:`~DependencyContainer.register_factory`, which expects a callable to lazily
initialize the object, :meth:`~DependencyContainer.register_instance` stores the
passed object directly in the container::

    from fairseq2.dependency import StandardDependencyContainer

    container = StandardDependencyContainer()

    class Foo:
        pass

    container.register_instance(Foo, Foo())

    obj = container.resolve(Foo)

    assert isinstance(obj, Foo)

Both :meth:`~DependencyContainer.register_factory` and :meth:`~DependencyContainer.register_instance`
also accept an optional key argument. When provided, the object will be
registered along with the key and will be resolved only when the same key is
passed to :meth:`~DependencyResolver.resolve`::

    from fairseq2.dependency import StandardDependencyContainer

    container = StandardDependencyContainer()

    class Foo:
        pass

    container.register_instance(Foo, Foo(), key="foo")

    obj = container.resolve(Foo, key="foo")

    assert isinstance(obj, Foo)

:meth:`~DependencyResolver.resolve` will raise a :class:`DependencyError` when an
object cannot be found. As an alternative, :meth:`~DependencyResolver.resolve_optional`
can be used which returns ``None`` instead::

    from fairseq2.dependency import StandardDependencyContainer

    container = StandardDependencyContainer()

    class Foo:
        pass

    obj = container.resolve_optional(Foo)

    assert obj is None

Registering Multiple Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When :meth:`~DependencyContainer.register_factory` or :meth:`~DependencyContainer.register_instance`
is called multiple times with the same type and, optionally, key,
:meth:`~DependencyResolver.resolve` will return only the last registered object::

    from fairseq2.dependency import StandardDependencyContainer

    container = StandardDependencyContainer()

    class Foo:
        pass

    foo1 = Foo()
    foo2 = Foo()

    container.register_instance(Foo, foo1)
    container.register_instance(Foo, foo2)

    obj = container.resolve(Foo)

    assert obj is foo2

If, not just the last registered object, but all registered objects of certain
type are needed, :meth:`~DependencyResolver.resolve_all` can be used. It returns
all non-keyed objects in the order they were registered as an iterable::

    from fairseq2.dependency import StandardDependencyContainer

    container = StandardDependencyContainer()

    class Foo:
        pass

    foo1 = Foo()
    foo2 = Foo()

    container.register_instance(Foo, foo1)
    container.register_instance(Foo, foo2)

    itr = container.resolve_all(Foo)

    assert next(itr) is foo1
    assert next(itr) is foo2

:meth:`~DependencyResolver.resolve_all_keyed` is similar to :meth:`~DependencyResolver.resolve_all`,
but works for keyed objects. It returns them as key-object pairs in the order
they were registered as an iterable::

    from fairseq2.dependency import StandardDependencyContainer

    container = StandardDependencyContainer()

    class Foo:
        pass

    foo1 = Foo()
    foo2 = Foo()

    container.register_instance(Foo, foo1, key="foo1")
    container.register_instance(Foo, foo2, key="foo2")

    itr = container.resolve_all(Foo)

    key1, obj1 = next(itr)
    key2, obj2 = next(itr)

    assert key1 == "foo1"
    assert key2 == "foo2"

    assert obj1 is foo1
    assert obj2 is foo2

A More Complete Example
=======================

The example below is slightly more complex and shows how an object with a
dependency can be registered to the container. It also demonstrates the use of
interfaces as registration types::

    from abc import ABC, abstractmethod

    from fairseq2.dependency import StandardDependencyContainer

    container = StandardDependencyContainer()

    # `Writer` Interface
    class Writer(ABC):
        @abstractmethod
        def write(self, s: str) -> None:
            ...

    # `Writer` Implementation
    class StdOutWriter(Bar):
        def write(self, s: str) -> None:
            print(s)

    # `Foo` Interface
    class Foo(ABC):
        @abstractmethod
        def write_foo(self) -> None:
            ...

    # `Foo` Implementation
    class FooImpl(Foo):
        # depends on a `Writer` instance.
        def __init__(self, writer: Writer) -> None:
            self.writer = writer

        def write_foo(self) -> None:
            self.writer.write("foo")

    # Registers `Writer`.
    container.register_instance(Writer, StdOutWriter())

    def create_foo(resolver: DependencyResolver) -> Foo:
        # Resolves the registered `Writer` object from the container.
        writer = resolver.resolve(Writer)

        return FooImpl(writer)

    # Registers `Foo`.
    container.register_factory(Foo, create_foo)

    # Internally calls `create_foo` to create the `Foo` instance.
    foo1 = container.resolve(Foo)

    assert foo1 is FooImpl

    # Prints "FooImpl" to stdout.
    foo1.write_foo()

    # Secondary calls return the same object.
    foo2 = container.resolve(Foo)

    assert foo1 is foo2

Using fairseq2 Container
^^^^^^^^^^^^^^^^^^^^^^^^

The examples above all used a newly-created :class:`StandardDependencyContainer`
for demonstration purposes. In real-world, objects are typically registered with
fairseq2's global container. There are two ways to access it:

- For a Python package, an extension function can be used to make the objects of
  the package available to all fairseq2 users.  See :doc:`runtime_extensions`
  for details.
- In a plain Python script, :func:`get_container` can be used. :func:`~fairseq2.setup_fairseq2`
  has to be called first though which is responsible for initializing the global
  container.

::

    from fairseq2 import setup_fairseq2
    from fairseq2.assets import AssetStore
    from fairseq2.dependency import get_container

    # Must be called before any other fairseq2 calls.
    setup_fairseq2()

    # Returns the global `DependencyContainer` initialized by `setup_fairseq2`.
    container = get_container()

    # Resolves the registered `AssetStore` object, which is by default an
    # instance of `StandardAssetStore`.
    store = container.resolve(AssetStore)
