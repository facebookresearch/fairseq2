=================
Design Philosophy
=================

One of the core goals of fairseq2 is to make it possible for researchers to
explore new ideas and implement novel features without having to fork fairseq2.
Instead of having a monolithic repository that can only be modified by
copy-pasting large chunks of code, in fairseq2, all major APIs follow the
interface/implementation convention along with `dependency inversion principle`__.
This means, each API has an *interface* (i.e. an abstract :class:`~abc.ABC`
class) that defines the contract of that API, and one or more concrete
implementations of that interface. Different implementations can be integrated
with the rest of fairseq2 via its lightweight `dependency injection`__ API.

.. __: https://en.wikipedia.org/wiki/Dependency_inversion_principle
.. __: https://en.wikipedia.org/wiki/Dependency_injection

Interface/Implementation Convention
===================================

.. currentmodule:: fairseq2.nn

The diagram below shows the :doc:`position encoder API </reference/fairseq2.nn/position_encoders>`
as an example. The API is defined by the abstract :class:`PositionEncoder`
PyTorch module. :class:`SinusoidalPositionEncoder`, :class:`LearnedPositionEncoder`,
and :class:`RotaryEncoder` implement :class:`PositionEncoder` for their
respective algorithms.

.. currentmodule:: fairseq2.data.text

When several implementations of an API share common logic, a typical pattern is
to have an intermediate abstract class, prefixed with ``Abstract``,  between the
interface and the concrete implementations.  For example, the :doc:`text tokenizer
API </reference/fairseq2.data.text/text_tokenizers>` has :class:`AbstractTextTokenizer`
that holds the common logic for :class:`SentencePieceTokenizer` and
:class:`TiktokenTokenizer`.


Dependency Inversion
====================

The dependency inversion principle is critical to have a clean, well-tested, and
extensible API. The example below shows the ``__init__()`` method of the
:class:`StandardTransformerDecoderLayer`::

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
and its dependencies enables composing diverse object hierarchies, such as
different model architectures, with minimal redundancy (i.e. code duplication).

Dependency Injection
====================



Runtime Discovery of Dependencies
=================================
