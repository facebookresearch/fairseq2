.. _position_encoder:

fairseq2.nn.PositionEncoder
===========================

.. currentmodule:: fairseq2.nn

The diagram below shows the :doc:`position encoder API </reference/fairseq2.nn.PositionEncoder>`
as an example. The API is defined by the abstract :class:`PositionEncoder`
PyTorch module. :class:`SinusoidalPositionEncoder`, :class:`LearnedPositionEncoder`,
and :class:`RotaryEncoder` implement :class:`PositionEncoder` for their
respective algorithms. Technically, any of these position encoders can be used
wherever a :class:`PositionEncoder` is expected.

.. mermaid::

    classDiagram
        class Module {
            <<torch.nn.Module>>
            +parameters()
            +forward()*
            +train()
            +eval()
        }

        class PositionEncoder {
            <<abstract>>
            +encoding_dim: int
            +forward(seqs, seqs_layout, state_bag)*
        }

        class SinusoidalPositionEncoder {
            +max_seq_len: int
            +sin_offset: int
            +freqs: Tensor
            +forward(seqs, seqs_layout, state_bag)
            +reset_parameters()
        }

        class LearnedPositionEncoder {
            +max_seq_len: int
            +weight: Parameter
            +forward(seqs, seqs_layout, state_bag)
            +reset_parameters()
        }

        class RotaryEncoder {
            +max_seq_len: int
            +theta: float
            +freqs: Tensor
            +forward(seqs, seqs_layout, state_bag)
            +reset_parameters()
        }

        class ReferenceRotaryEncoder {
            +max_seq_len: int
            +theta: float
            +cos_freqs: Tensor
            +sin_freqs: Tensor
            +forward(seqs, seqs_layout, state_bag)
            +reset_parameters()
        }

        Module <|-- PositionEncoder
        PositionEncoder <|-- SinusoidalPositionEncoder
        PositionEncoder <|-- LearnedPositionEncoder
        PositionEncoder <|-- RotaryEncoder
        PositionEncoder <|-- ReferenceRotaryEncoder


.. autoclass:: PositionEncoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SinusoidalPositionEncoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LearnedPositionEncoder
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RotaryEncoder
   :members:
   :undoc-members:
   :show-inheritance:

**Example Usage:**

.. code-block:: python

    from fairseq2.nn import SinusoidalPositionEncoder, RotaryEncoder

    # Sinusoidal position encoding
    pos_encoder = SinusoidalPositionEncoder(
        encoding_dim=512,
        max_seq_len=2048
    )

    # Use with BatchLayout for proper position handling
    seqs = torch.randn(4, 6, 512)  # (batch, seq, features)
    batch_layout = BatchLayout.of(seqs, seq_lens=[4, 2, 3, 5])

    pos_encodings = pos_encoder(seqs, seqs_layout=batch_layout)
