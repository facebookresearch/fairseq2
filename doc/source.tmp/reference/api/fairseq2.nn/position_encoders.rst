============================
Position Encoders
============================

.. currentmodule:: fairseq2.nn

A set of PyTorch modules to encode sequences with positional information.

**ABCs**

* :class:`PositionEncoder`

**Classes**

* :class:`SinusoidalPositionEncoder`
* :class:`LearnedPositionEncoder`
* :class:`RotaryEncoder`

ABCs
====

.. autoclass:: PositionEncoder

    .. autoclasstree:: fairseq2.nn.PositionEncoder fairseq2.nn.SinusoidalPositionEncoder fairseq2.nn.LearnedPositionEncoder fairseq2.nn.RotaryEncoder
        :full:

Classes
=======


.. autoclass:: SinusoidalPositionEncoder


    .. autoclasstree:: fairseq2.nn.SinusoidalPositionEncoder
        :full:

    The positional encodings are initialized as in tensor2tensor which differs
    slightly from the description in section 3.5 of
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`. This means instead of:

    .. math::

        PE_{(pos, 2i)}   = \text{sin}(pos/10000^{2i/d_{model}})

        PE_{(pos, 2i+1)} = \text{cos}(pos/10000^{2i/d_{model}})

    we use:

    .. math::

        PE_{(pos, i)} = \text{sin}(pos/10000^{i/d_{model}})\;\text{for}\;i\;   <\frac{d_{model}}{2}

        PE_{(pos, i)} = \text{cos}(pos/10000^{i/d_{model}})\;\text{for}\;i\;\geq\frac{d_{model}}{2}

    See `here <https://github.com/tensorflow/tensor2tensor/pull/177>`_ for more
    information.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
    >>>
    >>> m = SinusoidalPositionEncoder(encoding_dim=4, max_seq_len=16)
    >>>
    >>> seqs = torch.ones((3, 4))
    >>>
    >>> m(seqs)
    tensor([[ 1.0000e+00,  1.0000e+00,  2.0000e+00,  2.0000e+00],  # pos 0
            [ 9.4147e-01,  2.0000e-04,  6.4030e-01,  2.0000e+00],  # pos 1
            [ 1.0930e-02,  3.0000e-04, -5.1615e-01,  2.0000e+00]]) # pos 2


.. autoclass:: LearnedPositionEncoder

    .. autoclasstree:: fairseq2.nn.LearnedPositionEncoder
        :full:

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.position_encoder import LearnedPositionEncoder
    >>>
    >>> m = LearnedPositionEncoder(encoding_dim=4, max_seq_len=16)
    >>>
    >>> seqs = torch.ones((3, 4))
    >>>
    >>> m(seqs)
    tensor([[ 1.1135,  0.5548,  0.4293,  2.0112],                               # pos 0
            [ 0.2364,  0.6009,  3.3865, -2.4810],                               # pos 1
            [-0.4746,  0.4544,  0.2761,  0.8828]], grad_fn=<SqueezeBackward1>)  # pos 2




.. autoclass:: RotaryEncoder

    .. autoclasstree:: fairseq2.nn.RotaryEncoder
        :full:
