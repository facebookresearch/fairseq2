.. _api-models-olmo:

====================
fairseq2.models.olmo
====================

.. currentmodule:: fairseq2.models.olmo

The OLMo module provides support for OLMo2 and OLMo3 language models from the
Allen Institute for AI. It includes model configurations, hub access, tokenizers,
and utilities for loading and working with OLMo models.

Quick Start
-----------

.. code-block:: python

    from fairseq2.models.olmo import get_olmo_model_hub, load_olmo_tokenizer

    # Get the model hub
    hub = get_olmo_model_hub()

    # List available architectures
    for arch in sorted(hub.get_archs()):
        print(f"  - {arch}")

    # Load a model
    model = hub.load_model("olmo2_7b")

    # Load corresponding tokenizer
    tokenizer = load_olmo_tokenizer("olmo2_7b")

Available Models
----------------

**OLMo2 Series** — standard causal attention, 4K context:

- ``olmo2_1b`` - 1B parameters
- ``olmo2_7b`` - 7B parameters
- ``olmo2_13b`` - 13B parameters
- ``olmo2_32b`` - 32B parameters (GQA)

**OLMo3 Series** — hybrid sliding window + full attention, 8K–65K context with YaRN:

- ``olmo3_7b`` - 7B parameters
- ``olmo3_32b`` - 32B parameters (GQA)

Model Configuration
-------------------

OLMOConfig
~~~~~~~~~~

.. autoclass:: OLMOConfig
    :members:
    :show-inheritance:

    Configuration class for OLMo models. Extends :class:`~fairseq2.models.llama.LLaMAConfig`
    with OLMo-specific architecture choices such as post-norm residual connections,
    Q/K normalization, and optional hybrid sliding window attention (OLMo3).

    The default values correspond to the OLMo2 1B architecture.

    **Key Parameters:**

    * ``model_dim`` - Model dimensionality (default: 2048)
    * ``num_layers`` - Number of decoder layers (default: 16)
    * ``num_attn_heads`` - Number of attention heads (default: 16)
    * ``num_key_value_heads`` - Key/value heads for GQA; equals ``num_attn_heads`` for MHA (default: 16)
    * ``max_seq_len`` - Maximum sequence length (default: 4096)
    * ``vocab_size`` - Vocabulary size (default: 100,352)
    * ``sliding_window`` - Sliding window size for OLMo3 hybrid attention; ``None`` for OLMo2 (default: ``None``)
    * ``yarn_scale_config`` - YaRN scaling for OLMo3 long-context models (default: ``None``)

YaRNScaleConfig
~~~~~~~~~~~~~~~

.. autoclass:: YaRNScaleConfig
    :members:

    Configuration for YaRN (Yet another RoPE extensioN) scaling, used by OLMo3 to
    extend context length from 8K to 65K tokens. YaRN scaling is applied selectively
    to full-attention layers; sliding window layers use standard RoPE.

    Reference: https://arxiv.org/abs/2309.00071

Tokenizer
---------

OlmoTokenizer
~~~~~~~~~~~~~

.. autoclass:: OlmoTokenizer
    :members:
    :show-inheritance:

OlmoTokenizerConfig
~~~~~~~~~~~~~~~~~~~

.. autoclass:: OlmoTokenizerConfig
    :members:
    :show-inheritance:

load_olmo_tokenizer
~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_olmo_tokenizer

Hub
---

get_olmo_model_hub
~~~~~~~~~~~~~~~~~~

.. autodata:: get_olmo_model_hub

    Returns the model hub accessor for OLMo models.

    .. code-block:: python

        from fairseq2.models.olmo import get_olmo_model_hub

        hub = get_olmo_model_hub()
        model = hub.load_model("olmo2_7b", device=device)

Constants
---------

OLMO_FAMILY
~~~~~~~~~~~

.. autodata:: OLMO_FAMILY
    :annotation: = "olmo"

    The family name identifier for OLMo models.

Complete Examples
-----------------

Basic Model Usage
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch

    from fairseq2.device import get_default_device
    from fairseq2.models.olmo import get_olmo_model_hub, load_olmo_tokenizer
    from fairseq2.nn import BatchLayout

    device = get_default_device()

    hub = get_olmo_model_hub()
    model = hub.load_model("olmo2_7b", device=device)
    tokenizer = load_olmo_tokenizer("olmo2_7b")

    texts = ["The capital of France is", "The capital of Germany is"]
    encoder = tokenizer.create_encoder()
    tokens = torch.vstack([encoder(text) for text in texts]).to(device)

    model.eval()
    with torch.inference_mode():
        seqs_layout = BatchLayout.of(tokens)
        output = model(tokens, seqs_layout=seqs_layout)

Custom Architecture
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models.olmo import get_olmo_model_hub

    hub = get_olmo_model_hub()

    config = hub.get_arch_config("olmo2_7b")
    config.max_seq_len = 2048
    config.dropout_p = 0.1

    model = hub.create_new_model(config)

See Also
--------

* :doc:`/reference/fairseq2.models.hub` - Model hub API reference
* :doc:`/guides/add_model` - Tutorial on adding new models
* :doc:`/basics/assets` - Understanding the asset system
