.. _api-models-gemma4:

======================
fairseq2.models.gemma4
======================

.. currentmodule:: fairseq2.models.gemma4

The Gemma 4 module provides support for Google's Gemma 4 model family, including dense
(E4B, 31B) and Mixture-of-Experts (26B-A4B) variants with base and instruction-tuned
versions. The architecture features Per-Layer Embeddings (PLE), partial rotary position
encodings, KV sharing across attention layers, QK/V-norm, logit soft-capping, and an
optional Conformer-based audio tower for multimodal inference.

Architecture Overview
---------------------

.. image:: /_static/img/gemma4/gemma4_architecture.svg
   :alt: Gemma 4 decoder architecture
   :align: center

Model Variants
--------------

.. image:: /_static/img/gemma4/gemma4_variants.svg
   :alt: Gemma 4 model variant comparison
   :align: center

.. list-table:: Model Variant Summary
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Variant
     - Params
     - Layers
     - model_dim
     - Features
     - Active Params
   * - E4B / E4B-it
     - 7.5B
     - 34
     - 2560
     - PLE, Audio, KV-share
     - 7.5B (dense)
   * - 31B / 31B-it
     - 30.7B
     - 48
     - 4608
     - KV-share, Double MLP
     - 30.7B (dense)
   * - 26B-A4B
     - 25.2B
     - 34
     - 2560
     - MoE (128 experts, top-2)
     - ~3.8B

Quick Start
-----------

.. code-block:: python

    from fairseq2.models.gemma4 import get_gemma4_model_hub, get_gemma4_tokenizer_hub

    # Get the model hub
    hub = get_gemma4_model_hub

    # Load a model
    model = hub.load_model("gemma4_e4b")

    # Load corresponding tokenizer
    tokenizer = get_gemma4_tokenizer_hub.load_tokenizer("gemma4_e4b")

    # Encode text
    encoder = tokenizer.create_encoder()
    tokens = encoder("The future of AI is")

Available Models
----------------

The following model architectures are registered:

- ``gemma4_e2b`` / ``gemma4_e2b_it`` - 2B parameters (dense)
- ``gemma4_e4b`` / ``gemma4_e4b_it`` - 7.5B parameters (dense, with PLE + audio)
- ``gemma4_31b`` / ``gemma4_31b_it`` - 30.7B parameters (dense)
- ``gemma4_26b_a4b`` / ``gemma4_26b_a4b_it`` - 25.2B total / ~3.8B active (MoE)

All models use:

- Vocabulary size: 262,144
- Tied embeddings with soft-capped final logits (cap=30.0)
- Mixed sliding (window=512) and full (global) attention layers
- QK-norm and V-norm (non-learnable) on attention
- Partial rotary position encoding (50%) on full attention layers
- GELU(tanh) activation in GLU feed-forward networks

Key Architectural Features
--------------------------

**Per-Layer Embeddings (PLE)** — E4B only
    A learned projection splits the input embedding into per-layer contributions,
    each gated by a sigmoid before being added to the hidden state at each decoder
    layer. This replaces the traditional single-embedding approach.

**KV Sharing**
    Adjacent sliding-attention layers share key-value projections. A SOURCE layer
    computes K/V; CONSUMER layers reuse the pre-computed K/V. This saves memory
    and compute without sacrificing quality.

**K=V Attention**
    On full (global) attention layers, the value projection is removed and the
    key projection output is reused as both K and V (after separate norms).

**Mixture of Experts (MoE)** — 26B-A4B only
    Each decoder layer contains a router that selects top-2 experts from a pool
    of 128 experts. The shared FFN runs in parallel with the expert mixture.

**Audio Tower** — E4B only
    A Conformer encoder processes log-mel spectrograms into audio embeddings
    that are injected at ``<audio>`` token positions in the input sequence.

Model Configuration
-------------------

Gemma4Config
~~~~~~~~~~~~

.. autoclass:: Gemma4Config
    :members:
    :show-inheritance:

    **Key Parameters:**

    * ``model_dim`` — Model dimensionality (2560 for E4B/26B-A4B, 4608 for 31B)
    * ``num_layers`` — Number of decoder layers (34 or 48)
    * ``num_attn_heads`` — Number of attention heads
    * ``num_key_value_heads`` — Number of key/value heads for GQA
    * ``head_dim`` — Head dimension for sliding attention (128)
    * ``global_head_dim`` — Head dimension for full attention (256)
    * ``sliding_window`` — Sliding attention window size (512)
    * ``partial_rotary_factor`` — Fraction of head_dim using RoPE (0.5)
    * ``has_ple`` — Whether to use Per-Layer Embeddings
    * ``enable_moe`` — Whether to use Mixture of Experts
    * ``layer_types`` — List of ``"sliding_attention"`` or ``"full_attention"`` per layer
    * ``attention_k_eq_v`` — Whether K=V on full attention layers

Configuration Factories
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_gemma4_e4b_config
.. autofunction:: get_gemma4_e2b_config
.. autofunction:: get_gemma4_31b_config
.. autofunction:: get_gemma4_26b_a4b_config
.. autofunction:: register_gemma4_configs

Model
-----

Gemma4Model
~~~~~~~~~~~

.. autoclass:: Gemma4Model
    :members:
    :show-inheritance:

    Top-level causal language model combining frontend, decoder, and final
    projection. Supports optional audio input via ``audio_features`` parameter.

Gemma4Factory
~~~~~~~~~~~~~

.. autoclass:: Gemma4Factory
    :members:
    :show-inheritance:

.. autofunction:: create_gemma4_model

Components
----------

Gemma4Frontend
~~~~~~~~~~~~~~

.. autoclass:: Gemma4Frontend
    :members:
    :show-inheritance:

    Handles token embedding, PLE computation, and optional audio embedding injection.

Gemma4Decoder
~~~~~~~~~~~~~

.. autoclass:: Gemma4Decoder
    :members:
    :show-inheritance:

    Stacks decoder layers with KV sharing management across layers.

Gemma4DecoderLayer
~~~~~~~~~~~~~~~~~~

.. autoclass:: Gemma4DecoderLayer
    :members:
    :show-inheritance:

    Single decoder layer with attention, FFN, optional PLE, and optional MoE.

Gemma4Attention
~~~~~~~~~~~~~~~

.. autoclass:: Gemma4Attention
    :members:
    :show-inheritance:

    Multi-head attention with QK-norm, V-norm, optional K=V, partial RoPE,
    and KV sharing support (source/consumer roles).

MoE Components
~~~~~~~~~~~~~~

.. autoclass:: Gemma4Router
    :members:
    :show-inheritance:

.. autoclass:: Gemma4Experts
    :members:
    :show-inheritance:

Audio Tower
-----------

.. autoclass:: Gemma4AudioConfig
    :members:
    :show-inheritance:

.. autoclass:: Gemma4AudioTower
    :members:
    :show-inheritance:

.. autoclass:: Gemma4ConformerEncoder
    :members:
    :show-inheritance:

.. autoclass:: Gemma4ConformerBlock
    :members:
    :show-inheritance:

.. autoclass:: Gemma4ConformerAttention
    :members:
    :show-inheritance:

.. autoclass:: Gemma4SubsampleConvProjection
    :members:
    :show-inheritance:

.. autoclass:: Gemma4MultimodalAudioEmbedder
    :members:
    :show-inheritance:

Tokenizer
---------

Gemma4Tokenizer
~~~~~~~~~~~~~~~~

.. autoclass:: Gemma4Tokenizer
    :members:
    :show-inheritance:

    Tokenizer for Gemma 4 models. Uses SentencePiece with a 262,144-token vocabulary.
    Supports chat template formatting for instruction-tuned variants.

.. autofunction:: load_gemma4_tokenizer

Hub Accessors
-------------

.. autodata:: get_gemma4_model_hub
.. autodata:: get_gemma4_tokenizer_hub

HuggingFace Interop
--------------------

.. autofunction:: convert_gemma4_state_dict

    Bidirectional state dict conversion between HuggingFace Transformers and fairseq2
    formats. Handles weight transpositions, key remapping, PLE weight splitting/merging,
    and MoE parameter reshaping.

Distributed Training
--------------------

.. autofunction:: apply_fsdp_to_gemma4

    Apply Fully Sharded Data Parallelism to a Gemma 4 model.

.. autofunction:: apply_ac_to_gemma4

    Apply activation checkpointing to a Gemma 4 model.

.. autofunction:: get_gemma4_shard_specs

    Get tensor parallel shard specifications for a Gemma 4 model configuration.

Constants
---------

.. autodata:: GEMMA4_FAMILY
    :annotation: = "gemma4"

    The family name identifier for Gemma 4 models.

SFT Recipe Configs
------------------

Pre-built SFT recipe configurations for GSM8K fine-tuning are provided:

- ``recipes/lm/sft/configs/gemma4_e4b_gsm8k.yaml``
- ``recipes/lm/sft/configs/gemma4_31b_gsm8k.yaml``
- ``recipes/lm/sft/configs/gemma4_26b_a4b_gsm8k.yaml``

Example usage:

.. code-block:: bash

    fairseq2 lm sft --config recipes/lm/sft/configs/gemma4_e4b_gsm8k.yaml

See Also
--------

* :doc:`/reference/fairseq2.models.hub` — Model hub API reference
* :doc:`/guides/add_model` — Tutorial on adding new models
* :doc:`/basics/assets` — Understanding the asset system
