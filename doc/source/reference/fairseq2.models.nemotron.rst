.. _api-models-nemotron:

========================
fairseq2.models.nemotron
========================

.. currentmodule:: fairseq2.models.nemotron

The NemotronH module provides support for NVIDIA's Nemotron-H model family, a
Mamba2-Transformer hybrid Mixture-of-Experts architecture. Phase 1 covers the
text-only 30B-A3B variant: a 52-layer decoder with a 3-way hybrid pattern of
Mamba2 selective state-space layers, Mixture-of-Experts FFN layers, and grouped
attention layers.

Architecture Overview
---------------------

The 30B-A3B model uses a fixed layer pattern (``hybrid_override_pattern``) of
52 single-purpose blocks:

- **23 Mamba2 SSM layers** — selective state-space mixers with chunked scan.
- **23 MoE FFN layers** — 128 experts, top-6 sigmoid routing with bias
  correction, plus one always-on shared expert.
- **6 GQA attention layers** — 32 query heads, 2 key/value heads, RoPE
  (``theta=10000``).

Each block holds exactly one mixer kind. Position information is supplied by
the Mamba2 mixers; the attention layers run without RoPE/ALiBi by default.

Quick Start
-----------

.. code-block:: python

    from fairseq2.models.nemotron import get_nemotron_h_model_hub

    hub = get_nemotron_h_model_hub
    model = hub.load_model("nemotron_h_30b_a3b")

Available Models
----------------

The following architecture is registered:

- ``nemotron_h_30b_a3b`` — 30B total parameters, ~3B active per token.

Model Configuration
-------------------

NemotronHConfig
~~~~~~~~~~~~~~~

.. autoclass:: NemotronHConfig
    :members:
    :show-inheritance:

Configuration Factories
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: register_nemotron_h_configs

Model
-----

NemotronHFactory
~~~~~~~~~~~~~~~~

.. autoclass:: NemotronHFactory
    :members:
    :show-inheritance:

.. autofunction:: create_nemotron_h_model

Components
----------

NemotronHBlock
~~~~~~~~~~~~~~

.. autoclass:: NemotronHBlock
    :members:
    :show-inheritance:

    Single-purpose 3-way hybrid block. Holds exactly one of: Mamba2 mixer,
    MoE FFN, or grouped attention.

NemotronHMamba2Mixer
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: NemotronHMamba2Mixer
    :members:
    :show-inheritance:

    Mamba2 selective state-space mixer. Uses CUDA-accelerated triton kernels
    when available, with a PyTorch fallback for CPU and unsupported devices.

NemotronHMoE
~~~~~~~~~~~~

.. autoclass:: NemotronHMoE
    :members:
    :show-inheritance:

    Mixture-of-Experts feed-forward layer with sigmoid routing, bias
    correction, optional group-level pre-selection, and a parallel shared
    expert. Activations are squared ReLU.

Hub Accessors
-------------

.. autodata:: get_nemotron_h_model_hub

HuggingFace Interop
-------------------

.. autofunction:: convert_nemotron_h_state_dict

    Bidirectional state-dict conversion between HuggingFace Transformers and
    fairseq2 formats. Handles Mamba2 fused projections, MoE expert stacking,
    and GQA head layout.

Distributed Training
--------------------

.. autofunction:: get_nemotron_h_shard_specs

    Tensor-parallel shard specifications for attention, embedding, MoE
    (column/row-sharded experts with TP all-reduce), and the final
    projection.

.. autoclass:: NemotronHMoESharder
    :members:
    :show-inheritance:

Constants
---------

.. autodata:: NEMOTRON_H_FAMILY
    :annotation: = "nemotron_h"

    The family name identifier for NemotronH models.

See Also
--------

* :doc:`/reference/fairseq2.models.hub` — Model hub API reference
* :doc:`/basics/assets` — Understanding the asset system
