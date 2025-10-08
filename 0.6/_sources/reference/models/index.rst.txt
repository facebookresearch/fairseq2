===============
fairseq2.models
===============

.. currentmodule:: fairseq2.models

The models module provides pre-trained models and model architectures for various
tasks including language modeling, machine translation, and speech recognition.

The models module includes:

- LLaMA models
- Qwen models
- Mistral models
- Wav2Vec2 models
- NLLB translation models
- Model loading and configuration utilities
- Model hub interface for advanced operations

.. toctree::
   :maxdepth: 1

   hub
   qwen
   llama

Quick Start
-----------

Loading a Model
~~~~~~~~~~~~~~~

.. code-block:: python

    from fairseq2.models import load_model
    from fairseq2.device import get_default_device

    device = get_default_device()
    model = load_model("qwen3_0.6b", device=device)


Listing Available Models
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # List models from command line
    python -m fairseq2.assets list --kind model


Getting Model Information
~~~~~~~~~~~~~~~~~~~~~~~~~

.. doctest::

    >>> from fairseq2.assets import get_asset_store
    >>> asset_store = get_asset_store()
    >>> card = asset_store.retrieve_card("qwen25_7b")
    >>> print(f"Model family: {card.field('model_family').as_(str)}")
    Model family: qwen
    >>> print(f"Architecture: {card.field('model_arch').as_(str)}")
    Architecture: qwen25_7b

Supported Model Families
-------------------------

Please check ``src/fairseq2/models/`` directories for supported model families.
You can also list available models from command line:

.. code-block:: bash

    # List models from command line
    $ python -m fairseq2.assets list --kind model
