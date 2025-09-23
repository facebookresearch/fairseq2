fairseq2.models.hg
==================

The :mod:`fairseq2.models.hg` module provides seamless integration with HuggingFace Transformers models within the fairseq2 framework. This module allows you to load and use any HuggingFace model with fairseq2's training and inference pipelines.

API Reference
-------------

High-Level API
~~~~~~~~~~~~~~

.. autofunction:: fairseq2.models.hg.load_hg_model_simple

.. autofunction:: fairseq2.models.hg.load_hg_tokenizer_simple

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fairseq2.models.hg.load_causal_lm

.. autofunction:: fairseq2.models.hg.load_seq2seq_lm

.. autofunction:: fairseq2.models.hg.load_multimodal_model

Configuration
~~~~~~~~~~~~~

.. autoclass:: fairseq2.models.hg.HuggingFaceModelConfig
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fairseq2.models.hg.HgTokenizerConfig
    :members:
    :undoc-members:
    :show-inheritance:

Factory Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: fairseq2.models.hg.create_hg_model

.. autofunction:: fairseq2.models.hg.register_hg_model_class

Tokenizer Classes
~~~~~~~~~~~~~~~~~

.. autoclass:: fairseq2.models.hg.HgTokenizer
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: fairseq2.models.hg.load_hg_tokenizer

Hub Integration
~~~~~~~~~~~~~~~

.. autofunction:: fairseq2.models.hg.get_hg_model_hub

.. autofunction:: fairseq2.models.hg.get_hg_tokenizer_hub

Exceptions
~~~~~~~~~~

.. autoexception:: fairseq2.models.hg.HuggingFaceModelError
    :members:
    :undoc-members:


Examples
--------

Basic Model Loading
~~~~~~~~~~~~~~~~~~~

Use DialoGPT for conversational AI:

.. code-block:: python

    from fairseq2.models.hg import load_causal_lm, load_hg_tokenizer_simple
    import torch

    # Load DialoGPT model
    model = load_causal_lm("microsoft/DialoGPT-small")
    tokenizer = load_hg_tokenizer_simple("microsoft/DialoGPT-small")

    # Conversation
    user_input = "How are you doing today?"
    inputs = tokenizer.encode(user_input + tokenizer.eos_token).unsqueeze(0)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 20,
            num_return_sequences=1,
            pad_token_id=tokenizer.vocab_info.eos_idx,
            do_sample=True,
            temperature=0.7,
        )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Bot: {response[len(user_input):]}")

Sequence-to-Sequence Tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use T5 for translation and summarization:

.. code-block:: python

    from fairseq2.models.hg import load_seq2seq_lm, load_hg_tokenizer_simple
    import torch

    # Load T5 model
    model = load_seq2seq_lm("t5-small")
    tokenizer = load_hg_tokenizer_simple("t5-small")

    # Translation task
    text = "translate English to French: Hello, how are you?"
    inputs = tokenizer.encode(text).unsqueeze(0)

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50)

    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Translation: {translation}")


Custom Model Registration
~~~~~~~~~~~~~~~~~~~~~~~~~

Register custom models not supported by Auto classes:

.. code-block:: python

    from fairseq2.models.hg import register_hg_model_class, load_hg_model_simple

    # Register a custom model class
    register_hg_model_class(
        config_class_name="Qwen2_5OmniConfig",
        model_class="Qwen2_5OmniForConditionalGeneration",
        processor_class="Qwen2_5OmniProcessor",
    )

    # Now load the custom model
    model = load_hg_model_simple(
        "Qwen/Qwen2.5-Omni-3B",
        model_type="custom",
        use_processor=True,
        trust_remote_code=True,
    )

Hub Loading
~~~~~~~~~~~

Load models/tokenizers using the fairseq2 hub system:

.. code-block:: python

    from fairseq2.models.hg import get_hg_model_hub, get_hg_tokenizer_hub

    name = "hg_qwen25_omni_3b"

    # Load a pre-configured model
    model_hub = get_hg_model_hub()
    model = model_hub.load_model(name)

    # Load the corresponding tokenizer
    tokenizer_hub = get_hg_tokenizer_hub()
    tokenizer = tokenizer_hub.load_tokenizer(name)


.. note::
    This module requires the ``transformers`` library. Install it with:
    ``pip install transformers``

.. warning::
    Some models require ``trust_remote_code=True`` for custom architectures.
    Only use this with trusted model sources.


Module Structure
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   fairseq2.models.hg.api
   fairseq2.models.hg.config
   fairseq2.models.hg.factory
   fairseq2.models.hg.hub
   fairseq2.models.hg.tokenizer
