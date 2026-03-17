==================================================
HuggingFace Model SFT Training with fairseq2
==================================================

.. currentmodule:: fairseq2.models.hg

fairseq2 provides integration with HuggingFace Transformers models through the
:mod:`fairseq2.models.hg` module. This guide walks through a complete
supervised fine-tuning (SFT) example using a Gemma model loaded via HuggingFace
and trained with fairseq2's recipe system and distributed training
infrastructure.

Prerequisites
=============

Ensure fairseq2 is installed and the virtual environment is activated:

.. code:: bash

    source .venv/bin/activate

The example uses ``google/gemma-3-1b-it`` from HuggingFace Hub and the
``facebook/fairseq2-lm-gsm8k`` dataset. Both are downloaded automatically on
first use.

Running the Recipe
==================

fairseq2 uses a YAML-based recipe system for training. A pre-configured
example for Gemma SFT on GSM8K is provided at
``recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml``.

**Single GPU** (recommended):

.. code:: bash

    python -m recipes.lm.sft \
        --config-file recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml \
        /path/to/output

**Multi-GPU with FSDP** (requires 2+ GPUs):

.. code:: bash

    torchrun --nproc_per_node=2 -m recipes.lm.sft \
        --config-file recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml \
        /path/to/output

**Override config values from the command line**:

.. code:: bash

    python -m recipes.lm.sft \
        --config-file recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml \
        --config regime.num_steps=100 dataset.max_seq_len=2048 \
        /path/to/output

**Dump the full default config** (useful for reference):

.. code:: bash

    python -m recipes.lm.sft --dump-config

Understanding the Configuration
================================

The complete Gemma SFT config is shown below. Each section is explained
afterwards.

.. code:: yaml

    model:
      name: null
      family: "hg"
      arch: "causal_lm"
      config_overrides:
        hf_name: "google/gemma-3-1b-it"
        model_type: "causal_lm"
        trust_remote_code: true

    tokenizer:
      path: "google/gemma-3-1b-it"
      family: "hg"

    dataset:
      max_seq_len: 4096
      max_num_tokens: 8192
      chat_mode: false
      config_overrides:
        sources:
          train:
          - path: "hg://facebook/fairseq2-lm-gsm8k"
            split: "sft_train"
            weight: 1.0

    common:
      metric_recorders:
        wandb:
          enabled: False
          project: "fairseq2"
          run_name: "sft_gemma_3_1b_it_gsm8k"

    regime:
      num_steps: 500
      checkpoint_every_n_steps: 500
      validate_every_n_steps: 10000
      checkpoint_every_n_data_epochs: 100
      keep_last_n_checkpoints: 1
      publish_metrics_every_n_steps: 10
      save_model_only: true
      export_hugging_face: false

Model Section
-------------

.. code:: yaml

    model:
      name: null
      family: "hg"
      arch: "causal_lm"
      config_overrides:
        hf_name: "google/gemma-3-1b-it"
        model_type: "causal_lm"
        trust_remote_code: true

- ``name: null`` disables the default fairseq2 model lookup so the model is
  loaded entirely through the HuggingFace integration.
- ``family: "hg"`` selects the HuggingFace model family, which uses
  :func:`load_causal_lm` internally.
- ``arch: "causal_lm"`` tells the factory to use ``AutoModelForCausalLM``.
- ``config_overrides`` passes fields to
  :class:`~fairseq2.models.hg.config.HuggingFaceModelConfig`:

  - ``hf_name`` is the HuggingFace Hub model identifier.
  - ``model_type: "causal_lm"`` ensures the model is wrapped in
    :class:`~fairseq2.models.hg.adapter.HgCausalLMAdapter`, which
    adapts the HuggingFace model to fairseq2's
    :class:`~fairseq2.models.clm.CausalLM` interface.

Compare this with a native fairseq2 model config (e.g. LLaMA) which only needs
``name``:

.. code:: yaml

    model:
      name: "llama3_2_1b"

Tokenizer Section
-----------------

.. code:: yaml

    tokenizer:
      path: "google/gemma-3-1b-it"
      family: "hg"

- ``path`` specifies the HuggingFace tokenizer to load (same identifier as the
  model). This uses ``AutoTokenizer.from_pretrained`` under the hood via
  :func:`load_hg_tokenizer_simple`.
- ``family: "hg"`` selects the HuggingFace tokenizer family.

Dataset Section
---------------

.. code:: yaml

    dataset:
      max_seq_len: 4096
      max_num_tokens: 8192
      chat_mode: false
      config_overrides:
        sources:
          train:
          - path: "hg://facebook/fairseq2-lm-gsm8k"
            split: "sft_train"
            weight: 1.0

- ``max_seq_len: 4096`` drops sequences longer than 4096 tokens.
- ``max_num_tokens: 8192`` enables dynamic batching — each batch contains at
  most 8192 tokens total, automatically adjusting the number of sequences per
  batch based on their lengths.
- ``chat_mode: false`` uses standard SFT behavior where all tokens after the
  source are treated as training targets. HuggingFace tokenizers do not generate
  the ``assistant_mask`` required by fairseq2's chat mode.
- ``sources`` defines the training data. The ``hg://`` prefix loads datasets
  from HuggingFace Hub. Multiple sources with different weights can be specified
  for weighted sampling.

The dataset expects JSONL entries with ``src`` and ``tgt`` fields:

.. code:: json

    {"src": "What is 2 + 2?", "tgt": "2 + 2 = 4. The answer is 4."}

Regime Section
--------------

.. code:: yaml

    regime:
      num_steps: 500
      checkpoint_every_n_steps: 500
      validate_every_n_steps: 10000
      keep_last_n_checkpoints: 1
      publish_metrics_every_n_steps: 10
      save_model_only: true
      export_hugging_face: false

- ``num_steps: 500`` trains for 500 optimizer steps (roughly 5 epochs over the
  GSM8K dataset on a single GPU).
- ``save_model_only: true`` saves only the model weights, not the optimizer
  state. This produces smaller checkpoints suitable for inference.
- ``export_hugging_face: false`` should be disabled for HuggingFace models since
  they are already in HuggingFace format.

Key Differences from Native fairseq2 Models
============================================

When using HuggingFace models with the recipe system, there are a few
differences compared to native fairseq2 model families like LLaMA:

1. **Model loading**: Set ``family: "hg"`` and provide ``config_overrides``
   with the HuggingFace model identifier instead of using a fairseq2
   ``name``.

2. **Tokenizer**: Set ``family: "hg"`` and use ``path`` (not ``name``) to
   point to the HuggingFace tokenizer.

3. **Chat mode**: Disable ``chat_mode`` unless your HuggingFace tokenizer
   generates ``assistant_mask`` fields. Standard SFT (all target tokens
   contribute to loss) works out of the box.

4. **HuggingFace export**: Set ``export_hugging_face: false`` — the model is
   already in HuggingFace format.

For comparison, a native LLaMA config is much shorter because the model and
tokenizer are registered in fairseq2's asset system:

.. code:: yaml

    model:
      name: "llama3_2_1b"
    tokenizer:
      name: "llama3_2_1b"
    dataset:
      max_seq_len: 4096
      max_num_tokens: 8192
      chat_mode: true
      config_overrides:
        sources:
          train:
          - path: "hg://facebook/fairseq2-lm-gsm8k"
            split: "sft_train"
            weight: 1.0

Adapting to Other HuggingFace Models
=====================================

To fine-tune a different HuggingFace model, copy the Gemma config and change
the model identifier:

.. code:: yaml

    model:
      name: null
      family: "hg"
      arch: "causal_lm"
      config_overrides:
        hf_name: "mistralai/Mistral-7B-Instruct-v0.3"
        model_type: "causal_lm"

    tokenizer:
      path: "mistralai/Mistral-7B-Instruct-v0.3"
      family: "hg"

For seq2seq models (e.g. T5), change ``arch`` and ``model_type``:

.. code:: yaml

    model:
      name: null
      family: "hg"
      arch: "seq2seq_lm"
      config_overrides:
        hf_name: "google-t5/t5-small"
        model_type: "seq2seq_lm"

Checkpointing and Resumption
=============================

The recipe uses fairseq2's
:class:`~fairseq2.checkpoint.StandardCheckpointManager` for robust
checkpointing:

- **Automatic resume**: Re-running the same command with the same output
  directory automatically resumes from the last checkpoint.
- **Distributed-safe**: Works correctly in multi-GPU setups.
- **Checkpoints saved to**: ``{output_dir}/checkpoints/step_{N}/``

.. code:: bash
    :caption: Example - Resume Training

    # First run (trains and saves checkpoint)
    python -m recipes.lm.sft \
        --config-file recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml \
        /path/to/output

    # Resume from checkpoint (automatically detects and loads)
    python -m recipes.lm.sft \
        --config-file recipes/lm/sft/configs/gemma_3_1b_it_gsm8k.yaml \
        /path/to/output

Multi-GPU Notes
===============

When running with FSDP (the default for multi-GPU):

- Each rank sees its local device (e.g., ``cuda:0`` from rank 0's perspective).
  The model is actually sharded across all GPUs — this is expected FSDP
  behavior.
- Only rank 0 prints output to avoid clutter.
- Adjust ``max_num_tokens`` if you run out of memory on multi-GPU setups.

.. note::

    For production training, consider tuning the optimizer and learning rate
    scheduler. The recipe system exposes ``optimizer`` and ``lr_scheduler``
    config sections — run ``python -m recipes.lm.sft --dump-config`` to see
    all available options.
