.. _basics-cli:

:octicon:`terminal` CLI
=======================

The Command-Line Interface (CLI) is a crucial feature in fairseq2, offering users a powerful and flexible way to interact with the framework.
With the CLI, you can quickly and easily execute tasks, customize recipes and configurations, and perform complex operations such as sweep runs and benchmarking.

Basic Usage
-----------

Here are some basic examples of using the CLI:

.. code-block:: bash

    # Get help about available commands
    fairseq2 -h

    # Get help about a specific command group (e.g. recipe lm)
    fairseq2 lm -h

    # Get help about a specific command (e.g. recipe lm::instruction_finetune)
    fairseq2 lm instruction_finetune -h

    # List available configuration presets for a recipe (e.g. recipe lm::instruction_finetune)
    fairseq2 lm instruction_finetune --list-preset-configs

    # Dump the default configuration for a recipe (e.g. recipe lm::instruction_finetune)
    fairseq2 lm instruction_finetune --dump-config

    # Run a recipe with default settings (e.g. recipe lm::instruction_finetune)
    fairseq2 lm instruction_finetune <OUTPUT_DIR>

    # Run a recipe with a custom config file (e.g. recipe lm::instruction_finetune)
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config-file <YOUR_CONFIG>.yaml

Configuration Customization
---------------------------

fairseq2 provides multiple ways to customize recipe configurations:

1. Using Config Files
^^^^^^^^^^^^^^^^^^^^^

You can specify one or multiple YAML config files:

.. code-block:: bash

    # Single config file
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config-file config1.yaml

    # Multiple config files (merged from left to right)
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config-file base.yaml --config-file override.yaml

2. Command Line Overrides
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``--config`` to override specific values:

.. code-block:: bash

    # Override single value
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config dataset.max_num_tokens=512

    # Override nested values
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config optimizer.config.lr=4e-5

    # Override multiple values
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config dataset.max_num_tokens=512 dataset.max_seq_len=512

    # Override a tuple
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config profile="[500,10]"

or add, delete values:

.. code-block:: bash

    # Delete a configuration key
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config del:common.metric_recorders.tensorboard

    # Add a configuration key
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config set:common.metric_recorders.tensorboard="{enabled: true}"

.. note::

  Unlike ``--config-file``, only one ``--config`` argument can be used.

3. Adding and Removing Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``set:`` and ``del:`` directives for more advanced configuration:

.. code-block:: bash

    # Add a new configuration value
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config set:new_param=value

    # Remove a configuration value
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config del:unwanted_param

4. Combining Different Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can combine all these methods, with later values taking precedence:

.. code-block:: bash

    fairseq2 lm instruction_finetune <OUTPUT_DIR> \
        --config-file base.yaml \
        --config-file override.yaml \
        --config max_num_tokens=512 \
        optimizer_config.lr=4e-5 \
        set:custom_param=value

5. Running Sweeps
^^^^^^^^^^^^^^^^^

Sweep tags help organize different runs by creating subdirectories based on configuration values.
The default sweep tag will be generated with the format ``"ps_{preset}.ws_{world_size}.{hash}"``.
You can customize the sweep tag format with the ``--sweep-format`` argument:

.. code-block:: bash

    # Use a custom sweep tag format
    fairseq2 lm preference_finetune <OUTPUT_DIR> --config-file config.yaml --sweep-format="lr_{optimizer.config.lr}/criterion_{criterion.name}"

    # If you don't want the sweep tag, you can use --no-sweep-dir
    fairseq2 lm preference_finetune <OUTPUT_DIR> --config-file config.yaml --no-sweep-dir


The following features are available in fairseq2 sweep tags generator:

**1. Accessing nested configuration values:**

.. code-block:: bash

    fairseq2 lm instruction_finetune <OUTPUT_DIR> --sweep-format="dropout_{model.config.dropout_p}"

**2. Including multiple parameters:**

.. code-block:: bash

    fairseq2 lm instruction_finetune <OUTPUT_DIR> --sweep-format="model_{model.name}.bs_{dataset.batch_size}.lr_{optimizer.config.lr}"

**3. Special placeholders:**

* ``{preset}`` - The configuration preset name
* ``{world_size}`` - The distributed training world size
* ``{hash}`` - A unique hash based on configuration values

**4. Custom directory structure:**

.. code-block:: bash

    # Create nested directory structure with forward slashes
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --sweep-format="model_{model.name}/{optimizer.config.lr}"

Asset Management
----------------

fairseq2 provides commands to manage and inspect assets:

.. code-block:: bash

    # List all available assets
    fairseq2 assets list

    # List assets filtered by type
    fairseq2 assets list --type model
    fairseq2 assets list --type dataset
    fairseq2 assets list --type tokenizer

    # Show details of a specific asset
    fairseq2 assets show llama3_1_8b_instruct

LLaMA Utilities
---------------

fairseq2 provides utilities for working with LLaMA models:

.. code-block:: bash

    # Convert fairseq2 LLaMA checkpoints to reference format
    fairseq2 llama convert_checkpoint <MODEL_NAME> <INPUT_DIR> <OUTPUT_DIR>

    # Write LLaMA configurations in Hugging Face format
    fairseq2 llama write_hf_config <MODEL_NAME> <OUTPUT_DIR>

Available Recipe Groups
-----------------------

fairseq2 includes several recipe groups for different tasks:

- ``asr``: ASR (Automatic Speech Recognition) recipes
- ``lm``: Language model recipes (instruction fine-tuning, preference optimization, etc.)
- ``mt``: Machine translation recipes
- ``wav2vec2``: wav2vec 2.0 pretraining recipes
- ``wav2vec2_asr``: wav2vec 2.0 ASR recipes

For more details about the recipe configurations, please refer to :ref:`basics-recipe`.

See More
--------

For more technical details about implementing custom CLIs and extensions, see:
