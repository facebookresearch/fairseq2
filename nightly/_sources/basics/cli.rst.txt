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

    # List available presets for a recipe (e.g. recipe lm::instruction_finetune)
    fairseq2 lm instruction_finetune --list-presets

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
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config max_num_tokens=512

    # Override nested values
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config optimizer_config.lr=4e-5

    # Override multiple values
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config max_num_tokens=512 max_seq_len=512

    # Override a tuple
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config profile="[500,10]"

.. note::

  Unlike ``--config-file``, only one ``--config`` argument can be used.

3. Adding and Removing Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``add:`` and ``del:`` directives for more advanced configuration:

.. code-block:: bash

    # Add a new configuration value
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --config add:new_param=value

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
        add:custom_param=value

Asset Management
----------------

fairseq2 provides commands to manage and inspect assets:

.. code-block:: bash

    # List all available assets
    fairseq2 assets list

    # Show details of a specific asset
    fairseq2 assets show llama3_1_8b_instruct

    # List assets filtered by type
    fairseq2 assets list --type model
    fairseq2 assets list --type dataset
    fairseq2 assets list --type tokenizer

See More
--------

For more technical details about implementing custom CLIs and extensions, see:

- :doc:`/reference/api/fairseq2.recipes/cli`
