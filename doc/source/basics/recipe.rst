.. _basics-recipe:

:octicon:`gift` Recipes
=======================


Recipe Configuration Structure
------------------------------

Recipes in fairseq2 use a structured configuration system based on dataclasses. Let's examine the configuration structure using the instruction fine-tuning recipe as an example.

Core Configuration Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Model Configuration
""""""""""""""""""""""

Controls model architecture and parameters:

.. code-block:: yaml

    model:
      name: llama3_1_8b_instruct    # Model name/identifier
      config:                       # Model-specific configuration
        dropout_p: 0.0              # Dropout probability

2. Dataset Configuration
""""""""""""""""""""""""
Defines dataset and data loading parameters:

.. code-block:: yaml

    dataset:
      name: foo                     # Dataset name
      family: instruction           # Dataset family
      path: /path/to/data           # Path to dataset
      train_split: default          # Training split name
      max_seq_len: 8192             # Maximum sequence length
      max_num_tokens: 16384         # Maximum tokens per batch
      batch_size: null              # Fixed batch size (if specified)
      example_shuffle_window: 10000 # Window size for example shuffling
      batch_shuffle_window: 1000    # Window size for batch shuffling
      num_prefetch: 4               # Number of batches to prefetch

3. Trainer Configuration
""""""""""""""""""""""""

Specifies training behavior and hardware settings:

.. code-block:: yaml

    trainer:
      dtype: bfloat16                # Training data type
      data_parallelism: fsdp         # Data parallel strategy (ddp or fsdp)
      activation_checkpointing: true # Use activation checkpointing
      gradient_accumulation: 1       # Gradient accumulation steps
      torch_compile: false           # Use torch.compile
      mixed_precision: static        # Mixed precision mode
      fsdp:                          # FSDP-specific settings
        granularity: layer
        reshard_after_forward: true

4. Optimizer Configuration
""""""""""""""""""""""""""

Controls optimization parameters:

.. code-block:: yaml

    optimizer:
      name: adamw                  # Optimizer type
      config:                      # Optimizer-specific config
        lr: 5.5e-6                 # Learning rate
        betas: [0.9, 0.95]         # Adam betas
        weight_decay: 0.1          # Weight decay

5. Learning Rate Scheduler
""""""""""""""""""""""""""

Defines learning rate scheduling:

.. code-block:: yaml

    lr_scheduler:
      name: cosine                # Scheduler type
      config:                     # Scheduler-specific config
        final_lr_scale: 0.2       # Final LR as fraction of initial

6. Training Regime
""""""""""""""""""

Defines training loop behavior:

.. code-block:: yaml

    regime:
      num_steps: 5000                   # Total training steps
      validate_every_n_steps: 100       # Validation frequency
      checkpoint_every_n_steps: 1000    # Checkpoint frequency
      keep_last_n_checkpoints: 1        # Number of checkpoints to keep
      publish_metrics_every_n_steps: 10 # Metrics logging frequency

Using Preset Configurations
---------------------------

fairseq2 provides preset configurations for common scenarios:

.. code-block:: python

    # Available presets for instruction fine-tuning:
    - llama3_1_instruct             # Base LLaMA 3 1.8B
    - llama3_1_instruct_constant_lr # With constant learning rate
    - llama3_1_instruct_lr_anneal_0 # With LR annealing to 0
    - llama3_1_70b_instruct         # LLaMA 3 70B
    - llama2_7b_chat                # LLaMA 2 7B
    - llama2_70b_chat               # LLaMA 2 70B

To use a preset:

.. code-block:: bash

    fairseq2 lm instruction_finetune <OUTPUT_DIR> --preset llama2_7b_chat

Customizing Configurations
--------------------------

You can customize configurations in several ways:

1. Using a YAML file with the configuration override syntax. Note the ``_set_``
   directive:

.. code-block:: yaml

    # config.yaml
    dataset:
        _set_:
            path: /data/my_dataset
            max_num_tokens: 4096
    optimizer:
        config:
            _set_:
                lr: 5e-7

2. Using command line overrides:

.. code-block:: bash

    fairseq2 lm instruction_finetune <OUTPUT_DIR> \
        --preset llama2_7b_chat \
        --config-file config.yaml \
        --config optimizer.config.lr=4e-5

For more details about the CLI usage, see :ref:`basics-cli`.
