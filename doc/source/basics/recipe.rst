.. _basics-recipe:

:octicon:`gift` Recipes
=======================

Instruction Fine-tuning Example
--------------------------------

The recipe handler for instruction fine-tuning is registered at :meth:`fairseq2.recipes.lm._setup_cli`. 
The recipe itself resides at :meth:`fairseq2.recipes.lm.load_instruction_finetuner`. 
This recipe loads a set of components into a :class:`fairseq2.recipes.trainer.Trainer`, including:

- Model
- Data reader (train and valid)
- Checkpoint manager
- Criterion
- Metric recorder

The only inputs required are the configuration and an output directory for checkpoints, training events, and more.

How to Configure a Recipe
--------------------------

Recipe configuration is defined as a class, such as :class:`fairseq2.recipes.lm.instruction_finetune.InstructionFinetuneConfig` for instruction fine-tuning.

Customized configurations can be loaded from YAML files. For example, if you create a YAML configuration file at ``$YOUR_CONFIG.yaml``:

.. code-block:: yaml

    dataset: /data/gsm8k_data/sft
    model: llama3_1_8b
    max_num_tokens: 4096
    max_seq_len: 4096
    max_num_steps: 1000
    max_num_data_epochs: 20
    checkpoint_every_n_steps: 1000
    keep_last_n_checkpoints: 1
    keep_last_n_models: 1
    publish_metrics_every_n_steps: 5

This configuration can be loaded using the following command:

.. code-block:: bash

    fairseq2 lm instruction_finetune --config-file $YOUR_CONFIG.yaml ...

Key Points on Configuration
----------------------------

* **Using YAML Files:**

  * The YAML file path is provided as the argument value to ``--config-file``.

  * The YAML file does not need to represent the entire configuration.

  * You can dump the default preset configuration to view default values using the ``--dump-config`` argument with the recipe command.

  * Multiple ``--config-file`` arguments can be used, and configurations will be merged from left to right, with the last value overriding previous ones.

* **Overriding via CLI:**

  * Configuration values can be adjusted directly via the CLI using ``--config k=v``.

    * For example: ``--config optimizer_config.lr=4e-5``

  * Multiple key-value pairs can be passed:

    * Example: ``--config max_num_tokens=512 max_seq_len=512``

  * CLI overrides can be combined with ``--config-file``, where CLI values will take precedence over YAML values.

* **Using Add/Del Directives:**

  * Directives such as ``add`` or ``del`` can be used for more advanced overrides:

    * Add a new value: ``--config add:xyz=value``

    * Remove a value: ``--config del:xyz``
