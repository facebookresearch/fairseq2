.. _tutorial-presets:

====================================
:octicon:`gear` Working with Presets
====================================

.. dropdown:: What you will learn
    :icon: multi-select
    :animate: fade-in

    * What presets are and why they are useful
    
    * How to use built-in presets
    
    * How to create custom presets
    
    * How to override preset configurations

.. dropdown:: Prerequisites
    :icon: multi-select
    :animate: fade-in

    * Get familiar with fairseq2 basics (:ref:`basics-overview`)
    
    * Ensure you have fairseq2 installed (:ref:`installation`)

    * Familiarize yourself with recipes (:doc:`Recipe </basics/recipe>`)

    * Optionally, checkout the end to end fine-tuning tutorial (:ref:`tutorial-end-to-end-fine-tuning`)



Overview
--------

Presets are pre-defined configurations that help you quickly get started with common training scenarios.
They encapsulate best practices and tested hyperparameters for specific use cases.
They also allows quick hyperparameter sweeps.

The key benefits of using presets are:

* Reduce boilerplate configuration code
* Start with proven configurations
* Easily customize for your needs
* Share configurations across experiments


Using Built-in Presets
----------------------

fairseq2 comes with several built-in presets for common scenarios. To use a preset:

1. List available presets:

.. code-block:: bash

    fairseq2 lm instruction_finetune --list-preset-configs

2. Use a preset:

.. code-block:: bash

    fairseq2 lm instruction_finetune $OUTPUT_DIR --preset llama3_1_instruct

The preset will set default values for all configuration parameters.
You can override any of these values using ``--config``.

Available Presets
^^^^^^^^^^^^^^^^^

For instruction fine-tuning:

* ``llama3_1_instruct`` - Base LLaMA 3 1.8B configuration
* ``llama3_1_instruct_constant_lr`` - With constant learning rate
* ``llama3_1_instruct_lr_anneal_0`` - With LR annealing to 0
* ``llama3_1_70b_instruct`` - LLaMA 3 70B configuration

For preference optimization (DPO/CPO/ORPO/SimPO):

* Similar presets are available with additional criterion-specific configurations


Overriding Preset Values
------------------------

You can override any preset values in two ways:

1. Using command line arguments:

.. code-block:: bash

    fairseq2 lm instruction_finetune $OUTPUT_DIR \
        --preset llama3_1_instruct \
        --config optimizer.config.lr=2e-4 dataset.batch_size=16

2. Using a YAML configuration file:

.. code-block:: yaml
    
    # my_config.yaml
    optimizer:
      config:
        _set_:
            lr: 2e-4

.. code-block:: bash

    fairseq2 lm instruction_finetune $OUTPUT_DIR \
        --preset llama3_1_instruct \
        --config-file my_config.yaml

The override precedence is:

1. Command line overrides (highest priority)
2. Config file values  
3. Preset defaults (lowest priority)

Best Practices
--------------

* Start with an existing preset close to your use case
* Create custom presets for configurations you use frequently
* Document preset parameters and their effects
* Use meaningful preset names that indicate their purpose
* Keep presets focused on specific scenarios
* Version control your custom presets

Go Beyond
---------

Once you are familiar with presets, you can go beyond and easily run hyperparameter sweeps.

.. dropdown:: A dummy slurm example
    :icon: code
    :animate: fade-in

    .. code-block:: bash

        presets=(
            "preset_fast"
            "preset_accurate"
            "preset_default"
        )

        batch_sizes=(
            "16"
            "32"
            "64"
        )

        output_dir=<your_output_dir>

        for preset in "${presets[@]}"; do
            for batch_size in "${batch_sizes[@]}"; do
                echo "Running preset::$preset | batch_size::$batch_size"
                srun fairseq2 <your_recipe> train $output_dir/$preset/batch_size_$batch_size \
                    --preset $preset \
                    --config dataset.batch_size=$batch_size
            done
        done

It will be much easier for you to manage your experiments and benchmark training speed to multiple nodes.

.. image:: /_static/img/tutorials/presets/tutorial_presets_benchmark.png
    :width: 600px
    :align: center
    :alt: Benchmark

See Also
--------

- :doc:`Recipe </basics/recipe>`
- :doc:`CLI </basics/cli>`
