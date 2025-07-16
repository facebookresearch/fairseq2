.. _tutorial-preference-optimization:

======================================
:octicon:`law` Preference Optimization
======================================

.. dropdown:: What you will learn
    :icon: multi-select
    :animate: fade-in

    * How to run preference optimization recipe

    * How to customize the criterion units and use DPO/CPO/ORPO/SimPO

    * How to run preference optimization with multiple nodes in fairseq2

.. dropdown:: Prerequisites
    :icon: multi-select
    :animate: fade-in

    * Get familiar with fairseq2 basics (:ref:`basics-overview`)

    * Ensure you have fairseq2 installed (:ref:`installation`)

    * Get familiar with the tutorial on end-to-end fine-tuning (:ref:`tutorial-end-to-end-fine-tuning`)


Overview
--------

#. **Prepare**

    * Download the `LLaMA3.1 8B instruction finetuned model`_ from HuggingFace

    * Download the `gsm8k data`_ prepared for this tutorial

#. **Fine-Tune**

    * One simple command to run the preference optimization recipe

    * Choose between different preference optimization methods (DPO/CPO/ORPO/SimPO)

    * Customize configurations for each method

    * Accelerate the training with multiple nodes

#. **Generate**

    * One simple command to generate from the finetuned model

    * Since this step is similar to what has been covered in :ref:`tutorial-end-to-end-fine-tuning`, we would not elaborate on this topic.


Prepare
-------

Model
^^^^^

Follow the `HuggingFace Models Tutorial`_ to download the `LLaMA3.1 8B instruction finetuned model`_, which can be run on volta32gb GPUs.
Once you have the model in your local path, (`e.g.`` ``/models/Llama-3.1-8B/original/consolidated.00.pth``), 
you need to register the model in a YAML card so that fairseq2 will know from where to pull the model 
(read more about :ref:`basics-assets`). To do that:

* Create a YAML file (e.g. ``my_llama3_1_8b_instruct.yaml``) with the following content:

.. code-block:: yaml

    name: llama3_1_8b_instruct@user
    checkpoint: "/models/Meta-Llama-3-8B-Instruct/original/consolidated.00.pth"
    tokenizer: "/models/Meta-Llama-3-8B-Instruct/original/tokenizer.model"

.. tip::

    The ``@user`` specifies this is your special environment. This can also be extended to help resolve different domain name for your clusters


* Save the file in one of the following locations:

    * `Option 1`: Place it in the default fairseq2 asset directory

        * ``mkdir -p ~/.config/fairseq2/assets``

        * ``mv my_llama3_1_8b_instruct.yaml ~/.config/fairseq2/assets/``

    * `Option 2`: Specify a custom directory and point ``FAIRSEQ2_USER_ASSET_DIR`` to it

        * ``export FAIRSEQ2_USER_ASSET_DIR=/path/to/custom/asset/directory``

        * ``mv my_llama3_1_8b_instruct.yaml /path/to/custom/asset/directory/``

You can check out the predefined fairseq2 LLaMA model cards `here`_.

Dataset
^^^^^^^

Follow the `HuggingFace Datasets Tutorial`_ to download the `gsm8k data`_, (formatted with fairseq2 flavor) to your local path (`e.g.` ``/datasets/facebook/fairseq2-lm-gsm8k/``).
We will use the ``dpo/train.jsonl`` to fine-tune the model and use the ``test/test.jsonl`` for evaluation.


Fine-Tune
---------

One-Liner
^^^^^^^^^

Running the preference optimization recipe is as simple as:

.. code-block:: bash

    fairseq2 lm preference_finetune $OUTPUT_DIR --config \
        dataset.path=/datasets/facebook/fairseq2-lm-gsm8k/dpo \
        model.name=llama3_1_8b_instruct \
        trainer.dtype=float16 \
        regime.num_steps=1000 \
        regime.num_data_epochs=20 \
        regime.checkpoint_every_n_steps=1000

By default, DPO (direct preference optimization) is applied (``--config criterion.name=dpo``).
The use of other methods (CPO/ORPO/SimPO) is documented below.
The configuration fields are detailed in the page :ref:`basics-recipe`.
The fields follows a nested structure, where each field is a key-value pair.
In the example above, we have made changes to config sections including ``dataset``, ``model``, ``trainer``, ``regime``.

Dumping Configuration
^^^^^^^^^^^^^^^^^^^^^

For a quick overview of all the sections and fields, you can use the ``--dump-config`` command:

.. code-block:: bash

    fairseq2 lm preference_finetune --dump-config

Preference Optimization Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fairseq2 supports four different preference optimization methods:

1. **DPO (Direct Preference Optimization)**

    * Paper: https://arxiv.org/abs/2305.18290
    * Key configuration parameters:
        - ``beta``: Coefficient of regularization towards reference model (default: 0.1)
        - ``nll_scale``: Coefficient of NLL loss (default: 0.0)
        - ``length_normalization``: Whether to use length normalized rewards (default: False)
        - ``reference_model``: Name of reference model (default: llama3_1_8b_instruct)
        - ``reference_dtype``: Data type of reference model (default: float16)

.. dropdown:: Example preset for DPO
    :icon: code
    :animate: fade-in

    Here's an example preset config for DPO:

    .. code-block:: yaml

        # dpo.yaml
        model:
            _set_:
                name: llama3_1_8b_instruct
        dataset:
            _set_:
                name: gsm8k_dpo
                path: null
                family: generic_preference
                source_encode_mode: prompt
                target_encode_mode: prompt_response
                mask_source_tokens: true
                min_seq_len: 1
                max_seq_len: 8192
                max_num_tokens: 16384
                batch_size: null
                example_shuffle_window: 10000
                batch_shuffle_window: 1000
                num_prefetch: 4
                extras: {}
        criterion:
            _set_:
                name: dpo
                config:
                    reference_model:
                        name: llama3_1_8b_instruct
                    reference_dtype: bfloat16
                    beta: 0.1
                    nll_scale: 0.0
                    length_normalization: false
        gang:
            _set_:
                tensor_parallel_size: 1
                timeout: 15
                high_priority: true
                monitored: false
        trainer:
            _set_:
                dtype: bfloat16
                data_parallelism: fsdp
                mixed_precision: static
                gradient_accumulation: 1
                activation_checkpointing: true
                max_gradient_norm: null
                fp16_loss_scale:
                - 128.0
                - 0.0001
                torch_compile: false
                profile: null
                gradient_check: false
                anomaly_detection: false
            fsdp:
                _set_:
                    version: v1
                    granularity: layer
                    hybrid: false
                    reshard_after_forward: true
                    fp32_reduce: true
        optimizer:
            _set_:
                name: adamw
            config:
                _set_:
                    lr: 5.5e-06
                    betas:
                    - 0.9
                    - 0.95
                    eps: 1.0e-08
                    weight_decay: 0.1
                    amsgrad: false
                    maximize: false
                    capturable: false
                    differentiable: false
                    impl: auto
                    use_fp32: false
        lr_scheduler:
            _set_:
                name: cosine_annealing
            config:
                _set_:
                    cycle_len: null
                    num_warmup_steps: 0
                    cycle_mul: 1.0
                    lr_mul: 1.0
                    start_lr: 0.0
                    final_lr: null
                    final_lr_scale: 0.2
        regime:
            _set_:
                num_steps: 5000
                num_data_epochs: null
                score_metric: null
                lower_score_better: false
                validate_after_n_steps: 0
                validate_every_n_steps: null
                validate_after_n_data_epochs: 0
                validate_every_n_data_epochs: null
                checkpoint_after_n_steps: 0
                checkpoint_every_n_steps: 1000
                checkpoint_after_n_data_epochs: 0
                checkpoint_every_n_data_epochs: null
                keep_last_n_checkpoints: 1
                keep_best_n_checkpoints: null
                keep_checkpoint_every_n_steps: null
                publish_metrics_after_n_steps: 0
                publish_metrics_every_n_steps: 10
                publish_metrics_after_n_data_epochs: 0
                publish_metrics_every_n_data_epochs: null
        common:
            _set_:
                seed: 2
            metric_recorders:
                log:
                    _set_:
                        enabled: true
                jsonl:
                    _set_:
                        enabled: true
                tensorboard:
                    _set_:
                        enabled: true
                wandb:
                    _set_:
                        enabled: false
                        project: null
                        run: null
            profilers:
                torch:
                    _set_:
                        enabled: false
                        skip_n_steps: 4
                        wait_n_steps: 0
                        num_warmup_steps: 1
                        num_active_steps: 4
                        repeat: 1
            assets:
                _set_:
                    extra_path: null
                    checkpoint_dir: null

    .. code-block:: bash

        fairseq2 lm preference_finetune $OUTPUT_DIR --config-file /path/to/dpo.yaml

2. **CPO (Contrastive Preference Optimization)**

    * Paper: https://arxiv.org/abs/2401.08417
    * Key configuration parameters:
        - ``beta``: Coefficient for preferred vs dispreferred sequences (default: 1.0)
        - ``nll_scale``: Coefficient of NLL loss (default: 1.0)

.. dropdown:: Example preset for CPO
    :icon: code
    :animate: fade-in

    Here's an example preset config for CPO:

    .. code-block:: yaml

        # cpo.yaml
        model:
            _set_:
                name: llama3_1_8b_instruct
        dataset:
            _set_:
                path: /checkpoint/seamless/data/gsm8k_data/dpo
                batch_size: 1
        criterion:
            _set_:
                name: cpo
                config:
                    beta: 0.1
                    nll_scale: 0.0

    Then, to run the preference finetuning recipe with CPO unit:

    .. code-block:: bash

        fairseq2 lm preference_finetune $OUTPUT_DIR --config-file /path/to/cpo.yaml

3. **ORPO (Odds Ratio Preference Optimization)**

    * Paper: https://arxiv.org/abs/2403.07691
    * Key configuration parameters:
        - ``orpo_lambda``: Coefficient of odds-ratio component (default: 1.0)
        - ``nll_scale``: Coefficient of NLL loss (default: 1.0)

.. dropdown:: Example preset for ORPO
    :icon: code
    :animate: fade-in

    Here's an example preset config for ORPO:

    .. code-block:: yaml

        # orpo.yaml
        model:
            _set_:
                name: llama3_1_8b_instruct
        dataset:
            _set_:
                path: /checkpoint/seamless/data/gsm8k_data/dpo
                batch_size: 1
        criterion:
            _set_:
                name: orpo
                config:
                    nll_scale: 0.0
                    orpo_lambda: 0.1

    Then, to run the preference finetuning recipe with ORPO unit:

    .. code-block:: bash

        fairseq2 lm preference_finetune $OUTPUT_DIR --config-file /path/to/orpo.yaml


4. **SimPO (Simple Preference Optimization)**

    * Paper: https://arxiv.org/abs/2405.14734
    * Key configuration parameters:
        - ``beta``: Coefficient of KL-divergence regularization (default: 1.0)
        - ``gamma``: Target reward margin between completions (default: 0.5)
        - ``nll_scale``: Coefficient of NLL loss (default: 0.0)

.. dropdown:: Example preset for SimPO
    :icon: code
    :animate: fade-in

    Here's an example preset config for SimPO:

    .. code-block:: yaml

        # simpo.yaml
        model:
            _set_:
                name: llama3_1_8b_instruct
        dataset:
            _set_:
                path: /checkpoint/seamless/data/gsm8k_data/dpo
                batch_size: 1
        criterion:
            _set_:
                name: simpo
                config:
                    beta: 2
                    nll_scale: 0.0

    Then, to run the preference finetuning recipe with SimPO unit:

    .. code-block:: bash

        fairseq2 lm preference_finetune $OUTPUT_DIR --config-file /path/to/simpo.yaml


Iterative Training
^^^^^^^^^^^^^^^^^^

Sometimes you may want to continue fine-tuning from a previously trained checkpoint, either to:

- Resume interrupted training
- Fine-tune on additional data
- Perform iterative fine-tuning with different hyperparameters

fairseq2 provides a clean way to handle this through the checkpoint system (learn more about :ref:`basics-ckpt-management`):

.. code-block:: bash

    fairseq2 lm preference_finetune $OUTPUT_DIR --config \
        common.assets.checkpoint_dir=/path/to/checkpoint \
        model.name=last_checkpoint \  # this will pick up the last checkpoint
        dataset.path=/path/to/data

.. dropdown:: To pick up a specific checkpoint
    :icon: code
    :animate: fade-in

    .. code-block:: bash

        CKPT_DIR="/checkpoint/user/experiments/run_0/checkpoints"
        CKPT="checkpoint_step_1000"  # e.g. checkpoint of step 1000

        fairseq2 lm preference_finetune $OUTPUT_DIR --config \
            common.assets.checkpoint_dir=$CKPT_DIR \
            model.name=$CKPT \
            dataset.path=/path/to/new/data \
            dataset.max_num_tokens=4096 \
            trainer.dtype=float16

    .. note::

        If you want to pick a specific checkpoint instead of the last checkpoint, the ``model`` parameter must be set to ``checkpoint_step_X`` where X matches the step number of the checkpoint you want to load.

Multi-Node
^^^^^^^^^^

To help accelerate the training, fairseq2 is able to automatically detect multi-node setup.

- `Option 1`: Slurm

    .. code-block:: bash

        srun --nodes=2 --ntasks-per-node=8 \
            fairseq2 lm preference_finetune $OUTPUT_DIR \
            ...

- `Option 2`: Torchrun

    .. code-block:: bash

        torchrun --standalone --nproc-per-node 8 --no-python \
            fairseq2 lm preference_finetune $OUTPUT_DIR \
            ...

Generate
--------

Once we have finished the training, we can find in the ``$OUTPUT_DIR`` the model checkpoints in ``$OUTPUT_DIR/checkpoints``.
With that, we can now generate over the test dataset!

You can either use fairseq2 native generation recipe:

.. code-block:: bash

    CKPT_DIR="/checkpoint/$USER/my_experiment/checkpoints"
    CKPT="last_checkpoint"
    SAVE_DIR="/checkpoint/$USER/my_experiment/generations"
    DATASET="/datasets/facebook/fairseq2-lm-gsm8k/test/test.jsonl"

    fairseq2 lm generate $SAVE_DIR --no-sweep-dir --config \
        common.assets.checkpoint_dir=$CKPT_DIR \
        model.name=$CKPT \
        seq_generator.config.temperature=0.1 \
        dataset.path=$DATASET

Or accelerate with VLLM:

.. code-block:: python

    from vllm import LLM

    llm = LLM(
        model=<path_to_fs2_checkpoint>,  # path of your model
        tokenizer=<name_or_path_of_hf_tokenizer>,  # path of your tokenizer files
    )
    output = llm.generate("Hello, my name is")
    print(output)

For the simplicity of our documentation, please refer to :ref:`tutorial-end-to-end-fine-tuning` for more details.

See Also
--------

- :doc:`Design Philosophy </basics/design_philosophy>`
- :doc:`Recipe </basics/recipe>`
- :doc:`CLI </basics/cli>`
- :doc:`Assets </basics/assets>`
- :ref:`tutorial-end-to-end-fine-tuning`


.. _LLaMA3.1 8B instruction finetuned model: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main
.. _gsm8k data: https://huggingface.co/datasets/facebook/fairseq2-lm-gsm8k
.. _here: https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/assets/cards/models/llama.yaml
.. _HuggingFace Models Tutorial: https://huggingface.co/docs/hub/en/models-downloading
.. _HuggingFace Datasets Tutorial: https://huggingface.co/docs/hub/en/datasets-downloading
.. _HF script: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
.. _VLLM documentation: https://vllm.readthedocs.io/en/latest/
