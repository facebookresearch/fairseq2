.. _tutorial-end-to-end-fine-tuning:

====================================================
:octicon:`comment-discussion` End-to-End Fine-Tuning
====================================================

.. dropdown:: What you will learn
    :icon: multi-select
    :animate: fade-in

    * How to customize your assets (`e.g.` models, datasets, tokenizers)

    * How to run instruction fine-tuning recipe

    * How to use fairseq2 to generate (inference)

    * How to convert fairseq2 ckpt to huggingface ckpt for accelerated vllm inference

    * How to run fairseq2 with multiple nodes

.. dropdown:: Prerequisites
    :icon: multi-select
    :animate: fade-in

    * Get familiar with fairseq2 basics (:ref:`basics-overview`)

    * Ensure you have fairseq2 installed (:ref:`installation`)


Overview
--------

#. **Prepare**

    * Download the `LLaMA3.2 1B model`_ from HuggingFace

    * Download the `gsm8k data`_ prepared for this tutorial

#. **Fine-Tune**

    * One simple command to run the instruction fine-tuning recipe

    * Accelerate the training with multiple nodes

#. **Generate**

    * One simple command to generate from the finetuned model

    * Convert fairseq2 model ckpt to hf ckpt for accelerated vllm inference

#. **Go Beyond**

    * Use fairseq2 to accelerate your research


Prepare
-------


Model
^^^^^

Follow the `HuggingFace Models Tutorial`_ to download the `LLaMA3.2 1B model`_, which can be run on volta32gb GPUs.
Once you have the model in your local path, (`e.g.`` ``/models/Llama-3.2-1B/original/consolidated.00.pth``), 
you need to register the model in a YAML card so that fairseq2 will know from where to pull the model 
(read more about :ref:`basics-assets`). To do that:

* Create a YAML file (e.g. ``my_llama3_2_1b.yaml``) with the following content:

.. code-block:: yaml

    name: llama3_2_1b@user
    checkpoint: "/models/Llama-3.2-1B/original/consolidated.00.pth"

    ---

    name: llama3@user
    tokenizer: "/models/Llama-3.2-1B/original/tokenizer.model"

.. tip::

    The ``@user`` specifies this is your special environment. This can also be extended to help resolve different domain name for your clusters


* Save the file in one of the following locations:

    * `Option 1`: Place it in the default fairseq2 asset directory

        * ``mkdir -p ~/.config/fairseq2/assets``

        * ``mv my_llama3_2_1b.yaml ~/.config/fairseq2/assets/``

    * `Option 2`: Specify a custom directory and point ``FAIRSEQ2_USER_ASSET_DIR`` to it

        * ``export FAIRSEQ2_USER_ASSET_DIR=/path/to/custom/asset/directory``

        * ``mv my_llama3_2_1b.yaml /path/to/custom/asset/directory/``

Dataset
^^^^^^^

Follow the `HuggingFace Datasets Tutorial`_ to download the `gsm8k data`_, (formatted with fairseq2 flavor) to your local path (`e.g.` ``/datasets/facebook/fairseq2-lm-gsm8k/``).
We will use the ``sft/train.jsonl`` to fine-tune the model and use the ``test/test.jsonl`` for evaluation.


Fine-Tune
---------

One-Liner
^^^^^^^^^

Running the fine-tuning recipe is as simple as:

.. code-block:: bash

    fairseq2 lm instruction_finetune $OUTPUT_DIR --config \
        dataset=/datasets/facebook/fairseq2-lm-gsm8k/sft \
        model=llama3_2_1b \
        max_num_tokens=4096 \
        dtype=float16 \
        max_num_steps=1000 \
        max_num_data_epochs=20 \
        checkpoint_every_n_steps=1000


.. dropdown:: You can also put the configuration in a YAML file
    :icon: code
    :animate: fade-in

    .. code-block:: yaml

        # /configs/example.yaml
        dataset: /datasets/facebook/fairseq2-lm-gsm8k/sft
        model: llama3_2_1b
        max_num_tokens: 4096
        max_seq_len: 4096
        max_num_steps: 1000
        max_num_data_epochs: 20
        checkpoint_every_n_steps: 1000
        keep_last_n_checkpoints: 1
        keep_last_n_models: 1
        publish_metrics_every_n_steps: 5
        dtype: float16  # volta32gb gpus do not support bfloat16

    Then run:

    .. code-block:: bash

        CONFIG_FILE=/configs/example.yaml
        fairseq2 lm instruction_finetune $OUTPUT_DIR --config-file $CONFIG_FILE

For more details about the configuration, please refer to :ref:`basics-recipe`.

Multi-Node
^^^^^^^^^^

To help accelerate the training, fairseq2 is able to automatically detect multi-node setup.

- `Option 1`: Slurm

    .. code-block:: bash

        srun --nodes=2 --ntasks-per-node=8 \
            fairseq2 lm instruction_finetune $OUTPUT_DIR \
            ...

- `Option 2`: Torchrun

    .. code-block:: bash

        torchrun --standalone --nproc-per-node 8 --no-python \
            fairseq2 lm instruction_finetune $OUTPUT_DIR \
            ...

Generate
--------

Once we have finished the training, we can find in the ``$OUTPUT_DIR`` the model checkpoints in ``$OUTPUT_DIR/checkpoints``. With that, we can now generate over the test dataset!


Native Support
^^^^^^^^^^^^^^

fairseq2 natively supports inference:

.. code-block:: bash

    CKPT_PATH="/checkpoint/$USER/experiments/$EXPERIMENT_NAME/checkpoints/step_1000"
    CKPT_DIR=$(dirname "$CKPT_PATH")
    CKPT="checkpoint_$(basename "$CKPT_DIR")"  # e.g., checkpoint_step_1000
    SAVE_DIR="$CKPT_DIR/generation"
    DATASET="/datasets/facebook/fairseq2-lm-gsm8k/test/test.jsonl"

    fairseq2 lm generate $SAVE_DIR --no-sweep-dir --config \
    checkpoint_dir=$CKPT_DIR \
    model=$CKPT \
        generator_config.temperature=0.1 \
        dataset=$DATASET


VLLM Support
^^^^^^^^^^^^


To accelerate the inference process, we can convert fairseq2 checkpoints to HuggingFace checkpoints, which can be deployed with VLLM. This takes 2 steps:

**Step 1: Convert fairseq2 checkpoint to XLFormer checkpoint**

The first step is to use the fairseq2 command-line (:ref:`basics-cli`) tool to convert the fairseq2 checkpoint to an XLF checkpoint. The command structure is as follows:

.. code-block:: bash

    fairseq2 llama convert_checkpoint --arch <architecture> <fairseq2_checkpoint_dir> <xlf_checkpoint_dir>


* ``<architecture>``: Specify the architecture of the model (`e.g.`, ``llama3``)

* ``<fairseq2_checkpoint_dir>``: Path to the directory containing the Fairseq2 checkpoint

* ``<xlf_checkpoint_dir>``: Path where the XLF checkpoint will be saved


.. note::

    Architecture ``--arch`` must exist and be defined in `e.g.` :meth:`fairseq2.models.llama.archs.register_archs`.


**Step 2: Convert XLFormer checkpoint to HF checkpoint**

After obtaining the XLFormer checkpoint, the next step is to convert it to the Hugging Face format. Please refer to the official `HF script`_.


**Step 3: Deploy with VLLM**

.. code-block:: python

    from vllm import LLM

    llm = LLM(model=<path_to_hf_checkpoint>)  # path of your model
    output = llm.generate("Hello, my name is")
    print(output)

Please refer to the `VLLM documentation`_ for more details.

Check the Accuracy
^^^^^^^^^^^^^^^^^^

Once you generated the output, it is relatively trivial to compute the accuracy. Overall, you just need to:

* Load the generated dataset

* Load the original test dataset as ground truth

* Compare and count the number of correct items

.. dropdown:: Some example utils functions
    :icon: code
    :animate: fade-in

    .. code-block:: python

        import re

        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        INVALID_ANS = "[invalid]"


        def extract_answer(completion: str) -> str:
            """
            Extract the answer from the completion.

            :param completion: The completion.
            :return: The answer.
            """
            global ANS_RE, INVALID_ANS
            match = ANS_RE.search(completion)
            if match:
                match_str = match.group(1).strip()
                match_str = match_str.replace(",", "")
                return match_str
            else:
                return INVALID_ANS


        def is_correct(model_completion: str, gt_example: str) -> bool:
            """
            Check if the model completion is correct.

            :param model_completion: The model completion.
            :param gt_example: The ground truth example.
            :return: True if the model completion is correct, False otherwise.
            """
            gt_answer = extract_answer(gt_example)
            assert gt_answer != INVALID_ANS
            return extract_answer(model_completion) == gt_answer


Go Beyond
---------


That's pretty much it to get you started. But you can do a lot more. fairseq2 is a powerful tool to help you accelerate and scale up your research. It allows:

* Experiment with different hyper-parameter configurations;

.. image:: /_static/img/tutorials/end_to_end_fine_tuning/tutorial_example_elements_per_second.png
    :width: 580px
    :align: center
    :alt: Elements per Second

* Compare performance across various datasets or model architectures;

.. image:: /_static/img/tutorials/end_to_end_fine_tuning/tutorial_example_accuracy.png
    :width: 580px
    :align: center
    :alt: Model Comparison

* Profile resource usage and optimize training workflows;

.. image:: /_static/img/tutorials/end_to_end_fine_tuning/tutorial_example_trace.png
    :width: 580px
    :align: center
    :alt: Tracing

* Connect to your WanDB and monitor your experiments in real-time;

.. image:: /_static/img/tutorials/end_to_end_fine_tuning/tutorial_example_wandb.png
    :width: 580px
    :align: center
    :alt: WandB

Now, up for you to discover!!!



.. _LLaMA3.2 1B model: https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main
.. _gsm8k data: https://huggingface.co/datasets/facebook/fairseq2-lm-gsm8k
.. _HuggingFace Models Tutorial: https://huggingface.co/docs/hub/en/models-downloading
.. _HuggingFace Datasets Tutorial: https://huggingface.co/docs/hub/en/datasets-downloading
.. _HF script: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
.. _VLLM documentation: https://vllm.readthedocs.io/en/latest/
