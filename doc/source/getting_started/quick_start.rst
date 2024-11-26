.. _quick_start:

==========================================
:octicon:`rocket` Quick Start
==========================================

Language Model (LM)
-------------------

Supervised Fine-Tuning (SFT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    fairseq2 lm instruction_finetune $OUTPUT_DIR --config \
        dataset=/datasets/facebook/fairseq2-lm-gsm8k/sft \
        model=llama3_2_1b \
        max_num_tokens=4096 \
        dtype=float16 \
        max_num_steps=1000 \
        max_num_data_epochs=20 \
        checkpoint_every_n_steps=1000

Read more about this recipe in :ref:`tutorial-end-to-end-fine-tuning`.


Generating Text
^^^^^^^^^^^^^^^

After fine-tuning a language model, you can generate text with the following command:

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