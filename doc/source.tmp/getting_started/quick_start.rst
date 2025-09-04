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
        dataset.path=/datasets/facebook/fairseq2-lm-gsm8k/sft \
        dataset.max_num_tokens=4096 \
        model.name=llama3_2_1b \
        trainer.dtype=float16 \
        regime.num_steps=1000 \
        regime.num_data_epochs=20 \
        regime.checkpoint_every_n_steps=1000

Read more about this recipe in :ref:`tutorial-end-to-end-fine-tuning`.

Preference Optimization (PO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    fairseq2 lm preference_finetune $OUTPUT_DIR --config \
        dataset.path=/datasets/facebook/fairseq2-lm-gsm8k/dpo \
        model.name=llama3_2_1b \
        trainer.dtype=float16 \
        regime.num_steps=1000 \
        regime.num_data_epochs=20 \
        regime.checkpoint_every_n_steps=1000

Read more about this recipe in :ref:`tutorial-preference-optimization`.

Generating Text
^^^^^^^^^^^^^^^

After fine-tuning a language model, you can generate text with the following command:

.. code-block:: bash

    CKPT_DIR="/checkpoint/$USER/experiments/$EXPERIMENT_NAME/checkpoints"
    SAVE_DIR="$CKPT_DIR/generation"
    DATASET="/datasets/facebook/fairseq2-lm-gsm8k/test/test.jsonl"

    fairseq2 lm generate $SAVE_DIR --no-sweep-dir --config \
        common.assets.checkpoint_dir=$CKPT_DIR \
        model.name="last_checkpoint" \
        seq_generator.config.temperature=0.1 \
        dataset.path=$DATASET


See Also
--------

- :doc:`Design Philosophy </basics/design_philosophy>`
- :doc:`Recipe </basics/recipe>`
- :doc:`CLI </basics/cli>`
- :doc:`Assets </basics/assets>`
