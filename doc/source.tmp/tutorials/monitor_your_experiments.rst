.. _tutorials-monitor-your-experiments:

:octicon:`codescan-checkmark` Monitor Your Experiments
======================================================


.. dropdown:: What you will learn
    :icon: multi-select
    :animate: fade-in

    * How to monitor your experiments using Tensorboard

    * How to monitor your experiments using WanDB

.. dropdown:: Prerequisites
    :icon: multi-select
    :animate: fade-in

    * Get familiar with fairseq2 basics (:ref:`basics-overview`)

    * Ensure you have fairseq2 installed (:ref:`installation`)

TensorBoard
-----------


.. image:: /_static/img/tutorials/end_to_end_fine_tuning/tutorial_example_trace.png
    :width: 580px
    :align: center
    :alt: TensorBoard


fairseq2 saves checkpoints and tensorboard events to the defined ``$OUTPUT_DIR``, which allows you to investigate into the details in your jobs.

.. code-block:: bash

    # run tensorboard at your ckpt path
    tensorboard --logdir $CHECKPOINT_PATH

    # example
    tensorboard --logdir /checkpoint/$USER/outputs/ps_llama3_1_instruct.ws_16.a73dad52/tb/train

If you ran your experiment on your server, you probably need to port forward the tensorboard service to your local machine:

.. code-block:: bash

    ssh -L 6006:localhost:6006 $USER@$SERVER_NAME

Then you can view the tensorboard service in your browser `http://localhost:6006 <http://localhost:6006>`__.


WanDB
------


.. image:: /_static/img/tutorials/end_to_end_fine_tuning/tutorial_example_wandb.png
    :width: 580px
    :align: center
    :alt: WandB

fairseq2 natively support WanDB (Weights & Biases) - a powerful tool for monitoring and managing machine learning experiments.
WanDB provides a centralized platform to track, compare, and analyze the performance of different models, making it easier to identify trends, optimize hyperparameters, and reproduce results.
Follow the `quick start guide <https://docs.wandb.ai/quickstart>`__ to initialize it in your environment.

What you need to do is simply add the following line in your config YAML file:

.. code-block:: yaml

    common:
        metric_recorders:
            wandb:
                _set_:
                    enabled: true
                    project: <YOUR_PROJECT_NAME>
                    run: <YOUR_JOB_RUN_NAME>

Then run your recipe with ``fairseq2 ... --config-file <YOUR_CONFIG>.yaml``.

Then you can open up your WanDB Portal and check the results in real-time.


.. dropdown:: A step-by-step example
    :icon: code
    :animate: fade-in

    .. code-block:: bash

        ENV_NAME=...  # YOUR_ENV_NAME
        CONFIG_FILE=...  # YOUR_CONFIG_FILE
        OUTPUT_DIR=...  # YOUR_OUTPUT_DIR

        conda activate $ENV_NAME
        # install wandb
        pip install wandb
        # initialize wandb, copy paste your token when prompted
        wandb login --host=...  # your wandb hostname

        # now you are good to go
        fairseq2 lm instruction_finetune $OUTPUT_DIR \
        --config-file $CONFIG_FILE \

        # cleanup
        conda deactivate
