.. _training:

Training a model
================

.. highlight:: bash

In this tutorial, you will launch the training of a Machine Translation model, on WMT14 dataset.

This tutorial uses `Hugging Face`_ datasets library
to load the data,
make sure you have it installed using::

  pip install datasets

.. _Hugging Face: https://huggingface.co/datasets

``train``
---------

First create a workspace folder to save the trained models.
For example::

  WORKSPACE=/checkpoint/$USER/fairseq2/wmt14


`train_mt.py <https://github.com/fairinternal/fairseq2/blob/main/examples/train_mt.py>`_ is the training script for this tutorial.
Start a process training the model on your machine::

  fairseq2 train examples/cli/train_mt.py train -w $WORKSPACE

If you're on a Slurm cluster you can use it by specifying which Slurm partition you want to use::

  fairseq2 train examples/cli/train_mt.py train -w $WORKSPACE --partition PARTITION

When starting ``fairseq2`` creates a folder for this run under the workspace folder, and tells it it's name.

This sends the training job to Slurm queue instead of running it locally.
The main process monitors the job and shows errors in case of crash.
You can safely exit it the actual training job, continues in the background, managed by Slurm.

:py:func:`fairseq2.cli.train` also has a ``--num_gpu`` flag, that controls how many GPUs to use for the training.
This is most useful in combination with Slurm to start
a multi-gpu training.

.. note:: you can use ``--help`` to get help on ``train`` supported arguments.

Note that ``train_mt.py`` doesn't implement early stopping,
and you didn't specified a number of steps.
The training runs as long as your cluster admins allow it.
To stop earlier you can pass ``--max_steps`` to ``train``.


Setting hyper-parameters
~~~~~~~~~~~~~~~~~~~~~~~~

You can also set hyper-parameters from the terminal::

  fairseq2 train examples/cli/train_mt.py train -w $WORKSPACE lr=1e-5

.. note:: The syntax for hyper-parameters ``lr=1e-5`` is different from the syntax for training settings.
  The difference is that the ``train`` function has settings to control how the experiment is *launched*,
  whereas the hyper-parameters control how the model is *training*.

The list of possible hyper-parameters depends on the specific script you wrote.
You can print the list by using ``help`` command::

  fairseq2 train examples/cli/train_mt.py help


W&B integration
~~~~~~~~~~~~~~~

.. TODO:: replace with tensorboard instructions

If you have a `W&B <https://wandb.ai>`_ account, ``fairseq2`` can upload the runs to a given project.
First install ``pip install wandb``, then run ``wandb login``.
If this is the first time you do this,
``wandb`` prompts instructions to get a secret token that allows it to connect to W&B in your name.
Then you can use::

  fairseq2 train examples/cli/train_mt.py --partition PARTITION wandb_project=<entity>/<project>

The job is then visible at https://wandb.ai/<entity>/<project>.

Be careful with typo, because ``wandb`` happily creates a new project without confirmation.
Now you should be able to see your job training and the ``train/loss`` metric decreasing overtime.

``evaluate``
------------

Now it's time to start evaluating your model on the validation set.
To do this first locate a snapshot created by fairseq2::

  ls $WORKSPACE/*/epoch*_step*

If you don't see any, it's probably because the job didn't made a checkpoint yet, so wait a bit more.
Call this folder ``SNAPSHOT``::

  SNAPSHOT=$(ls $WORKSPACE/*/epoch*_step* | tail -1)

You'll notice that this folder contains both weights of the model a ``hubconf.py`` file and a ``hubconf.yaml`` file.
Checkout the content of ``hubconf.yaml``::

  > ls $SNAPSHOT
  > cat $SNAPSHOT/hubconf.yaml

``hubconf.yaml`` captures hyper-parameters that where used to train this model.
``hubconf.py`` file is a copy of the ``train_mt.py`` training script,
and it can be used with the ``evaluate`` command to run evaluation::

  python $SNAPSHOT/hubconf.py evaluate

Similarly to ``train`` this uses your local machine by default,
but ``evaluate`` also has ``--partition`` and ``--num-gpus`` to run on a SLURM cluster.

Since the training script is saved next to the checkpoint, don't be afraid to edit it as you want between experiments.

Each experiment script can chose which metrics to compute,
and ``train_mt.py`` is are using the ``TranslationTask`` that computes BLEU, ChrF and ChrF++ during evaluation.
Note that ``fairseq2`` models save their tokenizer alongside the rest of the model weights
and so they can read and output strings.
This mean it can use ``sacrebleu`` and compute the BLEU scores on the reconstructed sentences.

If you used W&B for the training run, the evaluation results are also uploaded to W&B.
``fairseq2`` creates "run groups" to group the train and evaluate runs.

``eval_server``
---------------

Evaluating those intermediary checkpoint by hand can be tedious.
Use ``fairseq2 eval_server $WORKSPACE`` to monitor $WORKSPACE
and automatically run evaluation every time a new snapshot appears.
This allows to let your training job run at full speed and handle the evaluation asynchronously.
Typically you want the evaluation server to run in the background.
You can use a program like ``screen`` to do that::

  screen fairseq2 eval_server $WORKSPACE

.. note::
  you can fallback to the traditional behavior of stopping the training
  from time to time to run evaluation by passing `--eval_freq=1000` (in steps) to `fairseq2 train`.

``inference``
-------------

You can also interact with a model by running::

  python $SNAPSHOT/hubconf.py inference

this reads sentence from standard input and outputs their translations.
