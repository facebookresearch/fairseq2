Model Training
==============

In this tutorial, we will launch the training of a Machine Translation model, on WMT14 dataset.

This tutorial uses Huggingface datasets library
to load the data,
make sure you have it installed using
`pip install datasets`.

# train

First we need to create a workspace folder to save the trained models.
In this tutorial we will use `WORKSPACE=/checkpoint/$USER/fairseq2/wmt14`.

[examples/train_mt.py] is the training script we want to run.
`fairseq2 train examples/train_mt.py -w $WORKSPACE` will start a process training the model on your machine.
When starting `fairseq2` will create a subfolder inside the workspace for this run.
If you're on a Slurm cluster you can use it by specifying which Slurm partition you want to use:
`fairseq2 train examples/train_mt.py -w $WORKSPACE --partition PARTITION`
This will send the training job to Slurm queue.
The current process will monitor the job and will show errors in case of crash. You can safely exit it the actual training job will continue in the background.
`fairseq2 train` also has a `--num_gpu` flag, that controls how many GPU to use for the training.
This implies using SLURM, local multi gpu training isn't supported yet.
`fairseq2 train` assumes you have 8 GPUs per training host.
Note: you can always use `fairseq2 --help` or `fairseq2 CMD --help` to get help on the CLI.


## W&B integration

If you have a W&B account, `fairseq2` can upload the runs to a given project.
First install `pip install wandb`, then run `wandb login`.
If this is the first time you do this, `wandb` will prompt instructions to get a secret token that allows it to connect to W&B in your name.
Then you can use `fairseq2 train examples/train_mt.py --partition PARTITION wandb_project=<entity>/<project>`.
The job will appear at [https://wandb.ai/<entity>/<project>].

Be careful with typo, because `wandb` will happily create a new project without confirmation.
Note that `--partition` and `wandb_project` uses a different syntax.
This is because `--partition` is a flag of `fairseq2 train` itself, while `wandb_project` is one of the arguments of our training script `train_mt.py`.

Now you should be able to see your job training and the `train/loss` metric decreasing overtime.

Note that `train_mt.py` doesn't implement early stopping, nor we specified a number of steps.
The training will run undefinitively (or as long as your cluster admins allow it).
**TODO add a flag to contol max duration**

# evaluation

Now it's time to start evaluating our model on the validation set.
To do this first locate a snapshot made by fairseq2.
`ls $WORKSPACE/*/epoch*_step*`

If you don't see any, it's probably because the job didn't made a checkpoint yet, so wait a bit more.
Let's call this folder `SNAPSHOT=$(ls $WORKSPACE/*/epoch*_step* | tail -1)`.

Then you can run `fairseq2 evaluate $SNAPSHOT`.
This will use your local machine by default, but `fairseq2 evaluate` also has `--partition` and `--num-gpus`.

This will load the snapshot and run the evaluation.

Each experiment script can chose which metrics to compute, but in our case we are using the `TranslationTask` that computes BLEU, ChrF and ChrF++ during evaluation.
Note that `fairseq2` models include their own tokenizer and so they can read and output strings.
This mean we can use `sacrebleu` and compute the true BLEU scores using proper tokenization specific to the language at hand.

If you used W&B for the training run, the evaluation results will also be uploaded to W&B.
`fairseq2` will automatically create "run groups" and will add the evaluation result to the same groups than the training run.

# eval_server

Evaluating those intermediary checkpoint by hand can be tedious,
so it would be nice if we could automatically run evaluation every time a new snapshot is emitted.
`fairseq2 eval_server $WORKSPACE` does exactly that.
This allows to let your training job run at full speed and handle the evaluation asynchronously.
Typically you want the eval server to run in the background.
You can use a program like `screen` to do that:
`screen fairseq2 eval_server $WORKSPACE`.

**Note:** you can fallback to the traditional behavior of stopping the training from time to time to run evaluation by passing `--eval_freq=1000` (in steps) to `fairseq2 train`.

# inference

You can also interact with a model by running:
`fairseq2 inference $SNAPSHOT`
this will read input sentence from stdin and output their translation on the stdout.
