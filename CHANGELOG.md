# Changelog
All notable changes to fairseq2 are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.6.0] - TBD
- The optimizer and learning rate scheduler recipe configurations now support
  multiple parameter groups. This is in particular convenient for models that
  require more than one learning rate to train (e.g. GAN models). Check out
  [this PR](https://github.com/facebookresearch/fairseq2/pull/1332) for details.
- The `regime.save_model_only` recipe option now accepts 'all' and 'all_but_last'
  as alternatives to a boolean value. Setting the option to 'all' is equivalent
  to `True` and means that only the model state is saved during checkpointing.
  This is beneficial for short-lived training jobs where the user does not
  expect to resume the job but requires frequent snapshots of the model for
  evaluation purposes. In this mode, checkpointing is faster and disk space is
  saved by avoiding the storage of trainer, optimizer, and data reader states.

  The 'all_but_last' option is similar to 'all', except that the full state is
  saved only for the last checkpoint while all previous checkpoints will store
  only the model state, as in the 'all' mode. This is helpful to avoid
  unnecessary disk space use if the user does not plan to branch off the
  training from a previous checkpoint.
- The default resume mode for Weights & Biases metric recorder changed from
  'allow' to ``None`` to avoid noisy, safe-to-ignore warnings when resuming a
  preempted job.
