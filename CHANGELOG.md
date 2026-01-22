# Changelog
All notable changes to fairseq2 are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.8.0] - TBD
- TBD

## [0.7.0] - Nov 4th, 2025
- `RecipeModel` is now callable and forwards the call to `RecipeModel.module`
  for a cleaner, more convenient syntax.
- A new `get_asset_download_manager` helper function to download assets in
  procedural code.
- A new `register_recipe_assets` helper function that can be used to register
  recipe-specific asset cards that cannot be (accidentally) overwritten by users.
  [More info](https://github.com/facebookresearch/fairseq2/pull/1373)
- Reference API documentation has been flattened and updated for better readability.
- Revised Wav2Vec 2.0 recipes have been merged back and are available under the
  recipes/wav2vec2 directory.

## [0.6.0] - Oct 7th, 2025
- `fairseq2.sharder` is deprecated. fairseq2 now expects parallelism strategies
  to be applied within model factories. This gives model authors full control
  over how parallelism is applied to their models. [More info](https://github.com/facebookresearch/fairseq2/pull/1349) 
- `Gangs` can now be used as a context manager, along with a new `maybe_get_current_gangs()`
  helper function. This feature is particularly useful in procedural programming,
  as it eliminates the need to pass a `Gangs` instance through every function call.
  [More info](https://facebookresearch.github.io/fairseq2/stable/concepts/gang.html#how-to-use-gangs-in-deeply-nested-functions)
- An experimental implementation of LLaMA 4 Scout model is now available.
- The recipe command line interface now accepts a new `--no-exit-on-error` flag
  to allow post-mortem debugging of recipe processes. [More info](https://github.com/facebookresearch/fairseq2/pull/1337)
- The optimizer and learning rate scheduler recipe configurations now support
  multiple parameter groups. This is in particular convenient for models that
  require more than one learning rate to train (e.g. GAN models). [More info](https://github.com/facebookresearch/fairseq2/pull/1332)
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
