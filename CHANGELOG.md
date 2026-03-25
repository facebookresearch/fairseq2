# Changelog
All notable changes to fairseq2 are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.8.0] - March 25th, 2026
- fsspec integration for remote filesystem support. Checkpoints can be saved to and loaded from S3 via `--checkpoint-dir s3://bucket/path/`. Requires `s3fs`. (#1126)
- New `GlobalFileSystem` replaces `LocalFileSystem` as default, dispatching to the appropriate backend based on URI scheme. (#1126)
- PyTorch 2.9.1 and 2.10 (forward compatibility) are now supported. PyTorch 2.9 introduced breaking changes to LR scheduler return types, which have been addressed. (#1477, #1491, #1456)
- **Breaking**: Trainer, evaluator, generator, validator, and task moved from `fairseq2.recipe` to `fairseq2` package root. (#1417)
- **Breaking**: LM recipes restructured: `text_generate` renamed to `generate`, SFT configs removed/renamed, recipe config classes changed. (#1431, #1432, #1433)
- **Breaking**: `RecipeModel` is deprecated. Access the model directly via `.module` instead. (#1403)
- **Breaking**: `pq.ParquetDataset` replaced with `pyarrow.dataset` interface. (#1490)
- **Breaking**: `resolve_optional` renamed to `maybe_resolve`. (#1462)
- **Breaking**: Revised `ModelCheckpointLoader` API. (#1475)
- **Breaking**: Refactored tensor sharded modules (embedding, projection, FFN, attention). (#1476)
- New context managers for procedural programming: `GangContext`, `DeviceContext`, `DataTypeContext`, `current_dtype`. Eliminates need to pass state through nested function calls. (#1474, #1473, #1464)
- `CheckpointManager`, `Optimizer`, and `LRScheduler` now exposed in `RecipeContext`. (#1461)
- Synchronous asset loading across ranks for models and tokenizers. Use when all ranks need identical assets loaded simultaneously. (#1429, #1426)
- `CheckpointManager.register_save_hook` allows custom logic during checkpoint saves. (#1439)
- Config files now support `${env:<NAME>}` to interpolate environment variables. (#1435)
- `--no-rich` CLI flag disables rich text output for log parsing. (#1421)
- Hugging Face export now runs in isolated process with saved command line and logs for debugging. (#1459, #1458, #1437, #1434)
- Improved support for gated Hugging Face models. (#1422)
- `get_family` utility functions for detecting model families. (#1454)
- Gemma3n model family (E2B/E4B) with text + audio inference and SFT training. (#1496)
- Generic HuggingFace model integration: load, shard, and train any HuggingFace CausalLM model directly through `HgCausalLMAdapter` without requiring a native fairseq2 reimplementation. Includes FSDP sharding, HF tokenizer integration, and SFT recipe support. (#1479)
- `AssetDownloadManager` gains `local_only` parameter and custom download subpath support. (#1423, #1425)
- Recipes now set Python `random` and `numpy` seeds for reproducibility. (#1419)
- Wandb metric recorder now respects wandb environment variables. (#1440)
- Improved `share_parameters` implementation. (#1484)
- Fixed `cross_entropy` with `reduction="mean"` to properly exclude padding tokens from the denominator. (#1455)
- Fixed `Flash3SDPA` to support the `flash-attn-3` v3.0.0 package API (`flash_attn_3._C` / `torch.ops.flash_attn_3`) in addition to the legacy `flash_attn_3_cuda` module. (#1495)
- Fixed data pipeline sampling bug when `allow_repeats=False` with many pipelines. (#1471)
- Fixed `DataParallelFacade` weakref errors. (#1447, #1436)
- Fixed WER calculation to use lists instead of tensors. (#1413)

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
