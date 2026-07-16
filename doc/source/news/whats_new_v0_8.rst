====================================
:octicon:`report` What's New in v0.8
====================================

fairseq2 v0.8 is now available on PyPI::

    pip install --upgrade fairseq2

(v0.8.1 is identical to v0.8.0 minus an accidentally bundled artifact.)

🤖 **New Models & HuggingFace Integration**
============================================

**Gemma3n (E2B/E4B)**
    Text + audio inference and SFT training with HuggingFace parity.

**OLMo2/3**
    Text inference and SFT training with HuggingFace parity.

**Generic HuggingFace Model Integration**
    Load, shard, and train any HuggingFace CausalLM directly via
    ``HgCausalLMAdapter`` — no native fairseq2 reimplementation needed.
    Includes FSDP sharding, HF tokenizer integration, and SFT recipe support.
    Primarily useful for training; use vLLM or the HF interface for inference.

🚀 **New Features**
====================

**S3 Checkpoint Storage**
    fsspec integration enables ``--checkpoint-dir s3://bucket/path/``
    for saving and loading checkpoints to and from S3. Requires ``s3fs``.

**HuggingFace Export Revised**
    HF export now runs in an isolated process with saved command line
    and logs for debugging. Improved support for gated HF models.

**Context Managers for Procedural Programming**
    New ``GangContext``, ``DeviceContext``, ``DataTypeContext``, and
    ``current_dtype`` eliminate the need to thread state through nested
    function calls.

**PyTorch 2.9 / 2.10 Support**
    PyTorch 2.9.1 and 2.10 (forward compatibility) are now supported.
    Breaking changes to LR scheduler return types in 2.9 have been
    addressed.

**Environment Variable Interpolation**
    Config files now support ``${env:<NAME>}`` to interpolate environment
    variables.

**CLI Improvements**
    ``--no-rich`` flag disables rich text output for log parsing.
    ``get_family`` utility functions for detecting model families.

🔧 **Fixes**
=============

**Flash3SDPA**
    Updated for the ``flash-attn-3`` v3.0.0 package API
    (``flash_attn_3._C`` / ``torch.ops.flash_attn_3``) in addition to
    the legacy ``flash_attn_3_cuda`` module.

**cross_entropy**
    ``reduction="mean"`` now properly excludes padding tokens from the
    denominator.

**Data Pipeline**
    Fixed sampling bug when ``allow_repeats=False`` with many pipelines.
    Fixed ``DataParallelFacade`` weakref errors.

**Other**
    ``datetime.utcnow()`` deprecation warnings cleaned up.
    Fixed WER calculation to use lists instead of tensors.

💥 **Breaking Changes**
========================

Backward-compatibility shims are provided where noted and will be
removed in v0.12.

**Trainer, Evaluator, Generator Moved**
    Moved from ``fairseq2.recipe`` to the ``fairseq2`` package root.
    Backward compat shims provided until v0.12.

**RecipeModel Deprecated**
    Access the model directly via ``.module`` instead.

**resolve_optional → maybe_resolve**
    Shim provided until v0.12.

**ModelCheckpointLoader API Revised**
    Shim provided until v0.12.

**LM Recipes Restructured**
    ``text_generate`` renamed to ``generate``. SFT configs
    removed/renamed. Recipe config classes changed.

**ParquetDataset**
    ``pq.ParquetDataset`` replaced with ``pyarrow.dataset`` interface.

**Tensor Sharded Modules**
    Refactored embedding, projection, FFN, and attention sharding
    modules.

📢 **A Note on Maintenance**
=============================

Our team's priorities have shifted toward enabling collaboration between
FAIR and other orgs — work that is not being developed within fairseq2.
As a result, fairseq2 is moving into **maintenance mode** rather than
active development for the time being.

To set expectations clearly:

- **Not deprecated** — fairseq2 is still here and usable
- **No active feature development** for now
- Critical fixes are still on the table, but we cannot commit to timelines
- No guaranteed SLA on issues or requests

This could change depending on where things go, but for now this is
where we are.

Thanks for using fairseq2 and for understanding.

— fairseq2 team
