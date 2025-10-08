====================================
:octicon:`report` What's New in v0.5
====================================

fairseq2 v0.5 represents a major milestone with significant improvements across
the entire toolkit. This release focuses on simplifying user onboarding,
enhancing performance, and expanding integration capabilities.

🚀 **Recipe Authoring & User Experience**
=========================================

**Simplified Recipe APIs**
    Significant overhaul and simplification of recipe authoring APIs to ease
    onboarding to fairseq2 and make fairseq2 features more discoverable to new users.
    The new APIs provide a more intuitive and streamlined experience for both
    beginners and advanced users.

**Enhanced Discoverability**
    fairseq2 features are now more discoverable, with improved documentation,
    better API organization, and clearer examples to help users get started quickly.

🤖 **New Model Support**
========================

**Qwen 2.5 & Qwen 3 Models**
    Parity-checked implementations of dense Qwen 2.5 and Qwen 3 models have been
    added, expanding the range of state-of-the-art language models available in fairseq2.

**New Language Model Pretraining Recipe**
    Implementation of a new language model pretraining recipe, parity-checked by a
    LLaMA-3 8B - 1T DCLM Baseline run. This provides a robust foundation for
    training large language models from scratch.

🔗 **Hugging Face Integration**
===============================

**Native Checkpoint Support**
    Native support for reading and writing Hugging Face checkpoints for models that
    implement necessary integration APIs, enabling easier leverage of the Hugging Face
    ecosystem during async evaluation jobs (*e.g.*, vLLM) within fairseq2 training.

**Hugging Face Tokenizers**
    Native support for Hugging Face tokenizers across all relevant fairseq2 APIs,
    providing seamless integration with the broader NLP ecosystem.

**Asset Cards with HF Hub**
    Asset Cards now support the "hg://" scheme to download models, tokenizers, and
    datasets directly from the Hugging Face Hub, making it easier to work with
    community models and datasets.

📊 **Data Processing & Batching**
=================================

**Unified Batching APIs**
    Consolidation of padded and packed batching APIs under a single :ref:`batch_layout` type.
    All ``fairseq2.nn`` modules are updated to handle both modes consistently, simplifying
    data processing workflows.

**High-Performance Data Pipeline**
    A new high-performance C++-based data pipeline packing operation specifically
    designed for language model pretraining jobs, significantly improving data
    throughput and training efficiency.

⚡ **Performance & Memory Optimizations**
=========================================

**Flash3 Attention Support**
    Support for (varlen) flash3 attention, with ``torch.compile`` integration, providing
    state-of-the-art attention performance and memory efficiency.

**Torch.compile Integration**
   Activation memory budget setting of ``torch.compile``'s min-cut partitioner is
   now exposed in all first-party recipes, giving users fine-grained control over
   memory usage during compilation.

💾 **Advanced Checkpoint Management**
=====================================

**New Checkpoint Format**
   A new checkpoint format serves as a lightweight alternative to PyTorch DCP,
   offering similar dynamic resharding capabilities. No need to set up process
   groups for checkpoint saving or loading, thanks to integration with the new
   3-D model sharding APIs.

**User-Inspectable Checkpoints**
   Generated checkpoints are regular, user-inspectable PyTorch tensor files
   (i.e., ".pt") for easier troubleshooting and analysis.

**Asynchronous Checkpoint Manager**
   A new asynchronous checkpoint manager is tightly integrated with the new format.
   It is fully deterministic and includes special handling of NFS lookup caches
   to prevent race conditions in async evaluation jobs.

**Model-Only Checkpoints**
   Ability to save only models instead of entire checkpoints during training,
   especially helpful for short-running post-training jobs to reduce disk overhead.

⚙️ **Advanced Model Sharding**
==============================

**3-D Model Sharding API**
   A new, extensible 3-D model sharding API supported both in offline (checkpoint)
   and online (training) settings, enabling more flexible and efficient distributed
   training configurations.

📈 **Metrics & Monitoring**
===========================

**Revised Metric API**
   The metric API has been revised for greater flexibility and no longer requires
   individual ``MetricBag`` subclasses for use with recipe units, simplifying
   custom metric implementation.

🔧 **Architecture & Maintainability**
=====================================

**Dependency Injection Framework**
   Many internal APIs revised to use a new, lightweight yet full-featured dependency
   injection framework for better testability and maintainability.

**API Improvements**
   Numerous improvements and extensions to various internal APIs, enhancing overall
   code quality and developer experience.

🎯 **Migration Guide**
======================

When upgrading to v0.5, users should be aware of the following key changes:

**Recipe APIs**
   Recipe authoring APIs have been significantly simplified. Existing recipes
   may need updates to work with the new APIs. Check the updated documentation
   and examples for migration guidance.

**BatchLayout Changes**
   The consolidation of batching APIs under ``BatchLayout`` may require updates
   to custom data processing code. The new unified API provides better consistency
   and performance.

**Checkpoint Format**
   While the new checkpoint format offers significant advantages, existing
   checkpoints will need to be converted or retrained. The new format provides
   better performance and easier troubleshooting capabilities.

**What's Next**
===============

The fairseq2 team continues to work on:

- Enhanced model support and integrations
- Further performance optimizations
- Expanded tutorial and documentation coverage
- Community-driven feature requests

Stay tuned for future releases and improvements!
