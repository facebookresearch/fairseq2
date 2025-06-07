:github_url: https://github.com/facebookresearch/fairseq2

=================================
Welcome to fairseq2 Documentation
=================================

fairseq2 is a sequence modeling toolkit that allows researchers and developers
to train custom models for translation, summarization, language modeling, and
other content generation tasks.

.. grid:: 3

    .. grid-item-card::  Quick Start
        :link: tutorial-end-to-end-fine-tuning
        :link-type: ref

        Run a quick start tutorial.

    .. grid-item-card::  Basics
        :link: basics-overview
        :link-type: ref

        Get familiar with fairseq2.

    .. grid-item-card::  API Reference
        :link: reference-api
        :link-type: ref

        Jump to the code.


Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation/index
   getting_started/quick_start

.. toctree::
   :maxdepth: 1
   :caption: Basics

   basics/overview
   basics/design_philosophy
   basics/cli
   basics/assets
   basics/data_pipeline
   basics/parquet_dataloader
   basics/ckpt
   basics/recipe
   basics/runtime_extensions
   basics/gang
   basics/trainer

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/end_to_end_fine_tuning
   tutorials/preference_optimization
   tutorials/monitor_your_experiments
   tutorials/presets
   tutorials/pudb
   tutorials/models

.. toctree::
   :maxdepth: 2
   :caption: Notebooks

   notebooks/datapipeline
   notebooks/dataset_gsm8k_sft
   notebooks/parquet_dataloader
   notebooks/hf_parquet_integration
   notebooks/models/load_model

.. toctree::
   :maxdepth: 1
   :caption: FAQ

   faq/contributing

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/api/index
   reference/bibliography

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
