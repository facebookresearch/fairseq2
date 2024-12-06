===================
fairseq2.recipes.lm
===================

.. module:: fairseq2.recipes.lm

Overview
========
The ``fairseq2.recipes.lm`` module provides utilities and recipes for language model training and fine-tuning.
This includes tools for both pre-training and instruction tuning of language models.

Key Features
============
- Language model pre-training utilities
- Instruction fine-tuning support
- CLI setup for language model training
- Common training recipes and configurations

Components
==========

Functions
---------

.. autofunction:: _setup_lm_cli
   :noindex:

The ``_setup_lm_cli`` function configures command-line interface options for language model training and fine-tuning tasks.

Submodules
----------

.. toctree::
    :maxdepth: 1

    instruction_finetune

The ``instruction_finetune`` module provides specialized utilities for instruction-based fine-tuning of language models.

Usage Examples
==============

- :ref:`tutorial-end-to-end-fine-tuning`
