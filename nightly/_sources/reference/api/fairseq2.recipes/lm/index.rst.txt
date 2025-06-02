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

Instruction Fine-Tuning
=======================

Classes
~~~~~~~

.. autoclass:: InstructionFinetuneConfig

.. autoclass:: InstructionFinetuneCriterion

.. autoclass:: InstructionFinetuneDatasetSection

.. autoclass:: InstructionFinetuneUnit

.. autoclass:: InstructionLossEvalUnit


Functions
~~~~~~~~~

.. autofunction:: load_instruction_finetuner


Preference Fine-Tuning
======================

Classes (PO)
~~~~~~~~~~~~

.. autoclass:: POFinetuneConfig

.. autoclass:: POCriterionSection

.. autoclass:: POFinetuneDatasetSection


.. autoclass:: CpoFinetuneConfig

.. autoclass:: CpoFinetuneUnit

.. autoclass:: CpoFinetuneUnitHandler


.. autoclass:: DpoFinetuneConfig

.. autoclass:: DpoFinetuneUnit

.. autoclass:: DpoFinetuneUnitHandler


.. autoclass:: OrpoFinetuneConfig

.. autoclass:: OrpoFinetuneUnit

.. autoclass:: OrpoFinetuneUnitHandler


.. autoclass:: SimPOFinetuneConfig

.. autoclass:: SimPOFinetuneUnit

.. autoclass:: SimPOFinetuneUnitHandler

Functions (PO)
~~~~~~~~~~~~~~

.. autofunction:: load_po_finetuner


Text Generation
================

Classes (Text Generation)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TextGenerateConfig

.. autoclass:: TextGenerateDatasetSection

.. autoclass:: TextGenerateUnit

Functions (Text Generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_text_generator


Usage Examples
==============

- :ref:`tutorial-end-to-end-fine-tuning`

