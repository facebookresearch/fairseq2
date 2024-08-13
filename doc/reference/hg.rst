Hugging Face Recipes
====================
.. body

.. currentmodule:: fairseq2.recipes.hg

``fairseq2.recipes.hg`` is a tool for evaluating fairseq2 models on HuggingFace's `datasets`_ using different metrics available in the `evaluate`_ and against `transformers`_ baselines. See an example of evaluating Automatic Speech Recognition (ASR) models in the :ref:`asr-example`_.

.. _datasets: https://huggingface.co/docs/datasets/
.. _evaluate: https://huggingface.co/docs/evaluate/en/index
.. _transformers: https://huggingface.co/docs/transformers/en/index

For this API to work, you need to have the following libraries installed:

- evaluate
- datasets
- transformers

Current supported evaluation tasks in ``fairseq2.recipes.hg``:

- Automatic Speech Recognition (ASR)
- Machine Translation (MT) - In development

For each task, there is a default preset configuration that manages the dataset, model to be used, as well as the I/O input (where to store the results, precision for model loading, etc.)

.. _asr-example:

Usage Example
-------------

In the example below, we evaluate a ASR model (default Wav2Vec) using the default preset configuration (LibriSpeech dataset, using the BLEU metric):

.. code-block:: bash

   $ fairseq2 hg asr /path/to/output_dir

   ...
   [08/11/24 20:23:56] INFO     fairseq2.recipes.hg.evaluator - Running evaluation on 1 device(s).
   [08/11/24 20:24:22] INFO     fairseq2.recipes.hg.evaluator - Eval Metrics - BLEU: 0.597487 | Elapsed Time: 26s | Wall Time: 27s | brevity_penalty: 0.9428731438548749 |
                                length_ratio: 0.9444444444444444 | precisions: [0.8235294117647058, 0.6666666666666666, 0.5384615384615384, 0.5454545454545454] |
                                reference_length: 18 | translation_length: 17
                       INFO     fairseq2.recipes.hg.evaluator - Evaluation complete in 27 seconds

**Overriding Default Configuration:**

You can override the default configuration by specifying parameters directly or a config file. For example, to use a different model or adjust evaluation settings, modify your command as follows:

.. code-block:: bash

   $ fairseq2 hg asr /tmp/fairseq2/ --config max_samples=2 model_name=openai/whisper-tiny.en dtype=torch.float32

   ...   
   [08/11/24 20:30:35] INFO     fairseq2.recipes.hg.evaluator - Eval Metrics - BLEU: 0.468458 | Elapsed Time: 3s | Wall Time: 4s | brevity_penalty: 1.0 | length_ratio:    
                                1.1666666666666667 | precisions: [0.6666666666666666, 0.5263157894736842, 0.4117647058823529, 0.3333333333333333] | reference_length: 18 | 
                                translation_length: 21                                                                                                                     
                       INFO     fairseq2.recipes.hg.evaluator - Evaluation complete in 4 seconds                                                                           


ASR Evaluation
--------------

The module includes specific functionality for ASR evaluation. It supports evaluation of ASR models on Hugging Face datasets, including Wav2Vec2 and Whisper models.

**Configuration:**

The `AsrEvalConfig` class holds the configuration for ASR evaluation. It defines parameters for data processing, model evaluation, and other settings. The configuration can be overridden through the command line or a configuration file.

.. autoclass:: fairseq2.recipes.hg.asr_eval.AsrEvalConfig
   :members:
   :show-inheritance:

**Data Processing:**

.. currentmodule:: fairseq2.recipes.hg.asr_eval

.. autosummary::
    :toctree: generated/hg/data_processing

    extract_features
    to_batch
    prepare_dataset

**Evaluation Functions:**

.. autosummary::
    :toctree: generated/hg/evaluation_functions

    load_asr_evaluator
    load_wav2vec2_asr_evaluator
    load_hg_asr_evaluator

Utilities
---------

.. currentmodule:: fairseq2.recipes.hg.dataset

.. autosummary:: 
    :toctree: generated/hg/utilities

   create_hf_reader

Extending the Framework
-----------------------

(Details come soon..)

While ASR evaluation is currently implemented, the framework is designed to be extensible. You can implement additional recipes and integrate other Hugging Face models and datasets as needed.