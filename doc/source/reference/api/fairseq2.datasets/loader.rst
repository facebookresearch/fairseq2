.. module:: fairseq2.datasets.loader

===============
Dataset Loaders
===============

.. autoclasstree:: fairseq2.datasets.loader
   :full:
   :zoom:

The dataset loader system in fairseq2 provides a flexible and extensible way to load different types of datasets.
The system uses the concept of dataset families to organize and manage different dataset formats.

Dataset Family
--------------

A dataset family represents a specific format or structure of data that requires specialized loading logic.
Each dataset is associated with a family through the ``dataset_family`` field in its asset card.

Built-in Dataset Families
^^^^^^^^^^^^^^^^^^^^^^^^^

fairseq2 includes several built-in dataset families:

- ``generic_text``: For plain text datasets
- ``generic_parallel_text``: For parallel text/translation datasets
- ``generic_asr``: For automatic speech recognition datasets
- ``generic_speech``: For speech-only datasets
- ``generic_instruction``: For instruction-tuning datasets
- ``generic_preference_optimization``: For preference optimization datasets

Example Asset Card
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    name: librispeech_asr
    dataset_family: generic_asr
    tokenizer: "https://example.com/tokenizer.model"
    tokenizer_family: char_tokenizer

Core Components
---------------

DatasetLoader Protocol
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fairseq2.datasets.loader.DatasetLoader
   :members:
   :special-members: __call__

AbstractDatasetLoader
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fairseq2.datasets.loader.AbstractDatasetLoader
   :members:
   :show-inheritance:

DelegatingDatasetLoader
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: fairseq2.datasets.loader.DelegatingDatasetLoader
   :members:
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: fairseq2.datasets.loader.is_dataset_card

.. autofunction:: fairseq2.datasets.loader.get_dataset_family

Usage Examples
--------------

1. Loading a Dataset Using Family
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fairseq2.datasets import load_text_dataset

    # Load using dataset name (will look up asset card)
    dataset = load_text_dataset("my_text_dataset")

    # Load using explicit asset card
    card = AssetCard(name="custom_dataset", dataset_family="generic_text")
    dataset = load_text_dataset(card)

2. Registering a Custom Dataset Loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from fairseq2.datasets import DelegatingDatasetLoader

    # Create your custom dataset loader
    class MyCustomDatasetLoader(AbstractDatasetLoader[MyDataset]):
        def _load(self, path: Path, card: AssetCard) -> MyDataset:
            return MyDataset.from_path(path)

    # Register with a family name
    loader = MyCustomDatasetLoader()
    load_dataset = DelegatingDatasetLoader()
    load_dataset.register("my_custom_family", loader)

See Also
--------

- :doc:`Text Dataset </reference/api/fairseq2.data.text/index>`
