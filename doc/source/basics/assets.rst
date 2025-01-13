.. _basics-assets:

===========================
:octicon:`container` Assets
===========================

.. currentmodule:: fairseq2.assets

In fairseq2, "assets" refer to the various components that make up a sequence or language modeling task, such as datasets, models, tokenizers, etc. These assets are essential for training, evaluating, and deploying models.
``fairseq2.assets`` provides API to load the different models using the "model cards" from different "stores".

Cards: YAML Files in fairseq2
-----------------------------

To organize these assets, fairseq2 uses a concept called "cards," which are essentially YAML files that describe the assets and their relationships.
For example, you can find all the "cards" in fairseq2 `here <https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/assets/cards>`__.
Cards provide a flexible way to define and manage the various components of an NLP task, making it easier to reuse, share, and combine different assets.

How Cards Help Organize Assets
------------------------------

* **Asset Definition**: Cards define the assets used in an NLP task, including datasets, models, tokenizers, and other resources.

* **Relationship Management**: Cards specify the relationships between assets, such as which dataset is used with which model or tokenizer.

* **Reusability**: Cards enable reusability of assets across different tasks and projects, reducing duplication and increasing efficiency.

* **Sharing and Collaboration**: Cards facilitate sharing and collaboration by providing a standardized way to describe and exchange assets.


How to Customize Your Assets
----------------------------

* How to add a dataset

    * Make sure that you have the dataset in place

    * Add the ``name``, ``dataset_family``, and ``data`` fields, which allows fairseq2 to find the corresponding dataset loader

    * For more detailed information about ``dataset_family``, please refer to :doc:`Dataset Loaders </reference/api/fairseq2.datasets/index>`

.. code-block:: yaml

    name: gsm8k_sft
    dataset_family: generic_instruction

    ---

    name: gsm8k_sft@awscluster
    data: "/data/gsm8k_data/sft"


* How to add a model

    * Make sure that you have the model checkpoint

    * Add the ``name`` and ``checkpoint`` fields

.. code-block:: yaml

    name: llama3_2_1b@awscluster
    checkpoint: "/models/Llama-3.2-1B/original/consolidated.00.pth"


Advanced Topics
---------------

Model Store
~~~~~~~~~~~

A store is a place where all the model cards are stored. In fairseq2, a store is accessed via 
:py:class:`fairseq2.assets.AssetStore`. Multiple stores are allowed. By default, fairseq2 will look up the following stores:

* System asset store: Cards that are shared by all users. By default, the system store is `/etc/fairseq2/assets`,
    but this can be changed via the environment variable `FAIRSEQ2_ASSET_DIR`

* User asset store: Cards that are only available to the user. By default, the user store is 
    `~/.config/fairseq2/assets`, but this can be changed via the environment variable `FAIRSEQ2_USER_ASSET_DIR`

To register a new store, implement a :py:class:`fairseq2.assets.AssetMetadataProvider` and add them to 
:py:class:`fairseq2.assets.asset_store`. Here is an example to register a new directory as a model store:

.. code-block:: python

    from pathlib import Path
    from fairseq2.assets import FileAssetMetadataProvider, asset_store

    my_dir = Path("/path/to/model_store")
    asset_store.metadata_providers.append(FileAssetMetadataProvider(my_dir))


Model Card
~~~~~~~~~~

A model card is a .YAML file that contains information about a model and instructs a 
:py:class:`fairseq2.models.utils.generic_loaders.ModelLoader` on how to load the model into the memory. Each model card
must have 2 mandatory attributes: `name` and `checkpoint`. `name` will be used to identify the model card, and it must
be unique `across` all 
fairseq2 provides example cards for different LLMs in
:py:mod:`fairseq2.assets.cards`. 

In fairseq2, a model card is accessed via :py:class:`fairseq2.assets.AssetCard`. Alternatively, one can call 
`fairseq2.assets.AssetMetadataProvider.get_metadata(name: str)` to get the meta data of a given model card name.

See Also
--------

- :doc:`Datasets </reference/api/fairseq2.datasets/index>`
