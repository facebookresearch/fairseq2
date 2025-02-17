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

    name: gsm8k_sft@user
    data: "/data/gsm8k_data/sft"


* How to add a model

    * Make sure that you have the model checkpoint

    * Add the ``name`` and ``checkpoint`` fields

.. code-block:: yaml

    name: llama3_2_1b@user
    checkpoint: "/models/Llama-3.2-1B/original/consolidated.00.pth"


Advanced Topics
---------------

Asset Store
~~~~~~~~~~~

A store is a place where all the model cards are stored. In fairseq2, a store is accessed via 
:py:class:`fairseq2.assets.AssetStore`. By default, fairseq2 will look up the following paths to
find asset cards:

* System: Cards that are shared by all users. By default, the system store is `/etc/fairseq2/assets`,
    but this can be changed via the environment variable `FAIRSEQ2_ASSET_DIR`.

* User: Cards can be created with name with the suffix ``@user`` (`e.g.` ``llama3_2_1b@user``) that are only available to the user.
    By default, the user store is ``~/.config/fairseq2/assets``, but this can be changed via the environment variable `FAIRSEQ2_USER_ASSET_DIR`.

Here is an example on how to register a new directory to the a asset store:

.. code-block:: python

    from pathlib import Path
    from fairseq2.assets import FileAssetMetadataLoader, StandardAssetStore

    def register_my_models(asset_store: StandardAssetStore) -> None:
        my_dir = Path("/path/to/model_store")
        loader = FileAssetMetadataLoader(my_dir)
        asset_provider = loader.load()
        asset_store.metadata_providers.append(asset_provider)


Asset Card
~~~~~~~~~~

A model card is a .YAML file that contains information about an asset such as
a model, dataset, or tokenizer. Each asset card must have a mandatory attribute
`name`. `name` will be used to identify the relevant asset, and it must be
unique across all fairseq2 provides example cards for different assets in
:py:mod:`fairseq2.assets.cards`.

See Also
--------

- :doc:`Datasets </reference/api/fairseq2.datasets/index>`
