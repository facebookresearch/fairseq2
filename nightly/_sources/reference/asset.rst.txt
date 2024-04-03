fairseq2.assets
===============
.. body

.. currentmodule:: fairseq2.assets

``fairseq2.asset`` provides API to load the different model using the "model cards" from different "stores".


.. autosummary::
    :toctree: generated/data

    AssetStore
    AssetCard
    AssetMetadataProvider

Model store
~~~~~~~~~~~

A store is a place where all the model cards are stored. In fairseq2, a store is accessed via 
`fairseq2.assets.AssetStore`. Multiple stores are allowed. By default, fairseq2 will look up the following stores:

* System asset store: Cards that are shared by all users. By default, the system store is `/etc/fairseq2/assets`,
    but this can be changed via the environment variable `FAIRSEQ2_ASSET_DIR`

* User asset store: Cards that are only available to the user. By default, the user store is 
    `~/.config/fairseq2/assets`, but this can be changed via the environment variable `FAIRSEQ2_USER_ASSET_DIR`

To register a new store, implement a :py:class:`fairseq2.assets.AssetMetadataProvider` and add them to 
:py:class:`fairseq2.assets.asset_store`. Here is an example to register a new directory as a model store:

    from pathlib import Path
    from fairseq2.assets import FileAssetMetadataProvider, asset_store

    my_dir = Path("/path/to/model_store")
    asset_store.metadata_providers.append(FileAssetMetadataProvider(my_dir))


Model card
~~~~~~~~~~~

A model card is a .YAML file that contains information about a model and instructs a 
:py:class:`fairseq2.models.utils.generic_loaders.ModelLoader` on how to load the model into the memory. Each model card
must have 2 mandatory attributes: `name` and `checkpoint`. `name` will be used to identify the model card, and it must
be unique _across_ all 
fairseq2 provides example cards for differen LLMs in
`fairseq2.assets.cards`. 

In fairseq2, a model card is accessed via :py:class:`fairseq2.assets.AssetCard`. Alternatively, one can call 
`fairseq2.assets.AssetMetadataProvider.get_metadata(name: str)` to get the meta data of a given model card name.
