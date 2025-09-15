.. _dataset_hub:

=====================
fairseq2.datasets.hub
=====================

.. currentmodule:: fairseq2.datasets.hub

The dataset hub provides a centralized way to manage and access datasets in fairseq2.
It offers functionality for discovering available datasets, loading datasets, and working
with custom dataset configurations.

Dataset Registration and Asset Cards
------------------------------------

Datasets in fairseq2 can be registered in two ways:

1. **Through Dataset Families**:
    Use ``register_dataset_family`` to register custom datasets:

    .. code-block:: python

         from fairseq2.composition import register_dataset_family

        register_dataset_family(
            container,             # DependencyContainer instance
            "custom_dataset",      # family name
            CustomDataset,         # dataset class
            CustomDatasetConfig,   # config class
            opener=custom_opener   # opener function
        )

2. **Through Asset Cards**:
    Create YAML files describing your datasets in any of these locations:

    - Built-in: ``fairseq2/assets/cards/``
    - System-wide: ``/etc/fairseq2/assets/``
    - User-specific: ``~/.config/fairseq2/assets/``
    - Recipe-local: Add to recipe config common section via ``extra_paths``:

    .. code-block:: yaml

        # Example: Adding recipe-local asset paths in config (absolute path)
        common:
            assets:
                extra_paths: ["/path/to/assets"]
        
        # or relative path
        # e.g. if the common section is in /path/to/recipe/config.yaml
        # it recursively retrieves assets from /path/to/recipe/assets
        common:
            assets:
                extra_paths: ["${dir}/assets"]

Core Classes
------------

DatasetHub
~~~~~~~~~~

.. autoclass:: DatasetHub
    :members:
    :show-inheritance:

    The main hub class for managing datasets. Provides methods for:

    - Listing available datasets (:meth:`iter_cards`)
    - Loading dataset configurations (:meth:`get_dataset_config`)
    - Opening datasets (:meth:`open_dataset`)
    - Opening custom datasets (:meth:`open_custom_dataset`)

    Example usage:

    .. code-block:: python

        get_my_dataset_hub = DatasetHubAccessor(
            MY_DATA_FAMILY_NAME, kls=MyDataset, config_kls=MyDatasetConfig
        )

        # Get the dataset hub
        hub = get_my_dataset_hub()

        # List all available datasets
        for card in hub.iter_cards():
            print(f"Found dataset: {card.name}")

        # Load a dataset configuration
        config = hub.get_dataset_config("my_dataset")

        # Open a dataset
        dataset = hub.open_dataset("my_dataset")

        # Open a custom dataset with specific configuration
        custom_dataset = hub.open_custom_dataset(config)

DatasetHubAccessor
~~~~~~~~~~~~~~~~~~

.. autoclass:: DatasetHubAccessor
    :members:
    :show-inheritance:

    Factory class for creating :class:`DatasetHub` instances for specific dataset families.
    Can be used by dataset implementors to create hub accessors for their dataset families.

    Example implementation of a dataset hub accessor:

    .. code-block:: python

        from fairseq2.datasets.hub import DatasetHubAccessor
        from my_dataset import MyDataset, MyDatasetConfig

        # Create a hub accessor for your dataset family
        get_my_dataset_hub = DatasetHubAccessor(
            "my_dataset_family",  # dataset family name
            MyDataset,           # concrete dataset class
            MyDatasetConfig      # concrete dataset config class
        )

Exceptions
----------

DatasetNotKnownError
~~~~~~~~~~~~~~~~~~~~

.. autoexception:: DatasetNotKnownError
    :show-inheritance:

    Raised when attempting to open a dataset that is not registered in the asset store.

    Example:

    .. code-block:: python

        try:
            dataset = hub.open_dataset("non_existent_dataset")
        except DatasetNotKnownError as e:
            print(f"Dataset not found: {e.name}")

DatasetFamilyNotKnownError
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: DatasetFamilyNotKnownError
    :show-inheritance:

    Raised when attempting to access a dataset family that is not registered in the system.

    Example:

    .. code-block:: python

        try:
            hub = DatasetHubAccessor("unknown_family", MyDataset, MyConfig)()
        except DatasetFamilyNotKnownError as e:
            print(f"Dataset family not found: {e.name}")

See Also
--------

- :ref:`tokenizer_hub` for tokenizer hub reference documentation.
- :ref:`api-models-hub` for model hub reference documentation.