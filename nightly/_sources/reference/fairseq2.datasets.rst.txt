.. _api_datasets:

=================
fairseq2.datasets
=================

.. currentmodule:: fairseq2.datasets

The datasets module provides flexibility in creating and managing datasets for various tasks.
It supports both built-in datasets and custom dataset implementations.

Key Features
------------

- **Dataset Family Registration**: Datasets can be registered using ``register_dataset_family`` for seamless integration
- **Flexible Configuration**: Dataset configurations can be defined through YAML asset cards

Dataset Registration and Asset Cards
------------------------------------

Datasets in fairseq2 can be registered as follows:

1. **Register Dataset Families**:
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

    This function can potentially be called:

    - in ``Recipe.register()`` (read more in :ref:`basics-building-recipes`), or
    - in a fairseq2 extension function (read more in :ref:`basics-runtime-extension`).

2. **More Variants w/ Asset Cards**:
    Create YAML files describing your datasets in any of these locations:

    - Built-in cards: ``fairseq2/assets/cards/``
    - System-wide cards: ``/etc/fairseq2/assets/`` (overridden by ``FAIRSEQ2_ASSET_DIR`` if set)
    - User-specific cards: ``~/.config/fairseq2/assets/`` (overridden by ``FAIRSEQ2_USER_ASSET_DIR`` if set)
    - Recipe-local cards: Add to recipe config common section via ``extra_paths``:

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


Creating Custom Datasets
------------------------

Custom datasets can be created independently in recipes by:

1. Implementing a dataset class
2. Creating a configuration class
3. Registering the dataset family
4. Providing asset card(s) in YAML format

Here's a basic example:

.. code-block:: python

    from fairseq2.composition import register_dataset_family
    from fairseq2.datasets import DataReader

    # 1. Dataset Implementation
    class CustomDataset:
        def create_reader(self, ...) -> DataReader:
            # Implementation

    # 2. Configuration Class
    @dataclass
    class CustomDatasetConfig:
        """
        This configuration matches the keys after the top-level `dataset_config:` key
        in the YAML asset definition:

        ```yaml
        name: mydataset
        dataset_config:
            data: (all keys here must have a companion parameter in this config)
        ```
        """
        path: Path
        # Other config options

    # 3. Register in Recipe
    class YourRecipe(TrainRecipe):  # or EvalRecipe/GenerationRecipe
        @override
        def register(self, container: DependencyContainer) -> None:
            register_dataset_family(
                container,
                "custom_dataset",           # family name
                CustomDataset,              # dataset class
                CustomDatasetConfig,        # config class
                opener=open_custom_dataset  # opener function
            )

The corresponding dataset asset card in YAML format could be for example:

.. code-block:: yaml

    name: mydataset
    dataset_family: custom_dataset

    ---

    name: mydataset@user
    dataset_config:
        data: "/path/to/local/datasets/librilight/10h"

    ---

    name: mydataset@mycluster
    dataset_config:
        data: "/path/to/cluster/datasets/librilight/10h"

Advanced Dataset Opening
------------------------

While the basic ``opener`` function is sufficient for most use cases, fairseq2 also provides
an advanced opening mechanism through ``advanced_opener`` for cases where access to fairseq2's
dependency injection system is needed.

The key differences are:

* ``opener(config) -> Dataset``: Simple function that takes only the config
* ``advanced_opener(resolver, config) -> Dataset``: Takes a :class:`DependencyResolver` as first parameter

The ``DependencyResolver`` provides access to other services and objects registered in fairseq2,
making it useful for more complex dataset implementations that need to interact with other
parts of the system.

Example usage:

.. code-block:: python

    from fairseq2.dependency import DependencyResolver
    
    def my_advanced_opener(resolver: DependencyResolver, config: MyDatasetConfig) -> MyDataset:
        # Access other fairseq2 objects through the resolver
        some_object = resolver.get("object_name")

        # Use the object in dataset creation
        return MyDataset(config, some_object)

    # Register with advanced opener
    register_dataset_family(
        container,
        "my_dataset",
        MyDataset,
        MyDatasetConfig,
        advanced_opener=my_advanced_opener  # Note: don't provide both opener and advanced_opener
    )

.. note::
    You must provide either ``opener`` or ``advanced_opener``, but not both. For simple
    dataset implementations that don't need access to other fairseq2 objects, using
    the basic ``opener`` is recommended.

* :doc:`/reference/fairseq2.datasets.hub`
