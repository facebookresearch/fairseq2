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
- **Multiple Asset Sources**: Dataset assets can be loaded from various locations:
  
  - Built-in: ``fairseq2/assets/cards/``
  - System-wide: ``/etc/fairseq2/assets/``
  - User-specific: ``~/.config/fairseq2/assets/``
  - Recipe-local: Specified via ``extra_paths`` in recipe config

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
                opener=open_custom_dataset  # optional opener function
            )

.. toctree::
    :maxdepth: 1

    hub