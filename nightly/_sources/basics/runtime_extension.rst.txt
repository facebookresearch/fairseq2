.. _basics-runtime-extension:

:octicon:`plug` Runtime Extension
=================================

fairseq2 provides a flexible runtime extension system that allows you to extend its functionality without modifying the core codebase.
You can register custom models, architectures, and assets through a simple setup function.

Overview
--------

The extension system is built around a dependency injection container (learn more in :ref:`basics-design-philosophy`) that manages fairseq2's components.
Through this system, you can:

* Register new models and architectures
* Add custom assets and asset providers
* Extend fairseq2's capabilities
* All without touching the fairseq2 codebase

Basic Usage
-----------

First, create a setup function that registers your custom components:

.. code-block:: python

    # in my_package/__init__.py
    from fairseq2.runtime.dependency import DependencyContainer

    def setup_my_fairseq2_extension(container: DependencyContainer) -> None:
        # Your extension setup code here...
        pass


Then, there are two ways to set up your extension:

**For Python packages** - Use setuptools entry points in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."fairseq2.extension"]
    my_extension = "my_package.module:setup_my_fairseq2_extension"

**For standalone scripts** - Use ``init_fairseq2`` with extras:

.. code-block:: python

    from fairseq2 import init_fairseq2
    from my_package import setup_my_fairseq2_extension

    if __name__ == "__main__":
        init_fairseq2(extras=setup_my_fairseq2_extension)

Example Extension Setup
-----------------------

Here's a complete example that shows how to register assets, models, and architectures:

.. code-block:: python

    from fairseq2.runtime.config_registry import ConfigRegistrar
    from fairseq2.runtime.dependency import DependencyContainer
    from fairseq2.composition import register_package_assets, register_file_assets, register_dataset_family, register_model_family
    from my_package.models.my_custom_model import MyCustomModel, MyCustomModelConfig, create_my_custom_model

    def setup_my_fairseq2_extension(container: DependencyContainer) -> None:
        # Register custom objects here...
        container.register(...)

        # Register assets (yaml files) from your package, which extends fairseq2
        register_package_assets(container, "my_package.assets")

        # Or register assets from a file path, where you put your asset yaml files
        register_file_assets(container, Path("path/to/assets"))

        # Register model families (if any)
        register_model_family(
            container,
            "my_custom_model",  # model family name
            kls=MyCustomModel,  # model class
            config_kls=MyCustomModelConfig,  # model config class
            factory=create_my_custom_model,  # factory function
            # ... other parameters
        )

        # Register dataset families (if any)
        register_dataset_family(
            container,             # DependencyContainer instance
            "custom_dataset",      # family name
            CustomDataset,         # dataset class
            CustomDatasetConfig,   # config class
            opener=custom_opener   # opener function
        )

        # Register tokenizer families (if any)
        register_tokenizer_family(
            container,
            "custom_tokenizer",     # tokenizer family name
            CustomTokenizer,        # tokenizer class
            CustomTokenizerConfig,  # tokenizer config class
            loader=custom_loader,   # loader function
        )

        # Register model architectures
        arch = ConfigRegistrar(container, MyCustomModelConfig)

        @arch("my_custom_arch_variant")  # architecture name
        def my_custom_arch_variant() -> MyCustomModelConfig:
            config = MyCustomModelConfig()
            # ... customize your config here...
            return config

Error Handling
--------------

The extension system includes error handling to maintain system stability:

* Failed extensions log warnings by default
* Set ``FAIRSEQ2_EXTENSION_TRACE`` environment variable for detailed error traces
* Invalid extension functions raise ``fairseq2.composition.ExtensionError``

.. code-block:: bash

    export FAIRSEQ2_EXTENSION_TRACE=1

See Also
--------

* :doc:`/basics/assets` for more information on assets
