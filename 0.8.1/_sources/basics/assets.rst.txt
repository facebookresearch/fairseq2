.. _basics-assets:

===========================
:octicon:`container` Assets
===========================

.. currentmodule:: fairseq2.assets

In fairseq2, "assets" refer to the various components that make up a machine learning task, such as models, datasets, tokenizers, and other resources.
These assets are essential for training, evaluating, and deploying models.
The :py:mod:`fairseq2.assets` module provides a unified API to manage and load these different assets using "asset cards" from various "stores".

Understanding the Asset System
------------------------------

The fairseq2 asset system consists of three main components:

1. **Asset Cards**: YAML files that describe assets and their metadata
2. **Asset Stores**: Collections of asset cards from various sources:

   - Built-in cards: ``fairseq2/assets/cards/``
   - System-wide cards: ``/etc/fairseq2/assets/`` (overridden by ``FAIRSEQ2_ASSET_DIR`` if set)
   - User-specific cards: ``~/.config/fairseq2/assets/`` (overridden by ``FAIRSEQ2_USER_ASSET_DIR`` if set)
   - Recipe-local cards: Specified via ``extra_paths`` in recipe config (see :ref:`api_datasets`)

3. **Asset Loaders**: Code that knows how to load specific asset types

This design allows for:

- **Centralized Management**: All assets are described in a consistent format
- **Environment Flexibility**: Different configurations for different environments
- **Easy Discovery**: Assets can be listed, searched, and queried
- **Source Abstraction**: Assets can come from local files, Hugging Face Hub, or other sources
- **Local Overrides**: Recipe-specific assets (e.g. datasets) can be defined locally

CLI Usage
---------

The fairseq2 asset CLI provides convenient commands to interact with the asset system:

.. code-block:: console

    # List all available assets
    $ python -m fairseq2.assets list

    # List only models
    $ python -m fairseq2.assets list --kind model

    # List only datasets
    $ python -m fairseq2.assets list --kind dataset

    # List only tokenizers
    $ python -m fairseq2.assets list --kind tokenizer

    # Show detailed information about a specific asset
    $ python -m fairseq2.assets show qwen3_8b

**Example Output:**

.. code-block:: console

    $ python -m fairseq2.assets list --kind model

    package:fairseq2.assets.cards
        - model:jepa_vith16@
        - model:jepa_vith16_384@
        - model:jepa_vitl16@
        - model:llama2@
        - model:llama2_13b@
        - model:llama2_13b_chat@
        ...
        - model:wav2vec2_large@
        - model:wav2vec2_large_lv60k@

    $ python -m fairseq2.assets show qwen3_8b
    qwen3_8b
        source          : 'package:fairseq2_ext.cards'
        model_family    : 'qwen'
        model_arch      : 'qwen3_8b'
        checkpoint      : '/datasets/pretrained-llms/Qwen3-8B'
        tokenizer       : '/datasets/pretrained-llms/Qwen3-8B'
        tokenizer_family: 'qwen'
        tokenizer_config: {'use_im_end': True}

Asset Cards: YAML Configuration Files
-------------------------------------

Asset cards are YAML files that describe the assets and their relationships.
You can find all the built-in asset cards in the `fairseq2 repository <https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/assets/cards>`__.

**Key Benefits of Asset Cards:**

* **Standardized Metadata**: Consistent format for describing assets across different types
* **Environment Management**: Support for different configurations per environment
* **Dependency Tracking**: Cards can reference other cards as dependencies
* **Source Flexibility**: Support for multiple download sources (Hugging Face, local files, HTTP)

**Basic Asset Card Structure:**

.. code-block:: yaml

    name: my_model_name
    model_family: transformer
    model_arch: transformer_lm
    checkpoint: "hg://facebook/my-model"
    tokenizer: "hg://facebook/my-model"
    tokenizer_family: sentencepiece

**Multi-Document YAML:**

Multiple assets can be defined in a single file using YAML document separators:

.. code-block:: yaml

    name: qwen25_7b
    model_family: qwen
    model_arch: qwen25_7b
    checkpoint: "hg://qwen/qwen2.5-7b"
    tokenizer: "hg://qwen/qwen2.5-7b"
    tokenizer_family: qwen

    ---

    name: qwen25_7b_instruct
    model_family: qwen
    model_arch: qwen25_7b
    checkpoint: "hg://qwen/qwen2.5-7b-instruct"
    tokenizer: "hg://qwen/qwen2.5-7b-instruct"
    tokenizer_family: qwen
    tokenizer_config:
      use_im_end: true

Creating Custom Assets
----------------------

Adding a Custom Dataset
~~~~~~~~~~~~~~~~~~~~~~~

To add a custom dataset, create an asset card with the required fields:

.. code-block:: yaml

    name: my_custom_dataset
    dataset_family: generic_instruction
    dataset_config:
      # free-form configuration for the dataset


**Required Fields for Datasets:**

- ``name``: Unique identifier for the dataset
- ``dataset_family``: The dataset loader family to use
- ``dataset_config``: Free-form configuration for the dataset loader.

For ``dataset_config``, you can use any configuration that the dataset loader supports.
You can find an example of registering a custom dataset loader below.

.. dropdown:: Example: Registering a custom dataset loader

    .. code-block:: python

      from fairseq2.composition import register_dataset_family
      from fairseq2.recipe import TrainRecipe

      @final
      class MyDataset:
        ...

        @classmethod
        def from_path(cls, path: Path) -> "MyDataset":
          ...

      @dataclass
      class MyDatasetConfig:
        """A dummy dataset config for demonstration purposes."""
        data: Path = field(default_factory=Path)

      def open_my_dataset(config: MyDatasetConfig) -> MyDataset:
        """The mapping between the dataset asset card definition and MyDataset."""
        return MyDataset.from_path(config.data)

      @final
      class MyRecipe(TrainRecipe):
        """A dummy train recipe."""

        @override
        def register(self, container: DependencyContainer) -> None:
            register_dataset_family(
                container,
                "my_dataset",  # dataset family name
                MyDataset,
                MyDatasetConfig,
                opener=open_my_dataset,
            )


Adding a Custom Model
~~~~~~~~~~~~~~~~~~~~~

To add a custom model, you need both the architecture configuration and the asset card:

.. code-block:: yaml

    name: my_custom_model@user
    model_family: llama
    model_arch: llama3_8b  # Use existing architecture
    checkpoint: "/path/to/my/model.pt"
    tokenizer: "hg://meta-llama/Llama-3-8b"
    tokenizer_family: llama

**Required Fields for Models:**

- ``name``: Unique identifier for the model
- ``model_family``: The model family (e.g., 'llama', 'qwen', 'mistral')
- ``checkpoint``: Path or URI to the model checkpoint

Advanced Configuration
----------------------

Environment-Specific Assets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assets can have environment-specific configurations using the ``@environment`` syntax:

.. code-block:: yaml

    # Base configuration
    name: my_model
    model_family: llama
    model_arch: llama3_8b
    checkpoint: "hg://meta-llama/Llama-3-8b"

    ---

    # User-specific override
    name: my_model@user
    base: my_model
    checkpoint: "/home/user/models/my_custom_llama.pt"

    ---

    # Cluster-specific override
    name: my_model@my_cluster
    base: my_model
    model_config:
      max_seq_len: 4096  # Shorter context for production


For more detailed information about registering asset cards on various clusters, please see the :ref:`basics-runtime-extension` documentation.

Base Assets and Inheritance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assets can inherit from other assets using the ``base`` field:

.. code-block:: yaml

    # Base model configuration
    name: base_model
    model_family: qwen
    model_arch: qwen25_7b
    tokenizer_family: qwen

    ---

    # Instruct version inheriting from base
    name: base_model_instruct
    base: base_model
    checkpoint: "hg://qwen/qwen2.5-7b-instruct"
    tokenizer: "hg://qwen/qwen2.5-7b-instruct"
    tokenizer_config:
      use_im_end: true

Asset Store Configuration
-------------------------

The Asset Store System
~~~~~~~~~~~~~~~~~~~~~~

fairseq2 uses a multi-layered asset store system that searches for asset **definitions** in the following order:

1. **User Assets** (``@user`` suffix): Personal assets for the current user
2. **Environment-Specific Assets**: Assets for the detected environment
3. **Base Assets**: Default/fallback assets

**Asset Search Paths:**

fairseq2 looks for **asset cards** in these locations (in order):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Store Type
     - Default Path
     - Environment Variable
   * - Built-in
     - ``fairseq2/assets/cards/``
     - N/A (package resources)
   * - System
     - ``/etc/fairseq2/assets/`` (overridden by ``FAIRSEQ2_ASSET_DIR`` if set)
     - ``FAIRSEQ2_ASSET_DIR``
   * - User
     - ``~/.config/fairseq2/assets/`` (overridden by ``FAIRSEQ2_USER_ASSET_DIR`` if set)
     - ``FAIRSEQ2_USER_ASSET_DIR``
   * - :ref:`basics-runtime-extension`
     - e.g. ``my_package/cards/`` (if registered with ``register_package_assets(container, "my_package.cards")``)
     - N/A (package resources)


If you are working with recipe, you can also specify the asset store to use with the config override ``--config common.asset.extra_paths="['/path/to/assets/dir', '/path/to/yet_other_assets/dir']"`` option.

**Cache Directory:**

Downloaded assets are cached in:

- Default: ``~/.cache/fairseq2/assets/``
- Override: ``FAIRSEQ2_CACHE_DIR`` environment variable

Programmatic Asset Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can register additional asset directories programmatically:

.. code-block:: python

    from pathlib import Path
    from fairseq2.composition import register_file_assets

    def setup_my_fairseq2(container: DependencyContainer) -> None:
      register_file_assets(container, Path("/path/to/my/assets"))

    init_fairseq2(extras=setup_my_fairseq2)

    # or register via setuptools entry_point.
    # or via `Recipe.register()`

For more detailed information about registering via setuptools, please see the :ref:`basics-runtime-extension` documentation.

**Dynamic Asset Creation:**

.. doctest::

    >>> from fairseq2 import DependencyContainer, init_fairseq2
    >>> from fairseq2.assets import get_asset_store
    >>> from fairseq2.composition import register_in_memory_assets
    >>> entries = [{"name": "foo1", "model_family": "foo"}, {"name": "foo2", "model_family": "foo"}]
    >>> def setup_fs2_extension(container: DependencyContainer) -> None:
    ...     register_in_memory_assets(container, source="my_in_mem_source", entries=entries)
    ...
    >>> _ = init_fairseq2(extras=setup_fs2_extension)
    >>> # Now you can load the asset
    >>> asset_store = get_asset_store()
    >>> asset_store.retrieve_card("foo1")
    foo1={'model_family': 'foo', '__source__': 'my_in_mem_source'}

Asset Card Reference
--------------------

Common Asset Fields
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Field
     - Description
     - Required
   * - ``name``
     - Unique identifier for the asset
     - Yes
   * - ``base``
     - Name of parent asset to inherit from
     - No
   * - ``model_family``
     - Model family name (for models)
     - Models only
   * - ``model_arch``
     - Model architecture name (for models)
     - Models only
   * - ``checkpoint``
     - Model checkpoint location (for models)
     - Models only
   * - ``tokenizer``
     - Tokenizer location
     - Models only
   * - ``tokenizer_family``
     - Tokenizer family name
     - Models only
   * - ``dataset_family``
     - Dataset loader family (for datasets)
     - Datasets only


**Source URI Formats:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Format
     - Example
     - Description
   * - Hugging Face Hub (only safetensors)
     - ``hg://qwen/qwen2.5-7b``
     - Downloads from Hugging Face
   * - Local File
     - ``file:///path/to/model.pt``
     - Local filesystem path
   * - HTTP/HTTPS
     - ``https://example.com/model.pt``
     - Direct download URL
   * - Relative Path
     - ``./models/model.pt``
     - Relative to asset card location

Configuration Overrides
~~~~~~~~~~~~~~~~~~~~~~~

Asset cards can override default configurations:

.. code-block:: yaml

    name: custom_qwen
    model_family: qwen
    model_arch: qwen25_7b
    checkpoint: "hg://qwen/qwen2.5-7b"
    tokenizer: "hg://qwen/qwen2.5-7b"
    tokenizer_family: qwen

    # Override model configuration
    model_config:
      max_seq_len: 8192      # Custom sequence length
      dropout_p: 0.1         # Add dropout for fine-tuning

    # Override tokenizer configuration
    tokenizer_config:
      use_im_end: true       # Use special end tokens
      max_length: 8192       # Match model sequence length

Best Practices
--------------

**Asset Naming:**

- Use descriptive names: ``qwen25_7b``, ``llama3_8b_instruct``
- Include size indicators where relevant
- Use consistent naming patterns within families
- Add suffixes for variants: ``_instruct``, ``_chat``, ``_base``

**Environment Management:**

- Use ``@user`` for personal/development assets
- Use environment names for deployment-specific configs
- Keep base assets generic and use overrides for specifics

**File Organization:**

- Group related assets in the same YAML file
- Use clear directory structures: ``models/``, ``datasets/``, etc.
- Document custom assets with comments

**Version Control:**

- Store asset cards in version control
- Use meaningful commit messages when adding assets
- Test asset loading before committing

Troubleshooting
---------------

**Asset Not Found:**

.. code-block:: python

    # Check if asset exists
    from fairseq2.assets import get_asset_store
    from fairseq2.assets.store import AssetNotFoundError

    asset_store = get_asset_store()
    print("Available assets:", list(asset_store.asset_names))

    # Try to load asset
    try:
        card = asset_store.retrieve_card("my_asset")
        print(f"Found: {card.name}")
    except AssetNotFoundError as e:
        print(f"Asset not found: {e}")

**Path Issues:**

.. code-block:: bash

    # Check asset directories
    echo "System: $FAIRSEQ2_ASSET_DIR"
    echo "User: $FAIRSEQ2_USER_ASSET_DIR"
    echo "Cache: $FAIRSEQ2_CACHE_DIR"

    # List user asset directory
    ls -la ~/.config/fairseq2/assets/


See Also
--------

- :doc:`Models </reference/fairseq2.models>` - Model-specific asset management
- :doc:`Datasets </reference/fairseq2.datasets>` - Dataset-specific asset management
- :doc:`Data Tokenizers </reference/fairseq2.data.tokenizers>` - Tokenizer asset management
