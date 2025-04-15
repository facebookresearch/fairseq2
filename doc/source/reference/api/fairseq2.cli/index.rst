============
fairseq2.cli
============

.. currentmodule:: fairseq2.cli

.. toctree::
    :maxdepth: 1

    llama/index


Classes
-------

.. autoclass:: Cli
   :members:

.. autoclass:: CliGroup
   :members:

.. autoclass:: CliCommand
   :members:

.. autoclass:: CliCommandHandler
   :members:

.. autoclass:: fairseq2.cli.commands.recipe.RecipeCommandHandler
   :members:


Examples
--------

Creating a Custom CLI
=====================

To create a custom CLI, you'll need to:

1. Create a CLI group
2. Add commands to the group
3. Register your CLI extension
4. (optional) Add extra sweep keys

Here's a complete example:

.. code-block:: python

    from fairseq2.cli import Cli, CliCommandHandler
    from fairseq2.cli.commands.recipe import RecipeCommandHandler
    
    def setup_custom_cli(cli: Cli) -> None:
        # Create a new command group
        group = cli.add_group(
            "custom",
            help="Custom recipes and utilities"
        )
        
        group.add_command(
            name="custom_command",
            handler=custom_handler(),  # this is the command handler fn.
            help="Run custom recipe"
        )

    def setup_recipe_cli(cli: Cli) -> None:
        # Create a new command group for recipes
        group = cli.add_group("recipe_name", help="Recipe commands")

        # (optional) Add extra sweep keys
        extra_sweep_keys = {"extra_key1", "extra_key2"}

        # create a recipe command handler first
        recipe_handler = RecipeCommandHandler(
            loader=recipe_loader,
            config_kls=recipe_config_kls,
            default_preset="recipe_default_preset",
            extra_sweep_keys=extra_sweep_keys,
        )

        group.add_command(
            name="recipe_command_name",
            handler=recipe_handler,
            help="recipe_command_help",
        )


You can find more examples in our recipe command examples:

* :mod:`fairseq2.recipes.lm.instruction_finetune`
* :mod:`fairseq2.recipes.llama.convert_checkpoint`
* :mod:`fairseq2.recipes.wav2vec2.train`


Recipe Command Handler
======================

The :class:`RecipeCommandHandler` class provides a standardized way to handle recipe commands. It automatically sets up:

- Configuration management (presets, files, overrides)
- Output directory handling
- Logging setup
- Environment setup for distributed training

Example implementation:

.. code-block:: python

    from dataclasses import dataclass
    from pathlib import Path
    from typing import Callable
    
    @dataclass
    class CustomConfig:
        param1: str
        param2: int

    def load_custom_recipe(config: CustomConfig, output_dir: Path) -> Callable[[], None]:
        def run_recipe() -> None:
            # Recipe implementation
            pass
            
        return run_recipe

    # Create preset configs
    custom_presets = ConfigRegistry(CustomConfig)
    custom_presets.register("default", CustomConfig(param1="value", param2=42))

Understanding Sweep Tags and Extra Sweep Keys
=============================================

Sweep tags are generated based on configuration values and are used to organize output directories for different runs. The default format is ``"ps_{preset}.ws_{world_size}.{hash}"``.

Default Sweep Keys
------------------

fairseq2 includes a comprehensive set of default sweep keys defined in :meth:`fairseq2.recipes.utils.sweep_tag.get_default_sweep_keys`. These keys include:

- Common configuration keys: ``name``, ``family``, ``config``
- Top-level keys: ``model``, ``dataset``, ``gang``, ``trainer``, ``criterion``, ``optimizer``
- Model configuration: ``arch``, ``checkpoint``
- And many others covering various aspects of training configuration

When to Use Extra Sweep Keys
----------------------------

You should add extra sweep keys when:

1. You have custom configuration keys that aren't included in the default list
2. You want to include specific nested configuration keys in the sweep tag generation
3. You're adding new configuration categories to your recipe

How path resolution works
-------------------------

When adding extra sweep keys, you only need to add the missing path components. For example:
    
    - To include ``my_custom_category.config.parameter`` in sweep tags, add ``extra_sweep_keys={"my_custom_category", "parameter"}``
    - You don't need to add ``config`` because it's already in the default sweep keys

For example, if you have a custom configuration like:

.. code-block:: python

    {
        "custom_module": {
            "specialized_param": "value",
            "config": {
                "nested_param": 42
            }
        }
    }

You would need to add ``"custom_module"`` and possibly ``"specialized_param"`` and ``"nested_param"`` to your extra sweep keys.

Customizing Sweep Tag Format
----------------------------

The sweep tag format can be customized to include specific configuration values:

.. code-block:: python

    # Basic format with default placeholders
    fmt="ps_{preset}.ws_{world_size}.{hash}"
    
    # Including specific configuration values
    fmt="model_{model.name}.lr_{optimizer.config.lr}"
    
    # Including deeply nested values
    fmt="dropout_{model.config.dropout_p}.batch_{dataset.batch_size}"

Available placeholders include:
- Any configuration key path that exists in your configuration (you can use ``--dump-config`` to see the full configuration)
- Special values: ``preset``, ``world_size``, and ``hash``

CLI Usage Example
-----------------

.. code-block:: bash

    # Use a custom sweep tag format to organize runs by learning rate
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --sweep-format="lr_{optimizer.config.lr}"
    
    # Complex format with multiple parameters 
    fairseq2 lm instruction_finetune <OUTPUT_DIR> --sweep-format="model_{model.name}.bs_{dataset.batch_size}.lr_{optimizer.config.lr}"

CLI Initialization Process
==========================

The CLI system is initialized in the following order:

1. :class:`Cli` instance is created in :meth:`fairseq2.recipes.main`
2. Core CLI groups are registered in :meth:`fairseq2.recipes._setup_cli`
3. Extension CLI groups are registered via :meth:`fairseq2.recipes._setup_cli_extensions`

To add your own CLI extension:

1. Create a Python package for your extension
2. Create an entry point in your package's ``setup.py``:

.. code-block:: python

    setup(
        name="my_fairseq2_extension",
        entry_points={
            "fairseq2.cli": [
                "custom = my_extension.cli:setup_custom_cli"
            ]
        }
    )