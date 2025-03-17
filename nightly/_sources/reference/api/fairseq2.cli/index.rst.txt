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

Here's a complete example:

.. code-block:: python

    from fairseq2.cli import Cli, CliCommandHandler
    from fairseq2.cli.commands.recipe import RecipeCommandHandler
    from fairseq2.context import RuntimeContext
    
    def setup_custom_cli(context: RuntimeContext, cli: Cli) -> None:
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
        
        # create a recipe command handler first
        recipe_handler = RecipeCommandHandler(
            loader=recipe_loader,
            config_kls=recipe_config_kls,
            default_preset="recipe_default_preset",
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

    from fairseq2.context import RuntimeContext
    from fairseq2.recipes import Recipe
    
    @dataclass
    class CustomConfig:
        param1: str
        param2: int

    def load_custom_recipe(
        context: RuntimeContext, config: CustomConfig, output_dir: Path
    ) -> Recipe:
        def run_recipe() -> None:
            # Recipe implementation
            pass
            
        return run_recipe

    # Create preset configs
    custom_presets = ConfigRegistry(CustomConfig)
    custom_presets.register("default", CustomConfig(param1="value", param2=42))

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
