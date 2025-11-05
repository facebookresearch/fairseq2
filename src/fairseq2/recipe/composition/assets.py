# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
from pathlib import Path

from fairseq2.composition import register_file_assets
from fairseq2.error import InternalError
from fairseq2.runtime.dependency import DependencyContainer


def register_recipe_assets(
    container: DependencyContainer, rel_path: Path | str, *, stack_level: int = 1
) -> None:
    """
    Registers asset cards located under the specified path, relative to the
    calling module’s path (i.e. ``module.__path__``).

    The specified path should be either a YAML file or a directory containing
    YAML files. If a directory is specified, it will be scanned recursively.
    Each YAML file must define one or more asset cards, which are used by
    :class:`AssetStore` to represent assets such as models, datasets, and
    tokenizers.

    Check out the :doc:`/concepts/assets` concept documentation to learn more
    about assets and asset cards.

    This function is intended to be called within the :meth:`~register` method
    of a recipe, to register assets that should always be available alongside
    the recipe.

    .. code:: python

        from fairseq2.runtime.dependency import DependencyContainer
        from fairseq2.recipe import TrainRecipe
        from fairseq2.recipe.composition import register_recipe_assets

        class MyRecipe(TrainRecipe):
            def register(self, container: DependencyContainer) -> None:
                register_recipe_assets(container, "configs/assets")

    ``stack_level`` can be used to specify the actual calling module. For
    example, if ``register_recipe_assets`` is invoked within a wrapper helper
    function, setting ``stack_level`` to 2 indicates that the caller module is
    the one that called the helper function.

    .. code:: python
        :caption: A helper function in a Python file named "my_helpers.py"

        from fairseq2.recipe.composition import register_recipe_assets
        from fairseq2.runtime.dependency import DependencyContainer

        def my_recipe_helper_function(container: DependencyContainer) -> None:
            # Note that `stack_level` is set to 2, so that the actual calling
            # module, and not "my_helpers.py", is used to resolve the path.
            register_recipe_assets(container, "path/to/assets", stack_level=2)

    Note that this function is primarily intended for use by recipe authors.
    Users of a recipe can specify additional paths to search for asset cards by
    using the :attr:`AssetsConfig.extra_paths` configuration option.

    .. code:: yaml
        :caption: A YAML recipe configuration

        common:
          assets:
            extra_paths:
              - /path/to/extra/recipes1
              - /path/to/extra/recipes2

    :raises ValueError: If ``rel_path`` specifies an absolute path
    :raises ValueError: If ``rel_path`` is a string and does not represent a
        valid path
    :raises ValueError: If ``stack_level`` is less than 1 or larger than the
        size of the call stack.
    """
    if isinstance(rel_path, str):
        rel_path = Path(rel_path)

    if rel_path.is_absolute():
        raise ValueError("`rel_path` must be a relative path.")

    if stack_level < 1:
        raise ValueError(
            f"`stack_level` must be greater than or equal to 1, but is {stack_level} instead."
        )

    call_stack = inspect.stack()

    if stack_level >= len(call_stack):
        raise ValueError(
            f"`stack_level` must be less than the size of the call stack ({len(call_stack)}), but is {stack_level} instead."
        )

    filename = call_stack[stack_level].filename

    try:
        module_path = Path(filename)
    except ValueError as ex:
        raise InternalError(
            f"Frame {stack_level} in the call stack does not have a valid `filename` attribute. `filename`: {filename}"
        ) from ex

    asset_path = module_path.parent.joinpath(rel_path)

    register_file_assets(container, asset_path)
