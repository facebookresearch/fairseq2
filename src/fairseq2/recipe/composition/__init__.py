# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The documentation for this module is work in progress.
"""

from __future__ import annotations

from fairseq2.recipe.composition.assets import (
    register_recipe_assets as register_recipe_assets,
)
from fairseq2.recipe.composition.config import (
    register_config_section as register_config_section,
)
from fairseq2.recipe.composition.dataset import register_dataset as register_dataset
from fairseq2.recipe.composition.eval_model import (
    register_reference_model as register_reference_model,
)
from fairseq2.recipe.composition.root import (
    _register_eval_recipe as _register_eval_recipe,
)
from fairseq2.recipe.composition.root import (
    _register_generation_recipe as _register_generation_recipe,
)
from fairseq2.recipe.composition.root import (
    _register_train_recipe as _register_train_recipe,
)
from fairseq2.recipe.composition.tokenizer import (
    register_tokenizer as register_tokenizer,
)
