# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.composition.assets import (
    register_checkpoint_models as register_checkpoint_models,
)
from fairseq2.composition.assets import register_file_assets as register_file_assets
from fairseq2.composition.assets import (
    register_in_memory_assets as register_in_memory_assets,
)
from fairseq2.composition.assets import (
    register_package_assets as register_package_assets,
)
from fairseq2.composition.datasets import (
    register_dataset_family as register_dataset_family,
)
from fairseq2.composition.extensions import ExtensionError as ExtensionError
from fairseq2.composition.lib import _register_library as _register_library
from fairseq2.composition.models import register_model_family as register_model_family
from fairseq2.composition.tokenizers import (
    register_tokenizer_family as register_tokenizer_family,
)
