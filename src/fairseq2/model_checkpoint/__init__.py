# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides a memory-efficient model checkpoint loading API that
supports lazy loading of various checkpoint formats while also supporting
distributed configurations with tensor resharding capability.

The loaders support:

- Memory-efficient lazy loading to avoid loading entire checkpoints into
  memory at once if the underlying format allows it. In particular relevant for
  large checkpoints that may not fit entirely into memory.
- On-the-fly tensor resharding across different distributed configurations.
- Optional memory mapping for reduced memory footprint.
- State dict conversion for format compatibility.
- Automatic format detection.

.. code:: python
    :caption: Example Usage

    from fairseq2.model_checkpoint import get_model_checkpoint_loader
    from fairseq2.nn import get_shard_dims

    model = ... # PyTorch Module

    checkpoint_path = ...  # Checkpoint file

    # Get shard dimensions of each parameter of the model.
    shard_dims = get_shard_dims(model)

    # Load checkpoint.
    for key, tensor in loader.lazy_load(checkpoint_path, shard_dims):
        # Process each tensor lazily without loading entire checkpoint.
"""

from __future__ import annotations

from fairseq2.model_checkpoint.basic import (
    _BasicModelCheckpointLoader as _BasicModelCheckpointLoader,
)
from fairseq2.model_checkpoint.common import reshard_tensor as reshard_tensor
from fairseq2.model_checkpoint.delegating import (
    _DelegatingModelCheckpointLoader as _DelegatingModelCheckpointLoader,
)
from fairseq2.model_checkpoint.loader import (
    CorruptModelCheckpointError as CorruptModelCheckpointError,
)
from fairseq2.model_checkpoint.loader import (
    ModelCheckpointLoader as ModelCheckpointLoader,
)
from fairseq2.model_checkpoint.loader import (
    ModelCheckpointLoadOptions as ModelCheckpointLoadOptions,
)
from fairseq2.model_checkpoint.loader import StateDictConverter as StateDictConverter
from fairseq2.model_checkpoint.loader import (
    get_model_checkpoint_loader as get_model_checkpoint_loader,
)
from fairseq2.model_checkpoint.native import (
    _NativeModelCheckpointLoader as _NativeModelCheckpointLoader,
)
from fairseq2.model_checkpoint.safetensors import (
    _SafetensorsCheckpointLoader as _SafetensorsCheckpointLoader,
)
