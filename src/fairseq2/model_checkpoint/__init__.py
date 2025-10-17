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
  large checkpoints that may not fit entirely in memory.
- On-the-fly tensor resharding across different distributed configurations.
- Optional memory mapping for reduced memory footprint.
- State dict conversion for format compatibility.
- Automatic format detection.

.. code:: python
    :caption: Example Usage

    from fairseq2.model_checkpoint import DelegatingModelCheckpointLoader

    gangs = ...  # Setup gangs

    # Delegates model loading to the appropriate loader based on the checkpoint
    # format.
    loader = DelegatingModelCheckpointLoader()

    checkpoint_path = ...  # Path to checkpoint file

    # Load checkpoint parameters lazily
    for key, tensor in loader.lazy_load(checkpoint_path, gangs):
        # Process tensor without loading entire checkpoint
"""

from __future__ import annotations

from fairseq2.model_checkpoint.basic import (
    BasicModelCheckpointLoader as BasicModelCheckpointLoader,
)
from fairseq2.model_checkpoint.common import reshard_tensor as reshard_tensor
from fairseq2.model_checkpoint.delegating import (
    DelegatingModelCheckpointLoader as DelegatingModelCheckpointLoader,
)
from fairseq2.model_checkpoint.loader import (
    ModelCheckpointError as ModelCheckpointError,
)
from fairseq2.model_checkpoint.loader import (
    ModelCheckpointLoader as ModelCheckpointLoader,
)
from fairseq2.model_checkpoint.loader import StateDictConverter as StateDictConverter
from fairseq2.model_checkpoint.native import (
    NativeModelCheckpointLoader as NativeModelCheckpointLoader,
)
from fairseq2.model_checkpoint.safetensors import (
    SafetensorsCheckpointLoader as SafetensorsCheckpointLoader,
)
