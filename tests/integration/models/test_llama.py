# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.llama import create_llama_model, llama_archs
from fairseq2.models.llama.loader import (
    convert_llama_checkpoint,
    convert_to_llama_checkpoint,
)
from fairseq2.models.utils.checkpoint import load_checkpoint
from fairseq2.typing import CPU
from tests.common import device


@pytest.mark.skipif(
    "FAIR_ENV_CLUSTER" not in os.environ, reason="checkpoints only on faircluster"
)
def test_convert_to_llama_checkpoint() -> None:
    card = asset_store.retrieve_card("llama2_7b")

    path = download_manager.download_checkpoint(
        card.field("checkpoint").as_uri(), model_name="llama2_7b", progress=False
    )

    # Load and convert the reference checkpoint to fairseq2.
    checkpoint = load_checkpoint(
        path, map_location=CPU, restrict=True, converter=convert_llama_checkpoint
    )

    # Convert it back to the reference format.
    checkpoint = convert_to_llama_checkpoint(checkpoint)

    # Now, convert back to fairseq2 again.
    checkpoint = convert_llama_checkpoint(checkpoint)

    # Try to load the model with the converted fairseq2 checkpoint.
    model_config = llama_archs.get_config("llama2_7b")

    model = create_llama_model(model_config, device=device)

    # This should work.
    model.load_state_dict(checkpoint["model"])
