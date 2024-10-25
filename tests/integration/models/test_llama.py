# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

from fairseq2.assets import default_asset_store, default_download_manager
from fairseq2.models.llama import create_llama_model, llama_archs
from fairseq2.models.llama.integ import convert_to_reference_checkpoint
from fairseq2.models.llama.loader import convert_llama_checkpoint
from fairseq2.typing import CPU
from fairseq2.utils.file import StandardTensorLoader
from tests.common import device


@pytest.mark.skipif(
    "FAIR_ENV_CLUSTER" not in os.environ, reason="checkpoints only on faircluster"
)
def test_convert_to_reference_checkpoint() -> None:
    model_config = llama_archs.get("llama2_7b")

    card = default_asset_store.retrieve_card("llama2_7b")

    checkpoint_uri = card.field("checkpoint").as_uri()
    checkpoint_checksum = card.field("checksum").get_as_(str)

    path = default_download_manager.download_checkpoint(
        checkpoint_uri, checkpoint_checksum, model_name="llama2_7b", progress=False
    )

    tensor_loader = StandardTensorLoader()

    checkpoint = tensor_loader(path, map_location=CPU, restrict=True)

    # Convert the reference checkpoint to fairseq2.
    checkpoint = convert_llama_checkpoint(checkpoint, model_config)

    # Convert it back to the reference format.
    checkpoint = convert_to_reference_checkpoint(checkpoint)

    # Now, convert back to fairseq2 again.
    checkpoint = convert_llama_checkpoint(checkpoint, model_config)

    # Try to load the model with the converted fairseq2 checkpoint.
    model = create_llama_model(model_config, device=device)

    # This should work.
    model.load_state_dict(checkpoint["model"])
