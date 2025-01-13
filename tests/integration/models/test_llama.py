# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import cast

import pytest

from fairseq2.context import get_runtime_context
from fairseq2.models.llama import LLaMAConfig, LLaMAFactory, convert_llama_checkpoint
from fairseq2.models.llama.integ import convert_to_reference_checkpoint


@pytest.mark.skipif(
    "FAIR_ENV_CLUSTER" not in os.environ, reason="checkpoints only on faircluster"
)
def test_convert_to_reference_checkpoint() -> None:
    context = get_runtime_context()

    model_config_registry = context.get_config_registry(LLaMAConfig)

    model_config = model_config_registry.get("llama2_7b")

    model_factory = LLaMAFactory(model_config)

    model = model_factory.create_model()

    state_dict = model.state_dict()

    checkpoint: dict[str, object] = {"model": state_dict}

    checkpoint = convert_to_reference_checkpoint(checkpoint)

    checkpoint = convert_llama_checkpoint(checkpoint, model_config)

    state_dict = cast(dict[str, object], checkpoint["model"])

    # This should work.
    model.load_state_dict(state_dict)
