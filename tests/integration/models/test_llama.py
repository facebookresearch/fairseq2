# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

from fairseq2 import get_dependency_resolver
from fairseq2.config import get_config
from fairseq2.models.llama import LLaMAConfig, LLaMAFactory, convert_llama_checkpoint
from fairseq2.models.llama.integ import convert_to_reference_llama_checkpoint


@pytest.mark.skipif(
    "FAIR_ENV_CLUSTER" not in os.environ, reason="checkpoints only on faircluster"
)
def test_convert_to_reference_checkpoint() -> None:
    resolver = get_dependency_resolver()

    model_config = get_config(resolver, LLaMAConfig, "llama2_7b")

    model_factory = LLaMAFactory(model_config)

    model = model_factory.create_model()

    checkpoint = model.state_dict()

    checkpoint = convert_to_reference_llama_checkpoint(checkpoint)

    checkpoint = convert_llama_checkpoint(checkpoint, model_config)

    # This should work.
    model.load_state_dict(checkpoint)
