# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

from fairseq2.models.llama import (
    LLaMAConfig,
    LLaMAFactory,
    convert_llama_state_dict,
    convert_to_ref_llama_state_dict,
)
from fairseq2.runtime.config_registry import get_config
from fairseq2.runtime.dependency import get_dependency_resolver


@pytest.mark.skipif(
    "FAIR_ENV_CLUSTER" not in os.environ, reason="checkpoints only on faircluster"
)
def test_convert_to_ref_checkpoint() -> None:
    resolver = get_dependency_resolver()

    model_config = get_config(resolver, LLaMAConfig, "llama2_7b")

    model_factory = LLaMAFactory(model_config)

    model = model_factory.create_model()

    state_dict = model.state_dict()

    state_dict = convert_to_ref_llama_state_dict(state_dict)

    state_dict = convert_llama_state_dict(state_dict, model_config)

    # This should work.
    model.load_state_dict(state_dict)
