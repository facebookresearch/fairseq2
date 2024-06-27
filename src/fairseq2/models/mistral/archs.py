# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.config_registry import ConfigRegistry
from fairseq2.models.mistral.factory import MistralConfig

mistral_archs = ConfigRegistry[MistralConfig]()

mistral_arch = mistral_archs.decorator


def _7b() -> MistralConfig:
    return MistralConfig()


def _register_mistral_archs() -> None:
    mistral_archs.register("7b", _7b)
