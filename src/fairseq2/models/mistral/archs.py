# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.mistral.factory import MistralConfig, mistral_arch


@mistral_arch("7b")
def _7b() -> MistralConfig:
    return MistralConfig()
