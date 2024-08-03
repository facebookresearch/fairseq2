# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.generation.beam_search.algo import (
    BeamSearchAlgorithm,
    StandardBeamSearchAlgorithm,
)

beam_search_factories = ConfigBoundFactoryRegistry[[], BeamSearchAlgorithm]()

beam_search_factory = beam_search_factories.decorator


@dataclass
class StandardBeamSearchConfig:
    """Holds the configuration of a :class:`StandardBeamSearchConfig`."""


beam_search_factories.register(
    "standard", lambda _: StandardBeamSearchAlgorithm(), StandardBeamSearchConfig
)
