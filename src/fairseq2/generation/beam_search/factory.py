# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fairseq2.factory_registry import ConfigBoundFactoryRegistry
from fairseq2.generation.beam_search.algo import (
    BeamSearchAlgorithm,
    StandardBeamSearchAlgorithm,
)

# typing_extensions in Python<=3.9 has a bug that causes this line to fail with:
#    Parameters to generic types must be types. Got []
# See https://stackoverflow.com/questions/73974069/generic-paramspec-on-python-3-9.
# See https://github.com/python/typing/discussions/908
if TYPE_CHECKING:  # compat: remove when Python 3.9 support is dropped.
    beam_search_factories = ConfigBoundFactoryRegistry[[], BeamSearchAlgorithm]()
else:
    beam_search_factories = ConfigBoundFactoryRegistry()

beam_search_factory = beam_search_factories.decorator


@dataclass
class StandardBeamSearchConfig:
    """Holds the configuration of a :class:`StandardBeamSearchConfig`."""


beam_search_factories.register(
    "standard", lambda _: StandardBeamSearchAlgorithm(), StandardBeamSearchConfig
)
