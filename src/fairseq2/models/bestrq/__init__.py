# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.bestrq.factory import BESTRQ_FAMILY as BESTRQ_FAMILY
from fairseq2.models.bestrq.factory import BestRQBuilder as BestRQBuilder
from fairseq2.models.bestrq.factory import BestRQConfig as BestRQConfig
from fairseq2.models.bestrq.factory import (
    BestRQEncoderBuilder as BestRQEncoderBuilder,
)
from fairseq2.models.bestrq.factory import (
    BestRQEncoderConfig as BestRQEncoderConfig,
)
from fairseq2.models.bestrq.factory import (
    create_bestrq_model as create_bestrq_model,
)
from fairseq2.models.bestrq.factory import bestrq_arch as bestrq_arch
from fairseq2.models.bestrq.factory import bestrq_archs as bestrq_archs
from fairseq2.models.bestrq.factory import (
    bestrq_encoder_arch as bestrq_encoder_arch,
)
from fairseq2.models.bestrq.factory import (
    bestrq_encoder_archs as bestrq_encoder_archs,
)
from fairseq2.models.bestrq.masker import RandomNoiseMasker as RandomNoiseMasker
from fairseq2.models.bestrq.model import BestRQFeatures as BestRQFeatures
from fairseq2.models.bestrq.model import BestRQLoss as BestRQLoss
from fairseq2.models.bestrq.model import BestRQModel as BestRQModel
from fairseq2.models.bestrq.model import BestRQOutput as BestRQOutput
from fairseq2.models.bestrq.quantizer import MultiRandomVectorQuantizerOutput

# isort: split

from fairseq2.dependency import DependencyContainer
from fairseq2.models.bestrq.archs import register_archs


def register_bestrq(container: DependencyContainer) -> None:
    register_archs()
