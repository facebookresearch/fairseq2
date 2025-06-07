# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.dependency import DependencyResolver
from fairseq2.device import CPU, Device
from fairseq2.recipe.config import CommonSection, get_config_section
from fairseq2.utils.rng import SeedHolder, manual_seed


def create_seed_holder(resolver: DependencyResolver) -> SeedHolder:
    common_section = get_config_section(resolver, "common", CommonSection)

    return SeedHolder(common_section.seed)


def set_manual_rng_seed(resolver: DependencyResolver) -> None:
    common_section = get_config_section(resolver, "common", CommonSection)

    device = resolver.resolve(Device)

    seed = common_section.seed

    manual_seed(seed, CPU, device)
