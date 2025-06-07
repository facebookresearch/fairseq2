# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.device import CPU, Device
from fairseq2.logging import log
from fairseq2.recipe.config import CommonSection, get_config_section
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.rng import RngBag, SeedHolder


def _create_seed_holder(resolver: DependencyResolver) -> SeedHolder:
    common_section = get_config_section(resolver, "common", CommonSection)

    return SeedHolder(common_section.seed)


def _manual_seed(resolver: DependencyResolver) -> None:
    common_section = get_config_section(resolver, "common", CommonSection)

    device = resolver.resolve(Device)

    rng_bag = RngBag.from_device_defaults(CPU, device)

    rng_bag.manual_seed(common_section.seed)

    log.info("Random number generator seed set to {}.", common_section.seed)
