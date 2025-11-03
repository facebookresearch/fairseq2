# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from fairseq2.recipe.config import ConfigSectionNotFoundError
from fairseq2.runtime.dependency import DependencyResolver

SectionT = TypeVar("SectionT")


@dataclass
class _RecipeConfigHolder:
    config: object


def _get_config_section(
    resolver: DependencyResolver, name: str, kls: type[SectionT]
) -> SectionT:
    config_holder = resolver.resolve(_RecipeConfigHolder)

    try:
        section = getattr(config_holder.config, name)
    except AttributeError:
        raise ConfigSectionNotFoundError(name) from None

    if not isinstance(section, kls):
        raise TypeError(
            f"{name} recipe configuration section must be of type `{kls}`, but is of type `{type(section)}` instead."
        )

    return section
