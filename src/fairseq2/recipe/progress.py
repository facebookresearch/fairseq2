# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.cluster import WorldInfo
from fairseq2.runtime.dependency import DependencyResolver
from fairseq2.utils.progress import ProgressReporter
from fairseq2.utils.rich import RichProgressReporter, get_error_console


def _create_progress_reporter(resolver: DependencyResolver) -> ProgressReporter:
    console = get_error_console()

    world_info = resolver.resolve(WorldInfo)

    return RichProgressReporter(console, world_info.rank)
