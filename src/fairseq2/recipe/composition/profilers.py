# Copyright (c) Meta Platforms, Inc. and affiliates.e
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.profilers import CompositeProfiler, Profiler
from fairseq2.recipe.internal.profilers import _MaybeTorchProfilerFactory
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def _register_profilers(container: DependencyContainer) -> None:
    container.register_type(Profiler, CompositeProfiler)

    # Torch Profiler or None
    def maybe_create_torch_profiler(resolver: DependencyResolver) -> Profiler | None:
        profiler_factory = resolver.resolve(_MaybeTorchProfilerFactory)

        return profiler_factory.maybe_create()

    container.collection.register(Profiler, maybe_create_torch_profiler, singleton=True)

    container.register_type(_MaybeTorchProfilerFactory)
