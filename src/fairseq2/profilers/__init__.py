# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.profilers.composite import CompositeProfiler as CompositeProfiler
from fairseq2.profilers.profiler import NOOP_PROFILER as NOOP_PROFILER
from fairseq2.profilers.profiler import Profiler as Profiler
from fairseq2.profilers.torch import TorchProfiler as TorchProfiler
