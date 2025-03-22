# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import torch

from fairseq2.context import RuntimeContext
from fairseq2.logging import log
from fairseq2.recipes import RecipeError
from fairseq2.recipes.config import CommonSection, get_config_section
from fairseq2.utils.env import get_rank
from fairseq2.utils.threading import ThreadingError, get_num_threads


def setup_torch(
    context: RuntimeContext, recipe_config: object, output_dir: Path | None = None
) -> None:
    common_section = get_config_section(recipe_config, "common", CommonSection)

    _set_environment_variables(context, output_dir)

    _set_num_threads(context, common_section.num_threads)

    _set_tf32(common_section.allow_tf32)


def _set_num_threads(context: RuntimeContext, num_threads: int | None) -> None:
    if "OMP_NUM_THREADS" in context.env:
        if num_threads is not None:
            log.warning("`OMP_NUM_THREADS` environment variable set. Ignoring `common.num_threads` value.")  # fmt: skip

        return

    if num_threads is None:
        try:
            num_threads = get_num_threads(context.env)
        except ThreadingError as ex:
            raise RecipeError(
                "The number of threads to use for intra-op parallelism cannot be determined. See the nested exception for details."
            ) from ex

    torch.set_num_threads(num_threads)

    log.info("Setting the number of threads used for intra-op parallelism to {}.", num_threads)  # fmt: skip


def _set_tf32(allow_tf32: bool) -> None:
    if not torch.cuda.is_available():
        return

    # Matrix Multiplications
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    # Convolutions
    torch.backends.cudnn.allow_tf32 = allow_tf32


def _set_environment_variables(
    context: RuntimeContext, output_dir: Path | None
) -> None:
    if output_dir is None:
        return

    env = context.env

    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    # torch.compile Trace. It has low overhead and can be enabled by default.
    if "TORCH_TRACE" not in env:
        trace_dir = output_dir.joinpath("cache/torch/trace")

        # PyTorch does not offer any function to set the compilation trace
        # directory programmatically. This is a hacky workaround.
        trace_handler = torch._logging._internal.LOG_TRACE_HANDLER
        if trace_handler is not None:
            trace_handler.root_dir = str(trace_dir)

        env["TORCH_TRACE"] = str(trace_dir)

    # Triton Cache
    if "TRITON_CACHE_DIR" not in env:
        rank = get_rank(env)

        triton_cache_dir = output_dir.joinpath(f"cache/triton/rank_{rank}")

        env["TRITON_CACHE_DIR"] = str(triton_cache_dir)

    # Just-in-Time Kernel Compilation Cache
    if "PYTORCH_KERNEL_CACHE_PATH" not in env:
        jit_cache_dir = output_dir.joinpath("cache/torch/jit")

        env["PYTORCH_KERNEL_CACHE_PATH"] = str(jit_cache_dir)
