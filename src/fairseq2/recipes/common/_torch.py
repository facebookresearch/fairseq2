# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from pathlib import Path

import torch

from fairseq2.context import RuntimeContext
from fairseq2.logging import log
from fairseq2.models.transformer import (
    Flash2SDPA,
    Flash3SDPA,
    NaiveSDPA,
    TorchSDPA,
    set_default_sdpa_factory,
)
from fairseq2.recipes import RecipeError
from fairseq2.recipes.config import TorchSection
from fairseq2.utils.env import get_rank
from fairseq2.utils.threading import ThreadingError, get_num_threads
from fairseq2.utils.version import torch_greater_or_equal


def setup_torch(
    context: RuntimeContext, torch_section: TorchSection, output_dir: Path | None
) -> None:
    _set_environment_variables(context, output_dir)

    _set_num_threads(context, torch_section.num_threads)

    _set_numerics(torch_section)

    _set_default_sdpa_variant(torch_section.default_sdpa)

    torch._functorch.config.activation_memory_budget = (
        torch_section.compiled_region_activation_memory_budget
    )


def _set_environment_variables(
    context: RuntimeContext, output_dir: Path | None
) -> None:
    if output_dir is None:
        return

    file_system = context.file_system

    env = context.env

    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    # Non-actionable warning when tracing enabled.
    warnings.filterwarnings(
        action="ignore", message=r".*Skipping serialization of skipfiles_inline_module_allowlist value.*"  # fmt: skip
    )

    # Triton Cache
    if "TRITON_CACHE_DIR" not in env:
        rank = get_rank(env)

        triton_cache_dir = output_dir.joinpath(f"cache/triton/rank_{rank}")

        try:
            file_system.make_directory(triton_cache_dir)
        except OSError:
            log.warning("'{}' directory for Triton cache cannot be created.", triton_cache_dir)  # fmt: skip
        else:
            env["TRITON_CACHE_DIR"] = str(triton_cache_dir)

    # Just-in-Time Kernel Compilation Cache
    if "PYTORCH_KERNEL_CACHE_PATH" not in env:
        jit_cache_dir = output_dir.joinpath("cache/torch/jit")

        try:
            file_system.make_directory(jit_cache_dir)
        except OSError:
            log.warning("'{}' directory for PyTorch JIT kernel cache cannot be created.", jit_cache_dir)  # fmt: skip
        else:
            env["PYTORCH_KERNEL_CACHE_PATH"] = str(jit_cache_dir)


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


def _set_numerics(torch_section: TorchSection) -> None:
    if not torch.cuda.is_available():
        return

    # Matrix Multiplications
    torch.backends.cuda.matmul.allow_tf32 = torch_section.allow_tf32

    # Convolutions
    torch.backends.cudnn.allow_tf32 = torch_section.allow_tf32

    # Reduced Precision GEMMs
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        torch_section.fp16_reduced_precision
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        torch_section.bf16_reduced_precision
    )


def _set_default_sdpa_variant(name: str) -> None:
    match name:
        case "torch":
            set_default_sdpa_factory(TorchSDPA)
        case "torch_math" | "torch_mem_efficient" | "torch_flash":
            set_default_sdpa_factory(TorchSDPA)

            backend = name[6:]

            try:
                _set_torch_sdpa_backend(backend)
            except (ImportError, AttributeError):
                log.warning("PyTorch SDPA kernel cannot be set to '{}'. Falling back to auto mode.", backend)  # fmt: skip
        case "flash2":
            set_default_sdpa_factory(Flash2SDPA)
        case "flash3":
            set_default_sdpa_factory(Flash3SDPA)
        case "naive":
            set_default_sdpa_factory(NaiveSDPA)
        case _:
            raise ValueError(
                f"`name` must be a known SDPA variant, but is '{name}' instead."
            )

    log.info("Default SDPA variant set to '{}'.", name)


def _set_torch_sdpa_backend(name: str) -> None:
    if torch_greater_or_equal(2, 6):
        _disable_torch_sdpa_backends()
    else:
        _disable_torch_sdpa_backends_legacy()

    match name:
        case "math":
            _enable_torch_sdpa_backend("math")
        case "mem_efficient":
            _enable_torch_sdpa_backend("mem_efficient")
        case "flash":
            _enable_torch_sdpa_backend("flash")
            _enable_torch_sdpa_backend("cudnn")


def _disable_torch_sdpa_backends() -> None:
    from torch.nn.attention import (
        _backend_names as backend_names,  # type: ignore[attr-defined]
    )

    for name in backend_names.keys():
        _enable_torch_sdpa_backend(name, False)


def _disable_torch_sdpa_backends_legacy() -> None:
    for name in ["math", "mem_efficient", "flash", "cudnn"]:
        _enable_torch_sdpa_backend(name, False)


def _enable_torch_sdpa_backend(name: str, value: bool = True) -> None:
    try:
        enable_backend = getattr(torch.backends.cuda, f"enable_{name}_sdp")
    except AttributeError:
        return

    enable_backend(value)
