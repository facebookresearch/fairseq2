# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

import torch

from fairseq2.device import Device
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.models.transformer import (
    Flash2SDPA,
    Flash3SDPA,
    FlexSDPA,
    NaiveSDPA,
    TorchSDPA,
    set_default_sdpa_factory,
)
from fairseq2.recipe.cluster import WorldInfo
from fairseq2.recipe.config import (
    CommonSection,
    TorchSection,
    get_config_section,
    get_output_dir,
)
from fairseq2.runtime.dependency import DependencyNotFoundError, DependencyResolver
from fairseq2.utils.env import get_env
from fairseq2.utils.threading import get_num_threads
from fairseq2.utils.version import torch_greater_or_equal


def _configure_torch(resolver: DependencyResolver) -> None:
    common_section = get_config_section(resolver, "common", CommonSection)

    _set_environment_variables(resolver)

    _set_num_threads(resolver, common_section.torch.num_threads)

    _set_numerics(common_section.torch)

    _set_default_device(resolver)

    _set_default_sdpa_variant(common_section.torch.default_sdpa)

    torch._functorch.config.activation_memory_budget = (
        common_section.torch.compiled_region_activation_memory_budget
    )


def _set_environment_variables(resolver: DependencyResolver) -> None:
    file_system = resolver.resolve(FileSystem)

    try:
        output_dir = get_output_dir(resolver)
    except DependencyNotFoundError:
        output_dir = None

    env = get_env(resolver)

    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    # Non-actionable warning when tracing enabled.
    warnings.filterwarnings(
        action="ignore", message=r".*Skipping serialization of skipfiles_inline_module_allowlist value.*"  # fmt: skip
    )

    if output_dir is None:
        return

    # Triton Cache
    if "TRITON_CACHE_DIR" not in env:
        world_info = resolver.resolve(WorldInfo)

        triton_cache_dir = output_dir.joinpath(f"cache/triton/rank_{world_info.rank}")

        try:
            file_system.make_directory(triton_cache_dir)
        except OSError:
            log.warning("A system error has occurred while creating the '{}' Triton cache directory.", triton_cache_dir)  # fmt: skip
        else:
            env["TRITON_CACHE_DIR"] = str(triton_cache_dir)

    # Just-in-Time Kernel Compilation Cache
    if "PYTORCH_KERNEL_CACHE_PATH" not in env:
        jit_cache_dir = output_dir.joinpath("cache/torch/jit")

        try:
            file_system.make_directory(jit_cache_dir)
        except OSError:
            log.warning("A system error has occurred while creating the '{}' PyTorch JIT kernel cache directory.", jit_cache_dir)  # fmt: skip
        else:
            env["PYTORCH_KERNEL_CACHE_PATH"] = str(jit_cache_dir)


def _set_num_threads(resolver: DependencyResolver, num_threads: int | None) -> None:
    env = get_env(resolver)

    if "OMP_NUM_THREADS" in env:
        if num_threads is not None:
            log.warning("`OMP_NUM_THREADS` environment variable set. Ignoring `common.num_threads` value.")  # fmt: skip

        return

    if num_threads is None:
        world_info = resolver.resolve(WorldInfo)

        num_threads = get_num_threads(world_info.local_size)

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


def _set_default_device(resolver: DependencyResolver) -> None:
    device = resolver.resolve(Device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    log.info("Setting '{}' as the default device of the process.", device)


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
        case "flex":
            set_default_sdpa_factory(FlexSDPA)
        case "naive":
            set_default_sdpa_factory(NaiveSDPA)
        case _:
            raise ValueError(
                f"`name` must be a known SDPA variant, but is '{name}' instead."
            )

    log.info("Setting '{}' as the default SDPA variant.", name)


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
