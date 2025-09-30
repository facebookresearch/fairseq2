# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from pathlib import Path
from typing import final

import torch

from fairseq2.device import Device
from fairseq2.file_system import FileSystem
from fairseq2.logging import log
from fairseq2.models.transformer import (
    Flash2SDPA,
    Flash3SDPA,
    NaiveSDPA,
    TorchSDPA,
    set_default_sdpa_factory,
)
from fairseq2.recipe.config import CommonSection
from fairseq2.utils.env import Environment
from fairseq2.utils.rng import RngBag
from fairseq2.utils.threading import get_num_threads
from fairseq2.world_info import WorldInfo


@final
class _TorchConfigurer:
    def __init__(
        self,
        section: CommonSection,
        world_info: WorldInfo,
        env: Environment,
        device: Device,
        rng_bag: RngBag,
        output_dir: Path,
        file_system: FileSystem,
    ) -> None:
        self._section = section
        self._world_info = world_info
        self._env = env
        self._device = device
        self._rng_bag = rng_bag
        self._output_dir = output_dir
        self._file_system = file_system

    def configure(self) -> None:
        self._set_seed()

        self._set_environment_variables()

        self._set_num_threads()

        self._set_numerics()

        self._set_default_device()

        self._set_default_sdpa_variant()

        torch._functorch.config.activation_memory_budget = (
            self._section.torch.compiled_region_activation_memory_budget
        )

    def _set_seed(self) -> None:
        self._rng_bag.manual_seed(self._section.seed)

        log.info("Random number generator seed set to {}.", self._section.seed)

    def _set_environment_variables(self) -> None:
        self._env.set("TORCH_SHOW_CPP_STACKTRACES", "1")

        # Non-actionable warning when tracing enabled.
        warnings.filterwarnings(
            action="ignore", message=r".*Skipping serialization of skipfiles_inline_module_allowlist value.*"  # fmt: skip
        )

        # Triton Cache
        if not self._env.has("TRITON_CACHE_DIR"):
            triton_cache_dir = self._output_dir.joinpath(
                f"cache/triton/rank_{self._world_info.rank}"
            )

            try:
                self._file_system.make_directory(triton_cache_dir)
            except OSError:
                log.warning("A system error occurred while creating the '{}' Triton cache directory.", triton_cache_dir)  # fmt: skip
            else:
                self._env.set("TRITON_CACHE_DIR", str(triton_cache_dir))

        # Just-in-Time Kernel Compilation Cache
        if not self._env.has("PYTORCH_KERNEL_CACHE_PATH"):
            jit_cache_dir = self._output_dir.joinpath("cache/torch/jit")

            try:
                self._file_system.make_directory(jit_cache_dir)
            except OSError:
                log.warning("A system error occurred while creating the '{}' PyTorch JIT kernel cache directory.", jit_cache_dir)  # fmt: skip
            else:
                self._env.set("PYTORCH_KERNEL_CACHE_PATH", str(jit_cache_dir))

    def _set_num_threads(self) -> None:
        num_threads = self._section.torch.num_threads

        if self._env.has("OMP_NUM_THREADS"):
            if num_threads is not None:
                log.warning("OMP_NUM_THREADS environment variable set. Ignoring `common.num_threads`.")  # fmt: skip

            return

        if num_threads is None:
            num_threads = get_num_threads(self._world_info.local_size)

        torch.set_num_threads(num_threads)

        log.info("Number of threads used for intra-op parallelism set to {}.", num_threads)  # fmt: skip

    def _set_numerics(self) -> None:
        if not torch.cuda.is_available():
            return

        torch_config = self._section.torch

        # Matrix Multiplication
        torch.backends.cuda.matmul.allow_tf32 = torch_config.allow_tf32

        # Convolution
        torch.backends.cudnn.allow_tf32 = torch_config.allow_tf32

        # Reduced Precision GEMM
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            torch_config.fp16_reduced_precision
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            torch_config.bf16_reduced_precision
        )

    def _set_default_device(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        log.info("Device of the process set to {}.", self._device)

    def _set_default_sdpa_variant(self) -> None:
        name = self._section.torch.default_sdpa

        match name:
            case "torch":
                set_default_sdpa_factory(TorchSDPA)
            case "torch_math" | "torch_mem_efficient" | "torch_flash":
                set_default_sdpa_factory(TorchSDPA)

                backend = name[6:]

                try:
                    self._set_torch_sdpa_backend(backend)
                except (ImportError, AttributeError):
                    log.warning("PyTorch SDPA kernel cannot be set to {}. Falling back to auto mode.", backend)  # fmt: skip
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

        log.info("Default SDPA variant set to {}.", name)

    @classmethod
    def _set_torch_sdpa_backend(cls, name: str) -> None:
        cls._disable_torch_sdpa_backends()

        match name:
            case "math":
                cls._enable_torch_sdpa_backend("math")
            case "mem_efficient":
                cls._enable_torch_sdpa_backend("mem_efficient")
            case "flash":
                cls._enable_torch_sdpa_backend("flash")
                cls._enable_torch_sdpa_backend("cudnn")

    @classmethod
    def _disable_torch_sdpa_backends(cls) -> None:
        from torch.nn.attention import (
            _backend_names as backend_names,  # type: ignore[attr-defined]
        )

        for name in backend_names.keys():
            cls._enable_torch_sdpa_backend(name, False)

    @staticmethod
    def _enable_torch_sdpa_backend(name: str, value: bool = True) -> None:
        try:
            enable_backend = getattr(torch.backends.cuda, f"enable_{name}_sdp")
        except AttributeError:
            return

        enable_backend(value)
