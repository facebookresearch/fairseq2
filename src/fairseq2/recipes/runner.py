# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Hashable, Set
from functools import cache
from pathlib import Path
from signal import SIGUSR1, signal
from types import FrameType
from typing import Protocol, TypeAlias, TypeVar, cast, runtime_checkable

from fairseq2.error import SetupError
from fairseq2.gang import get_rank, get_world_size, is_torchrun
from fairseq2.logging import log
from fairseq2.recipes.cluster import ClusterRegistry, register_clusters
from fairseq2.recipes.logging import setup_logging
from fairseq2.recipes.utils.log import log_config
from fairseq2.recipes.utils.sweep_tagger import SweepTagger
from fairseq2.setup import setup_fairseq2
from fairseq2.typing import DataClass
from fairseq2.utils.structured import unstructure
from fairseq2.utils.yaml import dump_yaml


@runtime_checkable
class Stoppable(Protocol):
    """Represents a task that supports graceful stopping."""

    def request_stop(self) -> None:
        ...


Recipe: TypeAlias = Callable[[], None]


RecipeConfigT = TypeVar("RecipeConfigT", bound=DataClass)

RecipeConfigT_contra = TypeVar(
    "RecipeConfigT_contra", bound=DataClass, contravariant=True
)


class RecipeLoader(Protocol[RecipeConfigT_contra]):
    def __call__(self, config: RecipeConfigT_contra, output_dir: Path) -> Recipe:
        ...


def run_recipe(
    loader: RecipeLoader[RecipeConfigT],
    preset: str,
    config: RecipeConfigT,
    output_dir: Path,
    *,
    cluster: str = "auto",
    no_sweep_dir: bool = False,
    sweep_format: str | None = None,
    extra_sweep_keys: Set[Hashable] | None = None,
    debug: bool = False,
) -> None:
    set_cluster_environment(cluster)

    unstructured_config = unstructure(config)

    output_dir = output_dir.expanduser().resolve()

    if not no_sweep_dir:
        tag = generate_sweep_tag(
            preset, unstructured_config, sweep_format, extra_sweep_keys
        )

        output_dir = output_dir.joinpath(tag)

    setup_distributed_logging(output_dir, debug)

    log_config(unstructured_config, log)

    dump_config_to_file(output_dir, unstructured_config)

    setup_fairseq2()

    recipe = loader(config, output_dir)

    # If the recipe is stoppable, use SIGUSR1 as the stop signal.
    if isinstance(recipe, Stoppable):

        def request_stop(signum: int, frame: FrameType | None) -> None:
            log.info("SIGUSR1 received. Requesting recipe to stop.")

            cast(Stoppable, recipe).request_stop()

        signal(SIGUSR1, request_stop)

    recipe()


def set_cluster_environment(cluster: str) -> None:
    registry = ClusterRegistry(is_torchrun=is_torchrun())

    register_clusters(registry)

    handler = registry.get(cluster)

    handler.set_torch_distributed_variables()


def generate_sweep_tag(
    preset: str,
    unstructured_config: object,
    sweep_format: str | None,
    extra_sweep_keys: Set[Hashable] | None,
) -> str:
    world_size = get_world_size()

    tagger = create_sweep_tagger(world_size, extra_sweep_keys)

    return tagger.generate(preset, unstructured_config, sweep_format)


def create_sweep_tagger(
    world_size: int, extra_sweep_keys: Set[Hashable] | None
) -> SweepTagger:
    sweep_keys = get_default_sweep_keys()

    if extra_sweep_keys is not None:
        sweep_keys = sweep_keys | extra_sweep_keys

    return SweepTagger(world_size, sweep_keys)


@cache
def get_default_sweep_keys() -> Set[Hashable]:
    return {
        "batch_shuffle_window",
        "betas",
        "data_parallelism",
        "dataset",
        "dtype",
        "example_shuffle_window",
        "final_lr_ratio",
        "final_lr_scale",
        "fp16_loss_scale",
        "fsdp_reshard_after_forward",
        "fsdp_wrap_granularity",
        "gradient_accumulation",
        "label_smoothing",
        "lr",
        "lr_stage_ratios",
        "max_gradient_norm",
        "max_num_elements",
        "max_num_steps",
        "max_num_tokens",
        "max_seq_len",
        "mixed_precision",
        "model",
        "model_arch",
        "model_config",
        "num_lr_warmup_steps",
        "pretrained_model",
        "seed",
        "split",
        "start_lr",
        "start_lr_scale",
        "tensor_parallel_size",
        "tokenizer",
        "train_split",
        "valid_split",
        "weight_decay",
    }


def setup_distributed_logging(output_dir: Path, debug: bool) -> None:
    log_file = output_dir.joinpath("logs/rank_{rank}.log")

    try:
        setup_logging(log_file, debug=debug)
    except OSError as ex:
        raise SetupError(
            "The distributed logging setup has failed. See the nested exception for details."
        ) from ex

    log.info("The log files are stored under the '{}' directory.", log_file.parent)


def dump_config_to_file(output_dir: Path, unstructured_config: object) -> None:
    rank = get_rank()
    if rank != 0:
        return

    config_file = output_dir.joinpath("config.yaml")

    try:
        dump_yaml(unstructured_config, config_file)
    except OSError as ex:
        raise SetupError(
            f"The recipe configuration cannot be saved to the '{config_file}' file. See the nested exception for details."
        ) from ex
