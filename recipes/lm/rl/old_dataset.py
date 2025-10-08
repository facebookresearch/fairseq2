# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, List, MutableMapping, cast, final

import torch
from typing_extensions import override

from fairseq2.data.data_pipeline import (
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    read_sequence,
    CollateOptionsOverride
)

from fairseq2.datasets import (
    DataReader,
    SequenceBatch,
    SyncMode
)
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DatasetHubAccessor
)
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang

@dataclass
class StaticBatching:
    """Specifies batching where each batch has the same number of examples."""

    batch_size: int
    """The number of examples in each batch."""

@dataclass(kw_only=True)
class DataReadOptions:
    batching: StaticBatching = field(default_factory=lambda: StaticBatching(1))
    """The batching strategy for returned examples."""

    example_shuffle_window: int = 0
    """
    The size of the sliding window for shuffling examples. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    batch_shuffle_window: int = 0
    """
    The size of the sliding window for shuffling batches. If ``1``, no
    shuffling is performed; if ``0``, true shuffling is performed by loading the
    entire dataset.
    """

    drop_remainder: bool = False
    """
    If ``True``, drops the last set of batches if they have in total fewer
    examples than requested.
    """

    sync_batches: bool = True
    """
    If ``True``, ensures that each process in the gang reads the same number of
    batches. Typically used when the amount of data to be read can vary per
    process (e.g. due to unbalanced sharding or non-static batching) and it is
    critical for each process to iterate over the same number of batches (e.g.
    during training).
    """

    sync_mode: SyncMode = SyncMode.UNTIL_FIRST
    """
    The data synchronization mode among processes in the gang. Only effective if
    :attr:`sync_batches` is ``True``.
    """

    max_num_batches: int | None = None
    """The maximum number of batches to return."""

    num_accumulate: int = 1
    """
    The number of batches to accumulate in each iteration. Typically used with
    gradient accumulation during training.
    """

    prefetch: int = 1
    """The number of batches to prefetch in background."""

    npc: int = 10
    """The reference number of parallel calls that data reader can do."""

    seed: int = 2
    """The seed to initialize the random number generators used internally."""

    extras: MutableMapping[str, object] = field(default_factory=dict)
    """The reader-specific extra options."""

    sample: bool = False
    """
    If ``True``, instruction sources (e.g. JSONL files) will be sampled in
    proportion to their weights.
    """

    source_encode_mode: str = "prompt"
    """The tokenizer mode to encode the source text."""

    target_encode_mode: str = "prompt_response"
    """The tokenizer mode to encode the target text."""

    chat_mode: bool = False

@dataclass(kw_only=True)
class PromptReadOptions(DataReadOptions):
    sample: bool = False
    """
    If ``True``, instruction sources (e.g. JSONL files) will be sampled in
    proportion to their weights.
    """

    mask_source_tokens: bool = True
    """
    If ``False``, calculates loss on the source tokens (prompt) as well as the
    target tokens.
    """

    source_encode_mode: str = "prompt"
    """The tokenizer mode to encode the source text."""

    src_key: str = "src"

    repeat_batch_n_times: int = 1


@dataclass
class PromptBatch:
    """Represents a preference optimization dataset batch."""

    prompts: List[List[int]]
    meta_info: List[Any]

    @property
    def batch_size(self) -> int:
        """The size of the batch dimension."""
        return len(self.prompts)

    @property
    def prompt_lengths(self) -> list[int]:
        return [len(p) for p in self.prompts]

    @override
    def to(self, device: Device, *, non_blocking: bool = False) -> None:
        # no device moving since we only carry tokens prompts here
        pass

# TODO: FIX, INFER
npc = 10


RL_TRAIN_DATASET: Final = "rl_dataset"

@final
class RLDataset:
    """Represents a generic JSONL preference optimization dataset."""

    _name: str
    _splits: dict[str, tuple[Sequence[Path], Sequence[float]]]

    def __init__(
        self, name: str, splits: dict[str, tuple[Sequence[Path], Sequence[float]]]
    ) -> None:
        """
        :param files:
            The instruction files.
        :param weights:
            The weight of each file in ``files``.
        """
        self._name = name

        for split, (files, weights) in splits.items():
            if len(files) != len(weights):
                raise ValueError(
                    f"The lengths of the file and weight lists of the '{split}' split must match, but they are {len(files)} and {len(weights)} instead."
                )

        self._splits = splits

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: Tokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: PromptReadOptions | None = None,
    ) -> DataReader[PromptBatch]:
        if options is None:
            options = PromptReadOptions()

        seed = options.seed
        src_key = options.src_key

        files_weights = self._splits.get(split)
        if files_weights is None:
            raise UnknownSplitError(self._name, split, self._splits.keys())

        if options is None:
            options = PromptReadOptions()

        seed = options.seed

        files, weights = files_weights

        if len(files) == 1:
            builder = self._read_jsonl(files[0], tokenizer)
        else:
            pipelines = []

            for file in files:
                pipeline = self._read_jsonl(file, tokenizer).and_return()

                pipelines.append(pipeline)

            if options.sample:
                builder = DataPipeline.sample(pipelines, weights=weights, seed=seed)

                seed += 1
            else:
                builder = DataPipeline.concat(pipelines)

        # Shuffle files. Must be consistent across all processes.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed=seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        # copy original prompt text in the example to use it in the reward models etc.
        # user must specify f"{src_key}_text" in keep jsonl keys arg to keep pass it to batch
        def copy_prompt_text(example: dict[str, Any]) -> dict[str, Any]:
            example[f"{src_key}_text"] = example[f"{src_key}"]
            return example

        builder.map(copy_prompt_text, num_parallel_calls=npc)

        # Encode source and target texts.
        source_encoder = tokenizer.create_encoder(mode=options.source_encode_mode)

        builder.map(source_encoder, selector=src_key, num_parallel_calls=npc)

        def process_examples(example: dict[str, Any]) -> dict[str, Any]:
            id_ = example.get("id", None)

            source_indices = example[src_key].tolist()

            total_tokens = len(source_indices)

            # below is an example of using extras field of data reader options
            if "keep_jsonl_keys" in options.extras:
                jsonl_keys = options.extras["keep_jsonl_keys"]
                if not (
                    isinstance(jsonl_keys, list)
                    and all(isinstance(i, str) for i in jsonl_keys)
                ):
                    raise ValueError(f"{jsonl_keys} must be a list of strings")
                jsonl_content = {}
                for k in jsonl_keys:
                    if k not in example:
                        raise KeyError(
                            f"Required key '{k}' not found in example dictionary."
                        )
                    jsonl_content[k] = example[k]
            else:
                jsonl_content = None

            return {
                "id": id_,
                "indices_prompt": source_indices,
                "total_tokens": total_tokens,
                "keep_jsonl_keys": jsonl_content,
            }

        builder.map(process_examples, num_parallel_calls=npc)

        batching = options.batching

        if isinstance(batching, StaticBatching):
            # Filter out long examples.
            def skip(example: dict[str, Any]) -> bool:
                total_tokens = example["total_tokens"]

                return total_tokens <= max_seq_len and total_tokens >= min_seq_len

            builder.filter(skip)

            # Bucket `batch_size` examples.
            builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        else:
            raise NotSupportedError(f"`{batching}` is not supported.")

        # Shuffle buckets.

        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed=seed)

        seed += 1

        # Return only the first `max_num_batches`.
        if options.max_num_batches is not None:
            builder.take(options.max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(options.num_prefetch)

        def glue_jsonl_columns(list_of_dicts, keep_jsonl_keys):
            result = {}
            for k in keep_jsonl_keys:
                result[k] = [d["keep_jsonl_keys"][k] for d in list_of_dicts]

            return result

        # Wrap examples with `PromptBatch`.
        def to_batch(example: dict[str, Any]) -> PromptBatch:
            prompts = [e["indices_prompt"] for e in example]
            meta_info = glue_jsonl_columns(example, options.extras["keep_jsonl_keys"])
            return PromptBatch(prompts=prompts, meta_info=meta_info)

        pipeline = (
            builder.map(to_batch)
            .yield_from(
                lambda x: read_sequence([x] * options.repeat_batch_n_times).and_return()
            )
            .and_return()
        )

        return DataPipelineReader[PromptBatch](
            self._name, split, pipeline, gang, options
        )

    def _read_jsonl(self, path: Path, tokenizer: Tokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open() as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)

def _load_files_and_weights(
    dataset_name: str, path: Path
) -> tuple[list[Path], list[float]]:
    path = path.expanduser().resolve()

    if not path.is_dir():
        return [path], [1.0]

    manifest_file = path.joinpath("MANIFEST")

    try:
        with manifest_file.open(encoding="utf-8") as fp:
            content = list(fp)
    except FileNotFoundError:
        content = None
    except OSError as ex:
        raise DatasetLoadError(
            dataset_name, f"The '{manifest_file}' manifest file of the '{dataset_name}' dataset cannot be read. See the nested exception for details."  # fmt: skip
        ) from ex

    # If the directory does not contain a MANIFEST file, treat all JSONL
    # files as part of the dataset with equal weight.
    if content is None:
        try:
            files = list(path.glob("**/*.jsonl"))
        except OSError as ex:
            raise DatasetLoadError(
                dataset_name, f"The JSONL files under the '{path}' directory of the '{dataset_name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
            ) from ex

        weights = [1.0 for _ in range(len(files))]

        return files, weights

    # Sort the JSONL files in alphabetical order.
    content.sort()

    files = []

    weights = []

    # Each line of the MANIFEST file corresponds to the path of a JSONL file
    # and its weight (e.g. number of examples).
    for idx, line in enumerate(content):

        def error():
            return RuntimeError(
                dataset_name, f"Each line in the '{manifest_file}' manifest file of the '{dataset_name}' dataset must represent a path to a JSONL file and a weight, but line {idx} is '{line}' instead."  # fmt: skip
            )

        fields = line.rstrip().split("\t")

        if len(fields) != 2:
            raise error()

        file_path = fields[0].strip()
        if not file_path:
            raise error()

        try:
            file = path.joinpath(file_path)
        except ValueError:
            raise error() from None

        if not file.exists():
            raise RuntimeError(
                dataset_name, f"The '{file}' path referred at line {idx} in the '{manifest_file}' manifest file of the '{dataset_name}' dataset does not exist."  # fmt: skip
            )

        files.append(file)

        try:
            weight = float(fields[1].strip())
        except ValueError:
            raise error() from None

        weights.append(weight)

    return files, weights

@dataclass(kw_only=True)
class RLDatasetConfig:
    path: str

@staticmethod
def open_rl_dataset(config: RLDatasetConfig) -> RLDataset:
    splits: dict[str, tuple[Sequence[Path], Sequence[float]]] = {}

    if config.path.is_dir():
        try:
            child_dirs = [p for p in config.path.iterdir() if p.is_dir()]
        except OSError as ex:
            raise RuntimeError(
                f"The files under the '{config.path}' directory of the dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
            ) from ex

        for child_dir in child_dirs:
            files, weights = _load_files_and_weights(name, child_dir)

            splits[child_dir.name] = (files, weights)

    if not splits:
        files, weights = _load_files_and_weights(name, path)

        splits["default"] = (files, weights)

    return RLDataset(name, splits)

def collate_with_target_mask(
    list_of_tensors, prompt_lengths, pad_value=0, device="cpu"
):
    # list_of_tensors contain prompt+rollout tokens, we use prompt_len to define the target loss mask here
    to_collate = []
    for seq, prompt_len in zip(list_of_tensors, prompt_lengths):
        target_loss_mask = torch.arange(len(seq)) >= prompt_len
        to_collate.append({"seqs": seq, "target_loss_mask": target_loss_mask})

    target_mask_collate_opts = [
        CollateOptionsOverride("target_loss_mask", pad_value=False),
    ]
    collater = Collater(
        pad_value=pad_value, pad_to_multiple=1, overrides=target_mask_collate_opts
    )
    # from fairseq2.utils.env import get_rank
    # from os import environ
    # if get_rank(environ) == 0:
    #     import ipdb; ipdb.set_trace()
    # torch.distributed.barrier()

    seq_data = cast(SequenceData, collater(to_collate))

    batch = SequenceBatch(
        seq_data["seqs"]["seqs"],
        seq_data["seqs"]["seq_lens"],
        target_mask=seq_data["target_loss_mask"]["seqs"],
    )
    batch.to(device)

    return batch