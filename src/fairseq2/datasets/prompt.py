# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, List, cast, final
from functools import partial
import torch
from typing_extensions import override
import re
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import (
    DataPipelineReader,
    DataReadOptions,
    DatasetHubAccessor,
    DatasetLoadError,
    LengthBatching,
    StaticBatching,
    UnknownSplitError,
)
from fairseq2.datasets.utils._manifest import _load_files_and_weights
from fairseq2.device import Device
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gang


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


class PromptDataset(ABC):
    """Represents a preference optimization dataset."""

    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: PromptReadOptions | None = None,
    ) -> DataPipelineReader[PromptBatch]:
        """Create a dataset reader.

        :param tokenizer:
            The tokenizer to encode text.
        :param gang:
            The gang over which to shard the dataset.
        :param min_seq_len:
            The minimum sequence length of each example. Examples shorter than
            this value will be dropped.
        :param max_seq_len:
            The maximum sequence length of each example. Examples longer than
            this value will be dropped.
        :param options:
            The read options.
        """


# TODO: FIX, INFER
npc = 10


GENERIC_PROMPT_DATASET_FAMILY: Final = "prompt_dataset"


@final
class GenericPromptDataset(PromptDataset):
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

    @staticmethod
    def from_path(path: Path, name: str) -> GenericPromptDataset:
        splits: dict[str, tuple[Sequence[Path], Sequence[float]]] = {}

        if path.is_dir():
            try:
                child_dirs = [p for p in path.iterdir() if p.is_dir()]
            except OSError as ex:
                raise DatasetLoadError(
                    name, f"The files under the '{path}' directory of the '{name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
                ) from ex

            for child_dir in child_dirs:
                files, weights = _load_files_and_weights(name, child_dir)

                splits[child_dir.name] = (files, weights)

        if not splits:
            files, weights = _load_files_and_weights(name, path)

            splits["default"] = (files, weights)

        return GenericPromptDataset(name, splits)

    @override
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_seq_len: int,
        max_seq_len: int,
        options: PromptReadOptions | None = None,
    ) -> DataPipelineReader[PromptBatch]:
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

        def apply_prmopt_template(
            example: dict[str, Any], prompt_template: str, remove_suffix_field: str
        ) -> dict[str, Any]:
            if remove_suffix_field is not None:
                example[f"{src_key}"] = example[f"{src_key}"].removesuffix(
                    example[remove_suffix_field]
                )
            if prompt_template is not None:
                example[f"{src_key}"] = prompt_template.replace(
                    "{prompt}", example[f"{src_key}"]
                )
            return example

        if (
            "prompt_template" in options.extras
            or "remove_suffix_field" in options.extras
        ):
            builder.map(
                partial(
                    apply_prmopt_template,
                    prompt_template=options.extras.get("prompt_template", None),
                    remove_suffix_field=options.extras.get("remove_suffix_field", None),
                ),
                num_parallel_calls=npc,
            )

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

    def _read_jsonl(self, path: Path, tokenizer: TextTokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open() as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads, num_parallel_calls=npc)


get_preference_dataset_hub = DatasetHubAccessor(PromptDataset)
