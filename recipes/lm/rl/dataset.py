# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Set, cast, final
from typing_extensions import override
from functools import partial

from fairseq2.device import Device, SupportsDeviceTransfer
import torch

from fairseq2.assets import HuggingFaceHub
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
)
from fairseq2.data.data_pipeline import DataPipeline, read_sequence
from fairseq2.data.text import read_text
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data.tokenizers.hg import HuggingFaceTokenEncoder
from fairseq2.datasets import DataPipelineReader, SequenceBatch, SyncMode
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.utils.uri import Uri

from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


LM_RL_DATASET: Final = "lm_rl"

def default_apply_chat_template_args():
    return {
        "add_generation_prompt": True,
    }

@dataclass
class PromptBatch(SupportsDeviceTransfer):
    """Represents a preference optimization dataset batch."""

    prompts: List[List[int]]
    meta_info: dict[str,list]

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

    extras: Dict[str, object] = field(default_factory=dict)
    """The reader-specific extra options."""

    source_encode_mode: str = "prompt"
    """The tokenizer mode to encode the source text."""

    chat_mode: bool = False

    src_key: str = "src"

    repeat_batch_n_times: int = 1

    apply_chat_template_args: Dict[str, object] = field(default_factory=default_apply_chat_template_args)
    """This will be passed as is to apply chat template e.g. enable_thinking"""

    subsample: bool = False
    """
    If ``True``, instruction sources (e.g. JSONL files) will be sampled in
    proportion to their weights.
    """


@final
class LMRLDataset:
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

    def _read_jsonl(self, path: Path, tokenizer: Tokenizer) -> DataPipelineBuilder:
        lines = []

        # TODO(balioglu): Do in C++.
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                lines.append(line)

        return read_sequence(lines).map(json.loads)

    def create_reader(
        self,
        split: str,
        tokenizer: Tokenizer,
        gangs: Gangs,
        options: DataReadOptions | None = None,
    ) -> DataPipelineReader[SequenceBatch]:

        files_weights = self._splits.get(split)
        if files_weights is None:
            raise ValueError(f"files_weights for split '{split}' is None")
        files, weights = files_weights

        if options is None:
            options = DataReadOptions()

        seed = options.seed

        builder = read_sequence(files)

        def read_json_or_jsonl(path: Path) -> DataPipelineBuilder:
            lines = []

            if path.suffix == ".jsonl":
                with path.open() as fp:
                    for line in fp:
                        lines.append(line)

                return read_sequence(lines).map(json.loads, num_parallel_calls=options.npc)
            
            elif path.suffix == ".json":
                lines = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(lines, list):
                    raise RuntimeError("json files expect to contain list of dicts")
                return read_sequence(lines)
            
            else:
                raise NotImplementedError(f"Only jsonl and json files are supported")


        if len(files) == 1:
            builder = read_json_or_jsonl(files[0])
        else:
            pipelines = []

            for file in files:
                pipeline = read_json_or_jsonl(file).and_return()

                pipelines.append(pipeline)

            if options.subsample:
                builder = DataPipeline.sample(pipelines, weights=weights, seed=seed)

                seed += 1
            else:
                builder = DataPipeline.concat(pipelines)

        # Shuffle files. Must be consistent across all processes.
        if options.example_shuffle_window != 1:
            builder.shuffle(options.example_shuffle_window, seed=seed)

        seed += 1

        # Shard.
        builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)

        seed += gangs.dp.rank

        if options.chat_mode is True:
            # not passing any encoding modes here, because we use apply_chat_template here
            encoder = tokenizer.create_encoder()
            if not isinstance(encoder, HuggingFaceTokenEncoder):
                raise RuntimeError(
                    "Huggingface tokenizer must be used when chat_mode is True"
                )
            else:

                def encoding_chat(example: dict[str, Any]) -> dict[str, Any]:
                    chat = example.get("chat", None)

                    if not chat:
                        chat = [
                            {"role": "user", "content": example.get(options.src_key)},
                        ]

                    encoded_output = encoder.apply_chat_template(
                        chat,
                        return_dict=True,
                        return_tensors="pt",
                        **options.apply_chat_template_args
                    )

                    indices = encoded_output["input_ids"][0]

                    example[f"{options.src_key}_indices"] = indices

                    return example
                
                builder.map(encoding_chat)

        else:
            def copy_key(src_key: str, tgt_key: str, example: dict[str, Any]) -> dict[str, object]:
                example[tgt_key] = example[src_key]
                return example
            
            builder.map(partial(copy_key, options.src_key, f"{options.src_key}_indices"))

            # Encode source using tokenizer without template wrap
            source_encoder = tokenizer.create_encoder(mode=options.source_encode_mode)
            builder.map(source_encoder, selector=f"{options.src_key}_indices")

        def process_examples(example: dict[str, Any]) -> dict[str, Any]:
            id_ = example.get("id", None)

            source_indices = example[f"{options.src_key}_indices"].tolist()

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

        builder.map(process_examples)

        batching = options.batching

        # Bucket `batch_size` examples.
        builder.bucket(batching.batch_size, drop_remainder=options.drop_remainder)
        
        # Shuffle buckets.
        if options.batch_shuffle_window != 1:
            builder.shuffle(options.batch_shuffle_window, seed=seed)

        seed += 1

        # Prefetch `prefetch` batches in background.
        builder.prefetch(options.prefetch)

        def glue_jsonl_columns(list_of_dicts, keep_jsonl_keys):
            result = {}
            for k in keep_jsonl_keys:
                result[k] = [d["keep_jsonl_keys"][k] for d in list_of_dicts]

            return result

        # Wrap examples with `PromptBatch`.
        def to_batch(example: dict[str, Any]) -> PromptBatch:
            prompts = [e["indices_prompt"] for e in example]
            if "keep_jsonl_keys" in options.extras:
                meta_info = glue_jsonl_columns(example, options.extras["keep_jsonl_keys"])
            else:
                meta_info = None
            return PromptBatch(prompts=prompts, meta_info=meta_info)

        pipeline = (
            builder.map(to_batch)
            .yield_from(
                lambda x: read_sequence([x] * options.repeat_batch_n_times).and_return()
            )
            .and_return()
        )

        return DataPipelineReader[PromptBatch](
            pipeline, gangs, num_accumulate=options.num_accumulate,
            drop_remainder=options.drop_remainder,
            sync=options.sync_batches,
            sync_mode=options.sync_mode,
        )


@dataclass
class LMRLDatasetConfig:
    path: str | None = None


def open_rl_dataset(config: LMRLDatasetConfig) -> LMRLDataset:
    name = "default"  # FIXME
    splits: dict[str, tuple[Sequence[Path], Sequence[float]]] = {}

    if config.path is None:
        raise ValueError("config.path cannot be None")

    uri = Uri.maybe_parse(config.path)
    if uri and uri.scheme == "hg":
        path = HuggingFaceHub().download_dataset(uri, config.path)
    else:
        path = Path(config.path)

    if path.is_dir():
        try:
            child_dirs = [p for p in path.iterdir() if p.is_dir()]
        except OSError as ex:
            raise DatasetLoadError(
                name, f"The files under the '{path}' directory of the '{name}' dataset cannot be retrieved. See the nested exception for details."  # fmt: skip
            ) from ex

        for child_dir in child_dirs:
            files, weights = load_files_and_weights(name, child_dir)

            splits[child_dir.name] = (files, weights)

    if not splits:
        files, weights = load_files_and_weights(name, path)

        splits["default"] = (files, weights)

    return LMRLDataset(name, splits)


class DatasetLoadError(Exception):
    dataset_name: str

    def __init__(self, dataset_name: str, message: str) -> None:
        super().__init__(message)

        self.dataset_name = dataset_name


class UnknownSplitError(ValueError):
    dataset_name: str
    split: str
    available_splits: Set[str]

    def __init__(
        self, dataset_name: str, split: str, available_splits: Set[str]
    ) -> None:
        s = ", ".join(sorted(available_splits))

        super().__init__(
            f"'{split}' is not a known split of the '{dataset_name}' dataset. The following splits are available: {s}"
        )

        self.dataset_name = dataset_name
        self.split = split
        self.available_splits = available_splits


@dataclass
class StaticBatching:
    """Specifies batching where each batch has the same number of examples."""

    batch_size: int
    """The number of examples in each batch."""


def load_files_and_weights(
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
            files = list(path.glob("**/*.json*"))
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

        def error() -> DatasetLoadError:
            return DatasetLoadError(
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
            raise DatasetLoadError(
                dataset_name, f"The '{file}' path referred at line {idx} in the '{manifest_file}' manifest file of the '{dataset_name}' dataset does not exist."  # fmt: skip
            )

        files.append(file)

        try:
            weight = float(fields[1].strip())
        except ValueError:
            raise error() from None

        weights.append(weight)

    return files, weights

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