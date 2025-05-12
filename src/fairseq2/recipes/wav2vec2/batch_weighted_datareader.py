# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from copy import deepcopy
from typing import Dict, Final, List, Tuple

from fairseq2.context import RuntimeContext
from fairseq2.data import DataPipeline
from fairseq2.data.text.tokenizers import TextTokenizer
from fairseq2.datasets import DataPipelineReader
from fairseq2.datasets.asr import AsrDataset, AsrReadOptions
from fairseq2.gang import Gang, Gangs
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.recipes.common import load_dataset
from fairseq2.recipes.config import DatasetSection

MIXTURE_DATASET_FAMILY: Final = "weighted_dataset_mixture"


class BatchMixtureDataset:

    _datasets: Dict[str, AsrDataset]
    _weights: Dict[str, float]
    _name: str

    name_pattern = r"^(?P<dataset_name>[a-zA-Z0-9_]+):(?P<weight>\d+(?:\.\d+)?)$"
    split_pattern = r"^(?P<dataset_name>[a-zA-Z0-9_]+)=(?P<split>[a-zA-Z0-9_]+)$"

    multi_split_pattern = (
        r"([a-zA-Z0-9_]+)=\[([^\]]+)\]|([a-zA-Z0-9_]+)=([a-zA-Z0-9_]+)"
    )

    def __init__(
        self, name: str, datasets: Dict[str, AsrDataset], weights: Dict[str, float]
    ) -> None:
        self._datasets = datasets
        self._weights = weights
        self._name = name

    @classmethod
    @classmethod
    def parse_split_config_with_multiple_splits(cls, split_config: str) -> List[str]:
        """
        input: "dataset0=[train,valid],dataset1=[test],dataset2=[train,valid,test]"
        output: ["dataset0=train", "dataset0=valid", "dataset1=test", "dataset2=train", "dataset2=valid", "dataset2=test"]
        """
        result = []
        # Use regex to find all dataset configurations
        matches = re.findall(cls.multi_split_pattern, split_config)

        for match in matches:
            if match[0]:  # Multi-split format: dataset=[split1,split2]
                dataset_name = match[0]
                splits = [s.strip() for s in match[1].split(",")]
                for split in splits:
                    result.append(f"{dataset_name}={split}")
            else:  # Single split format: dataset=split
                dataset_name = match[2]
                split = match[3]
                result.append(f"{dataset_name}={split}")

        return result

    @classmethod
    def parse_dataset_config(cls, config_str: str) -> List[Tuple[str, float]]:
        """
        Parse a dataset configuration string in the format "dataset0:10,dataset1:1,dataset2:3"
        and return a list of (dataset_name, weight) tuples.

        Args:
            config_str: String in the format "dataset0:10,dataset1:1,dataset2:3"

        Returns:
            List of (dataset_name, weight) tuples
        """
        result = []
        for item in config_str.split(","):
            match = re.match(cls.name_pattern, item.strip())
            if match is None:
                raise ValueError(f"Invalid dataset config format: {item}")

            dataset_name = match.group("dataset_name")
            weight = float(match.group("weight"))
            result.append((dataset_name, weight))

        return result

    @classmethod
    def parse_split_config(cls, config_str: str) -> List[Tuple[str, str]]:
        """
        Parse a dataset configuration string in the format "dataset0=train,dataset1=test"
        and return a list of (dataset_name, split) tuples.

        """
        result = []
        for item in config_str.split(","):
            match = re.match(cls.split_pattern, item.strip())
            if match is None:
                raise ValueError(f"Invalid dataset config format: {item}")

            dataset_name = match.group("dataset_name")
            split = match.group("split")
            result.append((dataset_name, split))

        return result

    @classmethod
    def from_configs(
        cls,
        dataset_cls: type[AsrDataset],
        context: RuntimeContext,
        dataset_config: DatasetSection,
        gangs: Gangs,
    ) -> "BatchMixtureDataset":
        datasets = {}
        weights = {}

        assert dataset_config.family == MIXTURE_DATASET_FAMILY
        assert dataset_config.name is not None

        for name, weight in cls.parse_dataset_config(dataset_config.name):
            n_ds_config = deepcopy(dataset_config)
            n_ds_config.name = name
            dataset = load_dataset(dataset_cls, context, n_ds_config, gangs)
            datasets[name] = dataset
            weights[name] = weight
        return cls(dataset_config.name, datasets, weights)

    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        min_audio_len: int,
        max_audio_len: int,
        options: AsrReadOptions | None = None,
    ) -> DataPipelineReader[Seq2SeqBatch]:

        if options is None:
            options = AsrReadOptions()

        splits = self.parse_split_config(split)

        datapipelines = []
        weigths = []
        datareader = None
        for dataset_name, dataset_split in splits:
            datareader = self._datasets[dataset_name].create_reader(
                split=dataset_split,
                tokenizer=tokenizer,
                gang=gang,
                min_audio_len=min_audio_len,
                max_audio_len=max_audio_len,
                options=options,
            )
            datapipelines.append(datareader._pipeline)  # type: ignore
            weigths.append(self._weights[dataset_name])

        if len(datapipelines) == 1:
            assert isinstance(datareader, DataPipelineReader)
            return datareader

        mixed_datapipeline = DataPipeline.sample(
            datapipelines, weights=weigths, seed=options.seed
        ).and_return()

        return DataPipelineReader[Seq2SeqBatch](
            self._name, split, mixed_datapipeline, gang, options
        )
