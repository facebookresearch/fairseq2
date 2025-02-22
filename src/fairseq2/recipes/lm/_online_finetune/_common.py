# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor
from torcheval.metrics import Mean

from fairseq2.datasets.preference import PreferenceBatch
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch, SequenceModelOutput
from fairseq2.recipes.metrics import SequenceMetricBag

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker
import os

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
)
from fairseq2.nn.padding import get_seqs_and_padding_mask


@dataclass(kw_only=True)
class OnlineCriterionSection:
    name: str

    config: object


class OnlineFinetuneMetricBag(SequenceMetricBag):

    def __init__(self, gang: Gang) -> None:
        super().__init__(gang)


def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes) 
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl

@ray.remote
class NoEnvLLM(LLM):

    def __init__(self, *args, **kwargs):
        # a hack to make the script work.
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        del os.environ["CUDA_VISIBLE_DEVICES"]
        super().__init__(*args, **kwargs)

        self.ready = True  # Set a flag or return a signal

    def is_ready(self):
        return self.ready

class MyWorker(Worker):
    """
    The `MyWorker` class inherits from `Worker` to provide custom functions.
    For simplicity, we define the `MyWorker` class in this self-contained 
    script. Normally, we should define the `MyWorker` class in a separate 
    file and pass the qualified name of the class to the `worker_cls` 
    parameter.
    """

    def init_weight_update_group(self, master_address, master_port,
                                 rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        # wrap in fs2 style dict
        weights = {
            "model_key": "model",
            "model": {name: weight}
        }.items()
        #self.model_runner.model.load_weights(weights=[(name, weight)])
        self.model_runner.model.load_weights(weights=weights)

        del weight

def setup_vllm(actor_name, vllm_init_checkpoint_dir, vllm_init_tokenizer, tensor_parallel_size):

    pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * tensor_parallel_size)

    ray.get(pg_inference.ready())

    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )

    """
    launch the vLLM inference engine.
    here we use `enforce_eager` to reduce the start time.
    """ 
    llm = NoEnvLLM.options(name=actor_name,num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,get_if_exists=True).remote(
        model=vllm_init_checkpoint_dir,
        tokenizer=vllm_init_tokenizer,
        enforce_eager=True,
        worker_cls=MyWorker,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="ray",
    )

    # we block here until the engine is initialized
    ray.get(llm.is_ready.remote())

    return llm

def cat_source_and_target(example: dict[str, Any], mask_source_tokens=True) -> dict[str, Any]:
    id_ = example.get("id", None)

    source_indices = example["src"]
    target_indices_chosen = example["tgt_chosen"]
    target_indices_rejected = example["tgt_rejected"]

    indices_chosen = torch.cat([source_indices, target_indices_chosen])
    indices_rejected = torch.cat([source_indices, target_indices_rejected])

    if mask_source_tokens:
        source_len = len(source_indices)
        target_mask_chosen = torch.arange(len(indices_chosen)) >= source_len
        target_mask_rejected = torch.arange(len(indices_rejected)) >= source_len
    else:
        target_mask_chosen = torch.full([len(indices_chosen)], True)
        target_mask_rejected = torch.full([len(indices_rejected)], True)

    total_tokens = (
        2 * len(source_indices)
        + len(target_indices_chosen)
        + len(target_indices_rejected)
    )

    # below is an example of using extras field of data reader options
    # if "keep_jsonl_keys" in options.extras:
    #     jsonl_keys = options.extras["keep_jsonl_keys"]
    #     if not (
    #         isinstance(jsonl_keys, list)
    #         and all(isinstance(i, str) for i in jsonl_keys)
    #     ):
    #         raise ValueError(f"{jsonl_keys} must be a list of strings")
    #     jsonl_content = {k: example.get(k, None) for k in jsonl_keys}
    # else:
    #     jsonl_content = None

    return {
        "id": id_,
        "indices_prompt": source_indices,
        "indices_chosen": indices_chosen,
        "indices_rejected": indices_rejected,
        "reference_score_chosen": example.get("reference_score_chosen", None),
        "reference_score_rejected": example.get(
            "reference_score_rejected", None
        ),
        "target_mask_chosen": target_mask_chosen,
        "target_mask_rejected": target_mask_rejected,
        "total_tokens": total_tokens,
        # "keep_jsonl_keys": jsonl_content,
    }

def to_preference_batch(example: dict[str, Any], device) -> PreferenceBatch:
    indices_chosen = cast(SequenceData, example["indices_chosen"])
    indices_rejected = cast(SequenceData, example["indices_rejected"])

    seqs_chosen, padding_mask_chosen = get_seqs_and_padding_mask(
        indices_chosen, device
    )
    seqs_rejected, padding_mask_rejected = get_seqs_and_padding_mask(
        indices_rejected, device
    )

    target_mask_chosen = example["target_mask_chosen"]["seqs"].to(device)
    target_mask_rejected = example["target_mask_rejected"]["seqs"].to(device)  # fmt: skip

    batch_chosen = SequenceBatch(
        seqs_chosen,
        padding_mask_chosen,
        target_mask_chosen,
        example=example,
    )

    batch_rejected = SequenceBatch(
        seqs_rejected,
        padding_mask_rejected,
        target_mask_rejected,
        example=example,
    )

    batch_reference_scores_chosen = None
    if all(example["reference_score_chosen"]):
        batch_reference_scores_chosen = torch.Tensor(
            example["reference_score_chosen"]
        ).to(device)
    batch_reference_scores_rejected = None
    if all(example["reference_score_rejected"]):
        batch_reference_scores_rejected = torch.Tensor(
            example["reference_score_rejected"]
        ).to(device)

    return PreferenceBatch(
        batch_chosen,
        batch_rejected,
        batch_reference_scores_chosen,
        batch_reference_scores_rejected,
    )