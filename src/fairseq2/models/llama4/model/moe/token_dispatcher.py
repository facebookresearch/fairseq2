# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import torch
from typing_extensions import override

from fairseq2.gang import Gang


class ShardingStrategy(Enum):
    TP = auto()
    # Not supported in fairseq2 for now
    # TP2EP = auto()
    DP2EP = auto()


class MoETokenDispatcher(ABC):
    def __init__(
        self,
        sharding_strategy: ShardingStrategy,
        num_experts: int,
    ) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        # Router weight is split along the expert dimension across TP group.
        # Each TP rank gets num_experts / TP * num_global_tokens_per_expert tokens.
        # For tp sharding, we apply allgather among the TP ranks to get the global tokens.
        # For tp2ep sharding, there is no need to do.
        # For dp2ep sharding, we apply all_to_all_sp_tp_sp among TP ranks to swap the layout
        # from [num_tokens/tp, dim] to [num_tokens, dim/tp], apply the a2a across EP ranks,
        # do a reverse all2all along TP ranks to revert the layout back to [num_tokens/tp, dim],
        # and an allgather among the TP ranks.
        self.sharding_strategy = sharding_strategy
        self.num_experts = num_experts

    @abstractmethod
    def token_permutation(
        self,
        tokens: torch.Tensor,  # [num_tokens, dim]
        permute_metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor: ...

    def wait_token_permutation(
        self,
        tokens: torch.Tensor,
        permute_metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return tokens

    @abstractmethod
    def token_unpermutation(
        self,
        tokens: torch.Tensor,
        permute_metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor: ...

    def wait_token_unpermutation(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return tokens


class TPTokenDispatcher(MoETokenDispatcher):
    """
    TP MoE Token Dispatcher. See token_permutation/token_unpermutation for more details.
    """

    def __init__(
        self,
        num_experts: int,
    ) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        super().__init__(ShardingStrategy.TP, num_experts)

    @override
    def token_permutation(
        self,
        tokens: torch.Tensor,  # [num_tokens, dim]
        permute_metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Dispatch tokens to local experts using TP sharding.

        Each TP rank gets num_experts experts.

        Args:
            tokens (torch.Tensor) [num_tokens, hidden_dim]:
                Input tokens.
            permute_metadata (Dict[str, torch.Tensor]):
                The metadata used for unpermutation.

        Returns:
            permuted_tokens (torch.Tensor) [num_local_experts*num_tokens_per_expert, hidden_dim]:
                Permuted tokens for local experts.
        """
        return tokens

    @override
    def token_unpermutation(
        self,
        tokens: torch.Tensor,
        permute_metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return tokens


def _all_to_all(
    gang: Gang,
    input_: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    async_op: bool = False,
) -> torch.Tensor:
    # Bypass the function if we are using only 1 GPU.
    if gang.size == 1:
        return input_

    input_ = input_.contiguous()
    if output_split_sizes is None:
        # Equal split (all2all)
        output = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        output = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
    handle = torch.distributed.all_to_all_single(
        output,
        input_,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=gang.as_process_group(),
        async_op=async_op,
    )
    return (output, handle) if async_op else output  # type: ignore[return-value]


def _gather_output_splits(
    input_splits: List[torch.Tensor], ep_gang: Gang
) -> List[torch.Tensor]:
    """Gather tensors and concatenate along the last dimension."""
    ep_size = ep_gang.size
    # Bypass the function if we are using only 1 GPU.
    if ep_size == 1:
        return input_splits

    local_output_splits = torch.tensor(
        input_splits[0],
        dtype=torch.int,
        device=torch.cuda.current_device(),
    )
    output_splits = torch.zeros(
        (ep_size),
        dtype=torch.int,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_gather_into_tensor(
        output_splits,
        local_output_splits,
        group=ep_gang.as_process_group(),
    )
    return output_splits.int().view(-1).tolist()


class DP2EPTokenDispatcher(MoETokenDispatcher):
    """
    DP2EP MoE Token Dispatcher. See token_permutation/token_unpermutation for more details.
    """

    ep_gang: Gang
    num_local_experts: int
    local_expert_indices: List[int]
    # This can be used when batch_size and seq_len are identical on DP ranks
    are_output_input_splits_equal: bool

    def __init__(
        self,
        num_experts: int,
        ep_gang: Gang,
        are_output_input_splits_equal: bool = True,
    ) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        super().__init__(ShardingStrategy.DP2EP, num_experts)
        self.ep_gang = ep_gang
        num_local_experts = num_experts // ep_gang.size
        self.num_local_experts: int = num_local_experts

        self.local_expert_indices: List[int] = list(
            range(
                ep_gang.rank * num_local_experts,
                (ep_gang.rank + 1) * num_local_experts,
            )
        )
        self.are_output_input_splits_equal = are_output_input_splits_equal

    @override
    def token_permutation(
        self,
        tokens: torch.Tensor,  # [num_tokens, dim]
        permute_metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Dispatch tokens to local experts using DP2EP sharding.

        Tokens in the ep DP groups will be dispatched to different experts.
        Each DP rank gets num_experts/EP local experts.
        Note that router_scores and router_indices are computed on bsz*seqlen*ep tokens,
        whereas x_aD only contains tokens from the current DP rank, i.e., num_tokens=bsz*seqlen.
        We use all2all to dispatch the tokens to the corresponding experts.

        When using sequence parallel, the input tokens are in the shape of [num_tokens/tp, dim].
        We will first apply an intra-node a2a to shuffle the tokens to [num_tokens, dim/tp].
        Then, we use all2all to dispatch the tokens to the assigned experts.
        On each TP rank, we then apply an allgather to obtain the routed tokens in the shape of
        [nmm_tokens, dim] before enterring the tensor parallel region.

        Args:
            permute_metadata (Dict[str, torch.Tensor]):
                The metadata used for unpermutation, which is a dictionary containing the following keys:
                a2a_input_permute_order (torch.Tensor) [num_tokens]:
                    The permutation order used by permuting the local tokens by experts used as a2a inputs.
                a2a_output_permute_order (torch.Tensor) [num_tokens]:
                    The permutation order used by permuting the a2a outputs by experts.
                input_splits (torch.Tensor) [ep]:
                    The input splits used by all2all.
                output_splits (torch.Tensor) [ep]:
                    The output splits used by all2all.

        Returns:
            routed_in_egD (torch.Tensor) [num_local_experts, num_global_tokens_per_expert, hidden_dim]:
                Permuted tokens for local experts.
        """
        # Prepare inputs_splits and output_splits for all2all
        if tokens.dim() == 1:
            B_T = tokens.shape
            D = 1
        else:
            B_T, D = tokens.shape  # type: ignore[assignment]

        assert permute_metadata is not None

        with torch.profiler.record_function("metadata preparation"), torch.no_grad():
            if "input_splits" not in permute_metadata["nt"]:
                # compute input/output splits for all2all
                tokens_per_expert = permute_metadata["nt"][
                    "tokens_per_expert"  # type: ignore[index]
                ]
                input_splits = [
                    tokens_per_expert * self.num_local_experts
                ] * self.ep_gang.size
                if (
                    self.are_output_input_splits_equal
                    or torch.cuda.graphs.is_current_stream_capturing()
                ):
                    # if cudagraph (decode), we assume all ranks are having the same
                    # batch size, otherwise cudagraph will be broken
                    output_splits = input_splits
                else:
                    # for eager, we will first collect the tokens from other experts
                    # to make sure alltoall will match. This will be slow due to the
                    # extra allgather but will be safe (if we cannot guarantee the
                    # prefill will have exactly the same token size)
                    output_splits = _gather_output_splits(
                        input_splits, ep_gang=self.ep_gang
                    )
                permute_metadata["nt"][
                    "input_splits"  # type: ignore[index]
                ] = input_splits  # type: ignore[assignment]
                permute_metadata["nt"][
                    "output_splits"  # type: ignore[index]
                ] = output_splits  # type: ignore[assignment]

        # Perform expert parallel a2a
        with torch.profiler.record_function("a2a"):
            tokens_eg_D = _all_to_all(
                self.ep_gang,
                tokens,
                permute_metadata["nt"]["output_splits"],  # type: ignore
                permute_metadata["nt"]["input_splits"],  # type: ignore
                async_op=True,  # return type is an async handle
            )
        return tokens_eg_D

    def wait_token_permutation(
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, Any],
        permute_metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        tokens, req = tokens[0], tokens[1]
        # req is an async handle
        req.wait()  # type: ignore
        return self.finalize_token_permutation(tokens, permute_metadata)

    def finalize_token_permutation(
        self,
        tokens_eg_D: torch.Tensor,
        permute_metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        assert permute_metadata is not None
        # Sort a2a outputs by local experts when num_local_experts > 1
        if tokens_eg_D.dim() == 1:
            D = 1
        else:
            _, D = tokens_eg_D.shape
        with torch.profiler.record_function("token local unpermute"):
            if self.num_local_experts > 1:
                if torch.cuda.graphs.is_current_stream_capturing():
                    tokens_per_expert = permute_metadata["nt"]["tokens_per_expert"]  # type: ignore[index]
                    tokens_eg_D_splits = torch.split(
                        tokens_eg_D,
                        tokens_per_expert,  # type: ignore[arg-type]
                        dim=0,
                    )
                else:
                    output_splits = permute_metadata["nt"]["output_splits"]  # type: ignore[index]
                    per_expert_output_splits = []
                    for i in output_splits:
                        per_expert_output_splits.extend(
                            [i // self.num_local_experts] * self.num_local_experts
                        )
                    tokens_eg_D_splits = torch.split(
                        tokens_eg_D, per_expert_output_splits, dim=0
                    )

                tokens_eg_D_reorder = []
                for eid in range(self.num_local_experts):
                    for epid in range(self.ep_gang.size):
                        tokens_eg_D_reorder.append(
                            tokens_eg_D_splits[epid * self.num_local_experts + eid]
                        )
                tokens_eg_D = torch.cat(tokens_eg_D_reorder, dim=0)

        return tokens_eg_D

    @override
    def token_unpermutation(
        self,
        tokens: torch.Tensor,
        permute_metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """ "Revert the token permutation to restore the original order.

        Args:
            routed_out_egD/tokens (torch.Tensor) [num_local_experts, num_tokens_per_expert, hidden_dim]:
                Output from local experts.
            permute_metadata (Dict[str, Any]):
                The metadata used for unpermutation, which is a dictionary containing the following keys:
                input_permute_order (torch.Tensor) [num_tokens]:
                    The permutation order used by permuting the local tokens by experts used as a2a inputs.
                output_permute_order (torch.Tensor) [num_tokens]:
                    The permutation order used by permuting the a2a outputs by experts.
                input_splits (torch.Tensor) [ep]:
                    The input splits used by all2all.
                output_splits (torch.Tensor) [ep]:
                    The output splits used by all2all.
        """
        assert permute_metadata is not None
        if tokens.dim() == 1:
            B_T = tokens.shape
            D = 1
        else:
            B_T, D = tokens.shape  # type: ignore[assignment]
        # Unpermute the output to a2a inputs when num_local_experts > 1
        with torch.profiler.record_function("token local unpermute"):
            if self.num_local_experts > 1:
                if torch.cuda.graphs.is_current_stream_capturing():
                    # for cudagraph, we know all ranks will have the same token size,
                    # therefore we can take the easy route of transpose*
                    tokens_per_expert = permute_metadata["nt"]["tokens_per_expert"]  # type: ignore
                    tokens_eg_D = (
                        tokens.view(
                            self.num_local_experts,
                            self.ep_gang.size,
                            tokens_per_expert,  # type: ignore
                            -1,
                        )
                        .transpose(0, 1)
                        .flatten(end_dim=2)
                    )
                else:
                    output_splits = permute_metadata["nt"]["output_splits"]  # type: ignore
                    tokens = tokens.view(-1, D)
                    per_expert_output_splits = [
                        i // self.num_local_experts for i in output_splits
                    ] * self.num_local_experts
                    tokens_reorder = torch.split(
                        tokens,
                        per_expert_output_splits,  # type: ignore[arg-type]
                        dim=0,
                    )
                    tokens_eg_D_reorder = []
                    for epid in range(self.ep_gang.size):
                        for eid in range(self.num_local_experts):
                            tokens_eg_D_reorder.append(
                                tokens_reorder[eid * self.ep_gang.size + epid]
                            )
                    tokens_eg_D = torch.cat(tokens_eg_D_reorder, dim=0)

            else:
                tokens_eg_D = tokens

        with torch.profiler.record_function("a2a"):
            tokens_eg_D = _all_to_all(
                self.ep_gang,
                tokens_eg_D,
                permute_metadata["nt"]["input_splits"],  # type: ignore
                permute_metadata["nt"]["output_splits"],  # type: ignore
                async_op=True,
            )  # [num_tokens, D]

        return tokens_eg_D

    def wait_token_unpermutation(
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, Any],
    ) -> torch.Tensor:
        tokens, req = tokens[0], tokens[1]
        # req is an async handle
        req.wait()  # type: ignore

        return self.finalize_token_unpermutation(tokens)

    def finalize_token_unpermutation(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return tokens
