# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Factory for creating NemotronH models.

Builds a TransformerLM with the 3-way hybrid decoder:
- Mamba2 SSM blocks at 'M' positions
- MoE FFN blocks at 'E' positions
- Standard GQA Attention blocks at 'A' positions

Tensor Parallelism:
- Attention layers are "sharding-aware" via StandardMultiheadAttention(gangs=...)
- MoE layers: shared + routed experts sharded via ColumnShardedLinear/RowShardedLinear
  when gangs.tp.size > 1; tp_gang set for final all-reduce.
- Mamba2 layers: TP deferred to future PR (requires splitting SSM heads across ranks).
- Embedding and final_proj: use VocabShardedEmbedding and ColumnShardedLinear
  which are inherently sharding-aware.
"""

from __future__ import annotations

from torch.nn import Module

from fairseq2.gang import Gang, Gangs, get_current_gangs
from fairseq2.models.nemotron.config import NemotronHConfig
from fairseq2.models.nemotron.decoder_layer import NemotronHBlock
from fairseq2.models.nemotron.mamba2 import NemotronHMamba2Mixer
from fairseq2.models.nemotron.moe import NemotronHMoE
from fairseq2.models.transformer import (
    CausalAttentionBias,
    MultiheadAttention,
    StandardMultiheadAttention,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    create_default_sdpa,
)
from fairseq2.models.transformer_lm import (
    StandardTransformerLMDecoder,
    TransformerLM,
    TransformerLMDecoder,
    TransformerLMDecoderLayer,
)
from fairseq2.nn import (
    ColumnShardedLinear,
    Embedding,
    LayerNorm,
    Projection,
    RMSNorm,
    RowShardedLinear,
    TiedProjection,
    VocabShardedEmbedding,
)


def create_nemotron_h_model(config: NemotronHConfig) -> TransformerLM:
    """Create a NemotronH language model."""
    return NemotronHFactory(config).create_model()


class NemotronHFactory:
    """Factory for building NemotronH models.

    Follows the modern fairseq2 pattern where parallelism (TP, FSDP) is
    handled within the factory rather than via external sharders:

    - ``StandardMultiheadAttention`` accepts ``gangs`` for automatic TP sharding.
    - ``VocabShardedEmbedding`` and ``ColumnShardedLinear`` are inherently
      sharding-aware.
    - MoE experts are explicitly sharded when ``gangs.tp.size > 1``.
    - Mamba2 SSM TP is deferred (requires splitting SSM heads across ranks).
    """

    def __init__(self, config: NemotronHConfig) -> None:
        self._config = config
        self._gangs = self._resolve_gangs()

    def _resolve_gangs(self) -> Gangs | None:
        """Get the current gangs if in a distributed context."""
        try:
            return get_current_gangs()
        except RuntimeError:
            return None

    @property
    def _tp_gang(self) -> Gang | None:
        """Get the TP gang, or None if TP is not active."""
        if self._gangs is None:
            return None
        if self._gangs.tp.size <= 1:
            return None
        return self._gangs.tp

    def create_model(self) -> TransformerLM:
        config = self._config

        embed = self.create_embedding()

        decoder_frontend = self.create_decoder_frontend(embed)

        decoder = self.create_decoder()

        final_proj = self.create_final_projection(embed)

        pad_idx = None

        return TransformerLM(
            config.model_dim,
            decoder_frontend,
            decoder,
            final_proj,
            pad_idx,
            config.max_seq_len,
        )

    def create_embedding(self) -> Embedding:
        config = self._config

        return VocabShardedEmbedding(config.vocab_size, config.model_dim)

    def create_decoder_frontend(self, embed: Embedding) -> TransformerFrontend:
        config = self._config

        return TransformerEmbeddingFrontend(
            config.model_dim,
            embed,
            pos_encoder=None,
            no_scale=True,
            dropout_p=config.dropout_p,
        )

    def create_decoder(self) -> TransformerLMDecoder:
        config = self._config

        layer_types = config.layer_types

        layers: list[TransformerLMDecoderLayer] = []

        for idx in range(config.num_layers):
            block_type = layer_types[idx]
            layer = self.create_decoder_layer(idx, block_type)
            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoder(layers, layer_norm)

    def create_decoder_layer(
        self,
        layer_idx: int,
        block_type: str,
    ) -> TransformerLMDecoderLayer:
        config = self._config

        norm = self.create_layer_norm()

        mixer: Module
        if block_type == "mamba":
            mixer = self.create_mamba2_mixer(layer_idx)
        elif block_type == "attention":
            mixer = self.create_self_attention()
        elif block_type == "moe":
            mixer = self.create_moe_block(layer_idx)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        return NemotronHBlock(
            block_type=block_type,  # type: ignore[arg-type]
            mixer=mixer,
            norm=norm,
            layer_idx=layer_idx,
            num_layers=config.num_layers,
        )

    def create_mamba2_mixer(self, layer_idx: int) -> NemotronHMamba2Mixer:
        config = self._config

        # TODO: Add Mamba2 TP support. This requires splitting SSM heads
        # across TP ranks and coordinating the conv1d and selective scan
        # operations. For now, each rank holds the full Mamba2 layer.
        return NemotronHMamba2Mixer(
            config.model_dim,
            num_heads=config.mamba_num_heads,
            head_dim=config.mamba_head_dim,
            state_size=config.ssm_state_size,
            n_groups=config.mamba_n_groups,
            conv_kernel=config.conv_kernel,
            chunk_size=config.chunk_size,
            time_step_min=config.time_step_min,
            time_step_max=config.time_step_max,
            use_conv_bias=config.use_conv_bias,
            proj_bias=config.mamba_proj_bias,
            eps=config.rms_norm_eps,
            layer_idx=layer_idx,
        )

    def create_self_attention(self) -> MultiheadAttention:
        config = self._config

        attn_bias = CausalAttentionBias()
        sdpa = create_default_sdpa(attn_bias)

        # NemotronH does NOT use RoPE in attention layers.
        # The Mamba2 layers handle position awareness implicitly through
        # sequential state processing, so attention layers only do
        # global context aggregation without positional encoding.
        #
        # Passing gangs= enables automatic TP sharding of Q/K/V/output_proj
        # via ColumnShardedLinear/RowShardedLinear inside StandardMultiheadAttention.
        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            head_dim=config.attn_head_dim,
            num_key_value_heads=config.num_key_value_heads,
            bias=False,  # attention_bias = False
            pos_encoder=None,  # No RoPE
            output_proj_bias=False,
            gangs=self._gangs,
        )

    def create_moe_block(self, layer_idx: int) -> NemotronHMoE:
        config = self._config

        moe = NemotronHMoE(
            config.model_dim,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            moe_intermediate_size=config.moe_intermediate_size,
            shared_expert_intermediate_size=config.shared_expert_intermediate_size,
            routed_scaling_factor=config.routed_scaling_factor,
            n_group=config.moe_n_group,
            topk_group=config.moe_topk_group,
            norm_topk_prob=config.norm_topk_prob,
            bias=False,
        )

        # Apply TP sharding to MoE if tensor parallelism is active
        tp_gang = self._tp_gang
        if tp_gang is not None:
            self._shard_moe(moe, tp_gang)

        return moe

    def _shard_moe(self, moe: NemotronHMoE, tp_gang: Gang) -> None:
        """Apply tensor parallelism to a MoE block.

        Shards each expert's intermediate dimension across TP ranks:
        - up_proj: column-sharded (split output/intermediate dim)
        - down_proj: row-sharded (split input/intermediate dim)
        Sets tp_gang for the final all-reduce in forward().
        """
        moe.tp_gang = tp_gang

        # Shard shared expert
        moe.shared_experts.up_proj = ColumnShardedLinear.from_linear(  # type: ignore[assignment]
            moe.shared_experts.up_proj,  # type: ignore[arg-type]
            tp_gang,
            gather_output=False,
        )
        moe.shared_experts.down_proj = RowShardedLinear.from_linear(  # type: ignore[assignment]
            moe.shared_experts.down_proj,  # type: ignore[arg-type]
            tp_gang,
            reduce_output=False,
        )

        # Shard each routed expert
        for expert in moe.experts:
            expert.up_proj = ColumnShardedLinear.from_linear(  # type: ignore[assignment]
                expert.up_proj,  # type: ignore[arg-type]
                tp_gang,
                gather_output=False,
            )
            expert.down_proj = RowShardedLinear.from_linear(  # type: ignore[assignment]
                expert.down_proj,  # type: ignore[arg-type]
                tp_gang,
                reduce_output=False,
            )

    def create_final_projection(self, embed: Embedding) -> Projection:
        config = self._config

        if config.tied_embeddings:
            if not isinstance(embed, VocabShardedEmbedding):
                raise TypeError(
                    f"`embed` is expected to be of type `{VocabShardedEmbedding}` "
                    f"when `config.tied_embeddings` is `True`, but is of type "
                    f"`{type(embed)}` instead."
                )
            return TiedProjection(embed.weight, bias=None)

        return ColumnShardedLinear(config.model_dim, config.vocab_size, bias=False)

    def create_layer_norm(self, dim: int | None = None) -> LayerNorm:
        config = self._config

        if dim is None:
            dim = config.model_dim

        return RMSNorm(dim, bias=False, eps=config.rms_norm_eps)
