# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from torch import Tensor
import torch
from fairseq2.data import VocabularyInfo
from fairseq2.gang import Gang
from fairseq2.models.decoder import DecoderModel
from fairseq2.models.sequence import SequenceModelOutput, SpeechTextReprOutput
from fairseq2.models.transformer.frontend import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.nn import (
    ColumnShardedLinear,
    Linear,
    Projection,
    RowShardedLinear,
    ShardedEmbedding,
    StandardEmbedding,
    VocabShardedEmbedding,
)
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import (
    GLUFeedForwardNetwork,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerDecoder,
)
from fairseq2.typing import override
from fairseq2.datasets.speech_text import SpeechTextAlignBatch
from fairseq2.models.sequence import SequenceBatch


@final
class SpeechTransformerDecoderModel(DecoderModel):
    """Represents a Transformer-based decoder model."""

    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        max_seq_len: int,
        vocab_info: VocabularyInfo,
        speech_encoder = None,
        speech_dim_adapter = None,
        speech_decoder = None,
    ) -> None:
        """
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to decoder outputs.
        :param max_seq_len:
            The maximum length of sequences produced by the model.
        :param vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__(decoder.model_dim, max_seq_len, vocab_info)
        self.decoder_frontend = decoder_frontend # from llama3
        self.decoder = decoder # from llama3
        self.speech_encoder = speech_encoder # is frozen
        self.speech_dim_adapter = speech_dim_adapter
        self.speech_decoder = speech_decoder
        self.final_proj = final_proj # from llama3



    @staticmethod
    def prepare_blockwise_attention_mask(audio_repr, audio_seq_lens, token_lens, boundary_index):
        repr_lens = torch.floor(audio_seq_lens / 640).long().to(audio_repr.device)
        if repr_lens.max().item() > audio_repr.shape[1]:
            repr_lens[repr_lens == repr_lens.max().item()] = audio_repr.shape[1]
            # print("Audio shorter than its precomputed repr len, reshape repr len")
            # print(repr_lens, audio_repr.shape[1])
        audio_repr_padding_mask = PaddingMask(repr_lens, batch_seq_len=audio_repr.shape[1])

        max_seq_len = audio_repr.shape[1]
        batch_size = repr_lens.shape[0]
        attention_map = audio_repr.new_full((max_seq_len, max_seq_len), 1).long()
        attention_map.tril_(diagonal=0)
        # bz x max_seq_len x max_seq_len
        # Note that we have to use repeat rather then expand to assign memory!!
        attention_map = attention_map.unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(batch_size):
            # print(boundary_index[i, :])
            cur_repr_len = repr_lens[i]
            if cur_repr_len < max_seq_len:
                # this is actually not necessary since the padding mask inside self-attention will achieve the same thing
                attention_map[i, :, cur_repr_len:] = 0
            
            cur_token_len = token_lens[i]
            if boundary_index[i, cur_token_len-1] >= cur_repr_len - 1:
                boundary_index[i, cur_token_len-1] = cur_repr_len - 1
            # create blockwise masks
            # TODO: optimize this code. this double for-loop is slow especially if number of boundary index is large
            for boundary_id in range(cur_token_len):
                boundary_column = boundary_index[i, boundary_id]
                if boundary_column + 1 < max_seq_len:
                    # we could use :boundary_column + 1 as well, but I want it to be able to 
                    # attend to 1 more repr before in case of bad alignment
                    # we can also adjust this window size to allow for more lenient window
                    attention_map[i, boundary_column+1:, :boundary_column] = 0
        return attention_map, audio_repr_padding_mask, boundary_index

 
    @override
    def forward(self, audio: SequenceBatch, text_token: SequenceBatch, boundary_index: Tensor) -> SpeechTextReprOutput:      
        # bz x repr_len x 768
        audio_repr = self.speech_encoder(audio).detach() # ensure that no gradient flows back to dinosr encoder
        audio_repr = self.speech_dim_adapter(audio_repr)
        blockwise_attn_mask, audio_repr_padding_mask, boundary_index = self.prepare_blockwise_attention_mask(
            audio_repr, audio.example["seq_lens"], text_token.example["seq_lens"], boundary_index)

        audio_repr, audio_repr_padding_mask = self.speech_decoder(
            audio_repr, audio_repr_padding_mask, blockwise_attn_mask=blockwise_attn_mask)

        selected_repr = audio_repr[torch.arange(audio_repr.shape[0]).to(audio_repr.device).unsqueeze(1), boundary_index, :]
        # compute loss against dtext_embedding
        text_seqs, text_mask = text_token.seqs, text_token.padding_mask
        text_embed, _= self.decoder_frontend(text_seqs, text_mask, return_raw_embed=True)
        return SpeechTextReprOutput(speech_repr=selected_repr, text_repr=text_embed, mask=text_mask)

    @override
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, PaddingMask]:
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        decoder_output, decoder_padding_mask = self.decoder(
            seqs, padding_mask, state_bag=state_bag
        )

        return decoder_output, decoder_padding_mask

    @override
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.vocab_info.pad_idx)


# mypy: disable-error-code="arg-type"


def shard_transformer_decoder_model(
    model: SpeechTransformerDecoderModel, gang: Gang, shard_embed_dim: bool = False
) -> None:
    """Shard ``model`` over ``gang`` for tensor parallelism.

    :param model:
        The model to shard.
    :param gang:
        The gang used for tensor parallelism.
    :param shard_embed_dim:
        If ``True``, shards :class:`StandardEmbedding` instances over the
        embedding dimension; otherwise, over the vocabulary dimension.
    """
    if gang.size == 1:
        return

    def shard_embed_frontend(m: TransformerEmbeddingFrontend) -> None:
        if not isinstance(m.embed, StandardEmbedding):
            return

        if shard_embed_dim:
            m.embed = ShardedEmbedding.from_embedding(m.embed, gang)
        else:
            m.embed = VocabShardedEmbedding.from_embedding(m.embed, gang)

    def shard_mha(m: StandardMultiheadAttention) -> None:
        for proj in (m.q_proj, m.k_proj, m.v_proj, m.output_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.q_proj = ColumnShardedLinear.from_linear(m.q_proj, gang, gather_output=False)
        m.k_proj = ColumnShardedLinear.from_linear(m.k_proj, gang, gather_output=False)
        m.v_proj = ColumnShardedLinear.from_linear(m.v_proj, gang, gather_output=False)

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, gang, scatter_input=False
        )

        m.num_heads = m.num_heads // gang.size
        m.num_key_value_heads = m.num_key_value_heads // gang.size

    def shard_ffn(m: StandardFeedForwardNetwork) -> None:
        for proj in (m.inner_proj, m.outer_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.inner_proj = ColumnShardedLinear.from_linear(
            m.inner_proj, gang, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, gang, scatter_input=False
        )

    def shard_glu_ffn(m: GLUFeedForwardNetwork) -> None:
        for proj in (m.gate_proj, m.inner_proj, m.output_proj):
            if not isinstance(proj, Linear):
                return

        # Scatter.
        m.gate_proj = ColumnShardedLinear.from_linear(
            m.gate_proj, gang, gather_output=False
        )

        m.inner_proj = ColumnShardedLinear.from_linear(
            m.inner_proj, gang, gather_output=False
        )

        # Gather.
        m.output_proj = RowShardedLinear.from_linear(
            m.output_proj, gang, scatter_input=False
        )

    for m in model.modules():
        if isinstance(m, TransformerEmbeddingFrontend):
            shard_embed_frontend(m)

            continue

        if isinstance(m, StandardMultiheadAttention):
            shard_mha(m)

            continue

        if isinstance(m, StandardFeedForwardNetwork):
            shard_ffn(m)

            continue

        if isinstance(m, GLUFeedForwardNetwork):
            shard_glu_ffn(m)

            continue

    if isinstance(model.final_proj, Linear):
        model.final_proj = ColumnShardedLinear.from_linear(model.final_proj, gang)
