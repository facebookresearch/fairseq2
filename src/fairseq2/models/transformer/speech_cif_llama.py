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
from fairseq2.models.sequence import SequenceModelOutput, SpeechTextReprOutput, SpeechTextPPLOutput
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
from fairseq2.models.transformer.cif import cif_function
import pdb

@final
class SpeechCIFTransformerDecoderModel(DecoderModel):
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
        cif_scorer = None,
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
        self.cif_scorer = cif_scorer
        self.final_proj = final_proj # from llama3


 
    @override
    def forward(self, audio: SequenceBatch, text_token: SequenceBatch, boundary_index: Tensor) -> SpeechTextReprOutput:
        audio_repr = self.speech_encoder(audio).detach() # ensure that no gradient flows back to dinosr encoder
        audio_repr = audio_repr.to(torch.bfloat16)  

        # compute scores for CIF aggregation
        repr_for_score = audio_repr
        for scorer in self.cif_scorer:
            repr_for_score = scorer(repr_for_score)
        alphas = torch.sigmoid(repr_for_score).squeeze(-1)

        audio_repr = self.speech_dim_adapter(audio_repr)
        audio_seq_lens =  audio.example["seq_lens"]
        repr_lens = torch.floor(audio_seq_lens / 640).long().to(audio_repr.device)
        if repr_lens.max().item() > audio_repr.shape[1]:
            repr_lens[repr_lens == repr_lens.max().item()] = audio_repr.shape[1]
        audio_repr_padding_mask = PaddingMask(repr_lens, batch_seq_len=audio_repr.shape[1])
        audio_repr, audio_repr_padding_mask = self.speech_decoder(
            audio_repr, audio_repr_padding_mask)

        # aggregate audio_repr with scores
        text_lens = text_token.example["seq_lens"].to(audio_repr.device)
        padding_mask_for_cif = ~audio_repr_padding_mask.materialize()
        cif_out = cif_function(
            inputs=audio_repr,
            alpha=alphas,
            target_lengths=text_lens,
            padding_mask=padding_mask_for_cif,
            beta=1.0
        )
        agg_audio_repr = cif_out["cif_out"][0]
        alphas = alphas.masked_fill(padding_mask_for_cif, 0.0)

        quantity_loss = (alphas.sum(dim=1) - text_lens) ** 2
        quantity_loss = quantity_loss.mean()

        text_seqs, text_mask = text_token.seqs, text_token.padding_mask
        text_embed, _= self.decoder_frontend(text_seqs, text_mask, return_raw_embed=True)
        return SpeechTextReprOutput(
            speech_repr=agg_audio_repr, 
            text_repr=text_embed, 
            mask=text_mask,
            quantity_loss=quantity_loss
        )
    

    def forward_nll(
            self, 
            audio: SequenceBatch, 
            text_token: SequenceBatch,
            boundary_index: Tensor,
            input_seqs: Tensor,
            output_seqs: Tensor,
            target_mask: Tensor,
            padding_mask: PaddingMask,
        ) -> SpeechTextPPLOutput:
        audio_repr = self.speech_encoder(audio).detach()
        audio_repr = audio_repr.to(torch.bfloat16)  
        audio_repr = self.speech_dim_adapter(audio_repr)
        
        blockwise_attn_mask, audio_repr_padding_mask, boundary_index = self.prepare_blockwise_attention_mask(
            audio_repr, audio.example["seq_lens"], text_token.example["seq_lens"], boundary_index)
        audio_repr, audio_repr_padding_mask = self.speech_decoder(
            audio_repr, audio_repr_padding_mask, blockwise_attn_mask=blockwise_attn_mask)

        selected_repr = audio_repr[torch.arange(audio_repr.shape[0]).to(audio_repr.device).unsqueeze(1), boundary_index, :]

        text_embed, _= self.decoder_frontend(input_seqs, None, return_raw_embed=True)
        swapped_embed = text_embed.clone()
        swapped_embed[:, 1:] = selected_repr[:, :-1]
        swapped_embed, _ = self.decoder_frontend.forward_with_embed(swapped_embed, None)
        decoder_output, decoder_padding_mask = self.decoder(swapped_embed, padding_mask)
     
        logits = self.final_proj(decoder_output)
        return SpeechTextPPLOutput(
            target_tokens=output_seqs,
            target_mask=target_mask,
            logits=logits
        )
    
    def forward_nll_mmlu(
            self,
            audio: SequenceBatch,
            boundary_index: Tensor,
            text_seq_lens: Tensor,
            input_tokens: Tensor,
            output_tokens: Tensor,
            speech_positions: Optional[Tensor] = None,
    ):
        
        audio_repr = self.speech_encoder(audio).detach()
        audio_repr = audio_repr.to(torch.bfloat16)  
        audio_repr = self.speech_dim_adapter(audio_repr)

        blockwise_attn_mask, audio_repr_padding_mask, boundary_index = self.prepare_blockwise_attention_mask(
            audio_repr, audio.example["seq_lens"], text_seq_lens, boundary_index)
        
        audio_repr, audio_repr_padding_mask = self.speech_decoder(
            audio_repr, audio_repr_padding_mask, blockwise_attn_mask=blockwise_attn_mask)

        batch_arange = torch.arange(audio_repr.shape[0]).to(audio_repr.device).unsqueeze(1)
        selected_repr = audio_repr[batch_arange, boundary_index, :]
        text_embed, _= self.decoder_frontend(input_tokens, None, return_raw_embed=True)

        swapped_embed = text_embed.clone()
        prompt_len = text_seq_lens[0]
        if speech_positions is None:
            # no demonstrations
            swapped_embed[:, 1:1+prompt_len] = selected_repr
        else:
            swapped_embed[batch_arange, speech_positions, :] = selected_repr

        swapped_embed, _ = self.decoder_frontend.forward_with_embed(swapped_embed, None)
        decoder_output, _ = self.decoder(swapped_embed, None)
     
        logits = self.final_proj(decoder_output)
        target_mask = output_tokens != -100
        return SpeechTextPPLOutput(
            target_tokens=output_tokens,
            target_mask=target_mask,
            logits=logits
        )
    

    def forward_text_nll(
            self, 
            input_seqs: Tensor,
            output_seqs: Tensor,
            target_mask: Tensor,
            padding_mask: PaddingMask,
        ) -> SpeechTextPPLOutput:
        text_embed, _= self.decoder_frontend(input_seqs, padding_mask)
        decoder_output, _ = self.decoder(text_embed, padding_mask)
        logits = self.final_proj(decoder_output)
        return SpeechTextPPLOutput(
            target_tokens=output_seqs,
            target_mask=target_mask,
            logits=logits
        )
    

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
    model: SpeechCIFTransformerDecoderModel, gang: Gang, shard_embed_dim: bool = False
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
