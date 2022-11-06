import dataclasses
from typing import Optional

from fairseq2.nn.embedding import Embedding
from fairseq2.nn.positional_embedding import (
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq2.nn.projection import TiedProjection
from fairseq2.nn.transformer.decoder import (
    StandardTransformerDecoder,
    TransformerDecoder,
)
from fairseq2.nn.transformer.decoder_layer import (
    StandardTransformerDecoderLayer,
    TransformerDecoderLayer,
)
from fairseq2.nn.transformer.encoder import (
    StandardTransformerEncoder,
    TransformerEncoder,
)
from fairseq2.nn.transformer.encoder_layer import (
    StandardTransformerEncoderLayer,
    TransformerEncoderLayer,
)
from fairseq2.nn.transformer.ffn import FeedForwardNetwork, StandardFeedForwardNetwork
from fairseq2.nn.transformer.model import StandardTransformer, Transformer
from fairseq2.nn.transformer.multihead_attention import (
    MultiheadAttention,
    StandardMultiheadAttention,
)
from fairseq2.typing import DataType, Device


@dataclasses.dataclass(frozen=True)
class StandardTransformerBuilder:
    """Builds a Transformer model as described in the original paper.

    If you want to tweak the architecture, please subclass this Builder class,
    and override the method corresponding to the part of the architecture you
    want to change.
    """

    model_dim: int
    ffn_inner_dim: int
    num_attn_heads: int
    num_layers: int
    batch_first: bool = True
    dropout_p: float = 0.1
    attn_dropout_p: float = 0.1

    def build(
        self,
        vocab_size: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> Transformer:
        embed = self.build_embeddings(
            vocab_size=vocab_size,
            device=device,
            dtype=dtype,
        )
        positional_embed = self.build_positional_embeddings(
            embedding_dim=embed.embedding_dim,
            device=device,
        )
        encoder = self.build_encoder(
            embed=embed,
            positional_embed=positional_embed,
            device=device,
            dtype=dtype,
        )
        decoder = self.build_decoder(
            embed=embed,
            positional_embed=positional_embed,
            device=device,
            dtype=dtype,
        )

        # Share the weight matrix between the embedding layers and the
        # pre-softmax score projection as described in the original paper.
        score_proj = TiedProjection(embed.weight)

        return StandardTransformer(encoder, decoder, score_proj, use_log_softmax=True)

    def build_embeddings(
        self,
        vocab_size: int,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> Embedding:
        embeds = Embedding(
            num_embed=vocab_size,
            embedding_dim=self.model_dim,
            device=device,
            dtype=dtype,
        )
        return embeds

    def build_positional_embeddings(
        self, embedding_dim: int, device: Optional[Device]
    ) -> PositionalEmbedding:
        return SinusoidalPositionalEmbedding(
            max_seq_len=4096,
            embedding_dim=self.model_dim,
            device=device,
        )

    def build_encoder(
        self,
        embed: Embedding,
        positional_embed: PositionalEmbedding,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> TransformerEncoder:
        return StandardTransformerEncoder(
            embed,
            positional_embed,
            [
                self.build_encoder_layer(i, device=device, dtype=dtype)
                for i in range(self.num_layers)
            ],
            embed_dropout_p=self.dropout_p,
            device=device,
            dtype=dtype,
        )

    def build_encoder_layer(
        self,
        layer: int,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> TransformerEncoderLayer:
        self_attn = self.build_attn(device=device, dtype=dtype)
        ffn = StandardFeedForwardNetwork(
            model_dim=self.model_dim,
            inner_dim=self.ffn_inner_dim,
            device=device,
            dtype=dtype,
        )
        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.dropout_p,
            device=device,
            dtype=dtype,
        )

    def build_ffn(
        self,
        layer: int,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> FeedForwardNetwork:
        return StandardFeedForwardNetwork(
            model_dim=self.model_dim,
            inner_dim=self.ffn_inner_dim,
            device=device,
            dtype=dtype,
        )

    def build_attn(
        self,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> MultiheadAttention:
        assert (
            self.model_dim % self.num_attn_heads == 0
        ), f"Can't divide model_dim ({self.model_dim}) with num_attn_heads ({self.num_attn_heads})!"

        mha = StandardMultiheadAttention(
            model_dim=self.model_dim,
            num_heads=self.num_attn_heads,
            attn_dropout_p=self.attn_dropout_p,
            batch_first=self.batch_first,
            device=device,
            dtype=dtype,
        )
        mha.to(device)
        return mha

    def build_decoder(
        self,
        embed: Embedding,
        positional_embed: PositionalEmbedding,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> TransformerDecoder:
        return StandardTransformerDecoder(
            embed,
            positional_embed,
            [
                self.build_decoder_layer(i, device=device, dtype=dtype)
                for i in range(self.num_layers)
            ],
            embed_dropout_p=self.dropout_p,
            device=device,
            dtype=dtype,
        )

    def build_decoder_layer(
        self,
        layer: int,
        device: Optional[Device],
        dtype: Optional[DataType],
    ) -> TransformerDecoderLayer:
        # Teaser: the next example will mix MoE and distributed decoder layers for
        # demonstration purposes (e.g. ShardedFeedForwardNetwork)

        self_attn = self.build_attn(device=device, dtype=dtype)
        enc_dec_attn = self.build_attn(device=device, dtype=dtype)

        ffn = self.build_ffn(layer, device=device, dtype=dtype)

        return StandardTransformerDecoderLayer(
            self_attn,
            enc_dec_attn,
            ffn,
            dropout_p=self.dropout_p,
            device=device,
            dtype=dtype,
        )
