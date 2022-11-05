import dataclasses

import torch

import fairseq2.nn
from fairseq2.nn import transformer


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
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "transformer.Transformer":
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

        # Share the weight matrix between the embedding layers and the pre-softmax
        # score projection as described in the original paper.
        score_proj = fairseq2.nn.TiedProjection(embed.weight)
        return transformer.StandardTransformer(
            encoder=encoder,
            decoder=decoder,
            score_proj=score_proj,
            use_log_softmax=True,
        )

    def build_embeddings(
        self,
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> fairseq2.nn.Embedding:
        embs = fairseq2.nn.Embedding(
            num_embed=vocab_size,
            embedding_dim=self.model_dim,
            device=device,
            dtype=dtype,
        )
        return embs

    def build_positional_embeddings(
        self,
        embedding_dim: int,
        device: torch.device,
    ) -> fairseq2.nn.PositionalEmbedding:
        return fairseq2.nn.SinusoidalPositionalEmbedding(
            max_seq_len=4096,
            embedding_dim=self.model_dim,
            device=device,
        )

    def build_encoder(
        self,
        embed: fairseq2.nn.Embedding,
        positional_embed: fairseq2.nn.PositionalEmbedding,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "transformer.TransformerEncoder":
        return transformer.StandardTransformerEncoder(
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
        device: torch.device,
        dtype: torch.dtype,
    ) -> "transformer.TransformerEncoderLayer":
        self_attn = self.build_attn(device=device, dtype=dtype)
        ffn = transformer.StandardFeedForwardNetwork(
            model_dim=self.model_dim,
            inner_dim=self.ffn_inner_dim,
            device=device,
            dtype=dtype,
        )
        return transformer.StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.dropout_p,
            device=device,
            dtype=dtype,
        )

    def build_ffn(
        self,
        layer: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "transformer.FeedForwardNetwork":
        return transformer.StandardFeedForwardNetwork(
            model_dim=self.model_dim,
            inner_dim=self.ffn_inner_dim,
            device=device,
            dtype=dtype,
        )

    def build_attn(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "transformer.MultiheadAttention":
        assert (
            self.model_dim % self.num_attn_heads == 0
        ), f"Can't divide model_dim ({self.model_dim}) with num_attn_heads ({self.num_attn_heads})!"

        mha = transformer.StandardMultiheadAttention(
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
        embed: fairseq2.nn.Embedding,
        positional_embed: fairseq2.nn.PositionalEmbedding,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "transformer.TransformerDecoder":
        return transformer.StandardTransformerDecoder(
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
        device: torch.device,
        dtype: torch.dtype,
    ) -> "transformer.TransformerDecoderLayer":
        # Teaser: the next example will mix MoE and distributed decoder layers for
        # demonstration purposes (e.g. ShardedFeedForwardNetwork)

        self_attn = self.build_attn(device=device, dtype=dtype)
        enc_dec_attn = self.build_attn(device=device, dtype=dtype)

        ffn = self.build_ffn(layer, device=device, dtype=dtype)

        return transformer.StandardTransformerDecoderLayer(
            self_attn,
            enc_dec_attn,
            ffn,
            dropout_p=self.dropout_p,
            device=device,
            dtype=dtype,
        )
