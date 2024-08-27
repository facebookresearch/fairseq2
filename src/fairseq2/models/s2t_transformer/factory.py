# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from torch.nn import SiLU

from fairseq2.config_registry import ConfigRegistry
from fairseq2.data import VocabularyInfo
from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.factory import model_factories
from fairseq2.models.s2t_transformer.feature_extractor import Conv1dFbankSubsampler
from fairseq2.models.s2t_transformer.frontend import S2TTransformerFrontend
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    TransformerModel,
    init_final_projection,
)
from fairseq2.nn import (
    Linear,
    PositionEncoder,
    SinusoidalPositionEncoder,
    StandardEmbedding,
    init_scaled_embedding,
)
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    RelativePositionalEncoding,
    RelativePositionSDPA,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device

S2T_TRANSFORMER_FAMILY: Final = "s2t_transformer"


@dataclass(kw_only=True)
class S2TTransformerConfig:
    """Holds the configuration of an S2T Transformer model.

    The default values correspond to the medium architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.
    """

    model_dim: int = 512
    """The dimensionality of the model."""

    max_source_seq_len: int = 1024
    """The maximum source sequence length after feature extraction."""

    num_fbank_channels: int = 80
    """The number of source log-mel filterbank channels."""

    max_target_seq_len: int = 1024
    """The maximum target sequence length."""

    target_vocab_info: VocabularyInfo = field(
        default_factory=lambda: VocabularyInfo(
            size=10000, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        )
    )
    """The target vocabulary information."""

    use_relative_pos: bool = False
    """If ``True``, uses relative positional encodings for source sequences."""

    use_conformer: bool = False
    """If ``True``, uses Conformer blocks instead of encoder layers."""

    num_encoder_layers: int = 12
    """The number of encoder layers."""

    num_decoder_layers: int = 6
    """The number of decoder layers."""

    num_encoder_attn_heads: int = 8
    """The number of attention heads in encoder layers."""

    num_decoder_attn_heads: int = 8
    """The number of attention heads in decoder layers."""

    ffn_inner_dim: int = 512 * 4
    """The dimensionality of inner projection layers in feed-forward networks."""

    dropout_p: float = 0.15
    """The dropout probability on outputs of Transformer layers."""

    depthwise_conv_kernel_size: int = 0
    """The kernel size of depthwise convolutions in Conformer blocks."""


s2t_transformer_archs = ConfigRegistry[S2TTransformerConfig]()

s2t_transformer_arch = s2t_transformer_archs.decorator


class S2TTransformerBuilder:
    """Builds modules of an S2T Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1911.08460`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: S2TTransformerConfig
    _device: Device | None
    _dtype: DataType | None
    _rel_pos_encoding: RelativePositionalEncoding | None

    def __init__(
        self,
        config: S2TTransformerConfig,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self._config = config

        self._device, self._dtype = device, dtype

        self._rel_pos_encoding = None

    def build_model(self) -> TransformerModel:
        """Build a model."""
        encoder_frontend = self.build_encoder_frontend()
        encoder = self.build_encoder()

        decoder_frontend = self.build_decoder_frontend()
        decoder = self.build_decoder()

        final_proj = Linear(
            self._config.model_dim,
            self._config.target_vocab_info.size,
            bias=False,
            init_fn=init_final_projection,
            device=self._device,
            dtype=self._dtype,
        )

        model = TransformerModel(
            encoder_frontend,
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self._config.max_target_seq_len,
            self._config.target_vocab_info,
        )

        model.set_family(S2T_TRANSFORMER_FAMILY)

        return model

    def build_encoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer encoder front-end."""
        feat_extractor = Conv1dFbankSubsampler(
            num_channels=self._config.num_fbank_channels,
            inner_dim=1024,
            feature_dim=self._config.model_dim,
            kernel_sizes=[5, 5],
            device=self._device,
            dtype=self._dtype,
        )

        pos_encoder = self.build_source_position_encoder()

        return S2TTransformerFrontend(
            self._config.model_dim,
            feat_extractor,
            pos_encoder,
            proj=self._config.use_conformer,
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        embed = StandardEmbedding(
            num_embeddings=self._config.target_vocab_info.size,
            embedding_dim=self._config.model_dim,
            pad_idx=self._config.target_vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
            device=self._device,
            dtype=self._dtype,
        )

        pos_encoder = self.build_target_position_encoder()

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_source_position_encoder(self) -> PositionEncoder | None:
        """Build a position encoder for source sequences."""
        if self._config.use_relative_pos:
            return None

        return SinusoidalPositionEncoder(
            self._config.model_dim,
            self._config.max_source_seq_len,
            device=self._device,
        )

    def build_target_position_encoder(self) -> PositionEncoder:
        """Build a position encoder for target sequences."""
        return SinusoidalPositionEncoder(
            self._config.model_dim,
            self._config.max_target_seq_len,
            _legacy_pad_idx=1,
            device=self._device,
        )

    def build_encoder(self) -> TransformerEncoder:
        """Build a Transformer encoder."""
        num_layers = self._config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        if self._config.use_conformer:
            encoder_norm_order = TransformerNormOrder.POST
        else:
            encoder_norm_order = TransformerNormOrder.PRE

        return StandardTransformerEncoder(
            layers,
            norm_order=encoder_norm_order,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self._config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self._device,
            dtype=self._dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        if self._config.use_conformer:
            return self.build_conformer_block()

        self_attn = self.build_encoder_attention()

        ffn = self.build_ffn()

        if self._config.use_conformer:
            encoder_norm_order = TransformerNormOrder.POST
        else:
            encoder_norm_order = TransformerNormOrder.PRE

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self._config.dropout_p,
            norm_order=encoder_norm_order,
            device=self._device,
            dtype=self._dtype,
        )

    def build_conformer_block(self) -> TransformerEncoderLayer:
        """Build a Conformer block."""
        ffn1 = self.build_ffn(use_swish=True)

        self_attn = self.build_encoder_attention()

        conv = ConformerConvolution(
            self._config.model_dim,
            self._config.depthwise_conv_kernel_size,
            device=self._device,
            dtype=self._dtype,
        )

        ffn2 = self.build_ffn(use_swish=True)

        return ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_decoder_attention()

        encoder_decoder_attn = self.build_decoder_attention()

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=self._config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self._device,
            dtype=self._dtype,
        )

    def build_encoder_attention(self) -> MultiheadAttention:
        """Build a Transformer encoder multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self._config.dropout_p)

        if self._config.use_relative_pos:
            if self._rel_pos_encoding is None:
                self._rel_pos_encoding = RelativePositionalEncoding(
                    self._config.model_dim,
                    self._config.max_source_seq_len,
                    device=self._device,
                )

            sdpa = RelativePositionSDPA(
                self._config.model_dim,
                self._config.num_encoder_attn_heads,
                self._rel_pos_encoding,
                inner_sdpa=sdpa,
                device=self._device,
                dtype=self._dtype,
            )

        return StandardMultiheadAttention(
            self._config.model_dim,
            self._config.num_encoder_attn_heads,
            sdpa=sdpa,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder_attention(self) -> MultiheadAttention:
        """Build a Transformer decoder multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self._config.dropout_p)

        return StandardMultiheadAttention(
            self._config.model_dim,
            self._config.num_decoder_attn_heads,
            sdpa=sdpa,
            device=self._device,
            dtype=self._dtype,
        )

    def build_ffn(self, use_swish: bool = False) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self._config.model_dim,
            self._config.ffn_inner_dim,
            bias=True,
            inner_activation=SiLU() if use_swish else None,
            inner_dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )


def create_s2t_transformer_model(
    config: S2TTransformerConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> TransformerModel:
    """Create an S2T Transformer model."""
    return S2TTransformerBuilder(config, device=device, dtype=dtype).build_model()


model_factories.register(
    S2T_TRANSFORMER_FAMILY,
    create_s2t_transformer_model,
    S2TTransformerConfig,
    s2t_transformer_archs,
)
