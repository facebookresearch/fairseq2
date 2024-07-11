# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Final, Optional

from fairseq2.data import VocabularyInfo
from fairseq2.models.transformer import (
    TransformerDecoderModel,
    TransformerEmbeddingFrontend,
    TransformerFrontend,
    init_final_projection,
    SpeechTransformerDecoderModel
)

from fairseq2.nn import LayerNorm, Linear, RMSNorm, RotaryEncoder, StandardEmbedding
from fairseq2.nn.lora import LoRAConfig
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)

from fairseq2.models.decoder import DecoderModel
import torch
import torch.nn.functional as F
# from fairseq2.nn.transformer.attention_mask import BlockwiseCausalAttentionMaskFactory
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import DataType, Device
from fairseq2.logging import get_log_writer
from seamless_communication.models.unit_extractor.unit_extractor import UnitExtractor
log = get_log_writer(__name__)
LLAMA_FAMILY: Final = "llama"


@dataclass
class LLaMAConfig:
    """Holds the configuration of a LLaMA model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2302.13971`.
    """

    model_dim: int = 4096
    """The dimensionality of the model."""

    max_seq_len: int = 2048
    """The maximum sequence length."""

    vocab_info: VocabularyInfo = field(
        default_factory=lambda: VocabularyInfo(
            size=32000, unk_idx=0, bos_idx=1, eos_idx=2, pad_idx=None
        )
    )
    """The vocabulary information."""

    num_layers: int = 32
    """The number of decoder layers."""

    num_attn_heads: int = 32
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 32
    """The number of key/value heads for Grouped Query Attention."""

    ffn_inner_dim: int = 4096 * 4
    """The dimensionality of inner projection layers in feed-forward networks."""

    ffn_inner_dim_scale: float = 2 / 3
    """The scale factor for the dimensionality of inner projection layers in
    feed forward networks."""

    ffn_inner_dim_to_multiple: int = 256
    """The dimensionality of inner projection layers in feed-forward networks is
    rounded up to the nearest multiple of this value."""

    rope_theta: float = 10_000.0
    """The coefficient of the long-term decay of the Rotary position encoder."""

    dropout_p: float = 0.1
    """The dropout probability on outputs of Transformer layers."""

    use_speech_decoder: bool = False
    speech_decoder_layers: int = 4
    freeze_text_llama: bool = False
    """The number of layer for the causal decoder for speech representation"""


class SemanticModel:
    def __init__(self, dinosr_config, dinosr_km_path, device, output_layer_idx=6):
        self.unit_extractor = UnitExtractor(
            dinosr_config, dinosr_km_path, device=device
        )
        for param in self.unit_extractor.parameters():
            param.requires_grad = False
        self.output_layer_idx = output_layer_idx
        self.device = device

    def __call__(self, x):
        features = self.unit_extractor.model(x, self.output_layer_idx).squeeze(0)
        return features
        # print(features.shape)
        # print(features[0, -1, :])
        # print(features[0, 0, :])
        # exit(0)
        # print(x.shape)
        # x = x.transpose(0, 1) # bz x seqlen -> seqlen, bz
        # print(x.shape)
        # decoded_audio = {
        #     "waveform": x.to(dtype=torch.float32),
        #     "sample_rate": 16000,
        #     "format": -1,
        # }
        # src = self.unit_extractor.collate(decoded_audio)["waveform"]
        # print(src)
        # exit(0)
        # seqs, padding_mask = get_seqs_and_padding_mask(src)
        # seqs = seqs.view(1, -1)
        # seqs = F.layer_norm(seqs, seqs.shape)
        # batch = SequenceBatch(seqs=seqs, padding_mask=padding_mask)
        # # print(features.shape, flush=True)
        # return features
    
        


class LLaMABuilder:
    """Builds modules of a LLaMA model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2302.13971` and
    :cite:t:`https://doi.org/10.48550/arXiv.2307.09288`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: LLaMAConfig
    _device: Optional[Device]
    _dtype: Optional[DataType]
    _pos_encoder: Optional[RotaryEncoder]

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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

        self._pos_encoder = None

    @staticmethod
    def _freeze_module(module):
        for params in module.parameters():
            params.requires_grad = False

    def build_model(self) -> DecoderModel:
        """Build a model."""
        log.info("Building Llama Decoders")
        decoder_frontend = self.build_decoder_frontend()
        decoder = self.build_decoder()
        final_proj = Linear(
            self._config.model_dim,
            self._config.vocab_info.size,
            bias=False,
            init_fn=init_final_projection,
            device=self._device,
            dtype=self._dtype,
        )

        if self._config.use_speech_decoder:
            log.info('Building Speech Decoder')
            semantic_model_km_path = '/fsx-ust/dnn/models/kmeans_dinosr.np'
            device = torch.device("cuda") 
            speech_encoder = SemanticModel("dinosr_base", semantic_model_km_path, device)
            # map encoded speech (hard coded as 768) to model dimension
            dim_adapter = Linear(768, self._config.model_dim, 
                                 bias=False, 
                                 init_fn=init_final_projection,
                                 device=self._device,
                                 dtype=self._dtype)
            # print(dim_adapter.weight.data)
            speech_decoder = self.build_speech_decoder()
            if self._config.freeze_text_llama:
                log.info("Freezing Text Llama Parameters!")
                self._freeze_module(decoder_frontend)
                self._freeze_module(decoder)
                self._freeze_module(final_proj)
                
            return SpeechTransformerDecoderModel(
                decoder_frontend,
                decoder,
                final_proj,
                self._config.max_seq_len,
                self._config.vocab_info,
                speech_encoder,
                dim_adapter,
                speech_decoder
            )
        else:
            # normal Llama Model architecture
            return TransformerDecoderModel(
                decoder_frontend,
                decoder,
                final_proj,
                self._config.max_seq_len,
                self._config.vocab_info,
            )

    def build_decoder_frontend(self) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        embed = StandardEmbedding(
            num_embeddings=self._config.vocab_info.size,
            embedding_dim=self._config.model_dim,
            device=self._device,
            dtype=self._dtype,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder=None,
            no_scale=True,  # LLaMA does not use embedding scaling.
            dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_speech_decoder(self) -> TransformerDecoder:
        """Build a Transformer Speech decoder (shallow layers)."""
        num_layers = self._config.speech_decoder_layers
        layers = [self.build_decoder_layer(use_pos_encoder=False) for _ in range(num_layers)]
        return StandardTransformerDecoder(
            layers,
            dropout_p=self._config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype
        )
    
    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self._config.num_layers
        layers = [self.build_decoder_layer() for _ in range(num_layers)]
        return StandardTransformerDecoder(
            layers,
            dropout_p=self._config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_decoder_layer(self, use_pos_encoder=True) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(
            self._config.num_attn_heads, self._config.num_key_value_heads,
            use_pos_encoder=use_pos_encoder
        )

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn=None,
            ffn=ffn,
            norm_order=TransformerNormOrder.PRE,
            layer_norm_factory=self.build_layer_norm,
            device=self._device,
            dtype=self._dtype,
        )

    def build_attention(
        self, num_heads: int, num_key_value_heads: int, use_pos_encoder=True
    ) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self._config.dropout_p)

        if self._pos_encoder is None:
            self._pos_encoder = RotaryEncoder(
                self._config.model_dim // num_heads,
                self._config.max_seq_len,
                theta=self._config.rope_theta,
                device=self._device,
            )

        return StandardMultiheadAttention(
            self._config.model_dim,
            num_heads,
            num_key_value_heads=num_key_value_heads,
            sdpa=sdpa,
            pos_encoder=self._pos_encoder if use_pos_encoder else None,
            bias=False,
            device=self._device,
            dtype=self._dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return GLUFeedForwardNetwork(
            self._config.model_dim,
            self._config.ffn_inner_dim,
            bias=False,
            inner_dim_scale=self._config.ffn_inner_dim_scale,
            inner_dim_to_multiple=self._config.ffn_inner_dim_to_multiple,
            inner_dropout_p=self._config.dropout_p,
            device=self._device,
            dtype=self._dtype,
        )

    def build_layer_norm(
        self,
        model_dim: int,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> LayerNorm:
        """Build a Layer Normalization module."""
        return RMSNorm(model_dim, bias=False, device=device, dtype=dtype)


def create_llama_model(
    config: LLaMAConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> DecoderModel:
    """Create a LLaMA model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    model = LLaMABuilder(config, device=device, dtype=dtype).build_model()
    return model.set_family(LLAMA_FAMILY)


def get_llama_lora_config() -> LoRAConfig:
    return LoRAConfig(
        r=8,
        alpha=16.0,
        dropout_p=0.05,
        keys=[".*decoder.layers.*.self_attn.*(q_proj|v_proj)$"],
    )
