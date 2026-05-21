# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from fairseq2.runtime.config_registry import ConfigRegistrar
from fairseq2.runtime.dependency import DependencyContainer

QWEN_FAMILY: Final = "qwen"
QWEN35_FAMILY: Final = "qwen3_5"
QWEN36_FAMILY: Final = "qwen3_6"


@dataclass(kw_only=True)
class QwenConfig:
    model_dim: int = 3584
    """The dimensionality of the model."""

    max_seq_len: int = 32_768
    """The maximum sequence length."""

    vocab_size: int = 152_064
    """The size of the vocabulary."""

    tied_embeddings: bool = False
    """If ``True``, ties the embedding table and the output projection layer."""

    num_layers: int = 28
    """The number of decoder layers."""

    num_attn_heads: int = 28
    """The number of attention heads in decoder layers."""

    num_key_value_heads: int = 4
    """The number of key/value heads for Grouped Query Attention."""

    head_dim: int | None = None
    """
    The dimensionality of attention heads. If ``None``, uses the standard
    formula ``model_dim // num_attn_heads``.
    """

    qkv_proj_bias: bool = True
    """If ``True``, query, key, and value projections learn an additive bias."""

    q_norm: bool = False
    """If ``True``, applies Layer Normalization to projected attention queries."""

    k_norm: bool = False
    """If ``True``, applies Layer Normalization to projected attention keys."""

    ffn_inner_dim: int = 18_944
    """The dimensionality of inner projection layers in feed-forward networks."""

    rope_theta: float = 1_000_000.0
    """The coefficient of the long-term decay of the Rotary position encoder."""

    dropout_p: float = 0.0
    """The dropout probability on outputs of Transformer layers."""

    pad_idx: int | None = None
    """The index of the pad symbol in the vocabulary."""


# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Qwen35Config:
    """Holds the configuration of a Qwen 3.5 dense model."""

    model_dim: int = 4096
    max_seq_len: int = 32_768
    vocab_size: int = 248_320
    tied_embeddings: bool = False
    num_layers: int = 32
    num_attn_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    ffn_inner_dim: int = 12_288
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1_000_000.0
    dropout_p: float = 0.0
    layer_types: list[str] | None = None
    full_attention_interval: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    pad_idx: int | None = None
    """The index of the pad symbol in the vocabulary."""

    def __post_init__(self) -> None:
        if self.layer_types is None:
            interval = self.full_attention_interval
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval) else "full_attention"
                for i in range(self.num_layers)
            ]


def register_qwen35_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Qwen35Config)

    @arch("qwen35_0.8b")
    def qwen35_0p8b() -> Qwen35Config:
        return Qwen35Config(
            model_dim=1024,
            max_seq_len=262_144,
            vocab_size=248_320,
            tied_embeddings=True,
            num_layers=24,
            num_attn_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            ffn_inner_dim=3584,
            partial_rotary_factor=0.25,
            rope_theta=10_000_000.0,
            full_attention_interval=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
        )

    @arch("qwen35_2b")
    def qwen35_2b() -> Qwen35Config:
        return Qwen35Config(
            model_dim=2048,
            max_seq_len=262_144,
            vocab_size=248_320,
            tied_embeddings=True,
            num_layers=24,
            num_attn_heads=8,
            num_key_value_heads=2,
            head_dim=256,
            ffn_inner_dim=6144,
            partial_rotary_factor=0.25,
            rope_theta=10_000_000.0,
            full_attention_interval=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=16,
        )

    @arch("qwen35_9b")
    def qwen35_9b() -> Qwen35Config:
        return Qwen35Config(
            model_dim=4096,
            max_seq_len=262_144,
            vocab_size=248_320,
            tied_embeddings=False,
            num_layers=32,
            num_attn_heads=16,
            num_key_value_heads=4,
            head_dim=256,
            ffn_inner_dim=12_288,
            partial_rotary_factor=0.25,
            rope_theta=10_000_000.0,
            full_attention_interval=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
        )

    @arch("qwen35_27b")
    def qwen35_27b() -> Qwen35Config:
        return Qwen35Config(
            model_dim=5120,
            max_seq_len=262_144,
            vocab_size=248_320,
            tied_embeddings=False,
            num_layers=64,
            num_attn_heads=24,
            num_key_value_heads=4,
            head_dim=256,
            ffn_inner_dim=17_408,
            partial_rotary_factor=0.25,
            rope_theta=10_000_000.0,
            full_attention_interval=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=48,
        )


# ---------------------------------------------------------------------------
# Qwen 3.5 MoE Config
# ---------------------------------------------------------------------------

QWEN35_MOE_FAMILY: Final = "qwen3_5_moe"


@dataclass(kw_only=True)
class Qwen35MoeConfig(Qwen35Config):
    """Holds the configuration of a Qwen 3.5 MoE model."""

    model_dim: int = 2048
    num_layers: int = 40
    num_key_value_heads: int = 2
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    router_aux_loss_coef: float = 0.001


def register_qwen35_moe_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Qwen35MoeConfig)

    @arch("qwen35_moe_35b_a3b")
    def qwen35_moe_35b_a3b() -> Qwen35MoeConfig:
        return Qwen35MoeConfig()


# ---------------------------------------------------------------------------
# Qwen 3.6 (Multimodal VLM) Config
# ---------------------------------------------------------------------------

QWEN36_MOE_FAMILY: Final = "qwen3_6_moe"


@dataclass(kw_only=True)
class Qwen36VisionConfig:
    """Configuration for the Qwen 3.6 ViT vision encoder."""

    depth: int = 27
    """Number of vision transformer blocks."""

    hidden_size: int = 1152
    """Hidden dimensionality of vision encoder."""

    num_heads: int = 16
    """Number of attention heads in vision blocks."""

    intermediate_size: int = 4304
    """MLP intermediate dimensionality in vision blocks."""

    in_channels: int = 3
    """Number of input image channels."""

    patch_size: int = 16
    """Spatial patch size for Conv3d embedding."""

    temporal_patch_size: int = 2
    """Temporal patch size for Conv3d embedding (video)."""

    spatial_merge_size: int = 2
    """Merger groups NxN neighboring patches for downsampling."""

    num_position_embeddings: int = 2304
    """Number of learned position embeddings."""

    out_hidden_size: int = 5120
    """Output dimensionality of the merger (matches text model_dim)."""


@dataclass(kw_only=True)
class Qwen36Config:
    """Configuration for a Qwen 3.6 dense VLM (text + vision)."""

    text_config: Qwen35Config = field(default_factory=lambda: Qwen35Config(
        model_dim=5120,
        max_seq_len=262_144,
        vocab_size=248_320,
        tied_embeddings=False,
        num_layers=64,
        num_attn_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        ffn_inner_dim=17_408,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
    ))
    """Text backbone configuration (identical to Qwen 3.5)."""

    vision_config: Qwen36VisionConfig = field(
        default_factory=lambda: Qwen36VisionConfig(out_hidden_size=5120)
    )
    """Vision encoder configuration."""

    image_token_id: int = 248056
    """Token ID used as placeholder for image features."""

    video_token_id: int = 248057
    """Token ID used as placeholder for video features."""

    vision_start_token_id: int = 248053
    """Token ID marking the start of a vision sequence."""

    vision_end_token_id: int = 248054
    """Token ID marking the end of a vision sequence."""

    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    """Frequency pair counts for 3-section M-RoPE: [temporal, height, width]."""


@dataclass(kw_only=True)
class Qwen36MoeConfig:
    """Configuration for a Qwen 3.6 MoE VLM (text MoE + vision)."""

    text_config: Qwen35MoeConfig = field(default_factory=Qwen35MoeConfig)
    """Text backbone configuration (Qwen 3.5 MoE)."""

    vision_config: Qwen36VisionConfig = field(
        default_factory=lambda: Qwen36VisionConfig(out_hidden_size=2048)
    )
    """Vision encoder configuration."""

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])


def register_qwen36_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Qwen36Config)

    @arch("qwen36_27b")
    def qwen36_27b() -> Qwen36Config:
        return Qwen36Config()


def register_qwen36_moe_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Qwen36MoeConfig)

    @arch("qwen36_moe_35b_a3b")
    def qwen36_moe_35b_a3b() -> Qwen36MoeConfig:
        return Qwen36MoeConfig()


# ---------------------------------------------------------------------------
# Qwen 2.5 / 3.0 arch configs
# ---------------------------------------------------------------------------


def register_qwen_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, QwenConfig)

    @arch("qwen25_3b")
    def qwen25_3b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 2048
        config.vocab_size = 151_936
        config.num_layers = 36
        config.num_attn_heads = 16
        config.num_key_value_heads = 2
        config.ffn_inner_dim = 11_008
        config.rope_theta = 1_000_000

        return config

    @arch("qwen25_7b")
    def qwen25_7b() -> QwenConfig:
        return QwenConfig()

    @arch("qwen25_14b")
    def qwen25_14b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 5120
        config.num_layers = 48
        config.num_attn_heads = 40
        config.num_key_value_heads = 8
        config.ffn_inner_dim = 13_824

        return config

    @arch("qwen25_32b")
    def qwen25_32b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 5120
        config.num_layers = 64
        config.num_attn_heads = 40
        config.num_key_value_heads = 8
        config.ffn_inner_dim = 27_648

        return config

    @arch("qwen25_1_5b")
    def qwen25_1_5b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 1536
        config.vocab_size = 151_936
        config.tied_embeddings = True
        config.num_attn_heads = 12
        config.num_key_value_heads = 2
        config.ffn_inner_dim = 8960

        return config

    @arch("qwen3_0.6b")
    def qwen3_0p6b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 1024
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.tied_embeddings = True
        config.num_layers = 28
        config.num_attn_heads = 16
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 3072
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_1.7b")
    def qwen3_1p7b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 2048
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.tied_embeddings = True
        config.num_layers = 28
        config.num_attn_heads = 16
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 6144
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_4b")
    def qwen3_4b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 2560
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.tied_embeddings = True
        config.num_layers = 36
        config.num_attn_heads = 32
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 9728
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_8b")
    def qwen3_8b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 4096
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.num_layers = 36
        config.num_attn_heads = 32
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 12_288
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_14b")
    def qwen3_14b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 5120
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.num_layers = 40
        config.num_attn_heads = 40
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 17_408
        config.rope_theta = 1_000_000

        return config

    @arch("qwen3_32b")
    def qwen3_32b() -> QwenConfig:
        config = QwenConfig()

        config.model_dim = 5120
        config.max_seq_len = 40_960
        config.vocab_size = 151_936
        config.num_layers = 64
        config.num_attn_heads = 64
        config.num_key_value_heads = 8
        config.head_dim = 128
        config.qkv_proj_bias = False
        config.q_norm = True
        config.k_norm = True
        config.ffn_inner_dim = 25_600
        config.rope_theta = 1_000_000

        return config
