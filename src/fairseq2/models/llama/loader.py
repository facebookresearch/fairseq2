# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any, final

from torch import Tensor
from typing_extensions import override

from fairseq2.assets import AssetCard
from fairseq2.data.text import (
    AbstractTextTokenizerLoader,
    BasicSentencePieceTokenizer,
    TextTokenizer,
    load_text_tokenizer,
)
from fairseq2.gang import Gang
from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.llama.factory import (
    LLAMA_FAMILY,
    LLaMAConfig,
    create_llama_model,
    llama_archs,
)
from fairseq2.models.llama.tokenizer import LLaMA3Tokenizer
from fairseq2.models.loader import StandardModelLoader, load_model
from fairseq2.models.transformer import (
    TransformerDecoderModel,
    shard_transformer_decoder_model,
)
from fairseq2.models.utils.checkpoint import convert_model_state_dict

load_llama_config = StandardModelConfigLoader(LLAMA_FAMILY, LLaMAConfig, llama_archs)


@final
class LLaMAModelLoader(StandardModelLoader[TransformerDecoderModel, LLaMAConfig]):
    """Loads LLaMA models."""

    @override
    def _shard(
        self, model: TransformerDecoderModel, gangs: dict[str, Gang], card: AssetCard
    ) -> None:
        gang = gangs["tp"]  # tensor parallel

        shard_embed_dim = card.field("shard_embed_dim").get_as_(bool, True)

        shard_transformer_decoder_model(model, gang, shard_embed_dim=shard_embed_dim)


def convert_llama_checkpoint(
    checkpoint: dict[str, Any], config: LLaMAConfig
) -> dict[str, Any]:
    """Convert a reference or Hugging Face LLaMA checkpoint to fairseq2 format."""
    # Check if we have a fairseq2 checkpoint.
    if "model" in checkpoint:
        return checkpoint

    # Check if we have a reference or Hugging Face checkpoint.
    if "lm_head.weight" in checkpoint:  # HG
        head_dim = config.model_dim // config.num_attn_heads

        def permute_rotary(w: Tensor, num_heads: int) -> Tensor:
            # (H, M) -> (H_d, 2, D / 2, M)
            w = w.view(num_heads, 2, head_dim // 2, config.model_dim)

            # (H_d, 2, D / 2, M) -> (H_d, D / 2, 2, M)
            w = w.transpose(1, 2)

            # (H_d, D / 2, 2, M) -> (H, M)
            return w.reshape(-1, config.model_dim)

        for idx in range(config.num_layers):
            q_key = f"model.layers.{idx}.self_attn.q_proj.weight"
            k_key = f"model.layers.{idx}.self_attn.k_proj.weight"

            q_proj = checkpoint[q_key]
            k_proj = checkpoint[k_key]

            q_proj = permute_rotary(q_proj, config.num_attn_heads)
            k_proj = permute_rotary(k_proj, config.num_key_value_heads)

            checkpoint[q_key] = q_proj
            checkpoint[k_key] = k_proj

        key_map = {
            # fmt: off
            r"^model\.layers\.([0-9]+)\.self_attn\.q_proj\.":        r"decoder.layers.\1.self_attn.q_proj.",
            r"^model\.layers\.([0-9]+)\.self_attn\.k_proj\.":        r"decoder.layers.\1.self_attn.k_proj.",
            r"^model\.layers\.([0-9]+)\.self_attn\.v_proj\.":        r"decoder.layers.\1.self_attn.v_proj.",
            r"^model\.layers\.([0-9]+)\.self_attn\.o_proj\.":        r"decoder.layers.\1.self_attn.output_proj.",
            r"^model\.layers\.([0-9]+)\.post_attention_layernorm\.": r"decoder.layers.\1.ffn_layer_norm.",
            r"^model\.layers\.([0-9]+)\.mlp\.gate_proj\.":           r"decoder.layers.\1.ffn.gate_proj.",
            r"^model\.layers\.([0-9]+)\.mlp\.down_proj\.":           r"decoder.layers.\1.ffn.output_proj.",
            r"^model\.layers\.([0-9]+)\.mlp\.up_proj\.":             r"decoder.layers.\1.ffn.inner_proj.",
            r"^model\.layers\.([0-9]+)\.input_layernorm\.":          r"decoder.layers.\1.self_attn_layer_norm.",
            r"^model\.norm\.":                                       r"decoder.layer_norm.",
            r"^model\.embed_tokens\.":                               r"decoder_frontend.embed.",
            r"^lm_head\.":                                           r"final_proj.",
            # fmt: on
        }
    else:
        key_map = {
            # fmt: off
            r"^layers\.([0-9]+)\.attention\.wq\.":    r"decoder.layers.\1.self_attn.q_proj.",
            r"^layers\.([0-9]+)\.attention\.wk\.":    r"decoder.layers.\1.self_attn.k_proj.",
            r"^layers\.([0-9]+)\.attention\.wv\.":    r"decoder.layers.\1.self_attn.v_proj.",
            r"^layers\.([0-9]+)\.attention\.wo\.":    r"decoder.layers.\1.self_attn.output_proj.",
            r"^layers\.([0-9]+)\.attention_norm\.":   r"decoder.layers.\1.self_attn_layer_norm.",
            r"^layers\.([0-9]+)\.feed_forward\.w1\.": r"decoder.layers.\1.ffn.gate_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w2\.": r"decoder.layers.\1.ffn.output_proj.",
            r"^layers\.([0-9]+)\.feed_forward\.w3\.": r"decoder.layers.\1.ffn.inner_proj.",
            r"^layers\.([0-9]+)\.ffn_norm\.":         r"decoder.layers.\1.ffn_layer_norm.",
            r"^norm\.":                               r"decoder.layer_norm.",
            r"^tok_embeddings\.":                     r"decoder_frontend.embed.",
            r"^output\.":                             r"final_proj.",
            # fmt: on
        }

        # We do not need the pre-computed 'rope.freqs' buffers.
        checkpoint = {k: v for (k, v) in checkpoint.items() if "rope.freqs" not in k}

    checkpoint = convert_model_state_dict(checkpoint, key_map)

    return {"model": checkpoint}


load_llama_model = LLaMAModelLoader(
    config_loader=load_llama_config,
    factory=create_llama_model,
    checkpoint_converter=convert_llama_checkpoint,
)

load_model.register(LLAMA_FAMILY, load_llama_model)


@final
class LLaMATokenizerLoader(AbstractTextTokenizerLoader[TextTokenizer]):
    """Loads LLaMA tokenizers."""

    @override
    def _load(self, path: Path, card: AssetCard) -> TextTokenizer:
        if card.field("use_v2_tokenizer").get_as_(bool, False):
            f = card.field("model_config").field("vocab_info").field("eos_idx")

            eot_idx = 128_009  # end-of-turn

            return LLaMA3Tokenizer(path, instruct=f.get_as_(int) == eot_idx)

        return BasicSentencePieceTokenizer(path)


load_llama_tokenizer = LLaMATokenizerLoader()

load_text_tokenizer.register(LLAMA_FAMILY, load_llama_tokenizer)
