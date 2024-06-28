# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, final

from fairseq2.assets import AssetCard
from fairseq2.data.text import (
    AbstractTextTokenizerLoader,
    BasicSentencePieceTokenizer,
    TextTokenizer,
    load_text_tokenizer,
)
from fairseq2.gang import Gang
from fairseq2.models.config_loader import StandardModelConfigLoader
from fairseq2.models.llama.archs import llama_archs
from fairseq2.models.llama.factory import LLAMA_FAMILY, LLaMAConfig, create_llama_model
from fairseq2.models.llama.tokenizer import LLaMA3Tokenizer
from fairseq2.models.loader import DenseModelLoader, load_model
from fairseq2.models.transformer import (
    TransformerDecoderModel,
    shard_transformer_decoder_model,
)
from fairseq2.models.utils.checkpoint import convert_model_state_dict
from fairseq2.typing import override

load_llama_config = StandardModelConfigLoader(
    family=LLAMA_FAMILY, config_kls=LLaMAConfig, arch_configs=llama_archs
)


@final
class LLaMAModelLoader(DenseModelLoader[TransformerDecoderModel, LLaMAConfig]):
    """Loads LLaMA models."""

    @override
    def _shard(
        self, model: TransformerDecoderModel, gangs: Dict[str, Gang], card: AssetCard
    ) -> None:
        gang = gangs["tp"]  # tensor parallel

        shard_embed_dim = card.field("shard_embed_dim").get_as_(bool, True)

        shard_transformer_decoder_model(model, gang, shard_embed_dim=shard_embed_dim)


def convert_llama_checkpoint(
    checkpoint: Dict[str, Any], config: LLaMAConfig
) -> Dict[str, Any]:
    """Convert a reference LLaMA checkpoint to fairseq2 format."""
    # Check if we have a fairseq2 checkpoint.
    if "output.weight" not in checkpoint:
        return checkpoint

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
