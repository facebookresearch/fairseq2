# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.llama.builder import LLaMAConfig, create_llama_model, llama_archs
from fairseq2.models.llama.tokenizer import LLaMATokenizer
from fairseq2.models.transformer import TransformerDecoderModel
from fairseq2.models.utils import ConfigLoader, ModelLoader, TokenizerLoader
from fairseq2.models.utils.checkpoint import convert_model_state_dict


def convert_llama_checkpoint(
    checkpoint: Dict[str, Any], config: LLaMAConfig
) -> Dict[str, Any]:
    """Convert a reference LLaMA checkpoint to fairseq2."""
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


load_llama_config = ConfigLoader[LLaMAConfig](asset_store, llama_archs)

load_llama_model = ModelLoader[TransformerDecoderModel, LLaMAConfig](
    asset_store,
    download_manager,
    load_llama_config,
    create_llama_model,
    convert_llama_checkpoint,
)

load_llama_tokenizer = TokenizerLoader[LLaMATokenizer](
    asset_store, download_manager, LLaMATokenizer
)
