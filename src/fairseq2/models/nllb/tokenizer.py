# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Final, Optional, Sequence, Union, final

import torch
from overrides import final as finaloverride

from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    TokenDecoder,
    TokenEncoder,
    Tokenizer,
    VocabularyInfo,
)
from fairseq2.data.typing import PathLike
from fairseq2.models.nllb.config import supported_nllb_variants
from fairseq2.models.utils.download import download_tokenizer

_TOKENIZER: Final = "https://tinyurl.com/flores200sacrebleuspm"

_FAIRCLUSTER_TOKENIZER: Final = "/large_experiments/seamless/nllb/opensource/spm_200/sentencepiece.source.256000.model"


def load_nllb_tokenizer(
    variant_or_pathname: Union[str, PathLike], progress: bool = True
) -> "NllbTokenizer":
    """Load the tokenizer used by NLLB models.

    :param variant_or_pathname:
        The model variant, or the pathname of the NLLB SentencePiece model file.
    :param progress:
        If ``True``, displays a progress bar to stderr.
    """
    pathname = variant_or_pathname

    if isinstance(pathname, str) and pathname in supported_nllb_variants():
        if "FAIR_ENV_CLUSTER" not in os.environ:
            pathname = download_tokenizer(
                _TOKENIZER, tokenizer_name="NLLB", sub_dir="nllb", progress=progress
            )
        else:
            pathname = _FAIRCLUSTER_TOKENIZER

    return NllbTokenizer(pathname)


@final
class NllbTokenizer(Tokenizer):
    """Represents the tokenizer used by NLLB models."""

    model: SentencePieceModel
    """The model used for tokenization."""

    langs: Sequence[str]
    """The list of languages supported by the tokenizer."""

    def __init__(self, pathname: PathLike) -> None:
        """
        :param pathname:
            The pathname of the NLLB SentencePiece model file.
        """
        # We use a helper method for readability since the list of languages is
        # rather long.
        self.langs = NllbTokenizer._get_lang_list()

        # Each language is represented by a __<lang>__ token.
        control_tokens = ["__" + lang + "__" for lang in self.langs]

        # Internal control tokens that are not relevant for public use.
        control_tokens.extend(["<MINED_DATA>", "<NMT_BT_DATA>", "<SMT_BT_DATA>"])

        # The SentencePiece model of NLLB is peculiar as it does not define a
        # pad token. We use an undocumented feature of our underlying C++ API
        # to insert a pad token to the model at index 0.
        control_tokens.append("<pad>@0")

        self.model = SentencePieceModel(pathname, control_tokens)

        vocab_info = VocabularyInfo(
            self.model.vocab_size,
            self.model.unk_idx,
            self.model.bos_idx,
            self.model.eos_idx,
            self.model.pad_idx,
        )

        super().__init__(vocab_info)

    @finaloverride
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        pin_memory: bool = False,
        dtype: torch.dtype = torch.int64,
        disable_parallelism: bool = False,
    ) -> TokenEncoder:
        """Create a token encoder.

        :param task:
            The only valid value is 'translation'. If ``None``, defaults to
            'translation'.
        :param lang:
            A language from :attr:`langs`.
        :param mode:
            The valid values are 'source' and 'target'. Set to 'source' if
            ``lang`` is the source language of the translation; otherwise, set
            to 'target'. If ``None``, defaults to 'source'.
        :param batch_size:
            If the number of sentences to encode is less than ``batch_size``,
            the output will be padded.
        :param device:
            The device on which to initialize token indices.
        :param pin_memory:
            If ``True``, uses pinned memory before copying token indices to the
            target device. (only supported by CUDA devices)
        :param dtype:
            The integral data type of generated token indices.
        :param disabled_parallelism:
            If ``True``, disables parallelism and uses the calling thread only.
        """
        if task is not None and task != "translation":
            raise ValueError(f"`task` ({task}) must be 'translation'.")

        # If not specified, we fall back to English.
        if lang is None or lang == "":
            lang = "eng_Latn"
        elif lang not in self.langs:
            raise ValueError(f"`lang` ({lang}) is not a supported language.")

        if mode is None or mode == "source":
            # NLLB models expect a language token in place of BOS in source
            # sequences.
            prefix_tokens = ["__" + lang + "__"]
            suffix_tokens = ["</s>"]
        elif mode == "target":
            # Target sequences are expected to start with an EOS, followed by
            # the language token.
            prefix_tokens = ["</s>", "__" + lang + "__"]
            suffix_tokens = []
        else:
            raise ValueError(f"`mode` ({mode}) must be 'source' or 'target'")

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            batch_size=batch_size,
            device=device,
            pin_memory=pin_memory,
            dtype=dtype,
            disable_parallelism=disable_parallelism,
        )

    @finaloverride
    def create_decoder(self) -> TokenDecoder:
        return SentencePieceDecoder(self.model)

    @staticmethod
    def _get_lang_list() -> Sequence[str]:
        return [
            "ace_Arab",
            "ace_Latn",
            "acm_Arab",
            "acq_Arab",
            "aeb_Arab",
            "afr_Latn",
            "ajp_Arab",
            "aka_Latn",
            "amh_Ethi",
            "apc_Arab",
            "arb_Arab",
            "ars_Arab",
            "ary_Arab",
            "arz_Arab",
            "asm_Beng",
            "ast_Latn",
            "awa_Deva",
            "ayr_Latn",
            "azb_Arab",
            "azj_Latn",
            "bak_Cyrl",
            "bam_Latn",
            "ban_Latn",
            "bel_Cyrl",
            "bem_Latn",
            "ben_Beng",
            "bho_Deva",
            "bjn_Arab",
            "bjn_Latn",
            "bod_Tibt",
            "bos_Latn",
            "bug_Latn",
            "bul_Cyrl",
            "cat_Latn",
            "ceb_Latn",
            "ces_Latn",
            "cjk_Latn",
            "ckb_Arab",
            "crh_Latn",
            "cym_Latn",
            "dan_Latn",
            "deu_Latn",
            "dik_Latn",
            "dyu_Latn",
            "dzo_Tibt",
            "ell_Grek",
            "eng_Latn",
            "epo_Latn",
            "est_Latn",
            "eus_Latn",
            "ewe_Latn",
            "fao_Latn",
            "pes_Arab",
            "fij_Latn",
            "fin_Latn",
            "fon_Latn",
            "fra_Latn",
            "fur_Latn",
            "fuv_Latn",
            "gla_Latn",
            "gle_Latn",
            "glg_Latn",
            "grn_Latn",
            "guj_Gujr",
            "hat_Latn",
            "hau_Latn",
            "heb_Hebr",
            "hin_Deva",
            "hne_Deva",
            "hrv_Latn",
            "hun_Latn",
            "hye_Armn",
            "ibo_Latn",
            "ilo_Latn",
            "ind_Latn",
            "isl_Latn",
            "ita_Latn",
            "jav_Latn",
            "jpn_Jpan",
            "kab_Latn",
            "kac_Latn",
            "kam_Latn",
            "kan_Knda",
            "kas_Arab",
            "kas_Deva",
            "kat_Geor",
            "knc_Arab",
            "knc_Latn",
            "kaz_Cyrl",
            "kbp_Latn",
            "kea_Latn",
            "khm_Khmr",
            "kik_Latn",
            "kin_Latn",
            "kir_Cyrl",
            "kmb_Latn",
            "kon_Latn",
            "kor_Hang",
            "kmr_Latn",
            "lao_Laoo",
            "lvs_Latn",
            "lij_Latn",
            "lim_Latn",
            "lin_Latn",
            "lit_Latn",
            "lmo_Latn",
            "ltg_Latn",
            "ltz_Latn",
            "lua_Latn",
            "lug_Latn",
            "luo_Latn",
            "lus_Latn",
            "mag_Deva",
            "mai_Deva",
            "mal_Mlym",
            "mar_Deva",
            "min_Latn",
            "mkd_Cyrl",
            "plt_Latn",
            "mlt_Latn",
            "mni_Beng",
            "khk_Cyrl",
            "mos_Latn",
            "mri_Latn",
            "zsm_Latn",
            "mya_Mymr",
            "nld_Latn",
            "nno_Latn",
            "nob_Latn",
            "npi_Deva",
            "nso_Latn",
            "nus_Latn",
            "nya_Latn",
            "oci_Latn",
            "gaz_Latn",
            "ory_Orya",
            "pag_Latn",
            "pan_Guru",
            "pap_Latn",
            "pol_Latn",
            "por_Latn",
            "prs_Arab",
            "pbt_Arab",
            "quy_Latn",
            "ron_Latn",
            "run_Latn",
            "rus_Cyrl",
            "sag_Latn",
            "san_Deva",
            "sat_Beng",
            "scn_Latn",
            "shn_Mymr",
            "sin_Sinh",
            "slk_Latn",
            "slv_Latn",
            "smo_Latn",
            "sna_Latn",
            "snd_Arab",
            "som_Latn",
            "sot_Latn",
            "spa_Latn",
            "als_Latn",
            "srd_Latn",
            "srp_Cyrl",
            "ssw_Latn",
            "sun_Latn",
            "swe_Latn",
            "swh_Latn",
            "szl_Latn",
            "tam_Taml",
            "tat_Cyrl",
            "tel_Telu",
            "tgk_Cyrl",
            "tgl_Latn",
            "tha_Thai",
            "tir_Ethi",
            "taq_Latn",
            "taq_Tfng",
            "tpi_Latn",
            "tsn_Latn",
            "tso_Latn",
            "tuk_Latn",
            "tum_Latn",
            "tur_Latn",
            "twi_Latn",
            "tzm_Tfng",
            "uig_Arab",
            "ukr_Cyrl",
            "umb_Latn",
            "urd_Arab",
            "uzn_Latn",
            "vec_Latn",
            "vie_Latn",
            "war_Latn",
            "wol_Latn",
            "xho_Latn",
            "ydd_Hebr",
            "yor_Latn",
            "yue_Hant",
            "zho_Hans",
            "zho_Hant",
            "zul_Latn",
        ]
