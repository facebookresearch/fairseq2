from pathlib import Path
from typing import cast

import pytest
import torch

import fairseq2.generate
import fairseq2.nn
from fairseq2.compat.models.transformer import load_fairseq1_checkpoint
from fairseq2.models.transformer import TransformerConfig, create_transformer_model
from tests.common import assert_close, assert_equal, device

NLLB_MODELS = Path("/large_experiments/seamless/nllb/opensource/")
NLLB_SMALL = NLLB_MODELS / "nllb_200_dense_distill_600m/checkpoint.pt"
NLLB_TOK = NLLB_MODELS / "spm_200/sentencepiece.source.256000.model"

ENG = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
FRA_1 = "Lundi, des scientifiques de l'École de médecine de l'Université de Stanford ont annoncé l'invention d'un nouvel outil de diagnostic capable de trier les cellules par type: une minuscule puce imprimable qui peut être fabriquée à l'aide d'imprimantes à jet d'encre standard pour environ un centime de centime chacun."
FRA_5 = "Lundi, des scientifiques de l'École de médecine de l'Université de Stanford ont annoncé l'invention d'un nouvel outil de diagnostic capable de trier les cellules par type: une minuscule puce imprimable qui peut être fabriquée à l'aide d'imprimantes à jet d'encre standard à environ un centime chacun."


@pytest.mark.skipif(not NLLB_MODELS.exists(), reason="needs to run on FAIR cluster")
def test_loading_nllb200_small(tmp_path: Path) -> None:
    # Load fairseq checkpoint into fairseq2
    model2, tokenizer2, cfg = load_fairseq1_checkpoint(NLLB_SMALL, NLLB_TOK, device)

    assert tokenizer2.special_tokens["__ace_Arab__"] == 256001
    assert tokenizer2.vocab_size() == 256206

    model2.eval()

    src_bos_tok = tokenizer2.special_tokens["__eng_Latn__"]
    src_tokens2 = tokenizer2.encode_batch([ENG], bos=src_bos_tok)
    assert tokenizer2.decode_batch(src_tokens2) == [ENG]

    print(" ".join(tokenizer2.spm.encode_as_pieces(ENG)))
    try:
        tokenizer1 = fairseq2.generate.DictTokenizer.from_fairseq_dict_txt(
            NLLB_TOK.parent / "dictionary.txt"
        )
        src_tokens1 = tokenizer1.encode_batch(
            [" ".join(tokenizer2.spm.encode_as_pieces(ENG))], bos=src_bos_tok
        )
        assert_equal(src_tokens2[:, :-1], src_tokens1[:, :-1])
    except ImportError:
        # The above tests uses fairseq tokenization
        pass

    src_tokens2 = src_tokens2.to(device)
    assert_speaks_french(model2, tokenizer2, device)

    # Save NLLB200 model as a fairseq2 model
    state = {
        "model": model2.state_dict(),
        "cfg": cfg,
        "tokenizer": tokenizer2,
    }
    torch.save(state, tmp_path / "nllb200.fairseq2.pt")

    # Reload NLLB200 as a fairseq2 model
    state = torch.load(tmp_path / "nllb200.fairseq2.pt")
    cfg3 = cast(TransformerConfig, state["cfg"])
    model3 = create_transformer_model(cfg3)
    tokenizer3 = cast(fairseq2.generate.Tokenizer, state["tokenizer"])
    model3.load_state_dict(state["model"])  # type: ignore
    model3.eval()
    src_tokens3 = tokenizer3.encode_batch([ENG], bos=src_bos_tok)
    assert_equal(src_tokens3, src_tokens2)
    assert_close(model3.encode(src_tokens3)[0], model2.encode(src_tokens2)[0])
    assert_speaks_french(model3, tokenizer3, device)


@torch.inference_mode()
def assert_speaks_french(
    model: fairseq2.models.transformer.TransformerModel,
    tokenizer: fairseq2.generate.Tokenizer,
    device: torch.device,
) -> None:
    # for beam_size, ref in [(1, FRA_1), (5, FRA_5)]:
    for beam_size, ref in [(1, FRA_1)]:
        strategy = fairseq2.generate.BeamSearchStrategy(
            token_meta=tokenizer, beam_size=beam_size, max_len=256
        )

        fra = strategy.generate_str(
            model,
            tokenizer,
            [ENG],
            src_bos="__eng_Latn__",
            tgt_bos="__fra_Latn__",
            device=device,
        )[0]

        print(fra)
        assert fra == ref
