from pathlib import Path
from typing import cast

import pytest
import torch

import fairseq2.generate
import fairseq2.nn
from fairseq2 import services
from fairseq2.assets import AssetStore
from fairseq2.compat.models.transformer import load_fairseq1_checkpoint
from fairseq2.data.text import Tokenizer
from fairseq2.models.nllb import NllbLoader
from fairseq2.models.transformer import TransformerConfig, create_transformer_model
from tests.common import assert_close, assert_equal, device

NLLB_MODELS = Path("/large_experiments/seamless/nllb/opensource/")
NLLB_SMALL = NLLB_MODELS / "nllb_200_dense_distill_600m/checkpoint.pt"

ENG = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
FRA_1 = "Lundi, des scientifiques de l'École de médecine de l'Université de Stanford ont annoncé l'invention d'un nouvel outil de diagnostic capable de trier les cellules par type: une minuscule puce imprimable qui peut être fabriquée à l'aide d'imprimantes à jet d'encre standard pour environ un centime de centime chacun."
FRA_5 = "Lundi, des scientifiques de l'École de médecine de l'Université de Stanford ont annoncé l'invention d'un nouvel outil de diagnostic capable de trier les cellules par type: une petite puce puce imprimable qui peut être fabriquée à l'aide d'imprimantes à jet d'encre standard, éventuellement un centime cent chacun."


@pytest.mark.skipif(not NLLB_MODELS.exists(), reason="needs to run on FAIR cluster")
def test_loading_nllb200_small(tmp_path: Path) -> None:
    card = services.get(AssetStore).retrieve_card("nllb_dense_distill_600m")

    tokenizer1 = NllbLoader(card, progress=False).load_tokenizer()

    # Load fairseq checkpoint into fairseq2
    model1, cfg1 = load_fairseq1_checkpoint(NLLB_SMALL, tokenizer1.vocab_info, device)

    model1.eval()

    src_encoder1 = tokenizer1.create_encoder(
        lang="eng_Latn", mode="source", device=device
    )
    src_decoder1 = tokenizer1.create_decoder()

    src_tok1 = src_encoder1(ENG)
    assert src_decoder1(src_tok1) == [ENG]

    assert_speaks_french(model1, tokenizer1, device)

    # Save NLLB200 model as a fairseq2 model
    state = {
        "model": model1.state_dict(),
        "cfg": cfg1,
        "tokenizer": tokenizer1,
    }
    torch.save(state, tmp_path / "nllb200.fairseq2.pt")

    # Reload NLLB200 as a fairseq2 model
    state = torch.load(tmp_path / "nllb200.fairseq2.pt")
    cfg2 = cast(TransformerConfig, state["cfg"])
    tokenizer2 = cast(Tokenizer, state["tokenizer"])
    model2 = create_transformer_model(cfg2, tokenizer2.vocab_info)
    model2.load_state_dict(state["model"])  # type: ignore
    model2.eval()

    src_encoder2 = tokenizer2.create_encoder(
        lang="eng_Latn", mode="source", device=device
    )

    src_tok2 = src_encoder2(ENG)
    assert_equal(src_tok1, src_tok2)

    assert_close(model1.encode(src_tok1)[0], model2.encode(src_tok2)[0])

    assert_speaks_french(model2, tokenizer2, device)


@torch.inference_mode()
def assert_speaks_french(
    model: fairseq2.models.transformer.TransformerModel,
    tokenizer: Tokenizer,
    device: torch.device,
) -> None:
    for beam_size, ref in [(1, FRA_1), (5, FRA_5)]:
        strategy = fairseq2.generate.BeamSearchStrategy(
            vocab_info=tokenizer.vocab_info, beam_size=beam_size, max_len=256
        )

        fra = strategy.generate_str(
            model,
            tokenizer,
            [ENG],
            src_lang="eng_Latn",
            tgt_lang="fra_Latn",
            device=device,
        )

        print(fra)
        assert fra == [ref]
