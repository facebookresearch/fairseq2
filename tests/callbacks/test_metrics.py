import fairseq2.callbacks
from tests.common import assert_equal


def test_bleu() -> None:
    bleu = fairseq2.callbacks.Bleu()
    bleu.update("a b c d", ["a b d"])
    # hyp_len, ref_len, *correct, *total
    assert_equal(bleu.bleu_counts, [4, 3, 3, 1, 0, 0, 4, 3, 2, 1])
    assert 35 == round(bleu.compute().item())
