from pathlib import Path

from fairseq2.generate import tokenizer

DATA = Path(__file__).parents[0] / "data"
SPM_PATH = DATA / "eng_Latn.1000.spm"


def build_test_spm_tokenizer() -> tokenizer.SpmTokenizer:
    """Build a small testing SpmTokenizer from a local model.

    :return: an SpmTokenizer.
    """
    return tokenizer.SpmTokenizer.from_file(SPM_PATH)
