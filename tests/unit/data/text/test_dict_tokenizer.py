import pickle
from typing import Sequence

import pytest
import torch

from fairseq2.data.text import DictModel, DictTokenizer, VocabularyInfo
from fairseq2.data.typing import StringLike
from tests.common import assert_equal

vocab: Sequence[StringLike] = ["hello", "one", "two", "world"]  # sorted alphabetically


def test_duplicate_word_throws() -> None:
    vocab_duplicates: Sequence[StringLike] = ["hello", "one", "two", "hello", "world"]
    with pytest.raises(ValueError):
        DictModel(vocab_duplicates)


def test_tokenizer_model() -> None:
    model = DictModel(vocab)

    assert "<unk>" == model.index_to_token(model.unk_idx)
    assert "<s>" == model.index_to_token(model.bos_idx)
    assert "</s>" == model.index_to_token(model.eos_idx)
    assert "<pad>" == model.index_to_token(model.pad_idx)

    assert "hello" == model.index_to_token(4)
    assert 4 == model.token_to_index("hello")

    assert "one" == model.index_to_token(5)
    assert 5 == model.token_to_index("one")


def test_pickling_() -> None:
    tokenizer = DictTokenizer(10, vocab)
    encoder = tokenizer.create_encoder()
    decoder = tokenizer.create_decoder()

    dmp = pickle.dumps(tokenizer)
    tokenizer2 = pickle.loads(dmp)
    encoder2 = tokenizer2.create_encoder()
    decoder2 = tokenizer2.create_decoder()

    X = ["hello world it is me"]
    assert_equal(encoder(X), encoder2(X))
    Y = encoder(X)
    assert decoder(Y) == decoder2(Y)


def test_vocab_info() -> None:
    tokenizer = DictTokenizer(10, vocab)
    actual = tokenizer.vocab_info
    expected: VocabularyInfo = VocabularyInfo(len(vocab) + 4, 0, 1, 2, 3)

    assert actual == expected


def test_encoder_string_arg_throws() -> None:
    tokenizer = DictTokenizer(10, vocab)
    encoder = tokenizer.create_encoder()
    with pytest.raises(ValueError):
        encoder("hello world!")


def test_encoder_one_line() -> None:
    tokenizer = DictTokenizer(10, vocab)
    encoder = tokenizer.create_encoder()
    sentence = "hello world from one person"
    expected = torch.tensor([[1, 4, 7, 0, 5, 0, 2, 3, 3, 3]])
    actual = encoder([sentence])
    assert torch.equal(actual, expected)


def test_enocder_multi_line() -> None:
    tokenizer = DictTokenizer(10, vocab)
    encoder = tokenizer.create_encoder()
    sentences = [
        "hello world from one person",
        "hello person hello i am one from world",
        "hello one two three",
        "one two three four five six seven eight nine ten",
    ]
    expected = torch.tensor(
        [
            [1, 4, 7, 0, 5, 0, 2, 3, 3, 3],
            [1, 4, 0, 4, 0, 0, 5, 0, 7, 2],
            [1, 4, 5, 6, 0, 2, 3, 3, 3, 3],
            [1, 5, 6, 0, 0, 0, 0, 0, 0, 2],
        ]
    )

    actual = encoder(sentences)
    assert torch.equal(actual, expected)


def test_decoder_one_line() -> None:
    tokenizer = DictTokenizer(10, vocab)
    decoder = tokenizer.create_decoder()
    tensor = torch.tensor([1, 4, 7, 0, 5, 0, 2, 3, 3, 3])
    expected = ["<s> hello world <unk> one <unk> </s> <pad> <pad> <pad>"]

    assert decoder(tensor) == expected


def test_decoder_multi_line() -> None:
    tokenizer = DictTokenizer(10, vocab)
    decoder = tokenizer.create_decoder()
    tensor = torch.tensor(
        [
            [1, 4, 7, 0, 5, 0, 2, 3, 3, 3],
            [1, 4, 0, 4, 0, 0, 5, 0, 7, 2],
            [1, 4, 5, 6, 0, 2, 3, 3, 3, 3],
            [1, 5, 6, 0, 0, 0, 0, 0, 0, 2],
        ],
    )
    expected = [
        "<s> hello world <unk> one <unk> </s> <pad> <pad> <pad>",
        "<s> hello <unk> hello <unk> <unk> one <unk> world </s>",
        "<s> hello one two <unk> </s> <pad> <pad> <pad> <pad>",
        "<s> one two <unk> <unk> <unk> <unk> <unk> <unk> </s>",
    ]

    assert decoder(tensor) == expected
