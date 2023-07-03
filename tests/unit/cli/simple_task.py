import logging
import random
from typing import Iterable, NamedTuple, Optional, Sequence

import torch
import torchtnt.utils
from torch import Tensor

import fairseq2.cli
from fairseq2 import data
from fairseq2.cli.api import Env
from fairseq2.data import CString, StringLike
from fairseq2.data.text import TokenDecoder, TokenEncoder, Tokenizer, VocabularyInfo
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.nllb import NllbConfig, create_nllb_model
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.typing import DataType, Device

log = logging.getLogger(__name__)
random.seed(0)


class CustomClass(NamedTuple):
    foo: str = "foo"
    bar: str = "bar"


class NumberTokenizer(Tokenizer):
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
        dtype: DataType = torch.int64,
        disable_parallelism: bool = False,
    ) -> "TokenEncoder":
        def encode(sentences: Sequence[StringLike]) -> Tensor:
            if isinstance(sentences, CString):
                sentences = [sentences]
            return torch.tensor(
                [[int(x) for x in str(s).split()] for s in sentences],
                dtype=dtype,
                device=device,
            )

        return encode  # type: ignore

    def create_decoder(self) -> "TokenDecoder":
        return lambda tokens: [" ".join(str(x.item()) for x in row) for row in tokens]  # type: ignore


def tokenizer(custom: CustomClass) -> Tokenizer:
    return NumberTokenizer(VocabularyInfo(100, 0, 0, 0, 0))


def custom() -> CustomClass:
    return CustomClass("foo", "baz")


def generate_samples(
    env: Env, tokenizer: Tokenizer, batch_size: int, num_examples: int
) -> Iterable[Seq2SeqBatch]:
    vocab_size = tokenizer.vocab_info.size
    src_seq_lens = torch.ones((batch_size, 1), device=env.device, dtype=torch.int64) * 2
    tgt_seq_lens = torch.ones((batch_size, 1), device=env.device, dtype=torch.int64) * 2

    def text_example(_: int) -> str:
        a = random.randrange(vocab_size)
        b = random.randrange(vocab_size)
        c = (a + b) % vocab_size
        # TODO: this is a bit ugly. I should return src and tgt
        return f"{a} {b} 0 {c}"

    def make_batch(batch: Tensor) -> Seq2SeqBatch:
        bs = batch.size(0)
        return Seq2SeqBatch(
            batch[:, :2], src_seq_lens[:bs], batch[:, 2:], tgt_seq_lens[:bs]
        )

    return (
        data.read_sequence(range(num_examples))
        .map(text_example)
        .bucket(batch_size)
        .collate()
        .map(tokenizer.create_encoder(device=env.device, dtype=torch.int64))
        .map(make_batch)
        .and_return()
    )  # type: ignore[no-any-return]


def train_data(
    env: Env, tokenizer: Tokenizer, batch_size: int = 8
) -> Iterable[Seq2SeqBatch]:
    return generate_samples(env, tokenizer, batch_size, 500)


def valid_data(
    env: Env, tokenizer: Tokenizer, batch_size: int = 8
) -> Iterable[Seq2SeqBatch]:
    return generate_samples(env, tokenizer, batch_size, 10)


def module(env: Env, transformer: NllbConfig) -> EncoderDecoderModel:
    """The translation model, see transformer for configuration"""
    torchtnt.utils.seed(0)
    torch.cuda.manual_seed(0)
    return create_nllb_model(transformer, env.device)


# Override default values of NllbConfig
transformer = lambda: NllbConfig(
    model_dim=32,
    max_seq_len=1024,
    vocabulary_size=100,
    pad_idx=0,
    num_encoder_layers=2,
    num_decoder_layers=1,
    num_encoder_attn_heads=4,
    num_decoder_attn_heads=4,
    ffn_inner_dim=32,
    dropout_p=0,
)

# This is important, it tells torch.hub how to reload our "task" which contains model and tokenizer.
fairseq2_hub = fairseq2.cli.fairseq2_hub


if __name__ == "__main__":
    import fairseq2.cli.commands

    fairseq2.cli.commands.main(__file__)
