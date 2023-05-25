"""Train eng-fra translation model on Tatoeba dataset"""
from pathlib import Path

from fairseq2 import data
from fairseq2.generate import spm_train

TSV_FILE = Path("examples/fra.txt")
SPM_PATTERN = "examples/fra.{vocab_size}.spm"


def tokenizer(
    tsv_file: Path = TSV_FILE, vocab_size: int = 5_000, spm: str = SPM_PATTERN
) -> data.text.MultilingualTokenizer:
    """eng-fra sentencepiece tokenizer"""
    spm_path = Path(spm.format(vocab_size=vocab_size))
    if not spm_path.exists():
        cfg = spm_train.TrainSpmConfig(vocab_size=vocab_size)
        eng_fra = (
            data.text.read_text(str(tsv_file), rtrim=True, skip_header=1)
            .map(lambda line: "\n".join((str(line).split("\t")[:1])))
            .and_return()
        )
        spm_train.train_from_stream(cfg, eng_fra, spm_path)

    return data.text.MultilingualTokenizer(
        spm_path, "translation", {"eng"}, {"fra"}, "eng", "fra"
    )


from fairseq2.cli import Env


def train_data(
    tokenizer: data.text.MultilingualTokenizer,
    env: Env,
    batch_size: int = 32,
    tsv_file: Path = TSV_FILE,
) -> data.DataPipeline:
    def _read_tsv_column(
        encoder: data.text.TokenEncoder, column: int
    ) -> data.DataPipeline:
        return (
            data.text.read_text(tsv_file, rtrim=True)
            .map(lambda line: str(line).split("\t")[column])
            .map(encoder)
            .and_return()
        )

    src = _read_tsv_column(
        tokenizer.create_encoder(mode="source", lang="eng"), column=0
    )
    tgt = _read_tsv_column(
        tokenizer.create_encoder(mode="target", lang="fra"), column=1
    )

    pad_idx = tokenizer.vocab_info.pad_idx
    device = env.device
    batches = (
        data.zip_data_pipelines([src, tgt])
        .shuffle(10_000)
        .batch(batch_size, pad_idx=pad_idx)
        .map(
            lambda st: data.Seq2SeqBatch(
                source=st[0].to(device),
                # TODO: the tokenizer should compute those
                src_seq_lens=(st[0] != pad_idx).sum(dim=-1).to(device),
                target=st[1].to(device),
                tgt_seq_lens=(st[1][:, 1:] != pad_idx).sum(dim=-1).to(device),
            )
        )
        .and_return()
    )
    return batches


import dataclasses

import torch

import fairseq2.models.nllb


def module(
    model_cfg: fairseq2.models.nllb.NllbConfig,
    env: Env,
) -> torch.nn.Module:
    return fairseq2.models.nllb.create_nllb_model(model_cfg, env.device)


def model_cfg(
    tokenizer: data.text.Tokenizer, model_dim: int = 128, num_layers: int = 4
) -> fairseq2.models.nllb.NllbConfig:
    cfg = fairseq2.models.nllb.nllb_archs.get_config("dense_600m")
    return dataclasses.replace(
        cfg,
        vocabulary_size=tokenizer.vocab_info.size,
        pad_idx=tokenizer.vocab_info.pad_idx,
        model_dim=model_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        ffn_inner_dim=model_dim * 4,
    )
