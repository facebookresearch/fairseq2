"""
Trains a Transformer model for Speech To Speech Machine Translation on UST dataset.

Example command:
fairseq2 train examples/s2st.py -w /checkpoint/$USER/fairseq2/s2tt.eng-deu wandb_project=fairseq2_s2tt
"""
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torchaudio  # type: ignore
import torchtnt.utils
from torch import Tensor

import fairseq2.cli
import fairseq2.dataloader.huggingface
import fairseq2.models.s2t_transformer as s2t
import fairseq2.tasks
from fairseq2 import data
from fairseq2.data import DataPipelineBuilder, StringLike
from fairseq2.data.text import MultilingualTokenizer, Tokenizer, VocabularyInfo
from fairseq2.dataloader import Seq2SeqBatch
from fairseq2.distributed import Env
from fairseq2.generate import spm_train
from fairseq2.models.transformer import TransformerModel

log = logging.getLogger(__name__)

DATADIR: str = "/checkpoint/guw/fairseq2/data/must-c-v1.0.eng-deu"
task = fairseq2.tasks.Seq2Seq


def tokenizer(
    xp: fairseq2.cli.Xp,
    data_dir: str = DATADIR,
    lang: str = "eng_Latn",
    vocab_size: int = 1024 * 8,
) -> Tokenizer:
    spm_dir = xp.script.parent.parent / "spm"
    spm_dir.mkdir(exist_ok=True)
    spm_path = spm_dir / f"{Path(DATADIR).name}.train_asr.{vocab_size}.spm"
    if not spm_path.exists():
        cfg = spm_train.TrainSpmConfig(
            vocab_size=vocab_size,
            training_lines=1_000_000,
            control_tokens=[f"<audio:{lang}>", f"<lang:{lang}>"],
        )
        manifest_path = data_dir + "/train_asr.tsv"
        log.info(
            f"Will train {spm_path} with vocab_size={vocab_size} on {manifest_path}"
        )
        text_data = (
            data.text.read_text(str(manifest_path), rtrim=True, skip_header=1)
            .map(lambda line: str(line).split("\t")[3])
            .and_return()
        )
        spm_train.train_from_stream(cfg, text_data, spm_path)
    else:
        log.info(f"Will reuse tokenizer from {spm_path}")

    return MultilingualTokenizer(spm_path, "asr", set(), {lang}, lang, lang)


def module(
    env: Env,
    vocab_info: VocabularyInfo,
    transformer: s2t.S2TTransformerConfig,
    fsdp: bool = False,
) -> TransformerModel:
    """The translation model, see transformer for configuration.

    - fsdp: enable FSDP (default is DDP when using several GPUs)
    """
    torchtnt.utils.seed(0)
    torch.cuda.manual_seed(0)

    model = s2t.create_s2t_transformer_model(transformer, vocab_info, env.device)
    if env.world_size > 1:
        if fsdp:
            from torch.distributed.fsdp.fully_sharded_data_parallel import (
                FullyShardedDataParallel as FSDP,
            )

            return FSDP(model)  # type: ignore[return-value]
        else:
            return torch.nn.parallel.DistributedDataParallel(  # type: ignore[return-value]
                model, device_ids=[env.device.index]
            )

    return model


transformer = s2t.get_s2t_transformer_config


def train_data(
    tokenizer: Tokenizer,
    env: Env,
    lang: str = "eng_Latn",
    batch_size: int = 16,
    data_dir: str = DATADIR,
) -> Iterable[Seq2SeqBatch]:
    return load_data_from_manifest(
        tokenizer,
        data_dir + "/train_asr.tsv",
        batch_size,
        lang,
        env,
    )


def valid_data(
    tokenizer: Tokenizer,
    env: Env,
    lang: str = "eng_Latn",
    batch_size: int = 16,
    data_dir: str = DATADIR,
) -> Iterable[Seq2SeqBatch]:
    return load_data_from_manifest(
        tokenizer, data_dir + "/dev_asr.tsv", batch_size, lang, env
    )


def _read_tsv_shard(manifest_path: str, env: Env) -> DataPipelineBuilder:
    return (
        data.text.read_text(manifest_path, rtrim=True, skip_header=1)
        # Sharding
        .shard(env.global_rank, env.world_size)
    )


def load_data_from_manifest(
    tokenizer: Tokenizer,
    manifest_path: str,
    batch_size: int,
    lang: str,
    env: Env,
) -> Iterable[Seq2SeqBatch]:
    chunk_size = 8
    pad_idx = tokenizer.vocab_info.pad_idx
    src_audio_dataloader = (
        _read_tsv_shard(manifest_path, env)
        .map(_load_audio_feats, chunk_size=chunk_size)
        .batch(batch_size, pad_idx=0)
        .and_return()
    )
    src_n_frames_dataloader = (
        _read_tsv_shard(manifest_path, env)
        .map(lambda line: torch.tensor(int(str(line).split("\t")[2])))
        .batch(batch_size)
        .prefetch(1)
        .and_return()
    )
    tgt_text_dataloader = (
        _read_tsv_shard(manifest_path, env)
        .map(lambda line: str(line).split("\t")[3])
        .map(
            tokenizer.create_encoder(mode="target", lang=lang),
            chunk_size=chunk_size,
        )
        .batch(batch_size, pad_idx=pad_idx)
        .and_return()
    )

    device = env.device
    return (
        data.zip_data_pipelines(
            [src_audio_dataloader, src_n_frames_dataloader, tgt_text_dataloader]
        )
        .prefetch(8)
        .map(
            lambda b: Seq2SeqBatch(
                # Move batch to gpu
                # TODO use a dedicated cuda stream
                source=b[0].to(device),
                src_seq_lens=b[1].to(device),
                target=b[2].to(device),
                tgt_seq_lens=(b[2][:, :-1] != pad_idx).sum(dim=-1).to(device),
            )
        )
        # Don't prefetch too much on the GPU
        .prefetch(1)
        .and_return()
    )  # type: ignore[no-any-return]


def _load_audio_feats(line: StringLike) -> Tensor:
    line = str(line)
    audio_desc = line.split("\t")[1]
    colon_parts = audio_desc.split(":")
    assert len(colon_parts) == 3
    wav = _load_audio_feats_byteoffset(colon_parts)
    # Drop channel dim
    return wav.squeeze(0)


def _load_audio_feats_byteoffset(parts: List[str]) -> Tensor:
    assert len(parts) == 3
    # We don't even need the length, it's part of the numpy header
    audio_path, byte_offset = parts[0], int(parts[1])
    with open(audio_path, "rb") as f:
        f.seek(byte_offset)
        magic_header = f.peek(8)[:8]
        # Handle precomputed audio features
        if magic_header == b"\x93NUMPY\x01\x00":
            return torch.from_numpy(np.load(f))
        wav, sample_rate = torchaudio.load(f)
        return wav  # type: ignore[no-any-return]


def vocab_info(tokenizer: Tokenizer) -> VocabularyInfo:
    """Cache metadata about the tokenizer"""
    log.info(f"vocab_info: {tokenizer.vocab_info}")
    return tokenizer.vocab_info


# TODO: move this to fairseq2.cli.defaults
hub_task = fairseq2.cli.hub_export(task, __file__)

if __name__ == "__main__":
    import fairseq2.cli.commands

    fairseq2.cli.commands.main(__file__)
