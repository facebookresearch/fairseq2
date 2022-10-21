import dataclasses
import functools
import logging
import math
import os
import typing as tp
from pathlib import Path

import datasets  # type: ignore
import fairscale  # type: ignore
import sentencepiece  # type: ignore
import torch
import torch.nn.functional as F
import torchtnt.runner  # type: ignore
import torchtnt.utils  # type: ignore
import wandb  # type: ignore
from fairscale.nn.model_parallel.layers import (  # type: ignore
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import Tensor
from torcheval.metrics import Mean  # type: ignore
from torchtnt.runner.state import State  # type: ignore
from torchtnt.runner.unit import EvalUnit, TrainUnit  # type: ignore

from fairseq2.nn import (
    Embedding,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TiedProjection,
)
from fairseq2.nn.transformer import (
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformer,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclasses.dataclass
class ModelConfig:
    model_dim: int
    ffn_inner_dim: int
    num_attn_heads: int
    dropout_p: float
    attn_dropout_p: float
    num_layers: int = 6


class Batch(tp.NamedTuple):
    source: torch.Tensor
    target: torch.Tensor


class MachineTranslationTask(TrainUnit[Batch], EvalUnit[Batch]):
    def __init__(self, cfg: ModelConfig, vocab_size: int, device: str):
        super().__init__()
        # initialize module & optimizer

        self.module = build_model(cfg, vocab_size, device, torch.float32)
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9
        )

        self.train_loss = Mean(device=device)
        self.eval_loss = Mean(device=device)
        self.log_frequency_steps = 1
        self.lr_frequency_steps = 10_000
        # TODO: add wandb to state
        self.wandb = wandb.init()

    def train_step(self, state: State, data: Batch) -> None:
        loss = self._nll_loss(data)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # update metrics & logs
        self.train_loss.update(loss)

        steps = state.train_state.progress.num_steps_completed + 1
        if steps % self.log_frequency_steps == 0:
            self._flush_loss(state, "train", self.train_loss)
        if steps % self.lr_frequency_steps == 0:
            self.lr_scheduler.step()

    def _nll_loss(self, data: Batch) -> Tensor:
        net_output = self.module.forward(data.source, data.target)
        probs = F.log_softmax(net_output, dim=-1).transpose(2, 1)
        # TODO: use masking and normalize with true number of tokens
        loss = F.nll_loss(probs, data.target, reduction="mean")
        return loss

    def _flush_loss(self, state: State, prefix: str, metric: tp.Any) -> None:
        step = state.train_state.progress.num_steps_completed
        loss = metric.compute()
        metric.reset()
        ppl = 10 * math.log10(loss)
        # TODO use tnt.logger
        metrics = {f"{prefix}/loss": loss, f"{prefix}/ppl": ppl}
        if prefix == "train":
            metrics[f"{prefix}/lr"] = self.lr_scheduler.get_last_lr()[0]
        self.wandb.log(metrics, step)

    def on_eval_start(self, state: State) -> None:
        pass

    def on_eval_end(self, state: State) -> None:
        self._flush_loss(state, "eval", self.eval_loss)

    @torch.inference_mode()
    def eval_step(self, state: State, data: Batch) -> None:
        self.eval_loss.update(self._nll_loss(data))


def load_embeddings(
    cfg: ModelConfig, vocab_size: int, device: str
) -> ParallelEmbedding:
    init = functools.partial(torch.nn.init.uniform_, a=-0.05, b=0.05)
    embs = ParallelEmbedding(vocab_size, cfg.model_dim, init_method=init)
    if device:
        embs = embs.to(device)
    return embs


def build_model(
    cfg: ModelConfig, vocab_size: int, device: str, dtype: torch.dtype
) -> Transformer:
    """Builds a Transformer model as described in the original paper.

    In fairseq2 models are constructed by composing modules as building blocks.
    This follows the dependency inversion principle, which means instead of a
    model being responsible for instantiating its submodules, it expects them to
    be provided by the user. This avoids having to subclass or copy/edit entire
    model architectures, and gives a chance to modify the behavior of a model at
    a much granular level.
    """
    embed = load_embeddings(cfg, vocab_size, device)

    positional_embed = SinusoidalPositionalEmbedding(
        max_seq_len=4096, embedding_dim=cfg.model_dim, device=device
    )

    encoder = build_encoder(cfg, embed, positional_embed, device, dtype)

    decoder = build_decoder(cfg, embed, positional_embed, device, dtype)

    # Share the weight matrix between the embedding layers and the pre-softmax
    # score projection as described in the original paper.
    score_proj = TiedProjection(embed.weight)

    return StandardTransformer(encoder, decoder, score_proj, use_log_softmax=True)


def build_encoder(
    cfg: ModelConfig,
    embed: ParallelEmbedding,
    positional_embed: PositionalEmbedding,
    device: tp.Any,
    dtype: tp.Any,
) -> TransformerEncoder:
    layers = []

    for i in range(cfg.num_layers):
        layers.append(build_encoder_layer(cfg, device, dtype))

    return StandardTransformerEncoder(
        embed,
        positional_embed,
        layers,
        embed_dropout_p=cfg.dropout_p,
        device=device,
        dtype=dtype,
    )


def build_encoder_layer(
    cfg: ModelConfig, device: tp.Any, dtype: tp.Any
) -> TransformerEncoderLayer:
    self_attn = build_attn(cfg, device, dtype)
    ffn = StandardFeedForwardNetwork(
        model_dim=cfg.model_dim,
        inner_dim=cfg.ffn_inner_dim,
        device=device,
        dtype=dtype,
    )
    return StandardTransformerEncoderLayer(
        self_attn,
        ffn,
        dropout_p=cfg.dropout_p,
        device=device,
        dtype=dtype,
    )


def build_attn(
    cfg: ModelConfig, device: tp.Any, dtype: tp.Any
) -> StandardMultiheadAttention:
    assert (
        cfg.model_dim % cfg.num_attn_heads == 0
    ), "Can't devide model_dim with num_attn_heads !"

    init = functools.partial(torch.nn.init.xavier_uniform_, gain=2**-0.5)
    q_proj = ColumnParallelLinear(
        cfg.model_dim, cfg.model_dim, bias=False, gather_output=False, init_method=init
    )
    k_proj = ColumnParallelLinear(
        cfg.model_dim, cfg.model_dim, bias=False, gather_output=False, init_method=init
    )
    v_proj = ColumnParallelLinear(
        cfg.model_dim, cfg.model_dim, bias=False, gather_output=False, init_method=init
    )
    out_proj = RowParallelLinear(
        cfg.model_dim,
        cfg.model_dim,
        bias=False,
        input_is_parallel=True,
        init_method=torch.nn.init.xavier_uniform_,
    )

    mha = StandardMultiheadAttention(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        out_proj=out_proj,
        model_dim=cfg.model_dim,
        num_heads=cfg.num_attn_heads,
        attn_dropout_p=cfg.attn_dropout_p,
        device=device,
        dtype=dtype,
    )
    mha.to(device)
    return mha


def build_decoder(
    cfg: ModelConfig,
    embed: Embedding,
    positional_embed: PositionalEmbedding,
    device: tp.Any,
    dtype: tp.Any,
) -> TransformerDecoder:
    return StandardTransformerDecoder(
        embed,
        positional_embed,
        [build_decoder_layer(cfg, device, dtype) for _ in range(cfg.num_layers)],
        embed_dropout_p=cfg.dropout_p,
        device=device,
        dtype=dtype,
    )


def build_decoder_layer(
    cfg: ModelConfig, device: tp.Any, dtype: tp.Any
) -> TransformerDecoderLayer:
    # Teaser: the next example will mix MoE and distributed decoder layers for
    # demonstration purposes (e.g. ShardedFeedForwardNetwork)

    self_attn = build_attn(cfg, device=device, dtype=dtype)
    enc_dec_attn = build_attn(cfg, device=device, dtype=dtype)

    ffn = StandardFeedForwardNetwork(
        model_dim=cfg.model_dim,
        inner_dim=cfg.ffn_inner_dim,
        device=device,
        dtype=dtype,
    )

    return StandardTransformerDecoderLayer(
        self_attn,
        enc_dec_attn,
        ffn,
        dropout_p=cfg.dropout_p,
        device=device,
        dtype=dtype,
    )


def get_config_for_big_variant() -> ModelConfig:
    return ModelConfig(
        model_dim=512,
        ffn_inner_dim=4096,
        num_attn_heads=16,
        dropout_p=0.3,
        attn_dropout_p=0.3,
    )


def _finalize_batch(values: list, device: str) -> Batch:
    # TODO: add batch statistic
    return Batch(
        source=_make_batch([src for src, tgt in values]).to(device),
        target=_make_batch([tgt for src, tgt in values]).to(device),
    )


def _make_batch(
    values: list,
    pad_idx: int = 0,
    pad_to_length: int = 0,
    pad_to_multiple: int = 1,
    batch_size: int = 0,
    left_pad: bool = False,
    dtype: torch.dtype = torch.long,
) -> Tensor:
    """Convert a list of token-index list into a padded 2d tensor.
    Note: eos/bos are supposed to be already added.
    """
    size = max(len(v) for v in values)
    size = max(size, pad_to_length)
    if size % pad_to_multiple != 0:
        size = (size - size % pad_to_multiple) + pad_to_multiple

    batch_size = max(len(values), batch_size)
    res = torch.zeros((size, batch_size), dtype=dtype).fill_(pad_idx)

    for i, v in enumerate(values):
        if left_pad:
            res[size - len(v) :, i] = torch.tensor(v, dtype=dtype)
        else:
            res[: len(v), i] = torch.tensor(v, dtype=dtype)
    return res


class DatasetLoader(tp.Iterable[Batch]):
    def __init__(
        self,
        train: bool,
        src: str,
        tgt: str,
        spm: sentencepiece.SentencePieceProcessor,
        batch_size: int,
        global_rank: int,
        world_size: int,
        device: str,
    ):
        self.src = src
        self.tgt = tgt
        self.spm = spm
        self.batch_size = batch_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = device

        if train:
            try:
                data = datasets.load_dataset(
                    "allenai/nllb", f"{src}-{tgt}", save_infos=True, streaming=True
                )
            except Exception:
                data = datasets.load_dataset(
                    "allenai/nllb", f"{tgt}-{src}", save_infos=True, streaming=True
                )
            self.data = data["train"]
            self.extract_src = lambda sample: sample["translation"][src]
            self.extract_tgt = lambda sample: sample["translation"][tgt]
        else:
            data = datasets.load_dataset("facebook/flores", f"{src}-{tgt}")
            self.data = data["dev"]
            self.extract_src = lambda sample: sample["sentence_" + src]
            self.extract_tgt = lambda sample: sample["sentence_" + tgt]

    def __iter__(self) -> tp.Iterator[Batch]:

        batch = []
        for i, sample in enumerate(self.data):
            if i % self.world_size != self.global_rank:
                continue

            src_ids = self.spm.encode_as_ids(self.extract_src(sample))
            tgt_ids = self.spm.encode_as_ids(self.extract_tgt(sample))
            batch.append((src_ids, tgt_ids))
            if len(batch) == self.batch_size:
                yield _finalize_batch(batch, self.device)
                batch = []
        if batch:
            yield _finalize_batch(batch, self.device)


def init_env() -> None:
    local_rank = os.environ.get("LOCAL_RANK", 0)
    rank = os.environ.get("RANK", 0)
    group_rank = os.environ.get("GROUP_RANK", 0)
    world_size = os.environ.get("WORLD_SIZE", 1)
    master_port = os.environ.get("MASTER_PORT", 2000)
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    log.info(
        f"LOCAL_RANK: {local_rank}\n"
        f"RANK: {rank}\n"
        f"GROUP_RANK: {group_rank}\n"
        f"WORLD_SIZE: {world_size}"
    )
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["GROUP_RANK"] = str(group_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ.update(os.environ)


def main() -> None:
    import bdb
    import pdb

    try:
        init_env()
        device = torchtnt.utils.init_from_env()
        # Prevent double init when running in REPL
        if not fairscale.nn.model_parallel.initialize.model_parallel_is_initialized():
            fairscale.nn.model_parallel.initialize_model_parallel(1)
        spm_path = Path("/private/home/kevinheffernan/nllb.200.models/laser2.spm")
        spm = sentencepiece.SentencePieceProcessor()
        spm.load(str(spm_path))
        vocab_size = spm.GetPieceSize() + 2
        task = MachineTranslationTask(
            get_config_for_big_variant(), vocab_size, "cuda:0"
        )
        train_dataloader = DatasetLoader(
            train=True,
            src="eng_Latn",
            tgt="fra_Latn",
            spm=spm,
            batch_size=16,
            global_rank=int(os.environ.get("GROUP_RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            device=device,
        )
        valid_dataloader = DatasetLoader(
            train=False,
            src="eng_Latn",
            tgt="fra_Latn",
            spm=spm,
            batch_size=16,
            global_rank=int(os.environ.get("GROUP_RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            device=device,
        )

        torchtnt.runner.fit(
            task,
            train_dataloader,
            valid_dataloader,
            max_epochs=2**30,
            max_steps=1_000_000,
            evaluate_every_n_steps=1000,
        )
    except bdb.BdbQuit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        pdb.post_mortem()


if __name__ == "__main__":
    main()
