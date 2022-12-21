from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

import fairseq
import fairseq.tasks
from torch import Tensor

from fairseq2.typing import Device

from . import Seq2SeqBatch


class BilingualDataloader(Iterable[Seq2SeqBatch]):
    def __init__(
        self,
        data_folder: Path,
        src: str,
        tgt: str,
        split: str,
        device: Device,
        task_kwargs: Mapping[str, Any] = {},
        loader_kwargs: Mapping[str, Any] = {
            "max_tokens": 4096,
            "required_batch_size_multiple": 8,
            "num_workers": 1,
        },
    ):
        self.split = split
        cfg = fairseq.tasks.translation.TranslationConfig(
            data=str(data_folder),
            source_lang=src,
            target_lang=tgt,
            left_pad_target=False,
            # TODO: infer this from the file header
            dataset_impl="mmap",
            required_seq_len_multiple=1,
            **task_kwargs,
        )
        # TODO receive the dict from caller
        src_dict = fairseq.data.Dictionary.load(str(data_folder / f"dict.{src}.txt"))
        tgt_dict = fairseq.data.Dictionary.load(str(data_folder / f"dict.{tgt}.txt"))
        assert src_dict == tgt_dict
        task = fairseq.tasks.translation.TranslationTask(cfg, src_dict, tgt_dict)

        self.cfg = cfg
        self.task = task
        self.loader_kwargs = dict(loader_kwargs)
        self.device = device
        self.vocab = src_dict

        # This is equivalent to task.load_dataset(split) expect we use prepend_bos=True
        self.data = fairseq.tasks.translation.load_langpair_dataset(
            cfg.data,
            split,
            src,
            src_dict,
            tgt,
            tgt_dict,
            combine=False,
            dataset_impl=cfg.dataset_impl,
            upsample_primary=cfg.upsample_primary,
            left_pad_source=cfg.left_pad_source,
            left_pad_target=cfg.left_pad_target,
            max_source_positions=cfg.max_source_positions,
            max_target_positions=cfg.max_target_positions,
            load_alignments=cfg.load_alignments,
            truncate_source=cfg.truncate_source,
            num_buckets=cfg.num_batch_buckets,
            shuffle=(split == "train"),
            pad_to_multiple=cfg.required_seq_len_multiple,
            # This is the only difference with TranslationTask.load_dataset
            prepend_bos=True,
        )
        task.datasets[split] = self.data

        def shift_num_tokens(indices: Tensor) -> Tensor:
            """Overrides the default num_tokens_vec, to not count the added bos.

            Note that num_tokens is used for batching, so we can just fix it later
            at the batch level.
            """
            num_tokens = type(self.data).num_tokens_vec(self.data, indices)
            return num_tokens - 1  # type: ignore

        self.data.num_tokens_vec = shift_num_tokens
        self.epoch_iter = self.task.get_batch_iterator(
            self.data,
            ignore_invalid_inputs=False,
            **self.loader_kwargs,
        )

    def __iter__(self) -> Iterator[Seq2SeqBatch]:
        for batch in self.epoch_iter.next_epoch_itr():
            # Start target sentences with EOS, yeah fairseq is weird.
            batch["target"][:, 0] = self.vocab.eos_index
            yield Seq2SeqBatch(
                # Strip the BOS of source tokens
                batch["net_input"]["src_tokens"][:, 1:].to(self.device),
                batch["target"].to(self.device),
                num_tokens=batch["ntokens"] - batch["nsentences"],
            )

    def state_dict(self) -> Dict[str, Any]:
        it = self.epoch_iter
        epoch = it.epoch
        iter_in_epoch = it.iterations_in_epoch
        # Fixes bugs in fairseq/data/iterators@EpochBatchIterator.end_of_epoch()
        if it._cur_epoch_itr is not None and it._cur_epoch_itr.has_next():
            epoch += 1
            iter_in_epoch = 0

        return {
            "epoch_iter": {
                "version": 2,
                "epoch": epoch,
                "iterations_in_epoch": iter_in_epoch,
                "shuffle": it.shuffle,
            }
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.epoch_iter.load_state_dict(state_dict["epoch_iter"])
