from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import fairseq
import fairseq.tasks

from fairseq2.typing import Device

from . import Batch


class BilingualDataloader(Iterable[Batch]):
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
            shuffle=(split != "test"),
            pad_to_multiple=cfg.required_seq_len_multiple,
            # This is the only difference with TranslationTask.load_dataset
            prepend_bos=True,
        )
        task.datasets[split] = self.data

    def __iter__(self) -> Iterator[Batch]:
        epoch_iter = self.task.get_batch_iterator(
            self.data,
            ignore_invalid_inputs=False,
            **self.loader_kwargs,
        )
        for batch in epoch_iter.next_epoch_itr(shuffle=False):
            yield Batch(
                batch["net_input"]["src_tokens"].to(self.device),
                batch["target"].to(self.device),
                num_tokens=batch["ntokens"],
            )
