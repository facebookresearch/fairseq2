# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Sequence, Union, cast, final, Set
from functools import partial

import torch
from typing_extensions import NoReturn
from torch import Tensor
from fairseq2.assets import AssetCard, AssetError
from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    SequenceData,
    create_bucket_sizes,
    read_sequence,
    list_files
)
from fairseq2.datasets.utils import _retrieve_alignment
import torchaudio
from fairseq2.data.text import TextTokenizer
from fairseq2.datasets.batching import LengthBatching, StaticBatching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import override
from dataclasses import dataclass

@dataclass
class SpeechTextAlignBatch:
    audios: SequenceBatch
    text_tokens: SequenceBatch
    boundary_index: Tensor
    # blockwise_attn_mask: Tensor



class SpeechTextDataset(ABC):
    @abstractmethod
    def create_reader(
        self,
        split: str,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        min_seq_len: int,
        *,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataPipelineReader[SpeechTextAlignBatch]:
        ...

load_speech_text_dataset = DelegatingDatasetLoader[SpeechTextDataset]()
# TODO: FIX, INFER
npc = 10


# TODO: Work in progress!
@final
class AlignSpeechTextDataset(SpeechTextDataset):
    """Represents a generic JSONL instruction dataset."""
    _data_dir: Path
    load_librispeech: bool=False

    def __init__(self, data_dir: Path, load_librispeech=False) -> None:
        self._data_dir = data_dir
        self.load_librispeech = load_librispeech


    @override
    def create_reader(
        self,
        tokenizer: TextTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        min_seq_len: int,
        *,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        **extras: Any,
    ) -> DataPipelineReader[SpeechTextAlignBatch]:
        if self.load_librispeech:
            builder = list_files(self._data_dir, pattern="*test_small.jsonl")
        else:
            builder = list_files(self._data_dir, pattern="*.jsonl")
        # Shuffle files. Must be consistent across all processes.
        if example_shuffle_window != 1:
            builder.shuffle(shuffle_window=0, seed=seed)

        seed += 1

        # process text tokens and laod audios
        builder.yield_from(partial(self._read_jsonl, tokenizer=tokenizer))
    
        # Shuffle examples.
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed=seed)

        seed += 1

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        bucket_sizes = create_bucket_sizes(
            max_num_elements=max_num_tokens,
            max_seq_len=max_seq_len,
        )

        builder.bucket_by_length(
            bucket_sizes,
            selector="text",
            min_data_len=min_seq_len,
            skip_above_max_examples=True,
            skip_below_min_examples=True,
        )

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed=seed)

        seed += 1

        builder.map(partial(self._process_alignment, tokenizer=tokenizer), num_parallel_calls=npc)
     
        # Collate bucketed examples into a batch.
        override_opts = [
            CollateOptionsOverride("alignment", pad_value=0),
            CollateOptionsOverride("file_id", pad_value=1, pad_to_multiple=2),
        ]
        collater = Collater(pad_value=tokenizer.vocab_info.pad_idx or 0, overrides=override_opts)

        builder.map(collater, num_parallel_calls=npc)
        # Prefetch `num_prefetch` batches in b
        builder.prefetch(num_prefetch)

        # Wrap examples with `SequenceBatch`.
        def to_batch(example: Dict[str, Any]) -> SpeechTextAlignBatch:
            token_seqs, token_padding_mask = get_seqs_and_padding_mask(cast(SequenceData, example["text_tok"]), gang.device)
            # if example["file_id"]["seq_lens"].max().item() % 640 == 0:
            #     # if divisible by 640, the audio repr will be 1 length smaller, 
            #     # so we manually append a padding token to avoid out-of-index error
            #     tmp_pad = example["file_id"]["seqs"].new_full((example["file_id"]["seqs"].shape[0], 100), 1.0)
            #     example["file_id"]["seqs"] = torch.cat((example["file_id"]["seqs"], tmp_pad), dim=1)
            #     # example["file_id"]["seq_lens"] += 1
            audio_seqs, audio_padding_mask = get_seqs_and_padding_mask(cast(SequenceData, example["file_id"]), gang.device)
            audio_batch = SequenceBatch(audio_seqs, audio_padding_mask, example=example["file_id"])
            token_batch =  SequenceBatch(token_seqs, token_padding_mask, example=example["text_tok"])
            
            ################# prepare attention mask ######################
            # repr_lens = torch.floor(example["file_id"]["seq_lens"] / 640).long()
            # token_lens = example["text_tok"]["seq_lens"]
            # max_seq_len = repr_lens.max().item()
            
            # batch_size = repr_lens.shape[0]
            # attention_map = audio_seqs.new_full((max_seq_len, max_seq_len), 1).long()
            # attention_map.tril_(diagonal=0)
            # # bz x max_seq_len x max_seq_len
            # # Note that we have to use repeat rather then expand to assign memory!!
            # attention_map = attention_map.unsqueeze(0).repeat(batch_size, 1, 1)
            # before_attn_time = time.time()

            # for i in range(batch_size):
            #     # print(boundary_index[i, :])
            #     cur_repr_len = repr_lens[i]
            #     if cur_repr_len < max_seq_len:
            #         # this is actually not necessary since the padding mask inside self-attention will achieve the same thing
            #         attention_map[i, :, cur_repr_len:] = 0
                
            #     cur_token_len = token_lens[i]
            #     if boundary_index[i, cur_token_len-1] >= cur_repr_len - 1:
            #         boundary_index[i, cur_token_len-1] = cur_repr_len - 1
            #     # create blockwise masks
            #     # TODO: optimize this code. this double for-loop is slow especially if number of boundary index is large
            #     for boundary_id in range(cur_token_len):
            #         boundary_column = boundary_index[i, boundary_id]
            #         if boundary_column + 1 < max_seq_len:
            #             # we could use :boundary_column + 1 as well, but I want it to be able to 
            #             # attend to 1 more repr before in case of bad alignment
            #             # we can also adjust this window size to allow for more lenient window
            #             attention_map[i, boundary_column+1:, :boundary_column] = 0
            # end_time = time.time()
            # print(f"Batching before attn {before_attn_time-start_time}, after attn: {end_time-before_attn_time}")
            boundary_index = example["alignment"]["seqs"] - 1
            return SpeechTextAlignBatch(
                audios=audio_batch, 
                text_tokens=token_batch, 
                boundary_index=boundary_index,
                # blockwise_attn_mask=attention_map   
            )
        
        pipeline = builder.map(to_batch).and_return()
        return DataPipelineReader[SpeechTextAlignBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )

        

    def _read_jsonl(self, path: str, tokenizer: TextTokenizer) -> DataPipeline:
        def _lower_case_text(x):
            return x.lower()
            
        text_encoder = tokenizer.create_encoder(mode="speech_text_align")
        lines = []
        # currently we only read json file, later needs to add in the hubert file as well
        with Path(path).open() as fp:
            for line in fp:
                lines.append(line)
        builder = read_sequence(lines)
        builder.map(json.loads, num_parallel_calls=npc)\
            .map((lambda x: x.lower()), selector="text", num_parallel_calls=npc)\
            .map(text_encoder, selector="text", num_parallel_calls=npc)
        return builder.and_return()
    

    def _process_alignment(self, input_data, tokenizer):
        def obtain_audio_file(file_id):
            if self.load_librispeech:
                audio_file = file_id
            else:
                file_prefix = file_id.split("/")[-1]
                subdirs = file_prefix.split("_")
                audio_file = f"/datasets01/mls/mls_english/train/audio/{subdirs[0]}/{subdirs[1]}/{file_prefix}.flac"
            # return audio_file
            try:
                wav, sr = torchaudio.load(audio_file)
            except Exception:
                wav = None
            return wav
        
        def process_alignment(text_encoder, data):
            """
            retrieve tokens and alignment information from unity_duration
            prepare alignment array as well audios
            """
            if not self.load_librispeech:
                del data["time"]
            unity_toks, unity_duration = data["unity_toks"], data["unity_duration"]
            # obtain the alignment btw tokenized text and the unity duration
            alignment, filterd_toks = _retrieve_alignment(
                tokenizer=text_encoder, 
                unity_toks=unity_toks,
                unity_duration=unity_duration,
                text=data["text"], # already tokenized
                audio_size=data["file_id"].shape[0] # audio is 1xseq_len
            )
            if alignment is None:
                return None

            data["alignment"] = torch.tensor(alignment)
            data["text_tok"] = torch.tensor(filterd_toks)
            data["text"] = text_encoder.decode(filterd_toks)
            del data["unity_toks"]
            del data["unity_duration"]
            return data
        # import time
        # prev_time = time.time()
        
        # print(input_data)
        # audio_load_time, alignment_time = 0, 0
        text_encoder = tokenizer.create_encoder(mode="speech_text_align")
        output_data = []
        for item in input_data:
            if self.load_librispeech:
                wav = obtain_audio_file(item["audio_file"])
            else:
                wav = obtain_audio_file(item["file_id"])
            # audio_load_time += time.time() - prev_time
            # prev_time = time.time() 
            if wav is not None:
                item["file_id"] = wav.squeeze(0)
                item = process_alignment(text_encoder=text_encoder, data=item)
                if item is not None:
                    output_data.append(item)
            # alignment_time += time.time() - prev_time
            # prev_time = time.time()

        # print(f'forloop load_time: {audio_load_time}, alignment time : {alignment_time}')
        return output_data


    @override
    def splits(self) -> Set[str]:
        return {"train"}


@final
class AlignSpeechTextDatasetLoader(AbstractDatasetLoader[AlignSpeechTextDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> AlignSpeechTextDataset:
        return AlignSpeechTextDataset(path)

load_align_speech_text_dataset = AlignSpeechTextDatasetLoader()
load_speech_text_dataset.register(
    "align_speech_text", load_align_speech_text_dataset
)



@final
class AlignSpeechTextLibrispeechDatasetLoader(AbstractDatasetLoader[AlignSpeechTextDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> AlignSpeechTextDataset:
        return AlignSpeechTextDataset(path, load_librispeech=True)
    
load_align_speech_text_librispeech_dataset = AlignSpeechTextLibrispeechDatasetLoader()
load_speech_text_dataset.register(
    "align_speech_text_librispeech", load_align_speech_text_librispeech_dataset
)
