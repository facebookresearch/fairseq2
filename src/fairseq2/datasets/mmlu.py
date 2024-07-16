# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Sequence, Union, cast, final, Set, Optional
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
import os
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
from fairseq2.logging import get_log_writer
log = get_log_writer(__name__)

@dataclass
class MMLUBatch:
    # text based input and output always needs to be prepared
    input_tokens: Tensor
    output_tokens: Tensor
    answer: int
    # only prepared when speech mmlu is used
    audios: Optional[SequenceBatch] = None
    boundary_index: Optional[Tensor] = None
    speech_positions: Optional[Tensor] = None
    text_seq_lens: Optional[Tensor] = None



class MMLUDataset(ABC):
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
    ) -> Dict[str, DataPipelineReader[MMLUBatch]]:
        ...

load_mmlu_dataset = DelegatingDatasetLoader[MMLUDataset]()
# TODO: FIX, INFER
npc = 10


# TODO: Work in progress!
@final
class SpeechTextMMLUDataset(MMLUDataset):
    """Represents a generic JSONL instruction dataset."""
    _data_dir: Path

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def create_text_data_reader(
        self,
        data_path,
        tokenizer: TextTokenizer,
        gang: Gang,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        num_shots: int= 0,
    ):

        dev_file, test_file = f"{data_path}/dev.jsonl", f"{data_path}/test.jsonl"
        if not os.path.isfile(dev_file):
            # I forget to copy some of the dev to my dir, use original file
            dev_file = dev_file.replace("/fsx-ust/steventan0110/dataset/dinosr_eval/cont_mmlu_tts", "/fsx-ust/steventan0110/dataset/dinosr_eval/mmlu")

        dev_data = [] # list of dict
        with open(dev_file, "r") as dev_f:
            for line in dev_f:
                jsonl_obj = json.loads(line)
                dev_data.append(jsonl_obj)
        
        builder = list_files(data_path, pattern="*test.jsonl")
        # process text tokens and laod audios
        builder.yield_from(partial(self._read_jsonl))
        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        builder.map(partial(self.prepare_prompt, 
                            tokenizer=tokenizer, 
                            dev_data=dev_data,
                            num_shots=num_shots),
                            num_parallel_calls=npc)
        
        def to_batch(example):
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
            answr_index = mapping[example["answer"]]
            return MMLUBatch(input_tokens=example["input_tokens"].to(gang.device), 
                      output_tokens=example["output_tokens"].to(gang.device), 
                      answer=answr_index)
        
        pipeline = builder.map(to_batch).and_return()
        return DataPipelineReader[MMLUBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )
    

    def create_speech_data_reader(
        self,
        data_path,
        tokenizer: TextTokenizer,
        gang: Gang,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        num_shots: int= 0,
    ):
        builder = list_files(data_path, pattern="*test.jsonl")
        # process text tokens and laod audios
        builder.yield_from(partial(self._read_jsonl))
        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        builder.map(partial(self.prepare_speech_prompt, 
                            tokenizer=tokenizer),
                            num_parallel_calls=npc)
        
        def to_batch(example):
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
            answr_index = mapping[example["answer"]]
            audio = example["audio"].repeat(4, 1).to(gang.device)
            audio_seq_data = SequenceData(
                seqs=audio,
                is_ragged=False,
                seq_lens=(torch.ones(4) * audio.shape[1]).long().to(gang.device)
            )
            audio_seqs, audio_padding_mask = get_seqs_and_padding_mask(audio_seq_data, gang.device)
            audio_batch = SequenceBatch(audio_seqs, audio_padding_mask, example=audio_seq_data)
            boundary_index = torch.tensor(example["alignment"]) - 1
            boundary_index = boundary_index.unsqueeze(0).repeat(4, 1).to(gang.device)
            text_seq_lens = (torch.ones(4) * example["prompt_lens"]).long().to(gang.device)
            return MMLUBatch(
                input_tokens=example["input_tokens"].to(gang.device), 
                output_tokens=example["output_tokens"].to(gang.device), 
                answer=answr_index,
                audios=audio_batch,
                boundary_index=boundary_index,
                text_seq_lens=text_seq_lens
            )
        
        pipeline = builder.map(to_batch).and_return()
        return DataPipelineReader[MMLUBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )
    

    def create_speech_text_data_reader(
        self,
        data_path,
        tokenizer: TextTokenizer,
        gang: Gang,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        num_shots: int= 0,
    ):
        dev_file = f"{data_path}/dev.jsonl"
        if not os.path.isfile(dev_file):
            # I forget to copy some of the dev to my dir, use original file
            dev_file = dev_file.replace("/fsx-ust/steventan0110/dataset/dinosr_eval/cont_mmlu_tts", "/fsx-ust/steventan0110/dataset/dinosr_eval/mmlu")

        dev_data = [] # list of dict
        with open(dev_file, "r") as dev_f:
            for line in dev_f:
                jsonl_obj = json.loads(line)
                dev_data.append(jsonl_obj)
        
        builder = list_files(data_path, pattern="*test.jsonl")
        # process text tokens and laod audios
        builder.yield_from(partial(self._read_jsonl))
        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        builder.map(partial(self.prepare_speech_text_prompt, 
                            tokenizer=tokenizer,
                            dev_data=dev_data,
                            num_shots=num_shots),
                            num_parallel_calls=1,
                            )
        
        def to_batch(example):
            mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
            answr_index = mapping[example["answer"]]
            audio = example["audio"].repeat(4, 1).to(gang.device)
            audio_seq_data = SequenceData(
                seqs=audio,
                is_ragged=False,
                seq_lens=(torch.ones(4) * audio.shape[1]).long().to(gang.device)
            )
            audio_seqs, audio_padding_mask = get_seqs_and_padding_mask(audio_seq_data, gang.device)
            audio_batch = SequenceBatch(audio_seqs, audio_padding_mask, example=audio_seq_data)
            boundary_index = torch.tensor(example["alignment"]) - 1
            boundary_index = boundary_index.unsqueeze(0).repeat(4, 1).to(gang.device)
            text_seq_lens = (torch.ones(4) * example["prompt_lens"]).long().to(gang.device)
            speech_positions = example["speech_position"].unsqueeze(0).repeat(4, 1).to(gang.device)
            return MMLUBatch(
                input_tokens=example["input_tokens"].to(gang.device), 
                output_tokens=example["output_tokens"].to(gang.device), 
                answer=answr_index,
                audios=audio_batch,
                boundary_index=boundary_index,
                text_seq_lens=text_seq_lens,
                speech_positions=speech_positions
            )
        
        pipeline = builder.map(to_batch).and_return()
        return DataPipelineReader[MMLUBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=False,
            sync_batches=True,
        )
    

    @override
    def create_reader(
        self,
        tokenizer: TextTokenizer,
        gang: Gang,
        *,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        mode: str = None,
        **extras: Any,
    ) -> Dict[str, DataPipelineReader[MMLUBatch]]:
        mmlu_task = [d for d in os.listdir(self._data_dir)]
        mmlu_dataloader_dict = {}
        for task in mmlu_task:
            if mode == "text":
                # create text only dataloader
                dataloader = self.create_text_data_reader(
                    f"{self._data_dir}/{task}",
                    tokenizer,
                    gang,
                    num_accumulate,
                    num_prefetch,
                    seed,
                    num_shots = 5# hard-code to be consistent with spirit-lm baselines
                )
            elif mode == "speech":
                dataloader =  self.create_speech_data_reader(
                    f"{self._data_dir}/{task}",
                    tokenizer,
                    gang,
                    num_accumulate,
                    num_prefetch,
                    seed,
                    num_shots = 0 # not supported as there are no waveforms
                )
            elif mode == "speech_text":
                dataloader =  self.create_speech_text_data_reader(
                    f"{self._data_dir}/{task}",
                    tokenizer,
                    gang,
                    num_accumulate,
                    num_prefetch,
                    seed,
                    num_shots = 0 # not supported as there are no waveforms
                )
            else:
                raise RuntimeError(f"{self.mode} not supported!")
            mmlu_dataloader_dict[task] = dataloader
        return mmlu_dataloader_dict


    def _read_jsonl(self, path: str) -> DataPipeline:  
        def _remove_columns(data):
            del data["hubert"]
            del data["prosody"]
            del data["dataset"]
            return data
        
        # text_encoder = tokenizer.create_encoder(mode="speech_text_align")
        lines = []
        # currently we only read json file, later needs to add in the hubert file as well
        with Path(path).open() as fp:
            for line in fp:
                lines.append(line)
        builder = read_sequence(lines)
        builder.map(json.loads, num_parallel_calls=npc).map(_remove_columns, num_parallel_calls=10)
        return builder.and_return()
    
    def get_input_output_tokens(
        self,
        text: str,
        completion: str,
        tokenizer,
    ):
        x = tokenizer(text)
        len_completion = len(tokenizer(completion)) - 1 # has bos token
        y = [-100 if k < len(x) - len_completion else t.item() for k, t in enumerate(x)]
        return x[:-1].tolist(), y[1:]

    def prepare_prompt(self, input_data, tokenizer, dev_data, num_shots):
        text_tokenizer = tokenizer.create_encoder(mode="prompt")
        demo = []
        if num_shots > 0:
            assert num_shots <= len(dev_data), "num shots larger than available dev examples"
            # create demonstrations
            for i in range(num_shots):
                fewshot_example = dev_data[i]
                question = fewshot_example["question"]
                choices = fewshot_example["choices"]
                answer = fewshot_example["answer"]
                choice_str =  "\n".join([f"{l}. {choices[l]}" for l in ["A", "B", "C", "D"]])
                cur_out = f"{question}\n{choice_str}\nAnswer: {answer}"
                demo.append(cur_out)
        demos = "\n".join(demo)
        # now append the actual question and answer and prepare masking
        question = input_data["question"]
        choices = input_data["choices"]
        answer = input_data["answer"]
        choice_str =  "\n".join([f"{l}. {choices[l]}" for l in ["A", "B", "C", "D"]])
        output_prompt = demos + "\n" + f"{question}\n{choice_str}\nAnswer: "
    
        all_inputs, all_outputs = [], []
        for option in ["A", "B", "C", "D"]:
            cur_full_text = f"{output_prompt}{option}"
            input_seqs, output_seqs = self.get_input_output_tokens(cur_full_text, option, text_tokenizer)
            all_inputs.append(input_seqs)
            all_outputs.append(output_seqs)
        input_tokens = torch.tensor(all_inputs)
        output_tokens = torch.tensor(all_outputs)
        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "answer": answer}
    

    def prepare_speech_prompt(self, input_data, tokenizer):
        text_tokenizer = tokenizer.create_encoder(mode="prompt")
        # now append the actual question and answer and prepare masking
        question = input_data["question"].replace("?", "")
        choices = input_data["choices"]
        answer = input_data["answer"]
        choice_str =  " ".join([f"{l} {choices[l]}" for l in ["A", "B", "C", "D"]])
        transcript = question.lower() + " " + choice_str.lower()
        audio_path = input_data["wav_path"].replace("/data/users/yagaur/", "/fsx-onellm/benjaminmuller/speechlm/eval_dino/")
        audio, _ = torchaudio.load(audio_path)

        unity_toks, unity_duration = input_data["unity_toks"], input_data["unity_duration"]
        # obtain the alignment btw tokenized text and the unity duration
        alignment, filterd_toks = _retrieve_alignment(
            tokenizer=tokenizer.create_encoder(mode="speech_text_align"), 
            unity_toks=unity_toks,
            unity_duration=unity_duration,
            text=transcript, # already tokenized
            audio_size=audio.shape[1] # audio is 1xseq_len
        )
        # find alignment btw transcript and audio
        assert alignment is not None, input_data
        decoder = tokenizer.create_decoder()
        filtered_text = decoder(torch.tensor(filterd_toks))    
        prompt_lens = len(filterd_toks)    
        output_prompt = f"{filtered_text}\nThe answer is: "

        all_inputs, all_outputs = [], []
        for option in ["A", "B", "C", "D"]:
            cur_full_text = f"{output_prompt}{option}"
            input_seqs, output_seqs = self.get_input_output_tokens(cur_full_text, option, text_tokenizer)
            all_inputs.append(input_seqs)
            all_outputs.append(output_seqs)
        input_tokens = torch.tensor(all_inputs)
        output_tokens = torch.tensor(all_outputs)
        return {
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens, 
            "answer": answer,
            "audio": audio,
            "alignment": alignment,
            "prompt_lens": prompt_lens
        }
    
    def prepare_speech_text_prompt(self, input_data, tokenizer, dev_data, num_shots):
        text_tokenizer = tokenizer.create_encoder(mode="prompt")

        demo = []
        if num_shots > 0:
            assert len(dev_data) >= num_shots, f"Num shots {num_shots} larger than the dev size"
            for i in range(num_shots):
                fewshot_example = dev_data[i]
                question = fewshot_example["question"]
                choices = fewshot_example["choices"]
                answer = fewshot_example["answer"]
                choice_str =  "\n".join([f"{l}. {choices[l]}" for l in ["A", "B", "C", "D"]])
                cur_out = f"{question}\n{choice_str}\nAnswer: {answer}"
                demo.append(cur_out)
            demos = "\n".join(demo) + "\n"

        # now append the actual question and answer and prepare masking
        question = input_data["question"].replace("?", "")
        choices = input_data["choices"]
        answer = input_data["answer"]
        choice_str =  "\n".join([f"{l}. {choices[l]}" for l in ["A", "B", "C", "D"]])
        
        audio_path = input_data["wav_path"].replace("/data/users/yagaur/", "/fsx-onellm/benjaminmuller/speechlm/eval_dino/")
        audio, _ = torchaudio.load(audio_path)

        unity_toks, unity_duration = input_data["unity_toks"], input_data["unity_duration"]
        # obtain the alignment btw tokenized text and the unity duration
        alignment, filterd_toks = _retrieve_alignment(
            tokenizer=tokenizer.create_encoder(mode="speech_text_align"), 
            unity_toks=unity_toks,
            unity_duration=unity_duration,
            text=question.lower(), # only obtain the question part's audio
            audio_size=audio.shape[1] # audio is 1xseq_len
        )

        # find alignment btw transcript and audio
        assert alignment is not None, input_data
        decoder = tokenizer.create_decoder()
        filtered_text = decoder(torch.tensor(filterd_toks))    
        prompt_lens = len(filterd_toks)    
        output_prompt = f"{filtered_text}\n{choice_str}\nAnswer: "

        prefix_len = 1 if num_shots == 0 else len(text_tokenizer(demos))
        speech_position = torch.arange(prompt_lens) + prefix_len

        all_inputs, all_outputs = [], []
        for option in ["A", "B", "C", "D"]:
            cur_full_text = f"{output_prompt}{option}" if num_shots == 0 else f"{demos}{output_prompt}{option}"
            input_seqs, output_seqs = self.get_input_output_tokens(cur_full_text, option, text_tokenizer)
            all_inputs.append(input_seqs)
            all_outputs.append(output_seqs)
        
        input_tokens = torch.tensor(all_inputs)
        output_tokens = torch.tensor(all_outputs)
        return {
            "input_tokens": input_tokens, 
            "output_tokens": output_tokens, 
            "answer": answer,
            "audio": audio,
            "alignment": alignment,
            "prompt_lens": prompt_lens,
            "speech_position": speech_position
        }






    @override
    def splits(self) -> Set[str]:
        return {"test"}


@final
class AlignSpeechTextDatasetLoader(AbstractDatasetLoader[SpeechTextMMLUDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> SpeechTextMMLUDataset:
        return SpeechTextMMLUDataset(path)

load_speech_text_mmlu_dataset = AlignSpeechTextDatasetLoader()
load_mmlu_dataset.register(
    "speech_text_mmlu", load_speech_text_mmlu_dataset
)
