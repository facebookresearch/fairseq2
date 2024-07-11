# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import math
from fairseq2.gang import Gang
from fairseq2.logging import LogWriter


def _reduce_num_batches(num_batches: int, gang: Gang, log: LogWriter) -> int:
    all_num_batches = torch.zeros((gang.size,), device=gang.device, dtype=torch.int64)

    num_batches_ = torch.tensor(num_batches, device=gang.device)

    gang.all_gather(all_num_batches, num_batches_)

    min_num_batches = int(all_num_batches.min())
    if min_num_batches != 0:
        return min_num_batches

    # If not all processes have reached end of data, report the ones that have
    # reached for debugging purposes.
    if log.is_enabled_for(logging.DEBUG) and all_num_batches.sum() > 0:
        ranks = all_num_batches.bool().logical_not_().nonzero().squeeze(-1).tolist()

        s = ", ".join(str(r) for r in ranks)

        log.debug("End of data reached at rank(s) {}.", s)

    return 0


def _retrieve_alignment(tokenizer, unity_toks, unity_duration, text, audio_size):
    if isinstance(text, str):
        text_toks = tokenizer(text).cpu().numpy().tolist()
    else:
        text_toks = text.cpu().numpy().tolist()
    cum_dur = 0
    dur_list = []
    for _, dur in enumerate(unity_duration):
        cum_dur += dur
        dur_list.append(cum_dur)
    words_to_find = [tokenizer.decode([tok]).lstrip() for tok in text_toks]
    cur_idx = 0
    alignment = []
    found_word = []
    output_toks = []

    for word_id, word_to_find in enumerate(words_to_find):
        if len(word_to_find) == 0:
            continue
        while True:
            if cur_idx + len(word_to_find) > len(unity_toks):
                break
            if ''.join(unity_toks[cur_idx:cur_idx+len(word_to_find)]) == word_to_find:
                # the matched character-based units are found
                dur_value = dur_list[cur_idx+len(word_to_find)]
                speech_index = math.ceil(dur_value // 2) if word_id != len(words_to_find) - 1 else dur_value // 2
                if len(alignment) > 0 and speech_index == alignment[-1]:
                    # this should not happen as each token needs to have some duration, skip the current word
                    break
                alignment.append(speech_index)
                cur_idx = cur_idx + len(word_to_find)
                found_word.append(word_to_find)
                output_toks.append(text_toks[word_id])
                break
            else:
                cur_idx += 1
 
    if len(words_to_find) != len(alignment):
        # mismatched words and alignments, but mostly due to some empty token which can be filtered
        if len(alignment) == 0:
            print("Mismatched words and alignments, no words are found! Skip this example")
            return None, None
        else:
            return alignment, output_toks
    elif alignment[-1] > audio_size // 640:
        print("Audio shorter than expected, will cause indexing error, skip this sample")
        return None, None
    else:
        return alignment, output_toks