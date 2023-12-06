# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import argparse
import logging
import os
import csv
import tempfile
from collections import defaultdict
from pathlib import Path
import torchaudio
try:
    import webrtcvad
except ImportError:
    raise ImportError("Please install py-webrtcvad: pip install webrtcvad")
import pandas as pd
from tqdm import tqdm

from fairseq2.utils.denoise_and_vad.denoiser.pretrained import master64
import fairseq2.utils.denoise_and_vad.denoiser.utils as utils
from fairseq2.utils.denoise_and_vad.vad import (
    frame_generator, vad_collector, read_wave, write_wave, FS_MS, THRESHOLD,
    SCALE
)
from fairseq2.utils.denoise_and_vad.denoiser.utils import save_df_to_tsv
from fairseq2.utils.tfgridnet.enh_inference import SeparateSpeech

log = logging.getLogger(__name__)

PATHS = ["after_denoise", "after_vad"]
MIN_T = 0.05

def generate_tmp_filename(extension="txt"):
    return tempfile._get_default_tempdir() + "/" + \
           next(tempfile._get_candidate_names()) + "." + extension


def convert_sr(inpath, sr, output_path):
    cmd = f"sox {inpath} -r {sr} {output_path}"
    os.system(cmd)
    return output_path



def apply_vad(vad, inpath):
    audio, sample_rate = read_wave(inpath)
    frames = frame_generator(FS_MS, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, FS_MS, 300, vad, frames)
    merge_segments = list()
    timestamp_start = 0.0
    timestamp_end = 0.0
    # removing start, end, and long sequences of sils
    for i, segment in enumerate(segments):
        merge_segments.append(segment[0])
        if i and timestamp_start:
            sil_duration = segment[1] - timestamp_end
            if sil_duration > THRESHOLD:
                merge_segments.append(int(THRESHOLD / SCALE) * (b'\x00'))
            else:
                merge_segments.append(int((sil_duration / SCALE)) * (b'\x00'))
        timestamp_start = segment[1]
        timestamp_end = segment[2]
    segment = b''.join(merge_segments)
    return segment, sample_rate


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr, encoding="PCM_S",
                    bits_per_sample=16)

def process(args):
    # Making sure we are requested either denoise or vad
    if not args.denoise and not args.vad:
        log.error("No denoise or vad is requested.")
        return

    log.info("Creating out directories...")
    if args.denoise:
        out_denoise = Path(args.output_dir).absolute().joinpath(PATHS[0])
        out_denoise.mkdir(parents=True, exist_ok=True)
    if args.vad:
        out_vad = Path(args.output_dir).absolute().joinpath(PATHS[1])
        out_vad.mkdir(parents=True, exist_ok=True)

    
    # Denoise
    if args.denoise:
        snr = -1
        # Load pre-trained speech enhancement model and build VAD model
        if args.model == "SeparateSpeech":

            model = SeparateSpeech(
            train_config="/Users/pradumna/Downloads/config.yaml", 
            model_file="/Users/pradumna/Downloads/5epoch.pth",
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=4,
            normalize_output_wav=True)
                 
            filename = args.audio_input
            output_path_denoise = out_denoise.joinpath(Path(f"SeperateSpeech_{filename}").name)
            waveform, sr = torchaudio.load(filename)
            waveform = waveform.to("cpu")
            estimate = model(waveform)
            estimate = torch.tensor(estimate)
            torchaudio.save(output_path_denoise, estimate[0], 16_000, encoding="PCM_S", bits_per_sample=16)
        else:
            model = master64().to(args.device)  
            # Define the audio file path
            filename = args.audio_input

            #log.info(f"Processing audio file: {audio_file_path}")

            # Process the audio file
            #filename = str(Path(audio_file_path).name)
            final_output = filename
            keep_sample = True
            # Set the output path for denoised audio
            output_path_denoise = out_denoise.joinpath(Path(f"master64_{filename}").name)
     
            # Convert to 16kHz if the sample rate is different
            tmp_path = convert_sr(final_output, 16000, "/Users/pradumna/Downloads/input.wav")
            # Load audio file and generate the enhanced version
            out, sr = torchaudio.load(tmp_path)
            out = out.to(args.device)
            estimate = model(out)
            estimate = (1 - args.dry_wet) * estimate + args.dry_wet * out
            write(estimate[0], str(output_path_denoise), sr)

            snr = utils.cal_snr(out, estimate)
            snr = snr.cpu().detach().numpy()[0][0]
            final_output = str(output_path_denoise)

        vad = webrtcvad.Vad(int(args.vad_agg_level))

         # Apply VAD
        if args.vad:
        # Set the output path for VAD
            output_path_vad = out_vad.joinpath(Path(filename).name)
    
        # Apply VAD
            segment, sample_rate = apply_vad(vad, filename)
    
        # Check length after VAD
        if len(segment) < sample_rate * MIN_T:
             keep_sample = False
             print((
            f"WARNING: skip {filename} because it is too short "
            f"after VAD ({len(segment) / sample_rate} < {MIN_T})"
             ))
        else:
            # Write VAD output
            write_wave(str(output_path_vad), segment, sample_rate)
            final_output = str(output_path_vad)

    # Output dictionary
    output_dict = {
    "id": ["id"],
    "audio": [final_output],
    "n_frames": [0],  # Set to a default value or remove if not needed
    "tgt_text": ["tgt_text"],
    "speaker": ["speaker"],
    "src_text": ["src_text"],
    "snr": [snr]
    }

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-input", "-ai", required=True,
                    type=str, help="path to the input audio file in .wav format.")
    parser.add_argument(
        "--output-dir", "-o", required=True, type=str,
        help="path to the output dir. it will contain files after denoising and"
             " vad"
    )
    parser.add_argument("--vad-agg-level", "-a", type=int, default=2,
                        help="the aggresive level of the vad [0-3].")
    parser.add_argument(
        "--dry-wet", "-dw", type=float, default=0.01,
        help="the level of linear interpolation between noisy and enhanced "
             "files."
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cpu",
        help="the device to be used for the speech enhancement model: "
             "cpu | cuda."
    )
    parser.add_argument(
        "--model", "-m", type=str, default="master64",
        help="the speech enhancement model to be used: master64 | SeparateSpeech."
    )
    parser.add_argument("--denoise", action="store_true",
                        help="apply a denoising")
    parser.add_argument("--vad", action="store_true", help="apply a VAD")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()