#!/usr/bin/env python3
"""Gemma3n audio transcription using fairseq2.

Usage:
    python scripts/gemma3n_validation/infer_audio.py --audio mono-16k.wav
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fairseq2.data._memory import MemoryBlock
from fairseq2.data.audio import AudioDecoder
from fairseq2.models.gemma3n.hub import (
    get_gemma3n_model_hub,
    get_gemma3n_tokenizer_hub,
)
from fairseq2.nn import BatchLayout

SAMPLE_RATE = 16_000
AUDIO_TOKEN_ID = 262273
NUM_AUDIO_TOKENS = 188


# -- mel extraction (matches HF Gemma3nAudioFeatureExtractor) ----------------


def _create_mel_filterbank(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    fft_length: int,
) -> np.ndarray:
    all_freqs = np.arange(n_freqs, dtype=np.float32) * (
        sample_rate / fft_length
    )
    m_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = np.expand_dims(f_pts, 0) - np.expand_dims(all_freqs, 1)
    zero = np.zeros(1, dtype=np.float32)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return np.maximum(zero, np.minimum(down_slopes, up_slopes))


def extract_mel_features(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16_000,
    n_mels: int = 128,
    frame_length_ms: float = 32.0,
    hop_length_ms: float = 10.0,
    f_min: float = 125.0,
    f_max: float = 7600.0,
    preemphasis: float = 0.97,
    mel_floor: float = 1e-5,
) -> np.ndarray:
    """Extract log-mel spectrogram matching Gemma3n audio feature extractor.

    :param waveform: Raw audio samples, shape ``(T,)`` or ``(1, T)``.
    :returns: Log-mel features, shape ``(num_frames, n_mels)``.
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]

    frame_length = int(round(sample_rate * frame_length_ms / 1000.0))
    hop_length = int(round(sample_rate * hop_length_ms / 1000.0))
    fft_length = 2 ** math.ceil(math.log2(frame_length)) * 2  # overdrive

    hann = np.arange(frame_length, dtype=np.float32)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * hann / frame_length))

    mel_filters = _create_mel_filterbank(
        n_freqs=fft_length // 2 + 1,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        sample_rate=sample_rate,
        fft_length=fft_length,
    )

    frame_size = frame_length + 1
    n_samples = waveform.shape[1]
    n_frames = (n_samples - frame_size) // hop_length + 1

    strides = (
        waveform.strides[0],
        waveform.strides[1] * hop_length,
        waveform.strides[1],
    )
    frames = np.lib.stride_tricks.as_strided(
        waveform, shape=(1, n_frames, frame_size), strides=strides
    )

    # HTK preemphasis
    first = frames[..., :1] * (1.0 - preemphasis)
    rest = frames[..., 1:-1] - preemphasis * frames[..., :-2]
    frames_pe = np.concatenate([first, rest], axis=-1)

    frames_w = frames_pe * window
    stft = np.fft.rfft(frames_w, n=fft_length, axis=-1)
    mag = np.abs(stft)
    mel = mag @ mel_filters
    log_mel = np.log(np.maximum(mel, mel_floor))

    return log_mel.squeeze(0)


# -- tokenization ------------------------------------------------------------


def build_audio_input_ids(
    prompt: str,
    tokenizer: object,
    device: torch.device,
) -> Tensor:
    """Build token sequence with audio placeholders.

    :returns: Token IDs, shape ``(1, S)``.
    """
    conversation = [
        {"role": "user", "content": f"<audio_placeholder>\n{prompt}"},
    ]

    text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    placeholder_ids = tokenizer.encode(
        "<audio_placeholder>", add_special_tokens=False
    )

    placeholder_len = len(placeholder_ids)
    pos = -1
    for i in range(len(token_ids) - placeholder_len + 1):
        if token_ids[i : i + placeholder_len] == placeholder_ids:
            pos = i
            break

    if pos < 0:
        raise RuntimeError("Could not find placeholder in tokenized sequence")

    audio_tokens = torch.full(
        (NUM_AUDIO_TOKENS,), AUDIO_TOKEN_ID, dtype=torch.long
    )
    token_ids_t = torch.tensor(token_ids, dtype=torch.long)
    token_ids_t = torch.cat(
        [token_ids_t[:pos], audio_tokens, token_ids_t[pos + placeholder_len :]],
    )

    return token_ids_t.unsqueeze(0).to(device)


# -- inference ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Gemma3n (fairseq2)"
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to 16 kHz mono WAV"
    )
    parser.add_argument(
        "--prompt", type=str, default="Transcribe this audio."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma3n_e2b_instruct",
        help="fairseq2 model name (from asset cards)",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # -- load audio & extract mel features -----------------------------------
    print(f"Loading audio from {audio_path}...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    decoded = AudioDecoder(dtype=torch.float32)(MemoryBlock(audio_bytes))
    waveform = decoded["waveform"]
    sr = int(decoded["sample_rate"])

    if sr != SAMPLE_RATE:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    print(f"  Duration: {waveform.shape[-1] / SAMPLE_RATE:.2f}s")

    mel = extract_mel_features(waveform.numpy())
    mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(device)
    print(f"  Mel features: {mel_tensor.shape}")

    # -- load model + tokenizer via fairseq2 hub -----------------------------
    print(f"\nLoading model '{args.model_name}'...")
    model_hub = get_gemma3n_model_hub()
    model = model_hub.load_model(
        args.model_name, device=device, dtype=torch.float32
    )
    model.eval()
    print("  Model loaded")

    print("Loading tokenizer...")
    tok_hub = get_gemma3n_tokenizer_hub()
    tokenizer = tok_hub.load_tokenizer(args.model_name)
    eos_id = tokenizer.vocab_info.eos_idx or 1
    print("  Tokenizer loaded")

    # -- tokenize prompt with audio placeholders -----------------------------
    input_ids = build_audio_input_ids(args.prompt, tokenizer, device)
    print(f"  Input tokens: {input_ids.shape[1]}")

    # -- generate ------------------------------------------------------------
    print(f"\nGenerating (max {args.max_tokens} tokens)...")
    generated = input_ids.clone()

    with torch.no_grad():
        for step in range(args.max_tokens):
            layout = BatchLayout(
                generated.shape, seq_lens=[generated.size(1)]
            )
            logits = model(
                generated, layout, audio_features=mel_tensor
            )
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if next_id.item() == eos_id:
                print(f"  EOS after {step + 1} tokens")
                break

            generated = torch.cat([generated, next_id], dim=1)

            if (step + 1) % 20 == 0:
                print(f"  {step + 1} tokens...")

    # -- decode output -------------------------------------------------------
    tok_decoder = tokenizer.create_decoder(skip_special_tokens=True)
    output_text = tok_decoder(generated[0].cpu())

    print(f"\n{'=' * 60}")
    print(output_text)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
