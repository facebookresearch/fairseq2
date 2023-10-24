import collections.abc
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import soundfile
from typeguard import check_argument_types

class SoundScpWriter:
    """Writer class for 'wav.scp'

    Args:
        outdir:
        scpfile:
        format: The output audio format
        multi_columns: Save multi channel data
            as multiple monaural audio files
        output_name_format: The naming formam of generated audio files
        output_name_format_multi_columns: The naming formam of generated audio files
            when multi_columns is given
        dtype:
        subtype:

    Examples:
        >>> writer = SoundScpWriter('./data/', './data/wav.scp')
        >>> writer['aa'] = 16000, numpy_array
        >>> writer['bb'] = 16000, numpy_array

        aa ./data/aa.wav
        bb ./data/bb.wav

        >>> writer = SoundScpWriter(
            './data/', './data/feat.scp', multi_columns=True,
        )
        >>> numpy_array.shape
        (100, 2)
        >>> writer['aa'] = 16000, numpy_array

        aa ./data/aa-CH0.wav ./data/aa-CH1.wav

    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        format="wav",
        multi_columns: bool = False,
        output_name_format: str = "{key}.{audio_format}",
        output_name_format_multi_columns: str = "{key}-CH{channel}.{audio_format}",
        subtype: str = None,
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.format = format
        self.subtype = subtype
        self.output_name_format = output_name_format
        self.multi_columns = multi_columns
        self.output_name_format_multi_columns = output_name_format_multi_columns

        self.data = {}

    def __setitem__(
        self, key: str, value: Union[Tuple[int, np.ndarray], Tuple[np.ndarray, int]]
    ):
        value = list(value)
        if len(value) != 2:
            raise ValueError(f"Expecting 2 elements, but got {len(value)}")
        if isinstance(value[0], int) and isinstance(value[1], np.ndarray):
            rate, signal = value
        elif isinstance(value[1], int) and isinstance(value[0], np.ndarray):
            signal, rate = value
        else:
            raise TypeError("value shoulbe be a tuple of int and numpy.ndarray")

        if signal.ndim not in (1, 2):
            raise RuntimeError(f"Input signal must be 1 or 2 dimension: {signal.ndim}")
        if signal.ndim == 1:
            signal = signal[:, None]

        if signal.shape[1] > 1 and self.multi_columns:
            wavs = []
            for channel in range(signal.shape[1]):
                wav = self.dir / self.output_name_format_multi_columns.format(
                    key=key, audio_format=self.format, channel=channel
                )
                wav.parent.mkdir(parents=True, exist_ok=True)
                wav = str(wav)
                soundfile.write(wav, signal[:, channel], rate, subtype=self.subtype)
                wavs.append(wav)

            self.fscp.write(f"{key} {' '.join(wavs)}\n")

            # Store the file path
            self.data[key] = wavs
        else:
            wav = self.dir / self.output_name_format.format(
                key=key, audio_format=self.format
            )
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav = str(wav)
            soundfile.write(wav, signal, rate, subtype=self.subtype)
            self.fscp.write(f"{key} {wav}\n")

            # Store the file path
            self.data[key] = wav

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()