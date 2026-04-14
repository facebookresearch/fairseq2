# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union, final

import numpy as np
from torch import Tensor
from decord import VideoReader, cpu, gpu  # type: ignore
import torch

from fairseq2.data.data_pipeline import ByteStreamError
from fairseq2.logging import get_log_writer
from fairseq2.typing import DataType, Device, CPU

logger = get_log_writer()

dtype_mapping = {
    torch.float32: np.float32,
    torch.bfloat16: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

device_mapping = {
    "cpu" : cpu,
    "cuda": gpu,  
}


FrameIndices = Union[Sequence[int], np.ndarray]  # type: ignore


@dataclass
class LengthBasedSegmentation:
    """Specifies the static video segmentation by the segment length"""
    num_frames_in_segment: int
    
    
@dataclass
class CountBasedSegmentation:
    num_segments: int


@dataclass
class NoSegmentation:
    pass


VideoSegmentationConfig = LengthBasedSegmentation | CountBasedSegmentation | NoSegmentation

@final
class VideoDecoder:
    
    def __init__(
        self,
        min_video_size: int = 1024,
        max_video_size: int | None = None,
        frame_step: int = 4,
        seed: int | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param min_video_size:
            Skip the videos that are too short (smaller than `min_video_size` bytes)
        :param max_video_size:
            If specified, all videos that exceed `max_video_size` bytes will be skipped
        :param frame_step:
            Step size to sample the video frames
        :param frames_per_segment:
            If specified, will segement videos to equal-sized segments, each has
            `frames_per_clip` sampled frames, 
        :param segments:
            If specified, each video will be partitioned into a fixed number of segments
        :param skip_short_segments:
            If True, segments that are too short will be skipped
        :param device:
            device to perform the reading operation.
        """        
        self._min_video_size = min_video_size
        self._max_video_size = max_video_size
        self._frame_step = frame_step

        if device:
            _index = device.index
            device_type = device_mapping[device.type]
            _device = device_type(_index)
        else:
            _device = cpu(0)
        self._device = _device
        
        self._dtype = dtype or torch.float32
        
        if seed:
            np.random.seed(seed)
    
    def __call__(
        self,
        file_path: Path,
        segments: Sequence[FrameIndices] | None = None,
    ) -> Sequence[Tensor] | None:
        """
        Read video from a file path and return a sequence of tensor reprensenting the video segments.
        This function is for random access of the frames and segments, and not for big video streams.
        
        If a list of segments are provided, the decoding will be limited to only the frames present
        in the segments. Otherwise the entire video will be decoded as one single segment
        """
        fsize = file_path.stat().st_size
        
        # Skip videos that are too short or too big
        if self._min_video_size > fsize:
            logger.info(f"Skip {file_path} due to too small size ({fsize} bytes)")
            return None
        
        if self._max_video_size is not None and self._max_video_size < fsize:
            logger.info(f"Skip {file_path} due to excessive size ({fsize} bytes)")
            return None
        
        try:
            vr = VideoReader(file_path, ctx=self._device)
            vr.seek(0)
        except Exception as ex:
            raise ByteStreamError(f"Error in reading {file_path}") from ex
        
        # We load them all frames together to save I/O
        if segments is None:
            segments = [list(range(0, len(vr), self._frame_step))]
        
        elif not segments:
            # Segments are provided but empty, this means the segmenetation algorithm does not 
            # return valid segments
            return None

        all_indices = np.concatenate(segments)
        buffer = vr.get_batch(all_indices).asnumpy()
        buffer = buffer.astype(dtype_mapping[self._dtype])
        buffer = torch.from_numpy(buffer)
        
        if self._dtype == torch.bfloat16:
            buffer = buffer.to(torch.bfloat16)
        
        buffer = buffer.to(self._device, non_blocking=True)
        
        segment_lens = [len(seg) for seg in segments]
        outputs = []
        start = 0
        for seglen in segment_lens:
            end = start + seglen
            outputs.append(buffer[start: end])
            start = end
        
        return outputs


def segment_video_by_length(
    file_path: Path, num_frames_in_segment: int, frame_step: int = 4
) -> Sequence[FrameIndices]:
    """
    Partition a video into a number of partitions of fixed sizes. The last remaining frames are abadonded
    
    :param file_path:
        path to the video file
    :param num_frames_in_segment:
        the number of sampled frames for each segment
    :param frame_step:
        step of the frames sampling in one segment
    """

    try:
        vr = VideoReader(file_path, ctx=cpu(0))
        vr.seek(0)
    except Exception as ex:
        raise ByteStreamError(f"Error in reading {file_path}") from ex

    # If there are 3 sampled frames in the segment (num_segment_frames = 3) and frame_step = 4, then
    # the segment length is 8 (Sampled frames are 0, 4, 8)
    segment_len = int((num_frames_in_segment - 1) * frame_step)
    num_segments = len(vr) // segment_len

    segments = []
    for i in range(num_segments):
        start = i * (segment_len + 1)
        end = (i + 1) * segment_len + i  # maintain segment_len and next start = previous end + 1
        indices = np.linspace(start, end, num=num_frames_in_segment)
        segments.append(indices)

    return segments


def segment_video_by_count(
    file_path: Path, num_segments: int, frame_step: int = 4
) -> Sequence[FrameIndices]:
    """Partition a video in a fixed number of segments

    :param file_path:
        path to the video file
    :param num_segments:
        number of segments
    :param frame_step:
        step of the frames sampling in one segment
    """
    
    try:
        vr = VideoReader(file_path, ctx=cpu(0))
        vr.seek(0)
    except Exception as ex:
        raise ByteStreamError(f"Error in reading {file_path}") from ex
    
    segment_len = len(vr) // num_segments
    if segment_len < frame_step:
        return []
    
    num_frames_in_segment = segment_len // frame_step + 1
    
    segments = []
    for i in range(num_segments):
        start = i * (segment_len + 1)
        end = (i + 1) * segment_len + i  # maintain segment_len and next start = previous end + 1
        indices = np.linspace(start, end, num=num_frames_in_segment)
        segments.append(indices)
    
    return segments
