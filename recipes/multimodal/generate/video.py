# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def extract_frames(path: str | Path, num_frames: int) -> list[Image.Image]:
    """Extract evenly spaced frames from a video file.

    :param path: Path to the video file.
    :param num_frames: Number of frames to extract.
    :returns: List of PIL Images.
    """
    from decord import VideoReader

    vr = VideoReader(str(path))
    total = len(vr)

    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
    return [Image.fromarray(frame) for frame in frames]


def prepare_multimodal_messages(
    messages: list[dict[str, Any]], num_frames: int
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Rewrite messages to expand video blocks into image blocks.

    Gemma3's chat template handles image blocks but silently drops video blocks.
    This function extracts frames from videos and rewrites the content blocks
    so the model sees them as a sequence of images.

    :param messages: Chat messages with content blocks (text, image, video).
    :param num_frames: Number of frames to extract per video.
    :returns: (rewritten_messages, all_images_in_order)
    """
    all_images: list[Image.Image] = []
    rewritten: list[dict[str, Any]] = []

    for msg in messages:
        content = msg.get("content")

        # If content is a plain string, pass through unchanged.
        if isinstance(content, str):
            rewritten.append(msg)
            continue

        # Content is a list of blocks.
        new_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type")

            if block_type == "video":
                url = block.get("url", "")
                frames = extract_frames(url, num_frames)
                all_images.extend(frames)
                for _ in frames:
                    new_blocks.append({"type": "image"})

            elif block_type == "image":
                url = block.get("url")
                if url:
                    img = Image.open(url).convert("RGB")
                    all_images.append(img)
                new_blocks.append({"type": "image"})

            else:
                new_blocks.append(block)

        rewritten.append({**msg, "content": new_blocks})

    return rewritten, all_images
