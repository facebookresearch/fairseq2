# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentTypeError
from pathlib import Path
from typing import cast

from pytest import Config, Parser, Session

import tests.common
from fairseq2 import setup_extensions
from fairseq2.typing import Device


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--device",
        default="cpu",
        type=_parse_device,
        help="device on which to run tests (default: %(default)s)",
    )
    parser.addoption(
        "--integration",
        default=False,
        action="store_true",
        help="whether to run the integration tests",
    )


def pytest_sessionstart(session: Session) -> None:
    setup_extensions()

    tests.common.device = cast(Device, session.config.getoption("device"))


def pytest_ignore_collect(collection_path: Path, path: None, config: Config) -> bool:
    # Ignore integration tests unless we run `pytest --integration`.
    if "integration" in collection_path.parts:
        return not cast(bool, config.getoption("integration"))

    return False


def _parse_device(value: str) -> Device:
    try:
        return Device(value)
    except RuntimeError:
        raise ArgumentTypeError(f"'{value}' is not a valid device name.")
