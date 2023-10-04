# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from argparse import ArgumentTypeError
from pathlib import Path
from typing import cast

import pytest
from packaging.version import Version

import tests.common
from fairseq2.typing import Device


def pytest_addoption(parser: pytest.Parser) -> None:
    # fmt: off
    parser.addoption(
        "--device", default="cpu", type=parse_device_arg,
        help="device on which to run tests (default: %(default)s)",
    )
    parser.addoption(
        "--integration", default=False, action="store_true",
        help="whether to run the integration tests",
    )
    # fmt: on


def parse_device_arg(value: str) -> Device:
    try:
        return Device(value)
    except RuntimeError:
        raise ArgumentTypeError(f"'{value}' is not a valid device name.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "fairseq2n(version): mark test to run only on the specified fairseq2n version or greater",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    tests.common.device = cast(Device, session.config.getoption("device"))


def pytest_ignore_collect(
    collection_path: Path, path: None, config: pytest.Config
) -> bool:
    if "integration" in collection_path.parts:
        # Ignore integration tests unless we run `pytest --integration`.
        return not cast(bool, config.getoption("integration"))

    return False


def pytest_runtest_setup(item: pytest.Function) -> None:
    marker = item.get_closest_marker(name="fairseq2n")
    if marker is not None:
        skip_if_fairseq2n_newer(*marker.args, **marker.kwargs)


def skip_if_fairseq2n_newer(version: str) -> None:
    import fairseq2n

    installed_version = Version(fairseq2n.__version__)
    annotated_version = Version(version)

    # fmt: off
    if installed_version < annotated_version:
        pytest.skip(f"The test requires fairseq2n v{annotated_version} or greater.")
    elif (
        installed_version.major != annotated_version.major or
        installed_version.minor != annotated_version.minor
    ):
        warnings.warn(
            f"The test requires fairseq2n v{annotated_version} which is older than the current version (v{installed_version}). The marker can be safely removed."
        )
    # fmt: on
