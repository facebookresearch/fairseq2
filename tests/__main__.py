# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
from unittest.loader import defaultTestLoader
from unittest.runner import TextTestRunner

import torch

import tests
from tests.common import TestCase


def parse_device_arg(value: str) -> torch.device:
    try:
        return torch.device(value)
    except RuntimeError:
        raise ArgumentTypeError(f"'{value}' is not a valid device name.")


def parse_args() -> Namespace:
    parser = ArgumentParser(prog=tests.__name__, description="Runs fairseq2 tests.")

    parser.add_argument(
        "-d",
        "--device",
        default="cpu",
        type=parse_device_arg,
        help="device on which to run tests (default: %(default)s)",
    )

    parser.add_argument(
        "--locals",
        dest="tb_locals",
        action="store_true",
        help="show local variables in tracebacks",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="store_const",
        const=2,
        default=0,
        help="increase output verbosity",
    )

    return parser.parse_args()


def run_tests(verbosity: int, tb_locals: bool) -> bool:
    start_dir = Path(__file__).parent.parent

    test_suite = defaultTestLoader.discover(str(start_dir))

    runner = TextTestRunner(verbosity=verbosity, tb_locals=tb_locals)

    return runner.run(test_suite).wasSuccessful()


args = parse_args()

TestCase.device = args.device

succeeded = run_tests(args.verbosity, args.tb_locals)

sys.exit(0 if succeeded else 1)
