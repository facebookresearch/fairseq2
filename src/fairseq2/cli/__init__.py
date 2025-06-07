# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.cli.cli import Cli as Cli
from fairseq2.cli.cli import CliArgumentError as CliArgumentError
from fairseq2.cli.cli import CliCommand as CliCommand
from fairseq2.cli.cli import CliCommandError as CliCommandError
from fairseq2.cli.cli import CliCommandHandler as CliCommandHandler
from fairseq2.cli.cli import CliGroup as CliGroup
from fairseq2.cli.logging import setup_logging as setup_logging
from fairseq2.cli.main import main as main
from fairseq2.cli.setup import setup_cli as setup_cli
