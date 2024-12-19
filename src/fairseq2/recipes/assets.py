# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Any, cast, final

from rich.console import Console
from rich.pretty import pretty_repr
from typing_extensions import override

from fairseq2.assets import (
    AssetCard,
    AssetNotFoundError,
    AssetStore,
    default_asset_store,
)
from fairseq2.datasets import is_dataset_card
from fairseq2.logging import get_log_writer
from fairseq2.models import is_model_card
from fairseq2.recipes.cli import Cli, CliCommandHandler
from fairseq2.recipes.console import get_console
from fairseq2.setup import setup_fairseq2

log = get_log_writer(__name__)


def _setup_asset_cli(cli: Cli) -> None:
    group = cli.add_group(
        "assets", help="list and show assets (e.g. models, tokenizers, datasets)"
    )

    group.add_command(
        "list",
        ListAssetsCommand(),
        help="list assets",
    )

    group.add_command(
        "show",
        ShowAssetCommand(),
        help="show asset",
    )


@final
class ListAssetsCommand(CliCommandHandler):
    """Lists assets available in the current Python environment."""

    _asset_store: AssetStore

    def __init__(self, asset_store: AssetStore | None = None) -> None:
        """
        :param asset_store:
            The asset store from which to retrieve the asset cards. If ``None``,
            the default asset store will be used.
        """
        self._asset_store = asset_store or default_asset_store

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--type",
            choices=["all", "model", "dataset", "tokenizer"],
            default="all",
            help="type of assets to list",
        )

    @override
    def run(self, args: Namespace) -> int:
        setup_fairseq2()

        usr_assets = self._retrieve_assets(args, user=True)
        glb_assets = self._retrieve_assets(args, user=False)

        console = get_console()

        console.print("[green bold]user:")

        self._dump_assets(console, usr_assets)

        console.print("[green bold]global:")

        self._dump_assets(console, glb_assets)

        return 0

    def _retrieve_assets(
        self, args: Namespace, user: bool
    ) -> list[tuple[str, list[str]]]:
        assets: dict[str, list[str]] = defaultdict(list)

        names = self._asset_store.retrieve_names(scope="user" if user else "global")

        for name in names:
            try:
                card = self._asset_store.retrieve_card(
                    name, scope="all" if user else "global"
                )
            except AssetNotFoundError:
                log.warning("The asset '{}' has an invalid card. Skipping.", name)

                continue

            if name[-1] == "@":
                name = name[:-1]

            try:
                source = cast(str, card.metadata["__source__"])
            except KeyError:
                source = "unknown source"

            types = []

            if args.type == "all" or args.type == "model":
                if is_model_card(card):
                    types.append("model")

            if args.type == "all" or args.type == "dataset":
                if is_dataset_card(card):
                    types.append("dataset")

            if args.type == "all" or args.type == "tokenizer":
                if self._is_tokenizer_card(card):
                    types.append("tokenizer")

            if args.type == "all" and not types:
                types.append("other")

            if not types:
                continue

            source_assets = assets[source]

            for t in types:
                source_assets.append(f"{t}:{name}")

        return [(source, names) for source, names in assets.items()]

    @staticmethod
    def _is_tokenizer_card(card: AssetCard) -> bool:
        return card.field("tokenizer_family").exists()

    def _dump_assets(
        self, console: Console, assets: list[tuple[str, list[str]]]
    ) -> None:
        if assets:
            assets.sort(key=lambda a: a[0])  # sort by source.

            for source, names in assets:
                names.sort(key=lambda n: n[0])  # sort by name.

                console.print(f"  [blue bold]{source}")

                for idx, name in enumerate(names):
                    console.print(f"   - {name}")

                console.print()
        else:
            console.print("  n/a")
            console.print()


class ShowAssetCommand(CliCommandHandler):
    """Shows the metadata of an asset."""

    _asset_store: AssetStore

    def __init__(self, asset_store: AssetStore | None = None) -> None:
        """
        :param asset_store:
            The asset store from which to retrieve the asset cards. If ``None``,
            the default asset store will be used.
        """
        setup_fairseq2()

        self._asset_store = asset_store or default_asset_store

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--env",
            dest="envs",
            metavar="ENV",
            nargs="*",
            help="environments to check",
        )

        parser.add_argument(
            "--scope",
            choices=["all", "global", "user"],
            default="all",
            help="scope for query (default: %(default)s)",
        )

        parser.add_argument("name", help="name of the asset")

    @override
    def run(self, args: Namespace) -> int:
        try:
            card: AssetCard | None = self._asset_store.retrieve_card(
                args.name, envs=args.envs, scope=args.scope
            )
        except AssetNotFoundError:
            log.error("An asset with the name '{}' cannot be found.", args.name)

            sys.exit(1)

        while card is not None:
            self._print_metadata(dict(card.metadata))

            card = card.base

        return 0

    def _print_metadata(self, metadata: dict[str, Any]) -> None:
        console = get_console()

        name = metadata.pop("name")

        console.print(f"[green bold]{name}")

        try:
            source = metadata.pop("__source__")
        except KeyError:
            source = "unknown"

        items = list(metadata.items())

        items.insert(0, ("source", source))

        for key, value in items:
            # Skip dunder keys (e.g. __base_path__).
            if len(key) > 4 and key.startswith("__") and key.endswith("__"):
                continue

            console.print(f"  [bold]{key:<16}:[/bold] {pretty_repr(value)}")

        console.print()
