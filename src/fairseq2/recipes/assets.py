# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, final

from rich.console import Console
from rich.pretty import pretty_repr

from fairseq2.assets import (
    AssetCard,
    AssetNotFoundError,
    AssetStore,
    default_asset_store,
)
from fairseq2.console import get_console
from fairseq2.logging import get_log_writer
from fairseq2.recipes.cli import Cli, CliCommandHandler
from fairseq2.recipes.logging import setup_basic_logging
from fairseq2.typing import override

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

    def __init__(self, asset_store: Optional[AssetStore] = None) -> None:
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
    def __call__(self, args: Namespace) -> None:
        setup_basic_logging()

        usr_assets = self._retrieve_assets(args, user=True)
        glb_assets = self._retrieve_assets(args, user=False)

        console = get_console()

        console.print("[green bold]user:")

        self._dump_assets(console, usr_assets)

        console.print("[green bold]global:")

        self._dump_assets(console, glb_assets)

    def _retrieve_assets(
        self, args: Namespace, user: bool
    ) -> List[Tuple[str, List[str]]]:
        assets: Dict[str, List[str]] = defaultdict(list)

        names = self._asset_store.retrieve_names(scope="user" if user else "global")

        for name in names:
            card = self._asset_store.retrieve_card(
                name, scope="all" if user else "global"
            )

            if name[-1] == "@":
                name = name[:-1]

            try:
                source = card.metadata["__source__"]
            except KeyError:
                source = "unknown source"

            types = []

            if args.type == "all" or args.type == "model":
                if card.field("model_family").exists():
                    types.append("model")

            if args.type == "all" or args.type == "dataset":
                if card.field("dataset_family").exists():
                    types.append("dataset")

            if args.type == "all" or args.type == "tokenizer":
                for field_name in ("tokenizer_family", "tokenizer"):
                    if card.field(field_name).exists():
                        types.append("tokenizer")

                        break

            if args.type == "all" and not types:
                types.append("other")

            if not types:
                continue

            source_assets = assets[source]

            for t in types:
                source_assets.append(f"{t}:{name}")

        return [(source, names) for source, names in assets.items()]

    def _dump_assets(
        self, console: Console, assets: List[Tuple[str, List[str]]]
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

    def __init__(self, asset_store: Optional[AssetStore] = None) -> None:
        """
        :param asset_store:
            The asset store from which to retrieve the asset cards. If ``None``,
            the default asset store will be used.
        """
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
    def __call__(self, args: Namespace) -> None:
        setup_basic_logging()

        try:
            card: Optional[AssetCard] = self._asset_store.retrieve_card(
                args.name, envs=args.envs, scope=args.scope
            )
        except AssetNotFoundError:
            log.error("An asset with the name '{}' cannot be found.", args.asset)

            sys.exit(1)

        while card is not None:
            self._print_metadata(dict(card.metadata))

            card = card.base

    def _print_metadata(self, metadata: Dict[str, Any]) -> None:
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

            # compat
            if key == "model_type":
                continue

            console.print(f"  [bold]{key:<16}:[/bold] {pretty_repr(value)}")

        console.print()
