# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import final

from rich.console import Console
from rich.pretty import pretty_repr
from typing_extensions import override

from fairseq2.assets import AssetCard, AssetCardNotFoundError, AssetStore
from fairseq2.cli import CliCommandHandler
from fairseq2.context import RuntimeContext
from fairseq2.logging import log
from fairseq2.recipes.utils.rich import get_console


@final
class ListAssetsHandler(CliCommandHandler):
    """Lists assets available in the current Python environment."""

    @override
    def init_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--type",
            choices=["all", "model", "dataset", "tokenizer"],
            default="all",
            help="type of assets to list",
        )

    @override
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        console = get_console()

        console.print("[green bold]user:")

        assets = self._retrieve_assets(context.asset_store, args.type, user=True)

        self._dump_assets(console, assets)

        console.print("[green bold]global:")

        assets = self._retrieve_assets(context.asset_store, args.type, user=False)

        self._dump_assets(console, assets)

        return 0

    @classmethod
    def _retrieve_assets(
        cls, asset_store: AssetStore, asset_type: str, user: bool
    ) -> list[tuple[str, list[str]]]:
        assets: dict[str, list[str]] = defaultdict(list)

        asset_names = asset_store.retrieve_names(scope="user" if user else "global")

        for asset_name in asset_names:
            try:
                card = asset_store.retrieve_card(
                    asset_name, scope="all" if user else "global"
                )
            except AssetCardNotFoundError:
                log.warning("The '{}' asset card is not valid. Skipping.", asset_name)

                continue

            if asset_name[-1] == "@":
                asset_name = asset_name[:-1]

            source = card.metadata.get("__source__", "unknown source")
            if not isinstance(source, str):
                source = "unknown source"

            asset_types = []

            if asset_type == "all" or asset_type == "model":
                if cls._is_model_card(card):
                    asset_types.append("model")

            if asset_type == "all" or asset_type == "dataset":
                if cls._is_dataset_card(card):
                    asset_types.append("dataset")

            if asset_type == "all" or asset_type == "tokenizer":
                if cls._is_tokenizer_card(card):
                    asset_types.append("tokenizer")

            if asset_type == "all" and not asset_types:
                asset_types.append("other")

            if not asset_types:
                continue

            source_assets = assets[source]

            for t in asset_types:
                source_assets.append(f"{t}:{asset_name}")

        output = []

        for source, asset_names in assets.items():
            asset_names.sort()

            output.append((source, asset_names))

        output.sort(key=lambda e: e[0])  # sort by source

        return output

    @staticmethod
    def _is_model_card(card: AssetCard) -> bool:
        return card.field("model_family").exists()

    @staticmethod
    def _is_tokenizer_card(card: AssetCard) -> bool:
        return card.field("tokenizer_family").exists()

    @staticmethod
    def _is_dataset_card(card: AssetCard) -> bool:
        return card.field("dataset_family").exists()

    @staticmethod
    def _dump_assets(console: Console, assets: list[tuple[str, list[str]]]) -> None:
        if assets:
            for source, names in assets:
                console.print(f"  [blue bold]{source}")

                for idx, name in enumerate(names):
                    console.print(f"   - {name}")

                console.print()
        else:
            console.print("  n/a")
            console.print()


class ShowAssetHandler(CliCommandHandler):
    """Shows the metadata of an asset."""

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
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        card: AssetCard | None = context.asset_store.retrieve_card(
            args.name, envs=args.envs, scope=args.scope
        )

        while card is not None:
            self._print_metadata(dict(card.metadata))

            card = card.base

        return 0

    def _print_metadata(self, metadata: dict[str, object]) -> None:
        console = get_console()

        name = metadata.pop("name")

        console.print(f"[green bold]{name}")

        source = metadata.pop("__source__", "unknown")

        items = list(metadata.items())

        items.insert(0, ("source", source))

        for key, value in items:
            # Skip dunder keys (e.g. __base_path__).
            if len(key) > 4 and key.startswith("__") and key.endswith("__"):
                continue

            console.print(f"  [bold]{key:<16}:[/bold] {pretty_repr(value)}")

        console.print()
