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

from fairseq2.assets import (
    AssetCard,
    AssetCardError,
    AssetCardNotFoundError,
    AssetLookupScope,
    AssetStore,
)
from fairseq2.cli import CliArgumentError, CliCommandError, CliCommandHandler
from fairseq2.cli.utils.rich import get_console
from fairseq2.context import RuntimeContext
from fairseq2.logging import log


@final
class ListAssetsHandler(CliCommandHandler):
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
        asset_store = context.asset_store

        usr_assets = self._retrieve_assets(asset_store, args.type, user=True)
        sys_assets = self._retrieve_assets(asset_store, args.type)

        console = get_console()

        console.print("[green bold]user:")

        self._print_assets(console, usr_assets)

        console.print("[green bold]system:")

        self._print_assets(console, sys_assets)

        return 0

    @staticmethod
    def _retrieve_assets(
        asset_store: AssetStore, asset_type: str, user: bool = False
    ) -> list[tuple[str, list[str]]]:
        all_assets: dict[str, list[str]] = defaultdict(list)

        scope = AssetLookupScope.USER if user else AssetLookupScope.SYSTEM

        asset_names = asset_store.retrieve_names(scope=scope)

        for asset_name in asset_names:
            try:
                card = asset_store.retrieve_card(asset_name, scope=scope)
            except AssetCardNotFoundError:
                log.warning("'{}' asset card has no generic version. Skipping.", asset_name)  # fmt: skip

                continue
            except AssetCardError:
                log.warning("'{}' asset card cannot be loaded. Skipping.", asset_name)

                continue

            if asset_name[-1] == "@":
                asset_name = asset_name[:-1]

            source = card.metadata.get("__source__", "unknown source")
            if not isinstance(source, str):
                source = "unknown source"

            asset_types = []

            if asset_type == "all" or asset_type == "model":
                if card.field("model_family").exists():
                    asset_types.append("model")

            if asset_type == "all" or asset_type == "dataset":
                if card.field("dataset_family").exists():
                    asset_types.append("dataset")

            if asset_type == "all" or asset_type == "tokenizer":
                if card.field("tokenizer_family").exists():
                    asset_types.append("tokenizer")

            if asset_type == "all" and not asset_types:
                asset_types.append("other")

            if not asset_types:
                continue

            source_assets = all_assets[source]

            for t in asset_types:
                source_assets.append(f"{t}:{asset_name}")

        output = []

        for source, asset_names in all_assets.items():
            asset_names.sort()

            output.append((source, asset_names))

        output.sort(key=lambda pair: pair[0])  # sort by source

        return output

    @staticmethod
    def _print_assets(console: Console, assets: list[tuple[str, list[str]]]) -> None:
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
            choices=["all", "system", "user"],
            default="all",
            help="scope for query (default: %(default)s)",
        )

        parser.add_argument("name", help="name of the asset")

    @override
    def run(
        self, context: RuntimeContext, parser: ArgumentParser, args: Namespace
    ) -> int:
        match args.scope:
            case "all":
                scope = AssetLookupScope.ALL
            case "system":
                scope = AssetLookupScope.SYSTEM
            case "user":
                scope = AssetLookupScope.USER
            case _:
                parser.error("invalid scope")

                return 2

        try:
            card: AssetCard | None = context.asset_store.retrieve_card(
                args.name, envs=args.envs, scope=scope
            )
        except AssetCardNotFoundError:
            raise CliArgumentError(
                "name", f"'{args.name}' is not a known asset. Use `fairseq2 assets list` to see the available assets."  # fmt: skip
            ) from None
        except AssetCardError as ex:
            raise CliCommandError(
                f"The '{args.name}' asset card cannot be read. See the nested exception for details."
            ) from ex

        while card is not None:
            self._print_metadata(dict(card.metadata))

            card = card.base

        return 0

    @staticmethod
    def _print_metadata(metadata: dict[str, object]) -> None:
        console = get_console()

        name = metadata.pop("name")

        console.print(f"[green bold]{name}")

        def print_field(name: str, value: object) -> None:
            is_dunder = len(name) > 4 and name.startswith("__") and name.endswith("__")
            if not is_dunder:
                console.print(f"  [bold]{name:<16}:[/bold] {pretty_repr(value)}")

        source = metadata.pop("__source__", "unknown")

        print_field("source", source)

        for field_name, value in metadata.items():
            print_field(field_name, value)

        console.print()
