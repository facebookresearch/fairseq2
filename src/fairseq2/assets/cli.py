# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace
from collections import defaultdict

from rich.console import Console
from rich.pretty import pretty_repr

from fairseq2.assets.card import AssetCard, AssetCardError
from fairseq2.assets.metadata_provider import AssetMetadataError, AssetNotFoundError
from fairseq2.assets.store import AssetStore, get_asset_store
from fairseq2.error import ContractError, InternalError
from fairseq2.logging import log
from fairseq2.runtime.composition import ExtensionError
from fairseq2.utils.rich import configure_rich_logging, get_console


def _main() -> None:
    args = _parse_args()

    configure_rich_logging()

    try:
        args.command(args)
    except AssetNotFoundError as ex:
        log.error("'{}' is not a known asset.", ex.name)

        sys.exit(2)
    except AssetMetadataError as ex:
        log.exception("Asset metadata of '{}' cannot be loaded. See the logged stack trace for details.", ex.source)  # fmt: skip

        sys.exit(1)
    except AssetCardError as ex:
        log.exception("Asset '{}' cannot be loaded. See the logged stack trace for details.", ex.name)  # fmt: skip

        sys.exit(1)
    except ExtensionError as ex:
        log.exception("{} extension failed to initialize. See the logged stack trace for details.", ex.entry_point)  # fmt: skip

        sys.exit(1)
    except InternalError:
        log.exception("Command failed with an unexpected internal error. Please file a bug report.")  # fmt: skip

        sys.exit(1)
    except ContractError:
        log.exception("Command failed with an unexpected internal error caused by an extension. Please file a bug report to the corresponding extension author.")  # fmt: skip

        sys.exit(1)


def _parse_args() -> Namespace:
    parser = ArgumentParser()

    sub_parsers = parser.add_subparsers()

    # list
    sub_parser = sub_parsers.add_parser("list", help="list available assets")

    kinds = ["all", "model", "dataset", "tokenizer"]

    sub_parser.add_argument(
        "--kind", choices=kinds, default="all", help="asset kinds to list"
    )

    sub_parser.set_defaults(command=_list_assets)

    # show
    sub_parser = sub_parsers.add_parser("show", help="show asset")

    sub_parser.add_argument("name", help="name of the asset")

    sub_parser.set_defaults(command=_show_asset)

    return parser.parse_args()


def _list_assets(args: Namespace) -> None:
    console = get_console()

    asset_store = get_asset_store()

    command = _ListAssetsCommand(console, asset_store)

    command.run(args.kind)


class _ListAssetsCommand:
    _console: Console
    _asset_store: AssetStore

    def __init__(self, console: Console, asset_store: AssetStore) -> None:
        self._console = console
        self._asset_store = asset_store

    def run(self, asset_kind: str) -> None:
        console = self._console

        assets = self._retrieve_assets(asset_kind)
        if assets:
            for source, names in assets:
                console.print(f"  [blue bold]{source}")

                for idx, name in enumerate(names):
                    console.print(f"   - {name}")

                console.print()
        else:
            console.print("  n/a")
            console.print()

    def _retrieve_assets(self, asset_kind: str) -> list[tuple[str, list[str]]]:
        assets: dict[str, list[str]] = defaultdict(list)

        for name in self._asset_store.asset_names:
            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError as ex:
                raise InternalError(
                    f"'{name}' is in `asset_names`, but cannot be found in the store."
                ) from ex
            except AssetCardError:
                log.warning("'{}' asset card cannot be loaded. Skipping.", name)

                continue

            source = card.metadata.get("__source__")
            if not isinstance(source, str):
                source = "unknown source"

            asset_kinds = []

            if asset_kind == "all":
                for kind in ("model", "dataset", "tokenizer"):
                    if card.field(f"{kind}_family").exists():
                        asset_kinds.append(kind)

                if not asset_kinds:
                    asset_kinds.append("other")
            else:
                if card.field(f"{asset_kind}_family").exists():
                    asset_kinds.append(asset_kind)

            if not asset_kinds:
                continue

            source_assets = assets[source]

            for kind in asset_kinds:
                source_assets.append(f"{kind}:{name}")

        assets_by_source = []

        for source, asset_names in assets.items():
            asset_names.sort()

            assets_by_source.append((source, asset_names))

        assets_by_source.sort(key=lambda pair: pair[0])  # sort by source

        return assets_by_source


def _show_asset(args: Namespace) -> None:
    console = get_console()

    asset_store = get_asset_store()

    command = _ShowAssetCommand(console, asset_store)

    command.run(args.name)


class _ShowAssetCommand:
    _console: Console
    _asset_store: AssetStore

    def __init__(self, console: Console, asset_store: AssetStore) -> None:
        self._console = console
        self._asset_store = asset_store

    def run(self, name: str) -> None:
        card: AssetCard | None = self._asset_store.retrieve_card(name)

        while card is not None:
            self._print_asset_card(card)

            card = card.base

    def _print_asset_card(self, card: AssetCard) -> None:
        console = self._console

        console.print(f"[green bold]{card.name}")

        def print_field(name: str, value: object) -> None:
            console.print(f"  [bold]{name:<16}:[/bold] {pretty_repr(value)}")

        source = card.metadata.get("__source__", "unknown")

        print_field("source", source)

        for field, value in card.metadata.items():
            if not self._is_dunder(field):
                print_field(field, value)

        console.print()

    @staticmethod
    def _is_dunder(name: str) -> bool:
        return len(name) > 4 and name.startswith("__") and name.endswith("__")
