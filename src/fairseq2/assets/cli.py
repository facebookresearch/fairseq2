# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import final

from rich.console import Console
from rich.pretty import pretty_repr

from fairseq2 import init_fairseq2
from fairseq2.assets.card import AssetCard
from fairseq2.assets.dirs import InvalidAssetPathVariableError
from fairseq2.assets.metadata_provider import (
    AssetMetadataSourceError,
    AssetMetadataSourceNotFoundError,
    CorruptAssetMetadataSourceError,
)
from fairseq2.assets.store import (
    AssetStore,
    AssetStoreError,
    BaseAssetCardNotFoundError,
)
from fairseq2.composition import (
    ExtensionError,
    register_checkpoint_models,
    register_file_assets,
)
from fairseq2.error import InternalError, OperationalError
from fairseq2.logging import configure_logging, log
from fairseq2.runtime.dependency import (
    DependencyContainer,
    DependencyResolver,
    activate_dependency,
    wire_object,
)
from fairseq2.utils.rich import get_console


def main() -> int:
    try:
        _run()
    except CommandArgumentError as ex:
        log.error(str(ex))

        return 2
    except CommandError as ex:
        log.error(str(ex), exc=ex.__cause__)

        return 1
    except OperationalError:
        log.exception("Command failed due to an operational error.")

        return 1
    except Exception:
        log.exception("Command failed due to an unexpected error. File a bug report to the corresponding author.")  # fmt: skip

        return 1
    else:
        return 0


def _run() -> None:
    args = _parse_args()

    configure_logging()

    def extras(container: DependencyContainer) -> None:
        _register_commands(container, args)

    try:
        resolver = init_fairseq2(extras=extras)
    except ExtensionError as ex:
        raise CommandError(
            f"'{ex.entry_point}' extension failed to initialize."
        ) from ex

    try:
        activate_dependency(resolver, AssetStore)
    except AssetMetadataSourceNotFoundError as ex:
        raise CommandError(
            f"Failed to load asset store. '{ex.source}' asset metadata source is not found."
        ) from None
    except CorruptAssetMetadataSourceError as ex:
        raise CommandError(
            f"Failed to load asset store. '{ex.source}' asset metadata source is corrupt."
        ) from ex
    except AssetMetadataSourceError as ex:
        cause = ex.__cause__

        if isinstance(cause, InvalidAssetPathVariableError):
            raise CommandError(
                f"Failed to load asset store. `{cause.var_name}` environment variable of the '{ex.source}' asset metadata source is expected to be a pathname, but is '{cause.value}' instead."
            ) from None

        raise OperationalError("Failed to load asset store.") from ex

    args.command(resolver, args)


def _parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--extra-asset-path",
        type=Path,
        help="extra asset card path",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="fairseq2 checkpoint directory",
    )

    sub_parsers = parser.add_subparsers()

    # list
    sub_parser = sub_parsers.add_parser("list", help="list available assets")

    kinds = ["all", "model", "dataset", "tokenizer"]

    sub_parser.add_argument(
        "--kind",
        choices=kinds,
        default="all",
        help="asset kinds to list",
    )

    sub_parser.set_defaults(command=_list_assets)

    # show
    sub_parser = sub_parsers.add_parser("show", help="show asset")

    sub_parser.add_argument(
        "name",
        help="name of the asset",
    )

    sub_parser.set_defaults(command=_show_asset)

    return parser.parse_args()


def _register_commands(container: DependencyContainer, args: Namespace) -> None:
    if args.extra_asset_path:
        register_file_assets(container, args.extra_asset_path)

    if args.checkpoint_dir:
        register_checkpoint_models(container, args.checkpoint_dir)

    console = get_console()

    # ListAssets
    def create_list_assets_command(resolver: DependencyResolver) -> _ListAssetsCommand:
        return wire_object(resolver, _ListAssetsCommand, console=console)

    container.register(_ListAssetsCommand, create_list_assets_command)

    # ShowAsset
    def create_show_asset_command(resolver: DependencyResolver) -> _ShowAssetCommand:
        return wire_object(resolver, _ShowAssetCommand, console=console)

    container.register(_ShowAssetCommand, create_show_asset_command)


def _list_assets(resolver: DependencyResolver, args: Namespace) -> None:
    resolver.resolve(_ListAssetsCommand).run(args.kind)


@final
class _ListAssetsCommand:
    def __init__(self, console: Console, asset_store: AssetStore) -> None:
        self._console = console
        self._asset_store = asset_store

    def run(self, asset_kind: str) -> None:
        """
        :raises CommandError:
        """
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
                card = self._asset_store.maybe_retrieve_card(name)
            except BaseAssetCardNotFoundError as ex:
                log.warning("'{}' base asset card of '{}' is not found. Skipping.", ex.base_name, name)  # fmt: skip

                continue
            except AssetStoreError as ex:
                raise OperationalError(
                    f"Failed to retrieve '{name}' asset card."
                ) from ex

            if card is None:
                if name[-1] != "@":
                    log.warning("Base asset card of {} not found. Skipping.", name)

                    continue

                raise InternalError(
                    f"'{name[:-1]}' is in `asset_names`, but is not found in the store."
                )

            source = card.metadata.get("__source__")
            if not isinstance(source, str):
                source = "unknown source"

            asset_kinds = []

            if asset_kind == "all":
                for kind in ("model", "dataset", "tokenizer"):
                    if card.has_field(f"{kind}_family"):
                        asset_kinds.append(kind)

                if not asset_kinds:
                    asset_kinds.append("other")
            else:
                if card.has_field(f"{asset_kind}_family"):
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


def _show_asset(resolver: DependencyResolver, args: Namespace) -> None:
    resolver.resolve(_ShowAssetCommand).run(args.name)


@final
class _ShowAssetCommand:
    def __init__(self, console: Console, asset_store: AssetStore) -> None:
        self._console = console
        self._asset_store = asset_store

    def run(self, name: str) -> None:
        """
        :raises CommandError:
        """
        try:
            card = self._asset_store.maybe_retrieve_card(name)
        except BaseAssetCardNotFoundError as ex:
            raise CommandError(
                f"'{ex.base_name}' base asset card of '{name}' is not found."
            ) from None
        except AssetStoreError as ex:
            raise OperationalError(f"Failed to retrieve '{name}' asset card.") from ex

        if card is None:
            raise CommandArgumentError(f"'{name}' asset is not found.")

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


class CommandError(Exception):
    pass


class CommandArgumentError(CommandError):
    pass
