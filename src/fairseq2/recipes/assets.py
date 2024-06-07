# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, final

from fairseq2.assets import AssetStore, default_asset_store
from fairseq2.recipes.cli import Cli, CliCommandHandler
from fairseq2.recipes.logging import console
from fairseq2.typing import override


def _setup_asset_cli(cli: Cli) -> None:
    group = cli.add_group(
        "assets", help="list and show assets (e.g. models, tokenizers, datasets)"
    )

    handler = ListAssetsComand()

    group.add_command("list", handler, help="list assets")


@final
class ListAssetsComand(CliCommandHandler):
    """Lists assets available in the current Python environment."""

    _asset_store: AssetStore

    def __init__(self, asset_store: Optional[AssetStore] = None) -> None:
        """
        :param asset_store:
            The asset store from which to retrieve the asset cards.
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
        console.print("[green bold]user:")

        self._dump_assets(args, user=True)

        console.print()

        console.print("[green bold]global:")

        self._dump_assets(args, user=False)

    def _dump_assets(self, args: Namespace, user: bool) -> None:
        assets = self._retrieve_assets(args, user=user)

        if assets:
            assets.sort(key=lambda a: a[0])  # sort by source.

            for source, names in assets:
                names.sort(key=lambda n: n[0])  # sort by name.

                console.print(f"  [blue bold]{source}")

                for idx, name in enumerate(names):
                    console.print(f"   - {name}", highlight=False)

                console.print()
        else:
            console.print("  n/a")

    def _retrieve_assets(
        self, args: Namespace, user: bool
    ) -> List[Tuple[str, List[str]]]:
        assets: Dict[str, List[str]] = defaultdict(list)

        names = self._asset_store.retrieve_names(scope="user" if user else "global")

        for name in names:
            card = self._asset_store.retrieve_card(name)

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

            if not types:
                types.append("other")

            source_assets = assets[source]

            for t in types:
                source_assets.append(f"{t}:{name}")

        return [(source, names) for source, names in assets.items()]
