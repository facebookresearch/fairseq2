# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser, Namespace
from io import BytesIO
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Final, Optional, Sequence

import boto3

INDEX_TEMPLATE: Final = """<!DOCTYPE html>
<html>
<head>
    <title>PEP 503 Index</title>
</head>
<body>
%s
</body>
</html>"""

if TYPE_CHECKING:

    class Bucket:
        objects: Any


def main() -> None:
    args = parse_args()

    bucket = get_s3_bucket(args.bucket, args.profile_name)

    create_or_update_project_index(bucket, args.prefix, args.project_names)

    for name in args.project_names:
        create_or_update_package_index(bucket, args.prefix, name)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("bucket", metavar="BUCKET")
    parser.add_argument("prefix", metavar="PREFIX")
    parser.add_argument("project_names", metavar="PROJECT", nargs="+")
    parser.add_argument("--profile-name")

    return parser.parse_args()


def get_s3_bucket(bucket_name: str, profile_name: Optional[str] = None) -> "Bucket":
    session = boto3.Session(profile_name=profile_name)

    s3 = session.resource("s3")

    return s3.Bucket(bucket_name)  # type: ignore[no-any-return]


def create_or_update_project_index(
    bucket: "Bucket", prefix: str, project_names: Sequence[str]
) -> None:
    anchors = [f'<a href="{name}/">{name}</a><br>' for name in project_names]

    index = INDEX_TEMPLATE % "\n".join(anchors)

    upload_index(bucket, prefix, index)


def create_or_update_package_index(
    bucket: "Bucket", prefix: str, project_name: str
) -> None:
    anchors = []

    for obj in bucket.objects.filter(Prefix=f"{prefix}/{project_name}/"):
        if (obj_path := PurePath(obj.key)).suffix == ".whl":
            anchors.append(f'<a href="{obj_path.name}">{obj_path.name}</a><br>')

    index = INDEX_TEMPLATE % "\n".join(anchors)

    upload_index(bucket, f"{prefix}/{project_name}", index)


def upload_index(bucket: "Bucket", prefix: str, index: str) -> None:
    metadata = {"ContentType": "text/html", "CacheControl": "max-age=300"}

    with BytesIO(index.encode("utf-8")) as fp:
        bucket.upload_fileobj(  # type: ignore[attr-defined]
            fp, f"{prefix}/index.html", ExtraArgs=metadata
        )


if __name__ == "__main__":
    main()
