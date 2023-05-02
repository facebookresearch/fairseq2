from pathlib import Path
from typing import Any, NamedTuple

import torch
import yaml

from fairseq2.cli import yaml_ext


def dump_and_load(x: Any) -> Any:
    return yaml.load(yaml.dump(x, Dumper=yaml_ext.Dumper), Loader=yaml.Loader)


def test_serialize_device() -> None:
    device = torch.device("cuda:1")
    assert dump_and_load(device) == device


def test_serialize_path() -> None:
    path = Path("/checkpoint/$USER/fairseq2/cool_experiment")
    assert dump_and_load(path) == path


def test_serialize_dtype() -> None:
    dtype = torch.float16
    assert dump_and_load(dtype) is dtype


def test_serialize_dict() -> None:
    data = {
        "device": torch.device("cuda:1"),
        "path": Path("/checkpoint/$USER/fairseq2/cool_experiment"),
        "dtype": torch.float16,
    }

    assert dump_and_load(data) == data

    nice_yaml = yaml.dump(data, Dumper=yaml_ext.Dumper)
    print(nice_yaml)

    assert (
        nice_yaml
        == """
device: !!python/object/apply:torch.device
- cuda:1
dtype: !!python/name:torch.float16 ''
path: !!python/object/apply:pathlib.PosixPath
- /checkpoint/$USER/fairseq2/cool_experiment
""".lstrip()
    )


class X(NamedTuple):
    a: int
    b: torch.dtype


def test_serialize_named_tuple() -> None:
    x = X(1, torch.int64)
    assert dump_and_load(x) == x
    nice_yaml = yaml.dump(x, Dumper=yaml_ext.Dumper)
    print(nice_yaml)
    assert (
        nice_yaml
        == """
!!python/object/new:tests.cli.test_yaml_ext.X
- 1
- !!python/name:torch.int64 ''
""".lstrip()
    )
