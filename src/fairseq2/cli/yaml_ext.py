import pathlib
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import yaml

if TYPE_CHECKING:
    import torch

T = TypeVar("T")


def Dumper(*args: Any, **kwargs: Any) -> yaml.Dumper:
    import torch

    dumper = yaml.Dumper(*args, **kwargs)
    add_from_string(dumper, pathlib.PosixPath)
    add_from_string(dumper, pathlib.WindowsPath)
    add_from_string(dumper, pathlib.PurePath)
    add_from_string(dumper, torch.device)

    dumper.add_representer(torch.dtype, represent_dtype)
    return dumper


def represent_dtype(dumper: yaml.Dumper, dtype: "torch.dtype") -> yaml.ScalarNode:
    return dumper.represent_scalar(f"tag:yaml.org,2002:python/name:{dtype}", "")


def add_from_string(
    dumper: yaml.Dumper, cls: Callable[[str], T], to_str: Callable[[T], str] = str
) -> None:
    """Register representer for a type that is constructible from a string."""
    assert isinstance(cls, type)
    module, name = (cls.__module__, cls.__name__)

    def from_string(dumper: yaml.Dumper, data: T) -> yaml.ScalarNode:
        return dumper.represent_sequence(
            f"tag:yaml.org,2002:python/object/apply:{module}.{name}",
            [to_str(data)],
        )

    dumper.add_representer(cls, from_string)
