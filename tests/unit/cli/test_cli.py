from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from fairseq2.cli import XpScript

MY_SCRIPT = """
def f1(x: int, y: int = 2) -> int:
    '''adds two number

    - x: first integer
    - y: second integer
    '''
    return x + y

def f2(x: int, z: int = 2) -> int:
    return x + z

def z(x: int) -> int:
    return x * x

def ret_x(x: int = 1) -> int:
    return x

def loop1(loop2: int) -> int:
    return 3 * loop2

def loop2(loop1: int) -> int:
    return loop1 - 1

def obj():
    '''The object'''
    return object()

def obj_id(obj) -> int:
    '''The id of the object'''
    return id(obj)
"""


@pytest.fixture(scope="session")
def script_path(tmp_path_factory: Any) -> Path:
    cli_dir: Path = tmp_path_factory.mktemp("cli")
    script_path = cli_dir / "script.py"
    script_path.write_text(MY_SCRIPT)
    return script_path


def test_call_fn_only_optional(script_path: Path) -> None:
    module = XpScript.from_script(script_path)
    assert module.call_fn("ret_x", caller="test") == 1


def test_call_fn_overrides_default(script_path: Path) -> None:
    module = XpScript.from_script(script_path, overrides=["x=3", "y=5"])
    assert module.call_fn("ret_x", caller="test") == 3
    assert module.call_fn("f1", caller="test") == 3 + 5


def test_call_fn_multi_step(script_path: Path) -> None:
    module = XpScript.from_script(script_path, overrides=["x=3"])
    assert module.call_fn("ret_x", caller="test") == 3
    assert module.call_fn("z", caller="test") == 9
    assert module.call_fn("f2", caller="test") == 3 + 9
    assert module.call_fn("f1", caller="test") == 3 + 2


def test_call_fn_detects_loop(script_path: Path) -> None:
    module = XpScript.from_script(script_path)
    with pytest.raises(Exception, match="loop detected: loop1 -> loop2 -> loop1"):
        module.call_fn("loop1", caller="test")


def test_call_fn_overrides_fn(script_path: Path) -> None:
    module = XpScript.from_script(script_path, overrides=["f1=0", "loop1=5"])
    assert module.call_fn("f1", caller="test") == 0
    assert module.call_fn("loop1", caller="test") == 5


def test_call_fn_missing_arg(script_path: Path) -> None:
    module = XpScript.from_script(script_path)
    with pytest.raises(Exception, match=r"Can't call f1, missing args: \['x'\]"):
        module.call_fn("f1", caller="test")


def test_serialize_save_fn_calls(script_path: Path, tmp_path: Path) -> None:
    module = XpScript.from_script(script_path, overrides=["x=5"])
    f1 = module.call_fn("f1", caller="test")
    module.serialize(tmp_path / "config.yaml")

    module2 = XpScript.from_script(script_path, yaml_config=tmp_path / "config.yaml")
    assert module2.call_fn("f1", caller="test") == f1


def test_serialize_skip_objects(script_path: Path, tmp_path: Path) -> None:
    module = XpScript.from_script(script_path)
    obj_id = module.call_fn("obj_id", caller="test")
    module.serialize(tmp_path / "config.yaml")
    assert list(module._cache.keys()) == ["obj", "obj_id"]

    module2 = XpScript.from_script(script_path, yaml_config=tmp_path / "config.yaml")
    # We saved obj_id, but not obj since there is no trivial serialization for it.
    assert list(module2._cache.keys()) == ["obj_id"]
    assert module2.call_fn("obj_id", caller="test") == obj_id


def test_parse_duration() -> None:
    from fairseq2.cli.module_loader import _parse

    parse_duration = lambda x: _parse(timedelta, "x", x)

    assert parse_duration("10m") == timedelta(minutes=10)
    assert parse_duration("3d") == timedelta(days=3)
    assert parse_duration("0.5h") == timedelta(hours=0.5)

    with pytest.raises(ValueError, match="time unit"):
        parse_duration("1H")
    with pytest.raises(ValueError, match="Can't parse"):
        parse_duration("abcdefgh")


def test_dag(script_path: Path) -> None:
    module = XpScript.from_script(script_path)
    dag = module.dag(["f1", "f2"])
    print(dag)
    assert dag["f1"] == ["x", "y"]
    assert dag["f2"] == ["x", "z"]
    assert dag["z"] == ["x"]


def test_bfs(script_path: Path) -> None:
    module = XpScript.from_script(script_path)
    dag = module.dag(["f1", "f2"])
    bfs = list(dag.breadth_first_traversal())
    assert bfs == ["f1", "f2", "x", "y", "z"]


def test_help(script_path: Path) -> None:
    module = XpScript.from_script(script_path)
    module_help = module.help("f1", "obj_id")
    print(module_help)
    assert "f1" in module_help
    assert "f1.x (int): first integer (REQUIRED)" in module_help
    assert "f1.y (int): second integer (default=2)" in module_help
    assert "The object" in module_help
    assert "The id of the object" in module_help
