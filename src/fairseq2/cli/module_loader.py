import dataclasses
import importlib
import inspect
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Type

import func_argparse
import yaml

import fairseq2.distributed

log = logging.getLogger("fairseq2.cli")


def hub_export(fn: Callable[..., Any], script: str) -> "Callable[..., Any]":
    import torchsnapshot

    from fairseq2.cli import DynamicModule

    def hub_entry_point(snapshot_dir: str, device) -> Any:
        assert Path(snapshot_dir).exists(), f"Snapshot {snapshot_dir} not found."
        env = fairseq2.distributed.env(Path(snapshot_dir), device=device)
        # theoritically we don't need to reload hubconf.py,
        # but torchhub isn't passing the module to us,
        # so unless we want to grab it from the calling stack frame,
        # we need to reload it again.
        module = DynamicModule.from_script(Path(script))
        module["env"] = env
        task = module.call_fn(fn.__name__, caller="inference")
        snapshot = torchsnapshot.Snapshot(path=str(snapshot_dir))
        snapshot.restore(task.state_dict_for_inference())  # type: ignore

        return task

    return hub_entry_point


class XP(NamedTuple):
    script: Path
    config_file: Path
    overrides: Sequence[str]


class DynamicModule:
    @staticmethod
    def from_script(
        script: Path,
        overrides: List[str] = [],
        name: str = "",
        yaml_config: Optional[Path] = None,
    ) -> "DynamicModule":
        import fairseq2.cli.defaults

        module = _load_module(script, name)
        _extends_module(module, fairseq2.cli.defaults)
        m = DynamicModule(module, {})
        if yaml_config:
            assert yaml_config.exists(), f"{yaml_config} not found !"
        else:
            yaml_config = script.with_suffix(".yaml")
        if yaml_config.exists():
            m._load_flat_yaml(yaml_config)
        m._add_args(overrides)
        return m

    @staticmethod
    def from_module(
        module: Any, yaml_config: Path, overrides: List[str] = []
    ) -> "DynamicModule":
        m = DynamicModule(module, {})
        m._load_flat_yaml(yaml_config)
        m._add_args(overrides)
        return m

    def __init__(self, module: Any, args: Dict[str, str]):
        self.module = module
        self._raw_args = args
        self._cache: Dict[str, Any] = {}
        self._pending: Dict[str, str] = {}

    def _add_args(self, args: List[str] = []) -> None:
        arg_dict = self._raw_args
        for arg in args:
            k, v = arg.split("=", 1)  # TODO: nice error message
            arg_dict[k] = v

    def _load_flat_yaml(self, yaml_conf: Path) -> None:
        assert yaml_conf.exists(), f"Yaml config not found: {yaml_conf}"
        cache = self._cache
        log.info(f"reloading {yaml_conf}")
        # TODO: safe load -> Path, Env, Device
        conf = yaml.load(yaml_conf.read_text(), Loader=yaml.Loader)
        for key, val in conf.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    cache[f"{key}.{k}"] = v
            else:
                cache[key] = val

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def __getitem__(self, key: str) -> Any:
        k = key.split(".", 1)[-1]
        prefixed_key = key
        if prefixed_key in self._cache:
            return self._cache[prefixed_key]
        else:
            return self._cache[k]

    def _check_no_recursive(self, name: str, caller: str) -> None:
        if name not in self._pending:
            self._pending[name] = caller
            return

        callers_loop = [name, caller]
        prev_caller = callers_loop[-1]
        while prev_caller != name:
            prev_caller = self._pending[prev_caller]
            callers_loop.append(prev_caller)
        loop = " -> ".join(callers_loop[::-1])
        raise Exception("Dependency loop detected: " + loop)

    def call_fn(self, name: str, *, caller: str) -> Any:
        if name in self._cache:
            return self._cache[name]

        fn = getattr(self.module, name)
        if not fn:
            raise ValueError(
                f"Can't create a value for argument {name:!r}. No function {name:!r} found."
            )

        self._check_no_recursive(name, caller)

        # Allow to override the function itself using the "@"" syntax.
        fn_override = name in self._raw_args and self._raw_args[name].startswith("@")
        if fn_override:
            fn = resolve_function(name, self._raw_args[name][1:])

        spec = inspect.getfullargspec(fn.__init__ if isinstance(fn, type) else fn)  # type: ignore

        if name in self._raw_args and not fn_override:
            value = _parse(spec.annotations["return"], name, self._raw_args[name])
            self._cache[name] = value
            return value

        missing_args = spec.args
        missing_args = [
            arg for arg in spec.args if arg not in self._cache and arg != "self"
        ]
        missing_args = self._resolve_from_module(name, missing_args)
        missing_args = self._resolve_from_arglist(name, spec, missing_args)

        if missing_args:
            raise ValueError(
                f"{_lineno(fn)} Can't call {name}, missing args: {missing_args}. Try to supply it from CLI {missing_args}=..."
            )
        resolved_args = {
            arg: self[f"{name}.{arg}"] for arg in spec.args if arg != "self"
        }
        res = fn(**resolved_args)
        self._cache[name] = res

        return res

    def _resolve_from_arglist(
        self, name: str, spec: Any, missing_args: List[str]
    ) -> List[str]:
        still_missing = []
        # TODO: handle kw only args
        defaults = dict(zip(reversed(spec.args), reversed(spec.defaults or [])))

        for arg in missing_args:
            prefixed_key = f"{name}.{arg}"
            k = prefixed_key if prefixed_key in self._raw_args else arg

            if k in self._raw_args:
                # Ensure we do the parsing once
                raw_value = self._raw_args[k]
                if arg not in spec.annotations:
                    log.warning(
                        f"No type annotation for {arg} in fn {name}, parsing {raw_value} as floats"
                    )
                value = _parse(spec.annotations.get(arg, float), k, raw_value)
                self._cache[k] = value
                self._cache[prefixed_key] = value
            elif arg in defaults:
                self._cache[prefixed_key] = defaults[arg]
            else:
                still_missing.append(arg)

        return still_missing

    def _resolve_from_module(self, name: str, missing_args: List[str]) -> List[str]:
        still_missing = []
        for arg in missing_args:
            if not hasattr(self.module, arg):
                still_missing.append(arg)
                continue
            self.call_fn(arg, caller=name)

        return still_missing

    def serialize(self, conf_file: Path) -> None:
        assert conf_file.suffix == ".yaml"
        yaml_conf = yaml.dump(self.as_tree())
        conf_file.write_text(yaml_conf)

    def as_tree(self) -> Dict[str, Any]:
        tree: Dict[str, Any] = {}
        for key, val in self._cache.items():
            if hasattr(self.module, key):
                # Only serialize the result of a function call, if it's something
                # that can be easily overridden through CLI.
                if not can_serialize(val):
                    continue

            parts = key.split(".")
            node = tree
            for k in parts[:-1]:
                if k not in node:
                    node[k] = {}
                node = node[k]
            node[parts[-1]] = val
        return tree


def can_serialize(val: Any) -> bool:
    if isinstance(val, (int, float, str, Path, tuple)):
        return True
    if dataclasses.is_dataclass(val):
        return True
    # if isinstance(val, list):
    #     return len(val) == 0 or can_serialize(val[0])
    # if isinstance(val, dict):
    #     return len(val) == 0 or can_serialize(next(iter(val.values())))

    return False


def _load_module(script: Path, name: str = "") -> Any:
    # Prevent accidental overriding of other python packages.
    name = name or "fairseq2.user." + script.stem
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _copy_file(file: Path, workdir: Path) -> None:
    assert workdir.is_dir()
    workdir_file = workdir / file.name
    if workdir_file.resolve() == file.resolve():
        return

    workdir_file.write_bytes(file.read_bytes())


def _parse(t: Type[Any], key: str, raw_value: str) -> Any:
    if t is bool:
        return _parse_bool(raw_value)
    if t is timedelta:
        return _parse_timedelta(raw_value)
    return func_argparse._get_parser(t, [key])(raw_value)


def _parse_bool(boolean: str) -> bool:
    if boolean in ("true", "True", "1"):
        return True
    if boolean in ("false", "False", "0"):
        return False
    raise ValueError(f"Can't parse boolean value {boolean!r}")


TIME_UNITS = {
    "s": timedelta(seconds=1),
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
}


def _parse_timedelta(duration: str) -> timedelta:
    suffix = duration[-1] if duration else ""
    if suffix not in TIME_UNITS:
        raise ValueError(
            f"Time duration should be suffixed by a time unit (s)econds, (m)inutes, (h)ours, (d)ays. Received {duration!r}"
        )

    try:
        d = float(duration[:-1])
        return d * TIME_UNITS[suffix]
    except Exception:
        raise ValueError(f"Can't parse duration. Received {duration!r}")


def _lineno(fn: Callable[..., Any]) -> str:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    code = fn.__code__
    return f"{code.co_filename}:{code.co_firstlineno}:"


def _extends_module(module: Any, defaults: Any) -> None:
    for k in defaults.__all__:
        if not hasattr(module, k):
            setattr(module, k, getattr(defaults, k))


def resolve_function(key: str, qual_name: str) -> Callable[..., Any]:
    if "." not in qual_name:
        raise ValueError(
            f"Function {key}={qual_name!r} isn't known. Please use full qualified name for {qual_name!r}"
        )
    mod_name, fn_name = qual_name.rsplit(".", 1)
    if mod_name not in sys.modules:
        # TODO: we should try to import it ourselves
        raise ValueError(
            f"Can't resolve {key}={qual_name!r}. Module {mod_name} isn't imported. You can explictly import it in you experiment script."
        )

    module = sys.modules[mod_name]
    fn = getattr(module, fn_name, None)
    if fn is None:
        raise ValueError(
            f"Can't resolve {key}={qual_name!r}. Module {mod_name} has no attribute {fn_name}"
        )

    if not callable(fn):
        raise ValueError(
            f"Invalid value received {key}={qual_name!r}. {qual_name!r}({fn}) isn't callable."
        )

    return fn  # type: ignore
