import dataclasses
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type

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
        # TODO: avoid using the workdir to store intermediary file like the SPM.
        module["env"] = env._replace(workdir=module["", "env"].workdir)

        task = module.call_fn(fn.__name__, caller="inference")
        snapshot = torchsnapshot.Snapshot(path=str(snapshot_dir))
        snapshot.restore(task.state_dict_for_inference())  # type: ignore

        return task

    return hub_entry_point


class DynamicModule:
    @staticmethod
    def from_script(
        script: Path, overrides: List[str] = [], name: str = ""
    ) -> "DynamicModule":
        module = _load_module(script, name)
        m = DynamicModule(module, {})
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

    def __getitem__(self, key: Tuple[str, str]) -> Any:
        name, k = key
        prefixed_key = ".".join(key)
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

        fn = getattr(self.module, name)  # TODO: nice error message
        self._check_no_recursive(name, caller)

        spec = inspect.getfullargspec(fn.__init__ if isinstance(fn, type) else fn)  # type: ignore

        if name in self._raw_args:
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
            raise Exception(
                f"{_lineno(fn)} Can't call {name}, missing args: {missing_args}"
            )
        resolved_args = {arg: self[name, arg] for arg in spec.args if arg != "self"}
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
                assert (
                    arg in spec.annotations
                ), f"No type annotation for {arg} in fn {name}, not sure how to parse {raw_value}"
                value = _parse(spec.annotations[arg], k, raw_value)
                self._cache[k] = value
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

    def serialize(self, workdir: Path, script: Path) -> Path:
        _copy_file(script, workdir)
        tree_conf = self.as_tree()
        yaml_conf = yaml.dump(tree_conf)
        print(yaml_conf)
        workdir_script = workdir / script.name
        workdir_script.with_suffix(".yaml").write_text(yaml_conf)
        return workdir_script

    def as_tree(self) -> Dict[str, Any]:
        tree: Dict[str, Any] = {}
        for key, val in self._cache.items():
            if hasattr(self.module, key):
                # Only serialize the result of a function call, if it's something
                # that can be easily overridden through CLI.
                if not isinstance(val, (int, float, str, Path, NamedTuple)):
                    if not dataclasses.is_dataclass(val):
                        continue

            parts = key.split(".")
            node = tree
            for k in parts[:-1]:
                if k not in node:
                    node[k] = {}
                node = node[k]
            node[parts[-1]] = val
        return tree


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
    # TODO: handle Device, Dtype, ...
    return func_argparse._get_parser(t, [key])(raw_value)


def _lineno(fn: Callable[..., Any]) -> str:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    code = fn.__code__
    return f"{code.co_filename}:{code.co_firstlineno}:"
