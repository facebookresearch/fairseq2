from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import importlib
import inspect
import io
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    NamedTuple,
    Optional,
    Sequence,
)

import func_argparse
import torch
import yaml

import fairseq2.cli.distributed
from fairseq2.cli import yaml_ext

log = logging.getLogger("fairseq2.cli")
AnyCallable = Callable[..., Any]


def fairseq2_hub(snapshot_dir: str, device: Optional[torch.device] = None) -> Any:
    """
    Tells torch.hub.load how to reload a fairseq2 task.

    This function needs to exposed in the training script otherwise torch.hub won't find it.
    """
    import torchsnapshot

    assert Path(snapshot_dir).exists(), f"Snapshot {snapshot_dir} not found."
    env = fairseq2.cli.distributed.env(device=device)
    # theoretically we don't need to reload hubconf.py,
    # but torchhub isn't passing the module to us,
    # so unless we want to grab it from the calling stack frame,
    # we need to reload it again.
    module = XpScript.from_script(Path(snapshot_dir) / "hubconf.py")
    module["env"] = env
    task = module.call_fn("task", caller="torch.hub.load")
    snapshot = torchsnapshot.Snapshot(path=str(snapshot_dir))
    # Note: we could allow the user more control on what needs to be reloaded.
    snapshot.restore(task.state_dict_for_inference())  # type: ignore

    return task


@dataclasses.dataclass(frozen=True)
class Xp:
    """Represents the current experiment being run by fairseq2"""

    script: Path
    """Path to the experiment script"""

    config_file: Path
    """Yaml file representing all hyper-parameters used"""

    overrides: Sequence[str]
    """List of hyper-parameters set from the CLI"""

    sha_key: str = dataclasses.field(init=False)
    """A hash of the experiment script and its hyper-parameters"""

    def __post_init__(self) -> None:
        if hasattr(self, "sha_key"):
            return
        code = self.script.read_text()
        code += ";".join(self.overrides)
        sha = hashlib.sha256(code.encode("utf-8")).hexdigest()[:8]
        object.__setattr__(self, "sha_key", sha)


class FnSpec(NamedTuple):
    fn: AnyCallable
    spec: inspect.FullArgSpec
    doc: Optional[str]


class XpScript:
    @staticmethod
    def from_script(
        script: Path,
        overrides: list[str] = [],
        yaml_config: Optional[Path] = None,
    ) -> "XpScript":
        import fairseq2.cli.defaults

        # It's important to be consistent here, we always load the module under
        # the name of fairseq2.user.hubconf.
        module = _load_module(script, name="fairseq2.user.hubconf")
        _extends_module(module, fairseq2.cli.defaults)
        m = XpScript(module, {})
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
        module: Any, yaml_config: Path, overrides: list[str] = []
    ) -> "XpScript":
        m = XpScript(module, {})
        m._load_flat_yaml(yaml_config)
        m._add_args(overrides)
        return m

    def __init__(self, module: Any, args: Dict[str, str]):
        self.module = module
        self._raw_args = args
        self._cache: Dict[str, Any] = {}
        self._pending: Dict[str, str] = {}

    def _add_args(self, args: list[str] = []) -> None:
        arg_dict = self._raw_args
        for arg in args:
            k, v = arg.split("=", 1)  # TODO: nice error message
            arg_dict[k] = v

    def _load_flat_yaml(self, yaml_conf: Path) -> None:
        assert yaml_conf.exists(), f"Yaml config not found: {yaml_conf}"
        cache = self._cache
        log.info(f"reloading {yaml_conf}")
        # TODO: safe load -> Path, Env, torch.device
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

    def _check_no_cycles(self, name: str, caller: str) -> None:
        # TODO: this should be done in "dag()"
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
        # TODO use "dag"
        if name in self._cache:
            return self._cache[name]

        fn = self._resolve_fn(name)
        if fn is None:
            raise ValueError(
                f"Can't create a value for argument {name!r}. No function {name!r} found."
            )
        spec = fn.spec

        self._check_no_cycles(name, caller)

        fn_res_override = not self._raw_args.get(name, "@").startswith("@")
        if fn_res_override:
            value = _parse(spec.annotations["return"], name, self._raw_args[name])
            self._cache[name] = value
            return value

        missing_args = spec.args
        missing_args = [
            arg for arg in spec.args if arg not in self._cache and arg != "self"
        ]
        missing_args = self._resolve_from_module(name, missing_args)
        _, missing_args = self._resolve_from_arglist(name, spec, missing_args)

        if missing_args:
            raise ValueError(
                f"{_lineno(fn.fn)} Can't call {name}, missing args: {missing_args}. Try to supply it from CLI: {missing_args[0]}=..."
            )
        resolved_args = {
            arg: self[f"{name}.{arg}"] for arg in spec.args if arg != "self"
        }
        res = fn.fn(**resolved_args)
        self._cache[name] = res

        return res

    @functools.lru_cache()
    def _resolve_fn(self, name: str) -> Optional[FnSpec]:
        # Allow to override the function itself using the "@" syntax.
        fn_override = self._raw_args.get(name, "").startswith("@")
        if fn_override:
            fn = resolve_function(name, self._raw_args[name][1:])
        else:
            fn = getattr(self.module, name, None)  # type: ignore
            if fn is None:
                return None

        # fn can actually be a class. In that case we want the type hints of the constructor.
        spec_fn = fn.__init__ if isinstance(fn, type) else fn  # type: ignore[misc]
        # TODO: how does that work with literals ?
        spec = inspect.getfullargspec(spec_fn)
        # For the documentation we prefer the class doc, and then the constructor doc.
        doc = _resolve_doc(fn)
        return FnSpec(fn, spec, doc)

    def _resolve_from_arglist(
        self, name: str, spec: Any, missing_args: list[str]
    ) -> tuple[list[str], list[str]]:
        resolved_args = []
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
                        f"No type annotation for {arg}={raw_value!r} in fn {name}"
                    )
                value = _parse(spec.annotations.get(arg, str), k, raw_value)
                self._cache[k] = value
                self._cache[prefixed_key] = value
                resolved_args.append(k)
            elif arg in defaults:
                self._cache[prefixed_key] = defaults[arg]
                resolved_args.append(k)
            else:
                still_missing.append(arg)

        return resolved_args, still_missing

    def _resolve_from_module(self, name: str, missing_args: list[str]) -> list[str]:
        still_missing = []
        for arg in missing_args:
            if not hasattr(self.module, arg):
                still_missing.append(arg)
                continue
            self.call_fn(arg, caller=name)

        return still_missing

    def serialize(self, conf_file: Path) -> None:
        assert conf_file.suffix == ".yaml"
        yaml_conf = yaml.dump(self.as_tree(), Dumper=yaml_ext.Dumper)
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

    def help(self, *entrypoints: str, hidden: list[str] = []) -> str:
        out = io.StringIO()
        if self.module.__doc__:
            print(self.module.__doc__.strip(), end="\n\n")

        dag = self.dag(entrypoints)
        documented: set[str] = set()
        for name in dag.breadth_first_traversal():
            if name in hidden:
                continue
            if name in ("__root__", "__sink__", "self"):
                continue
            if dag[name] == ["__sink__"]:
                continue

            documented.add(name)
            fn = self._resolve_fn(name)
            if fn is None:
                continue

            ta = _type_annotation(fn, "return")

            # Only keep the first line of fn documentation
            # TODO: show all docstring until first arg
            fn_doc = fn.doc or "UNDOCUMENTED"
            fn_doc = fn_doc.split("\n")[0].strip()
            print(f"**{name}** {ta}:", fn_doc, file=out)

            if not fn.spec.args:
                print("\t(no settings)", file=out)
                continue

            args_docs = _get_arguments_description(fn)
            deps = []
            for arg in fn.spec.args:
                key = f"{name}.{arg}"
                if arg == "self" or arg in hidden:
                    continue
                elif arg in documented:
                    deps.append(f"{arg} (see above)")
                    continue
                elif key in self._cache:
                    message = _join(args_docs.get(arg), f"(default={self._cache[key]})")
                elif dag[arg] == ["__sink__"]:
                    message = _join(args_docs.get(arg), "(REQUIRED)")
                else:
                    deps.append(f"{arg} (see below)")
                    continue
                ta = _type_annotation(fn, arg)
                print(f"\t{name}.{arg} {ta}:", message, file=out)
            if deps:
                print("\tUses", ", ".join(deps), file=out)
            print(file=out)

        return out.getvalue()

    def dag(self, entrypoints: Sequence[str]) -> "DAG":
        _dag = DAG(entrypoints)
        for e in entrypoints:
            _dag._add_incomplete(e)

        while True:
            name = _dag._next_incomplete_node()
            if name is None:
                break

            fn = self._resolve_fn(name)
            assert fn is not None, f"{name} isn't a valid entrypoint"

            spec = fn.spec
            fn_res_override = not self._raw_args.get(name, "@").startswith("@")
            if fn_res_override:
                # If fn result is explicit set on CLI,
                # we don't look into how fn is supposed to be called.
                value = _parse(spec.annotations["return"], name, self._raw_args[name])
                self._cache[name] = value
                continue

            _dag[name] = spec.args
            if not spec.args:
                continue

            missing_args = []
            for arg in spec.args:
                if arg == "self":
                    continue
                elif hasattr(self.module, arg):
                    _dag._add_incomplete(arg)
                elif arg not in _dag:
                    missing_args.append(arg)

            resolved_args, missing_args = self._resolve_from_arglist(
                name, spec, missing_args
            )
            for arg in missing_args:
                _dag[arg] = ["__sink__"]
            for arg in resolved_args:
                _dag[arg] = []

        return _dag


if TYPE_CHECKING:
    # this is only processed by mypy
    ODict = collections.OrderedDict[str, "list[str]"]
else:
    ODict = collections.OrderedDict


class DAG(ODict):
    def __init__(self, entrypoints: Sequence[str]) -> None:
        super().__init__([("__root__", list(entrypoints))])

    def _next_incomplete_node(self) -> Optional[str]:
        return next((key for key, neighbors in self.items() if neighbors is None), None)

    def _add_incomplete(self, key: str) -> None:
        if key not in self:
            self[key] = None  # type: ignore[assignment]

    def breadth_first_traversal(self) -> Iterator[str]:
        queue = list(self["__root__"])
        done = {"__sink__"}
        while queue:
            x = queue.pop(0)
            if x in done:
                continue
            yield x
            done.add(x)
            neighbors = self.get(x, [])
            for x in neighbors:
                if x not in done:
                    queue.append(x)


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


def _parse(t: type[Any], key: str, raw_value: str) -> Any:
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


def _lineno(fn: AnyCallable) -> str:
    if isinstance(fn, type):
        fn = fn.__init__  # type: ignore
    code = fn.__code__
    return f"{code.co_filename}:{code.co_firstlineno}:"


def _extends_module(module: Any, defaults: Any) -> None:
    for k in defaults.__all__:
        if not hasattr(module, k):
            setattr(module, k, getattr(defaults, k))


def _type_annotation(fn: FnSpec, name: str) -> str:
    func = fn.fn
    if isinstance(func, functools.partial):
        func = func.func

    if name == "return" and isinstance(func, type):
        type_annotation = qualname(func)
    else:
        type_annotation = fn.spec.annotations.get(name, "?")
        if isinstance(type_annotation, type):
            type_annotation = qualname(type_annotation)
    t = str(type_annotation)
    t = t.replace("typing.", "")
    return f"({t})"


def qualname(t: type[Any]) -> str:
    if t.__module__ == "builtins":
        return t.__name__
    return ".".join((t.__module__, t.__name__))


def resolve_function(key: str, qual_name: str) -> AnyCallable:
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


def _get_arguments_description(fn: FnSpec) -> Dict[str, str]:
    """Returns a description for each argument."""
    if not fn.doc:
        return {}
    descriptions = {}
    lines = list(filter(None, (l.strip("-* ") for l in fn.doc.splitlines())))
    for a in fn.spec.args:
        # TODO: some arguments may have more than one line of documentation.
        doc = next((l[len(a) :].strip(" :") for l in lines if l.startswith(a)), None)
        descriptions[a] = doc or ""

    return descriptions


def _join(*parts: Optional[str]) -> str:
    return " ".join((p for p in parts if p))


def _resolve_doc(fn: AnyCallable) -> str:
    if isinstance(fn, functools.partial):
        fn = fn.func

    init_doc = ""
    # tuple/NamedTuple have a very generic docstring
    if isinstance(fn, type) and fn.__init__ != tuple.__init__:  # type: ignore[misc]
        init_doc = fn.__init__.__doc__  # type: ignore[misc]

    return init_doc or fn.__doc__ or ""
