import contextlib
import importlib
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generic, Iterator, List, Type, TypeVar

import torchsnapshot

from fairseq2.typing import Device

try:
    import wandb as _  # noqa: F401
except Exception:
    # Wandb does weird thing when imported and it can actually fail
    # TODO: the logger shouldn't be in the hubstate itself
    pass

T = TypeVar("T")


class HubState(Generic[T]):
    """Saves extra state needed to build your Task.
    Make sure the HubState is saved with your model weights.

    HubState.add_state(model, **kwargs)

    TODO: reconsider all this.
    The thing I want is:
    * easy model loading: the train script should be a valid hubconf.py file
    * easy resuming: resume training with the train script
    * easy eval: with train script I can reload the model weights and run eval
    * easy inference: with train script I can load the model
    """

    def __init__(
        self,
        cls: Type[T],
        original_main: Path,
        main_code: str,
        kwargs: Dict[str, Any],
        dependencies: List[str] = [],
        key: str = "__hubstate__",
    ):
        """Init me with `self.extra_state = HubState(self, __file__, locals())`.

        self.__dict__.update(state_dict)
        """
        self.cls = cls
        self.original_main = original_main
        self.main_code = main_code
        self.key = key
        self.kwargs = kwargs
        self.dependencies = dependencies or []
        self._kwargs_pkl: Dict[str, bytes] = {}

    @staticmethod
    def add_state(
        instance: T,
        kwargs: Dict[str, Any],
        dependencies: List[str] = [],
        key: str = "__hubstate__",
    ) -> None:
        # We save the content of __main__ as part of the object, because
        # pickle doesn't work with classes defined in __main__ and that's
        # something we want to encourage.
        import __main__

        original_main = __main__.__file__
        kwargs.pop("self", None)
        kwargs.pop("__class__", None)
        if Path(original_main).exists():
            main_code = Path(original_main).read_text()
        else:
            main_code = ""
        hub_state = HubState(
            type(instance), Path(original_main), main_code, kwargs, dependencies, key
        )
        setattr(instance, key, hub_state)

    @staticmethod
    def empty() -> "HubState[None]":
        return HubState(type(None), Path(__file__), "", {}, [])

    def state_dict(self) -> Dict[str, Any]:
        if not self._kwargs_pkl:
            for k, arg in self.kwargs.items():
                self._kwargs_pkl[k] = pickle.dumps(arg)

        state = dict(self.__dict__)
        state.pop("kwargs")
        state["cls"] = self.cls.__qualname__
        state["cls_module"] = self.cls.__module__

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        with tempfile.TemporaryDirectory() as tmp_path:
            main = self._load_main_module(Path(tmp_path))
        with change_main(main):
            for k, v in self._kwargs_pkl.items():
                self.kwargs[k] = pickle.loads(v)

        # currently cls is still the class name not the actual class
        assert isinstance(self.cls, str)
        if self.cls_module == "__main__":
            cls_module = main
        else:
            cls_module = importlib.import_module(self.cls_module, self.cls)
        self.cls = getattr(cls_module, self.cls)

    def _load_main_module(self, tmp_path: Path) -> Any:
        if not self.main_code:
            return None
        tmp_file = tempfile.NamedTemporaryFile(
            dir=tmp_path, prefix=self.original_main.stem, suffix=".py"
        )
        module_file = tmp_file.name
        Path(module_file).write_text(self.main_code)

        # Import the model code
        spec = importlib.util.spec_from_file_location(
            self.original_main.stem, module_file
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def gen_hubconf(self, snapshot_dir: Path) -> None:
        """Writes a hub_conf.py in the directory that will be used to make the snapshots."""
        hub_conf = f"""
dependencies = ["fairseq2"]

from fairseq2.typing import Device
from fairseq2.hub import HubState

def model(name: str = "valid_best", device: Device = Device("cuda:0")):

    return HubState.load_snapshot(__file__, name, "{self.key}", device)
        """
        (snapshot_dir / "hubconf.py").write_text(hub_conf)

    def materialize(self) -> T:
        # TODO: this feel redoing some pickle work. Should we pickle the full instance ?
        task = self.cls(**self.kwargs)
        setattr(task, self.key, self)
        return task

    @staticmethod
    def load_snapshot(hub_conf: str, name: str, key: str, device: Device) -> Any:
        # TODO: add overrides
        hub_conf_dir = Path(hub_conf).parent
        snapshot_dir = hub_conf_dir / name
        assert snapshot_dir.exists(), f"Snapshot dir not found: {snapshot_dir}"

        snapshot = torchsnapshot.Snapshot(path=str(snapshot_dir))
        hub_state = HubState.empty()
        snapshot.restore({key: hub_state})

        if "device" in hub_state.kwargs:
            hub_state.kwargs["device"] = device

        task: Any = hub_state.materialize()
        app_state = task.app_state()
        # avoid reloading the hub_state again, by removing it from the app_state
        app_state.pop(hub_state.key, None)
        snapshot.restore(app_state)
        return task


@contextlib.contextmanager
def change_main(module: Any) -> Iterator[None]:
    main = sys.modules["__main__"]
    sys.modules["__main__"] = module
    try:
        yield
    finally:
        sys.modules["__main__"] = main
