# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Generic, TypeVar, cast, final, overload

import torch
from torch import Tensor
from torch.nn import Module

from fairseq2.assets import AssetCard, AssetCardError, AssetNotFoundError, AssetStore
from fairseq2.data_type import DataType
from fairseq2.device import CPU, Device, get_current_device
from fairseq2.error import InternalError
from fairseq2.gang import Gangs, create_fake_gangs
from fairseq2.models.family import ModelFamily
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.runtime.lookup import Lookup
from fairseq2.utils.warn import _warn_deprecated, _warn_progress_deprecated

ModelT = TypeVar("ModelT", bound=Module)

ModelConfigT = TypeVar("ModelConfigT")


@final
class ModelHub(Generic[ModelT, ModelConfigT]):
    """
    Provides a high-level interface for loading and creating models from a
    specific model family.

    This class serves as the primary entry point for working with models of a
    particular family (e.g., LLaMA, Qwen, etc.). It handles model discovery,
    configuration loading, and model instantiation.
    """

    def __init__(self, family: ModelFamily, asset_store: AssetStore) -> None:
        self._family = family
        self._asset_store = asset_store

    def iter_cards(self) -> Iterator[AssetCard]:
        """
        Iterates over all asset cards belonging to this model family.

        .. code:: python

            from fairseq2.models.qwen import get_qwen_model_hub

            # List all available Qwen models.
            for card in get_qwen_model_hub().iter_cards():
                print(f"Model: {card.name}")
        """
        return self._asset_store.find_cards("model_family", self._family.name)

    def get_archs(self) -> set[str]:
        """
        Returns the set of supported model architectures in this family.

        .. code:: python

            from fairseq2.models.qwen import get_qwen_model_hub

            # List all available Qwen architectures.
            for arch in get_qwen_model_hub().get_archs():
                print(f"Architecture: {arch}")
        """
        return self._family.get_archs()

    def get_arch_config(self, arch: str) -> ModelConfigT:
        """
        Returns the configuration for the specified model architecture.

        .. code:: python

            from fairseq2.models.qwen import get_qwen_model_hub

            config = get_qwen_model_hub().get_arch_config("qwen25_7b")

            print(config)

        :raises ModelArchitectureNotKnownError: If ``arch`` is not a known
            architecture in this family.
        """
        config = self.maybe_get_arch_config(arch)
        if config is None:
            raise ModelArchitectureNotKnownError(arch, self._family.name)

        return config

    def maybe_get_arch_config(self, arch: str) -> ModelConfigT | None:
        """
        Returns the configuration for the specified model architecture, or
        ``None`` if not known.
        """
        config = self._family.maybe_get_arch_config(arch)

        return cast(ModelConfigT | None, config)

    def get_model_config(self, card: AssetCard | str) -> ModelConfigT:
        """
        Returns the model configuration from an asset card.

        This method loads the base architecture configuration and applies any
        model-specific overrides specified in the asset card.

        As a convenience, this method also accepts an asset name instead of an
        asset card.

        .. code:: python

            from fairseq2.assets import get_asset_store
            from fairseq2.models.qwen import QwenConfig, get_qwen_model_hub

            card = get_asset_store().retrieve_card("qwen25_7b_instruct")

            qwen_config = get_qwen_model_hub().get_model_config(card)

            # As a convenience, the card can be omitted and the model name can
            # be passed directly to `get_model_config()`:
            qwen_config = get_qwen_model_hub().get_model_config("qwen25_7b_instruct")

            print(qwen_config)

        :raises ModelNotKnownError: If ``card`` is a string and no asset card
            with that name exists.

        :raises AssetCardError: If the asset card's model family does not match
            this hub's family.
        """
        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise ModelNotKnownError(name) from None
        else:
            name = card.name

        family_name = card.field("model_family").as_(str)

        if family_name != self._family.name:
            msg = f"family field of the {name} asset card is expected to be {self._family.name}, but is {family_name} instead."

            raise AssetCardError(name, msg)

        config = self._family.get_model_config(card)

        return cast(ModelConfigT, config)

    @overload
    def create_new_model(
        self,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        dtype: DataType | None = None,
        meta: bool = False,
    ) -> ModelT: ...

    @overload
    def create_new_model(
        self,
        config: ModelConfigT,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
        meta: bool = False,
    ) -> ModelT: ...

    def create_new_model(
        self,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        meta: bool = False,
    ) -> ModelT:
        """
        Creates a new model instance with the specified configuration.

        This method creates a fresh model without loading any pretrained weights.
        The model will be initialized with random parameters according to the
        architecture's default initialization scheme.

        If ``gangs`` is provided, it will be used to apply parallelism (i.e.
        model parallelism) to the initialized model. If the model family does
        not support a certain parallelism strategy, that strategy will be
        ignored. For instance if ``gangs.tp.size > 1``, but the model does not
        support tensor parallelism, the model will be instantiated with regular
        attention and feed-forward network blocks. If ``None``, the whole model
        will be initialized without any parallelism.

        If ``device`` is provided, the model will be created on the specified
        device; otherwise, the device returned from :func:`torch.get_default_device`
        will be used. Note that ``device`` and ``gangs`` cannot be provided
        together. If ``gangs`` is provided, ``gangs.root.device`` will be used.

        If ``dtype`` is provided, it will be used as the default data type of
        the model parameters and buffers; otherwise, the data type returned from
        :func:`torch.get_default_dtype` will be used.

        If ``meta`` is ``True``, the model will be created on the meta device
        for memory-efficient initialization. Only supported if the model family
        supports meta device.

        .. code:: python

            from fairseq2.models.qwen import QwenConfig, get_qwen_model_hub

            # Use the default Qwen configuration except the number of
            # decoder layers.
            config = QwenConfig(num_layers=16)

            qwen_model = get_qwen_model_hub().create_new_model(config)

        :raises ValueError: If both ``gangs`` and ``device`` are provided.

        :raises NotSupportedError: If ``meta`` is ``True`` but the model family
            doesn't support meta device.
        """
        gangs = _get_effective_gangs(gangs, device)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._family.create_new_model(config, gangs, dtype, meta)

        return cast(ModelT, model)

    @overload
    def load_model(
        self,
        card: AssetCard | str,
        *,
        gangs: Gangs | None = None,
        dtype: DataType | None = None,
        config: ModelConfigT | None = None,
        mmap: bool = False,
        progress: bool | None = None,
    ) -> ModelT: ...

    @overload
    def load_model(
        self,
        card: AssetCard | str,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
        config: ModelConfigT | None = None,
        mmap: bool = False,
        progress: bool | None = None,
    ) -> ModelT: ...

    def load_model(
        self,
        card: AssetCard | str,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        config: ModelConfigT | None = None,
        mmap: bool = False,
        progress: bool | None = None,
    ) -> ModelT:
        """
        Loads a pretrained model from an asset card.

        This method downloads the model checkpoint (if necessary) and loads the
        pretrained weights into a model instance. The model architecture and
        configuration are determined from the asset card metadata.

        As a convenience, this method also accepts an asset name instead of an
        asset card.

        If ``gangs`` is provided, it will be used to apply parallelism (i.e.
        model parallelism) to the initialized model. If the model family does
        not support a certain parallelism strategy, that strategy will be
        ignored. For instance if ``gangs.tp.size > 1``, but the model does not
        support tensor parallelism, the model will be instantiated with regular
        attention and feed-forward network blocks. If ``None``, the whole model
        will be initialized without any parallelism.

        If ``device`` is provided, the model will be created on the specified
        device; otherwise, the device returned from :func:`torch.get_default_device`
        will be used. Note that ``device`` and ``gangs`` cannot be provided
        together. If ``gangs`` is provided, ``gangs.root.device`` will be used.

        If ``dtype`` is provided, it will be used as the default data type of
        the model parameters and buffers; otherwise, the data type returned from
        :func:`torch.get_default_dtype` will be used.

        If ``config`` is provided, it overrides the default model configuration
        from the asset card. If ``None``, uses the configuration specified in
        the card. Typically used to perform slight adjustments to the model
        configuration such as tuning dropout probabilities without changing the
        architecture.

        If ``mmap`` is ``True``, the model checkpoint will be memory-mapped. This
        can reduce memory usage but may cause slower load times on some systems.

        ``progress`` is deprecated and will be removed in v0.13. Use
        ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress``
        parameter of :func:`init_fairseq` to disable progress bars.

        .. code:: python

            from fairseq2.assets import get_asset_store
            from fairseq2.models.qwen import QwenConfig, get_qwen_model_hub

            card = get_asset_store().retrieve_card("qwen25_7b_instruct")

            qwen_model = get_qwen_model_hub().load_model(card)

            # As a convenience, the card can be omitted and the model name can
            # be passed directly to `load_model()`:
            qwen_model = get_qwen_model_hub().load_model("qwen25_7b_instruct")

        :raises ModelNotKnownError: If ``card`` is a string and no asset card
            with that name exists.

        :raises AssetCardError: If the asset card's model family doesn't match
            this hub's family.

        :raises ValueError: If both ``gangs`` and ``device`` are provided.
        """
        _warn_progress_deprecated(progress)

        gangs = _get_effective_gangs(gangs, device)

        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise ModelNotKnownError(name) from None
        else:
            name = card.name

        family_name = card.field("model_family").as_(str)

        if family_name != self._family.name:
            msg = f"family field of the {name} asset card is expected to be {self._family.name}, but is {family_name} instead."

            raise AssetCardError(name, msg)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._family.load_model(card, gangs, dtype, config, mmap, progress=True)

        return cast(ModelT, model)

    @overload
    def load_custom_model(
        self,
        path: Path,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        dtype: DataType | None = None,
        mmap: bool = False,
        restrict: bool | None = None,
        progress: bool | None = None,
    ) -> ModelT: ...

    @overload
    def load_custom_model(
        self,
        path: Path,
        config: ModelConfigT,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
        mmap: bool = False,
        restrict: bool | None = None,
        progress: bool | None = None,
    ) -> ModelT: ...

    def load_custom_model(
        self,
        path: Path,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        device: Device | None = None,
        dtype: DataType | None = None,
        mmap: bool = False,
        restrict: bool | None = None,
        progress: bool | None = None,
    ) -> ModelT:
        """
        Loads a model from a custom checkpoint file.

        This method is useful for loading models from custom training runs or
        third-party checkpoints that are not available through the asset store.

        ``config`` specifies the model configuration. It must match the
        architecture of the saved checkpoint.

        If ``gangs`` is provided, it will be used to apply parallelism (i.e.
        model parallelism) to the initialized model. If the model family does
        not support a certain parallelism strategy, that strategy will be
        ignored. For instance if ``gangs.tp.size > 1``, but the model does not
        support tensor parallelism, the model will be instantiated with regular
        attention and feed-forward network blocks. If ``None``, the whole model
        will be initialized without any parallelism.

        If ``device`` is provided, the model will be created on the specified
        device; otherwise, the device returned from :func:`torch.get_default_device`
        will be used. Note that ``device`` and ``gangs`` cannot be provided
        together. If ``gangs`` is provided, ``gangs.root.device`` will be used.

        If ``dtype`` is provided, it will be used as the default data type of
        the model parameters and buffers; otherwise, the data type returned from
        :func:`torch.get_default_dtype` will be used.

        If ``mmap`` is ``True``, the model checkpoint will be memory-mapped. This
        can reduce memory usage but may cause slower load times on some systems.

        If ``restrict`` is ``True``, pickle (if used) will be restricted to load
        only tensors and types that can be safely serialized and deserialized.
        If ``None``, the default restriction setting of the family will be used.

        ``progress`` is deprecated and will be removed in v0.13. Use
        ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress``
        parameter of :func:`init_fairseq` to disable progress bars.

        .. code:: python

            from fairseq2.models.qwen import QwenConfig, get_qwen_model_hub

            checkpoint_path = ...

            # The checkpoint contains a Qwen model with 16 decoder layers.
            config = QwenConfig(num_layers=16)

            qwen_model = get_qwen_model_hub().load_custom_model(checkpoint_path, config)

        :raises ValueError: If both ``gangs`` and ``device`` are provided.

        :raises FileNotFoundError: If the checkpoint file does not exist.

        :raises ModelCheckpointError: If the checkpoint format is not valid or
            incompatible with the model.
        """
        _warn_progress_deprecated(progress)

        gangs = _get_effective_gangs(gangs, device)

        if dtype is None:
            dtype = torch.get_default_dtype()

        model = self._family.load_custom_model(
            path, config, gangs, dtype, mmap, restrict, progress=True
        )

        return cast(ModelT, model)

    def iter_checkpoint(
        self,
        path: Path,
        config: ModelConfigT,
        *,
        gangs: Gangs | None = None,
        mmap: bool = False,
        restrict: bool | None = None,
    ) -> Iterator[tuple[str, Tensor]]:
        """
        Lazily loads parameters from the specified model checkpoint path.

        Yields tensors one at a time to minimize memory usage if the underlying
        checkpoint format allows it.

        This method provides low-level access to checkpoint contents without
        loading the full model into memory. It's useful for checkpoint inspection,
        custom loading logic, or memory-efficient parameter processing.

        ``config`` specifies the model configuration used to determine the
        expected parameter structure in the checkpoint.

        If ``gangs`` is provided, it is used to determine the distributed target
        configuration and to shard yielded parameters accordingly. If ``None``,
        no sharding will be performed and full parameters will be yielded.

        If ``mmap`` is ``True``, the checkpoint will be memory-mapped. This can
        reduce memory usage but may cause slower load times on some systems.

        If ``restrict`` is ``True``, pickle (if used) will be restricted to load
        only tensors and types that can be safely serialized and deserialized.
        If ``None``, the default restriction setting of the family will be used.

        Yields pairs of ``(parameter name, parameter)`` for each parameter in
        the checkpoint.

        :raises FileNotFoundError: If the checkpoint file does not exist.

        :raises ModelCheckpointError: If the checkpoint format is not valid.
        """
        gangs = _get_effective_gangs(gangs, device=CPU)

        return self._family.iter_checkpoint(path, config, gangs, mmap, restrict)


@final
class ModelHubAccessor(Generic[ModelT, ModelConfigT]):
    """
    Creates a :class:`ModelHub` instance when called.

    This class provides a strongly-typed way to access model hubs. Its direct
    use is meant for model authors rather than library users.

    See ``src/fairseq2/models/llama/hub.py`` as an example.

    .. code::
        :caption: The use of `ModelHubAccessor` for model authors

        from fairseq2.models import ModelHubAccessor

        # Defined in the Python module where the model is implemented.
        get_my_model_hub = ModelHubAccessor(
            family_name="my_model_family", kls=MyModel, config_kls=MyModelConfig
        )

        # `get_my_model_hub()` is treated as a standalone function by the model
        # users in other parts of the code like below:
        model_config = MyModelConfig()

        model = get_my_model_hub().create_new_model(model_config)
    """

    def __init__(
        self, family_name: str, kls: type[ModelT], config_kls: type[ModelConfigT]
    ) -> None:
        self._family_name = family_name
        self._kls = kls
        self._config_kls = config_kls

    def __call__(self) -> ModelHub[ModelT, ModelConfigT]:
        resolver = get_dependency_resolver()

        asset_store = resolver.resolve(AssetStore)

        name = self._family_name

        family = resolver.resolve_optional(ModelFamily, key=name)
        if family is None:
            raise ModelFamilyNotKnownError(name)

        if not issubclass(family.kls, self._kls):
            raise InternalError(
                f"`kls` is `{self._kls}`, but the type of the {name} model family is `{family.kls}`."
            )

        if not issubclass(family.config_kls, self._config_kls):
            raise InternalError(
                f"`config_kls` is `{self._config_kls}`, but the configuration type of the {name} model family is `{family.config_kls}`."
            )

        return ModelHub(family, asset_store)


class ModelNotKnownError(Exception):
    """Raised when a requested model name is not found in the asset store."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known model.")

        self.name = name


class ModelFamilyNotKnownError(Exception):
    """Raised when a requested model family is not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is not a known model family.")

        self.name = name


class ModelArchitectureNotKnownError(Exception):
    """
    Raised when a requested model architecture is not supported by a model family.
    """

    def __init__(self, arch: str, family: str | None = None) -> None:
        """
        ``family`` defaults to ``None`` due to backwards-compatibility. New code
        must specify a model family when raising this error.
        """
        if family is None:
            _warn_deprecated(
                "`ModelArchitectureNotKnownError` will require a `family` argument starting fairseq2 v0.12."
            )

            super().__init__(f"{arch} is not a known model architecture.")
        else:
            super().__init__(f"{arch} is not a known {family} model architecture.")

        self.arch = arch
        self.family = family


@overload
def load_model(
    card: AssetCard | str,
    *,
    gangs: Gangs | None = None,
    dtype: DataType | None = None,
    config: object | None = None,
    mmap: bool = False,
    progress: bool | None = None,
) -> Module: ...


@overload
def load_model(
    card: AssetCard | str,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
    config: object | None = None,
    mmap: bool = False,
    progress: bool | None = None,
) -> Module: ...


def load_model(
    card: AssetCard | str,
    *,
    gangs: Gangs | None = None,
    device: Device | None = None,
    dtype: DataType | None = None,
    config: object | None = None,
    mmap: bool = False,
    progress: bool | None = None,
) -> Module:
    """
    Loads a pretrained model from an asset card.

    This function downloads the model checkpoint (if necessary) and loads the
    pretrained weights into a model instance. The model architecture and
    configuration are determined from the asset card metadata.

    As a convenience, this method also accepts an asset name instead of an
    asset card.

    The difference between ``load_model`` and :meth:`ModelHub.load_model()` is
    as follows:

    - ``load_model`` provides a unified interface for loading models across all
      model families. It determines the appropriate model family based on asset
      card metadata and delegates to the family-specific loading logic.
    - The tradeoff is that (1) the ``config`` parameter of ``load_model`` is not
      type-safe, (2) it is possible to accidentally load an unintended model
      since the function is not constrained to a specific family.
    - The general recommendation is to use :meth:`ModelHub.load_model` if the
      model family is known in advance, and to use ``load_model`` if the decision
      about the model and its family needs to be made at runtime.

    If ``gangs`` is provided, it will be used to apply parallelism (i.e.
    model parallelism) to the initialized model. If the model family does
    not support a certain parallelism strategy, that strategy will be
    ignored. For instance if ``gangs.tp.size > 1``, but the model does not
    support tensor parallelism, the model will be instantiated with regular
    attention and feed-forward network blocks. If ``None``, the whole model
    will be initialized without any parallelism.

    If ``device`` is provided, the model will be created on the specified
    device; otherwise, the device returned from :func:`torch.get_default_device`
    will be used. Note that ``device`` and ``gangs`` cannot be provided
    together. If ``gangs`` is provided, ``gangs.root.device`` will be used.

    If ``dtype`` is provided, it will be used as the default data type of
    the model parameters and buffers; otherwise, the data type returned from
    :func:`torch.get_default_dtype` will be used.

    If ``config`` is provided, it overrides the default model configuration
    from the asset card. If ``None``, uses the configuration specified in
    the card. Typically used to perform slight adjustments to the model
    configuration such as tuning dropout probabilities without changing the
    architecture.

    If ``mmap`` is ``True``, the model checkpoint will be memory-mapped. This
    can reduce memory usage but may cause slower load times on some systems.

    ``progress`` is deprecated and will be removed in v0.13. Use
    ``FAIRSEQ2_NO_PROGRESS=1`` environment variable or ``no_progress`` parameter
    of :func:`init_fairseq` to disable progress bars.

    .. code:: python

        from fairseq2.assets import get_asset_store
        from fairseq2.models.qwen import load_model

        card = get_asset_store().retrieve_card("qwen25_7b_instruct")

        qwen_model = load_model(card)

        # As a convenience, the card can be omitted and the model name can
        # be passed directly to `load_model()`:
        wav2vec2_model = load_model("wav2vec2_asr_base_10h")

    :raises ModelNotKnownError: If ``card`` is a string and no asset card
        with that name exists.

    :raises AssetCardError: If the asset card's model family doesn't match
        this hub's family.

    :raises ValueError: If both ``gangs`` and ``device`` are provided.
    """
    resolver = get_dependency_resolver()

    global_loader = resolver.resolve(GlobalModelLoader)

    return global_loader.load(card, gangs, device, dtype, config, mmap, progress)


@final
class GlobalModelLoader:
    """
    A global model loader that can load models from any registered model family.

    This class is used internally by the :func:`load_model` function to provide
    a unified interface for loading models across all model families. It resolves
    the appropriate model family based on asset card metadata and delegates to
    the family-specific loading logic.
    """

    def __init__(self, asset_store: AssetStore, families: Lookup[ModelFamily]) -> None:
        self._asset_store = asset_store
        self._families = families

    def load(
        self,
        card: AssetCard | str,
        gangs: Gangs | None,
        device: Device | None,
        dtype: DataType | None,
        config: object | None,
        mmap: bool,
        progress: bool | None,
    ) -> Module:
        """See :func:`load_model`."""
        _warn_progress_deprecated(progress)

        gangs = _get_effective_gangs(gangs, device)

        if isinstance(card, str):
            name = card

            try:
                card = self._asset_store.retrieve_card(name)
            except AssetNotFoundError:
                raise ModelNotKnownError(name) from None
        else:
            name = card.name

        family_name = card.field("model_family").as_(str)

        family = self._families.maybe_get(family_name)
        if family is None:
            msg = f"family field of the {name} asset card is expected to be a supported model family, but is {family_name} instead."

            raise AssetCardError(name, msg)

        if dtype is None:
            dtype = torch.get_default_dtype()

        return family.load_model(card, gangs, dtype, config, mmap, progress=True)


def _get_effective_gangs(gangs: Gangs | None, device: Device | None) -> Gangs:
    if gangs is not None:
        if device is not None:
            raise ValueError(
                "`gangs` and `device` must not be specified at the same time."
            )

        return gangs

    if device is None:
        device = get_current_device()

    if device.type == "meta":
        raise ValueError("`device` must be a real device.")

    return create_fake_gangs(device)
