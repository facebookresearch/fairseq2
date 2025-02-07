.. _tutorial-models:

:octicon:`ruby` Add Your Own Model
==================================

.. dropdown:: What you will learn
    :icon: multi-select
    :animate: fade-in

    * How to configure a model
    * How to implement a model class

    * How to create a model factory and handler

    * How to register your model with fairseq2

.. dropdown:: Prerequisites
    :icon: multi-select
    :animate: fade-in

    * Get familiar with fairseq2 basics (:ref:`basics-overview`)

    * Ensure you have fairseq2 installed (:ref:`installation`)

Overview
--------

The model system in fairseq2 consists of several key components:

#. **Model Config**

    * Defines the architecture and hyperparameters

    * Supports different model variants through config presets

#. **Model Class**

    * Implements the actual model architecture

    * Inherits from appropriate base classes

#. **Model Factory**

    * Creates model instances from configs

    * The most important is to have a ``create_model`` method

#. **Model Handler**

    * Manages model creation and checkpoint loading

    * Converts between different checkpoint formats

Directory Layout
----------------

The directory structure for a fairseq2 model typically looks like this:

.. code-block:: bash

    src/fairseq2/models/your_model/
    ├── __init__.py
    ├── config.py      # Model configuration and presets
    ├── factory.py     # Model factory
    ├── handler.py     # Model handler for creation and loading
    └── model.py       # Model implementation

Step-by-Step Guide
------------------

1. Define Model Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, create a configuration class in ``config.py``:

.. code-block:: python

    from dataclasses import dataclass
    from typing import Final

    from fairseq2.context import RuntimeContext
    from fairseq2.data import VocabularyInfo

    @dataclass(kw_only=True)
    class YourModelConfig:
        """Holds the configuration of your model."""
        
        model_dim: int = 512
        """The dimensionality of the model."""
        
        max_seq_len: int = 2048
        """The maximum sequence length."""
        
        vocab_info: VocabularyInfo
        """The vocabulary information."""

    def register_your_model_configs(context: RuntimeContext) -> None:
        """Register model architecture presets."""
        registry = context.get_config_registry(YourModelConfig)
        
        arch = registry.decorator
        
        @arch("base")
        def your_model_base() -> YourModelConfig:
            return YourModelConfig(
                vocab_info=VocabularyInfo(
                    size=32000,
                    unk_idx=0,
                    bos_idx=1,
                    eos_idx=2,
                    pad_idx=None
                )
            )

2. Create Model Class
^^^^^^^^^^^^^^^^^^^^^

Implement your model in ``model.py``:

.. code-block:: python

    from typing import final

    from torch import Tensor
    from typing_extensions import override

    from fairseq2.models.decoder import DecoderModel
    from fairseq2.nn import IncrementalStateBag
    from fairseq2.nn.padding import PaddingMask

    @final
    class YourModel(DecoderModel):
        """Your model implementation."""
        
        def __init__(
            self,
            model_dim: int,
            max_seq_len: int,
            vocab_info: VocabularyInfo,
        ) -> None:
            super().__init__(model_dim, max_seq_len, vocab_info)
            
            # Initialize your model components here
            
        @override
        def decode(
            self,
            seqs: Tensor,
            padding_mask: PaddingMask | None,
            *,
            state_bag: IncrementalStateBag | None = None,
        ) -> tuple[Tensor, PaddingMask]:
            # Implement your decoding logic
            pass

3. Implement Model Factory
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
    
    This factory pattern is a convention and not strictly required.
    It is helpful to subclass and change specific parts of the model construction logic if needed.
    The most important is to have a ``create_model(config: YourModelConfig) -> YourModel`` method to integrate with fairseq2.

Create a factory in ``factory.py``:

.. code-block:: python

    from fairseq2.models.your_model._config import YourModelConfig
    from fairseq2.models.your_model._model import YourModel

    class YourModelFactory:
        """Creates model instances."""
        
        _config: YourModelConfig
        
        def __init__(self, config: YourModelConfig) -> None:
            self._config = config
        
        def create_model(self) -> YourModel:
            """Creates a model instance."""
            config = self._config
            
            return YourModel(
                model_dim=config.model_dim,
                max_seq_len=config.max_seq_len,
                vocab_info=config.vocab_info,
            )


4. Create Model Handler
^^^^^^^^^^^^^^^^^^^^^^^

Implement a handler in ``handler.py``:

.. code-block:: python

    from typing import cast

    from torch.nn import Module
    from typing_extensions import override

    from fairseq2.models import AbstractModelHandler
    from fairseq2.models.your_model._config import YourModelConfig
    from fairseq2.models.your_model._factory import YourModelFactory
    from fairseq2.models.your_model._model import YourModel

    class YourModelHandler(AbstractModelHandler):
        # A 'family' represents a group of related models sharing a common
        # architecture. For instance, `llama` is the model family of
        # `llama_3_2_8b_instruct`.
        @override
        @property
        def family(self) -> str:
            return "my_model_family"
        
        @override
        @property
        def kls(self) -> type[Module]:
            return YourModel
        
        @override
        def _create_model(self, config: object) -> Module:
            config = cast(YourModelConfig, config)
            
            return YourModelFactory(config).create_model()

5. Register the Model
^^^^^^^^^^^^^^^^^^^^^

Add to your ``setup_fairseq2_extension``:

.. code-block:: python

    def setup_fairseq2_extension(context: RuntimeContext) -> None:
        # fairseq2's global model registry.
        model_registry = context.get_registry(ModelHandler)

        # Registry my model.
        configs = context.get_config_registry(YourModelConfig)

        default_arch = "base"

        handler = YourModelHandler(
            configs, default_arch, asset_download_manager, tensor_loader
        )

        model_registry.register(handler.family, handler)

        # Register my model architecture configurations.
        register_your_model_configs(context)

Best Practices
--------------

#. **Configuration**:
    * Make all parameters type-safe and well-documented
    * Use sensible defaults
    * Register different architectures as config presets

#. **Model Implementation**:
    * Inherit from appropriate base classes
    * Use type hints and proper documentation
    * Implement all required abstract methods

#. **Checkpoint Loading**:
    * Handle different checkpoint formats gracefully
    * Use ``convert_model_state_dict`` for key mapping
    * Validate checkpoint contents

#. **Testing**:
    * Add unit tests for model components
    * Test checkpoint loading
    * Verify model outputs

Common Pitfalls
---------------

#. **Type Safety**:
    * Always use type hints
    * Validate config parameters

#. **Checkpoint Compatibility**:
    * Handle missing or extra parameters
    * Verify tensor shapes and dtypes
    * Document supported checkpoint formats

#. **Model Registration**:
    * Register configs before using them
    * Set appropriate default architecture
    * Handle dependencies correctly
