.. _tutorial-models:

:octicon:`ruby` Add Your Own Model
==================================


.. dropdown:: What you will learn
    :icon: multi-select
    :animate: fade-in

    * How to configure a model

    * How to register a model architecture

    * How to use model factories to create models

    * How to use model loaders to load models

.. dropdown:: Prerequisites
    :icon: multi-select
    :animate: fade-in

    * Get familiar with fairseq2 basics (:ref:`basics-overview`)

    * Ensure you have fairseq2 installed (:ref:`installation`)

    * Get familiar with presets (:ref:`tutorial-presets`)

Overview
--------

The model configuration and loading system in fairseq2 consists of several key components:


#. **Model Config**

    * Defines the architecture and hyperparameters of a model (`e.g. number of layers, hidden size, learning rate, etc.`)

#. **Architecture Registry**

    * Stores predefined model architectures (`e.g. base, large, small, etc.`)

#. **Model Factory**

    * Creates model instances from configs

#. **Model Loader**

    * Handles model instantiation, checkpoint loading and format conversion (`e.g. loading from fairseq2 checkpoint, converting from HF checkpoint, etc.`)


Directory Layout
----------------

The directory structure for a typical fairseq2 model looks like this:

.. code-block:: bash

    fairseq2/models/
    ├── your_model/
    │   ├── __init__.py
    │   ├── archs.py        # Defines model architectures
    │   ├── factory.py      # Contains model factory and config classes
    │   ├── loader.py       # Handles model loading and checkpoint conversion
    │   └── model.py        # Actual model implementation

.. note::
   The actual layout might vary depending on your implementation.

Step-by-Step Guide
------------------

1. Define Model Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, create a configuration class in ``factory.py``:

.. code-block:: python

    from dataclasses import dataclass
    from fairseq2.typing import DataType
    from fairseq2.data import VocabularyInfo

    @dataclass(kw_only=True)
    class YourModelConfig:
        """Configuration for YourModel."""
        # Basic model parameters
        model_dim: int = 512
        """The dimensionality of the model."""

        num_layers: int = 6
        """The number of layers in the model."""

        num_heads: int = 8
        """The number of attention heads in the model."""
    
        ...

In the same file, create a registry for the model config:

.. code-block:: python

    your_model_config_registry = ConfigRegistry[YourModelConfig]()

    your_model_arch = your_model_config_registry.decorator

This ``your_model_arch`` is a decorator that can be later used to register model architectures.


2. Register Model Architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an architecture registry and define standard architectures in ``archs.py``:

.. code-block:: python

    from fairseq2.models.your_model.factory import your_model_arch

    @your_model_arch("base")
    def _base() -> YourModelConfig:
        """Base architecture."""
        return YourModelConfig()

    @your_model_arch("large")
    def _large() -> YourModelConfig:
        """Large architecture."""
        config = YourModelConfig()
        config.model_dim = 1024
        config.num_layers = 12
        config.num_heads = 16
        return config

.. note::
   Keep the architecture names descriptive and simple. Document differences between architectures.


.. dropdown:: Some real-world examples
    :icon: code
    :animate: fade-in

    * **Base Transformer Architecture**

    The base Transformer model provides a foundation that other models can build upon:

    .. code-block:: python

        # In transformer/archs.py
        from fairseq2.models.transformer.factory import TransformerConfig, transformer_arch

        @transformer_arch("base")
        def _base() -> TransformerConfig:
            """Base architecture with default parameters."""
            return TransformerConfig()

        @transformer_arch("big")
        def _big() -> TransformerConfig:
            """Larger architecture with modified parameters."""
            config = TransformerConfig()
            config.model_dim = 1024
            config.num_encoder_attn_heads = 16
            config.num_decoder_attn_heads = 16
            config.ffn_inner_dim = 4096
            config.dropout_p = 0.3
            return config


    * **NLLB (No Language Left Behind)**

    NLLB extends the base Transformer architecture with specific configurations for multilingual translation:

    .. code-block:: python

        # In nllb/archs.py
        @transformer_arch("nllb_dense_600m")
        def _dense_600m() -> TransformerConfig:
            config = _dense_1b()  # Inherits from larger architecture
            
            # Modify for smaller model
            config.num_encoder_layers = 12
            config.num_decoder_layers = 12
            config.ffn_inner_dim = 1024 * 4
            
            return config

        @transformer_arch("nllb_dense_1b")
        def _dense_1b() -> TransformerConfig:
            config = transformer_archs.get("base")  # Start from base transformer
            
            # Customize for NLLB
            config.model_dim = 1024
            config.vocab_info = VocabularyInfo(
                size=256206, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0
            )
            config.num_encoder_layers = 24
            config.num_decoder_layers = 24
            config.num_encoder_attn_heads = 16
            config.num_decoder_attn_heads = 16
            config.ffn_inner_dim = 1024 * 8
            config.norm_order = TransformerNormOrder.PRE
        
        return config


    * **LLaMA Architecture**

    LLaMA introduces its own configuration class with specific parameters for large language models:

    .. code-block:: python

        # In llama/archs.py
        @llama_arch("7b")
        def _7b() -> LLaMAConfig:
            """7B parameter model."""
            return LLaMAConfig()  # Uses default parameters

        @llama_arch("13b")
        def _13b() -> LLaMAConfig:
            """13B parameter model."""
            config = _7b()
            config.model_dim = 5120
            config.num_attn_heads = 40
            config.num_key_value_heads = 40
            config.ffn_inner_dim = 5120 * 4
            return config

        @llama_arch("llama2_70b")
        def _llama2_70b() -> LLaMAConfig:
            """LLaMA 2 70B parameter model."""
            config = _65b()
            config.max_seq_len = 4096
            config.num_key_value_heads = 8
            config.ffn_inner_dim = int(8192 * 4 * 1.3)  # See A.2.1 in LLaMA 2
            config.ffn_inner_dim_to_multiple = 4096
            return config


3. Create Model Factory
^^^^^^^^^^^^^^^^^^^^^^^

Implement a factory function in ``factory.py`` that creates model instances:

.. code-block:: python

    def create_your_model(config: YourModelConfig) -> YourModel:
        """Create a model instance from config."""
        model = YourModel(
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout_p=config.dropout_p,
            vocab_info=config.vocab_info,
        )
    
        # Convert to specified dtype
        model.to(dtype=config.dtype)
    
        return model


.. dropdown:: Some real-world examples
    :icon: code
    :animate: fade-in

    * **LLaMA Model Factory**

    We will use the ``fairseq2.models.llama.factory.create_llama_model`` function as an example.

    The ``create_llama_model`` function serves as a factory method for instantiating a LLaMA model.
    It encapsulates the process of building a model with the ``LLaMABuilder`` class, which constructs various components of the model based on the provided configuration.
    This design pattern allows for a clean separation of model creation logic, making it easier for users to customize and extend the model architecture.

    .. code-block:: python

        # In llama/factory.py
        class LLaMABuilder:
        ...

        def build_model(self) -> TransformerDecoderModel:
            """Build a model."""
            decoder_frontend = self.build_decoder_frontend()

            decoder = self.build_decoder()

            final_proj = Linear(...)

            model = TransformerDecoderModel(
                decoder_frontend,
                decoder,
                final_proj,
                ...
            )

            model.set_family(LLAMA_FAMILY)

            return model


        def create_llama_model(
            config: LLaMAConfig,
            *,
            device: Device | None = None,
            dtype: DataType | None = None,
        ) -> TransformerDecoderModel:
            """Create a LLaMA model."""
            return LLaMABuilder(config, device=device, dtype=dtype).build_model()


        model_factories.register(LLAMA_FAMILY, create_llama_model, LLaMAConfig, llama_archs)

    `create_llama_model` instantiates your builder class and call the `build_model` method that actually creates the model as a `TransformerDecoderModel`.
    Don't forget to register your model with the fairseq2 model factories so that it can be easily instantiated later.

4. Set Up Model Loader
^^^^^^^^^^^^^^^^^^^^^^

Create a loader in ``loader.py`` that handles model instantiation and checkpoint loading:

.. code-block:: python

    from fairseq2.models.config_loader import StandardModelConfigLoader
    from fairseq2.models.loader import StandardModelLoader, load_model

    # Create config loader
    load_your_model_config = StandardModelConfigLoader(
        YOUR_MODEL_FAMILY,
        YourModelConfig,
        your_model_archs
    )

    def convert_your_model_checkpoint(
        checkpoint: dict[str, Any], config: YourModelConfig
    ) -> dict[str, Any]:
        """Convert external checkpoints to fairseq2 format."""
        # Add checkpoint conversion logic here
        return {"model": checkpoint}

    # Create model loader
    load_your_model = StandardModelLoader(
        config_loader=load_your_model_config,
        factory=create_your_model,
        checkpoint_converter=convert_your_model_checkpoint,
    )

    # Register loader with global registry
    load_model.register(YOUR_MODEL_FAMILY, load_your_model)

.. dropdown:: Some real-world examples on ckpt conversion
    :icon: code
    :animate: fade-in

    The `convert_your_model_checkpoint` function is a checkpoint converter that converts external checkpoints to fairseq2 format.
    For example, in Mistral, the checkpoint format is different from fairseq2's.

    .. code-block:: python

        # In mistral/loader.py
        def convert_mistral_checkpoint(
            checkpoint: dict[str, Any], config: MistralConfig
        ) -> dict[str, Any]:
            """Convert Mistral checkpoint to fairseq2 format."""
            if "model" in checkpoint:  # Already in fairseq2 format
                return checkpoint

            # Map parameter names from Mistral to fairseq2 format
            key_map = {
                r"^layers\.([0-9]+)\.attention\.wq\.":    r"decoder.layers.\1.self_attn.q_proj.",
                r"^layers\.([0-9]+)\.attention\.wk\.":    r"decoder.layers.\1.self_attn.k_proj.",
                r"^layers\.([0-9]+)\.attention\.wv\.":    r"decoder.layers.\1.self_attn.v_proj.",
                # ... more mappings
            }

            checkpoint = convert_model_state_dict(checkpoint, key_map)
            return {"model": checkpoint}

    Overall, to support loading from different checkpoint formats:

    1. Modify the checkpoint converter function
    2. Add mapping logic for different parameter names
    3. Handle any necessary tensor transformations

.. dropdown:: Advanced topic: Sharding
    :icon: code
    :animate: fade-in

    The ``sharder`` argument in ``StandardModelLoader`` is a function that shards the model, which is useful for distributed training.
    This is natively supported by fairseq2, so you don't need to implement it yourself.
    For example, in LLaMA, the ``shard_llama_model`` function shards the model across multiple devices:

    .. code-block:: python

        # In llama/loader.py
        from fairseq2.models.transformer import shard_transformer_decoder_model
        from fairseq2.models.loader import StandardModelLoader

        def shard_llama_model(
            model: TransformerDecoderModel, config: LLaMAConfig, gangs: Mapping[str, Gang]
        ) -> None:
            gang = gangs["tp"]  # tensor parallel

            shard_embed_dim = config.max_seq_len < 8192  # LLaMA 1 or 2

            shard_transformer_decoder_model(model, gang, shard_embed_dim=shard_embed_dim)


        load_llama_model = StandardModelLoader(
            ...
            sharder=shard_llama_model,
        )

5. Using with Trainer
^^^^^^^^^^^^^^^^^^^^^

The model can be used with the fairseq2 trainer:

.. code-block:: python

    from fairseq2.models.loader import load_model
    from fairseq2.recipes.trainer import Trainer, TrainUnit
    from fairseq2.recipes.utils.asset import retrieve_asset_card

    model_card = retrieve_asset_card("llama3_2_1b")

    # Load model
    model = load_model(
        model_card,
        device=Device("cpu")
    )

    # Create training unit
    class YourTrainUnit(AbstractTrainUnit[SequenceBatch]):
        def __init__(self, model: YourModel) -> None:
            super().__init__(model)
        self._metric_bag = MetricBag()
        
        def __call__(self, batch: YourBatchType) -> tuple[Tensor, int]:
            loss = self._model(**batch)
            return loss, batch.num_targets

    # Set up trainer
    trainer = Trainer(
        unit=YourTrainUnit(model),
        data_reader=your_data_reader,
        optimizer=your_optimizer,
        # ... other trainer parameters
    )

    # Run training
    trainer()

For a real-world example, see the :mod:`fairseq2.recipes.lm` recipe.


Best Practices
--------------

#. **Configuration**:

   * Provide sensible defaults for all parameters
   * Document each config parameter

#. **Architecture Registry**:

   * Use descriptive names for architectures
   * Keep base architectures simple
   * Document differences between architectures

#. **Model Loading**:

   * Handle checkpoint format differences gracefully
   * Validate config parameters before model creation
   * Provide clear error messages for invalid configs

#. **Training Integration**:

   * Create a dedicated training unit for your model
   * Implement proper metric tracking
   * Handle device placement and dtype conversion

Common Pitfalls
---------------

#. **Checkpoint Compatibility**:

   * Ensure checkpoint conversion handles all parameter mappings
   * Verify tensor shapes and dtypes match
   * Handle missing or extra parameters gracefully

#. **Configuration Issues**:

   * Validate all config parameters before use
   * Handle interdependent parameters correctly
   * Document any parameter constraints

#. **Training Problems**:

   * Ensure proper device placement
   * Handle batch processing efficiently
   * Implement correct loss computation
