.. _basics-building-recipes:

==================================
:octicon:`rocket` Building Recipes
==================================

fairseq2 v0.5 introduces a simplified approach to creating custom training recipes.
This tutorial shows how to build a complete language model pretraining recipe with just a few focused classes, showcasing the power and flexibility of the new recipe system.

Overview
========

The new recipe system eliminates much of the complexity found in earlier versions, allowing you to focus on what matters most: your model, data, and training logic.
A complete custom recipe consists of just four main components:

.. mermaid::

   flowchart LR
       %% Styling
       classDef recipeBox fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b
       classDef configBox fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
       classDef datasetBox fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#1b5e20
       classDef entryBox fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100

       R[Recipe Class<br/>LMTrainRecipe]:::recipeBox
       C[Configuration<br/>LMTrainConfig]:::configBox
       D[Dataset<br/>LMTrainDataset]:::datasetBox
       E[Entry Point<br/>__main__.py]:::entryBox

       E --> R
       R --> C
       R --> D
       C --> D

- **Minimal boilerplate**: Just 3 methods to override in your recipe class
- **Automatic dependency injection**: Components are wired together automatically
- **Type-safe configuration**: Dataclass-based configs with IDE support
- **Pluggable datasets**: Easy to swap data sources and formats
- **One-line execution**: Single command to run your recipe

Let's go through the language model pretraining recipe step by step.

Step 0: Setup the Entry Point
=============================

The entry point is remarkably simple in v0.5:

.. code-block:: python

   from fairseq2.recipe.cli import train_main
   from .recipe import LMTrainRecipe

   # Create recipe instance
   recipe = LMTrainRecipe()

   # Run training with automatic CLI handling
   train_main(recipe)

That's it! Just 3 lines of code to create a complete training entry point.

**What :meth:`fairseq2.recipe.cli.train_main` provides automatically:**

- **Command line argument parsing** with recipe-specific options
- **Configuration loading and validation** from files or command line
- **Distributed training setup** with proper process group initialization
- **Logging configuration** with structured output and metrics
- **Error handling** with graceful shutdown and debugging support
- **Checkpoint management** with automatic save/load functionality

Then what we need to do is to build our ``LMTrainRecipe`` class.
As a quick preview, here is the skeleton of the recipe class:

.. code-block:: python

    from fairseq2.recipe.base import RecipeContext, TrainRecipe
    from fairseq2.composition import (
        register_dataset_family,
        register_model_family,
        register_tokenizer_family,
    )

    @final
    class LMTrainRecipe(TrainRecipe):
        """Language model pretraining recipe."""

        @override
        def register(self, container: DependencyContainer) -> None:
            """Register dataset/model/tokenizer family with the dependency container."""
            # Register model families (if any)
            register_model_family(
                container,
                "my_custom_model",  # model family name
                kls=MyCustomModel,  # model class
                config_kls=MyCustomModelConfig,  # model config class
                factory=create_my_custom_model,  # factory function
                # ... other parameters
            )

            # Register dataset families (if any)
            register_dataset_family(
                container,             # DependencyContainer instance
                "custom_dataset",      # family name
                CustomDataset,         # dataset class
                CustomDatasetConfig,   # config class
                opener=custom_opener   # opener function
            )

            # Register tokenizer families (if any)
            register_tokenizer_family(
                container,
                "custom_tokenizer",     # tokenizer family name
                CustomTokenizer,        # tokenizer class
                CustomTokenizerConfig,  # tokenizer config class
                loader=custom_loader,   # loader function
            )


        @override
        def create_trainer(self, context: RecipeContext) -> Trainer:
            """Create the trainer with model and data configuration."""
            ...
            # TODO: build this config class for our recipe
            config = context.config.as_(LMTrainConfig)

            # TODO: build this Train unit class which defines loss computation
            unit = LMTrainUnit(context.model)

            # TODO: build the dataset and create data reader
            dataset = context.default_dataset.as_(LMTrainDataset)
            data_reader = dataset.create_reader(...)

            # Create trainer using context helper
            return context.create_trainer(unit, data_reader)

        @property
        @override
        def config_kls(self) -> type[object]:
            """Return the configuration class for this recipe."""
            return LMTrainConfig


So, let's start with the first step.

Step 1: Define Your Configuration
=================================

Configuration in fairseq2 uses dataclasses with sensible defaults and clear structure:

**File: ``config.py``**

.. code-block:: python

    @dataclass(kw_only=True)
    class LMTrainConfig:
        """Configuration for language model pretraining."""

        # Model configuration
        model: ModelSection = field(
            default_factory=lambda: ModelSection(...)
        )

        # Dataset configuration
        dataset: LMTrainDatasetSection = field(
            default_factory=lambda: LMTrainDatasetSection(...)
        )

        # Tokenizer selection
        tokenizer: TokenizerSection = field(
            default_factory=lambda: TokenizerSection(...)
        )

        # Distributed training setup
        gang: GangSection = field(default_factory=lambda: GangSection())

        # Training parameters
        trainer: TrainerSection = field(
            default_factory=lambda: TrainerSection(...)
        )

        # Optimizer configuration
        optimizer: OptimizerSection = field(
            default_factory=lambda: OptimizerSection(...)
        )

        # Learning rate scheduler
        lr_scheduler: LRSchedulerSection | None = field(
            default_factory=lambda: LRSchedulerSection(...)
        )

        # Training regime
        regime: RegimeSection = field(
            default_factory=lambda: RegimeSection(...)
        )

        # Common settings
        common: CommonSection = field(default_factory=lambda: CommonSection(...))

    @dataclass(kw_only=True)
    class LMTrainDatasetSection(DatasetSection):
        """Dataset-specific configuration parameters."""
        ...

- **Simple Structure**: Each section controls a specific aspect of training
- **Sensible Defaults**: Ready-to-use settings for beginners
- **Type Safety**: Full IDE support with autocompletion
- **Customizable**: Easy to override values via command line or config files


Step 2: Implement Your Dataset
==============================

The dataset component handles data loading and preprocessing:

**File: `dataset.py`**

.. code-block:: python

    @final
    class LMTrainDataset:
        """Language model training dataset supporting JSONL files."""

        def __init__(self, files: Sequence[Path]) -> None:
            self._files = files

        def create_reader(
            self,
            tokenizer: Tokenizer,
            gangs: Gangs,
            *,
            ...
        ) -> DataReader[SequenceBatch]:
            """Create a data reader for distributed training."""

            ...

            # Create data pipeline
            builder = read_sequence(self._files)

            # Shard files across ranks for distributed training
            if file_world_size > 1:
                builder.shard(file_rank, file_world_size, allow_uneven=True)

            # Define how to read individual files
            def read_file(file: Path) -> DataPipeline:
                ...

            builder.yield_from(read_file)

            ...

            # Packing for efficient training
            builder.pack(...)

            ...

            # Background prefetching for performance
            builder.prefetch(prefetch)

            # Convert to SequenceBatch format
            def to_batch(example: dict[str, Any]) -> SequenceBatch:
                seqs, seq_lens = example["seqs"], example["seq_lens"]
                return SequenceBatch(seqs, seq_lens, packed=True)

            pipeline = builder.map(to_batch).and_return()

            return DataPipelineReader[SequenceBatch](
                pipeline,
                gangs,
                ...
            )

   @dataclass
   class LMTrainDatasetConfig:
       """Configuration for LM training dataset."""
       path: Path = field(default_factory=Path)

   def open_lm_train_dataset(config: LMTrainDatasetConfig) -> LMTrainDataset:
       """Factory function to create dataset from configuration."""
       path = config.path.expanduser().resolve()

       if not path.is_dir():
           # Single file
           files = [path]
       else:
           # Directory of JSONL files
           files = [f for f in path.glob("**/*.chunk.*.jsonl") if not f.is_dir()]
           files.sort()

       return LMTrainDataset(files)


- **Distributed by Design**: Automatic file sharding across data parallel ranks
- **Efficient Packing**: Sequences packed to maximize GPU utilization
- **Performance Optimized**: Background prefetching and pinned memory
- **Flexible Input**: Supports both single files and directories of files
- **torch.compile Ready**: Proper BatchLayout configuration for compilation

Step 3: Create Your Recipe Class
================================

The recipe class ties everything together with minimal boilerplate:

**File: `recipe.py`**

.. code-block:: python

    @final
    class LMTrainRecipe(TrainRecipe):
        """Language model pretraining recipe."""

        @override
        def register(self, container: DependencyContainer) -> None:
            """Register dataset family with the dependency container."""
            register_dataset_family(
                container,
                LM_TRAIN_DATASET,           # Dataset type identifier
                LMTrainDataset,             # Dataset class
                LMTrainDatasetConfig,       # Configuration class
                opener=open_lm_train_dataset,  # Factory function
            )

        @override
        def create_trainer(self, context: RecipeContext) -> Trainer:
            """Create the trainer with model and data configuration."""
            ...
            # Get typed configuration
            config = context.config.as_(LMTrainConfig)

            # Create training unit (defines loss computation)
            unit = LMTrainUnit(context.model)

            # Get dataset and create data reader
            dataset = context.default_dataset.as_(LMTrainDataset)
            data_reader = dataset.create_reader(...)

            # Create trainer using context helper
            return context.create_trainer(unit, data_reader)

        @property
        @override
        def config_kls(self) -> type[object]:
            """Return the configuration class for this recipe."""
            return LMTrainConfig

    @final
    class LMTrainUnit(TrainUnit[SequenceBatch]):
        """Training unit that defines how to process batches."""

        def __init__(self, model: RecipeModel) -> None:
            self._model = model

        @override
        def process_batch(
            self,
            batch: SequenceBatch,
            metric_bag: MetricBag
        ) -> tuple[Tensor, None]:
            """Process a single batch and compute loss."""
            # Split batch into input and target sequences
            input_batch, target_batch = batch.as_auto_regressive()

            # Get sequences and layout for model input
            seqs, seqs_layout = input_batch.as_input()

            # Compute loss using the model
            nll_loss = self._model.module(
                seqs,
                seqs_layout,
                ...
            )

            # Update metrics
            update_nll_loss_metric(metric_bag, nll_loss)
            update_seq_batch_metrics(metric_bag, batch)

            return nll_loss, None


- **Minimal Interface**: Only 3 methods to override (``register``, ``create_trainer``, ``config_kls``)
- **Automatic Dependency Injection**: Components are wired together by the framework
- **Type Safety**: Strong typing throughout with IDE support
- **Flexible Training Logic**: Easy to customize loss computation and metrics
- **Built-in Validation**: Context provides validation helpers

Running Your Recipe
===================

Once you've created these four files, running your recipe is straightforward:

**Basic Usage:**

.. code-block:: bash

    # Run with default configuration
    python -m recipes.lm.train /output/dir

    # Check the default configuration (yaml format)
    python -m recipes.lm.train --dump-config

    # Override configuration with your own yaml file + config overrides
    python -m your_package.lm.train \
        --config-file /path/to/config.yaml \
        --config model.name=llama3_2_1b_instruct regime.num_steps=20 lr_scheduler.config.num_warmup_steps=10

You can also specify the asset store to use with the config override ``--config common.asset.extra_paths="['/path/to/assets/dir', '/path/to/yet_other_assets/dir']"`` option.
For more detailed information about assets, see :doc:`/basics/assets`.

See Also
========

* :doc:`design_philosophy` - Core architectural principles
* :doc:`building_recipes` - Advanced recipe patterns with chatbot example
* :doc:`/reference/recipe` - Recipe system API reference
* :doc:`/news/whats_new_v0_5` - Complete list of v0.5 improvements
