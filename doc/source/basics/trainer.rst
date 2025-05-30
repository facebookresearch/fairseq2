.. _basics-trainer:

:octicon:`dependabot` Trainer
=============================

The :class:`fairseq2.recipes.trainer.Trainer` class is the core class for training models.

Overview
--------

The trainer in fairseq2 is designed to be flexible and model-agnostic, handling various training scenarios from simple models to complex distributed training setups.
It is probably the most complex system in fairseq2, but also the most powerful.

.. mermaid::

    flowchart LR
        %% Main Trainer Class
        A[Trainer] --> B[TrainUnit]
        A --> C[DataReader]
        A --> D[Optimizer]
        A --> E[CheckpointManager]
        A --> H[LRScheduler]
        A --> I[Gang System]
        A --> P[Metrics Logging]
        A --> V[Validation]

        %% TrainUnit Components
        B --> F[Model]

        %% Gang System
        I --> J[Root Gang]
        I --> K[DP Gang]
        I --> L[TP Gang]

        %% Metrics Logging
        P --> P1[TensorBoard]
        P --> P2[WandB]
        P --> P3[JSON Logger]
        
        %% Validation
        V --> Q[EvalUnit]
        V --> R[Validation DataReader]
        
        %% CheckpointManager Details
        E --> E1[Save State]
        E --> E2[Load State]
        E --> E3[Keep Best Checkpoints]
        E --> E4[Save FSDP Model]


Core Components
---------------

TrainUnit
^^^^^^^^^

The ``TrainUnit`` is an abstract class that encapsulates model-specific training logic:

.. code-block:: python

    class TrainUnit(ABC, Generic[BatchT_contra]):
        """Represents a unit to be used with Trainer."""

        @abstractmethod
        def __call__(self, batch: BatchT_contra) -> tuple[Tensor, int | None]:
            """Process batch and return loss and number of targets."""

        @abstractmethod
        def set_step_nr(self, step_nr: int) -> None:
            """Set current training step number."""

        @property
        @abstractmethod
        def model(self) -> Module:
            """The underlying model."""

.. dropdown:: Example implementation
   :icon: code
   :animate: fade-in

   .. code-block:: python

      class TransformerTrainUnit(TrainUnit[TransformerBatch]):
        def __init__(self, model: TransformerModel) -> None:
            super().__init__(model)
            
        def __call__(self, batch: TransformerBatch) -> tuple[Tensor, int]:
            outputs = self._model(**batch)
            return outputs.loss, batch.num_tokens

Trainer Configuration
^^^^^^^^^^^^^^^^^^^^^

The :class:`fairseq2.recipes.trainer.Trainer` class accepts a wide range of configuration options:

.. code-block:: python

    # Example Trainer Configuration
    trainer = Trainer(
        # Basic parameters
        unit=train_unit,                     # Training unit to compute loss
        data_reader=data_reader,             # Data reader for training batches
        optimizer=optimizer,                 # Optimizer
        checkpoint_manager=checkpoint_mgr,   # Checkpoint manager
        root_gang=root_gang,                 # Root gang for distributed training
        
        # Optional parameters
        dp_gang=dp_gang,                     # Data parallel gang
        tp_gang=tp_gang,                     # Tensor parallel gang
        dtype=torch.float32,                 # Model data type
        lr_scheduler=lr_scheduler,           # Learning rate scheduler
        max_num_steps=100_000,               # Maximum training steps
        max_num_data_epochs=10,              # Maximum training epochs
        
        # Validation parameters
        valid_units=[valid_unit],            # Validation units
        valid_data_readers=[valid_reader],   # Validation data readers
        validate_every_n_steps=1_000,        # Validation frequency
        
        # Checkpoint parameters
        checkpoint_every_n_steps=5_000,      # Checkpoint frequency
        keep_last_n_checkpoints=5,           # Number of checkpoints to keep
        keep_best_n_checkpoints=3,           # Number of best checkpoints to keep
        
        # Metric parameters
        publish_metrics_every_n_steps=100,   # Metric publishing frequency
        tb_dir=Path("runs"),                 # TensorBoard directory
        metrics_dir=Path("metrics"),         # Metrics directory
        
        # Advanced parameters
        fp16_loss_scale=(128.0, 0.0001),    # Initial and min loss scale for fp16
        max_gradient_norm=None,              # Max gradient norm for clipping
        amp=False,                           # Enable automatic mixed precision
        anomaly_detection=False,             # Enable autograd anomaly detection
        seed=2                               # Random seed
    )

Training Flow
-------------

The training process follows this simplified sequence:

.. mermaid::

    sequenceDiagram
        participant T as Trainer
        participant U as TrainUnit
        participant D as DataReader
        participant M as Model
        participant O as Optimizer
        
        T->>D: Request batch
        D-->>T: Return batch
        T->>U: Process batch
        U->>M: Forward pass
        M-->>U: Return loss
        U-->>T: Return loss, num_targets
        T->>M: Backward pass
        T->>O: Update parameters
        T->>T: Update metrics

.. dropdown:: Step-by-step breakdown
    :icon: code
    :animate: fade-in

    We provide a simplified step-by-step process for the trainer in the following code snippet to help you understand the training flow.

    1. **Initialization**: The trainer is initialized with the necessary components and configurations.

    .. code-block:: python

        def __init__(self, unit: TrainUnit[BatchT], data_reader: DataReader[BatchT], ...):
            self._model = unit.model
            self._unit = unit
            self._data_reader = data_reader
            # ... initialize other components

    2. **Training Loop**: The training loop is implemented in the ``_do_run`` method:

    .. code-block:: python

        def _do_run(self) -> None:
            while self._should_run_step():
                self._step_nr += 1
                
                # Run training step
                self._run_step()
                
                # Maybe validate
                if self._should_validate():
                    self._validate()
                
                # Maybe checkpoint
                if self._should_checkpoint():
                    self._checkpoint()

    3. **Step Execution**: The ``_run_step`` method is responsible for executing a single training step:

    .. code-block:: python

        def _run_step(self) -> None:
            # Collect batches
            batches = self._next_batches()
            
            # Process each batch
            for batch in batches:
                # Forward pass
                loss, num_targets = self._unit(batch)
                
                # Backward pass
                self._loss_scaler.backward(loss)
                
                # Update parameters
                self._loss_scaler.run_optimizer_step(self._step_nr)

    4. **Validation**: The validation loop is implemented in the ``_validate`` method:

    .. code-block:: python

        def _validate(self) -> None:
            self._model.eval()

            with summon_fsdp_for_validation(self._model):
                unit_scores = []

                for unit, data_reader in zip(self._valid_units, self._valid_data_readers):
                    unit_score = self._validate_unit(unit, data_reader)
                    if unit_score is not None:
                        unit_scores.append(unit_score)

                self._valid_score = self._compute_valid_score(unit_scores)

            self._model.train()

    5. **Checkpoint Management**: The trainer supports flexible checkpoint management:
        - Save checkpoints at regular intervals (steps or epochs)
        - Keep N most recent checkpoints
        - Keep N best checkpoints based on validation score
        - Separate policies for full checkpoints vs model-only checkpoints
        - Support for FSDP model consolidation

    6. **Metrics Logging**: The trainer supports multiple logging backends:
        - TensorBoard: Visualize training curves
        - JSON Logs: Store metrics in files
        - Weights & Biases (WandB): Collaborative experiment tracking

Best Practices
--------------

#. **Metric Tracking**:
   - Register all relevant metrics in the train unit
   - Use appropriate metric types (Mean, Sum, etc.)
   - Consider adding validation metrics

#. **Resource Management**:
   - Use appropriate batch sizes for your hardware
   - Enable ``amp`` for memory efficiency
   - Configure gradient accumulation as needed

#. **Checkpoint Management**:
   - Save checkpoints regularly
   - Use both ``keep_last_n_checkpoints`` and ``keep_best_n_checkpoints``
   - Consider separate policies for full checkpoints vs models

#. **Validation**:
   - Validate at appropriate intervals
   - Track relevant validation metrics
   - Implement early stopping if needed

Advanced Features
-----------------

#. **Early Stopping**:

    .. code-block:: python

        def early_stopper(step_nr: int, score: float) -> bool:
            # Custom early stopping logic
            return score < threshold

        metric_descriptors = get_runtime_context().get_registry(MetricDescriptor)

        try:
            score_metric_descriptor = metric_descriptors.get(metric_name)
        except LookupError:
            raise UnknownMetricDescriptorError(metric_name) from None

        trainer = Trainer(
            early_stopper=early_stopper,
            score_metric_descriptor=score_metric_descriptor,
            lower_better=True,
        )

#. **Custom Learning Rate Scheduling**:

    .. code-block:: python

        class CustomLRScheduler(LRScheduler):
            def get_lr(self) -> float:
                # Custom LR calculation
                return self.base_lr * decay_factor(self.step_nr)

        trainer = Trainer(
            lr_scheduler=CustomLRScheduler(optimizer),
        )

#. **Profiling**:

    .. code-block:: python

        num_skip_steps, num_record_steps = (100, 10)

        profile_dir = Path("logs/tb")

        profiler = TorchProfiler(
            num_skip_steps, num_record_steps, profile_dir, gangs.root
        )

        trainer = Trainer(
            profiler=profiler,
            ...
        )

