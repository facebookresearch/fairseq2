.. _basics-design-philosophy:

=====================================
:octicon:`infinity` Design Philosophy
=====================================

One of the core goals of fairseq2 is to make it possible for researchers to
explore new ideas and implement novel features without having to fork fairseq2.
Instead of having a monolithic repository that can only be modified by
copy-pasting large chunks of code, in fairseq2, all major APIs follow the
interface/implementation convention along with the `dependency inversion principle`__.
This means, each API has an *interface* (i.e. an abstract :class:`~abc.ABC`
class) that defines the contract of that API, and one or more concrete
implementations of that interface. Different implementations can be integrated
with the rest of fairseq2 via its powerful `dependency injection framework`__.

.. __: https://en.wikipedia.org/wiki/Dependency_inversion_principle
.. __: https://en.wikipedia.org/wiki/Dependency_injection

Interface/Implementation Convention
===================================

.. currentmodule:: fairseq2.nn

The diagram below shows the :doc:`position encoder API </reference/fairseq2.nn.PositionEncoder>`
as an example. The API is defined by the abstract :class:`PositionEncoder`
PyTorch module. :class:`SinusoidalPositionEncoder`, :class:`LearnedPositionEncoder`,
and :class:`RotaryEncoder` implement :class:`PositionEncoder` for their
respective algorithms. Technically, any of these position encoders can be used
wherever a :class:`PositionEncoder` is expected (see `Dependency Inversion`_
below).

.. image:: /_static/img/position_encoder.svg
    :width: 580px
    :align: center
    :alt: Position Encoder Hierarchy

These diagrams demonstrate fairseq2's interface-first approach: each API starts with
a clear abstract interface, followed by multiple concrete implementations that can
be used interchangeably.

Dependency Inversion
====================

.. currentmodule:: fairseq2.nn.transformer

The dependency inversion principle is critical to have a clean, well-tested, and
extensible API. The example below shows the (abbreviated) ``__init__()`` method
of the :class:`StandardTransformerDecoderLayer`::

    class StandardTransformerDecoderLayer(TransformerDecoderLayer):

        def __init__(
            self,
            self_attn: MultiheadAttention,
            self_attn_layer_norm: LayerNorm,
            encoder_decoder_attn: MultiheadAttention,
            encoder_decoder_attn_layer_norm: LayerNorm,
            ffn: FeedForwardNetwork,
            ffn_layer_norm: LayerNorm,
            *,
            ...
        ) -> None:
            ...

Instead of constructing the multihead attention and feed-forward network layers
within its ``__init__()`` method, :class:`StandardTransformerDecoderLayer`
expects the caller to provide instances of :class:`MultiheadAttention` and
:class:`FeedForwardNetwork` interfaces. This loose-coupling between an instance
and its dependencies enables composing diverse object graphs, such as different
model architectures, with minimal redundancy (i.e. code duplication).

Dependency Injection
====================

.. currentmodule:: fairseq2.runtime.dependency

fairseq2 v0.5 introduces a dependency injection framework that
significantly simplifies the construction and management of complex object graphs.
The core components are the :class:`DependencyContainer` and
:class:`DependencyResolver` classes, which provide automatic dependency resolution,
singleton management, and collection handling.

Core Components
^^^^^^^^^^^^^^^

The dependency injection system is built around several key abstractions:

.. autoclass:: DependencyResolver
   :members: resolve, resolve_optional, iter_keys
   :noindex:

.. autoclass:: DependencyContainer
   :members: register, register_type, register_instance
   :noindex:

.. autoclass:: DependencyProvider
   :noindex:

Basic Usage
^^^^^^^^^^^

The fairseq2 library uses the dependency injection system extensively for all
core components. The global container is initialized by :func:`fairseq2.init_fairseq2`
and can be accessed through :func:`get_dependency_resolver`::

    import fairseq2
    from fairseq2.runtime.dependency import get_dependency_resolver
    from fairseq2.assets import AssetStore
    from fairseq2.device import Device
    from fairseq2.models import load_model

    # Initialize the library - sets up the global container
    fairseq2.init_fairseq2()

    # Access the global resolver
    resolver = get_dependency_resolver()

    # Resolve library components
    asset_store = resolver.resolve(AssetStore)
    device = resolver.resolve(Device)
    card = asset_store.retrieve_card("llama3_1_8b_instruct")

    # These are all automatically configured through the DI system
    print(f"Default device: {device}")
    print(f"Retrieved card: {card}")

Recipe Execution Flow
^^^^^^^^^^^^^^^^^^^^^

The diagram below illustrates how fairseq2's dependency injection system orchestrates
recipe execution, from initial composition through to task execution:

.. mermaid::

    flowchart TD
        %% Styling
        classDef containerBox fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b
        classDef registryBox fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
        classDef executionBox fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#1b5e20
        classDef componentBox fill:#fff3e0,stroke:#f57c00,stroke-width:1px,color:#e65100

        %% Step 0: Composition Phase
        subgraph S0["🔧 Composition Phase"]
            direction TB

            subgraph Inputs["Core Dependencies"]
                direction TB
                WI[World Info]
                DV[Device]
                ENV[Environment]
                FS[File System]
                TH[Thread Pool]
                RNG[RNG Bag]
                componentBox:::componentBox
            end

            subgraph Registry["Extension Registry"]
                direction TB
                AS[Asset Store]
                MF[Model Families]
                TF[Tokenizer Families]
                EX[Extensions]
                CH[Checkpoint Handlers]
                registryBox:::registryBox
            end

            subgraph Recipe["Recipe Components"]
                direction TB
                RC[Recipe Config]
                OD[Output Directory]
                TR[Task Runner]
                executionBox:::executionBox
            end

            Inputs --> LR[Library Registration<br/>_register_library]
            Registry --> LR
            LR --> RR[Recipe Registration<br/>_register_*_recipe]
            Recipe --> RUR[Run Registration<br/>_register_run]
            RR --> RUR
        end

        %% Dependency Container
        RUR --> DC[📦 Dependency Container<br/>Auto-wiring & Resolution]
        DC:::containerBox

        %% Step 1: Execution Phase
        DC --> S1
        subgraph S1["⚡ Execution Phase (_run_recipe)"]
            direction TB

            CP[Cluster Preparer<br/>🏗️ Environment Setup]
            LC[Log Configurer<br/>📋 Distributed Logging]
            CD[Config Dumper<br/>💾 Save Configuration]
            TC[Torch Configurer<br/>🔥 PyTorch Setup]
            LH[Log Helper<br/>📊 System Info Logging]
            TR2[Task Runner<br/>🚀 Execute Recipe Task]

            CP --> LC --> CD --> TC --> LH --> TR2
            executionBox:::executionBox
        end

        %% Resolution arrows
        DC -.->|"resolve(ClusterPreparer)"| CP
        DC -.->|"resolve(LogConfigurer)"| LC
        DC -.->|"resolve(ConfigDumper)"| CD
        DC -.->|"resolve(TorchConfigurer)"| TC
        DC -.->|"resolve(LogHelper)"| LH
        DC -.->|"resolve(TaskRunner)"| TR2

This flow demonstrates several key concepts:

**1. Composition Phase** - All components are registered with the container:
   - **Core Dependencies**: Essential fairseq2 components (Device, WorldInfo, etc.)
   - **Extension Registry**: Pluggable components registered by extensions
   - **Recipe Components**: Task-specific configuration and runners

**2. Dependency Container** - Acts as the central orchestrator:
   - **Auto-wiring**: Automatically resolves dependencies through type inspection
   - **Singleton Management**: Ensures single instances of expensive resources
   - **Collection Support**: Handles multiple implementations (e.g., checkpoint loaders)

**3. Execution Phase** - Components are resolved and executed in sequence:
   - Each component is resolved on-demand from the container
   - Dependencies are automatically injected based on constructor annotations
   - The execution order is deterministic and well-defined

See Also
=========

* :doc:`/news/whats_new_v0_5` - What's new in v0.5 including DI improvements
