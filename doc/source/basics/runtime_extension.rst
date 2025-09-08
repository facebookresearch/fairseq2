.. _basics-runtime-extension:

:octicon:`plug` Runtime Extension
=================================

fairseq2 provides a flexible runtime extension system that allows you to extend its functionality without modifying the core codebase.
It uses ``setuptools`` entry points to dynamically load and register extensions during initialization.

Overview
--------

The extension system is built around a dependency injection container (learn more in :ref:`basics-design-philosophy`) that manages fairseq2's components.
Through this system, you can:

* Register new models and model cards
* Add custom asset providers and validators
* Extend the runtime context
* And more...

Basic Usage
-----------

Before using any fairseq2 APIs, you must initialize the framework with :meth:`fairseq2.init_fairseq2`:

.. code-block:: python

    from fairseq2 import init_fairseq2

    init_fairseq2()

Creating Extensions
-------------------

To create an extension, define a setup function:

.. code-block:: python

    from fairseq2.runtime.dependency import DependencyContainer

    def setup_my_extension(container: DependencyContainer) -> None:
        # Register your custom components here using `context`.
        pass

Registering Extensions
----------------------

Extensions are registered using setuptools entry points. You can configure them in either ``setup.py`` or ``pyproject.toml``:

Using ``setup.py``:

.. code-block:: python

    setup(
        name="my-fairseq2-extension",
        entry_points={
            "fairseq2.extension": [
                "my_extension = my_package.module:setup_my_extension",
            ],
        },
    )

Using ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."fairseq2.extension"]
    my_extension = "my_package.module:setup_my_extension"

Extension Loading Process
-------------------------

When ``init_fairseq2()`` is called, the following steps occur:

1. fairseq2 components are initialized
2. All registered extensions are discovered via entry points
3. Each extension's setup function is called

Complete Example
----------------

Here's a complete example of implementing a fairseq2 extension:

.. code-block:: python

    from fairseq2.assets import AssetEnvironmentResolver
    from fairseq2.composition import register_package_assets
    from fairseq2.runtime.dependency import DependencyContainer


    def setup_my_extension(container: DependencyContainer) -> None:

        # Register package metadata provider for cards
        register_package_assets(container, package="my_package.cards")

        def resolve_cluster_name(resolver: DependencyResolver) -> str:
            return "my_cluster"

        # To manage assets from a custom source, you can append a function that returns the asset source name to the list of environment resolvers
        # For example, the following code registers a function that returns "my_cluster" as the asset source name.
        # This allows you to add assets in the asset cards with identifiers that ends with "@my_cluster".
        container.collection.register(
            AssetEnvironmentResolver, lambda _: resolve_cluster_name
        )


Error Handling
--------------

The extension system includes error handling to maintain system stability:

* Failed extensions log warnings by default
* Set ``FAIRSEQ2_EXTENSION_TRACE`` environment variable for detailed error traces
* Invalid extension functions raise ``RuntimeError``

.. code-block:: bash

    export FAIRSEQ2_EXTENSION_TRACE=1


See Also
--------

* :doc:`/basics/assets` for more information on assets
