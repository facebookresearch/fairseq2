.. _guides-s3-checkpointing:

===========================
S3 Checkpointing
===========================

fairseq2 supports storing checkpoints on Amazon S3 while keeping other training
artifacts (logs, metrics, caches) on local or NFS storage. This hybrid approach
is useful when you want to leverage S3's scalability and durability for large
checkpoint files.

Prerequisites
=============

1. Install the ``s3fs`` package (included with fairseq2 by default):

   .. code-block:: bash

       pip install s3fs

2. Configure AWS credentials using one of the standard methods:

   - Environment variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``)
   - AWS credentials file (``~/.aws/credentials``)
   - IAM role (when running on AWS infrastructure)

Usage
=====

Use the ``--checkpoint-dir`` CLI option to redirect checkpoints to an S3 bucket:

.. code-block:: bash

    python -m recipes.lm.train /local/output/dir \
        --checkpoint-dir s3://my-bucket/checkpoints/experiment1

This will:

- Store checkpoints (and `model.yaml`) at ``s3://my-bucket/checkpoints/experiment1/config_hash/step_N/``
- Keep logs, metrics, and other artifacts in ``/local/output/dir/config_hash/``

The ``config_hash`` (e.g., ``ws_1.d2b3ae4f``) is automatically appended to both directories
based on the training configuration, ensuring consistent organization across local and remote storage.

Resuming from S3 Checkpoints
============================

To resume training from S3 checkpoints, use the same ``--checkpoint-dir`` option:

.. code-block:: bash

    python -m recipes.lm.train /local/output/dir \
        --checkpoint-dir s3://my-bucket/key/ \
        --resume-from last

The checkpoint manager will automatically detect and load the latest checkpoint
from S3.

Using S3 Paths in Asset Cards
=============================

You can define model or dataset cards that reference S3 paths directly. Here is
an example of a model card with checkpoints and tokenizer stored on S3:

.. code-block:: yaml

    # my_s3_model.yaml
    name: my_s3_model
    model_family: llama
    model_arch: llama3_8b
    checkpoint: "s3://my-bucket/models/my_model/checkpoint.pt"
    tokenizer: "s3://my-bucket/models/my_model/tokenizer.model"
    tokenizer_family: llama

And an example dataset card with data files on S3:

.. code-block:: yaml

    # my_s3_dataset.yaml
    name: my_s3_dataset
    dataset_family: generic_text
    path: "s3://my-bucket/datasets/my_dataset/"

To use these cards, add them to an asset store directory and specify it via:

.. code-block:: bash

    python -m recipes.lm.train /output/dir \
        --config common.asset.extra_paths="['/path/to/my/cards']"

Registering a Custom S3 Filesystem with AWS Profile
===================================================

If you need to use a specific AWS profile or custom S3 configuration (e.g., for
accessing buckets with different credentials), you can register a custom filesystem
before running your training:

.. code-block:: python

    import s3fs
    from fairseq2.file_system import FileSystemRegistry, FSspecFileSystem

    def register_s3_with_profile(
        profile_name: str,
        bucket_pattern: str | None = None,
    ) -> None:
        """
        Register an S3 filesystem with a specific AWS profile.

        Args:
            profile_name: The AWS profile name from ~/.aws/credentials
            bucket_pattern: Optional bucket name pattern to match. If provided,
                only S3 paths containing this pattern will use this filesystem.
                If None, this filesystem will be used for all S3 paths.
        """
        # Create S3 filesystem with the specified profile
        s3_fs = s3fs.S3FileSystem(profile=profile_name)

        # Define the pattern check function
        if bucket_pattern:
            def pattern_check(path) -> bool:
                path_str = str(path)
                return path_str.startswith("s3:/") and bucket_pattern in path_str
        else:
            def pattern_check(path) -> bool:
                return str(path).startswith("s3:/")

        # Wrap and register the filesystem
        wrapped_fs = FSspecFileSystem(s3_fs, "s3:/")
        FileSystemRegistry.register(pattern_check, lambda: wrapped_fs)

    # Example: Register S3 filesystem with "my-team-profile" for a specific bucket
    register_s3_with_profile(
        profile_name="my-team-profile",
        bucket_pattern="my-team-bucket",
    )

    # Now S3 paths like "s3://my-team-bucket/..." will use this profile


Programmatic Usage
==================

When using the training API directly, pass ``checkpoint_dir`` to the ``run()`` function:

.. code-block:: python

    from pathlib import Path
    from fairseq2.recipe import run

    run(
        recipe,
        config,
        output_dir=Path("/local/output/dir"),
        checkpoint_dir=Path("s3://my-bucket/key/"),
    )

Implementation Notes
====================

- **Atomic writes**: For local filesystems, checkpoints are written to a temporary
  directory (``step_N.tmp``) and atomically renamed upon completion. For S3,
  writes go directly to the final location since S3 doesn't support atomic renames.

- **Tested protocols**: Currently, only ``file`` (local), ``local``, and ``s3``
  protocols are officially supported. Other fsspec-compatible protocols may work
  but are not tested.

- **Filesystem priority**: When multiple filesystems are registered for
  overlapping patterns, the most recently registered one takes precedence.
  Register bucket-specific filesystems after the default S3 filesystem.

See Also
========

* :doc:`/basics/building_recipes` - Building custom training recipes
* :doc:`/basics/assets` - Understanding fairseq2 asset system
* :doc:`/reference/checkpoint` - Checkpoint API reference
