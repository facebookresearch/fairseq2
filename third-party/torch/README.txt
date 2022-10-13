PyTorch distributions 1.12.1 and 1.11.0 were missing some header files that were
transitively used by TorchScript. We store those files here so that we can build
our targets.

See https://github.com/pytorch/pytorch/issues/68876.

TODO: This directory should be deleted once we cease support for PyTorch 1.12.1.
