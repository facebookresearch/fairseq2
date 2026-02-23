"""Training unit for causal language modeling."""

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy

from fairseq2.datasets import SequenceBatch
from fairseq2.gang import Gangs
from fairseq2.metrics import Mean, MetricBag
from fairseq2.nn import BatchLayout
from fairseq2.trainer import TrainUnit

from config import TrainingConfig


class CausalLMTrainUnit(TrainUnit[SequenceBatch]):
    """
    Training unit for causal language modeling.

    This unit implements the forward pass, loss computation, and metric tracking
    for next-token prediction training.
    """

    def __init__(self, model: Module, gangs: Gangs, config: TrainingConfig) -> None:
        """
        Args:
            model: The language model to train
            gangs: Gang abstraction for distributed training
            config: Training configuration
        """
        self._model = model
        self._gangs = gangs
        self._config = config

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        """
        Prepares metrics to track during training.

        Args:
            metric_bag: Metric bag to populate with metrics
        """
        metric_bag.add("train_loss", Mean())
        metric_bag.add("train_ppl", Mean())
        metric_bag.add("num_tokens", Mean())

    def process_batch(
        self,
        batch: SequenceBatch,
        metric_bag: MetricBag,
    ) -> tuple[Tensor, int | None]:
        """
        Processes a batch and computes the loss.

        For causal language modeling, we:
        1. Run the model on input sequences
        2. Compute cross-entropy loss for next-token prediction
        3. Track metrics

        Args:
            batch: Batch of sequences to process
            metric_bag: Metric bag for tracking training metrics

        Returns:
            Tuple of (loss, num_targets) where:
            - loss: Scalar loss tensor for backpropagation
            - num_targets: Number of target tokens (for gradient normalization)
        """
        # Move batch to the correct device
        batch.to(self._gangs.device)

        # Extract input sequences
        input_seqs = batch.seqs

        # For causal LM: input is seqs[:-1], target is seqs[1:]
        input_ids = input_seqs[:, :-1]
        target_ids = input_seqs[:, 1:]

        # Create batch layout for the input sequences
        input_seq_lens = None
        if batch.seq_lens is not None:
            input_seq_lens = [max(1, length - 1) for length in batch.seq_lens]

        batch_layout = BatchLayout(
            shape=input_ids.shape,
            seq_lens=input_seq_lens,
            device=self._gangs.device,
        )

        # Forward pass through the model
        logits = self._model(input_ids, batch_layout)

        # Compute cross-entropy loss
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = target_ids.reshape(-1)

        loss = cross_entropy(
            logits_flat,
            targets_flat,
            reduction="mean",
            ignore_index=-1,
        )

        # Compute number of target tokens
        num_targets = target_ids.numel()

        # Track metrics
        batch_size = input_seqs.size(0)

        metric_bag.get("train_loss", Mean).update(loss.detach(), weight=batch_size)
        metric_bag.get("train_ppl", Mean).update(torch.exp(loss.detach()), weight=batch_size)
        metric_bag.get("num_tokens", Mean).update(num_targets, weight=1)

        return loss, num_targets
