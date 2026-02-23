"""Evaluation unit for causal language modeling."""

import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy

from fairseq2.datasets import SequenceBatch
from fairseq2.evaluator import EvalUnit
from fairseq2.gang import Gangs
from fairseq2.metrics import Mean, MetricBag
from fairseq2.nn import BatchLayout

from config import TrainingConfig


class CausalLMEvalUnit(EvalUnit[SequenceBatch]):
    """
    Evaluation unit for causal language modeling.

    This unit implements evaluation without gradients, computing loss
    and perplexity on held-out data.
    """

    def __init__(self, model: Module, gangs: Gangs, config: TrainingConfig) -> None:
        """
        Args:
            model: The language model to evaluate
            gangs: Gang abstraction for distributed training
            config: Training configuration
        """
        self._model = model
        self._gangs = gangs
        self._config = config

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        """
        Prepares metrics to track during evaluation.

        Args:
            metric_bag: Metric bag to populate with metrics
        """
        metric_bag.add("eval_loss", Mean())
        metric_bag.add("eval_ppl", Mean())
        metric_bag.add("eval_num_tokens", Mean())

    def process_batch(
        self,
        batch: SequenceBatch,
        metric_bag: MetricBag,
    ) -> None:
        """
        Processes a batch and computes the loss without gradients.

        Args:
            batch: Batch of sequences to process
            metric_bag: Metric bag for tracking evaluation metrics
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

        # Forward pass through the model (no gradients)
        with torch.no_grad():
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

        metric_bag.get("eval_loss", Mean).update(loss.detach(), weight=batch_size)
        metric_bag.get("eval_ppl", Mean).update(torch.exp(loss.detach()), weight=batch_size)
        metric_bag.get("eval_num_tokens", Mean).update(num_targets, weight=1)
