from typing import Optional

import torch.optim.lr_scheduler


class InverseSquareRootLR(torch.optim.lr_scheduler._LRScheduler):
    """Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`eps`) until the learning rate.
    Thereafter we decay proportional to the number of steps,
    with a decay factor set to align with the configured learning rate.

    During warmup::

      lrs = eps
      lr = lrs[update_num]

    After warmup::

      decay_factor = cfg.lr * sqrt(cfg.warmup_steps)
      lr = decay_factor / sqrt(update_num)
    """

    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        lr: float,
        warmup_steps: int = 4000,
        eps: float = 1.25e-07,
        last_epoch: int = -1
    ):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.eps = eps

        self._step_count = 0
        self._warmup_lr_step = (lr - eps) / warmup_steps
        self._warmup_end_correction = lr * warmup_steps**0.5

        super(InverseSquareRootLR, self).__init__(optimizer, last_epoch)

    def step(self, epoch: Optional[int] = None) -> None:
        # Note we aren't using epochs
        step = self._step_count
        self._step_count += 1

        if step < self.warmup_steps:
            lr = self.eps + step * self._warmup_lr_step
        else:
            lr = self._warmup_end_correction * step**-0.5
        self._last_lr = [lr]
        _update_opt(self.optimizer, lr)


def _update_opt(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
