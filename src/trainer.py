from transformers import Trainer
from transformers.utils import ExplicitEnum
from transformers.trainer_utils import SchedulerType
from transformers.optimization import get_scheduler
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from datasets.utils.logging import get_logger
import logging
import math
import numpy as np
import warnings

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

class LinearWarmupExponentialLR(LRScheduler):
    """
    Exponential LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_warmup_stop_steps, num_training_steps, lr_start, lr_end, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_warmup_stop_steps = num_warmup_stop_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        elif self.last_epoch < self.num_warmup_stop_steps:
            # figure out decay rate to use to get within 1e-10 of lr_end at end of training
            gammas = [np.exp(np.log(1e-10 / (base_lr - self.lr_end)) / (self.num_warmup_stop_steps - self.num_warmup_steps))
                      for base_lr in self.base_lrs]
            return [self.lr_end + (base_lr - self.lr_end) * gamma ** (self.last_epoch - self.num_warmup_steps) for base_lr, gamma in zip(self.base_lrs, gammas)]
        else:
            return [self.lr_end for base_lr in self.base_lrs]

class LinearWarmupCosineLR(LRScheduler):
    """
    Cosine LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_warmup_stop_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_warmup_stop_steps = num_warmup_stop_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start 
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        elif self.last_epoch < self.num_warmup_stop_steps:
            return [self.lr_end + (base_lr - self.lr_end) * (1 + math.cos(math.pi * (self.last_epoch - self.num_warmup_steps) / (self.num_warmup_stop_steps - self.num_warmup_steps))) / 2 for base_lr in self.base_lrs]
        else:
            return [self.lr_end for base_lr in self.base_lrs]
               
class LinearWarmupLinearLR(LRScheduler):
    """
    Lnear LR with linear warmup and decay to some end LR.
    """
    def __init__(self, optimizer, num_warmup_steps, num_warmup_stop_steps, num_training_steps, lr_start=1e-7, lr_end=0, last_epoch=-1, verbose=False):
        self.num_warmup_steps = num_warmup_steps
        self.num_warmup_stop_steps = num_warmup_stop_steps
        self.num_training_steps = num_training_steps
        self.lr_start = lr_start
        self.lr_end = lr_end
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning)

        if self.last_epoch > self.num_training_steps:
            return [group['lr'] for group in self.optimizer.param_groups]

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.lr_start + (base_lr - self.lr_start) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        elif self.last_epoch < self.num_warmup_stop_steps:
            return [self.lr_end + (base_lr - self.lr_end) * (self.num_warmup_stop_steps-self.last_epoch) / (self.num_warmup_stop_steps - self.num_warmup_steps) for base_lr in self.base_lrs]
        else:
            return [self.lr_end for base_lr in self.base_lrs]

class ExtendedSchedulerType(ExplicitEnum):
    LINEAR_WARMUP_EXPONENTIAL = "linear_warmup_exponential"
    LINEAR_WARMUP_COSINE = "linear_warmup_cosine"
    LINEAR_WARMUP_LINEAR = "linear_warmup_linear"
    

TYPE_TO_EXTENDED_SCHEDULER_FUNCTION = {
        ExtendedSchedulerType.LINEAR_WARMUP_EXPONENTIAL: LinearWarmupExponentialLR,
        ExtendedSchedulerType.LINEAR_WARMUP_COSINE: LinearWarmupCosineLR,
        ExtendedSchedulerType.LINEAR_WARMUP_LINEAR: LinearWarmupLinearLR,
}

def get_scheduler_extended(
    name,
    optimizer,
    num_warmup_steps,
    num_warmup_stop_steps,
    num_training_steps,
    lr_end=1e-4,
):
    try:
        name = ExtendedSchedulerType(name)
        schedule_func = TYPE_TO_EXTENDED_SCHEDULER_FUNCTION[name]
    except ValueError:
        return get_scheduler(name, optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    if num_warmup_stop_steps is None:
        num_warmup_stop_steps = num_training_steps

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_warmup_stop_steps=num_warmup_stop_steps, num_training_steps=num_training_steps, lr_start=1e-7, lr_end=lr_end)

class UpdatableTrainer(Trainer):
    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            if self.args.lr_scheduler_name is not None:
                lr_scheduler_name = self.args.lr_scheduler_name
            else:
                lr_scheduler_name = self.args.lr_scheduler_type
            self.lr_scheduler = get_scheduler_extended(
                lr_scheduler_name,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_warmup_stop_steps=self.args.num_warmup_stop_steps,
                num_training_steps=num_training_steps,
                lr_end=self.args.lr_end,
            )
        return self.lr_scheduler