"""
Custom Learning Rate Schedulers for MT3-PyTorch
"""

import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class InverseSqrtScheduler(LRScheduler):
    """
    Inverse Square Root Learning Rate Scheduler (T5-style).
    
    Used in original T5 and MT3 papers. After warmup, learning rate decays
    as 1/sqrt(step), which is slower than exponential or cosine decay.
    
    Formula:
        - Warmup phase (step < warmup_steps):
            lr = base_lr * (step / warmup_steps)
        
        - Decay phase (step >= warmup_steps):
            lr = base_lr / sqrt(step)
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps (linear increase)
        base_lr: Base learning rate (default: 1e-4)
        min_lr: Minimum learning rate (default: 1e-6)
        last_epoch: Last epoch/step number (default: -1)
    
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = InverseSqrtScheduler(optimizer, warmup_steps=10000)
        >>> for step in range(100000):
        ...     optimizer.step()
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        step = max(1, self.last_epoch)  # Avoid division by zero
        
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            # Inverse square root decay
            lr = self.base_lr / math.sqrt(step)
        
        # Clamp to minimum LR
        lr = max(lr, self.min_lr)
        
        return [lr for _ in self.optimizer.param_groups]


class InverseSqrtWithWarmup(LRScheduler):
    """
    Alternative implementation with explicit warmup parameter scaling.
    
    This version uses the warmup_steps as the reference point for decay,
    making the decay more stable: lr = base_lr / sqrt(max(step, warmup_steps))
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        step = max(1, self.last_epoch)
        
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            # Inverse square root decay (relative to warmup_steps)
            # This keeps LR more stable than decaying from step 1
            lr = self.base_lr * math.sqrt(self.warmup_steps / step)
        
        # Clamp to minimum LR
        lr = max(lr, self.min_lr)
        
        return [lr for _ in self.optimizer.param_groups]
