import numpy as np
import tensorflow as tf


class LearningRateScheduler:

    """
    Custom learning rate scheduler implementation using warmup and cosine decay 
    """
    def __init__(self, lr_min=1e-5, lr_max=1e-3, warmup_steps=5, total_steps=100):
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps 

    def __call__(self, step):
        # Linear warmup
        if step < self.warmup_steps:
            
            # gradually increase the LR
            lr = self.lr_min + (self.lr_max - self.lr_min) * (step / self.warmup_steps)
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            decay_step = step - self.warmup_steps
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_steps))
            lr = self.lr_min + (self.lr_max - self.lr_min) * cosine_decay
            
        return lr



