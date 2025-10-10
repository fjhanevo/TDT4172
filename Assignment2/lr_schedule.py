import numpy as np
import tensorflow as tf

class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

    """
    Custom learning rate scheduler implementation using warmup and cosine decay.
    Inherits from LearningRateSchedule so it can be used in the optimizer.
    """
    def __init__(self, lr_min=1e-5, lr_max=1e-3, warmup_steps=5, total_steps=100):
        self.lr_min = tf.cast(lr_min, tf.float32)
        self.lr_max = tf.cast(lr_max, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        is_warmup = step < self.warmup_steps
        # Linear warmup

        def warmup():
            return self.lr_min + (self.lr_max - self.lr_min) * (step / self.warmup_steps)

        # Cosine decay
        def cosine_decay(): 
            decay_steps = self.total_steps - self.warmup_steps
            decay_step = step - self.warmup_steps
            decay = 0.5 * (1 + tf.math.cos(np.pi * decay_step / decay_steps))
            return self.lr_min + (self.lr_max - self.lr_min) * decay
            
        return tf.cond(is_warmup, true_fn=warmup, false_fn=cosine_decay)

