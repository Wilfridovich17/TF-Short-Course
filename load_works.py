import tensorflow as tf
import numpy as np

from math import pi

grid = np.arange(-3.0, 3.0, 0.1)

mean =  0.0
sigma = 1.0

ckpt = tf.train.Checkpoint()
status = ckpt.restore(tf.train.latest_checkpoint('./tf_ckpts/'))

print(status)
