import numpy as np
import tensorflow as tf
import random
import os

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def smooth_labels(labels, factor=0.1):
    # apply label smoothing
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels
