"""!
\file tensorutils.py Functions involving creation of different type of tensors
"""
import numpy as np
import tensorflow as tf

from typing import Tuple, List


def random_tensor(ishape: List[int], min_v, max_v, dtype):
    """!
    \brief creates a random tensor of given shape
    """
    if any([a < 1 for a in ishape]):
        raise ValueError("input shape can not have an element less than 1")
    arr = np.random.rand(*ishape)
    arr = min_v + (max_v - min_v) * arr
    return tf.constant(arr, dtype=dtype)
